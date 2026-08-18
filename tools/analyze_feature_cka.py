#!/usr/bin/env python3
"""Offline representation-similarity diagnostic for heterogeneous W2S-CPS.

This script only loads checkpoints and runs deterministic validation inference.
It never trains, changes model parameters, or writes into checkpoint directories.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow execution as ``python tools/analyze_feature_cka.py`` from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.getnetwork import get_network


EPS = 1e-12
DEFAULT_EPOCHS = (20, 50, 100, 150, 200)
LAYER_MODULES = (
    ("E1", "Conv1"),
    ("E2", "Conv2"),
    ("E3", "Conv3"),
    ("E4", "Conv4"),
    ("Bottleneck", "Conv5"),
    ("DecoderLate", "Up_conv2"),
)


@dataclass(frozen=True)
class CheckpointPair:
    requested_epoch: int
    appearance_path: Path
    structure_path: Path
    appearance_actual_epoch: int
    structure_actual_epoch: int
    kind: str

    @property
    def actual_label(self) -> str:
        if self.appearance_actual_epoch == self.structure_actual_epoch:
            return str(self.appearance_actual_epoch)
        return "A:{}|S:{}".format(
            self.appearance_actual_epoch, self.structure_actual_epoch
        )

    @property
    def path_label(self) -> str:
        if self.appearance_path == self.structure_path:
            return str(self.appearance_path)
        return "A={}; S={}".format(self.appearance_path, self.structure_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="experiment directory; auto-selects the no-confidence GLaS baseline",
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=list(DEFAULT_EPOCHS))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets/GLaS/manifests/paper_seed_2026/validation.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/feature_cka")
    parser.add_argument("--max-samples-per-image", type=int, default=2048)
    parser.add_argument("--cka-seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--network", default="unet")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    if args.max_samples_per_image < 2:
        parser.error("--max-samples-per-image must be at least 2")
    if args.batch_size != 1:
        parser.error("batch size 1 is required for per-image aligned observation sampling")
    return args


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> float:
    """Linear CKA with observations in rows and arbitrary feature dimensions."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be [N,C] with the same N")
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two observations")
    # float64 accumulation is inexpensive after bounded spatial sampling and
    # avoids large cancellation error during feature centering.
    x = x.detach().to(device="cpu", dtype=torch.float64)
    y = y.detach().to(device="cpu", dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = cross.square().sum()
    denominator = torch.sqrt(xx.square().sum() * yy.square().sum()).clamp_min(eps)
    value = float((numerator / denominator).item())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite CKA")
    if value < -1e-8 or value > 1.0 + 1e-6:
        raise AssertionError("Linear CKA outside numerical [0,1] range: {}".format(value))
    return min(1.0, max(0.0, value))


def run_sanity_tests(seed: int = 1337) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(512, 32, generator=generator, dtype=torch.float64)
    identical = linear_cka(x, x)
    permutation = torch.randperm(x.shape[0], generator=generator)
    shuffled = linear_cka(x, x[permutation])
    q, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator, dtype=torch.float64))
    rotated = linear_cka(x, x @ q)
    if abs(identical - 1.0) > 1e-10:
        raise AssertionError("identical-feature CKA is not 1: {}".format(identical))
    if shuffled >= 0.25 or shuffled >= identical - 0.5:
        raise AssertionError("observation shuffling did not clearly lower CKA: {}".format(shuffled))
    if abs(rotated - 1.0) > 1e-10:
        raise AssertionError("orthogonal rotation invariance failed: {}".format(rotated))
    return {"identical": identical, "shuffled": shuffled, "orthogonal_rotation": rotated}


def _auto_experiment_dir() -> Path:
    root = PROJECT_ROOT / "checkpoints/w-o-confidence-threshold/GLaS"
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    candidates = [path for path in candidates if "weak_structure_strong" in path.name]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "expected exactly one no-confidence heterogeneous GLaS experiment below {}, got {}".format(
                root, len(candidates)
            )
        )
    return candidates[0]


def _periodic_actual_epochs(experiment_dir: Path) -> dict[tuple[int, str], int]:
    output: dict[tuple[int, str], int] = {}
    pattern = re.compile(
        r"current_epoch=(\d+).*branch=([AS]).*best_epoch=(\d+).*path=(\S+)"
    )
    for log_path in sorted(experiment_dir.glob("train_*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                output[(int(match.group(1)), match.group(2))] = int(match.group(3))
    return output


def resolve_checkpoints(experiment_dir: Path, epochs: Sequence[int]) -> list[CheckpointPair]:
    experiment_dir = experiment_dir.resolve()
    actual_epochs = _periodic_actual_epochs(experiment_dir)
    pairs = []
    for epoch in epochs:
        milestone = experiment_dir / "milestones" / "epoch_{:03d}.pth".format(epoch)
        if milestone.is_file():
            # A joint milestone is accepted only when its payload declares this
            # exact current epoch. Branch state_dicts are current weights; the
            # separate best_epochs metadata is not used as their epoch.
            payload = torch.load(milestone, map_location="cpu", weights_only=False)
            if int(payload.get("epoch", -1)) != epoch:
                raise ValueError("milestone epoch metadata mismatch: {}".format(milestone))
            if not all(key in payload for key in ("appearance_state_dict", "structure_state_dict")):
                raise KeyError("joint milestone lacks both branch states: {}".format(milestone))
            pairs.append(CheckpointPair(epoch, milestone, milestone, epoch, epoch, "milestone"))
            del payload
            continue

        candidates_a = sorted((experiment_dir / "periodic_best").glob(
            "epoch_{:04d}_best_A_Jc_*.pth".format(epoch)
        ))
        candidates_s = sorted((experiment_dir / "periodic_best").glob(
            "epoch_{:04d}_best_S_Jc_*.pth".format(epoch)
        ))
        if len(candidates_a) != 1 or len(candidates_s) != 1:
            raise FileNotFoundError("cannot uniquely resolve checkpoint pair for requested epoch {}".format(epoch))
        key_a, key_s = (epoch, "A"), (epoch, "S")
        if key_a not in actual_epochs or key_s not in actual_epochs:
            raise RuntimeError(
                "periodic-best actual epochs are absent from training log for requested epoch {}".format(epoch)
            )
        pairs.append(CheckpointPair(
            epoch, candidates_a[0].resolve(), candidates_s[0].resolve(),
            actual_epochs[key_a], actual_epochs[key_s], "periodic_best",
        ))
    return pairs


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def _state_from_checkpoint(path: Path, branch: str) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if _is_state_dict(payload):
        state = payload
    elif isinstance(payload, Mapping):
        aliases = (
            ("appearance_state_dict", "model1_state_dict", "model1")
            if branch == "appearance"
            else ("structure_state_dict", "model2_state_dict", "model2")
        )
        state = next((payload[key] for key in aliases if _is_state_dict(payload.get(key))), None)
        if state is None:
            raise KeyError("cannot find {} state_dict in {}".format(branch, path))
    else:
        raise TypeError("unsupported checkpoint payload: {}".format(path))
    return {key.removeprefix("module."): value for key, value in state.items()}


def build_validation_loader(manifest: Path, batch_size: int, num_workers: int) -> DataLoader:
    cfg = dataset_cfg("GlaS")
    dual_cfg = dict(cfg["dual_view"])
    dual_cfg["enabled"] = True
    # Prevent even constructing a randomized strong view during this diagnostic.
    dual_cfg["generate_strong_eval"] = False
    dataset = imagefloder_itn(
        manifest_path=str(manifest.resolve()),
        data_transform_1=data_transform_2d()["val"],
        data_normalize_1=data_normalize_2d(cfg["MEAN"], cfg["STD"]),
        dual_view_config=dual_cfg,
        sup=True,
        training=False,
    )
    if len(dataset) != 17:
        raise AssertionError("expected 17 GLaS validation images, found {}".format(len(dataset)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class ActivationCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = [name for _, name in LAYER_MODULES if name not in modules]
        if missing:
            raise KeyError("required semantic U-Net modules are missing: {}".format(missing))
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []
        for label, module_name in LAYER_MODULES:
            self.handles.append(modules[module_name].register_forward_hook(self._hook(label)))

    def _hook(self, label: str):
        def save(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            if not torch.is_tensor(output) or output.ndim != 4:
                raise TypeError("hook {} did not produce [B,C,H,W]".format(label))
            self.values[label] = output.detach()
        return save

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _uniform_indices(height: int, width: int, maximum: int) -> torch.Tensor:
    count = height * width
    if count <= maximum:
        return torch.arange(count, dtype=torch.long)
    # Integer midpoint sampling is deterministic, unique, and covers the full map.
    return torch.div(
        torch.arange(maximum, dtype=torch.long) * count + count // 2,
        maximum,
        rounding_mode="floor",
    )


def _sample_feature(feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if feature.shape[0] != 1:
        raise ValueError("sampling assumes one image per batch")
    flat = feature[0].permute(1, 2, 0).reshape(-1, feature.shape[1])
    return flat.index_select(0, indices.to(flat.device)).to(device="cpu", dtype=torch.float32)


def _parameter_versions(model: torch.nn.Module) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    parameters = tuple(model.parameters())
    return tuple(p._version for p in parameters), tuple(p.grad is None for p in parameters)


def _prediction_metrics(pred_a: torch.Tensor, pred_s: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    correct_a = pred_a.eq(gt)
    correct_s = pred_s.eq(gt)
    a_only = correct_a & ~correct_s
    s_only = ~correct_a & correct_s
    both_wrong = ~correct_a & ~correct_s
    total = float(gt.numel())
    disagreement = float(pred_a.ne(pred_s).sum()) / total
    # For binary segmentation, disagreement exactly partitions A-only/S-only correctness.
    if not torch.equal(pred_a.ne(pred_s), a_only | s_only):
        raise AssertionError("binary disagreement does not match correctness partition")
    error_union = a_only | s_only | both_wrong
    error_overlap = float(both_wrong.sum()) / max(float(error_union.sum()), 1.0)
    ca = correct_a.double().flatten()
    cs = correct_s.double().flatten()
    ca_centered, cs_centered = ca - ca.mean(), cs - cs.mean()
    corr_den = torch.sqrt(ca_centered.square().sum() * cs_centered.square().sum())
    correlation = float((ca_centered * cs_centered).sum() / corr_den) if corr_den > 0 else float("nan")
    a_only_n, s_only_n, both_wrong_n = map(float, (a_only.sum(), s_only.sum(), both_wrong.sum()))
    a_rescue = a_only_n / max(a_only_n + both_wrong_n, 1.0)
    s_rescue = s_only_n / max(s_only_n + both_wrong_n, 1.0)
    return {
        "prediction_disagreement": disagreement,
        "error_overlap_iou": error_overlap,
        "correctness_correlation": correlation,
        "modelA_only_correct_ratio": a_only_n / total,
        "modelS_only_correct_ratio": s_only_n / total,
        "both_wrong_ratio": both_wrong_n / total,
        "mean_bidirectional_rescue": 0.5 * (a_rescue + s_rescue),
    }


def diagnose_pair(
    pair: CheckpointPair,
    loader: DataLoader,
    device: torch.device,
    network: str,
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    cfg = dataset_cfg("GlaS")
    model_a = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_s = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_a.load_state_dict(_state_from_checkpoint(pair.appearance_path, "appearance"), strict=True)
    model_s.load_state_dict(_state_from_checkpoint(pair.structure_path, "structure"), strict=True)
    model_a.to(device).eval()
    model_s.to(device).eval()
    capture_a, capture_s = ActivationCapture(model_a), ActivationCapture(model_s)
    before_a, before_s = _parameter_versions(model_a), _parameter_versions(model_s)
    dataset_features_a: dict[str, list[torch.Tensor]] = defaultdict(list)
    dataset_features_s: dict[str, list[torch.Tensor]] = defaultdict(list)
    observed_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    per_image_rows: list[dict[str, Any]] = []
    all_pred_a, all_pred_s, all_gt = [], [], []

    try:
        with torch.no_grad():
            for batch in loader:
                appearance = batch["image_appearance_weak"].to(device, non_blocking=True)
                structure = batch["image_structure"].to(device, non_blocking=True)
                gt = batch["mask"].long().cpu()
                image_name = str(batch["ID"][0])

                capture_a.clear()
                logits_a = model_a(appearance)[0]
                features_a = {label: tensor for label, tensor in capture_a.values.items()}
                capture_s.clear()
                logits_s = model_s(structure)[0]
                features_s = {label: tensor for label, tensor in capture_s.values.items()}

                for label, _module_name in LAYER_MODULES:
                    fa, fs = features_a[label], features_s[label]
                    if fa.shape[-2:] != fs.shape[-2:]:
                        raise AssertionError(
                            "{} spatial mismatch {} vs {}; no resize was applied".format(
                                label, tuple(fa.shape), tuple(fs.shape)
                            )
                        )
                    indices = _uniform_indices(fa.shape[-2], fa.shape[-1], max_samples)
                    xa, ys = _sample_feature(fa, indices), _sample_feature(fs, indices)
                    value = linear_cka(xa, ys)
                    dataset_features_a[label].append(xa)
                    dataset_features_s[label].append(ys)
                    observed_shapes[label] = (tuple(fa.shape), tuple(fs.shape))
                    per_image_rows.append({
                        "requested_epoch": pair.requested_epoch,
                        "actual_checkpoint_epoch": pair.actual_label,
                        "image_name": image_name,
                        "layer": label,
                        "cka": value,
                    })

                all_pred_a.append(logits_a.argmax(dim=1).cpu())
                all_pred_s.append(logits_s.argmax(dim=1).cpu())
                all_gt.append(gt)

        cka_rows: list[dict[str, Any]] = []
        for label, module_name in LAYER_MODULES:
            # Concatenate each representation independently: C_A and C_S may differ.
            xa = torch.cat(dataset_features_a[label], dim=0)
            ys = torch.cat(dataset_features_s[label], dim=0)
            shape_a, shape_s = observed_shapes[label]
            cka_rows.append({
                "requested_epoch": pair.requested_epoch,
                "actual_checkpoint_epoch": pair.actual_label,
                "layer": label,
                "cka": linear_cka(xa, ys),
                "n_observations": xa.shape[0],
                "appearance_feature_shape": str(list(shape_a)).replace(" ", ""),
                "structure_feature_shape": str(list(shape_s)).replace(" ", ""),
                "checkpoint_path": pair.path_label,
            })
        metrics = _prediction_metrics(
            torch.cat(all_pred_a), torch.cat(all_pred_s), torch.cat(all_gt)
        )
    finally:
        capture_a.close()
        capture_s.close()

    after_a, after_s = _parameter_versions(model_a), _parameter_versions(model_s)
    if before_a != after_a or before_s != after_s:
        raise AssertionError("model parameters or gradients changed during no_grad diagnostic")
    return cka_rows, per_image_rows, metrics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def make_plots(output_dir: Path, cka_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted({int(row["requested_epoch"]) for row in cka_rows})
    values = {(int(row["requested_epoch"]), str(row["layer"])): float(row["cka"]) for row in cka_rows}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for layer, _ in LAYER_MODULES:
        ax.plot(epochs, [values[(epoch, layer)] for epoch in epochs], marker="o", label=layer)
    ax.set_xlabel("Requested checkpoint stage (see CSV for actual epoch pair)")
    ax.set_ylabel("Linear CKA")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "cka_vs_epoch.png", dpi=200)
    plt.close(fig)

    matrix = np.asarray([[values[(epoch, layer)] for epoch in epochs] for layer, _ in LAYER_MODULES])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(epochs)), epochs)
    ax.set_yticks(range(len(LAYER_MODULES)), [label for label, _ in LAYER_MODULES])
    ax.set_xlabel("Requested checkpoint stage")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] < 0.55 else "black"
            ax.text(column_index, row_index, "{:.3f}".format(matrix[row_index, column_index]),
                    ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="Linear CKA")
    fig.tight_layout()
    fig.savefig(output_dir / "cka_epoch_layer_heatmap.png", dpi=200)
    plt.close(fig)


def write_report(
    output_dir: Path,
    pairs: Sequence[CheckpointPair],
    cka_rows: Sequence[Mapping[str, Any]],
    aligned_rows: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, float],
) -> None:
    by_epoch_layer = {(int(r["requested_epoch"]), str(r["layer"])): float(r["cka"]) for r in cka_rows}
    epochs = [pair.requested_epoch for pair in pairs]
    deep = [by_epoch_layer[(epoch, "Bottleneck")] for epoch in epochs]
    overlap = [float(row["error_overlap_iou"]) for row in aligned_rows]
    rescue = [float(row["mean_bidirectional_rescue"]) for row in aligned_rows]
    corr_overlap, corr_rescue = spearman(deep, overlap), spearman(deep, rescue)
    deep_delta = deep[-1] - deep[0]
    overlap_delta = overlap[-1] - overlap[0]
    rescue_delta = rescue[-1] - rescue[0]
    tolerance = 0.02
    if deep_delta > tolerance and overlap_delta > 0 and rescue_delta < 0:
        interpretation = (
            "Evidence is consistent with progressive representation homogenization "
            "accompanying complementarity decay."
        )
    elif deep_delta < -tolerance:
        interpretation = (
            "Feature representations remain or become increasingly diverse; the late-stage "
            "CPS ceiling is unlikely to be explained by feature collapse."
        )
    else:
        interpretation = (
            "Prediction/error homogenization is observed without clear representation-level "
            "homogenization."
        )

    lines = [
        "Representation Similarity Diagnostic (offline, Linear CKA)",
        "",
        "Sanity tests:",
        "  identical={:.12f}".format(sanity["identical"]),
        "  shuffled={:.12f}".format(sanity["shuffled"]),
        "  orthogonal_rotation={:.12f}".format(sanity["orthogonal_rotation"]),
        "",
        "Hooked layers:",
    ]
    lines.extend("  {} -> {}".format(label, module) for label, module in LAYER_MODULES)
    lines.extend(["", "Checkpoints actually used:"])
    for pair in pairs:
        lines.append(
            "  requested={} actual={} kind={} path={}".format(
                pair.requested_epoch, pair.actual_label, pair.kind, pair.path_label
            )
        )
    lines.extend(["", "Dataset-level CKA:"])
    for epoch in epochs:
        pair = next(item for item in pairs if item.requested_epoch == epoch)
        lines.append("Epoch {} (actual {}):".format(epoch, pair.actual_label))
        for label, _ in LAYER_MODULES:
            lines.append("  {} {:.6f}".format(label, by_epoch_layer[(epoch, label)]))
    lines.extend([
        "",
        "Deep feature CKA trend:",
        "  " + " -> ".join("{}:{:.6f}".format(e, v) for e, v in zip(epochs, deep)),
        "Error-overlap trend:",
        "  " + " -> ".join("{}:{:.6f}".format(e, v) for e, v in zip(epochs, overlap)),
        "Rescue trend:",
        "  " + " -> ".join("{}:{:.6f}".format(e, v) for e, v in zip(epochs, rescue)),
        "",
        "Exploratory Spearman correlations (n={}, no significance/causal claim):".format(len(epochs)),
        "  Corr(deep CKA, ErrorOverlap)={:.6f}".format(corr_overlap),
        "  Corr(deep CKA, Rescue)={:.6f}".format(corr_rescue),
        "",
        "Neutral interpretation:",
        "  " + interpretation,
        "",
        "Caveat: periodic_best entries are historical branch-best snapshots and may pair",
        "different actual epochs. Requested-stage curves are not true epoch trajectories when",
        "kind=periodic_best; consult actual_checkpoint_epoch before scientific interpretation.",
    ])
    (output_dir / "cka_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sanity = run_sanity_tests(args.cka_seed)
    print("Sanity tests passed:", sanity)
    if args.sanity_only:
        return
    experiment_dir = (args.experiment_dir or _auto_experiment_dir()).resolve()
    pairs = resolve_checkpoints(experiment_dir, args.epochs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = build_validation_loader(args.manifest, args.batch_size, args.num_workers)
    device = torch.device(args.device)

    all_cka_rows, all_per_image_rows, aligned_rows = [], [], []
    for pair in pairs:
        print("Diagnosing requested epoch {} (actual {}, {})".format(
            pair.requested_epoch, pair.actual_label, pair.kind
        ), flush=True)
        cka_rows, per_image_rows, metrics = diagnose_pair(
            pair, loader, device, args.network, args.max_samples_per_image
        )
        all_cka_rows.extend(cka_rows)
        all_per_image_rows.extend(per_image_rows)
        deep_cka = next(float(row["cka"]) for row in cka_rows if row["layer"] == "Bottleneck")
        aligned_rows.append({
            "epoch": pair.requested_epoch,
            "requested_epoch": pair.requested_epoch,
            "actual_checkpoint_epoch": pair.actual_label,
            "deep_cka": deep_cka,
            **metrics,
            # Compact aliases requested by the example schema.
            "disagreement": metrics["prediction_disagreement"],
            "error_overlap": metrics["error_overlap_iou"],
            "rescue": metrics["mean_bidirectional_rescue"],
            "both_wrong": metrics["both_wrong_ratio"],
            "checkpoint_path": pair.path_label,
        })
        del cka_rows, per_image_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = []
    for cka_row in all_cka_rows:
        values = [float(row["cka"]) for row in all_per_image_rows
                  if row["requested_epoch"] == cka_row["requested_epoch"] and row["layer"] == cka_row["layer"]]
        summary_rows.append({
            "requested_epoch": cka_row["requested_epoch"],
            "actual_checkpoint_epoch": cka_row["actual_checkpoint_epoch"],
            "layer": cka_row["layer"],
            "mean_per_image_cka": float(np.mean(values)),
            "std_per_image_cka": float(np.std(values, ddof=1)),
            "dataset_level_cka": cka_row["cka"],
        })

    _write_csv(output_dir / "cka_by_epoch_layer.csv", all_cka_rows, list(all_cka_rows[0]))
    _write_csv(output_dir / "cka_per_image.csv", all_per_image_rows, list(all_per_image_rows[0]))
    _write_csv(output_dir / "cka_summary.csv", summary_rows, list(summary_rows[0]))
    _write_csv(output_dir / "cka_with_complementarity.csv", aligned_rows, list(aligned_rows[0]))
    make_plots(output_dir, all_cka_rows)
    write_report(output_dir, pairs, all_cka_rows, aligned_rows, sanity)
    print("Wrote diagnostic outputs to", output_dir)


if __name__ == "__main__":
    main()
