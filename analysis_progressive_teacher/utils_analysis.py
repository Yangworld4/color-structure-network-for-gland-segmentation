"""Shared utilities for the read-only progressive-teacher analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from dataload.dataset_2d import imagefloder_itn
from models.getnetwork import get_network


EPS = 1e-8
THRESHOLDS = np.arange(0.0, 0.9, 0.02, dtype=np.float64)


@dataclass(frozen=True)
class CheckpointPair:
    epoch: int
    appearance: Path
    structure: Path


@dataclass
class ValidationData:
    loader: DataLoader
    names: list[str]
    targets: np.ndarray
    image_paths: list[Path]
    manifest: Path


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu: str) -> torch.device:
    value = str(gpu).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not re.fullmatch(r"\d+", value):
        raise ValueError("--gpu must be a CUDA index, 'cpu', or 'auto'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError("CUDA device {} is unavailable".format(index))
    return torch.device("cuda:{}".format(index))


def _score_from_name(path: Path) -> float:
    match = re.search(r"(?i)(?:jc|iou)[_-]([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else float("-inf")


def _branch_from_name(path: Path) -> str | None:
    stem = path.stem.lower()
    patterns = {
        "appearance": (
            r"(?:^|[_-])best[_-]?a(?:[_-]|$)",
            r"(?:^|[_-])appearance(?:[_-]|$)",
            r"(?:^|[_-])result1(?:[_-]|$)",
        ),
        "structure": (
            r"(?:^|[_-])best[_-]?s(?:[_-]|$)",
            r"(?:^|[_-])structure(?:[_-]|$)",
            r"(?:^|[_-])result2(?:[_-]|$)",
        ),
    }
    matches = [
        branch
        for branch, expressions in patterns.items()
        if any(re.search(expression, stem) for expression in expressions)
    ]
    if len(matches) > 1:
        raise ValueError("Ambiguous branch in checkpoint name: {}".format(path))
    return matches[0] if matches else None


def _epoch_from_name(path: Path) -> int | None:
    match = re.search(r"(?i)(?:^|[_-])epoch[_-]?0*(\d+)(?:[_-]|$)", path.stem)
    return int(match.group(1)) if match else None


def _resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint_list(path: Path) -> list[CheckpointPair]:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            records = json.load(handle)
        else:
            records = []
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [part.strip() for part in re.split(r"[,\t]", line)]
                if len(parts) != 3:
                    raise ValueError(
                        "{}:{} must contain epoch,appearance,structure".format(
                            path, line_number
                        )
                    )
                records.append(
                    {"epoch": int(parts[0]), "appearance": parts[1], "structure": parts[2]}
                )
    if not isinstance(records, list) or not records:
        raise ValueError("Checkpoint list must be a non-empty list: {}".format(path))
    pairs = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Each checkpoint-list item must be an object")
        pairs.append(
            CheckpointPair(
                int(record["epoch"]),
                _resolve_path(record["appearance"], path.parent),
                _resolve_path(record["structure"], path.parent),
            )
        )
    return validate_pairs(pairs)


def discover_checkpoint_pairs(
    checkpoint_dir: Path,
    checkpoint_pattern: str | None,
    epoch_step: int,
) -> list[CheckpointPair]:
    root = checkpoint_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates = list(root.glob(checkpoint_pattern)) if checkpoint_pattern else [
        path for suffix in ("*.pth", "*.pt") for path in root.rglob(suffix)
    ]
    grouped: dict[int, dict[str, list[Path]]] = {}
    for path in candidates:
        epoch = _epoch_from_name(path)
        branch = _branch_from_name(path)
        if epoch is None or branch is None:
            continue
        if epoch_step and epoch % epoch_step != 0:
            continue
        grouped.setdefault(epoch, {}).setdefault(branch, []).append(path.resolve())

    pairs = []
    for epoch, branches in sorted(grouped.items()):
        for branch in ("appearance", "structure"):
            found = branches.get(branch, [])
            if len(found) != 1:
                raise ValueError(
                    "Epoch {} has {} {} checkpoint(s): {}. Use --checkpoint-list "
                    "to resolve ambiguity.".format(epoch, len(found), branch, found)
                )
        pairs.append(CheckpointPair(epoch, branches["appearance"][0], branches["structure"][0]))
    if not pairs:
        raise FileNotFoundError(
            "No paired epoch checkpoints found below {}{}".format(
                root,
                " with pattern {!r}".format(checkpoint_pattern)
                if checkpoint_pattern else "",
            )
        )
    return validate_pairs(pairs)


def validate_pairs(pairs: Sequence[CheckpointPair]) -> list[CheckpointPair]:
    ordered = sorted(pairs, key=lambda item: item.epoch)
    epochs = [item.epoch for item in ordered]
    if len(epochs) != len(set(epochs)):
        raise ValueError("Duplicate epochs in checkpoint pairs: {}".format(epochs))
    for pair in ordered:
        if pair.epoch <= 0:
            raise ValueError("Checkpoint epochs must be positive")
        if not pair.appearance.is_file() or not pair.structure.is_file():
            raise FileNotFoundError("Missing checkpoint in epoch {}".format(pair.epoch))
    return ordered


def discover_best(checkpoint_dir: Path, branch: str) -> Path:
    root = checkpoint_dir.expanduser().resolve()
    branch_token = "result1" if branch == "appearance" else "result2"
    exact = [
        path.resolve()
        for path in root.rglob("best_{}_Jc_*.pth".format(branch_token))
        if "periodic_best" not in path.parts
    ]
    if not exact:
        raise FileNotFoundError(
            "Could not auto-discover best {}; pass --best-{}.".format(
                branch, "a" if branch == "appearance" else "s"
            )
        )
    parent_dirs = {path.parent for path in exact}
    if len(parent_dirs) > 1:
        raise ValueError(
            "Best {} checkpoints were found in multiple experiment directories: {}. "
            "Pass the exact --best-{} path.".format(
                branch,
                sorted(str(path) for path in parent_dirs),
                "a" if branch == "appearance" else "s",
            )
        )
    exact.sort(key=lambda item: (_score_from_name(item), item.stat().st_mtime))
    return exact[-1]


def build_validation_data(args: Any, cfg: Mapping[str, Any], device: torch.device) -> ValidationData:
    split = "validation" if args.val_split in ("val", "validation") else "test"
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else (
            args.dataset_root.expanduser().resolve()
            / "manifests"
            / "paper_seed_2026"
            / "{}.csv".format(split)
        )
    )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg["MEAN"], cfg["STD"])
    dual_view_cfg = dict(cfg["dual_view"])
    dual_view_cfg["enabled"] = True
    dataset = imagefloder_itn(
        manifest_path=str(manifest),
        data_transform_1=transforms["val" if split == "validation" else "test"],
        data_normalize_1=normalize,
        sup=True,
        dual_view_config=dual_view_cfg,
        training=False,
    )
    names = [Path(path).name for path in dataset.mask_paths]
    if len(names) != len(set(names)):
        raise ValueError("Validation image IDs are not unique")
    targets = []
    observed_names = []
    target_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for batch in tqdm(target_loader, desc="Loading validation targets", unit="batch"):
        targets.append(batch["mask"].cpu().numpy().astype(np.uint8))
        observed_names.extend(list(batch["ID"]))
    if observed_names != names:
        raise AssertionError("Dataset order changed while loading validation targets")
    targets_array = np.concatenate(targets, axis=0)
    if not np.isin(targets_array, (0, 1)).all():
        raise ValueError("This analysis requires binary GT labels 0/1")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return ValidationData(
        loader=loader,
        names=names,
        targets=targets_array,
        image_paths=[Path(path) for path in dataset.img_paths_1],
        manifest=manifest,
    )


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def _strip_uniform_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return dict(state)


def extract_state_dict(payload: Any, branch: str) -> dict[str, torch.Tensor]:
    if _is_state_dict(payload):
        state = dict(payload)
    elif isinstance(payload, Mapping):
        branch_aliases = (
            ("appearance", "model1", "student1", "appearance_state_dict", "model1_state_dict")
            if branch == "appearance"
            else ("structure", "model2", "student2", "structure_state_dict", "model2_state_dict")
        )
        state = None
        for key in branch_aliases + ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if _is_state_dict(candidate):
                state = dict(candidate)
                break
        if state is None:
            for alias in branch_aliases:
                prefix = alias + "."
                selected = {
                    key[len(prefix):]: value
                    for key, value in payload.items()
                    if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
                }
                if selected:
                    state = selected
                    break
        if state is None:
            raise KeyError("No {} state_dict found in checkpoint".format(branch))
    else:
        raise TypeError("Checkpoint payload must be a mapping")
    for prefix in ("module.", "model."):
        state = _strip_uniform_prefix(state, prefix)
    return state


def load_checkpoint_readonly(path: Path, branch: str) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return extract_state_dict(payload, branch)


def _validate_state_dict(model: torch.nn.Module, state: Mapping[str, torch.Tensor], path: Path) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    print(
        "Checkpoint compatibility | {} | missing={} unexpected={} shape_mismatch={}".format(
            path, len(missing), len(unexpected), len(mismatched)
        )
    )
    if missing:
        print("  missing keys:", missing[:20])
    if unexpected:
        print("  unexpected keys:", unexpected[:20])
    if mismatched:
        print("  shape mismatches:", mismatched[:20])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint/model mismatch for {}. Check --network; no keys were ignored.".format(path)
        )


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    elif isinstance(output, Mapping):
        for key in ("logits", "out", "prediction"):
            if key in output:
                output = output[key]
                break
    if not torch.is_tensor(output) or output.ndim != 4:
        raise TypeError("Expected logits [N,C,H,W], got {}".format(type(output)))
    if output.shape[1] != 2:
        raise ValueError("This analysis requires two output classes")
    return output


def _cache_key(path: Path, branch: str, network: str, manifest: Path, threshold_mode: str, threshold: float) -> str:
    stat = path.stat()
    text = "|".join(
        (str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns), branch, network,
         str(manifest.resolve()), threshold_mode, str(threshold))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def select_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.0
    best_jaccard = -1.0
    target = targets.astype(bool)
    for threshold in THRESHOLDS:
        prediction = probabilities > threshold
        tp = np.logical_and(prediction, target).sum(dtype=np.int64)
        fp = np.logical_and(prediction, ~target).sum(dtype=np.int64)
        fn = np.logical_and(~prediction, target).sum(dtype=np.int64)
        jaccard = float(tp / (tp + fp + fn + EPS))
        if jaccard > best_jaccard:
            best_threshold, best_jaccard = float(threshold), jaccard
    return best_threshold, best_jaccard


def infer_checkpoint(
    path: Path,
    branch: str,
    args: Any,
    cfg: Mapping[str, Any],
    validation: ValidationData,
    device: torch.device,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    key = _cache_key(
        path, branch, args.network, validation.manifest,
        args.threshold_mode, args.threshold,
    )
    cache_path = cache_dir / "{}_{}_{}.npz".format(path.stem, branch[0], key)
    if cache_path.is_file() and not args.no_cache:
        with np.load(cache_path, allow_pickle=False) as cached:
            names = cached["names"].tolist()
            if names != validation.names:
                raise RuntimeError("Cached image order mismatch: {}".format(cache_path))
            predictions = cached["predictions"].astype(bool)
            metadata = json.loads(str(cached["metadata"].item()))
        if predictions.shape != validation.targets.shape:
            raise RuntimeError("Cached prediction shape mismatch: {}".format(cache_path))
        metadata["cache_hit"] = True
        return predictions, metadata

    model = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    state = load_checkpoint_readonly(path, branch)
    _validate_state_dict(model, state, path)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    probabilities = []
    names = []
    with torch.inference_mode():
        for batch in tqdm(
            validation.loader,
            desc="Inference {} {}".format(branch[0].upper(), path.stem),
            unit="batch",
            leave=False,
        ):
            image_key = "image_appearance" if branch == "appearance" else "image_structure"
            if image_key not in batch:
                raise KeyError("Validation batch lacks {}".format(image_key))
            images = batch[image_key].to(device, non_blocking=True)
            logits = extract_logits(model(images))
            if tuple(logits.shape[-2:]) != tuple(batch["mask"].shape[-2:]):
                raise RuntimeError(
                    "Logit/GT resolution mismatch for {}: {} vs {}. Original test.py "
                    "does not resize logits, so this analysis refuses to invent a resize rule.".format(
                        path, tuple(logits.shape[-2:]), tuple(batch["mask"].shape[-2:])
                    )
                )
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            names.extend(list(batch["ID"]))
    del model, state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if names != validation.names:
        raise AssertionError("Prediction image order differs from validation manifest")
    probability_array = np.concatenate(probabilities, axis=0)
    if args.threshold_mode == "auto":
        threshold, threshold_jaccard = select_threshold(probability_array, validation.targets)
        threshold_source = "same validation split, project 0.02 Jaccard grid"
    else:
        threshold, threshold_jaccard = float(args.threshold), None
        threshold_source = "fixed CLI threshold"
    predictions = probability_array > threshold
    metadata = {
        "checkpoint": str(path.resolve()),
        "branch": branch,
        "network": args.network,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_search_jaccard": threshold_jaccard,
        "manifest": str(validation.manifest),
        "cache_hit": False,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        predictions=predictions.astype(np.uint8),
        names=np.asarray(validation.names),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return predictions, metadata


def confusion_counts(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    valid = np.ones(target.shape, dtype=bool) if mask is None else mask.astype(bool)
    return {
        "tp": int(np.logical_and.reduce((prediction, target, valid)).sum()),
        "tn": int(np.logical_and.reduce((~prediction, ~target, valid)).sum()),
        "fp": int(np.logical_and.reduce((prediction, ~target, valid)).sum()),
        "fn": int(np.logical_and.reduce((~prediction, target, valid)).sum()),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    return {
        **dict(counts),
        "iou": tp / (tp + fp + fn + EPS),
        "dice": 2 * tp / (2 * tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "sensitivity": tp / (tp + fn + EPS),
        "specificity": tn / (tn + fp + EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + EPS),
        "fpr": fp / (tn + fp + EPS),
        "fnr": fn / (tp + fn + EPS),
    }


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    return metrics_from_counts(confusion_counts(prediction, target, mask))


def region_masks(targets: np.ndarray, boundary_width: int) -> dict[str, np.ndarray]:
    gt = torch.from_numpy(targets.astype(np.float32)).unsqueeze(1)
    radius = int(boundary_width)
    padded = F.pad(gt, (radius, radius, radius, radius), value=0.0)
    eroded = -F.max_pool2d(-padded, kernel_size=2 * radius + 1, stride=1)
    foreground = targets.astype(bool)
    boundary = foreground & ~(eroded.squeeze(1).numpy() > 0.5)
    interior = foreground & ~boundary
    background = ~foreground
    if np.any(boundary & interior) or not np.array_equal(boundary | interior, foreground):
        raise AssertionError("Boundary/interior sanity check failed")
    return {
        "all": np.ones_like(foreground, dtype=bool),
        "foreground": foreground,
        "background": background,
        "boundary": boundary,
        "interior": interior,
    }


def complementarity_counts(pred_a: np.ndarray, pred_s: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int | float]:
    valid = np.ones_like(target, dtype=bool) if mask is None else mask.astype(bool)
    correct_a = pred_a == target
    correct_s = pred_s == target
    both_correct = correct_a & correct_s & valid
    a_only = correct_a & ~correct_s & valid
    s_only = ~correct_a & correct_s & valid
    both_wrong = ~correct_a & ~correct_s & valid
    total = int(valid.sum())
    counts = {
        "valid_pixels": total,
        "both_correct": int(both_correct.sum()),
        "a_only_correct": int(a_only.sum()),
        "s_only_correct": int(s_only.sum()),
        "both_wrong": int(both_wrong.sum()),
        "disagreement": int(((pred_a != pred_s) & valid).sum()),
    }
    if sum(counts[key] for key in ("both_correct", "a_only_correct", "s_only_correct", "both_wrong")) != total:
        raise AssertionError("Complementarity partition does not cover valid pixels")
    for key in ("both_correct", "a_only_correct", "s_only_correct", "both_wrong", "disagreement"):
        counts[key + "_ratio"] = counts[key] / (total + EPS)
    return counts


def regional_expertise(pred_a: np.ndarray, pred_s: np.ndarray, target: np.ndarray, regions: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name in ("foreground", "background", "boundary", "interior"):
        mask = regions[name]
        correct_a = (pred_a == target) & mask
        correct_s = (pred_s == target) & mask
        a_wins = int((correct_a & ~correct_s).sum())
        s_wins = int((~correct_a & correct_s).sum())
        disagreements = a_wins + s_wins
        result.update({
            name + "_pixels": int(mask.sum()),
            name + "_A_accuracy": float(correct_a.sum() / (mask.sum() + EPS)),
            name + "_S_accuracy": float(correct_s.sum() / (mask.sum() + EPS)),
            name + "_A_only_correct": a_wins,
            name + "_S_only_correct": s_wins,
            name + "_disagreements": disagreements,
            name + "_A_win_rate": a_wins / (disagreements + EPS),
            name + "_S_win_rate": s_wins / (disagreements + EPS),
        })
    return result


def oracle_prediction(predictions: Iterable[np.ndarray], target: np.ndarray) -> np.ndarray:
    prediction_list = list(predictions)
    if not prediction_list:
        raise ValueError("Oracle requires at least one prediction")
    any_correct = np.zeros_like(target, dtype=bool)
    for prediction in prediction_list:
        any_correct |= prediction == target
    # Binary task: if no candidate is correct, every candidate equals 1-GT.
    return np.where(any_correct, target, ~target.astype(bool)).astype(bool)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError("Cannot infer CSV fields from empty rows")
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_image_means(values: Mapping[str, Sequence[float]], iterations: int, seed: int) -> dict[str, Any]:
    arrays = {key: np.asarray(item, dtype=np.float64) for key, item in values.items()}
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Bootstrap metrics must have the same non-zero image count")
    size = next(iter(lengths))
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(iterations, dtype=np.float64) for key in arrays}
    for iteration in tqdm(range(iterations), desc="Image-level bootstrap", unit="resample"):
        indices = rng.integers(0, size, size=size)
        for key, array in arrays.items():
            samples[key][iteration] = array[indices].mean()
    return {
        key: {
            "mean": float(array.mean()),
            "ci95_low": float(np.percentile(samples[key], 2.5)),
            "ci95_high": float(np.percentile(samples[key], 97.5)),
            "sampling_unit": "image",
            "num_images": size,
            "iterations": iterations,
            "seed": seed,
        }
        for key, array in arrays.items()
    }
