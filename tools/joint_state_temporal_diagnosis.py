"""Closed offline temporal CPS joint-state diagnosis.

The script evaluates synchronized A/B milestone payloads from one CPS run.
It reuses the E0 state definitions and the project's original evaluation
metric, dataloader, preprocessing, normalization, and model construction.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from config.eval_config.eval import evaluate  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from models.getnetwork import get_network  # noqa: E402
from test import extract_logits  # noqa: E402
from joint_state_error_diagnosis import (  # noqa: E402
    mean_or_nan,
    safe_divide,
    state_masks,
    write_csv,
)


DEFAULT_MILESTONE_DIR = (
    ROOT
    / "/data/mission/models/Exp/DUAL/checkpoints/semi_10_seed_24_rgbstain_dis_piexl_entropy/GLaS/"
    "CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image-input_mode=rgb_stain-cps_mode=entropy_routed/milestones"
)
DEFAULT_DATA_ROOT = ROOT / "datasets/GLaS"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics/joint_state_temporal_rgbstain_entropy_routed"
THRESHOLDS = (0.70, 0.80, 0.90, 0.95)

SUMMARY_COLUMNS = [
    "epoch", "dice_a", "dice_b", "jaccard_a", "jaccard_b",
    "metric_threshold_a", "metric_threshold_b", "agreement_rate",
    "disagreement_rate", "both_correct_rate_total", "both_wrong_rate_total",
    "both_wrong_rate_in_agreement", "shared_error_fraction_of_all_error_pixels",
    "a_correct_rate_in_disagreement", "b_correct_rate_in_disagreement",
    "shared_false_negative_pixels", "shared_false_positive_pixels",
    "shared_fn_fraction_of_both_wrong", "shared_fp_fraction_of_both_wrong",
    "mean_conf_both_correct", "mean_conf_both_wrong",
    "high_conf_shared_error_rate_070", "high_conf_shared_error_rate_080",
    "high_conf_shared_error_rate_090", "high_conf_shared_error_rate_095",
]

DELTA_COLUMNS = [
    "epoch", "delta_agreement_rate", "delta_disagreement_rate",
    "delta_both_wrong_rate_total", "delta_both_wrong_rate_in_agreement",
    "delta_shared_error_fraction", "delta_high_conf_shared_error_rate_090",
    "delta_jaccard_a", "delta_jaccard_b",
]

CORRELATION_COLUMNS = [
    "metric_x", "metric_y", "num_epochs", "pearson", "spearman",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-dir", default=str(DEFAULT_MILESTONE_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--network", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def epoch_from_payload(payload: Mapping[str, object], path: Path) -> int | None:
    value = payload.get("epoch")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    match = re.search(r"epoch[_-](\d+)", path.stem.lower())
    return int(match.group(1)) if match else None


def scan_synchronized_milestones(directory: Path, logger: logging.Logger) -> List[Tuple[int, Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Milestone directory not found: {directory}")
    candidates: Dict[int, List[Path]] = {}
    identified_epochs = set()
    for path in sorted(directory.glob("*.pth")):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            logger.info("[SKIP] file=%s: cannot read checkpoint (%s)", path, exc)
            continue
        if not isinstance(payload, dict):
            logger.info("[SKIP] file=%s: checkpoint payload is not a dict", path)
            continue
        epoch = epoch_from_payload(payload, path)
        if epoch is None:
            logger.info("[SKIP] file=%s: epoch cannot be identified", path)
            continue
        identified_epochs.add(epoch)
        has_a = isinstance(payload.get("appearance_state_dict"), dict)
        has_b = isinstance(payload.get("structure_state_dict"), dict)
        if not (has_a and has_b):
            logger.info("[SKIP] epoch=%02d: synchronized checkpoint pair incomplete", epoch)
            continue
        candidates.setdefault(epoch, []).append(path)
        logger.info(
            "[FOUND] epoch=%02d file=%s A=appearance_state_dict B=structure_state_dict",
            epoch, path,
        )

    synchronized = []
    for epoch, paths in sorted(candidates.items()):
        if len(paths) != 1:
            logger.info(
                "[SKIP] epoch=%02d: synchronized checkpoint pair ambiguous (%d payloads)",
                epoch, len(paths),
            )
            continue
        synchronized.append((epoch, paths[0]))
    # The requested run saves milestones every 10 epochs through epoch 200.
    # Report a completely missing payload with the same explicit skip message
    # used for an existing but incomplete payload.
    expected_epochs = set(range(10, 201, 10))
    if 0 in identified_epochs:
        expected_epochs.add(0)
    for epoch in sorted(expected_epochs - identified_epochs):
        logger.info("[SKIP] epoch=%02d: synchronized checkpoint pair incomplete", epoch)
    return synchronized


def load_models_from_milestone(
    path: Path,
    network: str,
    cfg: Mapping[str, object],
    device: torch.device,
):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Milestone payload is not a dict: {path}")
    if not isinstance(payload.get("appearance_state_dict"), dict):
        raise KeyError(f"Missing Model A appearance_state_dict: {path}")
    if not isinstance(payload.get("structure_state_dict"), dict):
        raise KeyError(f"Missing Model B structure_state_dict: {path}")
    model_a = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_b = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_a.load_state_dict(payload["appearance_state_dict"], strict=True)
    model_b.load_state_dict(payload["structure_state_dict"], strict=True)
    model_a.to(device).eval()
    model_b.to(device).eval()
    return model_a, model_b, payload


def metric_arrays(score_parts: List[torch.Tensor], mask_parts: List[torch.Tensor]):
    scores = torch.cat(score_parts, dim=0)
    masks = torch.cat(mask_parts, dim=0)
    return evaluate(scores, masks)


def correlation(x: Sequence[float], y: Sequence[float]) -> Tuple[int, float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if x_array.size < 2 or np.all(x_array == x_array[0]) or np.all(y_array == y_array[0]):
        return int(x_array.size), float("nan"), float("nan")
    pearson_result = pearsonr(x_array, y_array)
    spearman_result = spearmanr(x_array, y_array)
    pearson_value = getattr(pearson_result, "statistic", pearson_result[0])
    spearman_value = getattr(spearman_result, "statistic", spearman_result[0])
    return (
        int(x_array.size),
        float(pearson_value),
        float(spearman_value),
    )


def plot_single(output_dir: Path, filename: str, epochs: Sequence[int], series: Sequence[Tuple[str, Sequence[float]]], ylabel: str) -> None:
    figure = plt.figure(figsize=(8.5, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    for label, values in series:
        axis.plot(epochs, values, marker="o", linewidth=2, markersize=4, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)


def print_epoch_block(title: str, row: Mapping[str, object]) -> None:
    print(f"{title}:")
    print(f"epoch = {row['epoch']}")
    print(f"agreement = {float(row['agreement_rate']):.6f}")
    print(f"disagreement = {float(row['disagreement_rate']):.6f}")
    print(f"both_wrong_total = {float(row['both_wrong_rate_total']):.6f}")
    print(f"shared_error_fraction = {float(row['shared_error_fraction_of_all_error_pixels']):.6f}")
    print(f"high_conf_shared_error_090 = {float(row['high_conf_shared_error_rate_090']):.6f}")
    print(f"Jc_A = {float(row['jaccard_a']):.6f}")
    print(f"Jc_B = {float(row['jaccard_b']):.6f}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cps_joint_state_temporal")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(output_dir / "joint_state_temporal.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    milestone_dir = Path(args.milestone_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    synchronized = scan_synchronized_milestones(milestone_dir, logger)
    if not synchronized:
        raise RuntimeError(f"No synchronized milestone payloads found in {milestone_dir}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = dataset_cfg("GlaS")
    transforms = data_transform_2d()
    evaluation_transform = transforms["val" if args.split == "validation" else "test"]
    normalize = data_normalize_2d(cfg["MEAN"], cfg["STD"])
    manifest_path = data_root / "manifests/paper_seed_2026" / f"{args.split}.csv"
    dataset = imagefloder_itn(
        manifest_path=str(manifest_path),
        data_transform_1=evaluation_transform,
        data_normalize_1=normalize,
        sup=True,
        input_mode="rgb_rgb",
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    logger.info("milestone_dir=%s", milestone_dir)
    logger.info("dataset_root=%s", data_root)
    logger.info("split=%s", args.split)
    logger.info("manifest=%s", manifest_path)
    logger.info("device=%s", device)
    logger.info("num_images=%d", len(dataset))
    logger.info("evaluation_transform=project data_transform_2d[%s]: Resize(256,256)", args.split)
    logger.info("normalization=project data_normalize_2d(GLaS MEAN, STD)")
    logger.info("input_mode=rgb_rgb; both models receive the same project image tensor")
    logger.info("inference=model.eval() + torch.no_grad(); threshold=probability >= 0.5")
    logger.info("ignore_rule=project dataset converts binary mask to (mask > 0); no ignore label is exposed")
    logger.info("synchronized_epochs=%s", [epoch for epoch, _ in synchronized])

    rows: List[Dict[str, object]] = []
    for epoch, milestone_path in synchronized:
        try:
            first_payload = torch.load(milestone_path, map_location="cpu")
            metadata_network = first_payload.get("network") if isinstance(first_payload, dict) else None
            network = args.network or metadata_network
            if not isinstance(network, str) or not network:
                raise ValueError("network is absent from milestone metadata; pass --network")
            if args.network and metadata_network and args.network != metadata_network:
                raise ValueError(
                    f"requested network={args.network} differs from checkpoint metadata={metadata_network}"
                )
            model_a, model_b, payload = load_models_from_milestone(
                milestone_path, network, cfg, device
            )
            actual_epoch = epoch_from_payload(payload, milestone_path)
            if actual_epoch != epoch:
                raise ValueError(f"epoch mismatch: scanned={epoch}, payload={actual_epoch}")
        except Exception as exc:
            logger.info("[SKIP] epoch=%02d: synchronized checkpoint strict load failed (%s)", epoch, exc)
            continue

        logger.info("[LOAD] epoch=%02d", epoch)
        logger.info("Model A: %s", milestone_path)
        logger.info("Model B: %s", milestone_path)
        logger.info("Model A state key: appearance_state_dict")
        logger.info("Model B state key: structure_state_dict")
        logger.info("strict_load_A=True")
        logger.info("strict_load_B=True")

        totals = {
            "total_pixels": 0,
            "agreement_pixels": 0,
            "disagreement_pixels": 0,
            "both_correct_pixels": 0,
            "both_wrong_pixels": 0,
            "a_correct_b_wrong_pixels": 0,
            "a_wrong_b_correct_pixels": 0,
            "any_model_wrong_pixels": 0,
            "shared_false_negative_pixels": 0,
            "shared_false_positive_pixels": 0,
        }
        confidence_parts = {"both_correct": [], "both_wrong": []}
        high_conf_counts = {
            threshold: {"both_wrong": 0, "both_correct": 0}
            for threshold in THRESHOLDS
        }
        score_parts_a: List[torch.Tensor] = []
        score_parts_b: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                logits_a = extract_logits(model_a(images))
                logits_b = extract_logits(model_b(images))
                score_parts_a.append(logits_a.detach().cpu())
                score_parts_b.append(logits_b.detach().cpu())
                mask_parts.append(batch["mask"].detach().cpu())
                prob_a_batch = torch.softmax(logits_a, dim=1)[:, 1].cpu().numpy()
                prob_b_batch = torch.softmax(logits_b, dim=1)[:, 1].cpu().numpy()
                gt_batch = batch["mask"].cpu().numpy().astype(bool)

                for index in range(images.shape[0]):
                    prob_a = prob_a_batch[index]
                    prob_b = prob_b_batch[index]
                    gt = gt_batch[index]
                    pred_a = prob_a >= 0.5
                    pred_b = prob_b >= 0.5
                    states = state_masks(pred_a, pred_b, gt)
                    conf_a = np.maximum(prob_a, 1.0 - prob_a)
                    conf_b = np.maximum(prob_b, 1.0 - prob_b)
                    joint_mean_conf = (conf_a + conf_b) / 2.0
                    both_correct_mask = states["both_correct"]
                    both_wrong_mask = states["both_wrong"]
                    confidence_parts["both_correct"].append(joint_mean_conf[both_correct_mask])
                    confidence_parts["both_wrong"].append(joint_mean_conf[both_wrong_mask])

                    totals["total_pixels"] += int(gt.size)
                    totals["agreement_pixels"] += int(states["agree"].sum())
                    totals["disagreement_pixels"] += int(states["disagree"].sum())
                    totals["both_correct_pixels"] += int(both_correct_mask.sum())
                    totals["both_wrong_pixels"] += int(both_wrong_mask.sum())
                    totals["a_correct_b_wrong_pixels"] += int(states["a_correct_b_wrong"].sum())
                    totals["a_wrong_b_correct_pixels"] += int(states["a_wrong_b_correct"].sum())
                    totals["any_model_wrong_pixels"] += int(((pred_a != gt) | (pred_b != gt)).sum())
                    totals["shared_false_negative_pixels"] += int(states["shared_false_negative"].sum())
                    totals["shared_false_positive_pixels"] += int(states["shared_false_positive"].sum())

                    for threshold in THRESHOLDS:
                        high_conf_counts[threshold]["both_wrong"] += int(
                            (both_wrong_mask & (conf_a >= threshold) & (conf_b >= threshold)).sum()
                        )
                        high_conf_counts[threshold]["both_correct"] += int(
                            (both_correct_mask & (conf_a >= threshold) & (conf_b >= threshold)).sum()
                        )

        metric_a = metric_arrays(score_parts_a, mask_parts)
        metric_b = metric_arrays(score_parts_b, mask_parts)
        both_correct_conf = np.concatenate(confidence_parts["both_correct"]) if confidence_parts["both_correct"] else np.array([])
        both_wrong_conf = np.concatenate(confidence_parts["both_wrong"]) if confidence_parts["both_wrong"] else np.array([])
        row: Dict[str, object] = {
            "epoch": epoch,
            "dice_a": float(metric_a[2]),
            "dice_b": float(metric_b[2]),
            "jaccard_a": float(metric_a[1]),
            "jaccard_b": float(metric_b[1]),
            "metric_threshold_a": float(metric_a[0]),
            "metric_threshold_b": float(metric_b[0]),
            "agreement_rate": safe_divide(totals["agreement_pixels"], totals["total_pixels"]),
            "disagreement_rate": safe_divide(totals["disagreement_pixels"], totals["total_pixels"]),
            "both_correct_rate_total": safe_divide(totals["both_correct_pixels"], totals["total_pixels"]),
            "both_wrong_rate_total": safe_divide(totals["both_wrong_pixels"], totals["total_pixels"]),
            "both_wrong_rate_in_agreement": safe_divide(totals["both_wrong_pixels"], totals["agreement_pixels"]),
            "shared_error_fraction_of_all_error_pixels": safe_divide(totals["both_wrong_pixels"], totals["any_model_wrong_pixels"]),
            "a_correct_rate_in_disagreement": safe_divide(totals["a_correct_b_wrong_pixels"], totals["disagreement_pixels"]),
            "b_correct_rate_in_disagreement": safe_divide(totals["a_wrong_b_correct_pixels"], totals["disagreement_pixels"]),
            "shared_false_negative_pixels": totals["shared_false_negative_pixels"],
            "shared_false_positive_pixels": totals["shared_false_positive_pixels"],
            "shared_fn_fraction_of_both_wrong": safe_divide(totals["shared_false_negative_pixels"], totals["both_wrong_pixels"]),
            "shared_fp_fraction_of_both_wrong": safe_divide(totals["shared_false_positive_pixels"], totals["both_wrong_pixels"]),
            "mean_conf_both_correct": mean_or_nan(both_correct_conf),
            "mean_conf_both_wrong": mean_or_nan(both_wrong_conf),
        }
        for threshold in THRESHOLDS:
            suffix = f"{int(round(threshold * 100)):03d}"
            row[f"high_conf_shared_error_rate_{suffix}"] = safe_divide(
                high_conf_counts[threshold]["both_wrong"], totals["both_wrong_pixels"]
            )
        rows.append(row)
        del model_a, model_b

    if not rows:
        raise RuntimeError("No milestone survived strict loading and evaluation")
    rows.sort(key=lambda row: int(row["epoch"]))
    write_csv(output_dir / "joint_state_temporal_summary.csv", SUMMARY_COLUMNS, rows)

    by_epoch = {int(row["epoch"]): row for row in rows}
    delta_rows = []
    delta_fields = {
        "delta_agreement_rate": "agreement_rate",
        "delta_disagreement_rate": "disagreement_rate",
        "delta_both_wrong_rate_total": "both_wrong_rate_total",
        "delta_both_wrong_rate_in_agreement": "both_wrong_rate_in_agreement",
        "delta_shared_error_fraction": "shared_error_fraction_of_all_error_pixels",
        "delta_high_conf_shared_error_rate_090": "high_conf_shared_error_rate_090",
        "delta_jaccard_a": "jaccard_a",
        "delta_jaccard_b": "jaccard_b",
    }
    for row in rows:
        epoch = int(row["epoch"])
        previous = by_epoch.get(epoch - 10)
        delta = {"epoch": epoch}
        for output_name, source_name in delta_fields.items():
            delta[output_name] = (
                float(row[source_name]) - float(previous[source_name])
                if previous is not None and math.isfinite(float(row[source_name])) and math.isfinite(float(previous[source_name]))
                else float("nan")
            )
        delta_rows.append(delta)
    write_csv(output_dir / "joint_state_temporal_delta.csv", DELTA_COLUMNS, delta_rows)

    correlation_pairs = [
        ("agreement_rate", "shared_error_fraction_of_all_error_pixels"),
        ("agreement_rate", "both_wrong_rate_total"),
        ("disagreement_rate", "shared_error_fraction_of_all_error_pixels"),
        ("mean_jaccard", "both_wrong_rate_total"),
        ("mean_jaccard", "high_conf_shared_error_rate_090"),
    ]
    correlation_rows = []
    for x_name, y_name in correlation_pairs:
        x_values = []
        y_values = []
        for row in rows:
            x_values.append(
                (float(row["jaccard_a"]) + float(row["jaccard_b"])) / 2.0
                if x_name == "mean_jaccard" else float(row[x_name])
            )
            y_values.append(float(row[y_name]))
        count, pearson, spearman = correlation(x_values, y_values)
        correlation_rows.append({
            "metric_x": x_name,
            "metric_y": y_name,
            "num_epochs": count,
            "pearson": pearson,
            "spearman": spearman,
        })
    write_csv(output_dir / "joint_state_temporal_correlations.csv", CORRELATION_COLUMNS, correlation_rows)

    epochs = [int(row["epoch"]) for row in rows]
    plot_single(output_dir, "fig_agreement_disagreement_vs_epoch.png", epochs, [
        ("agreement_rate", [row["agreement_rate"] for row in rows]),
        ("disagreement_rate", [row["disagreement_rate"] for row in rows]),
    ], "Rate")
    plot_single(output_dir, "fig_shared_error_vs_epoch.png", epochs, [
        ("both_wrong_rate_total", [row["both_wrong_rate_total"] for row in rows]),
        ("both_wrong_rate_in_agreement", [row["both_wrong_rate_in_agreement"] for row in rows]),
    ], "Rate")
    plot_single(output_dir, "fig_shared_error_fraction_vs_epoch.png", epochs, [
        ("shared_error_fraction_of_all_error_pixels", [row["shared_error_fraction_of_all_error_pixels"] for row in rows]),
    ], "Fraction")
    plot_single(output_dir, "fig_high_conf_shared_error_vs_epoch.png", epochs, [
        ("high_conf_shared_error_rate_090", [row["high_conf_shared_error_rate_090"] for row in rows]),
    ], "Fraction")
    plot_single(output_dir, "fig_model_performance_vs_epoch.png", epochs, [
        ("jaccard_a", [row["jaccard_a"] for row in rows]),
        ("jaccard_b", [row["jaccard_b"] for row in rows]),
    ], "Jaccard")
    plot_single(output_dir, "fig_shared_error_type_vs_epoch.png", epochs, [
        ("shared_fn_fraction_of_both_wrong", [row["shared_fn_fraction_of_both_wrong"] for row in rows]),
        ("shared_fp_fraction_of_both_wrong", [row["shared_fp_fraction_of_both_wrong"] for row in rows]),
    ], "Fraction")

    early = rows[0]
    middle = by_epoch.get(100, rows[len(rows) // 2])
    final = rows[-1]
    logger.info("valid_synchronized_epochs=%s", epochs)
    for item in correlation_rows:
        logger.info(
            "correlation %s vs %s: n=%d pearson=%s spearman=%s",
            item["metric_x"], item["metric_y"], item["num_epochs"],
            item["pearson"], item["spearman"],
        )

    print("========== CPS TEMPORAL JOINT-STATE DIAGNOSIS ==========")
    print()
    print(f"Valid synchronized epochs: {epochs}")
    print(f"Number of epochs: {len(rows)}")
    print()
    print_epoch_block("Early epoch", early)
    print()
    print_epoch_block("Middle epoch", middle)
    print()
    print_epoch_block("Final epoch", final)
    print()
    print("Temporal change:")
    for label, field in (
        ("agreement", "agreement_rate"),
        ("disagreement", "disagreement_rate"),
        ("both_wrong_total", "both_wrong_rate_total"),
        ("shared_error_fraction", "shared_error_fraction_of_all_error_pixels"),
        ("high_conf_shared_error_090", "high_conf_shared_error_rate_090"),
    ):
        print(f"{label}: {float(early[field]):.6f} -> {float(final[field]):.6f} (delta {float(final[field]) - float(early[field]):+.6f})")
    print()
    if float(final["agreement_rate"]) > float(early["agreement_rate"]):
        print("Agreement increased during CPS training.")
    else:
        print("Agreement did not increase during CPS training under this evaluation.")
    if float(final["disagreement_rate"]) < float(early["disagreement_rate"]):
        print("Disagreement decreased during CPS training.")
    else:
        print("Disagreement did not decrease during CPS training under this evaluation.")
    if float(final["shared_error_fraction_of_all_error_pixels"]) > 0.5:
        print("Most remaining joint-error pixels are shared errors rather than disagreements.")
    else:
        print("Shared errors are not the majority of remaining joint-error pixels at the final epoch.")
    if float(final["high_conf_shared_error_rate_090"]) > 0.5:
        print("More than half of shared errors are simultaneously high-confidence predictions from both models.")
    else:
        print("At most half of shared errors are simultaneously high-confidence predictions from both models.")
    print()
    print("=========================================================")


if __name__ == "__main__":
    main()
