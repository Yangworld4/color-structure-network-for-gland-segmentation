"""Offline validation of historical recovery and heterogeneous-student dynamics.

This program is analysis-only. It never trains, backpropagates, creates an
optimizer, updates a checkpoint, or writes outside its dedicated output path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from config.dataset_config.dataset_cfg import dataset_cfg

from analysis_progressive_teacher.plots import (
    plot_complementarity,
    plot_cumulative,
    plot_leader,
    save_qualitative,
)
from analysis_progressive_teacher.utils_analysis import (
    EPS,
    CheckpointPair,
    bootstrap_image_means,
    build_validation_data,
    complementarity_counts,
    discover_best,
    discover_checkpoint_pairs,
    infer_checkpoint,
    load_checkpoint_list,
    oracle_prediction,
    region_masks,
    regional_expertise,
    resolve_device,
    seed_everything,
    segmentation_metrics,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        choices=("GlaS", "GLaS", "CRAG", "MoNuSeg"),
        default="GlaS",
        help="GlaS and GLaS are accepted as aliases",
    )
    parser.add_argument("--val-split", choices=("val", "validation", "test"), default="val")
    parser.add_argument("--manifest", type=Path, help="explicit validation/test manifest")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--best-a", type=Path, help="final best Appearance checkpoint")
    parser.add_argument("--best-s", type=Path, help="final best Structure checkpoint")
    parser.add_argument(
        "--checkpoint-list",
        type=Path,
        help="JSON list or text rows: epoch,appearance,structure (paths may be relative to this file)",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        help="recursive/relative glob below checkpoint-dir; filenames must encode epoch and branch",
    )
    parser.add_argument("--epoch-step", type=int, default=10)
    parser.add_argument("--network", default="xnet_sb", help="must exactly match checkpoint architecture")
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--gpu", default="0", help="CUDA index, cpu, or auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--threshold-mode",
        choices=("auto", "fixed"),
        default="auto",
        help="auto reproduces project validation: per-checkpoint Jaccard-optimal 0.02 grid",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="used only with --threshold-mode fixed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/progressive_teacher_validation"),
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore compatible prediction caches")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--save-qualitative", action="store_true")
    parser.add_argument("--qualitative-count", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--leader-tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--stable-leader-fraction", type=float, default=0.90)
    parser.add_argument("--expertise-win-threshold", type=float, default=0.60)
    parser.add_argument("--high-hrr-threshold", type=float, default=0.20)
    args = parser.parse_args()
    if args.dataset_name == "GLaS":
        args.dataset_name = "GlaS"
    if args.epoch_step < 1 or args.boundary_width < 1:
        parser.error("--epoch-step and --boundary-width must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("invalid batch size or worker count")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0,1]")
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive")
    if args.qualitative_count < 1:
        parser.error("--qualitative-count must be positive")
    for name in ("stable_leader_fraction", "expertise_win_threshold", "high_hrr_threshold"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error("--{} must be in [0,1]".format(name.replace("_", "-")))
    if args.leader_tie_tolerance < 0:
        parser.error("--leader-tie-tolerance must be non-negative")
    return args


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[CheckpointPair], str]:
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    best_a = args.best_a.expanduser().resolve() if args.best_a else discover_best(checkpoint_dir, "appearance")
    best_s = args.best_s.expanduser().resolve() if args.best_s else discover_best(checkpoint_dir, "structure")
    for path in (best_a, best_s):
        if not path.is_file():
            raise FileNotFoundError(path)
    pairs = (
        load_checkpoint_list(args.checkpoint_list)
        if args.checkpoint_list
        else discover_checkpoint_pairs(checkpoint_dir, args.checkpoint_pattern, args.epoch_step)
    )
    names = [pair.appearance.name.lower() for pair in pairs] + [pair.structure.name.lower() for pair in pairs]
    semantics = (
        "best_so_far_snapshots_inferred_from_filenames"
        if all("best" in name for name in names)
        else "epoch_snapshots_or_user_supplied; verify checkpoint provenance"
    )
    return best_a, best_s, pairs, semantics


def _per_image_metrics(prediction: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    return [segmentation_metrics(prediction[index], target[index]) for index in range(len(target))]


def _oracle_metrics(predictions: Sequence[np.ndarray], target: np.ndarray) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    oracle = oracle_prediction(predictions, target)
    return oracle, segmentation_metrics(oracle, target), _per_image_metrics(oracle, target)


def analyze_current(
    pred_a: np.ndarray,
    pred_s: np.ndarray,
    targets: np.ndarray,
    metadata_a: Mapping[str, Any],
    metadata_s: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metrics_a = segmentation_metrics(pred_a, targets)
    metrics_s = segmentation_metrics(pred_s, targets)
    complementarity = complementarity_counts(pred_a, pred_s, targets)
    both_wrong = (pred_a != targets) & (pred_s != targets)
    current_oracle, oracle_metrics, oracle_per_image = _oracle_metrics((pred_a, pred_s), targets)
    if int(both_wrong.sum()) != complementarity["both_wrong"]:
        raise AssertionError("Current Both-Wrong mask/count mismatch")
    summary = {
        "appearance": {**metrics_a, "inference": dict(metadata_a)},
        "structure": {**metrics_s, "inference": dict(metadata_s)},
        "complementarity": complementarity,
        "current_oracle": {
            **oracle_metrics,
            "gain_over_best_single_branch": oracle_metrics["dice"] - max(metrics_a["dice"], metrics_s["dice"]),
            "warning": "Analysis-only theoretical upper bound using validation GT; not deployable.",
        },
    }
    return summary, both_wrong, current_oracle, oracle_per_image


def analyze_recovery(
    pairs: Sequence[CheckpointPair],
    history: Mapping[int, tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    both_wrong: np.ndarray,
    names: Sequence[str],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    shape = targets.shape
    first_a = np.full(shape, -1, dtype=np.int32)
    first_s = np.full(shape, -1, dtype=np.int32)
    last_a = np.full(shape, -1, dtype=np.int32)
    last_s = np.full(shape, -1, dtype=np.int32)
    count_a = np.zeros(shape, dtype=np.uint16)
    count_s = np.zeros(shape, dtype=np.uint16)
    count_union = np.zeros(shape, dtype=np.uint16)
    for pair in tqdm(pairs, desc="Historical recovery", unit="pair"):
        pred_a, pred_s = history[pair.epoch]
        correct_a = (pred_a == targets) & both_wrong
        correct_s = (pred_s == targets) & both_wrong
        first_a[(first_a < 0) & correct_a] = pair.epoch
        first_s[(first_s < 0) & correct_s] = pair.epoch
        last_a[correct_a] = pair.epoch
        last_s[correct_s] = pair.epoch
        count_a += correct_a.astype(np.uint16)
        count_s += correct_s.astype(np.uint16)
        count_union += (correct_a | correct_s).astype(np.uint16)
    ever_a = (count_a > 0) & both_wrong
    ever_s = (count_s > 0) & both_wrong
    recoverable = (ever_a | ever_s) & both_wrong
    persistent = both_wrong & ~recoverable
    denominator = int(both_wrong.sum())
    summary = {
        "current_both_wrong_count": denominator,
        "A_recoverable_count": int(ever_a.sum()),
        "S_recoverable_count": int(ever_s.sum()),
        "union_recoverable_count": int(recoverable.sum()),
        "persistent_count": int(persistent.sum()),
        "HRR_A": float(ever_a.sum() / (denominator + EPS)),
        "HRR_S": float(ever_s.sum() / (denominator + EPS)),
        "HRR_union": float(recoverable.sum() / (denominator + EPS)),
        "PBR": float(persistent.sum() / (denominator + EPS)),
        "num_historical_pairs": len(pairs),
        "fixed_mask_definition": "Both wrong under final best Appearance and final best Structure",
    }
    if denominator and not np.isclose(summary["PBR"], 1.0 - summary["HRR_union"], atol=1e-8):
        raise AssertionError("PBR != 1 - HRR_union")
    per_image = []
    for index, name in enumerate(names):
        total = int(both_wrong[index].sum())
        a_count = int(ever_a[index].sum())
        s_count = int(ever_s[index].sum())
        union_count = int(recoverable[index].sum())
        persistent_count = int(persistent[index].sum())
        per_image.append({
            "image_name": name,
            "current_both_wrong_count": total,
            "A_recoverable_count": a_count,
            "S_recoverable_count": s_count,
            "union_recoverable_count": union_count,
            "persistent_count": persistent_count,
            "HRR_A": a_count / (total + EPS),
            "HRR_S": s_count / (total + EPS),
            "HRR_union": union_count / (total + EPS),
            "PBR": persistent_count / (total + EPS),
        })
    np.savez_compressed(
        output_dir / "both_wrong_temporal_analysis.npz",
        image_names=np.asarray(names),
        epochs=np.asarray([pair.epoch for pair in pairs], dtype=np.int32),
        current_both_wrong=both_wrong.astype(np.uint8),
        first_recovery_epoch_A=first_a,
        first_recovery_epoch_S=first_s,
        last_correct_epoch_A=last_a,
        last_correct_epoch_S=last_s,
        correct_count_A=count_a,
        correct_count_S=count_s,
        union_correct_count=count_union,
        A_correct_frequency=count_a.astype(np.float32) / max(len(pairs), 1),
        S_correct_frequency=count_s.astype(np.float32) / max(len(pairs), 1),
        union_correct_frequency=count_union.astype(np.float32) / max(len(pairs), 1),
        recoverable_mask=recoverable.astype(np.uint8),
        persistent_mask=persistent.astype(np.uint8),
    )
    arrays = {
        "ever_a": ever_a,
        "ever_s": ever_s,
        "recoverable": recoverable,
        "persistent": persistent,
        "count_a": count_a,
        "count_s": count_s,
        "count_union": count_union,
    }
    return summary, per_image, arrays


def cumulative_recovery(
    pairs: Sequence[CheckpointPair],
    history: Mapping[int, tuple[np.ndarray, np.ndarray]],
    best: tuple[np.ndarray, np.ndarray],
    targets: np.ndarray,
    both_wrong: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    sequences = {
        "chronological": [(str(pair.epoch), history[pair.epoch]) for pair in pairs] + [("best", best)],
        "reverse": [("best", best)] + [(str(pair.epoch), history[pair.epoch]) for pair in reversed(pairs)],
    }
    denominator = int(both_wrong.sum())
    for direction, sequence in sequences.items():
        ever_a = np.zeros_like(both_wrong)
        ever_s = np.zeros_like(both_wrong)
        included = []
        for step, (label, (pred_a, pred_s)) in enumerate(sequence, 1):
            included.append(label)
            ever_a |= (pred_a == targets) & both_wrong
            ever_s |= (pred_s == targets) & both_wrong
            union = ever_a | ever_s
            rows.append({
                "direction": direction,
                "num_checkpoint_pairs": step,
                "included_epochs": json.dumps(included),
                "HRR_A": float(ever_a.sum() / (denominator + EPS)),
                "HRR_S": float(ever_s.sum() / (denominator + EPS)),
                "HRR_union": float(union.sum() / (denominator + EPS)),
                "persistent_rate": float((both_wrong & ~union).sum() / (denominator + EPS)),
            })
    return rows


def analyze_leaders(
    pairs: Sequence[CheckpointPair],
    history: Mapping[int, tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for pair in pairs:
        pred_a, pred_s = history[pair.epoch]
        a = segmentation_metrics(pred_a, targets)
        s = segmentation_metrics(pred_s, targets)
        gap = a["dice"] - s["dice"]
        leader = "Tie" if abs(gap) <= tolerance else ("A" if gap > 0 else "S")
        rows.append({
            "epoch": pair.epoch,
            "A_Dice": a["dice"], "S_Dice": s["dice"],
            "A_IoU": a["iou"], "S_IoU": s["iou"],
            "A_precision": a["precision"], "S_precision": s["precision"],
            "A_recall": a["recall"], "S_recall": s["recall"],
            "A_specificity": a["specificity"], "S_specificity": s["specificity"],
            "A_FPR": a["fpr"], "S_FPR": s["fpr"],
            "A_FNR": a["fnr"], "S_FNR": s["fnr"],
            "dice_gap": gap, "leader": leader,
        })
    non_ties = [(row["epoch"], row["leader"]) for row in rows if row["leader"] != "Tie"]
    crossover_epochs = [
        epoch for (previous_epoch, previous), (epoch, current) in zip(non_ties, non_ties[1:])
        if current != previous
    ]
    a_count = sum(row["leader"] == "A" for row in rows)
    s_count = sum(row["leader"] == "S" for row in rows)
    tie_count = len(rows) - a_count - s_count
    dominant = "Tie/None" if a_count == s_count else ("Appearance" if a_count > s_count else "Structure")
    summary = {
        "A_leader_epochs_count": a_count,
        "S_leader_epochs_count": s_count,
        "tie_epochs_count": tie_count,
        "leader_crossover_count": len(crossover_epochs),
        "crossover_epochs": crossover_epochs,
        "maximum_A_lead": max(0.0, max(row["dice_gap"] for row in rows)),
        "maximum_S_lead": max(0.0, max(-row["dice_gap"] for row in rows)),
        "mean_absolute_dice_gap": float(np.mean([abs(row["dice_gap"]) for row in rows])),
        "dominant_global_leader": dominant,
        "note": "Milestones are evaluated exactly as supplied; current project periodic files are best-so-far snapshots.",
    }
    return rows, summary


def analyze_expertise(
    pairs: Sequence[CheckpointPair],
    history: Mapping[int, tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
    regions: Mapping[str, np.ndarray],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for pair in pairs:
        pred_a, pred_s = history[pair.epoch]
        rows.append({"epoch": pair.epoch, **regional_expertise(pred_a, pred_s, targets, regions)})
    summary: dict[str, Any] = {"auxiliary_expertise_win_threshold": threshold, "regions": {}}
    for region in ("foreground", "background", "boundary", "interior"):
        a_wins = sum(int(row[region + "_A_only_correct"]) for row in rows)
        s_wins = sum(int(row[region + "_S_only_correct"]) for row in rows)
        pooled_a_rate = a_wins / (a_wins + s_wins + EPS)
        mean_a_rate = float(np.mean([row[region + "_A_win_rate"] for row in rows]))
        a_epoch_advantage = sum(row[region + "_A_win_rate"] > row[region + "_S_win_rate"] for row in rows)
        s_epoch_advantage = sum(row[region + "_S_win_rate"] > row[region + "_A_win_rate"] for row in rows)
        dominant = "Appearance" if pooled_a_rate > 0.5 else ("Structure" if pooled_a_rate < 0.5 else "Tie")
        summary["regions"][region] = {
            "dominant_expert": dominant,
            "pooled_A_wins": a_wins,
            "pooled_S_wins": s_wins,
            "pooled_A_win_rate": pooled_a_rate,
            "pooled_S_win_rate": 1.0 - pooled_a_rate if a_wins + s_wins else 0.0,
            "mean_epoch_A_win_rate": mean_a_rate,
            "mean_epoch_S_win_rate": float(np.mean([row[region + "_S_win_rate"] for row in rows])),
            "A_advantage_epoch_count": a_epoch_advantage,
            "S_advantage_epoch_count": s_epoch_advantage,
            "appearance_expertise_flag": pooled_a_rate >= threshold,
            "structure_expertise_flag": (1.0 - pooled_a_rate) >= threshold if a_wins + s_wins else False,
        }
    return rows, summary


def analyze_complementarity_time(
    pairs: Sequence[CheckpointPair],
    history: Mapping[int, tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        pred_a, pred_s = history[pair.epoch]
        counts = complementarity_counts(pred_a, pred_s, targets)
        a = segmentation_metrics(pred_a, targets)
        s = segmentation_metrics(pred_s, targets)
        oracle = segmentation_metrics(oracle_prediction((pred_a, pred_s), targets), targets)
        rows.append({
            "epoch": pair.epoch,
            "both_correct_ratio": counts["both_correct_ratio"],
            "A_only_ratio": counts["a_only_correct_ratio"],
            "S_only_ratio": counts["s_only_correct_ratio"],
            "both_wrong_ratio": counts["both_wrong_ratio"],
            "disagreement_ratio": counts["disagreement_ratio"],
            "A_Dice": a["dice"], "S_Dice": s["dice"],
            "Oracle_Dice": oracle["dice"],
            "Oracle_Gain": oracle["dice"] - max(a["dice"], s["dice"]),
        })
    return rows


def both_wrong_regions(
    recoverable: np.ndarray,
    persistent: np.ndarray,
    regions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output = {}
    for label, pixels in (("recoverable", recoverable), ("persistent", persistent)):
        total = int(pixels.sum())
        output[label] = {"count": total}
        for region in ("foreground", "background", "boundary", "interior"):
            count = int((pixels & regions[region]).sum())
            output[label][region + "_count"] = count
            output[label][region + "_ratio_of_group"] = count / (total + EPS)
    output["note"] = "Boundary/interior partition foreground; therefore the four region ratios are not one four-way partition."
    return output


def make_interpretation(
    recovery: Mapping[str, Any],
    leaders: Mapping[str, Any],
    expertise: Mapping[str, Any],
    complementarity_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    total_epochs = leaders["A_leader_epochs_count"] + leaders["S_leader_epochs_count"] + leaders["tie_epochs_count"]
    dominant = leaders["dominant_global_leader"]
    dominant_count = leaders["A_leader_epochs_count"] if dominant == "Appearance" else leaders["S_leader_epochs_count"]
    stable = dominant in ("Appearance", "Structure") and dominant_count / max(total_epochs, 1) >= args.stable_leader_fraction
    flags = {
        "global_leader_stable": stable,
        "stable_global_leader": dominant if stable else None,
        "HRR_union_above_auxiliary_threshold": recovery["HRR_union"] >= args.high_hrr_threshold,
        "leader_crossovers_present": leaders["leader_crossover_count"] > 0,
    }
    for region, values in expertise["regions"].items():
        flags["appearance_has_{}_expertise".format(region)] = values["appearance_expertise_flag"]
        flags["structure_has_{}_expertise".format(region)] = values["structure_expertise_flag"]
    statements = []
    if recovery["HRR_union"] >= args.high_hrr_threshold and leaders["leader_crossover_count"] > 0:
        statements.append("The observed historical recovery and leader crossovers support further validation of historical knowledge and progressive model-level transitions.")
    elif recovery["HRR_union"] >= args.high_hrr_threshold and stable:
        statements.append("Historical recovery is substantial while the global leader is stable; this suggests testing a stable main EMA teacher plus a dynamic regional expertise advisor rather than mandatory global switching.")
    elif recovery["HRR_union"] < args.high_hrr_threshold:
        statements.append("Historical recovery is limited under the configured auxiliary threshold, suggesting that historical teachers may offer limited theoretical help for current Both-Wrong pixels.")
    gains = [row["Oracle_Gain"] for row in complementarity_rows]
    if len(gains) >= 4:
        window = max(1, len(gains) // 4)
        if np.mean(gains[-window:]) < 0.5 * np.mean(gains[:window]):
            statements.append("The A/S oracle gain declines markedly across milestones, which indicates possible specialization weakening and warrants caution about teacher-induced homogenization.")
    if not statements:
        statements.append("The measurements are descriptive and suggest follow-up ablations; they do not establish a causal benefit for an EMA teacher.")
    return flags, statements


def _format_report(summary: Mapping[str, Any]) -> str:
    current = summary["current_model"]
    recovery = summary["historical_recovery"]
    oracle = summary["oracle_analysis"]
    leaders = summary["leader_dynamics"]
    expertise = summary["expertise_dynamics"]["regions"]
    lines = [
        "Progressive EMA Teacher Feasibility — Offline Validation",
        "=" * 62,
        "A. Current model",
        "Best Appearance Dice: {:.6f}".format(current["appearance"]["dice"]),
        "Best Structure Dice:  {:.6f}".format(current["structure"]["dice"]),
        "Current Both Wrong:    {} ({:.4%})".format(current["complementarity"]["both_wrong"], current["complementarity"]["both_wrong_ratio"]),
        "Current disagreement:  {} ({:.4%})".format(current["complementarity"]["disagreement"], current["complementarity"]["disagreement_ratio"]),
        "Current Oracle Dice:   {:.6f}".format(oracle["current_oracle"]["dice"]),
        "Current Oracle gain:   {:+.6f}".format(oracle["current_oracle_gain_over_best_single"]),
        "",
        "B. Historical recovery",
        "HRR_A:                 {:.4%}".format(recovery["HRR_A"]),
        "HRR_S:                 {:.4%}".format(recovery["HRR_S"]),
        "HRR_union:             {:.4%}".format(recovery["HRR_union"]),
        "Persistent rate:       {:.4%}".format(recovery["PBR"]),
        "Historical Oracle Dice:{:.6f}".format(oracle["historical_oracle"]["dice"]),
        "Historical gain/current Oracle: {:+.6f}".format(oracle["historical_oracle_gain_over_current_oracle"]),
        "",
        "C. Leader dynamics",
        "A leader epochs:       {}".format(leaders["A_leader_epochs_count"]),
        "S leader epochs:       {}".format(leaders["S_leader_epochs_count"]),
        "Leader crossovers:     {}".format(leaders["leader_crossover_count"]),
        "Crossover epochs:      {}".format(leaders["crossover_epochs"]),
        "Dominant leader:       {}".format(leaders["dominant_global_leader"]),
        "",
        "D. Expertise dynamics",
    ]
    for region in ("foreground", "background", "boundary", "interior"):
        lines.append("{} dominant expert: {} (pooled A win {:.2%}, S win {:.2%})".format(
            region.capitalize(), expertise[region]["dominant_expert"], expertise[region]["pooled_A_win_rate"], expertise[region]["pooled_S_win_rate"]
        ))
    lines.extend(["", "E. Research interpretation"] + ["- " + item for item in summary["research_interpretation"]])
    lines.extend([
        "",
        "Important: all oracle/winner/recovery decisions use validation GT and are analysis-only, not deployable.",
        "Bootstrap confidence intervals use images—not pixels—as independent resampling units.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.gpu)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "prediction_cache"
    figure_dir = output_dir / "figures"
    if not args.no_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    best_a_path, best_s_path, pairs, checkpoint_semantics = _resolve_inputs(args)
    cfg = dataset_cfg(args.dataset_name)
    if cfg["NUM_CLASSES"] != 2:
        raise ValueError("This analysis supports binary segmentation only")
    validation = build_validation_data(args, cfg, device)
    regions = region_masks(validation.targets, args.boundary_width)

    print("=" * 70)
    print("Progressive Teacher Offline Validation (read-only)")
    print("Device:", device)
    print("Dataset/split:", args.dataset_name, args.val_split)
    print("Manifest:", validation.manifest)
    print("Dataset size:", len(validation.names))
    print("Best Appearance:", best_a_path)
    print("Best Structure:", best_s_path)
    print("Historical pairs:", len(pairs))
    print("Checkpoint semantics:", checkpoint_semantics)
    print("Output:", output_dir)
    print("=" * 70)

    jobs = [("best_A", "appearance", best_a_path), ("best_S", "structure", best_s_path)]
    for pair in pairs:
        jobs.extend((("epoch_{}_A".format(pair.epoch), "appearance", pair.appearance), ("epoch_{}_S".format(pair.epoch), "structure", pair.structure)))
    predictions: dict[str, np.ndarray] = {}
    inference_metadata: dict[str, Any] = {}
    for label, branch, path in tqdm(jobs, desc="Checkpoint loading/inference", unit="checkpoint"):
        prediction, metadata = infer_checkpoint(path, branch, args, cfg, validation, device, cache_dir)
        predictions[label] = prediction
        inference_metadata[label] = metadata

    best_predictions = (predictions["best_A"], predictions["best_S"])
    history = {
        pair.epoch: (predictions["epoch_{}_A".format(pair.epoch)], predictions["epoch_{}_S".format(pair.epoch)])
        for pair in pairs
    }
    inventory = {
        "best_appearance": str(best_a_path),
        "best_structure": str(best_s_path),
        "history_semantics": checkpoint_semantics,
        "pairs": [{"epoch": pair.epoch, "appearance": str(pair.appearance), "structure": str(pair.structure)} for pair in pairs],
        "inference": inference_metadata,
    }
    write_json(output_dir / "checkpoint_inventory.json", inventory)

    current, both_wrong, current_oracle_pred, current_oracle_per_image = analyze_current(
        *best_predictions, validation.targets, inference_metadata["best_A"], inference_metadata["best_S"]
    )
    write_json(output_dir / "current_best_summary.json", current)

    recovery, recovery_per_image, recovery_arrays = analyze_recovery(
        pairs, history, validation.targets, both_wrong, validation.names, output_dir
    )
    write_json(output_dir / "historical_recovery_summary.json", recovery)
    write_csv(output_dir / "historical_recovery_per_image.csv", recovery_per_image)

    cumulative_rows = cumulative_recovery(pairs, history, best_predictions, validation.targets, both_wrong)
    write_csv(output_dir / "cumulative_historical_recovery.csv", cumulative_rows)

    all_predictions = [best_predictions[0], best_predictions[1]]
    for pair in pairs:
        all_predictions.extend(history[pair.epoch])
    historical_oracle_pred, historical_oracle_metrics, historical_oracle_per_image = _oracle_metrics(all_predictions, validation.targets)
    metrics_a = current["appearance"]
    metrics_s = current["structure"]
    best_single_dice = max(metrics_a["dice"], metrics_s["dice"])
    oracle_analysis = {
        "best_appearance": {key: metrics_a[key] for key in ("dice", "iou")},
        "best_structure": {key: metrics_s[key] for key in ("dice", "iou")},
        "current_oracle": {key: current["current_oracle"][key] for key in ("dice", "iou")},
        "historical_oracle": historical_oracle_metrics,
        "current_oracle_gain_over_best_single": current["current_oracle"]["dice"] - best_single_dice,
        "historical_oracle_gain_over_best_single": historical_oracle_metrics["dice"] - best_single_dice,
        "historical_oracle_gain_over_current_oracle": historical_oracle_metrics["dice"] - current["current_oracle"]["dice"],
        "warning": "All oracle values use validation GT and are analysis-only theoretical upper bounds; they are not deployable.",
    }
    write_json(output_dir / "oracle_analysis.json", oracle_analysis)

    leader_rows, leader_summary = analyze_leaders(pairs, history, validation.targets, args.leader_tie_tolerance)
    write_csv(output_dir / "leader_dynamics.csv", leader_rows)
    write_json(output_dir / "leader_dynamics_summary.json", leader_summary)

    expertise_rows, expertise_summary = analyze_expertise(
        pairs, history, validation.targets, regions, args.expertise_win_threshold
    )
    write_csv(output_dir / "expertise_dynamics.csv", expertise_rows)
    write_json(output_dir / "expertise_dynamics_summary.json", expertise_summary)

    complementarity_rows = analyze_complementarity_time(pairs, history, validation.targets)
    write_csv(output_dir / "complementarity_over_time.csv", complementarity_rows)

    region_analysis = both_wrong_regions(recovery_arrays["recoverable"], recovery_arrays["persistent"], regions)
    write_json(output_dir / "both_wrong_region_analysis.json", region_analysis)

    per_a = _per_image_metrics(best_predictions[0], validation.targets)
    per_s = _per_image_metrics(best_predictions[1], validation.targets)
    bootstrap_values = {"HRR_union": [], "historical_oracle_gain": [], "current_oracle_gain": []}
    for index in range(len(validation.names)):
        bootstrap_values["HRR_union"].append(recovery_per_image[index]["HRR_union"])
        best_image_dice = max(per_a[index]["dice"], per_s[index]["dice"])
        bootstrap_values["current_oracle_gain"].append(current_oracle_per_image[index]["dice"] - best_image_dice)
        bootstrap_values["historical_oracle_gain"].append(historical_oracle_per_image[index]["dice"] - best_image_dice)
    bootstrap = bootstrap_image_means(bootstrap_values, args.bootstrap_iterations, args.bootstrap_seed)
    bootstrap["method"] = "Image-level nonparametric bootstrap of image-wise metrics; pixels are not resampling units."
    write_json(output_dir / "bootstrap_ci.json", bootstrap)

    flags, interpretations = make_interpretation(recovery, leader_summary, expertise_summary, complementarity_rows, args)
    final_summary = {
        "analysis_type": "read-only offline validation; no training or checkpoint updates",
        "dataset": {"name": args.dataset_name, "split": args.val_split, "manifest": str(validation.manifest), "images": len(validation.names)},
        "checkpoint_semantics": checkpoint_semantics,
        "threshold_protocol": {
            "mode": args.threshold_mode,
            "fixed_threshold": args.threshold if args.threshold_mode == "fixed" else None,
            "auto_protocol": "Per-checkpoint foreground threshold selected on this validation split using the project's 0.00:0.02:0.88 Jaccard grid.",
        },
        "current_model": current,
        "historical_recovery": recovery,
        "oracle_analysis": oracle_analysis,
        "leader_dynamics": leader_summary,
        "expertise_dynamics": expertise_summary,
        "automatic_flags": flags,
        "auxiliary_thresholds": {
            "stable_leader_fraction": args.stable_leader_fraction,
            "expertise_win_threshold": args.expertise_win_threshold,
            "high_hrr_threshold": args.high_hrr_threshold,
        },
        "bootstrap_ci": bootstrap,
        "research_interpretation": interpretations,
        "caution": "GT is used only for offline evaluation. Results suggest/support follow-up experiments and do not prove causality.",
    }
    write_json(output_dir / "final_summary.json", final_summary)
    (output_dir / "final_summary.txt").write_text(_format_report(final_summary), encoding="utf-8")

    if not args.no_figures:
        plot_cumulative(cumulative_rows, figure_dir / "cumulative_historical_recovery.png")
        plot_leader(leader_rows, figure_dir / "leader_dynamics_dice.png", figure_dir / "leader_gap.png")
        plot_complementarity(complementarity_rows, figure_dir / "complementarity_over_time.png")
    if args.save_qualitative:
        scores = np.asarray([row["HRR_union"] for row in recovery_per_image])
        selected = np.argsort(-scores)[: min(args.qualitative_count, len(scores))].tolist()
        save_qualitative(
            output_dir / "qualitative", selected, validation.names, validation.image_paths,
            validation.targets, best_predictions[0], best_predictions[1], both_wrong,
            recovery_arrays["recoverable"], recovery_arrays["persistent"], history,
        )

    print(_format_report(final_summary))
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
