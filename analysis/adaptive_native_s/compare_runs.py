#!/usr/bin/env python3
"""Compare Fixed Native-S with Direct Semantic-Gap Adaptive Native-S."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_root(pattern: str) -> Path:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run root for {pattern}, got {matches}")
    return Path(matches[0])


def train_curve(path: Path) -> pd.DataFrame:
    pattern = re.compile(r"epoch=(\d+)/\d+ .*?val_dice_student1=([0-9.eE+-]+) .*?val_dice_student2=([0-9.eE+-]+) .*?val_jaccard_student1=([0-9.eE+-]+) .*?val_jaccard_student2=([0-9.eE+-]+)")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            e, d1, d2, j1, j2 = match.groups()
            rows.append({"epoch": int(e), "Dice": (.5 * (float(d1) + float(d2))), "IoU": (.5 * (float(j1) + float(j2)))})
    result = pd.DataFrame(rows).drop_duplicates("epoch", keep="last").sort_values("epoch")
    if len(result) != 200:
        raise RuntimeError(f"Expected 200 validation rows in {path}, got {len(result)}")
    return result


def load_method(root: Path, tag: str, label: str, csv_root: Path):
    curve = train_curve(root / "train.log")
    shared = pd.read_csv(csv_root / f"validation_shared_errors_{tag}.csv")
    events = pd.read_csv(csv_root / f"validation_events_{tag}.csv")
    train = pd.read_csv(csv_root / f"training_{tag}.csv")
    merged = curve.merge(shared, on="epoch").merge(events, on="epoch")
    merged.insert(0, "method", label)
    train.insert(0, "method", label)
    return merged, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/native_gap_adaptive")
    parser.add_argument("--native-csv-root", default="outputs/native_scale_training")
    args = parser.parse_args()
    out = Path(args.output_dir); plots = out / "plots"; plots.mkdir(parents=True, exist_ok=True)
    csv_root = Path(args.native_csv_root)
    fixed_root = find_root("checkpoints/native_s_seed24/GLaS/*reference_mode=native_suppress")
    gap_root = find_root("checkpoints/native_gap_adaptive_s_seed24/GLaS/*reference_mode=native_suppress_adaptive")
    fixed_val, fixed_train = load_method(fixed_root, "native_s", "Fixed Native-S", csv_root)
    gap_val, gap_train = load_method(gap_root, "native_gap_adaptive_s", "Direct Gap Adaptive Native-S", csv_root)
    validation = pd.concat([fixed_val, gap_val], ignore_index=True)
    validation.to_csv(out / "validation_metrics.csv", index=False)
    gap_train.to_csv(out / "training_log.csv", index=False)
    gap_train.to_csv(out / "weight_statistics.csv", index=False)
    keep = ["method", "epoch", "Dice", "IoU", "WW_rate", "HC90WW_rate", "CC_rate", "Shared_FP_rate", "Shared_FN_rate", "WW_events", "HC90WW_events"]
    comparison = validation[keep].copy()
    deltas = []
    for epoch in sorted(validation.epoch.unique()):
        fixed = validation[(validation.method == "Fixed Native-S") & (validation.epoch == epoch)].iloc[0]
        gap = validation[(validation.method == "Direct Gap Adaptive Native-S") & (validation.epoch == epoch)].iloc[0]
        row = {"method": "Gap Adaptive - Fixed", "epoch": epoch}
        for col in keep[2:]: row[col] = float(gap[col] - fixed[col])
        deltas.append(row)
    pd.concat([comparison, pd.DataFrame(deltas)], ignore_index=True).to_csv(out / "fixed_vs_gap_adaptive.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    for method, group in validation.groupby("method"):
        for ax, col in zip(axes.flat, ["Dice", "IoU", "WW_rate", "HC90WW_rate", "CC_rate", "WW_events"]):
            ax.plot(group.epoch, group[col], label=method)
            ax.set_title(col); ax.set_xlabel("epoch"); ax.grid(alpha=.25)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(plots / "fixed_vs_gap_adaptive_curves.png", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for col, ax in [("mean_relative_gap", axes[0]), ("mean_adaptive_weight", axes[1])]:
        if col in gap_train: ax.plot(gap_train.epoch, gap_train[col]); ax.set_title(col); ax.set_xlabel("epoch"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots / "gap_weight_curves.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for col in ["gap_Q1_mean_weight", "gap_Q2_mean_weight", "gap_Q3_mean_weight", "gap_Q4_mean_weight"]:
        if col in gap_train: ax.plot(gap_train.epoch, gap_train[col], label=col)
    ax.set_title("Gap quartile weights"); ax.set_xlabel("epoch"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "gap_quartile_weights.png", bbox_inches="tight"); plt.close(fig)

    final = validation[validation.epoch == 200].set_index("method")
    fixed_complete = json.loads((fixed_root / "native_training_complete.json").read_text())
    gap_complete = json.loads((gap_root / "native_training_complete.json").read_text())
    gap_last = gap_train.iloc[-1]
    fixed = final.loc["Fixed Native-S"]; gap = final.loc["Direct Gap Adaptive Native-S"]
    delta = {key: float(gap[key] - fixed[key]) for key in keep[2:]}
    gap_q = {key: float(gap_last[key]) for key in ["gap_Q1_mean_gap", "gap_Q1_mean_weight", "gap_Q2_mean_gap", "gap_Q2_mean_weight", "gap_Q3_mean_gap", "gap_Q3_mean_weight", "gap_Q4_mean_gap", "gap_Q4_mean_weight"] if key in gap_last and pd.notna(gap_last[key])}
    outcome = "Outcome A — Direct semantic-gap weighting improves Fixed Native-S." if delta["Dice"] > 0 and delta["IoU"] > 0 and delta["WW_rate"] <= 0 and delta["HC90WW_rate"] <= 0 and delta["CC_rate"] >= 0 else ("Outcome B — Direct semantic-gap weighting is valid but offers little practical advantage over the simpler fixed suppression rule." if abs(delta["Dice"]) < .005 and abs(delta["IoU"]) < .005 else "Outcome C — Although UNI semantic gap correlates with shared-error severity offline, direct gap-to-weight mapping does not outperform the fixed suppression rule during training.")
    summary = {"implementation": {"adaptive_mode": "native_suppress_adaptive", "adaptive_strategy": "direct_relative_semantic_gap", "cdf_removed": True, "running_buffer_removed": True, "uni_input": "256x256", "native_grid": "16x16", "weight_formula": "w=1-max(S_opp-S_cur,0)/(abs(S_opp)+abs(S_cur)+1e-6) on agreement, else 1", "weight_range": [0.0, 1.0], "gt_used_for_weighting": False, "validation_used_for_weighting": False}, "training": {"epochs": 200, "seed": 24}, "gap_final_epoch": {"mean_relative_gap": float(gap_last.mean_relative_gap), "mean_weight": float(gap_last.mean_adaptive_weight), **gap_q}, "fixed_best": fixed_complete.get("best_synchronized_mean_val_jaccard"), "gap_best": gap_complete.get("best_synchronized_mean_val_jaccard"), "epoch200": {"Fixed Native-S": {key: float(fixed[key]) for key in keep[2:]}, "Direct Gap Adaptive Native-S": {key: float(gap[key]) for key in keep[2:]}}, "delta_gap_minus_fixed_epoch200": delta, "outcome": outcome}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text("# Direct Semantic-Gap Adaptive Native-S\n\nFormal seed-24, 200-epoch comparison against the completed Fixed Native-S run. Adaptive weighting uses only the current detached UNI token evidence; CDF, percentile rank, historical/running buffers, Student confidence, and GT/validation weighting are disabled.\n")
    print("=== Direct Semantic-Gap Adaptive Native-S ===")
    print("Fixed Native-S:", {key: float(fixed[key]) for key in keep[2:]})
    print("Gap-Adaptive Native-S:", {key: float(gap[key]) for key in keep[2:]})
    print("Gap-Adaptive - Fixed:", delta)
    print("Outcome:", outcome)
    print("Outputs:", out)


if __name__ == "__main__":
    main()
