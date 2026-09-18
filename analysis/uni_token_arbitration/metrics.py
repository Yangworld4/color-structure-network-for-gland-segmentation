"""Native-token record aggregation. No score map interpolation is permitted."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


SUBSETS = ("ALL_PURE", "PURE_FG", "PURE_BG", "CC", "WW", "HC90WW", "SHARED_FP", "SHARED_FN", "AC", "BC", "MIXED")


def finite_mean(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def subset_mask(records, subset):
    if subset == "ALL_PURE": return np.array([r["is_pure"] for r in records], bool)
    if subset == "PURE_FG": return np.array([r["is_pure"] and r["gt_token"] == "FG" for r in records], bool)
    if subset == "PURE_BG": return np.array([r["is_pure"] and r["gt_token"] == "BG" for r in records], bool)
    if subset == "MIXED": return np.array([r["is_mixed"] for r in records], bool)
    return np.array([r[f"is_{subset}"] for r in records], bool)


def calculate(records, subset):
    chosen = [r for r, keep in zip(records, subset_mask(records, subset)) if keep]
    n = len(chosen); image_ids = sorted({r["image_id"] for r in chosen})
    non_tie = [r for r in chosen if r["UNI_pred"] != "TIE"]
    accuracy = finite_mean([r["UNI_correct"] for r in non_tie])
    fg_acc = finite_mean([r["UNI_correct"] for r in non_tie if r["gt_token"] == "FG"])
    bg_acc = finite_mean([r["UNI_correct"] for r in non_tie if r["gt_token"] == "BG"])
    return {
        "subset": subset, "num_images": len(image_ids), "num_tokens": n,
        "num_ties": n - len(non_tie), "tie_rate": (n-len(non_tie))/n if n else float("nan"),
        "uni_accuracy": accuracy, "contradiction_rate": 1.0 - accuracy if np.isfinite(accuracy) else float("nan"), "balanced_accuracy": finite_mean([fg_acc, bg_acc]),
        "FG_recall": fg_acc, "BG_recall": bg_acc,
        "mean_S_FG": finite_mean([r["S_FG"] for r in chosen]), "mean_S_BG": finite_mean([r["S_BG"] for r in chosen]),
        "mean_margin": finite_mean([r["margin"] for r in chosen]), "mean_S_cur": finite_mean([r["S_cur"] for r in chosen]),
        "mean_S_opp": finite_mean([r["S_opp"] for r in chosen]), "mean_D": finite_mean([r["D"] for r in chosen]),
        "median_D": float(np.nanmedian([r["D"] for r in chosen])) if any(np.isfinite(r["D"]) for r in chosen) else float("nan"),
        "D_positive_rate": finite_mean([float(r["D"] > 0) for r in chosen if np.isfinite(r["D"])]),
    }


def image_rows(records):
    rows = []
    for epoch in sorted({r["epoch"] for r in records}):
        for image_id in sorted({r["image_id"] for r in records if r["epoch"] == epoch}):
            sample = [r for r in records if r["epoch"] == epoch and r["image_id"] == image_id]
            row = {"epoch": epoch, "image_id": image_id}
            for subset in ("ALL_PURE", "CC", "WW", "HC90WW"):
                metric = calculate(sample, subset)
                key = subset.lower()
                row[f"num_{key}"] = metric["num_tokens"]
                row[f"uni_accuracy_{key}"] = metric["uni_accuracy"]
                row[f"mean_D_{key}"] = metric["mean_D"]
            row["num_mixed"] = calculate(sample, "MIXED")["num_tokens"]
            # Explicit aliases follow the published diagnostic contract while
            # retaining the machine-friendly lower-case columns above.
            row["num_pure_tokens"] = row["num_all_pure"]
            row["num_CC"] = row["num_cc"]; row["num_WW"] = row["num_ww"]; row["num_HC90WW"] = row["num_hc90ww"]
            row["uni_accuracy_pure"] = row["uni_accuracy_all_pure"]
            row["uni_accuracy_CC"] = row["uni_accuracy_cc"]; row["uni_accuracy_WW"] = row["uni_accuracy_ww"]; row["uni_accuracy_HC90WW"] = row["uni_accuracy_hc90ww"]
            rows.append(row)
    return rows


def epoch_rows(records):
    rows = []
    for epoch in sorted({r["epoch"] for r in records}):
        sample = [r for r in records if r["epoch"] == epoch]
        row = {"epoch": epoch}
        for subset in ("CC", "WW", "HC90WW", "SHARED_FP", "SHARED_FN"):
            metric = calculate(sample, subset); key = subset.lower()
            row[f"num_{key}"] = metric["num_tokens"]; row[f"arbitration_accuracy_{key}"] = metric["uni_accuracy"]
            row[f"mean_D_{key}"] = metric["mean_D"]; row[f"D_positive_rate_{key}"] = metric["D_positive_rate"]
        rows.append(row)
    return rows


def mixed_stats(records):
    mixed = [r for r in records if r["is_mixed"]]
    occupancy = np.asarray([r["gt_occupancy"] for r in mixed], float); margin = np.asarray([r["margin"] for r in mixed], float)
    if len(mixed) >= 2 and np.unique(occupancy).size > 1:
        pearson = float(pearsonr(margin, occupancy)[0]); spearman = float(spearmanr(margin, occupancy)[0])
    else: pearson = spearman = float("nan")
    bins = []
    for lo, hi in ((.1,.3),(.3,.5),(.5,.7),(.7,.9)):
        group = [r for r in mixed if (r["gt_occupancy"] >= lo if lo == .1 else r["gt_occupancy"] > lo) and (r["gt_occupancy"] <= hi if hi == .9 else r["gt_occupancy"] < hi)]
        bins.append({"subset": f"MIXED_OCC_{lo:.1f}_{hi:.1f}", "num_images": len({r['image_id'] for r in group}), "num_tokens": len(group), "mean_S_FG": finite_mean([r["S_FG"] for r in group]), "mean_S_BG": finite_mean([r["S_BG"] for r in group]), "mean_margin": finite_mean([r["margin"] for r in group])})
    return pearson, spearman, bins


def bootstrap_image_ci(image_metrics, metric_key, subset_key, seed=2026, samples=1000):
    values = np.array([r[metric_key] for r in image_metrics if r[f"num_{subset_key}"] > 0 and np.isfinite(r[metric_key])], float)
    if not len(values): return [float("nan")] * 3
    rng = np.random.default_rng(seed); boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return [float(values.mean()), float(np.quantile(boot,.025)), float(np.quantile(boot,.975))]
