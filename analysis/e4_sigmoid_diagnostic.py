#!/usr/bin/env python3
"""Offline E4 raw-D -> sigmoid(-D) weight-range diagnostic.

This script only reads the existing E4 token record CSV. It does not import
the model code and performs no training or forward pass.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv"
OUTPUT = ROOT / "outputs/sigmoid_d_weight_diagnostic"
EPS = 1e-12


def sigmoid_neg(d: np.ndarray) -> np.ndarray:
    # D is positive here; this form is numerically safe for the observed range.
    return 1.0 / (1.0 + np.exp(d))


def quantile_dict(values: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    return {f"p{p:02d}": float(np.percentile(values, p)) for p in percentiles}


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not INPUT.exists():
        candidates = sorted(ROOT.glob("outputs/**/token_records.csv"))
        raise FileNotFoundError(f"Expected {INPUT}; candidates={candidates}")

    d_values: list[float] = []
    with INPUT.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"agreement", "contradiction", "D"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in reader:
            agreement = str(row["agreement"]).strip().lower() == "true"
            contradiction = str(row["contradiction"]).strip().lower() == "true"
            d = float(row["D"])
            if agreement and contradiction and d > 0.0:
                d_values.append(d)

    d = np.asarray(d_values, dtype=np.float64)
    if d.size == 0:
        raise RuntimeError("No rows satisfy agreement=true, contradiction=true, D>0")
    if not np.isfinite(d).all():
        raise ValueError("Filtered D contains non-finite values")

    w = sigmoid_neg(d)
    if not np.isfinite(w).all() or not ((w > 0.0).all() and (w < 0.5).all()):
        raise ValueError("sigmoid(-D) failed expected 0<w<0.5 range")

    d_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    d_q = quantile_dict(d, d_percentiles)
    weight_q = quantile_dict(w, d_percentiles)
    key_percentiles = [10, 25, 50, 75, 90]

    quantile_rows: list[dict[str, object]] = []
    for p in d_percentiles:
        d_value = d_q[f"p{p:02d}"]
        weight = float(sigmoid_neg(np.asarray([d_value]))[0])
        quantile_rows.append(
            {
                "percentile": f"p{p:02d}",
                "D": d_value,
                "sigmoid_weight": weight,
                "difference_from_fixed_0.5": weight - 0.5,
                "relative_change_vs_fixed": (weight - 0.5) / 0.5,
                "suppression_1_minus_weight": 1.0 - weight,
                "fixed_suppression": 0.5,
            }
        )

    weights_at_d_quantiles = {
        f"p{p:02d}": float(sigmoid_neg(np.asarray([d_q[f"p{p:02d}"]]))[0])
        for p in key_percentiles
    }
    delta_10_90 = weights_at_d_quantiles["p10"] - weights_at_d_quantiles["p90"]
    delta_25_75 = weights_at_d_quantiles["p25"] - weights_at_d_quantiles["p75"]
    ratio_10_90 = weights_at_d_quantiles["p10"] / (weights_at_d_quantiles["p90"] + EPS)

    q_edges = np.percentile(d, [0, 25, 50, 75, 100])
    quartile_rows: list[dict[str, object]] = []
    for i in range(4):
        if i < 3:
            mask = (d >= q_edges[i]) & (d < q_edges[i + 1])
        else:
            mask = (d >= q_edges[i]) & (d <= q_edges[i + 1])
        d_group, w_group = d[mask], w[mask]
        quartile_rows.append(
            {
                "quartile": f"Q{i + 1}",
                "num_tokens": int(d_group.size),
                "mean_D": mean_or_nan(d_group),
                "median_D": median_or_nan(d_group),
                "mean_sigmoid_weight": mean_or_nan(w_group),
                "median_sigmoid_weight": median_or_nan(w_group),
            }
        )
    if not all(
        quartile_rows[i]["mean_sigmoid_weight"] > quartile_rows[i + 1]["mean_sigmoid_weight"]
        for i in range(3)
    ):
        raise AssertionError("D quartile mean weights are not strictly descending")

    d_scale = {
        "P_D_lt_0.01": float(np.mean(d < 0.01)),
        "P_D_lt_0.02": float(np.mean(d < 0.02)),
        "P_D_lt_0.05": float(np.mean(d < 0.05)),
        "P_D_lt_0.10": float(np.mean(d < 0.10)),
        "P_D_gt_0.10": float(np.mean(d > 0.10)),
        "P_D_gt_0.20": float(np.mean(d > 0.20)),
        "P_D_gt_0.50": float(np.mean(d > 0.50)),
        "P_D_gt_1.00": float(np.mean(d > 1.00)),
    }

    # Diagnostic-only classification, with the requested text tied directly to
    # the observed dynamic range and concentration near the Fixed value.
    near_fixed = float(np.mean(np.abs(w - 0.5) < 0.025))
    if delta_10_90 < 0.05:
        conclusion = "Outcome B — Raw D is too small in scale for sigmoid(-D) to provide meaningful dynamic differentiation. The mapping is mathematically clean but practically close to Fixed Native-S."
    elif delta_10_90 < 0.15 and near_fixed >= 0.75:
        conclusion = "Outcome C — Raw-D sigmoid weighting is dominated by a small tail and is not a well-balanced adaptive mapping."
    else:
        conclusion = "Outcome A — Raw-D sigmoid weighting provides a meaningful parameter-free dynamic suppression range."

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT / "d_quantile_to_weight.csv",
        quantile_rows,
        [
            "percentile",
            "D",
            "sigmoid_weight",
            "difference_from_fixed_0.5",
            "relative_change_vs_fixed",
            "suppression_1_minus_weight",
            "fixed_suppression",
        ],
    )
    weight_rows = [
        {"metric": "num_contradiction_tokens", "value": int(d.size)},
        *[{"metric": f"D_{key}", "value": value} for key, value in d_q.items()],
        {"metric": "D_min", "value": float(d.min())},
        {"metric": "D_max", "value": float(d.max())},
        {"metric": "D_mean", "value": float(d.mean())},
        {"metric": "D_std", "value": float(d.std())},
        *[{"metric": f"weight_{key}", "value": value} for key, value in weight_q.items()],
        {"metric": "weight_min", "value": float(w.min())},
        {"metric": "weight_max", "value": float(w.max())},
        {"metric": "weight_mean", "value": float(w.mean())},
        {"metric": "weight_std", "value": float(w.std())},
        {"metric": "dynamic_range_p10_minus_p90", "value": delta_10_90},
        {"metric": "dynamic_range_p25_minus_p75", "value": delta_25_75},
        {"metric": "ratio_w_D_p10_over_w_D_p90", "value": ratio_10_90},
        *[{"metric": key, "value": value} for key, value in d_scale.items()],
        {"metric": "fraction_abs_weight_minus_0.5_lt_0.025", "value": near_fixed},
    ]
    write_csv(OUTPUT / "weight_distribution.csv", weight_rows, ["metric", "value"])

    summary = {
        "source": str(INPUT.relative_to(ROOT)),
        "analysis": "offline; existing E4 token_records.csv only; no model, training, or forward",
        "filter": "agreement == true and contradiction == true and D > 0",
        "num_contradiction_tokens": int(d.size),
        "D": {
            "min": float(d.min()),
            "max": float(d.max()),
            "mean": float(d.mean()),
            "std": float(d.std()),
            **{key: value for key, value in d_q.items()},
        },
        "weight_formula": "sigmoid(-D)=1/(1+exp(D))",
        "weight_at_D_quantiles": weights_at_d_quantiles,
        "weight_distribution": {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
            **weight_q,
        },
        "dynamic_range": {
            "p10_minus_p90": delta_10_90,
            "p25_minus_p75": delta_25_75,
            "ratio_10_90": ratio_10_90,
            "description": "very narrow" if delta_10_90 < 0.05 else ("moderate" if delta_10_90 < 0.15 else "substantial"),
        },
        "D_scale": d_scale,
        "quartiles": quartile_rows,
        "near_fixed_fraction_abs_weight_minus_0.5_lt_0.025": near_fixed,
        "conclusion": conclusion,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# E4 Raw-D Sigmoid Weight Diagnostic\n\n"
        "Offline-only diagnostic over the existing E4 `token_records.csv`. Rows were filtered to `agreement=true`, `contradiction=true`, and `D>0`; no model, training, or forward pass was used. The tested mapping is `w=sigmoid(-D)` with no temperature, scale, normalization, buffer, or floor.\n\n"
        f"Conclusion: {conclusion}\n"
    )

    try:
        import matplotlib.pyplot as plt

        grid = np.linspace(float(d.min()), float(d.max()), 500)
        plt.figure(figsize=(7, 4), dpi=150)
        plt.plot(grid, sigmoid_neg(grid), label="sigmoid(-D)")
        for p in key_percentiles:
            value = d_q[f"p{p:02d}"]
            weight = weights_at_d_quantiles[f"p{p:02d}"]
            plt.scatter([value], [weight], s=24)
            plt.annotate(f"p{p}", (value, weight), xytext=(4, 4), textcoords="offset points")
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Fixed 0.5")
        plt.xlabel("D")
        plt.ylabel("weight")
        plt.title("E4 raw-D sigmoid weight")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / "D_to_sigmoid_weight.png")
        plt.close()
    except Exception as exc:  # plotting is optional and never affects statistics
        (OUTPUT / "plot_status.txt").write_text(f"Plot skipped: {exc}\n")

    print("=== E4 Raw-D Sigmoid Weight Diagnostic ===")
    print(f"\nContradiction tokens: {d.size}")
    print("\nD:")
    for p in key_percentiles:
        print(f"  p{p}: {d_q[f'p{p:02d}']:.8f}")
    print("\nw = sigmoid(-D):")
    for p in key_percentiles:
        print(f"  D p{p} -> weight: {weights_at_d_quantiles[f'p{p:02d}']:.8f}")
    print("\nDynamic range:")
    print(f"  w(p10) - w(p90): {delta_10_90:.8f}")
    print(f"  w(p25) - w(p75): {delta_25_75:.8f}")
    print("\nCompared with Fixed 0.5:")
    for p in key_percentiles:
        row = quantile_rows[d_percentiles.index(p)]
        print(f"  p{p} delta: {row['difference_from_fixed_0.5']:.8f}")
    print("\nD scale:")
    for key in ["P_D_lt_0.05", "P_D_lt_0.10", "P_D_gt_0.20", "P_D_gt_0.50"]:
        print(f"  {key}: {d_scale[key]:.8f}")
    print("\nQuartiles:")
    for row in quartile_rows:
        print(f"  {row['quartile']} mean D / weight: {row['mean_D']:.8f} / {row['mean_sigmoid_weight']:.8f}")
    print(f"\nOutcome: {conclusion}")
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
