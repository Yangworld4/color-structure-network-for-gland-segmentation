#!/usr/bin/env python3
"""Closed, offline diagnostic of D as a continuous suppression-risk signal.

The script consumes the already generated E4 native-scale token records.  It
does not load models, images, GT, or the reference memory and performs no
training.  All risk labels are descriptive outcomes already present in E4
(token-level WW/HC90WW/CC ratios).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EPS = 1e-8
DEFAULT_SEED = 2026


def _mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def _median(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else None


def _rho(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan
    result = spearmanr(x[mask], y[mask])
    if hasattr(result, "statistic"):
        value = result.statistic
    elif hasattr(result, "correlation"):
        value = result.correlation
    else:
        value = result[0]
    return float(value)


def _safe_ratio(a, b):
    return float(a / (b + EPS))


def _risk_row(g: pd.DataFrame, group: str, include_d_range=True):
    row = {
        "group": group,
        "num_tokens": int(len(g)),
        "num_images": int(g["image_id"].nunique()) if len(g) else 0,
        "D_min": float(g["D"].min()) if len(g) else None,
        "D_max": float(g["D"].max()) if len(g) else None,
        "D_mean": _mean(g["D"]),
        "D_median": _median(g["D"]),
        "WW_ratio_mean": _mean(g["WW_ratio"]),
        "WW_ratio_median": _median(g["WW_ratio"]),
        "HC90WW_ratio_mean": _mean(g["HC90WW_ratio"]),
        "CC_ratio_mean": _mean(g["CC_ratio"]),
        "shared_FP_ratio_mean": _mean(g["shared_FP_ratio"]),
        "shared_FN_ratio_mean": _mean(g["shared_FN_ratio"]),
        "direction_accuracy": _mean(g["direction_correct"]),
        "P_WW_gt_025": _mean((g["WW_ratio"] > 0.25).astype(float)),
        "P_WW_gt_050": _mean((g["WW_ratio"] > 0.50).astype(float)),
        "P_HC90WW_gt_025": _mean((g["HC90WW_ratio"] > 0.25).astype(float)),
        "P_HC90WW_gt_050": _mean((g["HC90WW_ratio"] > 0.50).astype(float)),
        "P_CC_gt_050": _mean((g["CC_ratio"] > 0.50).astype(float)),
    }
    return row


def _write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, float_format="%.8g")


def _add_quantile_labels(df: pd.DataFrame, column: str, out_column: str, n: int):
    # rank(method=first) makes ties deterministic and guarantees balanced bins.
    ranks = df[column].rank(method="first")
    df[out_column] = pd.qcut(ranks, n, labels=[f"Q{i}" for i in range(1, n + 1)])


def _group_metrics(df: pd.DataFrame, label_col: str, labels):
    rows = []
    for label in labels:
        rows.append(_risk_row(df[df[label_col] == label], str(label)))
    return pd.DataFrame(rows)


def _stratified_correlations(df: pd.DataFrame, subset_name: str):
    rows = []
    for name, g in [("ALL_CONTRADICTION", df),
                    ("PURE", df[df["gt_pure_or_mixed"].astype(str).str.upper() == "PURE"]),
                    ("MIXED", df[df["gt_pure_or_mixed"].astype(str).str.upper() == "MIXED"]),
                    ("FG_CONSENSUS", df[df["hard_A"] == 1]),
                    ("BG_CONSENSUS", df[df["hard_A"] == 0])]:
        rows.append({
            "subset": subset_name,
            "stratum": name,
            "num_tokens": int(len(g)),
            "num_images": int(g["image_id"].nunique()) if len(g) else 0,
            "rho_D_WW": _rho(g["D"], g["WW_ratio"]),
            "rho_D_HC90WW": _rho(g["D"], g["HC90WW_ratio"]),
            "rho_D_CC": _rho(g["D"], g["CC_ratio"]),
        })
    return rows


def _ranking_metrics(agreement: pd.DataFrame, seed: int = DEFAULT_SEED):
    fractions = [0.05, 0.10, 0.20, 0.30]
    rng = np.random.default_rng(seed)
    rows = []
    total_ww = float(agreement["WW_ratio"].sum())
    total_hc = float(agreement["HC90WW_ratio"].sum())
    total_cc = float(agreement["CC_ratio"].sum())
    for frac in fractions:
        n = max(1, int(np.ceil(len(agreement) * frac)))
        candidates = {
            "D": agreement.nlargest(n, "r_d"),
            "confidence": agreement.nlargest(n, "r_c"),
            "D_confidence": agreement.nlargest(n, "r_dc"),
        }
        random_metrics = []
        for random_seed in range(5):
            rg = np.random.default_rng(random_seed)
            ids = rg.choice(len(agreement), size=n, replace=False)
            random_metrics.append(agreement.iloc[ids])
        candidates["random"] = pd.concat(random_metrics, ignore_index=True)
        for method, selected in candidates.items():
            # Random is five matched draws; report mean/std across draws.
            if method == "random":
                grouped = selected.groupby(np.arange(len(selected)) // n)
                values = {
                    "WW_precision": grouped["WW_ratio"].mean().mean(),
                    "WW_precision_std": grouped["WW_ratio"].mean().std(ddof=0),
                    "HC90WW_precision": grouped["HC90WW_ratio"].mean().mean(),
                    "HC90WW_precision_std": grouped["HC90WW_ratio"].mean().std(ddof=0),
                    "CC_precision": grouped["CC_ratio"].mean().mean(),
                    "CC_precision_std": grouped["CC_ratio"].mean().std(ddof=0),
                    "WW_pixel_coverage": grouped["WW_ratio"].sum().div(total_ww + EPS).mean(),
                    "WW_pixel_coverage_std": grouped["WW_ratio"].sum().div(total_ww + EPS).std(ddof=0),
                    "HC90WW_pixel_coverage": grouped["HC90WW_ratio"].sum().div(total_hc + EPS).mean(),
                    "HC90WW_pixel_coverage_std": grouped["HC90WW_ratio"].sum().div(total_hc + EPS).std(ddof=0),
                    "CC_exposure": grouped["CC_ratio"].sum().div(total_cc + EPS).mean(),
                    "CC_exposure_std": grouped["CC_ratio"].sum().div(total_cc + EPS).std(ddof=0),
                }
            else:
                values = {
                    "WW_precision": _mean(selected["WW_ratio"]),
                    "WW_precision_std": 0.0,
                    "HC90WW_precision": _mean(selected["HC90WW_ratio"]),
                    "HC90WW_precision_std": 0.0,
                    "CC_precision": _mean(selected["CC_ratio"]),
                    "CC_precision_std": 0.0,
                    "WW_pixel_coverage": float(selected["WW_ratio"].sum() / (total_ww + EPS)),
                    "WW_pixel_coverage_std": 0.0,
                    "HC90WW_pixel_coverage": float(selected["HC90WW_ratio"].sum() / (total_hc + EPS)),
                    "HC90WW_pixel_coverage_std": 0.0,
                    "CC_exposure": float(selected["CC_ratio"].sum() / (total_cc + EPS)),
                    "CC_exposure_std": 0.0,
                }
            rows.append({"coverage": frac, "num_selected": n, "method": method, **values})
    return pd.DataFrame(rows)


def _bootstrap_image_metrics(agreement_df: pd.DataFrame, n_boot: int, seed: int):
    images = np.array(sorted(agreement_df["image_id"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(images, size=len(images), replace=True)
        parts = [agreement_df[agreement_df["image_id"] == image] for image in sampled]
        g = pd.concat(parts, ignore_index=True)
        contradiction = g[g["D"] > 0].copy()
        q = contradiction.copy()
        # Keep fixed global quartile boundaries in the bootstrap: D itself is
        # the only ranking variable and no outcome is used to form bins.
        if len(q) >= 4:
            q["d_q"] = pd.qcut(q["D"].rank(method="first"), 4,
                                labels=["Q1", "Q2", "Q3", "Q4"])
        n20 = max(1, int(np.ceil(len(g) * .2)))
        rows.append({
            "bootstrap": b,
            "rho_D_WW": _rho(contradiction["D"], contradiction["WW_ratio"]),
            "rho_D_HC90WW": _rho(contradiction["D"], contradiction["HC90WW_ratio"]),
            "rho_D_CC": _rho(contradiction["D"], contradiction["CC_ratio"]),
            "Q4_minus_Q1_WW": _mean(q.loc[q["d_q"] == "Q4", "WW_ratio"]) - _mean(q.loc[q["d_q"] == "Q1", "WW_ratio"]),
            "Q4_minus_Q1_HC90WW": _mean(q.loc[q["d_q"] == "Q4", "HC90WW_ratio"]) - _mean(q.loc[q["d_q"] == "Q1", "HC90WW_ratio"]),
            "Q4_minus_Q1_CC": _mean(q.loc[q["d_q"] == "Q4", "CC_ratio"]) - _mean(q.loc[q["d_q"] == "Q1", "CC_ratio"]),
            "top20_D_WW": _mean(g.nlargest(n20, "r_d")["WW_ratio"]),
            "top20_D_conf_WW": _mean(g.nlargest(n20, "r_dc")["WW_ratio"]),
        })
    return pd.DataFrame(rows)


def _ci(series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return [None, None]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _plot_quantiles(q: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = np.arange(len(q))
    ax.plot(x, q["WW_ratio_mean"], marker="o", label="WW ratio")
    ax.plot(x, q["HC90WW_ratio_mean"], marker="o", label="HC90WW ratio")
    ax.plot(x, q["CC_ratio_mean"], marker="o", label="CC ratio")
    ax.set_xticks(x, q["group"])
    ax.set_xlabel("D quartile (contradiction tokens)")
    ax.set_ylabel("mean token occupancy ratio")
    ax.set_title("Risk versus contradiction strength")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "d_quartile_risk.png", bbox_inches="tight"); plt.close(fig)


def _plot_calibration(dec: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = np.arange(len(dec))
    ax.plot(x, dec["observed_WW_risk"], marker="o", label="WW")
    ax.plot(x, dec["observed_HC90WW_risk"], marker="o", label="HC90WW")
    ax.set_xticks(x, dec["D_decile"])
    ax.set_xlabel("D decile")
    ax.set_ylabel("observed risk")
    ax.set_title("D strength to observed shared-error risk")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "d_decile_calibration.png", bbox_inches="tight"); plt.close(fig)


def _heatmap(interaction: pd.DataFrame, value: str, title: str, filename: str, out: Path):
    piv = interaction.pivot(index="D_quartile", columns="confidence_quartile", values=value)
    piv = piv.reindex(index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])
    fig, ax = plt.subplots(figsize=(6.3, 5.2), dpi=160)
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(4), ["C-Q1", "C-Q2", "C-Q3", "C-Q4"])
    ax.set_yticks(range(4), ["D-Q1", "D-Q2", "D-Q3", "D-Q4"])
    ax.set_xlabel("confidence quartile"); ax.set_ylabel("D quartile")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            v = piv.iloc[i, j]
            if pd.notna(v): ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / filename, bbox_inches="tight"); plt.close(fig)


def _plot_ranking(rank: pd.DataFrame, out: Path):
    r = rank[rank["coverage"] == 0.20]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    order = ["random", "confidence", "D", "D_confidence"]
    r = r.set_index("method").reindex(order).reset_index()
    ax.bar(r["method"], r["WW_precision"], yerr=r["WW_precision_std"], capsize=4, label="WW ratio")
    ax.set_ylabel("top-20% mean WW ratio"); ax.set_title("Matched-budget risk ranking")
    fig.tight_layout(); fig.savefig(out / "ranking_top20_ww.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(r["method"], r["CC_exposure"], yerr=r["CC_exposure_std"], capsize=4, color="#777777")
    ax.set_ylabel("top-20% mean CC ratio"); ax.set_title("Matched-budget CC exposure")
    fig.tight_layout(); fig.savefig(out / "ranking_top20_cc.png", bbox_inches="tight"); plt.close(fig)


def _plot_epoch(e: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(e["epoch"], e["rho_D_WW"], marker="o", label="D vs WW")
    ax.plot(e["epoch"], e["rho_D_HC90WW"], marker="o", label="D vs HC90WW")
    ax.plot(e["epoch"], e["rho_D_CC"], marker="o", label="D vs CC")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("Spearman rho"); ax.set_title("Epoch stability")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "epoch_spearman.png", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token-records", default="outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv")
    ap.add_argument("--output-dir", default="outputs/adaptive_suppression_diagnostic/glas_standard_cps_uni256")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    out = Path(args.output_dir); plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True); plots.mkdir(exist_ok=True)

    df = pd.read_csv(args.token_records)
    required = ["image_id", "epoch", "D", "agreement", "contradiction", "WW_ratio", "HC90WW_ratio", "CC_ratio", "direction_correct", "gt_pure_or_mixed", "hard_A", "conf_A", "conf_B"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise RuntimeError(f"E4 token records missing required fields: {missing}")
    bool_cols = ["agreement", "contradiction", "hc_contradiction"]
    for c in bool_cols:
        if c in df: df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    numeric = ["D", "WW_ratio", "HC90WW_ratio", "CC_ratio", "shared_FP_ratio", "shared_FN_ratio", "direction_correct", "conf_A", "conf_B", "hard_A"]
    for c in numeric:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mean_confidence"] = (df["conf_A"] + df["conf_B"]) / 2
    df["D_plus"] = np.maximum(df["D"], 0)
    # Risk scores are diagnostics only; no outcome is used to define them.
    positive = df.loc[df["agreement"] & (df["D"] > 0), "D_plus"]
    d_scale = float(positive.median()) if len(positive) else 1.0
    d_scale = max(d_scale, EPS)
    df["r_d"] = df["D_plus"]
    df["r_c"] = 2 * (df["mean_confidence"] - 0.5)
    df["r_dc"] = np.tanh(df["D_plus"] / d_scale) * df["r_c"]
    agreement = df[df["agreement"]].copy()
    contradiction = agreement[agreement["D"] > 0].copy()
    _add_quantile_labels(contradiction, "D", "d_quartile", 4)
    _add_quantile_labels(contradiction, "D", "d_decile", 10)
    # Write all E4 rows with the continuous diagnostic columns for reuse.
    _write_csv(df, out / "d_token_records.csv")

    quartiles = _group_metrics(contradiction, "d_quartile", ["Q1", "Q2", "Q3", "Q4"])
    _write_csv(quartiles, out / "d_quantile_metrics.csv")
    decile_rows = []
    for label in [f"D{i}" for i in range(1, 11)]:
        g = contradiction[contradiction["d_decile"] == label]
        decile_rows.append({
            "D_decile": label, "num_tokens": len(g), "num_images": g["image_id"].nunique(),
            "D_mean": _mean(g["D"]), "D_median": _median(g["D"]),
            "observed_WW_risk": _mean(g["WW_ratio"]), "observed_HC90WW_risk": _mean(g["HC90WW_ratio"]),
            "observed_CC_ratio": _mean(g["CC_ratio"]), "direction_accuracy": _mean(g["direction_correct"]),
        })
    deciles = pd.DataFrame(decile_rows)
    _write_csv(deciles, out / "d_decile_metrics.csv")
    _write_csv(deciles, out / "d_calibration_metrics.csv")

    interaction_rows = []
    _add_quantile_labels(contradiction, "mean_confidence", "confidence_quartile", 4)
    for d_q in ["Q1", "Q2", "Q3", "Q4"]:
        for c_q in ["Q1", "Q2", "Q3", "Q4"]:
            g = contradiction[(contradiction["d_quartile"] == d_q) & (contradiction["confidence_quartile"] == c_q)]
            interaction_rows.append({"D_quartile": d_q, "confidence_quartile": c_q, "num_tokens": len(g), "num_images": g["image_id"].nunique(), "mean_D": _mean(g["D"]), "mean_confidence": _mean(g["mean_confidence"]), "WW_ratio": _mean(g["WW_ratio"]), "HC90WW_ratio": _mean(g["HC90WW_ratio"]), "CC_ratio": _mean(g["CC_ratio"]), "direction_accuracy": _mean(g["direction_correct"])})
    interaction = pd.DataFrame(interaction_rows)
    _write_csv(interaction, out / "confidence_interaction_metrics.csv")

    strat_rows = _stratified_correlations(contradiction, "contradiction")
    for name, g in [("HC_CONTRADICTION", contradiction[contradiction.get("hc_contradiction", False)] if "hc_contradiction" in contradiction else contradiction.iloc[0:0]), ("WW_HEAVY", contradiction[contradiction["WW_ratio"] > .5]), ("HC90WW_HEAVY", contradiction[contradiction["HC90WW_ratio"] > .5])]:
        strat_rows.extend(_stratified_correlations(g, name))
    _write_csv(pd.DataFrame(strat_rows), out / "direction_metrics.csv")

    rank = _ranking_metrics(agreement, args.seed)
    _write_csv(rank, out / "ranking_metrics.csv")

    # Epoch and image-level diagnostics are kept separate to avoid treating
    # repeated tokens as independent evidence of temporal stability.
    epoch_rows = []
    image_rows = []
    for epoch, eg in sorted(contradiction.groupby("epoch"), key=lambda x: x[0]):
        q4, q1 = eg[eg["d_quartile"] == "Q4"], eg[eg["d_quartile"] == "Q1"]
        epoch_rows.append({"epoch": epoch, "num_contradiction_tokens": len(eg), "num_images": eg["image_id"].nunique(), "rho_D_WW": _rho(eg["D"], eg["WW_ratio"]), "rho_D_HC90WW": _rho(eg["D"], eg["HC90WW_ratio"]), "rho_D_CC": _rho(eg["D"], eg["CC_ratio"]), "Q4_Q1_WW_ratio": _safe_ratio(_mean(q4["WW_ratio"]), _mean(q1["WW_ratio"])), "Q4_Q1_HC90WW_ratio": _safe_ratio(_mean(q4["HC90WW_ratio"]), _mean(q1["HC90WW_ratio"])), "Q4_direction_accuracy": _mean(q4["direction_correct"]), "positive_image_rho_fraction": None})
        for image, ig in eg.groupby("image_id"):
            image_rows.append({"image_id": image, "epoch": epoch, "num_contradiction_tokens": len(ig), "mean_D": _mean(ig["D"]), "rho_D_WW": _rho(ig["D"], ig["WW_ratio"]), "rho_D_HC90WW": _rho(ig["D"], ig["HC90WW_ratio"]), "rho_D_CC": _rho(ig["D"], ig["CC_ratio"]), "insufficient_tokens": len(ig) < 3})
    epoch_df = pd.DataFrame(epoch_rows); image_df = pd.DataFrame(image_rows)
    _write_csv(epoch_df, out / "epoch_metrics.csv"); _write_csv(image_df, out / "image_metrics.csv")

    boot = _bootstrap_image_metrics(agreement, args.bootstrap, args.seed)
    _write_csv(boot, out / "bootstrap_metrics.csv")

    _plot_quantiles(quartiles, plots); _plot_calibration(deciles, plots)
    _heatmap(interaction, "WW_ratio", "WW ratio: D × confidence", "interaction_ww.png", plots)
    _heatmap(interaction, "HC90WW_ratio", "HC90WW ratio: D × confidence", "interaction_hc90ww.png", plots)
    _heatmap(interaction, "CC_ratio", "CC ratio: D × confidence", "interaction_cc.png", plots)
    _heatmap(interaction, "direction_accuracy", "Direction accuracy: D × confidence", "interaction_direction.png", plots)
    _plot_ranking(rank, plots); _plot_epoch(epoch_df, plots)

    rho_ww = _rho(contradiction["D"], contradiction["WW_ratio"])
    rho_hc = _rho(contradiction["D"], contradiction["HC90WW_ratio"])
    rho_cc = _rho(contradiction["D"], contradiction["CC_ratio"])
    q = quartiles.set_index("group")
    rank20 = rank[rank["coverage"] == .2].set_index("method")
    image_rhos = image_df[~image_df["insufficient_tokens"]]
    positive_image_fraction = float((image_rhos["rho_D_WW"] > 0).mean()) if len(image_rhos) else None
    positive_image_hc_fraction = float((image_rhos["rho_D_HC90WW"] > 0).mean()) if len(image_rhos) else None
    # Fixed descriptive rule, declared in README: A requires positive global
    # risk association, Q4>Q1, D beats random at top20, and majority positive
    # image-level correlations. B is reserved for a clear combined-score gain.
    d20 = float(rank20.loc["D", "WW_precision"])
    dc20 = float(rank20.loc["D_confidence", "WW_precision"])
    random20 = float(rank20.loc["random", "WW_precision"])
    if rho_ww > 0.15 and rho_hc > 0.15 and q.loc["Q4", "WW_ratio_mean"] > q.loc["Q1", "WW_ratio_mean"] and d20 > random20 and (positive_image_fraction is None or (positive_image_fraction >= .5 and positive_image_hc_fraction >= .5)):
        outcome = "Outcome B — D is informative but incomplete." if dc20 > d20 + .02 else "Outcome A — D is a useful continuous contradiction-reliability signal."
    else:
        outcome = "Outcome C — Contradiction detection is useful, but D magnitude is not a reliable continuous risk estimator."

    summary = {
        "num_tokens_total": int(len(df)), "num_agreement_tokens": int(len(agreement)), "num_contradiction_tokens": int(len(contradiction)), "num_images": int(df["image_id"].nunique()), "epochs": sorted([int(x) for x in df["epoch"].unique()]),
        "spearman": {"D_vs_WW": rho_ww, "D_vs_HC90WW": rho_hc, "D_vs_CC": rho_cc, "bootstrap_CI": {"D_vs_WW": _ci(boot["rho_D_WW"]), "D_vs_HC90WW": _ci(boot["rho_D_HC90WW"]), "D_vs_CC": _ci(boot["rho_D_CC"])}},
        "quartile": {"Q1_WW": float(q.loc["Q1", "WW_ratio_mean"]), "Q4_WW": float(q.loc["Q4", "WW_ratio_mean"]), "Q1_HC90WW": float(q.loc["Q1", "HC90WW_ratio_mean"]), "Q4_HC90WW": float(q.loc["Q4", "HC90WW_ratio_mean"]), "Q1_CC": float(q.loc["Q1", "CC_ratio_mean"]), "Q4_CC": float(q.loc["Q4", "CC_ratio_mean"]), "Q1_direction_acc": float(q.loc["Q1", "direction_accuracy"]), "Q4_direction_acc": float(q.loc["Q4", "direction_accuracy"])},
        "combined_score": {"top10_WW": float(rank[rank.coverage == .1].set_index("method").loc["D_confidence", "WW_precision"]), "top20_WW": float(rank20.loc["D_confidence", "WW_precision"]), "top20_HC90WW": float(rank20.loc["D_confidence", "HC90WW_precision"]), "top20_CC": float(rank20.loc["D_confidence", "CC_exposure"])},
        "confidence_only": {"top20_WW": float(rank20.loc["confidence", "WW_precision"])},
        "D_only": {"top20_WW": d20, "top20_HC90WW": float(rank20.loc["D", "HC90WW_precision"]), "top20_CC": float(rank20.loc["D", "CC_exposure"])},
        "random": {"top20_WW_mean": random20, "top20_WW_std": float(rank20.loc["random", "WW_precision_std"])},
        "pure_mixed": {r["stratum"] + "_D_vs_WW": r["rho_D_WW"] for r in strat_rows if r["subset"] == "contradiction" and r["stratum"] in {"PURE", "MIXED"}},
        "epoch_stability": {"positive_image_rho_fraction_WW": positive_image_fraction, "positive_image_rho_fraction_HC90WW": positive_image_hc_fraction, "median_image_rho_WW": float(image_rhos["rho_D_WW"].median()) if len(image_rhos) else None, "median_image_rho_HC90WW": float(image_rhos["rho_D_HC90WW"].median()) if len(image_rhos) else None, "IQR_image_rho_WW": [float(image_rhos["rho_D_WW"].quantile(.25)), float(image_rhos["rho_D_WW"].quantile(.75))] if len(image_rhos) else None},
        "outcome": outcome,
        "bootstrap": {"n": args.bootstrap, "Q4_minus_Q1_WW_CI": _ci(boot["Q4_minus_Q1_WW"]), "Q4_minus_Q1_HC90WW_CI": _ci(boot["Q4_minus_Q1_HC90WW"]), "Q4_minus_Q1_CC_CI": _ci(boot["Q4_minus_Q1_CC"]), "top20_D_WW_CI": _ci(boot["top20_D_WW"]), "top20_D_conf_WW_CI": _ci(boot["top20_D_conf_WW"])},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "README.md").write_text(f"""# E4-D Adaptive Suppression Feasibility Diagnostic\n\nClosed offline analysis of `{args.token_records}`. No model forward, training, memory rebuild, or threshold selection is performed.\n\nD quartiles/deciles are formed using D only among agreement contradiction tokens (D>0). Ranking scores are descriptive and are not inserted into training. Bootstrap resamples image IDs, retaining all epochs for each sampled image, so token-level bootstrap is avoided.\n\nOutcome rule is fixed in code: Outcome A requires positive global D-risk correlations, Q4>Q1 WW risk, D top-20 ranking above random, and at least half of valid images with positive D-vs-WW rho. Outcome B is used when those conditions hold but D×confidence improves top-20 WW precision by >0.02 over D-only; otherwise Outcome C.\n\nResult: **{outcome}**\n""")

    print("=== E4-D Adaptive Suppression Feasibility ===")
    print(f"Tokens: {len(df)} | Contradiction tokens: {len(contradiction)}")
    print(f"Spearman D vs WW / HC90WW / CC: {rho_ww:.4f} / {rho_hc:.4f} / {rho_cc:.4f}")
    print(f"Q1→Q4 WW: {q.loc['Q1','WW_ratio_mean']:.4f} → {q.loc['Q4','WW_ratio_mean']:.4f}")
    print(f"Q1→Q4 HC90WW: {q.loc['Q1','HC90WW_ratio_mean']:.4f} → {q.loc['Q4','HC90WW_ratio_mean']:.4f}")
    print(f"Top-20% WW random / confidence / D / D×confidence: {random20:.4f} / {rank20.loc['confidence','WW_precision']:.4f} / {d20:.4f} / {dc20:.4f}")
    print(f"Outcome: {outcome}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
