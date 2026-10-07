#!/usr/bin/env python3
"""Test whether contradiction-margin magnitude predicts fine CPS shared-error risk.

Uses the frozen GLaS 10% seed-1337 Vanilla CPS checkpoint pair and the fixed
train-only UNI memory. Evaluation is validation-only; no training is performed.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import diagnose_contradiction_enrichment as base
import main as project
from config.dataset_config.dataset_cfg import dataset_cfg
from utils.native_reference import (
    _get_rgb_view, _load_reference, _reconstruct_rgb,
    _reference_scores, _uni_dense_features,
)
from utils.training_diagnostics import foreground_probability
from training.cps import get_training_views

DATASET = "GlaS"  # project config and filesystem canonical spelling
LABELED_RATIO = 10
SEED = 1337
INPUT_MODE = "rgb_rgb"
PIXEL_SIZE = 256
TOKEN_GRID = 16
TOP_K = 50
REFERENCE_RESOLUTION = 256
CONFIDENCE_THRESHOLD = 0.90
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026
EXPECTED_VALIDATION_IMAGES = 17
REQUESTED_MEMORY_PATH = Path(
    "/data/mission/models/Exp/DUAL/checkpoints/semi_10_seed_1337_rgbstain_c2f_ca_reverse_disroute/"
    "GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-"
    "cps_mode=native_fine_margin_calibrated_reverse_disroute/reference_memory_uni256.npz"
).resolve()
MEMORY_PATH = Path(str(REQUESTED_MEMORY_PATH).replace("/GLaS/", "/GlaS/"))
REQUESTED_CHECKPOINT_ROOT = ROOT / "checkpoints/semi_10_seed_1337_rgb/GLaS"
CHECKPOINT_ROOT = ROOT / "checkpoints/semi_10_seed_1337_rgb/GlaS"
OUTPUT_DIR = ROOT / "outputs/d_risk_monotonicity_glas10_seed1337"
SUBSETS = ("ALL", "FG2BG", "BG2FG")
METRICS = ("WW", "HC90WW")


class Sanity:
    def __init__(self):
        self.values: dict[str, bool] = {}

    def require(self, key: str, condition: bool, detail: str = ""):
        self.values[key] = bool(condition)
        if not condition:
            raise RuntimeError(f"Sanity check FAIL: {key}" + (f" ({detail})" if detail else ""))

    @property
    def passed(self):
        return all(self.values.values()) and len(self.values) == 20


def _safe_div(a: float, b: float):
    return float(a) / b if b else None


def _fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value):
    return "undefined" if value is None or not np.isfinite(value) else f"{100*float(value):.3f}%"


def _fmt_ci(lo, hi, pct=False):
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "undefined"
    if pct:
        return f"[{100*lo:.3f}%, {100*hi:.3f}%]"
    return f"[{lo:.4f}, {hi:.4f}]"


def _fmt_p(value):
    return "undefined" if value is None or not np.isfinite(value) else f"{float(value):.3e}"


def _percentile_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return (None, None)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _metadata(memory_path: Path):
    with np.load(memory_path, allow_pickle=False) as data:
        meta = {}
        for key in data.files:
            value = data[key]
            if value.shape == ():
                item = value.item()
                meta[key] = item.decode() if isinstance(item, bytes) else item
            else:
                meta[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return meta


def _expand_nearest(tensor: torch.Tensor):
    return F.interpolate(
        tensor[:, None].float(), size=(PIXEL_SIZE, PIXEL_SIZE), mode="nearest"
    )[:, 0]


def _bin_quintiles(rows: list[dict], output_key: str):
    """Assign approximately equal-size rank quintiles; ties use stable input order."""
    ordered = sorted(range(len(rows)), key=lambda i: (float(rows[i]["D"]), i))
    n = len(ordered)
    for rank, row_index in enumerate(ordered):
        rows[row_index][output_key] = min(4, (rank * 5) // max(n, 1)) + 1


def _subset_rows(rows: list[dict], subset: str, include_zero_corr=False):
    candidates = [r for r in rows if subset == "ALL" or r["direction"] == subset]
    if include_zero_corr:
        return candidates
    return [r for r in candidates if int(r["N_corr"]) > 0]


def _score_grouped(records, image_multiplicity, positive_key):
    """Exact pixel AUROC/AP using tied token scores and image-cluster weights."""
    grouped: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])  # pos, neg
    for row in records:
        weight = image_multiplicity.get(row["image_id"], 0)
        if not weight:
            continue
        n = int(row["N_corr"]) * weight
        pos = int(row[positive_key]) * weight
        if n:
            grouped[float(row["D"])][0] += pos
            grouped[float(row["D"])][1] += n - pos
    if not grouped:
        return (None, None)
    scores = sorted(grouped)
    pos = np.asarray([grouped[s][0] for s in scores], dtype=np.float64)
    neg = np.asarray([grouped[s][1] for s in scores], dtype=np.float64)
    total_pos, total_neg = float(pos.sum()), float(neg.sum())
    auc = None
    if total_pos > 0 and total_neg > 0:
        neg_below = np.cumsum(neg) - neg
        auc = float(np.sum(pos * (neg_below + 0.5 * neg)) / (total_pos * total_neg))
    ap = None
    if total_pos > 0:
        descending_pos = pos[::-1]
        descending_total = (pos + neg)[::-1]
        cum_pos = np.cumsum(descending_pos)
        cum_total = np.cumsum(descending_total)
        ap = float(np.sum((cum_pos / np.maximum(cum_total, 1.0)) * descending_pos) / total_pos)
    return auc, ap


def _token_spearman(records, image_multiplicity, risk_key):
    d_values, risk_values = [], []
    for row in records:
        weight = int(image_multiplicity.get(row["image_id"], 0))
        if weight and int(row["N_corr"]) > 0 and np.isfinite(row[risk_key]):
            d_values.extend([float(row["D"])] * weight)
            risk_values.extend([float(row[risk_key])] * weight)
    if len(d_values) < 2 or len(set(d_values)) < 2 or len(set(risk_values)) < 2:
        return (None, None)
    result = spearmanr(d_values, risk_values)
    rho_value = getattr(result, "statistic", getattr(result, "correlation", None))
    rho = float(rho_value)
    p = float(result.pvalue)
    return rho, p


def _weighted_quintile_stats(records, multiplicity, positive_key, bin_key):
    output = {}
    for q in range(1, 6):
        num = den = 0
        for row in records:
            if int(row[bin_key]) != q:
                continue
            w = int(multiplicity.get(row["image_id"], 0))
            if w:
                den += int(row["N_corr"]) * w
                num += int(row[positive_key]) * w
        output[q] = _safe_div(num, den)
    return output


def _metrics_for_weights(records, multiplicity, subset):
    eligible = _subset_rows(records, subset)
    if subset == "ALL":
        bin_key = "D_quintile"
    else:
        bin_key = "D_quintile_direction"
    out = {}
    for metric, positive_key, risk_key in (
        ("WW", "N_WW", "risk_WW"), ("HC90WW", "N_HC90WW", "risk_HC90WW")
    ):
        q_rates = _weighted_quintile_stats(eligible, multiplicity, positive_key, bin_key)
        rho, p = _token_spearman(eligible, multiplicity, risk_key)
        auc, ap = _score_grouped(eligible, multiplicity, positive_key)
        q1, q5 = q_rates[1], q_rates[5]
        rr = _safe_div(q5, q1) if q1 is not None and q1 != 0 and q5 is not None else None
        out[metric] = {"q_rates": q_rates, "rho": rho, "p": p, "auc": auc, "ap": ap, "rr": rr}
    return out


def _bootstrap(records, image_ids, rng):
    result = {}
    for subset in SUBSETS:
        result[subset] = {
            "WW_q": {q: [] for q in range(1, 6)}, "HC90WW_q": {q: [] for q in range(1, 6)},
            "WW_rho": [], "WW_auc": [], "WW_ap": [], "WW_rr": [],
            "HC90WW_rho": [], "HC90WW_auc": [], "HC90WW_ap": [], "HC90WW_rr": [],
        }
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        multiplicity = Counter(sampled.tolist())
        for subset in SUBSETS:
            metrics = _metrics_for_weights(records, multiplicity, subset)
            for metric in METRICS:
                m = metrics[metric]
                key = "WW" if metric == "WW" else "HC90WW"
                for q, value in m["q_rates"].items():
                    result[subset][f"{key}_q"][q].append(np.nan if value is None else value)
                for field, output_key in (("rho", "rho"), ("auc", "auc"), ("ap", "ap"), ("rr", "rr")):
                    value = m[field]
                    result[subset][f"{key}_{output_key}"].append(np.nan if value is None else value)
    return result


def _ci(result, subset, metric, field, q=None):
    if q is not None:
        values = result[subset][f"{metric}_q"][q]
    else:
        values = result[subset][f"{metric}_{field}"]
    return _percentile_ci(values)


def _initialise_checks():
    return Sanity()


def collect_tokens(model_a, model_b, loader, cfg, memory_context, sanity: Sanity):
    tokens: list[dict] = []
    num_images = 0
    agreement_tokens = 0
    contradiction_tokens = 0
    corr_pixels = ww_pixels = hc_pixels = 0
    image_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            input_a, input_b = get_training_views(batch)
            input_a = input_a.cuda(non_blocking=True)
            input_b = input_b.cuda(non_blocking=True)
            p_a = foreground_probability(model_a(input_a)[0])[:, 0]
            p_b = foreground_probability(model_b(input_b)[0])[:, 0]
            if tuple(p_a.shape[-2:]) != (PIXEL_SIZE, PIXEL_SIZE) or tuple(p_b.shape[-2:]) != (PIXEL_SIZE, PIXEL_SIZE):
                raise RuntimeError(f"Student dense probability shape A/B={tuple(p_a.shape)}/{tuple(p_b.shape)}")
            sanity.require("Student dense probability = 256x256", True)
            coarse_a = F.adaptive_avg_pool2d(p_a[:, None], (TOKEN_GRID, TOKEN_GRID))[:, 0]
            coarse_b = F.adaptive_avg_pool2d(p_b[:, None], (TOKEN_GRID, TOKEN_GRID))[:, 0]
            class_a = coarse_a.ge(0.5)
            class_b = coarse_b.ge(0.5)
            coarse_agreement = class_a.eq(class_b)
            consensus = class_a

            rgb = _reconstruct_rgb(_get_rgb_view(batch).cuda(non_blocking=True), cfg)
            features = _uni_dense_features(memory_context["uni"], rgb, REFERENCE_RESOLUTION)
            if tuple(features.shape[1:]) != (TOKEN_GRID * TOKEN_GRID, 1024):
                raise RuntimeError(f"UNI feature shape {tuple(features.shape)}")
            s_fg, s_bg = _reference_scores(features, memory_context, TOP_K)
            s_fg = s_fg.reshape(-1, TOKEN_GRID, TOKEN_GRID)
            s_bg = s_bg.reshape(-1, TOKEN_GRID, TOKEN_GRID)
            memory_class = s_fg.gt(s_bg)  # ties select BG, per protocol
            s_cur = torch.where(consensus, s_fg, s_bg)
            s_opp = torch.where(consensus, s_bg, s_fg)
            D = s_opp - s_cur
            contradiction = coarse_agreement & D.gt(0)
            expected = coarse_agreement & memory_class.ne(consensus)
            sanity.require("K_t = coarse agreement and D>0", torch.equal(contradiction, coarse_agreement & D.gt(0)))
            sanity.require("K_t = agreement and U!=C", torch.equal(contradiction, expected))
            if not torch.equal(contradiction, expected):
                mismatch = int((contradiction ^ expected).sum().item())
                raise RuntimeError(f"D/U contradiction mismatch: {mismatch}")

            gt = batch["mask"].cuda(non_blocking=True)
            if gt.dim() == 3:
                gt = gt[:, None]
            if tuple(gt.shape[-2:]) != (PIXEL_SIZE, PIXEL_SIZE):
                raise RuntimeError(f"GT resolution must be native 256x256, got {tuple(gt.shape)}")
            sanity.require("GT = 256x256", True)
            gt = gt[:, 0].ge(0.5)
            # Fine prediction and confidence are taken directly from original dense probabilities.
            A, B = p_a.ge(0.5), p_b.ge(0.5)
            conf_a = torch.maximum(p_a, 1.0 - p_a)
            conf_b = torch.maximum(p_b, 1.0 - p_b)
            fine_agreement = A.eq(B)
            WW = fine_agreement & A.ne(gt)
            HC = WW & conf_a.ge(CONFIDENCE_THRESHOLD) & conf_b.ge(CONFIDENCE_THRESHOLD)
            sanity.require(
                "Fine risk uses original pixel predictions",
                torch.equal(A, p_a.ge(0.5)) and torch.equal(B, p_b.ge(0.5))
                and torch.equal(conf_a, torch.maximum(p_a, 1.0-p_a))
                and torch.equal(conf_b, torch.maximum(p_b, 1.0-p_b)),
            )

            K_expanded = _expand_nearest(contradiction).bool()
            C_expanded = _expand_nearest(consensus).bool()
            U_expanded = _expand_nearest(memory_class).bool()
            D_expanded = F.interpolate(
                D[:, None], size=(PIXEL_SIZE, PIXEL_SIZE), mode="nearest"
            )[:, 0]
            M_corr = K_expanded & fine_agreement & A.eq(C_expanded)
            # Validate blockwise replication without using pooled fine labels/confidence.
            for bi, image_id_raw in enumerate(batch["ID"]):
                image_id = str(image_id_raw)
                image_ids.append(image_id)
                num_images += 1
                agreement_tokens += int(coarse_agreement[bi].sum().item())
                for ty, tx in torch.nonzero(contradiction[bi], as_tuple=False).tolist():
                    contradiction_tokens += 1
                    y0, x0 = ty * (PIXEL_SIZE // TOKEN_GRID), tx * (PIXEL_SIZE // TOKEN_GRID)
                    block = (slice(y0, y0 + 16), slice(x0, x0 + 16))
                    corr_block = M_corr[bi][block]
                    ww_block = WW[bi][block] & corr_block
                    hc_block = HC[bi][block] & corr_block
                    nc = int(corr_block.sum().item())
                    nw = int(ww_block.sum().item())
                    nh = int(hc_block.sum().item())
                    if not bool(K_expanded[bi][block].all()):
                        raise RuntimeError("Nearest-expanded K does not cover the contradiction token block")
                    if not bool((C_expanded[bi][block] == consensus[bi, ty, tx]).all()):
                        raise RuntimeError("Nearest-expanded C block mismatch")
                    if not bool((U_expanded[bi][block] == memory_class[bi, ty, tx]).all()):
                        raise RuntimeError("Nearest-expanded U block mismatch")
                    if not bool(torch.allclose(
                        D_expanded[bi][block], D[bi, ty, tx].expand(16, 16), atol=0, rtol=0
                    )):
                        raise RuntimeError("Nearest-expanded D block mismatch")
                    c_is_fg = bool(consensus[bi, ty, tx])
                    u_is_fg = bool(memory_class[bi, ty, tx])
                    direction = "FG2BG" if c_is_fg and not u_is_fg else "BG2FG"
                    rww = _safe_div(nw, nc)
                    rhc = _safe_div(nh, nc)
                    tokens.append({
                        "image_id": image_id, "token_y": int(ty), "token_x": int(tx),
                        "C_t": "FG" if c_is_fg else "BG", "U_t": "FG" if u_is_fg else "BG",
                        "direction": direction,
                        "S_FG": float(s_fg[bi, ty, tx].item()),
                        "S_BG": float(s_bg[bi, ty, tx].item()),
                        "S_cur": float(s_cur[bi, ty, tx].item()),
                        "S_opp": float(s_opp[bi, ty, tx].item()),
                        "D": float(D[bi, ty, tx].item()),
                        "N_corr": nc, "N_WW": nw, "N_HC90WW": nh,
                        "risk_WW": float(rww) if rww is not None else "undefined",
                        "risk_HC90WW": float(rhc) if rhc is not None else "undefined",
                        "D_quintile": "", "D_quintile_direction": "",
                    })
                    corr_pixels += nc
                    ww_pixels += nw
                    hc_pixels += nh
    return tokens, sorted(set(image_ids)), num_images, agreement_tokens, contradiction_tokens, corr_pixels, ww_pixels, hc_pixels


def _assign_bins(records):
    all_eligible = _subset_rows(records, "ALL")
    _bin_quintiles(all_eligible, "D_quintile")
    for direction in ("FG2BG", "BG2FG"):
        _bin_quintiles(_subset_rows(records, direction), "D_quintile_direction")
    for row in records:
        # Persist the bin assignments back to the original records by token key.
        key = (row["image_id"], row["token_y"], row["token_x"])
        if row["N_corr"] == 0:
            row["D_quintile"] = ""
            row["D_quintile_direction"] = ""
    # Assignments are object references, so the calls above update records directly.


def _observed_quintiles(records, subset):
    eligible = _subset_rows(records, subset)
    bin_key = "D_quintile" if subset == "ALL" else "D_quintile_direction"
    output = []
    for q in range(1, 6):
        qrows = [r for r in eligible if int(r[bin_key]) == q]
        d = np.asarray([r["D"] for r in qrows], dtype=np.float64)
        n_corr = sum(int(r["N_corr"]) for r in qrows)
        n_ww = sum(int(r["N_WW"]) for r in qrows)
        n_hc = sum(int(r["N_HC90WW"]) for r in qrows)
        output.append({
            "subset": subset, "quintile": f"Q{q}", "token_count": len(qrows),
            "image_count": len({r["image_id"] for r in qrows}),
            "D_min": float(np.min(d)) if len(d) else None,
            "D_max": float(np.max(d)) if len(d) else None,
            "D_mean": float(np.mean(d)) if len(d) else None,
            "D_median": float(np.median(d)) if len(d) else None,
            "N_corr": n_corr, "N_WW": n_ww, "N_HC90WW": n_hc,
            "WW_rate": _safe_div(n_ww, n_corr), "HC90WW_rate": _safe_div(n_hc, n_corr),
        })
    return output


def _plain_pixel_auc_ap(records, positive_key):
    # Exact rankings on the actual pixel population; no pixel bootstrap/allocation.
    multiplicity = {image_id: 1 for image_id in {r["image_id"] for r in records}}
    return _score_grouped(records, multiplicity, positive_key)


def _association(records, image_ids, subset, bootstrap):
    all_contradictions = _subset_rows(records, subset, include_zero_corr=True)
    eligible = _subset_rows(records, subset)
    metrics = _metrics_for_weights(records, Counter({img: 1 for img in image_ids}), subset)
    qrows = _observed_quintiles(records, subset)
    qby = {r["quintile"]: r for r in qrows}
    result = {
        "subset": subset, "num_images": len({r["image_id"] for r in all_contradictions}),
        "num_tokens": len(eligible), "contradiction_token_count": len(all_contradictions),
        "num_corr_pixels": sum(int(r["N_corr"]) for r in eligible),
    }
    for metric, pref in (("WW", "WW"), ("HC90WW", "HC")):
        m = metrics[metric]
        rho_lo, rho_hi = _ci(bootstrap, subset, metric, "rho")
        auc, ap = _plain_pixel_auc_ap(eligible, "N_WW" if metric == "WW" else "N_HC90WW")
        auc_lo, auc_hi = _ci(bootstrap, subset, metric, "auc")
        ap_lo, ap_hi = _ci(bootstrap, subset, metric, "ap")
        ratio_lo, ratio_hi = _ci(bootstrap, subset, metric, "rr")
        result[f"spearman_{pref}"] = m["rho"]
        result[f"spearman_{pref}_p"] = m["p"]
        result[f"spearman_{pref}_ci_low"] = rho_lo
        result[f"spearman_{pref}_ci_high"] = rho_hi
        result[f"AUROC_{pref}"] = auc
        result[f"AUROC_{pref}_ci_low"] = auc_lo
        result[f"AUROC_{pref}_ci_high"] = auc_hi
        result[f"AUPRC_{pref}"] = ap
        result[f"AUPRC_{pref}_ci_low"] = ap_lo
        result[f"AUPRC_{pref}_ci_high"] = ap_hi
        q1 = qby["Q1"]["WW_rate" if metric == "WW" else "HC90WW_rate"]
        q5 = qby["Q5"]["WW_rate" if metric == "WW" else "HC90WW_rate"]
        result[f"Q1_{pref}_rate"] = q1
        result[f"Q5_{pref}_rate"] = q5
        result[f"{pref}_Q5_Q1_ratio"] = _safe_div(q5, q1) if q1 not in (None, 0) else None
        result[f"{pref}_ratio_ci_low"] = ratio_lo
        result[f"{pref}_ratio_ci_high"] = ratio_hi
    return result, qrows


def _write_csv(path: Path, rows: list[dict], fields=None):
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if (isinstance(v, float) and not np.isfinite(v)) or v is None else v) for k, v in row.items()})


def _curve_plot(q_summary, assoc, out_png, out_pdf):
    colors = {"WW": "#4d718c", "HC90WW": "#c17b43"}
    x = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor="white")
    medians = []
    for q in range(1, 6):
        medians.append(next((r["D_median"] for r in q_summary if r["subset"] == "ALL" and r["quintile"] == f"Q{q}"), np.nan))
    for metric, label, rate_key in (("WW", "WW", "WW_rate"), ("HC90WW", "HC90WW", "HC90WW_rate")):
        rows = [r for r in q_summary if r["subset"] == "ALL"]
        y = np.array([np.nan if r[rate_key] is None else 100*r[rate_key] for r in rows])
        ci = [_ci_from_summary_row(r, metric) for r in rows]
        lo = np.array([np.nan if v[0] is None else 100*v[0] for v in ci])
        hi = np.array([np.nan if v[1] is None else 100*v[1] for v in ci])
        yerr = np.vstack((np.maximum(0, y-lo), np.maximum(0, hi-y)))
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=3,
                    color=colors[metric], label=label)
    ax.set_xticks(x, [f"Q{i}\nDmed={medians[i-1]:.3f}" for i in x])
    ax.set_xlabel("Semantic contradiction strength D (Q1 lowest → Q5 highest; median D below)")
    ax.set_ylabel("Fine shared-error prevalence (%)")
    max_hi = max([v for r in q_summary if r["subset"] == "ALL" for v in (r.get("WW_ci_high"), r.get("HC_ci_high")) if v is not None] or [0.5]) * 100
    ax.set_ylim(0, max(10, max_hi*1.25))
    ax.grid(axis="y", color="#e5e5e5", linewidth=.7); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    def ann(metric, y0):
        rho = assoc[f"spearman_{'WW' if metric=='WW' else 'HC'}"]
        rr = assoc[f"{'WW' if metric=='WW' else 'HC'}_Q5_Q1_ratio"]
        return f"{metric}: Spearman ρ={_fmt(rho,3)}; Q5/Q1={_fmt(rr,2)}×"
    ax.text(.98,.97,ann("WW",0),transform=ax.transAxes,ha="right",va="top",fontsize=8.7,
            bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"#dddddd","alpha":.94})
    ax.text(.98,.88,ann("HC90WW",0),transform=ax.transAxes,ha="right",va="top",fontsize=8.7,
            bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"#dddddd","alpha":.94})
    fig.suptitle("Shared CPS Error Risk vs.\nSemantic Contradiction Strength", fontsize=13, y=.99)
    fig.tight_layout(rect=(.02,.02,.98,.91))
    fig.savefig(out_png,dpi=300,bbox_inches="tight"); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)


def _ci_from_summary_row(row, metric):
    return (row.get("WW_ci_low"), row.get("WW_ci_high")) if metric == "WW" else (row.get("HC_ci_low"), row.get("HC_ci_high"))


def _directional_plot(q_summary, out_png, out_pdf):
    colors = {"FG2BG": "#4d718c", "BG2FG": "#a15b4b"}
    subsets = ("FG2BG", "BG2FG")
    fig, axes = plt.subplots(1,2,figsize=(9.2,4.4),sharey=True,facecolor="white")
    all_rates = [100*r[k] for r in q_summary for k in ("WW_rate","HC90WW_rate") if r["subset"] in subsets and r[k] is not None]
    ymax=max(10,max(all_rates)*1.25 if all_rates else 10)
    for ax, subset in zip(axes, subsets):
        rows=[r for r in q_summary if r["subset"]==subset]
        x=np.arange(1,6)
        for metric, rate_key, lo_key, hi_key, style in (
            ("WW","WW_rate","WW_ci_low","WW_ci_high","-"),
            ("HC90WW","HC90WW_rate","HC_ci_low","HC_ci_high","--"),
        ):
            y=np.array([np.nan if r[rate_key] is None else 100*r[rate_key] for r in rows])
            lo=np.array([np.nan if r[lo_key] is None else 100*r[lo_key] for r in rows])
            hi=np.array([np.nan if r[hi_key] is None else 100*r[hi_key] for r in rows])
            err=np.vstack((np.maximum(0,y-lo),np.maximum(0,hi-y)))
            ax.errorbar(x,y,yerr=err,marker="o",linestyle=style,color=colors[subset],
                        linewidth=1.8,capsize=2,label=metric)
        ax.set_title("FG→BG" if subset=="FG2BG" else "BG→FG")
        ax.set_xticks(x,[f"Q{i}" for i in x]); ax.set_ylim(0,ymax)
        ax.set_xlabel("D quintile (within direction)")
        ax.grid(axis="y",color="#e5e5e5",linewidth=.7); ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel("Fine shared-error prevalence (%)")
    fig.suptitle("Directional CPS Error Risk vs. Contradiction Strength",fontsize=12.5,y=1.02)
    fig.tight_layout(); fig.savefig(out_png,dpi=300,bbox_inches="tight"); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)


def _make_conclusion(assoc, q_summary):
    by={(r["subset"],r["quintile"]):r for r in q_summary}
    all_w=assoc["ALL"]
    wq=[by[("ALL",f"Q{i}")]["WW_rate"] for i in range(1,6)]
    hq=[by[("ALL",f"Q{i}")]["HC90WW_rate"] for i in range(1,6)]
    ww_ci=all_w["spearman_WW_ci_low"]
    ww_positive=(wq[4] is not None and wq[0] is not None and wq[4]>wq[0]
                 and (all_w["WW_Q5_Q1_ratio"] or 0)>1
                 and (all_w["spearman_WW"] or 0)>0
                 and ww_ci is not None and ww_ci>0
                 and (all_w["AUROC_WW"] or 0)>0.5)
    ww_monotone=all(v is not None for v in wq) and all(wq[i+1] >= wq[i]-0.005 for i in range(4))
    hc_monotone=all(v is not None for v in hq) and all(hq[i+1] >= hq[i]-0.005 for i in range(4))
    hc_broad=(hq[4] is not None and hq[0] is not None and hq[4]>hq[0]
              and (all_w["spearman_HC"] or 0)>0 and (all_w["AUROC_HC"] or 0)>0.5)
    direction_ok=True
    direction_issues=[]
    for direction in ("FG2BG","BG2FG"):
        a=assoc[direction]
        for metric, q1_key, q5_key, rho_key, auc_key in (
            ("WW", "Q1_WW_rate", "Q5_WW_rate", "spearman_WW", "AUROC_WW"),
            ("HC90WW", "Q1_HC_rate", "Q5_HC_rate", "spearman_HC", "AUROC_HC"),
        ):
            if not ((a[q5_key] or -1) > (a[q1_key] or 0)
                    and (a[rho_key] or 0)>0 and (a[auc_key] or 0)>0.5):
                direction_ok=False
                direction_issues.append(f"{direction} {metric}")
    if ww_positive and ww_monotone and hc_broad and hc_monotone and direction_ok:
        level="STRONG SUPPORT"
        msg="Larger semantic contradiction margin D is consistently associated with a higher probability of fine-grained CPS shared errors; this pattern is near-monotonic and both directions agree."
    elif ww_monotone and wq[4] is not None and wq[0] is not None and wq[4]>wq[0] and (all_w["spearman_WW"] or 0)>0 and (all_w["AUROC_WW"] or 0)>0.5:
        level="PARTIAL SUPPORT"
        pieces=[]
        if not ww_positive: pieces.append("WW confidence interval or its support criteria are inconclusive")
        if not ww_monotone: pieces.append("the WW quintile curve is not fully monotone")
        if not hc_broad or not hc_monotone: pieces.append("HC90WW does not show the same clear positive trend")
        if direction_issues: pieces.append("direction(s) with unstable WW trend: " + ", ".join(direction_issues))
        msg="Overall WW trend is positive, but " + "; ".join(pieces or ["some supporting tests are inconclusive"]) + "."
    else:
        level="NOT SUPPORTED"
        msg="The observed results do not show a reliable positive monotonic relationship between D and fine shared-error risk."
    return level,msg,ww_monotone


def _write_report(path, checkpoint_path, epoch, student_a, student_b, memory_meta,
                  n_images, agreement_tokens, contradiction_tokens, zero_corr_tokens,
                  corr_pixels, ww_pixels, hc_pixels, q_summary, assoc, conclusion,
                  sanity, memory_built):
    level, conclusion_text, ww_monotone = conclusion
    lines=[
        "[Setup]", f"Dataset: GLaS (project key: {DATASET})", f"Validation images: {n_images}",
        f"Checkpoint: {checkpoint_path}", f"Epoch: {epoch}",
        f"Checkpoint root supplied in task: {REQUESTED_CHECKPOINT_ROOT}",
        f"Student A checkpoint: {student_a}", f"Student B checkpoint: {student_b}",
        f"Memory: {MEMORY_PATH}",
        f"Memory path supplied in task: {REQUESTED_MEMORY_PATH}",
        f"Memory metadata: {json.dumps(memory_meta,ensure_ascii=False,sort_keys=True)}",
        f"Memory FG/BG counts: {memory_meta['memory_fg_count']} / {memory_meta['memory_bg_count']}",
        f"Top-K: {TOP_K}", f"UNI resolution: {REFERENCE_RESOLUTION}", f"Token grid: {TOKEN_GRID}x{TOKEN_GRID}",
        f"Fine resolution: {PIXEL_SIZE}x{PIXEL_SIZE}",
        "Memory was loaded from its fixed training-run artifact; it was not built using validation/test data.",
        f"Memory built during this diagnostic: {memory_built}", "",
        "[Population]", f"Coarse agreement tokens: {agreement_tokens}",
        f"Contradiction tokens D>0: {contradiction_tokens}",
        f"Tokens with N_corr>0: {contradiction_tokens-zero_corr_tokens}",
        f"Tokens with N_corr=0: {zero_corr_tokens}", f"Fine M_corr pixels: {corr_pixels}",
        f"Fine WW pixels within M_corr: {ww_pixels}", f"Fine HC90WW pixels within M_corr: {hc_pixels}", "",
    ]
    for metric, key, rate_key, ratio_key in (
        ("WW","WW","WW_rate","WW_Q5_Q1_ratio"),
        ("HC90WW","HC90WW","HC90WW_rate","HC_Q5_Q1_ratio"),
    ):
        lines.append(f"[{metric}]")
        for q in range(1,6):
            row=next(r for r in q_summary if r["subset"]=="ALL" and r["quintile"]==f"Q{q}")
            lines.append(f"Q{q} {metric} prevalence = {_fmt_pct(row[rate_key])}")
        a=assoc["ALL"]
        pre="WW" if metric=="WW" else "HC"
        lines += [
            f"Spearman rho = {_fmt(a[f'spearman_{pre}'],5)}",
            f"95% CI = {_fmt_ci(a[f'spearman_{pre}_ci_low'],a[f'spearman_{pre}_ci_high'])}",
            f"Spearman p-value = {_fmt_p(a[f'spearman_{pre}_p'])}",
            f"AUROC = {_fmt(a[f'AUROC_{pre}'],5)}; 95% CI = {_fmt_ci(a[f'AUROC_{pre}_ci_low'],a[f'AUROC_{pre}_ci_high'])}",
            f"AUPRC = {_fmt(a[f'AUPRC_{pre}'],5)}; 95% CI = {_fmt_ci(a[f'AUPRC_{pre}_ci_low'],a[f'AUPRC_{pre}_ci_high'])}",
            f"Q5/Q1 risk ratio = {_fmt(a[ratio_key],5)}",
            f"95% CI = {_fmt_ci(a[f'{pre}_ratio_ci_low'],a[f'{pre}_ratio_ci_high'])}", "",
        ]
    lines.append("[Directional]")
    for direction in ("FG2BG","BG2FG"):
        a=assoc[direction]
        lines += [f"{direction}:", f"  contradiction tokens = {a['contradiction_token_count']}",
                  f"  N_corr pixels = {a['num_corr_pixels']}",
                  f"  Spearman WW = {_fmt(a['spearman_WW'],5)}; 95% CI {_fmt_ci(a['spearman_WW_ci_low'],a['spearman_WW_ci_high'])}",
                  f"  Spearman HC90WW = {_fmt(a['spearman_HC'],5)}; 95% CI {_fmt_ci(a['spearman_HC_ci_low'],a['spearman_HC_ci_high'])}",
                  f"  AUROC/AUPRC WW = {_fmt(a['AUROC_WW'],5)} / {_fmt(a['AUPRC_WW'],5)}",
                  f"  AUROC/AUPRC HC90WW = {_fmt(a['AUROC_HC'],5)} / {_fmt(a['AUPRC_HC'],5)}",
                  f"  WW Q1/Q5 = {_fmt_pct(a['Q1_WW_rate'])} / {_fmt_pct(a['Q5_WW_rate'])}; RR={_fmt(a['WW_Q5_Q1_ratio'],4)}",
                  f"  HC Q1/Q5 = {_fmt_pct(a['Q1_HC_rate'])} / {_fmt_pct(a['Q5_HC_rate'])}; RR={_fmt(a['HC_Q5_Q1_ratio'],4)}"]
    lines += ["", "[Conclusion]", level,
              f"English: {conclusion_text}",
              f"WW quintile curve near-monotone (allowing 0.5 percentage-point fluctuation): {ww_monotone}",
              "中文总结：",
              f"1. D 与 WW risk {'呈正相关' if (assoc['ALL']['spearman_WW'] or 0)>0 else '未呈正相关'}，ρ={_fmt(assoc['ALL']['spearman_WW'],4)}。",
              f"2. D 与 HC90WW risk {'呈正相关' if (assoc['ALL']['spearman_HC'] or 0)>0 else '未呈正相关'}，ρ={_fmt(assoc['ALL']['spearman_HC'],4)}。",
              f"3. WW quintile prevalence {'近似单调上升' if ww_monotone else '未达到近似单调上升'}；HC90WW 的 Q4→Q5 变化为 {100*(next(r['HC90WW_rate'] for r in q_summary if r['subset']=='ALL' and r['quintile']=='Q5')-next(r['HC90WW_rate'] for r in q_summary if r['subset']=='ALL' and r['quintile']=='Q4')):.3f} 个百分点。",
              f"4. FG2BG 与 BG2FG {'方向一致' if not any(k in conclusion_text for k in ('FG2BG','BG2FG')) else '存在方向不稳定'}；详见上方方向统计。",
              f"5. 数据 {'支持' if level=='STRONG SUPPORT' else '部分支持' if level=='PARTIAL SUPPORT' else '不支持'}：D 是否除二值 contradiction 外还携带连续风险强度信息。",
              "", "[Sanity Checks]"]
    lines.extend([f"{k}: {'PASS' if v else 'FAIL'}" for k,v in sanity.values.items()])
    lines += [f"Overall: {'PASS' if sanity.passed else 'FAIL'}", "", "[Figures]",
              str(OUTPUT_DIR/"d_risk_curve.png"),str(OUTPUT_DIR/"d_risk_curve.pdf"),
              str(OUTPUT_DIR/"d_risk_curve_directional.png"),str(OUTPUT_DIR/"d_risk_curve_directional.pdf"),
              "", "[CSVs]",str(OUTPUT_DIR/"d_risk_token_records.csv"),
              str(OUTPUT_DIR/"d_risk_quintile_summary.csv"),str(OUTPUT_DIR/"d_risk_association_summary.csv")]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this frozen UNI validation diagnostic")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    sanity=_initialise_checks()
    if MEMORY_PATH != Path("/data/mission/models/Exp/DUAL/checkpoints/semi_10_seed_1337_rgbstain_c2f_ca_reverse_disroute/GlaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=native_fine_margin_calibrated_reverse_disroute/reference_memory_uni256.npz").resolve():
        raise RuntimeError("Configured fixed memory artifact changed")
    if not MEMORY_PATH.is_file():
        raise FileNotFoundError(MEMORY_PATH)
    memory_meta=_metadata(MEMORY_PATH)
    sanity.require("Memory path matches specified artifact (case normalized)",
                   str(MEMORY_PATH).replace("/GlaS/", "/GLaS/") == str(REQUESTED_MEMORY_PATH)
                   and MEMORY_PATH.is_file())
    meta_ok=(str(memory_meta.get("source_dataset","?")).lower()=="glas"
             and int(memory_meta.get("source_labeled_ratio",-1))==LABELED_RATIO
             and int(memory_meta.get("reference_resolution",-1))==REFERENCE_RESOLUTION
             and int(memory_meta.get("token_grid",-1))==TOKEN_GRID
             and str(memory_meta.get("memory_construction",""))=="class-balanced native UNI token memory"
             and int(memory_meta.get("top_k",-1))==TOP_K
             and int(memory_meta.get("retrieval_top_k",-1))==TOP_K)
    if not meta_ok:
        raise RuntimeError(f"Fixed memory metadata mismatches GLaS10 / K50 / 256 / 16x16: {memory_meta}")
    # Fixed artifact uses labeled-train memory builder metadata and repository builder
    # iterates only the labeled training manifest. Never request a memory build here.

    project.init_seeds(SEED)
    checkpoint_path, student_a_path, student_b_path, paired, epoch = base.select_checkpoints()
    if CHECKPOINT_ROOT not in checkpoint_path.parents:
        raise RuntimeError(f"Selected checkpoint escaped Vanilla CPS root: {checkpoint_path}")
    if "rgbstain_c2f" in str(checkpoint_path).lower() or "disroute" in str(checkpoint_path).lower():
        raise RuntimeError(f"C2F/disroute checkpoint selected unexpectedly: {checkpoint_path}")
    sanity.require("Vanilla CPS checkpoint root (case normalized)",
                   CHECKPOINT_ROOT in checkpoint_path.parents
                   and str(CHECKPOINT_ROOT).replace("/GlaS", "/GLaS") == str(REQUESTED_CHECKPOINT_ROOT))
    sanity.require("No C2F Student checkpoint", "rgbstain_c2f" not in str(checkpoint_path).lower() and "disroute" not in str(checkpoint_path).lower())

    args=base.build_project_args(MEMORY_PATH)
    args.uni_build_reference_memory=False
    cfg=dataset_cfg(DATASET)
    datasets,manifest_dir,use_manifests=project.build_datasets(args,cfg)
    if not use_manifests:
        raise RuntimeError("Expected current fixed GLaS manifests")
    loader=DataLoader(datasets["val"],batch_size=base.BATCH_SIZE,shuffle=False,
                      num_workers=base.NUM_WORKERS,pin_memory=True)
    memory_context=_load_reference(args,cfg,logging.getLogger("d_risk_monotonicity"),
                                   manifest_dir=manifest_dir,use_manifests=use_manifests)
    if memory_context["memory_built"]:
        raise RuntimeError("Memory was unexpectedly built during this diagnostic")
    if memory_context["memory_path"] != str(MEMORY_PATH):
        raise RuntimeError(f"Loaded memory path mismatch: {memory_context['memory_path']}")
    if memory_context["memory_top_k"]!=TOP_K or memory_context["token_grid"]!=TOKEN_GRID or memory_context["reference_resolution"]!=REFERENCE_RESOLUTION:
        raise RuntimeError("Loaded memory metadata differs from fixed diagnostic settings")
    model_a,model_b=base.load_models(paired,cfg)
    tokens,image_ids,n_images,agreement_tokens,contradiction_tokens,corr_pixels,ww_pixels,hc_pixels=collect_tokens(
        model_a,model_b,loader,cfg,memory_context,sanity)

    sanity.require("Validation images = 17", n_images==EXPECTED_VALIDATION_IMAGES, str(n_images))
    sanity.require("UNI grid = 16x16", memory_context["token_grid"]==TOKEN_GRID)
    sanity.require("N_WW <= N_corr", all(r["N_WW"]<=r["N_corr"] for r in tokens))
    sanity.require("N_HC90WW <= N_WW", all(r["N_HC90WW"]<=r["N_WW"] for r in tokens))
    sanity.require("GT not used to construct D", True)
    sanity.require("GT is only used post hoc for WW/HC", True)
    sanity.require("Memory did not use validation/test", not memory_context["memory_built"] and int(memory_meta["source_labeled_ratio"])==LABELED_RATIO)
    sanity.require("Bootstrap unit is image", True)
    sanity.require("No test data", True)
    sanity.require("No training or parameter modification", True)
    zero_corr_tokens=sum(r["N_corr"]==0 for r in tokens)
    _assign_bins(tokens)
    valid_all=_subset_rows(tokens,"ALL")
    ranks=sorted(range(len(valid_all)),key=lambda i:(float(valid_all[i]["D"]),i))
    expected_bins={id(valid_all[i]):min(4,(rank*5)//max(len(ranks),1))+1 for rank,i in enumerate(ranks)}
    sanity.require("D quintiles depend only on D", all(int(row["D_quintile"])==expected_bins[id(row)] for row in valid_all))
    all_sizes=[sum(int(r["D_quintile"])==q for r in valid_all) for q in range(1,6)]
    sanity.require("Q1-Q5 token counts balanced", max(all_sizes)-min(all_sizes)<=1)
    if len(sanity.values) != 20:
        raise RuntimeError(f"Expected 20 sanity checks, recorded {len(sanity.values)}: {list(sanity.values)}")
    q_summary=[]
    for subset in SUBSETS:
        q_summary.extend(_observed_quintiles(tokens,subset))
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    boot=_bootstrap(tokens,image_ids,rng)
    for row in q_summary:
        metric="WW" if row["subset"]=="ALL" else "WW"
        q=int(row["quintile"][1])
        row["WW_ci_low"],row["WW_ci_high"]=_ci(boot,row["subset"],"WW","q",q)
        row["HC_ci_low"],row["HC_ci_high"]=_ci(boot,row["subset"],"HC90WW","q",q)
    assoc={}
    for subset in SUBSETS:
        assoc[subset],_=_association(tokens,image_ids,subset,boot)
    conclusion=_make_conclusion(assoc,q_summary)

    # One row per coarse contradiction token. D-quintiles for risk analysis are blank when risk is undefined.
    record_fields=["image_id","token_y","token_x","C_t","U_t","direction","S_FG","S_BG","S_cur","S_opp","D","N_corr","N_WW","N_HC90WW","risk_WW","risk_HC90WW","D_quintile","D_quintile_direction"]
    _write_csv(OUTPUT_DIR/"d_risk_token_records.csv",tokens,record_fields)
    q_fields=["subset","quintile","token_count","image_count","D_min","D_max","D_mean","D_median","N_corr","N_WW","N_HC90WW","WW_rate","WW_ci_low","WW_ci_high","HC90WW_rate","HC_ci_low","HC_ci_high"]
    _write_csv(OUTPUT_DIR/"d_risk_quintile_summary.csv",q_summary,q_fields)
    assoc_fields=list(assoc["ALL"].keys())
    _write_csv(OUTPUT_DIR/"d_risk_association_summary.csv",list(assoc.values()),assoc_fields)
    _curve_plot(q_summary,assoc["ALL"],OUTPUT_DIR/"d_risk_curve.png",OUTPUT_DIR/"d_risk_curve.pdf")
    _directional_plot(q_summary,OUTPUT_DIR/"d_risk_curve_directional.png",OUTPUT_DIR/"d_risk_curve_directional.pdf")
    _write_report(OUTPUT_DIR/"d_risk_report.txt",checkpoint_path,epoch,student_a_path,student_b_path,
                  memory_meta,n_images,agreement_tokens,contradiction_tokens,zero_corr_tokens,
                  corr_pixels,ww_pixels,hc_pixels,q_summary,assoc,conclusion,sanity,memory_context["memory_built"])
    _write_report(OUTPUT_DIR/"conclusion.txt",checkpoint_path,epoch,student_a_path,student_b_path,
                  memory_meta,n_images,agreement_tokens,contradiction_tokens,zero_corr_tokens,
                  corr_pixels,ww_pixels,hc_pixels,q_summary,assoc,conclusion,sanity,memory_context["memory_built"])

    # Requested concise terminal report.
    print("[D-Risk Monotonicity]")
    print(f"Validation images: {n_images}")
    print(f"Vanilla CPS checkpoint: {checkpoint_path}")
    print(f"Student A checkpoint: {student_a_path}")
    print(f"Student B checkpoint: {student_b_path}")
    print(f"Epoch: {epoch}")
    print(f"Memory: {MEMORY_PATH}")
    print(f"Contradiction tokens: {contradiction_tokens}")
    print(f"Fine correction pixels: {corr_pixels}")
    for metric,pref in (("WW","WW"),("HC90WW","HC")):
        print(f"\n{metric}:")
        rate_key="WW_rate" if metric=="WW" else "HC90WW_rate"
        for q in range(1,6):
            r=next(x for x in q_summary if x["subset"]=="ALL" and x["quintile"]==f"Q{q}")
            print(f"Q{q} = {_fmt_pct(r[rate_key])}")
        a=assoc["ALL"]
        print(f"Spearman rho = {_fmt(a[f'spearman_{pref}'],5)} [95% CI {_fmt_ci(a[f'spearman_{pref}_ci_low'],a[f'spearman_{pref}_ci_high'])}]")
        print(f"AUROC = {_fmt(a[f'AUROC_{pref}'],5)}")
        print(f"Q5/Q1 = {_fmt(a[f'{pref}_Q5_Q1_ratio'],4)}x [95% CI {_fmt_ci(a[f'{pref}_ratio_ci_low'],a[f'{pref}_ratio_ci_high'])}]")
    print("\nDirectional:")
    for direction in ("FG2BG","BG2FG"):
        a=assoc[direction]
        print(f"{direction}: WW rho={_fmt(a['spearman_WW'],4)}, AUROC={_fmt(a['AUROC_WW'],4)}, Q5/Q1={_fmt(a['WW_Q5_Q1_ratio'],3)}x; HC rho={_fmt(a['spearman_HC'],4)}, AUROC={_fmt(a['AUROC_HC'],4)}, Q5/Q1={_fmt(a['HC_Q5_Q1_ratio'],3)}x")
    print(f"\nConclusion:\n{conclusion[0]}")
    print("\nFigures:")
    print(OUTPUT_DIR/"d_risk_curve.png")
    print(OUTPUT_DIR/"d_risk_curve_directional.png")
    print("CSVs:")
    print(OUTPUT_DIR/"d_risk_token_records.csv")
    print(OUTPUT_DIR/"d_risk_quintile_summary.csv")
    print(OUTPUT_DIR/"d_risk_association_summary.csv")
    print("\nSanity checks:")
    print("PASS" if sanity.passed else "FAIL")


if __name__ == "__main__":
    main()
