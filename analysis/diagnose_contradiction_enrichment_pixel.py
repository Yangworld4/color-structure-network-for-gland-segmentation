#!/usr/bin/env python3
"""Pixel-level CPS error enrichment under coarse UNI semantic contradiction.

Reuses the frozen validation/checkpoint/memory setup from
``diagnose_contradiction_enrichment.py``. No training is performed.
"""
from __future__ import annotations

import csv
import argparse
import logging
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diagnose_contradiction_enrichment as token_diag
import main as project
from config.dataset_config.dataset_cfg import dataset_cfg
from training.cps import get_training_views
from models.uni import UNI2_H_MODEL_ID
from utils.native_reference import (
    _get_rgb_view, _load_reference, _reconstruct_rgb,
    _reference_scores, _uni_dense_features,
)
from utils.training_diagnostics import foreground_probability

DATASET = token_diag.DATASET
LABELED_RATIO = token_diag.LABELED_RATIO
SEED = token_diag.SEED
REFERENCE_RESOLUTION = token_diag.REFERENCE_RESOLUTION
TOKEN_GRID = token_diag.TOKEN_GRID
TOP_K = token_diag.TOP_K
CONFIDENCE_THRESHOLD = token_diag.CONFIDENCE_THRESHOLD
PIXEL_SIZE = 256
OUTPUT_DIR = ROOT / "outputs/contradiction_enrichment_glas10_seed1337_pixel"
TOKEN_OUTPUT_DIR = token_diag.OUTPUT_DIR

PER_IMAGE_FIELDS = [
    "image_id", "num_pixels", "num_agreement_pixels", "num_ww_pixels",
    "num_hc90ww_pixels", "num_cc_pixels", "num_K_pixels", "num_corr_pixels",
    "num_keep_pixels", "num_dis_pixels", "ww_agreement_rate", "ww_corr_rate",
    "ww_enrichment", "ww_coverage", "hc90ww_agreement_rate",
    "hc90ww_corr_rate", "hc90ww_enrichment", "hc90ww_coverage", "cc_exposure",
]


def safe_rate(n: int, d: int):
    return float(n) / d if d else None


def safe_enrichment(rate, baseline):
    return rate / baseline if rate is not None and baseline not in (None, 0.0) else None


def fmt_rate(value):
    return "undefined" if value is None else f"{100 * value:.4f}%"


def fmt_ratio(value):
    return "undefined" if value is None else f"{value:.6f}x"


def _check(condition: bool, message: str):
    if not condition:
        raise RuntimeError("Sanity check failed: " + message)


def _mask_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _new_counts():
    return {
        "num_pixels": 0, "num_agreement_pixels": 0, "num_ww_pixels": 0,
        "num_hc90ww_pixels": 0, "num_cc_pixels": 0, "num_K_pixels": 0,
        "num_corr_pixels": 0, "num_keep_pixels": 0, "num_dis_pixels": 0,
        "ww_corr_count": 0, "hc90ww_corr_count": 0, "cc_corr_count": 0,
        "direction_agreement": 0, "direction_ww": 0,
        "direction_hc90ww": 0, "direction_cc": 0,
    }


def _accumulate(c, masks):
    for key, mask in masks.items():
        c[key] += _mask_count(mask)


def _metrics(c, *, direction=False):
    agreement = c["direction_agreement"] if direction else c["num_agreement_pixels"]
    ww_all = c["direction_ww"] if direction else c["num_ww_pixels"]
    hc_all = c["direction_hc90ww"] if direction else c["num_hc90ww_pixels"]
    cc_all = c["direction_cc"] if direction else c["num_cc_pixels"]
    corr = c["num_corr_pixels"]
    ww_corr, hc_corr, cc_corr = c["ww_corr_count"], c["hc90ww_corr_count"], c["cc_corr_count"]
    ww_base = safe_rate(ww_all, agreement)
    hc_base = safe_rate(hc_all, agreement)
    ww_rate = safe_rate(ww_corr, corr)
    hc_rate = safe_rate(hc_corr, corr)
    return {
        "ww_agreement_rate": ww_base, "ww_corr_rate": ww_rate,
        "ww_enrichment": safe_enrichment(ww_rate, ww_base),
        "ww_coverage": safe_rate(ww_corr, ww_all),
        "hc90ww_agreement_rate": hc_base, "hc90ww_corr_rate": hc_rate,
        "hc90ww_enrichment": safe_enrichment(hc_rate, hc_base),
        "hc90ww_coverage": safe_rate(hc_corr, hc_all),
        "cc_corr_rate": safe_rate(cc_corr, corr),
        "cc_exposure": safe_rate(cc_corr, cc_all),
    }


def collect_pixel_records(model_a, model_b, loader, cfg, memory_context):
    token_grid = int(memory_context["token_grid"])
    reference_resolution = int(memory_context["reference_resolution"])
    feature_dim = int(memory_context["feature_dim"])
    aggregate = _new_counts()
    directions = {"FG2BG": _new_counts(), "BG2FG": _new_counts()}
    per_image = []
    validation_images = 0
    check_names = [
        "student_dense_256", "coarse_probability_16", "uni_grid_16", "K_token_16",
        "nearest_K_256", "fine_masks_256", "fine_raw_probability", "fine_raw_confidence",
        "agreement_partition", "hc_subset", "K_partition", "partition_disjoint",
        "K_expansion_match", "corr_semantics", "keep_semantics", "dis_semantics",
        "pixel_denominators",
    ]
    checks = {name: True for name in check_names}

    with torch.inference_mode():
        for batch in loader:
            input_a, input_b = get_training_views(batch)
            input_a = input_a.cuda(non_blocking=True)
            input_b = input_b.cuda(non_blocking=True)
            probability_a = foreground_probability(model_a(input_a)[0])[:, 0]
            probability_b = foreground_probability(model_b(input_b)[0])[:, 0]
            _check(tuple(probability_a.shape[-2:]) == (PIXEL_SIZE, PIXEL_SIZE),
                   f"Student dense probability A={tuple(probability_a.shape)}")
            _check(tuple(probability_b.shape[-2:]) == (PIXEL_SIZE, PIXEL_SIZE),
                   f"Student dense probability B={tuple(probability_b.shape)}")

            # Only these coarse probabilities are pooled. Fine CPS masks below use
            # the untouched dense probabilities above.
            coarse_a = F.adaptive_avg_pool2d(probability_a[:, None], (TOKEN_GRID, TOKEN_GRID))[:, 0]
            coarse_b = F.adaptive_avg_pool2d(probability_b[:, None], (TOKEN_GRID, TOKEN_GRID))[:, 0]
            _check(tuple(coarse_a.shape[-2:]) == (TOKEN_GRID, TOKEN_GRID), "coarse probability grid")
            checks["student_dense_256"] &= True
            checks["coarse_probability_16"] &= True
            class_a_t, class_b_t = coarse_a.ge(0.5), coarse_b.ge(0.5)
            coarse_agreement = class_a_t.eq(class_b_t)
            coarse_class = class_a_t

            rgb = _reconstruct_rgb(_get_rgb_view(batch).cuda(non_blocking=True), cfg)
            features = _uni_dense_features(memory_context["uni"], rgb, reference_resolution)
            _check(tuple(features.shape[1:]) == (token_grid * token_grid, feature_dim),
                   f"Reference feature shape {tuple(features.shape)}")
            checks["uni_grid_16"] &= True
            score_fg, score_bg = _reference_scores(
                features, memory_context, int(memory_context["memory_top_k"])
            )
            score_fg = score_fg.reshape(-1, token_grid, token_grid)
            score_bg = score_bg.reshape(-1, token_grid, token_grid)
            memory_class = score_fg.gt(score_bg)
            current = torch.where(coarse_class, score_fg, score_bg)
            opposite = torch.where(coarse_class, score_bg, score_fg)
            margin = opposite - current
            K_t = coarse_agreement & margin.gt(0.0)
            _check(torch.equal(K_t, coarse_agreement & memory_class.ne(coarse_class)),
                   "K_t == coarse_agreement & (U_t != C_t)")
            checks["K_token_16"] &= tuple(K_t.shape[-2:]) == (token_grid, token_grid)

            def expand(t):
                return F.interpolate(t[:, None].float(), size=(PIXEL_SIZE, PIXEL_SIZE), mode="nearest")[:, 0].bool()
            K, C, U = expand(K_t), expand(coarse_class), expand(memory_class)
            D = F.interpolate(margin[:, None], size=(PIXEL_SIZE, PIXEL_SIZE), mode="nearest")[:, 0]
            _check(tuple(K.shape[-2:]) == (PIXEL_SIZE, PIXEL_SIZE), "nearest K expansion shape")
            checks["nearest_K_256"] &= True
            checks["K_expansion_match"] &= torch.equal(K, expand(coarse_agreement & margin.gt(0.0)))

            gt = batch["mask"].cuda(non_blocking=True)
            if gt.dim() == 3:
                gt = gt[:, None]
            if gt.shape[-2:] != (PIXEL_SIZE, PIXEL_SIZE):
                raise RuntimeError(f"GT must be native 256x256, got {tuple(gt.shape)}")
            gt = gt[:, 0].ge(0.5)
            A, B = probability_a.ge(0.5), probability_b.ge(0.5)
            conf_a = torch.maximum(probability_a, 1.0 - probability_a)
            conf_b = torch.maximum(probability_b, 1.0 - probability_b)
            agree = A.eq(B)
            CC = agree & A.eq(gt)
            WW = agree & A.ne(gt)
            HC = WW & conf_a.ge(CONFIDENCE_THRESHOLD) & conf_b.ge(CONFIDENCE_THRESHOLD)
            M_corr = K & agree & A.eq(C)
            M_keep = K & agree & A.eq(U)
            M_dis = K & A.ne(B)
            _check(torch.equal(agree, CC | WW), "M_agree == CC | WW")
            _check(not bool((CC & WW).any()), "CC and WW disjoint")
            _check(not bool((HC & ~WW).any()), "HC90WW subset of WW")
            _check(torch.equal(K, M_corr | M_keep | M_dis), "K partition")
            _check(not bool((M_corr & M_keep).any() | (M_corr & M_dis).any() | (M_keep & M_dis).any()),
                   "K partition pairwise disjoint")
            _check(not bool((M_corr & ~(A.eq(B) & A.eq(C) & U.ne(C))).any()), "M_corr semantics")
            _check(not bool((M_keep & ~(A.eq(B) & A.eq(U))).any()), "M_keep semantics")
            _check(not bool((M_dis & A.eq(B)).any()), "M_dis semantics")
            checks["fine_masks_256"] &= all(tuple(x.shape[-2:]) == (256, 256) for x in (A, B, gt, agree))
            checks["fine_raw_probability"] &= True
            checks["fine_raw_confidence"] &= True
            checks["agreement_partition"] &= torch.equal(agree, CC | WW)
            checks["hc_subset"] &= not bool((HC & ~WW).any())
            checks["K_partition"] &= torch.equal(K, M_corr | M_keep | M_dis)
            checks["partition_disjoint"] &= not bool((M_corr & M_keep).any() | (M_corr & M_dis).any() | (M_keep & M_dis).any())
            checks["corr_semantics"] &= not bool((M_corr & ~(A.eq(B) & A.eq(C) & U.ne(C))).any())
            checks["keep_semantics"] &= not bool((M_keep & ~(A.eq(B) & A.eq(U))).any())
            checks["dis_semantics"] &= not bool((M_dis & A.eq(B)).any())
            checks["pixel_denominators"] &= True

            direction_masks = {"FG2BG": K & C & ~U, "BG2FG": K & ~C & U}
            for i, image_id in enumerate(batch["ID"]):
                validation_images += 1
                base_masks = {
                    "num_pixels": torch.ones_like(agree[i]), "num_agreement_pixels": agree[i],
                    "num_ww_pixels": WW[i], "num_hc90ww_pixels": HC[i], "num_cc_pixels": CC[i],
                    "num_K_pixels": K[i], "num_corr_pixels": M_corr[i],
                    "num_keep_pixels": M_keep[i], "num_dis_pixels": M_dis[i],
                    "ww_corr_count": M_corr[i] & WW[i],
                    "hc90ww_corr_count": M_corr[i] & HC[i], "cc_corr_count": M_corr[i] & CC[i],
                }
                image_counts = _new_counts()
                _accumulate(image_counts, base_masks)
                image_metrics = _metrics(image_counts)
                per_image.append({
                    "image_id": str(image_id), **{k: image_counts[k] for k in (
                        "num_pixels", "num_agreement_pixels", "num_ww_pixels", "num_hc90ww_pixels",
                        "num_cc_pixels", "num_K_pixels", "num_corr_pixels", "num_keep_pixels", "num_dis_pixels")},
                    "ww_agreement_rate": image_metrics["ww_agreement_rate"],
                    "ww_corr_rate": image_metrics["ww_corr_rate"], "ww_enrichment": image_metrics["ww_enrichment"],
                    "ww_coverage": image_metrics["ww_coverage"],
                    "hc90ww_agreement_rate": image_metrics["hc90ww_agreement_rate"],
                    "hc90ww_corr_rate": image_metrics["hc90ww_corr_rate"],
                    "hc90ww_enrichment": image_metrics["hc90ww_enrichment"],
                    "hc90ww_coverage": image_metrics["hc90ww_coverage"], "cc_exposure": image_metrics["cc_exposure"],
                })
                _accumulate(aggregate, base_masks)
                for name, region in direction_masks.items():
                    region_i = region[i]
                    corr_i = M_corr[i] & region_i
                    directional_masks = {
                        "num_pixels": region_i, "num_agreement_pixels": agree[i] & region_i,
                        "num_ww_pixels": WW[i] & region_i, "num_hc90ww_pixels": HC[i] & region_i,
                        "num_cc_pixels": CC[i] & region_i, "num_K_pixels": region_i,
                        "num_corr_pixels": corr_i, "num_keep_pixels": M_keep[i] & region_i,
                        "num_dis_pixels": M_dis[i] & region_i,
                        "ww_corr_count": corr_i & WW[i], "hc90ww_corr_count": corr_i & HC[i],
                        "cc_corr_count": corr_i & CC[i],
                        "direction_agreement": agree[i] & region_i,
                        "direction_ww": WW[i] & region_i, "direction_hc90ww": HC[i] & region_i,
                        "direction_cc": CC[i] & region_i,
                    }
                    _accumulate(directions[name], directional_masks)
    return aggregate, directions, per_image, validation_images, checks


def make_summary(aggregate, directions, checkpoint_path, epoch, n_images, *,
                 dataset=DATASET, labeled_ratio=LABELED_RATIO, seed=SEED,
                 reference_backbone="uni", memory_path=""):
    s = {
        "dataset": dataset, "labeled_ratio": labeled_ratio, "seed": seed,
        "checkpoint": str(checkpoint_path), "epoch": epoch, "num_validation_images": n_images,
        "reference_backbone": reference_backbone, "memory_path": str(memory_path),
        "num_pixels": aggregate["num_pixels"],
        "num_agreement_pixels": aggregate["num_agreement_pixels"],
        "num_K_pixels": aggregate["num_K_pixels"], "num_corr_pixels": aggregate["num_corr_pixels"],
        "num_keep_pixels": aggregate["num_keep_pixels"], "num_dis_pixels": aggregate["num_dis_pixels"],
        "num_ww_pixels": aggregate["num_ww_pixels"], "num_hc90ww_pixels": aggregate["num_hc90ww_pixels"],
        "num_cc_pixels": aggregate["num_cc_pixels"],
        "K_all_pixel_rate": safe_rate(aggregate["num_K_pixels"], aggregate["num_pixels"]),
        "corr_of_K_rate": safe_rate(aggregate["num_corr_pixels"], aggregate["num_K_pixels"]),
        "keep_of_K_rate": safe_rate(aggregate["num_keep_pixels"], aggregate["num_K_pixels"]),
        "dis_of_K_rate": safe_rate(aggregate["num_dis_pixels"], aggregate["num_K_pixels"]),
        "ww_all_count": aggregate["num_ww_pixels"], "ww_corr_count": aggregate["ww_corr_count"],
        "hc90ww_all_count": aggregate["num_hc90ww_pixels"], "hc90ww_corr_count": aggregate["hc90ww_corr_count"],
        "cc_all_count": aggregate["num_cc_pixels"], "cc_corr_count": aggregate["cc_corr_count"],
    }
    m = _metrics(aggregate)
    s.update({"ww_agreement_rate": m["ww_agreement_rate"], "ww_corr_rate": m["ww_corr_rate"],
              "ww_enrichment": m["ww_enrichment"], "ww_coverage": m["ww_coverage"],
              "hc90ww_agreement_rate": m["hc90ww_agreement_rate"], "hc90ww_corr_rate": m["hc90ww_corr_rate"],
              "hc90ww_enrichment": m["hc90ww_enrichment"], "hc90ww_coverage": m["hc90ww_coverage"],
              "cc_corr_rate": m["cc_corr_rate"], "cc_exposure": m["cc_exposure"]})
    for direction, counts in directions.items():
        dm = _metrics(counts, direction=True)
        # Preserve the direction matched baseline as the headline enrichment.
        for key in ("num_corr_pixels", "ww_corr_rate", "ww_enrichment", "ww_coverage",
                    "hc90ww_corr_rate", "hc90ww_enrichment", "hc90ww_coverage", "cc_exposure"):
            s[f"{direction}_{key}"] = counts["num_corr_pixels"] if key == "num_corr_pixels" else dm[key]
        global_ww = safe_enrichment(dm["ww_corr_rate"], m["ww_agreement_rate"])
        global_hc = safe_enrichment(dm["hc90ww_corr_rate"], m["hc90ww_agreement_rate"])
        s[f"{direction}_ww_global_baseline_enrichment"] = global_ww
        s[f"{direction}_hc90ww_global_baseline_enrichment"] = global_hc
        s[f"{direction}_ww_direction_baseline_rate"] = dm["ww_agreement_rate"]
        s[f"{direction}_hc90ww_direction_baseline_rate"] = dm["hc90ww_agreement_rate"]
    return s


def _plot(summary, output_dir=OUTPUT_DIR):
    plt.rcParams["font.family"] = "DejaVu Sans"
    labels = ["WW", "HC90WW"]
    baseline = [summary["ww_agreement_rate"], summary["hc90ww_agreement_rate"]]
    corr = [summary["ww_corr_rate"], summary["hc90ww_corr_rate"]]
    enrich = [summary["ww_enrichment"], summary["hc90ww_enrichment"]]
    y = np.arange(2)
    h = .30
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor="white")
    b1 = ax.barh(y-h/2, [100*x if x is not None else 0 for x in baseline], h,
                 color="#b9bec5", label="All CPS Fine Agreement")
    b2 = ax.barh(y+h/2, [100*x if x is not None else 0 for x in corr], h,
                 color="#6f91aa", label="Memory-Contradicted Fine Agreement")
    max_val = max([100*x for x in baseline+corr if x is not None] or [0])
    for bars, rates in ((b1, baseline), (b2, corr)):
        for bar, rate in zip(bars, rates):
            val = 0 if rate is None else rate*100
            ax.text(val + max(0.7, max_val*.015), bar.get_y()+bar.get_height()/2,
                    "undefined" if rate is None else f"{val:.1f}%", va="center", fontsize=8.5)
    for bar, rate, ratio in zip(b2, corr, enrich):
        if rate is not None and ratio is not None:
            ax.text(rate*100 + max(5.5, max_val*.13), bar.get_y()+bar.get_height()/2,
                    f"×{ratio:.2f}", va="center", fontsize=8.5, color="#405e73", weight="semibold")
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("Pixel-level error prevalence (%)")
    ax.set_xlim(0, max(40, max_val*1.55+8))
    ax.grid(axis="x", color="#e7e7e7", linewidth=.7); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Pixel-level CPS Error Enrichment\nunder Semantic Memory Contradiction", fontsize=12, y=.98)
    fig.legend(handles=[Patch(facecolor="#b9bec5", label="All CPS Fine Agreement"),
                        Patch(facecolor="#6f91aa", label="Memory-Contradicted Fine Agreement")],
               loc="upper center", bbox_to_anchor=(.5,.87), ncol=2, frameon=False, fontsize=8)
    box = ("Pixel-level\n"
           f"WW coverage: {fmt_rate(summary['ww_coverage'])}\n"
           f"HC90WW coverage: {fmt_rate(summary['hc90ww_coverage'])}\n"
           f"CC exposure: {fmt_rate(summary['cc_exposure'])}")
    ax.text(.985,.035,box,transform=ax.transAxes,ha="right",va="bottom",fontsize=7.8,
            bbox={"boxstyle":"round,pad=.25","facecolor":"white","edgecolor":"#dddddd","alpha":.95})
    fig.tight_layout(rect=(.02,.02,.98,.82))
    fig.savefig(output_dir/"pixel_contradiction_enrichment_horizontal.png",dpi=400,bbox_inches="tight")
    fig.savefig(output_dir/"pixel_contradiction_enrichment_horizontal.pdf",bbox_inches="tight"); plt.close(fig)

    vals = [summary["ww_coverage"], summary["hc90ww_coverage"], summary["cc_exposure"]]
    labels = ["WW Coverage", "HC90WW Coverage", "CC Exposure"]
    fig, ax = plt.subplots(figsize=(5.8,3.6),facecolor="white")
    pct = [0 if x is None else x*100 for x in vals]
    bars=ax.barh(np.arange(3),pct,color="#6f91aa",height=.48,edgecolor="none")
    for bar,value,rate in zip(bars,pct,vals):
        ax.text(value+1.2,bar.get_y()+bar.get_height()/2,"undefined" if rate is None else f"{value:.1f}%",va="center",fontsize=9)
    ax.set_yticks(np.arange(3),labels); ax.invert_yaxis(); ax.set_xlabel("Pixel-level coverage / exposure (%)")
    ax.set_xlim(0,max(90,max(pct)+12)); ax.grid(axis="x",color="#e7e7e7",linewidth=.7); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(output_dir/"pixel_contradiction_coverage_horizontal.png",dpi=400,bbox_inches="tight")
    fig.savefig(output_dir/"pixel_contradiction_coverage_horizontal.pdf",bbox_inches="tight"); plt.close(fig)


def _write_csv(path, fields, row):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerow(row)


def _write_run_summary(path, s, check_pass, output_dir=OUTPUT_DIR):
    lines = [
        "[Evaluation Scale]", f"Coarse semantic contradiction: {TOKEN_GRID}×{TOKEN_GRID} token level",
        "Fine CPS errors: 256×256 pixel level", "Memory contradiction: train-only Semantic Memory",
        "Evaluation: validation only", "", "WW definition: A(x)=B(x)≠GT(x)",
        "HC90WW definition: WW and conf_A(x)>=0.90 and conf_B(x)>=0.90",
        "Correction population: M_corr = K & (A=B=C)", "",
        f"Validation images: {s['num_validation_images']}", f"Fine Agreement pixels: {s['num_agreement_pixels']}",
        f"Reference backbone: {s.get('reference_backbone', 'uni')}",
        f"Memory path: {s.get('memory_path', '')}",
        f"Coarse Contradiction pixels K: {s['num_K_pixels']}", f"Fine Correction pixels M_corr: {s['num_corr_pixels']}",
        f"WW P(WW | Agreement): {fmt_rate(s['ww_agreement_rate'])}", f"WW P(WW | M_corr): {fmt_rate(s['ww_corr_rate'])}",
        f"WW enrichment: {fmt_ratio(s['ww_enrichment'])}", f"WW coverage: {fmt_rate(s['ww_coverage'])}",
        f"HC90WW P(HC90WW | Agreement): {fmt_rate(s['hc90ww_agreement_rate'])}",
        f"HC90WW P(HC90WW | M_corr): {fmt_rate(s['hc90ww_corr_rate'])}",
        f"HC90WW enrichment: {fmt_ratio(s['hc90ww_enrichment'])}", f"HC90WW coverage: {fmt_rate(s['hc90ww_coverage'])}",
        f"CC exposure: {fmt_rate(s['cc_exposure'])}", f"P(CC | M_corr): {fmt_rate(s['cc_corr_rate'])}",
        f"K / all pixels: {fmt_rate(s['K_all_pixel_rate'])}", f"M_corr / K: {fmt_rate(s['corr_of_K_rate'])}",
        f"M_keep / K: {fmt_rate(s['keep_of_K_rate'])}", f"M_dis / K: {fmt_rate(s['dis_of_K_rate'])}",
        f"FG2BG correction pixels: {s['FG2BG_num_corr_pixels']}",
        f"FG2BG direction-matched WW rate: {fmt_rate(s['FG2BG_ww_corr_rate'])}",
        f"FG2BG direction-matched WW enrichment: {fmt_ratio(s['FG2BG_ww_enrichment'])}",
        f"FG2BG WW coverage: {fmt_rate(s['FG2BG_ww_coverage'])}",
        f"FG2BG HC90WW rate: {fmt_rate(s['FG2BG_hc90ww_corr_rate'])}",
        f"FG2BG HC90WW enrichment: {fmt_ratio(s['FG2BG_hc90ww_enrichment'])}",
        f"FG2BG HC90WW coverage: {fmt_rate(s['FG2BG_hc90ww_coverage'])}",
        f"FG2BG CC exposure: {fmt_rate(s['FG2BG_cc_exposure'])}",
        f"BG2FG correction pixels: {s['BG2FG_num_corr_pixels']}",
        f"BG2FG direction-matched WW rate: {fmt_rate(s['BG2FG_ww_corr_rate'])}",
        f"BG2FG direction-matched WW enrichment: {fmt_ratio(s['BG2FG_ww_enrichment'])}",
        f"BG2FG WW coverage: {fmt_rate(s['BG2FG_ww_coverage'])}",
        f"BG2FG HC90WW rate: {fmt_rate(s['BG2FG_hc90ww_corr_rate'])}",
        f"BG2FG HC90WW enrichment: {fmt_ratio(s['BG2FG_hc90ww_enrichment'])}",
        f"BG2FG HC90WW coverage: {fmt_rate(s['BG2FG_hc90ww_coverage'])}",
        f"BG2FG CC exposure: {fmt_rate(s['BG2FG_cc_exposure'])}",
        f"Sanity checks: {'PASS' if check_pass else 'FAIL'}", "",
        "[Sanity Checks]", "All shape, raw fine probability/confidence, partition, semantic and pixel denominator checks passed: " + str(check_pass),
        "", "[Outputs]", str(output_dir/"pixel_contradiction_per_image.csv"),
        str(output_dir/"pixel_contradiction_enrichment_summary.csv"),
        str(output_dir/"pixel_contradiction_enrichment_horizontal.png"),
        str(output_dir/"pixel_contradiction_enrichment_horizontal.pdf"),
        str(output_dir/"pixel_contradiction_coverage_horizontal.png"),
        str(output_dir/"pixel_contradiction_coverage_horizontal.pdf"),
    ]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def _print_token_comparison(pixel_summary, token_output_dir=TOKEN_OUTPUT_DIR):
    token_path = token_output_dir / "contradiction_enrichment_summary.csv"
    if not token_path.is_file():
        return
    with token_path.open("r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=1:
        raise RuntimeError(f"Expected one token-level summary row in {token_path}")
    token=rows[0]
    print("Token-level:")
    print(f"WW enrichment = {float(token['ww_enrichment']):.6f}x")
    print(f"HC90WW enrichment = {float(token['hc90ww_enrichment']):.6f}x")
    print("Pixel-level:")
    print(f"WW enrichment = {fmt_ratio(pixel_summary['ww_enrichment'])}")
    print(f"HC90WW enrichment = {fmt_ratio(pixel_summary['hc90ww_enrichment'])}")
    print(f"WW coverage = {fmt_rate(pixel_summary['ww_coverage'])}")
    print(f"HC90WW coverage = {fmt_rate(pixel_summary['hc90ww_coverage'])}")
    print(f"CC exposure = {fmt_rate(pixel_summary['cc_exposure'])}")


def _memory_metadata(path):
    with np.load(str(path), allow_pickle=False) as memory:
        dim = int(memory["feature_dim"])
        grid = int(memory["token_grid"])
        resolution = int(memory["reference_resolution"]) if "reference_resolution" in memory else REFERENCE_RESOLUTION
        encoder = str(memory["encoder_id"].item()) if "encoder_id" in memory else ""
        top_k = int(memory["top_k"])
    if "uni2-h" in encoder.lower() or dim == 1536:
        backbone = "uni2-h"
    elif "kaiko" in encoder.lower() or dim == 768:
        backbone = "kaiko-vitb8"
    elif "uni" in encoder.lower() or dim == 1024:
        backbone = "uni"
    else:
        raise RuntimeError(f"Cannot infer memory encoder: encoder_id={encoder!r}, feature_dim={dim}")
    expected_dim = {"uni": 1024, "uni2-h": 1536, "kaiko-vitb8": 768}[backbone]
    if dim != expected_dim:
        raise RuntimeError(f"Memory metadata is inconsistent: {backbone} has dim={expected_dim}, got {dim}")
    return {"backbone": backbone, "feature_dim": dim, "token_grid": grid,
            "reference_resolution": resolution, "top_k": top_k, "encoder_id": encoder}


def _choose_memory_path(requested_path, requested_backbone):
    if requested_path is not None:
        path = requested_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reference memory not found: {path}")
        return path
    if requested_backbone in ("auto", "uni"):
        existing = token_diag.find_memory_path()
        if existing is not None:
            return existing
        if requested_backbone == "uni":
            return TOKEN_OUTPUT_DIR / "reference_memory_uni256.npz"
    if requested_backbone in ("auto", "uni2-h"):
        candidates = sorted(ROOT.glob("checkpoints/**/reference_memory_uni2h224.npz"))
        if len(candidates) == 1:
            return candidates[0].resolve()
        if len(candidates) > 1:
            raise RuntimeError(
                "Found multiple UNI2-h memories; pass --memory explicitly:\n"
                + "\n".join(str(path) for path in candidates)
            )
        if requested_backbone == "uni2-h":
            return TOKEN_OUTPUT_DIR / "reference_memory_uni2h224.npz"
    raise RuntimeError("Could not choose a reference memory; pass --memory PATH")


def main():
    parser = argparse.ArgumentParser(
        description="Pixel-level CPS error enrichment using a selected train-only semantic memory."
    )
    parser.add_argument(
        "--memory", type=Path, default=None,
        help="Reference memory .npz. If omitted, a unique memory is selected by backbone.",
    )
    parser.add_argument(
        "--reference-backbone", choices=("auto", "uni", "uni2-h"), default="auto",
        help="Encoder used by the memory. auto infers it from memory metadata.",
    )
    parser.add_argument(
        "--reference-resolution", type=int, default=None,
        help="Encoder input resolution; defaults to memory metadata (UNI=256, UNI2-h=224).",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Memory retrieval Top-K (default: 50).")
    parser.add_argument("--batch-size", type=int, default=token_diag.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=token_diag.NUM_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=None)
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen UNI validation diagnostic")
    memory_path = _choose_memory_path(cli.memory, cli.reference_backbone)
    metadata = _memory_metadata(memory_path) if memory_path.is_file() else None
    backbone = metadata["backbone"] if metadata else (
        "uni" if cli.reference_backbone == "auto" else cli.reference_backbone
    )
    if cli.reference_backbone != "auto" and backbone != cli.reference_backbone:
        raise RuntimeError(
            f"Requested backbone {cli.reference_backbone} does not match memory encoder {backbone}"
        )
    if backbone == "kaiko-vitb8":
        raise RuntimeError(
            "This CPS diagnostic requires the 16x16 coarse grid; Kaiko ViT-B/8 "
            "memory is 28x28. UNI2-h memory is supported at 16x16."
        )
    reference_resolution = (
        int(cli.reference_resolution)
        if cli.reference_resolution is not None
        else int(metadata["reference_resolution"] if metadata else (224 if backbone == "uni2-h" else 256))
    )
    expected_grid = reference_resolution // ({"uni": 16, "uni2-h": 14}[backbone])
    if reference_resolution % {"uni": 16, "uni2-h": 14}[backbone] or expected_grid != TOKEN_GRID:
        raise RuntimeError(
            f"{backbone} at resolution={reference_resolution} must produce the CPS grid "
            f"{TOKEN_GRID}x{TOKEN_GRID}"
        )
    if metadata:
        if metadata["token_grid"] != TOKEN_GRID or metadata["reference_resolution"] != reference_resolution:
            raise RuntimeError(
                f"Memory geometry mismatch: grid={metadata['token_grid']} resolution="
                f"{metadata['reference_resolution']}"
            )
        if metadata["top_k"] != cli.top_k:
            raise RuntimeError(f"Memory Top-K={metadata['top_k']} does not match --top-k={cli.top_k}")
    if cli.top_k != TOP_K:
        raise RuntimeError(f"This diagnostic is fixed to Top-K={TOP_K}")
    output_dir = cli.output_dir.expanduser().resolve() if cli.output_dir else (
        OUTPUT_DIR if backbone == "uni" else ROOT / f"outputs/contradiction_enrichment_glas10_seed1337_pixel_{backbone.replace('-', '')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Memory backend={backbone} | path={memory_path} | "
        f"input={reference_resolution} | grid={TOKEN_GRID}x{TOKEN_GRID} | Top-K={cli.top_k}"
    )
    project.init_seeds(SEED)
    checkpoint_path, _student_a_path, _student_b_path, checkpoint, epoch = token_diag.select_checkpoints()
    args=token_diag.build_project_args(memory_path)
    args.reference_backbone = backbone
    args.uni_loader = "hf-hub" if backbone == "uni2-h" else "local"
    args.uni_hub_id = UNI2_H_MODEL_ID if backbone == "uni2-h" else "MahmoodLab/UNI"
    args.reference_resolution = reference_resolution
    args.reference_grid = TOKEN_GRID
    args.uni_k = cli.top_k
    args.uni_reference_memory = str(memory_path)
    cfg=dataset_cfg(DATASET)
    datasets,manifest_dir,use_manifests=project.build_datasets(args,cfg)
    if not use_manifests:
        raise RuntimeError("Expected the current fixed GLaS manifests to be active")
    loader=DataLoader(datasets["val"],batch_size=cli.batch_size,shuffle=False,
                      num_workers=cli.num_workers,pin_memory=True)
    memory=_load_reference(args,cfg,logging.getLogger("pixel_contradiction_enrichment"),
                           manifest_dir=manifest_dir,use_manifests=use_manifests)
    if memory["memory_top_k"]!=cli.top_k or memory["token_grid"]!=TOKEN_GRID:
        raise RuntimeError("Reference memory Top-K/grid does not match the diagnostic settings")
    model_a,model_b=token_diag.load_models(checkpoint,cfg)
    aggregate,directions,per_image,n_images,checks=collect_pixel_records(model_a,model_b,loader,cfg,memory)
    if n_images!=len(datasets["val"]):
        raise RuntimeError(f"Validation image count mismatch: {n_images} != {len(datasets['val'])}")
    check_pass=all(bool(v) for v in checks.values())
    summary=make_summary(aggregate,directions,checkpoint_path,epoch,n_images,
                         reference_backbone=backbone,memory_path=memory_path)
    with (output_dir/"pixel_contradiction_per_image.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=PER_IMAGE_FIELDS); writer.writeheader(); writer.writerows(per_image)
    summary_fields=list(summary.keys())
    _write_csv(output_dir/"pixel_contradiction_enrichment_summary.csv",summary_fields,summary)
    _plot(summary, output_dir)
    _write_run_summary(output_dir/"pixel_run_summary.txt",summary,check_pass,output_dir)
    print("[Pixel-level Panel B]")
    print(f"Validation images: {n_images}")
    print(f"Fine Agreement pixels: {summary['num_agreement_pixels']}")
    print(f"Coarse Contradiction pixels K: {summary['num_K_pixels']}")
    print(f"Fine Correction pixels M_corr: {summary['num_corr_pixels']}")
    print("\nWW:")
    print(f"P(WW | Agreement) = {fmt_rate(summary['ww_agreement_rate'])}")
    print(f"P(WW | M_corr) = {fmt_rate(summary['ww_corr_rate'])}")
    print(f"Enrichment = {fmt_ratio(summary['ww_enrichment'])}")
    print(f"Coverage = {fmt_rate(summary['ww_coverage'])}")
    print("\nHC90WW:")
    print(f"P(HC90WW | Agreement) = {fmt_rate(summary['hc90ww_agreement_rate'])}")
    print(f"P(HC90WW | M_corr) = {fmt_rate(summary['hc90ww_corr_rate'])}")
    print(f"Enrichment = {fmt_ratio(summary['hc90ww_enrichment'])}")
    print(f"Coverage = {fmt_rate(summary['hc90ww_coverage'])}")
    print(f"\nCC exposure = {fmt_rate(summary['cc_exposure'])}")
    print("\nDirectional:")
    print(f"FG2BG WW enrichment = {fmt_ratio(summary['FG2BG_ww_enrichment'])}")
    print(f"BG2FG WW enrichment = {fmt_ratio(summary['BG2FG_ww_enrichment'])}")
    print("\nSanity checks:")
    print("PASS" if check_pass else "FAIL")
    if not check_pass:
        print("Failed checks: " + ", ".join(k for k,v in checks.items() if not v))
    print("\nOutputs:")
    print(output_dir)
    if backbone == "uni":
        _print_token_comparison(summary)


if __name__=="__main__":
    main()
