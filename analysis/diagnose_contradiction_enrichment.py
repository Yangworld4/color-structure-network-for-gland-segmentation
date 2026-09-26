#!/usr/bin/env python3
"""GlaS CPS-error enrichment under UNI semantic contradiction.

This validation-only diagnostic loads the existing seed-1337 Vanilla CPS
checkpoint pair and the train-only UNI memory. It performs no training.
Pass ``--reuse-token-records`` to update metrics and figures from saved CSVs
without loading models or running inference.
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project
import test as test_project
from config.dataset_config.dataset_cfg import dataset_cfg
from config.uni_config.uni_config import UNI_CONFIG_DEFAULT, UNI_WEIGHTS_DEFAULT
from loss.loss_function import DiceLoss
from models.getnetwork import get_network
from training.cps import get_training_views
from utils.native_reference import (
    _get_rgb_view,
    _load_reference,
    _reconstruct_rgb,
    _reference_scores,
    _uni_dense_features,
)
from utils.training_diagnostics import foreground_probability


DATASET = "GlaS"
LABELED_RATIO = 10
SEED = 1337
NETWORK = "unet"
INPUT_MODE = "rgb_rgb"
BATCH_SIZE = 2
NUM_WORKERS = 4
REFERENCE_RESOLUTION = 256
TOKEN_GRID = 16
TOP_K = 50
CONFIDENCE_THRESHOLD = 0.90

CHECKPOINT_ROOT = ROOT / "checkpoints/semi_10_seed_1337_rgb/GlaS"
EXISTING_MEMORY_GLOB = (
    "checkpoints/semi_10_seed_1337_rgbstain_coarse2fine/"
    "GlaS/*/reference_memory_uni256.npz"
)
OUTPUT_DIR = ROOT / "outputs/contradiction_enrichment_glas10_seed1337"

TOKEN_FIELDS = [
    "image_id",
    "token_y",
    "token_x",
    "student_a_class",
    "student_b_class",
    "student_a_confidence",
    "student_b_confidence",
    "gt_class",
    "s_fg",
    "s_bg",
    "s_cur",
    "s_opp",
    "D",
    "contradiction",
    "is_cc",
    "is_ww",
    "is_hc90ww",
    "is_shared_fp",
    "is_shared_fn",
]

SUMMARY_FIELDS = [
    "dataset",
    "labeled_ratio",
    "seed",
    "checkpoint",
    "epoch",
    "num_validation_images",
    "num_agreement_tokens",
    "num_contradiction_tokens",
    "contradiction_ratio",
    "ww_all_count",
    "ww_all_rate",
    "ww_contradiction_count",
    "ww_contradiction_rate",
    "ww_enrichment",
    "hc90ww_all_count",
    "hc90ww_all_rate",
    "hc90ww_contradiction_count",
    "hc90ww_contradiction_rate",
    "hc90ww_enrichment",
    "cc_all_count",
    "cc_all_rate",
    "cc_contradiction_count",
    "cc_contradiction_rate",
    "ww_coverage",
    "hc90ww_coverage",
    "ww_contradiction_precision",
    "hc90ww_contradiction_precision",
]

BASELINE_COLOR = "#b9bec5"
CONTRADICTION_COLOR = "#6f91aa"
FONT_FAMILY = "DejaVu Sans"


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _safe_enrichment(condition_rate: float | None, baseline_rate: float | None) -> float | None:
    if condition_rate is None or baseline_rate in (None, 0.0):
        return None
    return condition_rate / baseline_rate


def _same_state_dict(first: dict, second: dict) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[key], second[key]) for key in first
    )


def select_checkpoints() -> tuple[Path, Path, Path, dict, int]:
    """Use test.py's validation-best selection for each CPS branch."""
    student_a_path = test_project.find_best_checkpoint(
        CHECKPOINT_ROOT, DATASET, input_view="student1"
    ).resolve()
    student_b_path = test_project.find_best_checkpoint(
        CHECKPOINT_ROOT, DATASET, input_view="student2"
    ).resolve()
    default_test_choice = test_project.find_best_checkpoint(
        CHECKPOINT_ROOT, DATASET, input_view="auto"
    ).resolve()

    if default_test_choice != student_b_path:
        raise RuntimeError(
            "The test.py default checkpoint is not the expected selected CPS branch: "
            f"{default_test_choice}"
        )
    if student_a_path.parent != student_b_path.parent:
        raise RuntimeError("Student A/B best checkpoints are from different CPS runs")

    paired_path = student_a_path.parent / "best_synchronized_mean_val.pth"
    if not paired_path.is_file():
        raise FileNotFoundError(f"Paired validation-best checkpoint not found: {paired_path}")
    paired = torch.load(paired_path, map_location="cpu", weights_only=False)
    required = {"epoch", "appearance_state_dict", "structure_state_dict"}
    if not required.issubset(paired):
        raise KeyError(f"Paired checkpoint is missing fields: {sorted(required - paired.keys())}")

    state_a = torch.load(student_a_path, map_location="cpu", weights_only=False)
    state_b = torch.load(student_b_path, map_location="cpu", weights_only=False)
    if not _same_state_dict(state_a, paired["appearance_state_dict"]):
        raise RuntimeError("Synchronized Student A is not the test.py validation-best Student A")
    if not _same_state_dict(state_b, paired["structure_state_dict"]):
        raise RuntimeError("Synchronized Student B is not the test.py validation-best Student B")

    epoch = int(paired["epoch"])
    train_log = student_a_path.parent / "train.log"
    if train_log.is_file():
        text = train_log.read_text(encoding="utf-8", errors="replace")
        final_match = re.search(
            r"Training completed \| result1_best_epoch=(\d+).*?"
            r"result2_best_epoch=(\d+)",
            text,
        )
        if final_match and tuple(map(int, final_match.groups())) != (epoch, epoch):
            raise RuntimeError(
                "Paired checkpoint epoch does not match the selected Student A/B "
                f"validation-best epochs: checkpoint={epoch}, log={final_match.groups()}"
            )
    return paired_path, student_a_path, student_b_path, paired, epoch


def find_memory_path() -> Path | None:
    matches = sorted(ROOT.glob(EXISTING_MEMORY_GLOB))
    if len(matches) > 1:
        raise RuntimeError(f"Expected one matching 10% seed-1337 GlaS memory, found: {matches}")
    return matches[0].resolve() if matches else None


def build_project_args(memory_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=DATASET,
        path_dataset=str(ROOT / "datasets/GlaS"),
        manifest_dir=None,
        labeled_ratio=LABELED_RATIO,
        input1="image",
        input_mode=INPUT_MODE,
        sup_mark="7",
        unsup_mark="61",
        network=NETWORK,
        cps_mode="standard",
        reference_mode="none",
        reference_resolution=REFERENCE_RESOLUTION,
        uni_k=TOP_K,
        uni_weights=UNI_WEIGHTS_DEFAULT,
        uni_config=UNI_CONFIG_DEFAULT,
        uni_reference_memory=str(memory_path),
        uni_build_reference_memory=False,
        batch_size=BATCH_SIZE,
    )


def load_models(paired_checkpoint: dict, cfg: dict):
    model_a = get_network(NETWORK, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    model_b = get_network(NETWORK, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    model_a.load_state_dict(paired_checkpoint["appearance_state_dict"], strict=True)
    model_b.load_state_dict(paired_checkpoint["structure_state_dict"], strict=True)
    model_a.eval()
    model_b.eval()
    return model_a, model_b


def _class_name(is_foreground: bool) -> str:
    return "FG" if is_foreground else "BG"


def collect_token_records(model_a, model_b, loader, cfg, memory_context):
    records: list[dict[str, object]] = []
    validation_images = 0
    agreement_count = 0
    contradiction_count = 0
    ww_count = 0
    ww_contradiction_count = 0
    hc_count = 0
    hc_contradiction_count = 0
    cc_count = 0
    cc_contradiction_count = 0

    with torch.inference_mode():
        for batch in loader:
            input_a, input_b = get_training_views(batch)
            input_a = input_a.cuda(non_blocking=True)
            input_b = input_b.cuda(non_blocking=True)
            logits_a = model_a(input_a)[0]
            logits_b = model_b(input_b)[0]

            probability_a = foreground_probability(logits_a)[:, 0]
            probability_b = foreground_probability(logits_b)[:, 0]
            token_probability_a = F.adaptive_avg_pool2d(
                probability_a[:, None], (TOKEN_GRID, TOKEN_GRID)
            )
            token_probability_b = F.adaptive_avg_pool2d(
                probability_b[:, None], (TOKEN_GRID, TOKEN_GRID)
            )
            token_probability_a = token_probability_a[:, 0]
            token_probability_b = token_probability_b[:, 0]
            token_class_a = token_probability_a.ge(0.5)
            token_class_b = token_probability_b.ge(0.5)
            confidence_a = torch.maximum(token_probability_a, 1.0 - token_probability_a)
            confidence_b = torch.maximum(token_probability_b, 1.0 - token_probability_b)
            agreement = token_class_a.eq(token_class_b)

            ground_truth = batch["mask"].cuda(non_blocking=True)
            if ground_truth.dim() == 3:
                ground_truth = ground_truth[:, None]
            token_gt_occupancy = F.adaptive_avg_pool2d(
                ground_truth.float(), (TOKEN_GRID, TOKEN_GRID)
            )[:, 0]
            token_gt_class = token_gt_occupancy.ge(0.5)

            rgb = _reconstruct_rgb(_get_rgb_view(batch).cuda(non_blocking=True), cfg)
            features = _uni_dense_features(
                memory_context["uni"], rgb, REFERENCE_RESOLUTION
            )
            if tuple(features.shape[1:]) != (TOKEN_GRID * TOKEN_GRID, 1024):
                raise RuntimeError(f"Unexpected UNI token shape: {tuple(features.shape)}")
            score_fg, score_bg = _reference_scores(
                features, memory_context, int(memory_context["memory_top_k"])
            )
            score_fg = score_fg.reshape(-1, TOKEN_GRID, TOKEN_GRID)
            score_bg = score_bg.reshape(-1, TOKEN_GRID, TOKEN_GRID)

            score_current = torch.where(token_class_a, score_fg, score_bg)
            score_opposite = torch.where(token_class_a, score_bg, score_fg)
            contradiction_margin = score_opposite - score_current
            contradiction = contradiction_margin.gt(0.0)

            is_cc = agreement & token_class_a.eq(token_gt_class)
            is_ww = agreement & token_class_a.ne(token_gt_class)
            is_hc90ww = (
                is_ww
                & confidence_a.ge(CONFIDENCE_THRESHOLD)
                & confidence_b.ge(CONFIDENCE_THRESHOLD)
            )
            shared_fp = is_ww & token_class_a & ~token_gt_class
            shared_fn = is_ww & ~token_class_a & token_gt_class

            for image_index, image_id in enumerate(batch["ID"]):
                validation_images += 1
                for token_y, token_x in torch.nonzero(
                    agreement[image_index], as_tuple=False
                ).tolist():
                    is_contradiction = bool(contradiction[image_index, token_y, token_x])
                    is_token_cc = bool(is_cc[image_index, token_y, token_x])
                    is_token_ww = bool(is_ww[image_index, token_y, token_x])
                    is_token_hc = bool(is_hc90ww[image_index, token_y, token_x])
                    agreement_count += 1
                    contradiction_count += int(is_contradiction)
                    ww_count += int(is_token_ww)
                    ww_contradiction_count += int(is_token_ww and is_contradiction)
                    hc_count += int(is_token_hc)
                    hc_contradiction_count += int(is_token_hc and is_contradiction)
                    cc_count += int(is_token_cc)
                    cc_contradiction_count += int(is_token_cc and is_contradiction)
                    records.append({
                        "image_id": str(image_id),
                        "token_y": token_y,
                        "token_x": token_x,
                        "student_a_class": _class_name(bool(token_class_a[image_index, token_y, token_x])),
                        "student_b_class": _class_name(bool(token_class_b[image_index, token_y, token_x])),
                        "student_a_confidence": float(confidence_a[image_index, token_y, token_x]),
                        "student_b_confidence": float(confidence_b[image_index, token_y, token_x]),
                        "gt_class": _class_name(bool(token_gt_class[image_index, token_y, token_x])),
                        "s_fg": float(score_fg[image_index, token_y, token_x]),
                        "s_bg": float(score_bg[image_index, token_y, token_x]),
                        "s_cur": float(score_current[image_index, token_y, token_x]),
                        "s_opp": float(score_opposite[image_index, token_y, token_x]),
                        "D": float(contradiction_margin[image_index, token_y, token_x]),
                        "contradiction": int(is_contradiction),
                        "is_cc": int(is_token_cc),
                        "is_ww": int(is_token_ww),
                        "is_hc90ww": int(is_token_hc),
                        "is_shared_fp": int(bool(shared_fp[image_index, token_y, token_x])),
                        "is_shared_fn": int(bool(shared_fn[image_index, token_y, token_x])),
                    })

    counts = {
        "validation_images": validation_images,
        "agreement": agreement_count,
        "contradiction": contradiction_count,
        "ww": ww_count,
        "ww_contradiction": ww_contradiction_count,
        "hc90ww": hc_count,
        "hc90ww_contradiction": hc_contradiction_count,
        "cc": cc_count,
        "cc_contradiction": cc_contradiction_count,
    }
    return records, counts


def summarize(records, counts, checkpoint_path: Path, checkpoint_epoch: int):
    n_agreement = counts["agreement"]
    n_contradiction = counts["contradiction"]
    values = {
        "dataset": DATASET,
        "labeled_ratio": LABELED_RATIO,
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint_epoch,
        "num_validation_images": counts["validation_images"],
        "num_agreement_tokens": n_agreement,
        "num_contradiction_tokens": n_contradiction,
        "contradiction_ratio": _safe_rate(n_contradiction, n_agreement),
        "ww_all_count": counts["ww"],
        "ww_all_rate": _safe_rate(counts["ww"], n_agreement),
        "ww_contradiction_count": counts["ww_contradiction"],
        "ww_contradiction_rate": _safe_rate(counts["ww_contradiction"], n_contradiction),
        "ww_enrichment": _safe_enrichment(
            _safe_rate(counts["ww_contradiction"], n_contradiction),
            _safe_rate(counts["ww"], n_agreement),
        ),
        "ww_coverage": _safe_rate(counts["ww_contradiction"], counts["ww"]),
        "ww_contradiction_precision": _safe_rate(
            counts["ww_contradiction"], n_contradiction
        ),
        "hc90ww_all_count": counts["hc90ww"],
        "hc90ww_all_rate": _safe_rate(counts["hc90ww"], n_agreement),
        "hc90ww_contradiction_count": counts["hc90ww_contradiction"],
        "hc90ww_contradiction_rate": _safe_rate(
            counts["hc90ww_contradiction"], n_contradiction
        ),
        "hc90ww_enrichment": _safe_enrichment(
            _safe_rate(counts["hc90ww_contradiction"], n_contradiction),
            _safe_rate(counts["hc90ww"], n_agreement),
        ),
        "hc90ww_coverage": _safe_rate(
            counts["hc90ww_contradiction"], counts["hc90ww"]
        ),
        "hc90ww_contradiction_precision": _safe_rate(
            counts["hc90ww_contradiction"], n_contradiction
        ),
        "cc_all_count": counts["cc"],
        "cc_all_rate": _safe_rate(counts["cc"], n_agreement),
        "cc_contradiction_count": counts["cc_contradiction"],
        "cc_contradiction_rate": _safe_rate(
            counts["cc_contradiction"], n_contradiction
        ),
    }
    return values


def write_token_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOKEN_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)


def _plot(summary: dict[str, object]) -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    categories = ["WW", "HC90WW"]
    positions = np.arange(len(categories), dtype=np.float64)
    all_rates = [float(summary["ww_all_rate"]), float(summary["hc90ww_all_rate"])]
    contradiction_rates = [
        float(summary["ww_contradiction_rate"]),
        float(summary["hc90ww_contradiction_rate"]),
    ]
    all_values = [100.0 * value for value in all_rates]
    contradiction_values = [100.0 * value for value in contradiction_rates]
    enrichments = [summary["ww_enrichment"], summary["hc90ww_enrichment"]]
    height = 0.30

    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor="white")
    bars_all = ax.barh(
        positions - height / 2, all_values, height,
        color=BASELINE_COLOR, edgecolor="none", label="All CPS Agreement",
    )
    bars_contradiction = ax.barh(
        positions + height / 2, contradiction_values, height,
        color=CONTRADICTION_COLOR, edgecolor="none", label="Memory Contradiction",
    )
    for bar, value in zip(bars_all, all_values):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", ha="left", va="center", fontsize=9)
    for bar, value, ratio in zip(bars_contradiction, contradiction_values, enrichments):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(value + 1.0, y, f"{value:.1f}%", ha="left", va="center", fontsize=9)
        if ratio is not None:
            ax.text(value + 9.2, y, f"×{float(ratio):.2f}", ha="left", va="center",
                    fontsize=9, fontweight="semibold", color="#405e73")

    ax.set_yticks(positions, categories)
    ax.invert_yaxis()
    ax.set_xlabel("Error prevalence (%)", fontsize=10.5)
    ax.set_xlim(0, max(82.0, max(contradiction_values) + 24.0))
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=3)
    fig.suptitle("CPS Error Enrichment under Semantic Memory Contradiction", fontsize=12, y=0.985)
    fig.legend(
        handles=[bars_all[0], bars_contradiction[0]],
        labels=["All CPS Agreement", "Memory Contradiction"],
        frameon=False, fontsize=8.7, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2,
    )
    coverage_text = (
        "Contradiction coverage\n"
        f"WW: {_format_rate(summary['ww_coverage'])}\n"
        f"HC90WW: {_format_rate(summary['hc90ww_coverage'])}"
    )
    ax.text(
        0.985, 0.035, coverage_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    fig.savefig(OUTPUT_DIR / "contradiction_enrichment_horizontal.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "contradiction_enrichment_horizontal.pdf", bbox_inches="tight")
    plt.close(fig)

    coverage = [summary["ww_coverage"], summary["hc90ww_coverage"]]
    coverage_values = [0.0 if value is None else 100.0 * float(value) for value in coverage]
    fig, ax = plt.subplots(figsize=(5.6, 3.3), facecolor="white")
    bars = ax.barh(positions, coverage_values, height=0.42,
                   color=CONTRADICTION_COLOR, edgecolor="none")
    for bar, value, rate in zip(bars, coverage_values, coverage):
        label = "undefined" if rate is None else f"{value:.1f}%"
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=9)
    ax.set_yticks(positions, categories)
    ax.invert_yaxis()
    ax.set_xlabel("Coverage (%)", fontsize=10.5)
    ax.set_xlim(0, max(90.0, max(coverage_values) + 12.0))
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="out", length=3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "contradiction_coverage_horizontal.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "contradiction_coverage_horizontal.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_rate(value: float | None) -> str:
    return "undefined (zero denominator)" if value is None else f"{100.0 * value:.4f}%"


def _format_ratio(value: float | None) -> str:
    return "undefined (zero denominator)" if value is None else f"{value:.6f}x"


def write_run_summary(
    path: Path,
    summary: dict[str, object],
    checkpoint_path: Path,
    student_a_path: Path,
    student_b_path: Path,
    memory_path: Path,
    memory_context: dict,
    manifest_dir: str,
    memory_built: bool,
) -> None:
    lines = [
        "[Experiment]",
        f"Dataset: {DATASET}",
        f"Labeled ratio: {LABELED_RATIO}%",
        f"Seed: {SEED}",
        "Input mode: rgb_rgb (Vanilla CPS)",
        "Validation manifest: " + str(Path(manifest_dir) / "validation.csv"),
        "",
        "[Checkpoint]",
        f"Actual checkpoint path: {checkpoint_path}",
        f"Checkpoint epoch: {summary['epoch']}",
        f"Student A test.py validation-best path: {student_a_path}",
        f"Student B test.py default-selected path: {student_b_path}",
        "Paired checkpoint states verified against both validation-best branch files: yes",
        "",
        "[Semantic Memory]",
        f"Memory path: {memory_path}",
        f"FG tokens: {memory_context['memory_fg_count']}",
        f"BG tokens: {memory_context['memory_bg_count']}",
        f"Top-K: {memory_context['memory_top_k']}",
        f"UNI input size: {REFERENCE_RESOLUTION}",
        f"Native grid: {TOKEN_GRID}x{TOKEN_GRID}",
        f"Labeled split: {Path(manifest_dir) / 'labeled_10.csv'}",
        "Memory class anchors: FG occupancy >= 0.90; BG occupancy <= 0.10",
        f"Memory constructed during this run: {memory_built}",
        "Memory construction source: labeled training images and GT only; validation GT excluded",
        "Memory token-selection RNG seed: 2026 (project memory rule; experiment seed remains 1337)",
        "",
        "[Population]",
        f"Validation images: {summary['num_validation_images']}",
        f"Agreement tokens: {summary['num_agreement_tokens']}",
        f"Contradiction tokens: {summary['num_contradiction_tokens']}",
        f"Contradiction ratio: {_format_rate(summary['contradiction_ratio'])}",
        "",
        "[WW]",
        f"P(WW | Agreement): {_format_rate(summary['ww_all_rate'])}",
        f"P(WW | Contradiction): {_format_rate(summary['ww_contradiction_rate'])}",
        f"WW enrichment: {_format_ratio(summary['ww_enrichment'])}",
        f"WW contradiction coverage: {_format_rate(summary['ww_coverage'])}",
        f"WW contradiction precision: {_format_rate(summary['ww_contradiction_precision'])}",
        "",
        "[HC90WW]",
        f"P(HC90WW | Agreement): {_format_rate(summary['hc90ww_all_rate'])}",
        f"P(HC90WW | Contradiction): {_format_rate(summary['hc90ww_contradiction_rate'])}",
        f"HC90WW enrichment: {_format_ratio(summary['hc90ww_enrichment'])}",
        f"HC90WW contradiction coverage: {_format_rate(summary['hc90ww_coverage'])}",
        f"HC90WW contradiction precision: {_format_rate(summary['hc90ww_contradiction_precision'])}",
        "",
        "[CC]",
        f"P(CC | Agreement): {_format_rate(summary['cc_all_rate'])}",
        f"P(CC | Contradiction): {_format_rate(summary['cc_contradiction_rate'])}",
        "",
        "[Outputs]",
        f"Token CSV: {OUTPUT_DIR / 'contradiction_token_records.csv'}",
        f"Summary CSV: {OUTPUT_DIR / 'contradiction_enrichment_summary.csv'}",
        f"Figure PNG: {OUTPUT_DIR / 'contradiction_enrichment.png'}",
        f"Figure PDF: {OUTPUT_DIR / 'contradiction_enrichment.pdf'}",
        f"Horizontal figure PNG: {OUTPUT_DIR / 'contradiction_enrichment_horizontal.png'}",
        f"Horizontal figure PDF: {OUTPUT_DIR / 'contradiction_enrichment_horizontal.pdf'}",
        f"Coverage figure PNG: {OUTPUT_DIR / 'contradiction_coverage_horizontal.png'}",
        f"Coverage figure PDF: {OUTPUT_DIR / 'contradiction_coverage_horizontal.pdf'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean in token records: {value!r}")


def update_from_saved_records() -> None:
    """Refresh metrics/figures from existing token records without inference."""
    token_path = OUTPUT_DIR / "contradiction_token_records.csv"
    summary_path = OUTPUT_DIR / "contradiction_enrichment_summary.csv"
    run_summary_path = OUTPUT_DIR / "run_summary.txt"
    with token_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
    required = {"is_ww", "is_hc90ww", "is_cc", "contradiction"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Token CSV is missing fields: {sorted(required - set(reader.fieldnames or []))}")
    flags = [
        {
            "ww": _csv_bool(row["is_ww"]),
            "hc": _csv_bool(row["is_hc90ww"]),
            "cc": _csv_bool(row["is_cc"]),
            "contradiction": _csv_bool(row["contradiction"]),
        }
        for row in records
    ]
    counts = {
        "agreement": len(flags),
        "contradiction": sum(row["contradiction"] for row in flags),
        "ww": sum(row["ww"] for row in flags),
        "ww_contradiction": sum(row["ww"] and row["contradiction"] for row in flags),
        "hc90ww": sum(row["hc"] for row in flags),
        "hc90ww_contradiction": sum(row["hc"] and row["contradiction"] for row in flags),
        "cc": sum(row["cc"] for row in flags),
        "cc_contradiction": sum(row["cc"] and row["contradiction"] for row in flags),
    }
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    if len(old_rows) != 1:
        raise ValueError(f"Expected one summary row, found {len(old_rows)}")
    old = old_rows[0]
    counts["validation_images"] = int(old["num_validation_images"])
    calculated = summarize(records, counts, Path(old["checkpoint"]), int(old["epoch"]))
    count_fields = (
        "num_agreement_tokens", "num_contradiction_tokens", "ww_all_count",
        "ww_contradiction_count", "hc90ww_all_count", "hc90ww_contradiction_count",
        "cc_all_count", "cc_contradiction_count",
    )
    for field in count_fields:
        if int(old[field]) != int(calculated[field]):
            raise RuntimeError(
                f"Existing summary differs from token records for {field}: "
                f"{old[field]} != {calculated[field]}; refusing to update"
            )
    rate_fields = (
        "contradiction_ratio", "ww_all_rate", "ww_contradiction_rate", "ww_enrichment",
        "hc90ww_all_rate", "hc90ww_contradiction_rate", "hc90ww_enrichment",
        "cc_all_rate", "cc_contradiction_rate",
    )
    for field in rate_fields:
        if not np.isclose(float(old[field]), float(calculated[field]), rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Existing summary differs from token records for {field}; refusing to update"
            )

    calculated["num_validation_images"] = old["num_validation_images"]
    calculated.update({key: old[key] for key in ("dataset", "labeled_ratio", "seed")})
    write_summary_csv(summary_path, calculated)
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    lines = run_summary_path.read_text(encoding="utf-8").splitlines()
    additions = {
        "[WW]": [
            f"WW contradiction coverage: {_format_rate(calculated['ww_coverage'])}",
            f"WW contradiction precision: {_format_rate(calculated['ww_contradiction_precision'])}",
        ],
        "[HC90WW]": [
            f"HC90WW contradiction coverage: {_format_rate(calculated['hc90ww_coverage'])}",
            f"HC90WW contradiction precision: {_format_rate(calculated['hc90ww_contradiction_precision'])}",
        ],
        "[Outputs]": [
            f"Horizontal figure PNG: {OUTPUT_DIR / 'contradiction_enrichment_horizontal.png'}",
            f"Horizontal figure PDF: {OUTPUT_DIR / 'contradiction_enrichment_horizontal.pdf'}",
            f"Coverage figure PNG: {OUTPUT_DIR / 'contradiction_coverage_horizontal.png'}",
            f"Coverage figure PDF: {OUTPUT_DIR / 'contradiction_coverage_horizontal.pdf'}",
        ],
    }
    new_labels = {line.split(":", 1)[0] for group in additions.values() for line in group}
    lines = [line for line in lines if line.split(":", 1)[0] not in new_labels]
    for section, values in additions.items():
        index = lines.index(section)
        end = next(
            (i for i in range(index + 1, len(lines)) if lines[i].startswith("[")), len(lines)
        )
        lines[end:end] = values
    run_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot(calculated)
    print(f"Token CSV rows: {counts['agreement']}")
    print(
        f"WW enrichment={calculated['ww_enrichment']:.6f} | "
        f"coverage={_format_rate(calculated['ww_coverage'])}"
    )
    print(
        f"HC90WW enrichment={calculated['hc90ww_enrichment']:.6f} | "
        f"coverage={_format_rate(calculated['hc90ww_coverage'])}"
    )
    print(f"Updated outputs: {OUTPUT_DIR}")


def main() -> None:
    if "--reuse-token-records" in sys.argv:
        update_from_saved_records()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen UNI validation diagnostic")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    project.init_seeds(SEED)

    checkpoint_path, student_a_path, student_b_path, checkpoint, checkpoint_epoch = (
        select_checkpoints()
    )
    existing_memory = find_memory_path()
    memory_path = (
        existing_memory
        if existing_memory is not None
        else OUTPUT_DIR / "reference_memory_uni256.npz"
    )

    project_args = build_project_args(memory_path)
    cfg = dataset_cfg(DATASET)
    datasets, manifest_dir, use_manifests = project.build_datasets(project_args, cfg)
    if not use_manifests:
        raise RuntimeError("Expected the current fixed GlaS manifests to be active")
    validation_loader = DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    memory_context = _load_reference(
        project_args,
        cfg,
        logging.getLogger("contradiction_enrichment"),
        manifest_dir=manifest_dir,
        use_manifests=use_manifests,
    )
    if memory_context["memory_top_k"] != TOP_K:
        raise RuntimeError(
            f"Existing memory Top-K={memory_context['memory_top_k']}; expected fixed K={TOP_K}"
        )
    if memory_context["token_grid"] != TOKEN_GRID:
        raise RuntimeError(f"Existing memory grid is not {TOKEN_GRID}x{TOKEN_GRID}")

    model_a, model_b = load_models(checkpoint, cfg)
    records, counts = collect_token_records(
        model_a, model_b, validation_loader, cfg, memory_context
    )
    if counts["validation_images"] != len(datasets["val"]):
        raise RuntimeError(
            "Validation image count mismatch: iterated={} dataset={}".format(
                counts["validation_images"], len(datasets["val"])
            )
        )
    if counts["agreement"] != len(records):
        raise RuntimeError("Token CSV population does not match the agreement-token count")

    summary = summarize(records, counts, checkpoint_path, checkpoint_epoch)
    write_token_csv(OUTPUT_DIR / "contradiction_token_records.csv", records)
    write_summary_csv(OUTPUT_DIR / "contradiction_enrichment_summary.csv", summary)
    _plot(summary)
    write_run_summary(
        OUTPUT_DIR / "run_summary.txt",
        summary,
        checkpoint_path,
        student_a_path,
        student_b_path,
        memory_path,
        memory_context,
        manifest_dir,
        memory_context["memory_built"],
    )

    print(f"Checkpoint: {checkpoint_path} (epoch {checkpoint_epoch})")
    print(f"Memory: {memory_path}")
    print(
        "Agreement={} | Contradiction={} ({})".format(
            summary["num_agreement_tokens"],
            summary["num_contradiction_tokens"],
            _format_rate(summary["contradiction_ratio"]),
        )
    )
    print(
        "WW: Agreement={} | Contradiction={} | Enrichment={}".format(
            _format_rate(summary["ww_all_rate"]),
            _format_rate(summary["ww_contradiction_rate"]),
            _format_ratio(summary["ww_enrichment"]),
        )
    )
    print(
        "HC90WW: Agreement={} | Contradiction={} | Enrichment={}".format(
            _format_rate(summary["hc90ww_all_rate"]),
            _format_rate(summary["hc90ww_contradiction_rate"]),
            _format_ratio(summary["hc90ww_enrichment"]),
        )
    )
    print(
        "CC: Agreement={} | Contradiction={}".format(
            _format_rate(summary["cc_all_rate"]),
            _format_rate(summary["cc_contradiction_rate"]),
        )
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
