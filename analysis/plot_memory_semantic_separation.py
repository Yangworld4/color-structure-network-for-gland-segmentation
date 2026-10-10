#!/usr/bin/env python3
"""Offline UNI FG/BG memory semantic-separation figure.

This script reads existing token-level records only. It never loads a model,
performs a forward pass, rebuilds memory, or changes any experiment output.

Example:
    python analysis/plot_memory_semantic_separation.py

If automatic discovery does not find the records:
    python analysis/plot_memory_semantic_separation.py \
        --input outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "memory_semantic_separation_kaiko_vitb8_100"
DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "outputs/native_scale_reference/glas_standard_cps_kaiko_vitb8_100/token_records.csv",
    PROJECT_ROOT / "outputs/e4/token_records.csv",
]


def _first_present(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    fields = {str(value).strip().lower(): value for value in fieldnames}
    for candidate in candidates:
        if candidate.lower() in fields:
            return fields[candidate.lower()]
    return None


def _to_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_class(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"fg", "foreground", "gland", "1", "true", "positive"}:
        return 1
    if text in {"bg", "background", "0", "false", "negative"}:
        return 0
    number = _to_float(value)
    if number is not None and number in (0.0, 1.0):
        return int(number)
    return None


def _find_input(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Token record file does not exist: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.is_file():
            return candidate

    discovered = sorted(
        PROJECT_ROOT.glob("outputs/**/token_records.csv"),
        key=lambda path: str(path),
    )
    if discovered:
        return discovered[0]

    raise FileNotFoundError(
        "Could not find token_records.csv. Provide --input with the existing E0/E4 CSV path."
    )


def _load_records(path: Path) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        fields = list(reader.fieldnames)

    fg_field = _first_present(
        fields,
        ["S_FG", "score_fg", "fg_score", "similarity_fg", "fg_similarity"],
    )
    bg_field = _first_present(
        fields,
        ["S_BG", "score_bg", "bg_score", "similarity_bg", "bg_similarity"],
    )
    class_field = _first_present(
        fields,
        ["gt_class", "gt_semantic_class", "semantic_class", "gt_token_class", "class"],
    )
    ratio_field = _first_present(
        fields,
        ["fg_ratio", "gt_fg_ratio", "gt_occupancy", "occupancy", "purity"],
    )
    top_k_field = _first_present(fields, ["top_k", "k", "retrieval_k"])

    if fg_field is None or bg_field is None:
        raise ValueError(
            "CSV must contain S_FG/S_BG or recognized score aliases; "
            f"found columns: {fields}"
        )
    if class_field is None and ratio_field is None:
        raise ValueError(
            "CSV needs gt_class or a GT foreground-ratio/purity column to derive token class; "
            f"found columns: {fields}"
        )

    column_info = {
        "s_fg": fg_field,
        "s_bg": bg_field,
        "gt_class": class_field or "",
        "gt_ratio": ratio_field or "",
        "top_k": top_k_field or "",
        "gt_rule": "direct gt class" if class_field else "ratio >= 0.9 FG; <= 0.1 BG; mixed ignored",
    }
    return rows, fields, column_info


def _extract_margins(
    rows: list[dict[str, str]],
    column_info: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margins: list[float] = []
    classes: list[int] = []
    pure_flags: list[bool] = []
    skipped = 0
    for row in rows:
        s_fg = _to_float(row.get(column_info["s_fg"]))
        s_bg = _to_float(row.get(column_info["s_bg"]))
        if s_fg is None or s_bg is None:
            skipped += 1
            continue

        gt_class = None
        is_pure = True
        if column_info["gt_class"]:
            gt_class = _parse_class(row.get(column_info["gt_class"]))
        else:
            ratio = _to_float(row.get(column_info["gt_ratio"]))
            if ratio is None:
                skipped += 1
                continue
            if ratio >= 0.9:
                gt_class = 1
            elif ratio <= 0.1:
                gt_class = 0
            else:
                is_pure = False

        if gt_class not in (0, 1) or not is_pure:
            skipped += 1
            continue
        margins.append(s_fg - s_bg)
        classes.append(gt_class)
        pure_flags.append(is_pure)

    if not margins:
        raise ValueError("No valid pure FG/BG validation tokens remained after parsing")
    if skipped:
        print(f"Skipped {skipped} rows with missing/ambiguous score or GT class")
    return np.asarray(margins, dtype=np.float64), np.asarray(classes), np.asarray(pure_flags)


def _stats_for(values: np.ndarray, positive_side: bool) -> dict[str, float | int]:
    correct = values > 0 if positive_side else values < 0
    return {
        "n": int(values.size),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
        "correct_sign_rate": float(correct.mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _kde(ax, values: np.ndarray, color: str) -> bool:
    if values.size < 2 or np.ptp(values) <= 0:
        return False
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return False
    kde = gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), 500)
    ax.plot(x, kde(x), color=color, linewidth=2.0)
    return True


def _plot(
    fg_values: np.ndarray,
    bg_values: np.ndarray,
    balanced_accuracy: float,
    auroc: float,
    output_dir: Path,
) -> None:
    all_values = np.concatenate([fg_values, bg_values])
    lower, upper = np.percentile(all_values, [0.5, 99.5])
    full_min = float(all_values.min())
    full_max = float(all_values.max())
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        lower, upper = full_min - 0.5, full_max + 0.5
    pad = max((upper - lower) * 0.05, 1e-3)
    x_range = (lower - pad, upper + pad)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), facecolor="white")
    bins = np.linspace(x_range[0], x_range[1], 45)
    ax.hist(
        bg_values, bins=bins, density=True, alpha=0.22,
        color="#377eb8", edgecolor="none", label="GT Background",
    )
    ax.hist(
        fg_values, bins=bins, density=True, alpha=0.22,
        color="#e41a1c", edgecolor="none", label="GT Gland",
    )
    kde_bg = _kde(ax, bg_values, "#377eb8")
    kde_fg = _kde(ax, fg_values, "#e41a1c")
    ax.axvline(
        0.0, color="black", linestyle="--", linewidth=1.4,
        label=r"Decision boundary ($\Delta = 0$)",
    )
    ax.set_xlim(*x_range)
    ax.set_xlabel(r"Semantic margin $\Delta = S_{FG} - S_{BG}$", fontsize=10.5, labelpad=43)
    ax.set_ylabel("Density", fontsize=10.5)
    fig.suptitle(
        "Semantic Separation of Unseen Validation Tokens by UNI FG/BG Memory",
        fontsize=12,
        y=0.98,
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.8,
        columnspacing=1.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, direction="out", length=3)
    ax.margins(y=0.32)
    ax.text(
        0.98, 0.97,
        "Balanced Accuracy: {:.1f}%\nAUROC: {:.1f}%".format(
            100.0 * balanced_accuracy,
            100.0 * auroc,
        ),
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
    )
    ax.text(
        0.01, -0.19, r"$\Delta < 0$: BG-favoring",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color="#377eb8", clip_on=False,
    )
    ax.text(
        0.99, -0.19, r"$\Delta > 0$: FG-favoring",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        color="#e41a1c", clip_on=False,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.88))
    fig.savefig(output_dir / "memory_margin_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "memory_margin_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    return kde_bg or kde_fg


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot offline UNI FG/BG memory margin separation")
    parser.add_argument("--input", type=Path, default=None, help="Existing token_records.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Expected retrieval K recorded in the input CSV.",
    )
    args = parser.parse_args()

    input_path = _find_input(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, _fields, column_info = _load_records(input_path)
    detected_top_k = None
    if column_info["top_k"]:
        detected = {
            int(float(str(row[column_info["top_k"]]).strip()))
            for row in rows
            if str(row.get(column_info["top_k"], "")).strip()
        }
        if len(detected) > 1:
            raise ValueError(f"Input contains multiple top-k values: {sorted(detected)}")
        if detected:
            detected_top_k = next(iter(detected))
    if args.top_k is not None and detected_top_k is not None and args.top_k != detected_top_k:
        raise ValueError(
            f"Requested --top-k={args.top_k}, but input records contain top_k={detected_top_k}"
        )
    top_k = args.top_k if args.top_k is not None else detected_top_k
    margins, classes, _pure_flags = _extract_margins(rows, column_info)
    fg_values = margins[classes == 1]
    bg_values = margins[classes == 0]
    if fg_values.size == 0 or bg_values.size == 0:
        raise ValueError(
            f"Both classes are required; got FG={fg_values.size}, BG={bg_values.size}"
        )

    fg_stats = _stats_for(fg_values, positive_side=True)
    bg_stats = _stats_for(bg_values, positive_side=False)
    balanced_accuracy = 0.5 * (
        float(fg_stats["correct_sign_rate"]) + float(bg_stats["correct_sign_rate"])
    )
    auroc = float(roc_auc_score(classes, margins))

    all_stats = _stats_for(margins, True)
    all_stats["correct_sign_rate"] = float(balanced_accuracy)
    stats_rows = [
        {"subset": "ALL", "gt_class": "ALL", **all_stats},
        {"subset": "ALL", "gt_class": "FG", **fg_stats},
        {"subset": "ALL", "gt_class": "BG", **bg_stats},
    ]
    _write_csv(
        output_dir / "memory_margin_statistics.csv",
        stats_rows,
        ["subset", "gt_class", "n", "mean_delta", "median_delta", "min_delta", "max_delta", "correct_sign_rate"],
    )

    # The main plot uses the complete pure FG/BG validation-token population.
    # The supplementary table is kept explicit even when no separate purity
    # field exists in the source records.
    supplementary_rows = []
    for subset_name, subset_mask in (("ALL", np.ones(margins.shape, dtype=bool)),):
        for class_value, class_name, positive_side in ((1, "FG", True), (0, "BG", False)):
            values = margins[subset_mask & (classes == class_value)]
            stats = _stats_for(values, positive_side)
            supplementary_rows.append({"subset": subset_name, "gt_class": class_name, **stats})
    _write_csv(
        output_dir / "memory_margin_supplementary.csv",
        supplementary_rows,
        ["subset", "gt_class", "n", "mean_delta", "median_delta", "min_delta", "max_delta", "correct_sign_rate"],
    )

    has_kde = _plot(fg_values, bg_values, balanced_accuracy, auroc, output_dir)
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "total_tokens": int(margins.size),
        "fg_tokens": int(fg_values.size),
        "bg_tokens": int(bg_values.size),
        "margin": {
            "formula": "S_FG - S_BG",
            "full_min": float(margins.min()),
            "full_max": float(margins.max()),
            "visual_range_p0.5_p99.5": [float(np.percentile(margins, 0.5)), float(np.percentile(margins, 99.5))],
        },
        "fg": fg_stats,
        "bg": bg_stats,
        "balanced_accuracy": float(balanced_accuracy),
        "auroc": auroc,
        "source_columns": column_info,
        "kde_available": bool(has_kde),
        "plot_forward_performed": False,
        "plot_memory_rebuilt": False,
        "interpretation": (
            "The class-aware UNI memory provides discriminative semantic evidence on unseen validation tokens: "
            "gland tokens are predominantly more similar to the foreground memory, while background tokens are "
            "predominantly more similar to the background memory."
            if fg_stats["correct_sign_rate"] > 0.5 and bg_stats["correct_sign_rate"] > 0.5
            else "The validation-token margin distributions do not both show majority-correct semantic sign separation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# UNI FG/BG Memory Semantic Separation

    Input: `{input_path}`

Retrieval Top-K: `{top_k if top_k is not None else 'not recorded'}`

The figure uses the existing validation token records and computes `Delta = S_FG - S_BG`.
No model forward, memory rebuild, training, or CPS modification is performed.

GT class source: `{column_info['gt_rule']}`.

The main population contains all valid pure FG/BG validation tokens according to the
existing direct class or 0.9/0.1 occupancy rule. It is not filtered by margin sign,
CPS agreement, or confidence. The dashed vertical line at `Delta=0` is the semantic
decision boundary.

Complete margin minimum/maximum are stored in `summary.json`. The plot display uses
the 0.5--99.5 percentile range only to prevent a few extreme outliers from compressing
the main distribution; no rows are removed from the statistics.

Outputs:

- `memory_margin_distribution.png` (300 dpi)
- `memory_margin_distribution.pdf`
- `memory_margin_statistics.csv`
- `memory_margin_supplementary.csv`
- `summary.json`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print("=== Memory Semantic Separation ===")
    print(f"Retrieval Top-K: {top_k if top_k is not None else 'not recorded'}")
    print("Validation tokens:")
    print(f"  Total: {margins.size}")
    print(f"  FG: {fg_values.size}")
    print(f"  BG: {bg_values.size}")
    print("\nMargin Δ = S_FG - S_BG")
    print("FG:")
    print(f"  mean: {fg_stats['mean_delta']:.6f}")
    print(f"  median: {fg_stats['median_delta']:.6f}")
    print(f"  correct-sign rate P(Δ>0): {fg_stats['correct_sign_rate']:.6f}")
    print("BG:")
    print(f"  mean: {bg_stats['mean_delta']:.6f}")
    print(f"  median: {bg_stats['median_delta']:.6f}")
    print(f"  correct-sign rate P(Δ<0): {bg_stats['correct_sign_rate']:.6f}")
    print(f"Balanced semantic accuracy: {balanced_accuracy:.6f}")
    print(f"AUROC: {auroc:.6f}")
    print("\nOutput:")
    print(output_dir)


if __name__ == "__main__":
    main()
