#!/usr/bin/env python3
"""Offline Figure 3: UNI memory semantic correctness on validation tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv"
DEFAULT_OUTPUT = ROOT / "outputs/memory_semantic_correctness"


def _field(fields, names):
    lowered = {str(name).lower(): name for name in fields}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _float(value):
    try:
        value = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _class(value):
    text = str(value).strip().lower()
    if text in {"fg", "foreground", "gland", "1", "true", "positive"}:
        return 1
    if text in {"bg", "background", "0", "false", "negative"}:
        return 0
    number = _float(value)
    if number in (0.0, 1.0):
        return int(number)
    return None


def _load(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    fg = _field(fields, ["S_FG", "score_fg", "fg_score"])
    bg = _field(fields, ["S_BG", "score_bg", "bg_score"])
    gt = _field(fields, ["gt_class", "gt_semantic_class", "semantic_class", "class"])
    purity = _field(fields, ["purity", "token_purity", "footprint_type", "subset"])
    if fg is None or bg is None or gt is None:
        raise ValueError(
            f"Required fields S_FG, S_BG, gt_class are missing. Columns: {fields}"
        )
    return rows, {"fg": fg, "bg": bg, "gt": gt, "purity": purity}


def _records(rows, fields):
    records = []
    skipped = 0
    for row in rows:
        s_fg = _float(row.get(fields["fg"]))
        s_bg = _float(row.get(fields["bg"]))
        gt = _class(row.get(fields["gt"]))
        if s_fg is None or s_bg is None or gt not in (0, 1):
            skipped += 1
            continue
        delta = s_fg - s_bg
        if delta > 0.0:
            prediction = 1
        elif delta < 0.0:
            prediction = 0
        else:
            prediction = None
        subset = "ALL"
        if fields["purity"] is not None:
            label = str(row.get(fields["purity"], "")).strip().upper()
            if label in {"PURE", "MIXED"}:
                subset = label
        records.append({"gt": gt, "prediction": prediction, "delta": delta, "subset": subset})
    if not records:
        raise ValueError("No valid token records remain after parsing")
    if skipped:
        print(f"Skipped {skipped} invalid token rows")
    return records


def _metrics(records):
    fg = [record for record in records if record["gt"] == 1]
    bg = [record for record in records if record["gt"] == 0]
    correct_fg = sum(record["prediction"] == 1 for record in fg)
    correct_bg = sum(record["prediction"] == 0 for record in bg)
    ties = sum(record["prediction"] is None for record in records)
    correct = correct_fg + correct_bg
    fg_accuracy = correct_fg / len(fg) if fg else 0.0
    bg_accuracy = correct_bg / len(bg) if bg else 0.0
    balanced = 0.5 * (fg_accuracy + bg_accuracy)
    return {
        "total_tokens": len(records),
        "fg_tokens": len(fg),
        "bg_tokens": len(bg),
        "correct_fg": correct_fg,
        "correct_bg": correct_bg,
        "incorrect_fg": len(fg) - correct_fg,
        "incorrect_bg": len(bg) - correct_bg,
        "ties": ties,
        "fg_accuracy": fg_accuracy,
        "bg_accuracy": bg_accuracy,
        "balanced_accuracy": balanced,
        "overall_accuracy": correct / len(records) if records else 0.0,
    }


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _main_figure(metrics, output_dir):
    labels = ["Foreground", "Background", "Balanced"]
    values = [
        100.0 * metrics["fg_accuracy"],
        100.0 * metrics["bg_accuracy"],
        100.0 * metrics["balanced_accuracy"],
    ]
    colors = ["#e15759", "#4e79a7", "#59a14f"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="white")
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="none")
    ax.axhline(50.0, color="#555555", linestyle="--", linewidth=1.0, label="Chance level")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Correctness (%)", fontsize=11)
    ax.set_title("Memory Semantic Correctness", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "memory_semantic_correctness.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "memory_semantic_correctness.pdf", bbox_inches="tight")
    plt.close(fig)


def _supplementary_figure(subset_metrics, output_dir):
    labels = ["ALL", "PURE", "MIXED"]
    values = [100.0 * subset_metrics[label]["balanced_accuracy"] for label in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor="white")
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.62, color=["#59a14f", "#76b7b2", "#f28e2b"], edgecolor="none")
    ax.axhline(50.0, color="#555555", linestyle="--", linewidth=1.0, label="Chance level")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Balanced correctness (%)", fontsize=11)
    ax.set_title("Pure/Mixed Memory Semantic Correctness", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 2.0, f"{value:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_pure_mixed_correctness.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "memory_pure_mixed_correctness.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate offline Figure 3 memory semantic correctness")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Token records not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fields = _load(input_path)
    records = _records(rows, fields)
    all_metrics = _metrics(records)

    csv_rows = []
    for subset_name, subset_metrics in [("ALL", all_metrics)]:
        for metric_name in ("fg_accuracy", "bg_accuracy", "balanced_accuracy", "overall_accuracy"):
            csv_rows.append({
                "subset": subset_name,
                "metric": metric_name,
                "value": subset_metrics[metric_name],
                "num_tokens": subset_metrics["total_tokens"],
            })

    subset_metrics = {"ALL": all_metrics}
    purity_labels = {record["subset"] for record in records}
    pure_mixed_available = {"PURE", "MIXED"}.issubset(purity_labels)
    if pure_mixed_available:
        for label in ("PURE", "MIXED"):
            subset = [record for record in records if record["subset"] == label]
            subset_metrics[label] = _metrics(subset)
            for metric_name in ("fg_accuracy", "bg_accuracy", "balanced_accuracy", "overall_accuracy"):
                csv_rows.append({
                    "subset": label,
                    "metric": metric_name,
                    "value": subset_metrics[label][metric_name],
                    "num_tokens": subset_metrics[label]["total_tokens"],
                })
    _write_csv(
        output_dir / "memory_semantic_correctness.csv",
        csv_rows,
        ["subset", "metric", "value", "num_tokens"],
    )

    confusion_rows = []
    for gt_value, gt_name in ((1, "FG"), (0, "BG")):
        for prediction, prediction_name in ((1, "FG"), (0, "BG"), (None, "tie")):
            confusion_rows.append({
                "subset": "ALL",
                "gt_class": gt_name,
                "predicted_class": prediction_name,
                "count": sum(
                    record["gt"] == gt_value and record["prediction"] == prediction
                    for record in records
                ),
            })
    _write_csv(
        output_dir / "memory_confusion_matrix.csv",
        confusion_rows,
        ["subset", "gt_class", "predicted_class", "count"],
    )

    _main_figure(all_metrics, output_dir)
    if pure_mixed_available:
        _supplementary_figure(subset_metrics, output_dir)

    summary = {
        "input": str(input_path),
        "total_tokens": all_metrics["total_tokens"],
        "fg_tokens": all_metrics["fg_tokens"],
        "bg_tokens": all_metrics["bg_tokens"],
        "fg_accuracy": all_metrics["fg_accuracy"],
        "bg_accuracy": all_metrics["bg_accuracy"],
        "balanced_accuracy": all_metrics["balanced_accuracy"],
        "overall_accuracy": all_metrics["overall_accuracy"],
        "correct_fg": all_metrics["correct_fg"],
        "correct_bg": all_metrics["correct_bg"],
        "incorrect_fg": all_metrics["incorrect_fg"],
        "incorrect_bg": all_metrics["incorrect_bg"],
        "ties": all_metrics["ties"],
        "pure": {
            "available": bool(pure_mixed_available),
            "balanced_accuracy": subset_metrics.get("PURE", {}).get("balanced_accuracy"),
            "num_tokens": subset_metrics.get("PURE", {}).get("total_tokens"),
        },
        "mixed": {
            "available": bool(pure_mixed_available),
            "balanced_accuracy": subset_metrics.get("MIXED", {}).get("balanced_accuracy"),
            "num_tokens": subset_metrics.get("MIXED", {}).get("total_tokens"),
        },
        "confusion": confusion_rows,
        "forward_performed": False,
        "memory_modified": False,
        "cps_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pure_mixed_text = (
        "Semantic discrimination is stronger on semantically pure tokens than on mixed tokens, "
        "indicating that the memory is reliable at the native semantic scale but should not be "
        "interpreted as a dense pixel-level oracle."
        if pure_mixed_available and subset_metrics["PURE"]["balanced_accuracy"] > subset_metrics["MIXED"]["balanced_accuracy"]
        else ""
    )
    readme = f"""# Memory Bank Semantic Correctness

Input records: `{input_path}`

Memory prediction is defined by `Delta = S_FG - S_BG`: positive predicts FG,
negative predicts BG, and exact zero is recorded as a tie. Ties remain in the
class denominators but are excluded from correctness numerators.

Results are computed on existing token-level validation records only. No model
forward, memory rebuild, CPS run, or training is performed by this Figure 3 script.

The main figure reports foreground accuracy, background accuracy, and balanced
accuracy for all valid validation tokens. It supports only native-scale semantic
validity of the memory bank; it does not claim dense pixel-label accuracy or CPS
confirmation-bias resolution.

{pure_mixed_text}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print("=== Memory Bank Semantic Correctness ===")
    print("Validation tokens:")
    print(f"  total: {all_metrics['total_tokens']}")
    print(f"  FG: {all_metrics['fg_tokens']}")
    print(f"  BG: {all_metrics['bg_tokens']}")
    print("Memory prediction:")
    print(f"  FG accuracy: {all_metrics['fg_accuracy']:.6f}")
    print(f"  BG accuracy: {all_metrics['bg_accuracy']:.6f}")
    print(f"  Balanced accuracy: {all_metrics['balanced_accuracy']:.6f}")
    print(f"  Overall accuracy: {all_metrics['overall_accuracy']:.6f}")
    print("Confusion:")
    print(f"  GT FG -> FG: {all_metrics['correct_fg']}")
    print(f"  GT FG -> BG: {all_metrics['incorrect_fg']}")
    print(f"  GT BG -> BG: {all_metrics['correct_bg']}")
    print(f"  GT BG -> FG: {all_metrics['incorrect_bg']}")
    print("PURE/MIXED:")
    print(f"  available: {pure_mixed_available}")
    print(f"  PURE balanced accuracy: {subset_metrics.get('PURE', {}).get('balanced_accuracy')}")
    print(f"  MIXED balanced accuracy: {subset_metrics.get('MIXED', {}).get('balanced_accuracy')}")
    print("Outputs:")
    print(f"  figure: {output_dir / 'memory_semantic_correctness.pdf'}")
    print(f"  csv: {output_dir / 'memory_semantic_correctness.csv'}")
    print(f"  summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
