"""Publication-style plots for the offline complementarity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


APPEARANCE_COLOR = "#0072B2"
STRUCTURE_COLOR = "#D55E00"
ORACLE_COLOR = "#009E73"
REGION_LABELS = {
    "all_pixels": "All",
    "foreground": "Foreground",
    "background": "Background",
    "boundary": "Boundary",
    "interior": "Interior",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, base_path: Path, formats=("pdf", "png", "svg")) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(base_path.with_suffix("." + extension), **kwargs)
    plt.close(fig)


def plot_disagreement_winner(
    disagreement_winner: Mapping[str, Mapping[str, float]],
    figures_dir: Path,
) -> None:
    _style()
    regions = tuple(REGION_LABELS)
    appearance = [100 * disagreement_winner[name]["A_win_rate"] for name in regions]
    structure = [100 * disagreement_winner[name]["S_win_rate"] for name in regions]
    x = np.arange(len(regions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars_a = ax.bar(x - width / 2, appearance, width, label="Appearance", color=APPEARANCE_COLOR)
    bars_s = ax.bar(x + width / 2, structure, width, label="Structure", color=STRUCTURE_COLOR)
    ax.set_ylabel("Win rate (%)")
    ax.set_xticks(x, [REGION_LABELS[name] for name in regions])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    for bars in (bars_a, bars_s):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    _save(fig, figures_dir / "fig_disagreement_winner")


def plot_confidence_coverage_accuracy(
    curve_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    styles = {
        "all_pixels": ("All", "#000000", "o"),
        "foreground": ("Foreground", APPEARANCE_COLOR, "s"),
        "background": ("Background", STRUCTURE_COLOR, "^"),
    }
    for region, (label, color, marker) in styles.items():
        rows = [row for row in curve_rows if row["region"] == region]
        rows.sort(key=lambda row: float(row["coverage"]))
        ax.plot(
            [100 * float(row["coverage"]) for row in rows],
            [100 * float(row["selection_accuracy"]) for row in rows],
            marker=marker,
            markersize=4,
            linewidth=1.5,
            color=color,
            label=label,
        )
    ax.set_xlabel("Disagreement coverage (%)")
    ax.set_ylabel("Higher-confidence selection accuracy (%)")
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 100)
    ax.grid(linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, figures_dir / "fig_confidence_coverage_accuracy")


def plot_per_image_metrics(
    per_image_rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> None:
    _style()
    appearance = np.asarray([float(row["appearance_dice"]) for row in per_image_rows])
    structure = np.asarray([float(row["structure_dice"]) for row in per_image_rows])
    gains = np.asarray([float(row["oracle_gain"]) for row in per_image_rows])

    fig, ax = plt.subplots(figsize=(4.5, 4.3))
    sizes = 26 + 180 * gains / max(float(gains.max()), 1e-8)
    ax.scatter(
        appearance,
        structure,
        s=sizes,
        c=gains,
        cmap="viridis",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    lower = min(float(appearance.min()), float(structure.min())) - 0.02
    upper = max(float(appearance.max()), float(structure.max())) + 0.02
    ax.plot([lower, upper], [lower, upper], "--", color="0.4", linewidth=1, label="y = x")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Appearance Dice")
    ax.set_ylabel("Structure Dice")
    ax.grid(linewidth=0.5, alpha=0.2)
    ax.legend(frameon=False)
    _save(fig, figures_dir / "fig_per_image_dice_scatter")

    order = np.argsort(-gains)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(np.arange(len(order)), gains[order], color=ORACLE_COLOR, width=0.75)
    ax.set_xlabel("Validation image (sorted by Oracle gain)")
    ax.set_ylabel("Oracle Dice gain over Structure")
    ax.set_xticks(
        np.arange(len(order)),
        [str(per_image_rows[index]["image_id"]) for index in order],
        rotation=55,
        ha="right",
    )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.grid(axis="y", linewidth=0.5, alpha=0.2)
    _save(fig, figures_dir / "fig_per_image_oracle_gain")


def plot_class_error_profile(
    appearance: Mapping[str, float],
    structure: Mapping[str, float],
    figures_dir: Path,
) -> None:
    _style()
    metrics = ("recall", "specificity", "fpr", "fnr")
    labels = ("Recall", "Specificity", "FPR", "FNR")
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(
        x - width / 2,
        [100 * appearance[key] for key in metrics],
        width,
        label="Appearance",
        color=APPEARANCE_COLOR,
    )
    ax.bar(
        x + width / 2,
        [100 * structure[key] for key in metrics],
        width,
        label="Structure",
        color=STRUCTURE_COLOR,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, figures_dir / "fig_class_error_profile")


def plot_qualitative_panel(
    selected_ids: Sequence[str],
    image_records: Mapping[str, Mapping[str, np.ndarray]],
    figures_dir: Path,
) -> None:
    _style()
    qualitative_dir = figures_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["#D9D9D9", "#E69F00", "#56B4E9", "#D55E00", "#000000"])
    columns = ("RGB", "GT", "Appearance", "Structure", "Complementarity")
    fig, axes = plt.subplots(
        len(selected_ids), len(columns),
        figsize=(12.0, 2.35 * len(selected_ids)),
        squeeze=False,
    )
    for row_index, image_id in enumerate(selected_ids):
        record = image_records[image_id]
        panels = (
            record["rgb"], record["gt"], record["pred_a"],
            record["pred_s"], record["relation"],
        )
        for column_index, panel in enumerate(panels):
            ax = axes[row_index, column_index]
            if column_index == 0:
                ax.imshow(panel)
            elif column_index == 4:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            if row_index == 0:
                ax.set_title(columns[column_index])
            if column_index == 0:
                ax.set_ylabel(str(image_id), rotation=90)
            ax.set_xticks([])
            ax.set_yticks([])

        safe_id = str(image_id).replace("/", "_")
        plt.imsave(qualitative_dir / f"{safe_id}_rgb.png", record["rgb"])
        plt.imsave(qualitative_dir / f"{safe_id}_gt.png", record["gt"], cmap="gray", vmin=0, vmax=1)
        plt.imsave(qualitative_dir / f"{safe_id}_pred_appearance.png", record["pred_a"], cmap="gray", vmin=0, vmax=1)
        plt.imsave(qualitative_dir / f"{safe_id}_pred_structure.png", record["pred_s"], cmap="gray", vmin=0, vmax=1)
        plt.imsave(
            qualitative_dir / f"{safe_id}_complementarity.png",
            record["relation"], cmap=cmap, vmin=0, vmax=4,
        )

    handles = [
        Patch(color="#D9D9D9", label="Both correct"),
        Patch(color="#E69F00", label="Appearance only"),
        Patch(color="#56B4E9", label="Structure only"),
        Patch(color="#D55E00", label="Both wrong"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.06, wspace=0.04, hspace=0.08)
    _save(
        fig,
        figures_dir / "fig_qualitative_complementarity",
        formats=("pdf", "png"),
    )
