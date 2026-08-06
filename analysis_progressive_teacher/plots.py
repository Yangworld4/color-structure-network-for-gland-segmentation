"""Matplotlib outputs for progressive-teacher feasibility analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


COLORS = {"A": "#0072B2", "S": "#D55E00", "union": "#009E73", "neutral": "#555555"}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure generation requires matplotlib already available in the environment; "
            "use --no-figures to skip plots."
        ) from exc
    return plt


def plot_cumulative(rows: Sequence[Mapping], output: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for axis, direction, title in zip(
        axes,
        ("chronological", "reverse"),
        ("Chronological", "Reverse history"),
    ):
        selected = [row for row in rows if row["direction"] == direction]
        x = [row["num_checkpoint_pairs"] for row in selected]
        for key, label, color in (
            ("HRR_A", "Appearance", COLORS["A"]),
            ("HRR_S", "Structure", COLORS["S"]),
            ("HRR_union", "Union", COLORS["union"]),
        ):
            axis.plot(x, [100 * row[key] for row in selected], marker="o", ms=3, label=label, color=color)
        axis.set_title(title)
        axis.set_xlabel("Included checkpoint pairs")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Historical recovery rate (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_leader(rows: Sequence[Mapping], dice_output: Path, gap_output: Path) -> None:
    plt = _pyplot()
    epochs = [row["epoch"] for row in rows]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(epochs, [row["A_Dice"] for row in rows], marker="o", label="Appearance", color=COLORS["A"])
    axis.plot(epochs, [row["S_Dice"] for row in rows], marker="s", label="Structure", color=COLORS["S"])
    axis.set(xlabel="Epoch milestone", ylabel="Dice", title="Global leader dynamics (best-so-far snapshots)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dice_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    gaps = np.asarray([row["dice_gap"] for row in rows])
    axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.plot(epochs, gaps, color=COLORS["neutral"], marker="o")
    axis.fill_between(epochs, 0, gaps, where=gaps >= 0, color=COLORS["A"], alpha=0.25, label="Appearance leads")
    axis.fill_between(epochs, 0, gaps, where=gaps < 0, color=COLORS["S"], alpha=0.25, label="Structure leads")
    axis.set(xlabel="Epoch milestone", ylabel="A Dice - S Dice", title="Leader gap")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(gap_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_complementarity(rows: Sequence[Mapping], output: Path) -> None:
    plt = _pyplot()
    epochs = [row["epoch"] for row in rows]
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(epochs, [100 * row["disagreement_ratio"] for row in rows], label="Disagreement", color=COLORS["A"])
    axis.plot(epochs, [100 * row["both_wrong_ratio"] for row in rows], label="Both wrong", color=COLORS["S"])
    axis.plot(epochs, [100 * row["Oracle_Gain"] for row in rows], label="Oracle gain", color=COLORS["union"])
    axis.set(xlabel="Epoch milestone", ylabel="Rate / Dice gain (percentage points)", title="Complementarity over time")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_qualitative(
    output_dir: Path,
    selected_indices: Sequence[int],
    names: Sequence[str],
    image_paths: Sequence[Path],
    targets: np.ndarray,
    pred_a: np.ndarray,
    pred_s: np.ndarray,
    both_wrong: np.ndarray,
    recoverable: np.ndarray,
    persistent: np.ndarray,
    historical_predictions: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in selected_indices:
        best_label = "none"
        best_prediction = pred_a[index]
        best_recovery = -1
        for epoch, (history_a, history_s) in historical_predictions.items():
            for branch, prediction in (("A", history_a[index]), ("S", history_s[index])):
                recovered = int(((prediction == targets[index]) & both_wrong[index]).sum())
                if recovered > best_recovery:
                    best_recovery = recovered
                    best_label = "{} epoch {}".format(branch, epoch)
                    best_prediction = prediction
        with Image.open(image_paths[index]) as image:
            rgb = np.asarray(image.convert("RGB").resize((targets.shape[2], targets.shape[1])))
        panels = (
            ("RGB", rgb),
            ("GT", targets[index]),
            ("Best Appearance", pred_a[index]),
            ("Best Structure", pred_s[index]),
            ("Current both wrong", both_wrong[index]),
            ("Historically recoverable", recoverable[index]),
            ("Persistent", persistent[index]),
            ("Best recovery: {}".format(best_label), best_prediction),
        )
        fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
        for axis, (title, panel) in zip(axes.ravel(), panels):
            axis.imshow(panel, cmap=None if panel.ndim == 3 else "gray", vmin=0, vmax=1 if panel.ndim == 2 else None)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(names[index])
        fig.tight_layout()
        safe_name = Path(names[index]).stem
        fig.savefig(output_dir / "{}_history_recovery.png".format(safe_name), dpi=300, bbox_inches="tight")
        plt.close(fig)
