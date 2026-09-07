#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot CPS temporal diagnosis motivation figures.

Input:
    joint_state_temporal_summary.csv

Output:
    Figure1a_agreement_disagreement.png/pdf
    Figure1b_shared_error.png/pdf
    Figure1c_high_conf_shared_error.png/pdf

Usage:
    python plot_joint_state_motivation.py \
        --csv diagnostics/joint_state_temporal_e0/joint_state_temporal_summary.csv \
        --outdir diagnostics/joint_state_temporal_e0/motivation_figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="../diagnostics/joint_state_temporal_rgbstain_entropy_routed/joint_state_temporal_summary.csv",
        help="Path to joint_state_temporal_summary.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../diagnostics/rgb_stain_entropy_routed_motivation_figures",
        help="Directory for output figures",
    )
    return parser.parse_args()


def check_columns(df):
    required = [
        "epoch",
        "agreement_rate",
        "disagreement_rate",
        "both_wrong_rate_total",
        "high_conf_shared_error_rate_090",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def setup_axis(ax, ylabel):
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.tick_params(axis="both", labelsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def annotate_endpoint(ax, x, y, text=None, offset=(5, 5)):
    if text is None:
        text = f"{y:.1f}%"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )


def save_figure(fig, outdir, name):
    png_path = outdir / f"{name}.png"
    pdf_path = outdir / f"{name}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


def plot_figure_1a(df, outdir):
    """
    Figure 1(a):
    Models increasingly agree.

    agreement_rate ↑
    disagreement_rate ↓
    """

    epoch = df["epoch"].to_numpy()

    agreement = df["agreement_rate"].to_numpy() * 100
    disagreement = df["disagreement_rate"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        epoch,
        agreement,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Agreement",
    )

    ax.plot(
        epoch,
        disagreement,
        marker="s",
        markersize=4,
        linewidth=2,
        label="Disagreement",
    )

    setup_axis(ax, "Pixel rate")

    ax.set_xticks(epoch)
    ax.tick_params(axis="x", rotation=45)

    ax.set_ylim(0, 105)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc="center right",
    )

    # Highlight first and final epoch
    annotate_endpoint(
        ax,
        epoch[0],
        agreement[0],
        f"{agreement[0]:.1f}%",
        (4, 6),
    )

    annotate_endpoint(
        ax,
        epoch[-1],
        agreement[-1],
        f"{agreement[-1]:.1f}%",
        (-34, 7),
    )

    annotate_endpoint(
        ax,
        epoch[0],
        disagreement[0],
        f"{disagreement[0]:.1f}%",
        (4, 6),
    )

    annotate_endpoint(
        ax,
        epoch[-1],
        disagreement[-1],
        f"{disagreement[-1]:.1f}%",
        (-32, 8),
    )

    ax.set_title(
        "Cross-model predictions increasingly agree",
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        fig,
        outdir,
        "Figure1a_agreement_disagreement",
    )


def plot_figure_1b(df, outdir):
    """
    Figure 1(b):
    Agreement does not eliminate shared errors.

    Plot absolute:
        P(A = B != Y)

    IMPORTANT:
    Do not use shared_error_fraction_of_all_error_pixels here,
    because that metric contains disagreement in its denominator.
    """

    epoch = df["epoch"].to_numpy()

    both_wrong = (
        df["both_wrong_rate_total"].to_numpy() * 100
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        epoch,
        both_wrong,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Shared error",
    )

    setup_axis(ax, "All-pixel rate")

    ax.set_xticks(epoch)
    ax.tick_params(axis="x", rotation=45)

    ymax = max(25, np.ceil(both_wrong.max() / 5) * 5 + 5)
    ax.set_ylim(0, ymax)

    # First / last values
    annotate_endpoint(
        ax,
        epoch[0],
        both_wrong[0],
        f"{both_wrong[0]:.1f}%",
        (5, 7),
    )

    annotate_endpoint(
        ax,
        epoch[-1],
        both_wrong[-1],
        f"{both_wrong[-1]:.1f}%",
        (-35, 8),
    )

    # Explicitly emphasize final residual shared error
    ax.axhline(
        both_wrong[-1],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    ax.set_title(
        "Erroneous consensus remains substantial",
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        fig,
        outdir,
        "Figure1b_shared_error",
    )


def plot_figure_1c(df, outdir):
    """
    Figure 1(c):
    Shared errors become increasingly high-confidence.

    Curve 1:
        P(conf_A >= 0.9 and conf_B >= 0.9 | A=B!=Y)

    Curve 2:
        P(A=B!=Y and conf_A>=0.9 and conf_B>=0.9)

    Curve 2 is derived as:

        both_wrong_rate_total
        *
        high_conf_shared_error_rate_090
    """

    epoch = df["epoch"].to_numpy()

    high_conf_conditional = (
        df["high_conf_shared_error_rate_090"].to_numpy()
        * 100
    )

    high_conf_total = (
        df["both_wrong_rate_total"].to_numpy()
        * df["high_conf_shared_error_rate_090"].to_numpy()
        * 100
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        epoch,
        high_conf_conditional,
        marker="o",
        markersize=4,
        linewidth=2,
        label="High-conf. fraction within shared errors",
    )

    ax.plot(
        epoch,
        high_conf_total,
        marker="s",
        markersize=4,
        linewidth=2,
        label="High-conf. shared error / all pixels",
    )

    setup_axis(ax, "Rate")

    ax.set_xticks(epoch)
    ax.tick_params(axis="x", rotation=45)

    ax.set_ylim(0, 100)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
    )

    # Conditional high-confidence shared-error rate
    annotate_endpoint(
        ax,
        epoch[0],
        high_conf_conditional[0],
        f"{high_conf_conditional[0]:.1f}%",
        (5, 6),
    )

    annotate_endpoint(
        ax,
        epoch[-1],
        high_conf_conditional[-1],
        f"{high_conf_conditional[-1]:.1f}%",
        (-35, 7),
    )

    # Absolute high-confidence shared-error burden
    annotate_endpoint(
        ax,
        epoch[0],
        high_conf_total[0],
        f"{high_conf_total[0]:.1f}%",
        (5, 6),
    )

    annotate_endpoint(
        ax,
        epoch[-1],
        high_conf_total[-1],
        f"{high_conf_total[-1]:.1f}%",
        (-35, 7),
    )

    ax.set_title(
        "Shared errors become increasingly confident",
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        fig,
        outdir,
        "Figure1c_high_conf_shared_error",
    )

    # Also save derived values for later paper analysis
    derived = pd.DataFrame(
        {
            "epoch": epoch,
            "high_conf_shared_error_rate_090":
                df["high_conf_shared_error_rate_090"],
            "both_wrong_rate_total":
                df["both_wrong_rate_total"],
            "high_conf_both_wrong_rate_total":
                df["both_wrong_rate_total"]
                * df["high_conf_shared_error_rate_090"],
        }
    )

    derived_path = (
        outdir
        / "high_conf_shared_error_derived.csv"
    )

    derived.to_csv(
        derived_path,
        index=False,
    )

    print(f"[SAVE] {derived_path}")


def print_summary(df):
    first = df.iloc[0]
    last = df.iloc[-1]

    first_high_total = (
        first["both_wrong_rate_total"]
        * first["high_conf_shared_error_rate_090"]
    )

    last_high_total = (
        last["both_wrong_rate_total"]
        * last["high_conf_shared_error_rate_090"]
    )

    print()
    print("=" * 65)
    print("CPS MOTIVATION FIGURE SUMMARY")
    print("=" * 65)

    print(
        f"Epoch range: "
        f"{int(first['epoch'])} -> {int(last['epoch'])}"
    )

    print()
    print("Figure 1(a)")
    print(
        f"Agreement:    "
        f"{first['agreement_rate'] * 100:.2f}% "
        f"-> {last['agreement_rate'] * 100:.2f}%"
    )
    print(
        f"Disagreement: "
        f"{first['disagreement_rate'] * 100:.2f}% "
        f"-> {last['disagreement_rate'] * 100:.2f}%"
    )

    print()
    print("Figure 1(b)")
    print(
        f"Shared error / all pixels: "
        f"{first['both_wrong_rate_total'] * 100:.2f}% "
        f"-> {last['both_wrong_rate_total'] * 100:.2f}%"
    )

    print()
    print("Figure 1(c)")
    print(
        f"High-conf fraction within shared errors: "
        f"{first['high_conf_shared_error_rate_090'] * 100:.2f}% "
        f"-> {last['high_conf_shared_error_rate_090'] * 100:.2f}%"
    )

    print(
        f"High-conf shared error / all pixels: "
        f"{first_high_total * 100:.2f}% "
        f"-> {last_high_total * 100:.2f}%"
    )

    print("=" * 65)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(csv_path)

    check_columns(df)

    # Ensure temporal ordering
    df = (
        df.sort_values("epoch")
        .reset_index(drop=True)
    )

    print(
        f"[INFO] Loaded {len(df)} epochs "
        f"from {csv_path}"
    )

    print(
        "[INFO] Epochs:",
        df["epoch"].tolist(),
    )

    plot_figure_1a(df, outdir)
    plot_figure_1b(df, outdir)
    plot_figure_1c(df, outdir)

    print_summary(df)


if __name__ == "__main__":
    main()