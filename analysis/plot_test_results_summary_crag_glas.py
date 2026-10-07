#!/usr/bin/env python3
"""Render the CRAG/GlaS test summary, including multi-seed CPS-XNet-SB."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "outputs/test_results_summary_glass_crag_seeds.png"
OUT_PDF = ROOT / "outputs/test_results_summary_glass_crag_seeds.pdf"
SEEDS = (24, 42, 1337)


def read_score(dataset: str, run_pattern: str, ratio: int, seed: int):
    matches = glob.glob(str(ROOT / f"test_results/{dataset}/{run_pattern.format(seed=seed, ratio=ratio)}/test/metrics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one metric JSON for {dataset} {ratio}% seed {seed}; found {matches}")
    item = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    if item.get("split") != "test":
        raise ValueError(f"Expected test split in {matches[0]}")
    return float(item["dice"]) * 100, float(item["jaccard"]) * 100


def run_rows(dataset, pattern, ratios, seeds=SEEDS, include_sd=True):
    rows = []
    for ratio in ratios:
        values = {seed: read_score(dataset, pattern, ratio, seed) for seed in seeds}
        for seed in seeds:
            dice, jac = values[seed]
            rows.append([f"{ratio}%", str(seed), f"{dice:.2f}", f"{jac:.2f}", False])
        if include_sd:
            arr = np.asarray([values[seed] for seed in seeds], dtype=float)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1)
            rows.append([f"{ratio}%", "均值 ± SD", f"{mean[0]:.2f} ± {sd[0]:.2f}", f"{mean[1]:.2f} ± {sd[1]:.2f}", True])
    return rows


def main():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # GLaS 10/20 seed 1337 run folders carry a `_glas` suffix.
    glas_rows = []
    for ratio in (10, 20):
        vals = {}
        for seed in SEEDS:
            suffix = f"_{ratio}%_glas" if seed == 1337 else f"_{ratio}%"
            run_name = f"cps_seed_{seed}{suffix}"
            p = ROOT / f"test_results/GlaS/{run_name}/test/metrics.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            vals[seed] = (float(d["dice"])*100, float(d["jaccard"])*100)
        for seed in SEEDS:
            glas_rows.append([f"{ratio}%", str(seed), f"{vals[seed][0]:.2f}", f"{vals[seed][1]:.2f}", False])
        a = np.asarray([vals[s] for s in SEEDS])
        m, sd = a.mean(0), a.std(0, ddof=1)
        glas_rows.append([f"{ratio}%", "均值 ± SD", f"{m[0]:.2f} ± {sd[0]:.2f}", f"{m[1]:.2f} ± {sd[1]:.2f}", True])
    sections = [
        ("GLaS · UNet（测试集 80 张）", glas_rows),
        ("CRAG · UNet（测试集 40 张）", run_rows("CRAG", "cps_seed_{seed}_{ratio}%", (10, 20))),
        ("CRAG · CPS-XNet-SB（测试集 40 张）", run_rows("CRAG", "cps_xnet_seed_{seed}_{ratio}%", (10, 20))),
    ]

    header = ["设置", "Seed", "Dice", "Jaccard"]
    fig = plt.figure(figsize=(16, 18.3), dpi=140, facecolor="white")
    fig.text(.5, .975, "CRAG 与 GlaS 测试集结果汇总", ha="center", va="top", fontsize=27, fontweight="bold")
    fig.text(.5, .945, "指标以百分数显示（原始值 × 100）", ha="center", va="top", fontsize=16, color="#58677b")

    top, bottom = .905, .105
    gap = .07
    section_height = (top - bottom - gap * (len(sections)-1)) / len(sections)
    col_widths = [0.16, 0.26, 0.29, 0.29]
    for index, (title, rows) in enumerate(sections):
        y_top = top - index * (section_height + gap)
        title_y = y_top
        table_bottom = y_top - section_height + .012
        fig.text(.01, title_y, title, ha="left", va="top", fontsize=19, fontweight="bold")
        ax = fig.add_axes([.01, table_bottom, .98, section_height - .045])
        ax.axis("off")
        table_data = [header] + [row[:4] for row in rows]
        tbl = ax.table(cellText=table_data, cellLoc="center", colWidths=col_widths, bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(15)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d4deea")
            cell.set_linewidth(1.0)
            if r == 0:
                cell.set_facecolor("#24496d")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            elif rows[r-1][4]:
                cell.set_facecolor("#e5eff9")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white" if r % 2 else "#f0f4f8")

    fig.text(.01, .027,
             "UNet 汇总为三个 seed 的均值 ± 样本标准差；CPS-XNet-SB 使用三个 seed（24、42、1337）。",
             ha="left", va="bottom", fontsize=12.5, color="#536274")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
