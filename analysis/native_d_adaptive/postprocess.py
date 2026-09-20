#!/usr/bin/env python3
"""Archive the completed D-Adaptive Native-S run without model inference.

This script only reads the training/validation CSVs and epoch logs emitted by
main.py.  It intentionally performs no checkpoint loading and no forward pass.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/data/mission/models/Exp/DUAL")
TRAIN_OUT = ROOT / "outputs/native_scale_training"
OUT = ROOT / "outputs/native_d_adaptive"
PLOTS = OUT / "plots"
EPOCHS = [40, 80, 120, 160, 200]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TRAIN_OUT / name)


def parse_epoch_metrics(root_name: str) -> pd.DataFrame:
    logs = list((ROOT / root_name).glob("**/train.log"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one train.log under {root_name}, found {logs}")
    rows = []
    pat = re.compile(
        r"epoch=(\d+)/200 .*?val_dice_student1=([0-9.eE+-]+) "
        r".*?val_dice_student2=([0-9.eE+-]+) .*?"
        r"val_jaccard_student1=([0-9.eE+-]+) .*?"
        r"val_jaccard_student2=([0-9.eE+-]+) .*?"
        r"student12_both_correct=([0-9.eE+-]+) .*?"
        r"student12_both_wrong=([0-9.eE+-]+)"
    )
    for line in logs[0].read_text(errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        e, d1, d2, j1, j2, cc, ww = m.groups()
        rows.append({
            "epoch": int(e),
            "Dice": (float(d1) + float(d2)) / 2.0,
            "IoU": (float(j1) + float(j2)) / 2.0,
            "CC_rate_log": float(cc),
            "WW_rate_log": float(ww),
        })
    df = pd.DataFrame(rows).sort_values("epoch").drop_duplicates("epoch")
    if len(df) != 200:
        raise RuntimeError(f"parsed {len(df)} epoch metric rows from {logs[0]}")
    return df


def merge_metrics(tag: str, root_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = read_csv(f"validation_shared_errors_{tag}.csv")
    events = read_csv(f"validation_events_{tag}.csv")
    logs = parse_epoch_metrics(root_name)
    merged = errors.merge(events, on="epoch", validate="one_to_one").merge(logs, on="epoch", validate="one_to_one")
    train = read_csv(f"training_{tag}.csv")
    return merged, train


def metric_row(df: pd.DataFrame, epoch: int, method: str, selection: str) -> dict:
    r = df.loc[df.epoch == epoch].iloc[0]
    return {
        "method": method,
        "selection": selection,
        "epoch": int(epoch),
        "Dice": float(r.Dice),
        "IoU": float(r.IoU),
        "WW_rate": float(r.WW_rate),
        "HC90WW_rate": float(r.HC90WW_rate),
        "CC_rate": float(r.CC_rate),
        "Shared_FP_rate": float(r.Shared_FP_rate),
        "Shared_FN_rate": float(r.Shared_FN_rate),
        "WW_events": int(r.WW_events),
        "HC90WW_events": int(r.HC90WW_events),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    fixed, fixed_train = merge_metrics("native_s", "checkpoints/native_s_seed24")
    adaptive, adaptive_train = merge_metrics(
        "native_d_adaptive_s", "checkpoints/native_d_adaptive_s_seed24"
    )

    # Keep the raw, identical-evaluator validation comparison in a standalone file.
    rows = []
    for e in range(1, 201):
        for method, df in [("Fixed Native-S", fixed), ("D-Adaptive Native-S", adaptive)]:
            rows.append(metric_row(df, e, method, "all_epochs"))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "fixed_vs_d_adaptive.csv", index=False)

    # Required milestone plus each method's best mean validation Dice epoch.
    best_fixed_epoch = int(fixed.loc[fixed.Dice.idxmax(), "epoch"])
    best_adaptive_epoch = int(adaptive.loc[adaptive.Dice.idxmax(), "epoch"])
    selected = []
    for e in EPOCHS:
        selected += [metric_row(fixed, e, "Fixed Native-S", f"epoch_{e}"),
                     metric_row(adaptive, e, "D-Adaptive Native-S", f"epoch_{e}")]
    selected += [metric_row(fixed, best_fixed_epoch, "Fixed Native-S", "best_Dice"),
                 metric_row(adaptive, best_adaptive_epoch, "D-Adaptive Native-S", "best_Dice")]
    pd.DataFrame(selected).to_csv(OUT / "fixed_vs_d_adaptive_selected.csv", index=False)

    # Independent copies of the requested logs.
    fixed_train.assign(method="Fixed Native-S").to_csv(OUT / "fixed_training_log.csv", index=False)
    adaptive_train.assign(method="D-Adaptive Native-S").to_csv(OUT / "training_log.csv", index=False)
    adaptive_train.to_csv(OUT / "weight_statistics.csv", index=False)
    fixed_errors = read_csv("validation_shared_errors_native_s.csv")
    adaptive_errors = read_csv("validation_shared_errors_native_d_adaptive_s.csv")
    validation = pd.concat([
        fixed_errors.assign(method="Fixed Native-S"),
        adaptive_errors.assign(method="D-Adaptive Native-S"),
    ], ignore_index=True)
    validation.to_csv(OUT / "validation_metrics.csv", index=False)

    # Curves required for direct method comparison.
    plt.figure(figsize=(9, 5))
    for name, df in [("Fixed Native-S", fixed), ("D-Adaptive Native-S", adaptive)]:
        plt.plot(df.epoch, df.Dice, label=f"{name} Dice")
        plt.plot(df.epoch, df.IoU, linestyle="--", label=f"{name} IoU")
    plt.xlabel("Epoch"); plt.ylabel("score"); plt.grid(alpha=.25); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(PLOTS / "dice_iou.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    for col in ["WW_rate", "HC90WW_rate", "CC_rate"]:
        plt.plot(fixed.epoch, fixed[col], label=f"Fixed {col}")
        plt.plot(adaptive.epoch, adaptive[col], linestyle="--", label=f"D {col}")
    plt.xlabel("Epoch"); plt.ylabel("rate"); plt.grid(alpha=.25); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(PLOTS / "shared_errors.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    for name, tr in [("Fixed", fixed_train), ("D-Adaptive", adaptive_train)]:
        plt.plot(tr.epoch, tr.mean_CPS_weight, label=f"{name} mean CPS weight")
    plt.plot(adaptive_train.epoch, adaptive_train.mean_contradiction_weight, label="D contradiction weight")
    plt.xlabel("Epoch"); plt.ylabel("weight"); plt.grid(alpha=.25); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / "weights.png", dpi=160); plt.close()

    # Use epoch 200 as the fixed, reproducible training-state weight summary.
    wr = adaptive_train.loc[adaptive_train.epoch == 200].iloc[0]
    qrows = []
    for q in range(1, 5):
        qrows.append({"quartile": f"Q{q}", "mean_D": float(wr[f"D_Q{q}_mean_D"]),
                      "mean_weight": float(wr[f"D_Q{q}_mean_weight"])})
    monotonic = all(qrows[i]["mean_D"] < qrows[i+1]["mean_D"] for i in range(3)) and \
        all(qrows[i]["mean_weight"] > qrows[i+1]["mean_weight"] for i in range(3))

    def vals(df, epoch):
        r = df.loc[df.epoch == epoch].iloc[0]
        return {k: float(r[k]) for k in ["Dice", "IoU", "WW_rate", "HC90WW_rate", "CC_rate", "Shared_FP_rate", "Shared_FN_rate"]}

    e200_f, e200_a = vals(fixed, 200), vals(adaptive, 200)
    bf, ba = vals(fixed, best_fixed_epoch), vals(adaptive, best_adaptive_epoch)
    # Outcome is decided at the matched, final 200-epoch checkpoint.  The
    # separate best-Dice rows are reported, but comparing different epochs
    # would confound the method with checkpoint selection.
    improved = (e200_a["Dice"] > e200_f["Dice"] and e200_a["IoU"] > e200_f["IoU"] and
                e200_a["WW_rate"] <= e200_f["WW_rate"] and e200_a["HC90WW_rate"] <= e200_f["HC90WW_rate"] and
                e200_a["CC_rate"] >= e200_f["CC_rate"])
    close = abs(e200_a["Dice"] - e200_f["Dice"]) < 0.002 and abs(e200_a["IoU"] - e200_f["IoU"]) < 0.002
    outcome = "Outcome A" if improved else ("Outcome B" if close else "Outcome C")
    conclusion = {
        "Outcome A": "D-dependent extra suppression improves Fixed Native-S.",
        "Outcome B": "D-dependent suppression is valid but provides little practical advantage over Fixed Native-S.",
        "Outcome C": "Stronger D-dependent suppression does not improve over the fixed 0.5 rule.",
    }[outcome]

    summary = {
        "implementation": {
            "mode": "native_suppress_d_adaptive",
            "formula": "w=1 if disagreement or D<=0; otherwise max(0,0.5-D)",
            "uni_input": [256, 256], "native_grid": [16, 16],
            "cdf": False, "running_buffer": False, "extra_hyperparameters": "none",
            "validation_used_for_weighting": False, "gt_used_for_weighting": False,
        },
        "formal_training": {"epochs": 200, "seed": 24},
        "best_epochs": {"fixed_dice": best_fixed_epoch, "d_adaptive_dice": best_adaptive_epoch},
        "fixed_best_dice": bf, "d_adaptive_best_dice": ba,
        "epoch_200": {"fixed": e200_f, "d_adaptive": e200_a},
        "outcome_basis": "matched epoch 200 (best-Dice rows are reported separately)",
        "d_weight_epoch_200": {"mean_D": float(wr.mean_D_contradiction),
            "D_p10": float(wr.D_p10), "D_p50": float(wr.D_p50), "D_p90": float(wr.D_p90),
            "mean_contradiction_weight": float(wr.mean_contradiction_weight),
            "weight_p10": float(wr.contradiction_weight_p10),
            "weight_p50": float(wr.contradiction_weight_p50),
            "weight_p90": float(wr.contradiction_weight_p90),
            "fraction_zero_weight": float(wr.fraction_zero_weight)},
        "D_quartiles_epoch_200": qrows,
        "D_to_weight_monotonic": bool(monotonic),
        "outcome": outcome,
        "conclusion": conclusion,
        "artifacts": {"training_log": "training_log.csv", "validation_metrics": "validation_metrics.csv",
                       "weight_statistics": "weight_statistics.csv", "comparison": "fixed_vs_d_adaptive_selected.csv"},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "config.json").write_text(json.dumps(summary["implementation"] | {"seed": 24, "num_epochs": 200}, indent=2) + "\n")
    readme = f"""# D-Adaptive Native-S\n\nFormal 200-epoch run, seed 24. No forward pass is performed by this postprocessor.\n\nFormula: `w=1` for disagreement or non-contradiction; `w=max(0, 0.5-D)` for `D>0` agreement tokens.\n\nOutcome: **{outcome}** — {conclusion}\n\nBest mean-Dice epochs: Fixed={best_fixed_epoch}, D-Adaptive={best_adaptive_epoch}.\n\nThe main comparison is `fixed_vs_d_adaptive_selected.csv`; raw training and evaluator exports are retained alongside it.\n"""
    (OUT / "README.md").write_text(readme)
    report = f"""# D-Adaptive Native-S 离线报告

## 实验

- 模式：`native_suppress_d_adaptive`
- 公式：无 contradiction 或 Student disagreement 时 `w=1`；agreement 且 `D>0` 时 `w=max(0, 0.5-D)`。
- 训练：seed 24，正式 200 epoch；Fixed Native-S 路径未修改。
- CDF、running buffer、percentile、额外超参数和 GT weighting：均未使用。
- 结果判定采用 matched epoch 200；best-Dice epoch 仅作为独立报告，避免不同 checkpoint 选择造成混淆。

## Epoch 200 对照

| 指标 | Fixed Native-S | D-Adaptive Native-S | Δ |
|---|---:|---:|---:|
| Dice | {e200_f['Dice']:.6f} | {e200_a['Dice']:.6f} | {e200_a['Dice']-e200_f['Dice']:+.6f} |
| IoU | {e200_f['IoU']:.6f} | {e200_a['IoU']:.6f} | {e200_a['IoU']-e200_f['IoU']:+.6f} |
| WW | {e200_f['WW_rate']:.6f} | {e200_a['WW_rate']:.6f} | {e200_a['WW_rate']-e200_f['WW_rate']:+.6f} |
| HC90WW | {e200_f['HC90WW_rate']:.6f} | {e200_a['HC90WW_rate']:.6f} | {e200_a['HC90WW_rate']-e200_f['HC90WW_rate']:+.6f} |
| CC | {e200_f['CC_rate']:.6f} | {e200_a['CC_rate']:.6f} | {e200_a['CC_rate']-e200_f['CC_rate']:+.6f} |

Fixed best mean-Dice epoch is **{best_fixed_epoch}** ({bf['Dice']:.6f}); D-Adaptive best mean-Dice epoch is **{best_adaptive_epoch}** ({ba['Dice']:.6f}).

## Weight diagnostic at epoch 200

`D` mean/p10/p50/p90 = `{wr.mean_D_contradiction:.6f} / {wr.D_p10:.6f} / {wr.D_p50:.6f} / {wr.D_p90:.6f}`.
Contradiction weight mean/p10/p50/p90 = `{wr.mean_contradiction_weight:.6f} / {wr.contradiction_weight_p10:.6f} / {wr.contradiction_weight_p50:.6f} / {wr.contradiction_weight_p90:.6f}`.
Zero-weight fraction = `{wr.fraction_zero_weight:.6f}`. D-to-weight quartile monotonicity = `{monotonic}`.

## 结论

**{outcome}** — {conclusion}

详细逐 epoch 数据见 `fixed_vs_d_adaptive.csv`、`training_log.csv`、`validation_metrics.csv` 和 `weight_statistics.csv`。
"""
    (OUT / "final_report.md").write_text(report)
    print("=== D-Adaptive Native-S ===")
    print("Implementation:")
    print("  mode: native_suppress_d_adaptive")
    print("  UNI input: 256x256")
    print("  native grid: 16x16")
    print("  formula: w=1 if no contradiction; max(0,0.5-D) otherwise")
    print("  CDF: false")
    print("  running buffer: false")
    print("  extra hyperparameters: none")
    print("Weight (epoch 200):")
    print(f"  mean D: {wr.mean_D_contradiction:.6f}")
    print(f"  D p10/p50/p90: {wr.D_p10:.6f} / {wr.D_p50:.6f} / {wr.D_p90:.6f}")
    print(f"  mean contradiction weight: {wr.mean_contradiction_weight:.6f}")
    print(f"  weight p10/p50/p90: {wr.contradiction_weight_p10:.6f} / {wr.contradiction_weight_p50:.6f} / {wr.contradiction_weight_p90:.6f}")
    print(f"  zero-weight fraction: {wr.fraction_zero_weight:.6f}")
    print("D quartiles (mean D / mean weight):")
    for q in qrows: print(f"  {q['quartile']}: {q['mean_D']:.6f} / {q['mean_weight']:.6f}")
    print("Fixed Native-S:", best_fixed_epoch, bf)
    print("D-Adaptive Native-S:", best_adaptive_epoch, ba)
    print("Outcome:", outcome, "-", conclusion)
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
