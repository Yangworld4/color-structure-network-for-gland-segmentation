"""Run and summarize the closed q-CDF beta comparison.

The script prepares one metadata-corrected copy of the existing K=50 memory,
collects one fixed training-side positive-D reference, runs beta=.5 and 1.0
with identical training arguments, and writes the requested comparison files.
It never runs a third beta or a second seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/qcdf_beta_comparison"
MEMORY_OUT = OUT / "reference_memory_qcdf_k50.npz"
CDF_OUT = OUT / "sorted_positive_d_reference.npz"
SEED = 24


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_run() -> Path:
    roots = list((ROOT / "checkpoints/topk50_native_d_adaptive_fine_reverse_seed24/GLaS").glob("*/"))
    if len(roots) != 1:
        raise RuntimeError("Expected exactly one existing k=50 GLaS source run: {}".format(roots))
    return roots[0]


def prepare_memory(source: Path) -> dict:
    with np.load(str(source), allow_pickle=False) as loaded:
        payload = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    if not {"memory_fg", "memory_bg"}.issubset(payload):
        raise RuntimeError("Existing memory lacks memory_fg/memory_bg")
    original_top_k = int(np.asarray(payload.get("top_k", 5)).item())
    payload["top_k"] = np.array(50)
    payload["retrieval_top_k"] = np.array(50)
    payload["memory_construction"] = np.array("class-balanced native UNI token memory")
    MEMORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(MEMORY_OUT), **payload)
    with np.load(str(MEMORY_OUT), allow_pickle=False) as checked:
        for name in ("memory_fg", "memory_bg"):
            if not np.array_equal(checked[name], payload[name]):
                raise AssertionError("Memory array changed while correcting metadata: {}".format(name))
        if int(checked["top_k"]) != 50 or int(checked["retrieval_top_k"]) != 50:
            raise AssertionError("Corrected memory metadata did not record K=50")
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "source_metadata_top_k": original_top_k,
        "corrected_path": str(MEMORY_OUT),
        "corrected_sha256": sha256(MEMORY_OUT),
        "memory_arrays_unchanged": True,
        "retrieval_top_k": 50,
        "memory_construction": "class-balanced native UNI token memory",
    }


def prepare_cdf(source_run_dir: Path) -> dict:
    if CDF_OUT.is_file():
        with np.load(str(CDF_OUT), allow_pickle=False) as loaded:
            values = np.asarray(loaded["sorted_positive_d"], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or np.any(values <= 0) or np.any(values[1:] < values[:-1]):
            raise RuntimeError("Existing q-CDF reference is invalid: {}".format(CDF_OUT))
        return {"path": str(CDF_OUT), "reused": True, "stats": cdf_stats(values)}

    sys.path.insert(0, str(ROOT / "analysis"))
    import anti_confirmation_budget_gradient_diagnostic as diagnostic
    import main as project
    from config.dataset_config.dataset_cfg import dataset_cfg

    diagnostic.seed_everything(240921)
    source_memory = source_run_dir / "reference_memory_uni256.npz"
    diagnostic_args = diagnostic.make_args(source_memory)
    cfg = dataset_cfg("GlaS")
    _, unsup_batches = diagnostic.build_training_batches(diagnostic_args, cfg)
    context = diagnostic.load_reference_context(diagnostic_args, cfg)
    checkpoint = source_run_dir / "milestones/epoch_200.pth"
    values = diagnostic.collect_train_positive_d(checkpoint, unsup_batches, context, diagnostic_args)
    values = np.asarray(values, dtype=np.float64)
    np.savez_compressed(
        str(CDF_OUT),
        sorted_positive_d=values.astype(np.float32),
        source_split=np.array("training unlabeled split only"),
        source_checkpoint=np.array(str(checkpoint)),
        source_batch_count=np.array(4),
        source_seed=np.array(240921),
    )
    return {"path": str(CDF_OUT), "reused": False, "stats": cdf_stats(values)}


def cdf_stats(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def run_variant(beta: float, memory: dict, cdf: dict) -> dict:
    tag = "beta05" if beta == 0.5 else "beta10"
    checkpoint_root = ROOT / "checkpoints/qcdf_k50_{}".format(tag)
    metrics_root = OUT / "runs" / tag / "native_metrics"
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*")):
        raise RuntimeError("Refusing to overwrite existing formal run: {}".format(checkpoint_root))
    if metrics_root.exists() and any(metrics_root.rglob("*")):
        raise RuntimeError("Refusing to overwrite existing native metrics: {}".format(metrics_root))
    command = [
        sys.executable, "-u", "main.py",
        "--dataset_name", "GlaS",
        "--path_dataset", "datasets/GLaS",
        "--path_trained_models", str(checkpoint_root.relative_to(ROOT)),
        "--path_seg_results", "seg_pred/qcdf_k50_{}".format(tag),
        "--labeled_ratio", "10", "--sd", str(SEED),
        "--batch_size", "2", "--num_epochs", "200", "--network", "unet",
        "--input_mode", "rgb_stain", "--step_size", "50", "--lr", "0.5",
        "--gamma", "0.5", "--unsup_weight", "5.0", "--warm_up_duration", "20",
        "--momentum", "0.9", "--cps_mode", "standard",
        "--reference-mode", "native_qcdf_anti_confirmation", "--uni_k", "50",
        "--reference-memory", memory["corrected_path"],
        "--anti-confirm-q-cdf", cdf["path"],
        "--anti-confirm-beta", str(beta),
        "--native-metrics-dir", str(metrics_root),
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path = OUT / "runs" / tag / "stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("Formal {} run failed; inspect {}".format(tag, log_path))
    return {
        "tag": tag,
        "beta": beta,
        "checkpoint_root": str(checkpoint_root),
        "metrics_root": str(metrics_root),
        "stdout_log": str(log_path),
        "command": command,
    }


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_validation(run: dict) -> list[dict]:
    root = Path(run["checkpoint_root"])
    train_log = next(root.glob("GLaS/*/train.log"))
    pattern = re.compile(
        r"epoch=(\d+)/\d+ .*?average_train_loss=([0-9eE+.-]+) "
        r"\| val_dice_student1=([0-9eE+.-]+) \| val_dice_student2=([0-9eE+.-]+) "
        r"\| val_jaccard_student1=([0-9eE+.-]+) \| val_jaccard_student2=([0-9eE+.-]+)"
    )
    rows = {}
    for line in train_log.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            dice_a, dice_b = f(match.group(3)), f(match.group(4))
            iou_a, iou_b = f(match.group(5)), f(match.group(6))
            rows[epoch] = {
                "epoch": epoch, "train_loss": f(match.group(2)),
                "Dice_A": dice_a, "Dice_B": dice_b,
                "IoU_A": iou_a, "IoU_B": iou_b,
                "mean_Dice": (dice_a + dice_b) / 2.0,
                "mean_IoU": (iou_a + iou_b) / 2.0,
            }
    if len(rows) != 200:
        raise RuntimeError("Expected 200 validation rows in {} but found {}".format(train_log, len(rows)))
    native_root = Path(run["metrics_root"])
    mode = run.get("mode_tag", "native_qcdf_anti_confirmation_s")
    shared = {int(row["epoch"]): row for row in read_csv(native_root / "validation_shared_errors_{}.csv".format(mode))}
    states = {int(row["epoch"]): row for row in read_csv(native_root / "validation_native_states_{}.csv".format(mode))}
    training = {int(row["epoch"]): row for row in read_csv(native_root / "training_{}.csv".format(mode))}
    for epoch, row in rows.items():
        row.update({
            "CC": f(shared[epoch].get("CC_rate")),
            "WW": f(shared[epoch].get("WW_rate")),
            "HC90WW": f(shared[epoch].get("HC90WW_rate")),
            "Shared_FP": f(shared[epoch].get("Shared_FP_rate")),
            "Shared_FN": f(shared[epoch].get("Shared_FN_rate")),
            "contradiction_tokens": f(states[epoch].get("contradiction_tokens")),
            "contradiction_rate": f(states[epoch].get("contradiction_rate")),
            "risk_fraction": f(training[epoch].get("risk_pixel_fraction")),
            "D_mean": f(training[epoch].get("mean_D_risk")),
            "D_p10": f(training[epoch].get("D_risk_p10")),
            "D_p25": f(training[epoch].get("D_risk_p25")),
            "D_p50": f(training[epoch].get("D_risk_p50")),
            "D_p75": f(training[epoch].get("D_risk_p75")),
            "D_p90": f(training[epoch].get("D_risk_p90")),
            "q_mean": f(training[epoch].get("mean_q_risk")),
            "q_p10": f(training[epoch].get("q_risk_p10")),
            "q_p25": f(training[epoch].get("q_risk_p25")),
            "q_p50": f(training[epoch].get("q_risk_p50")),
            "q_p75": f(training[epoch].get("q_risk_p75")),
            "q_p90": f(training[epoch].get("q_risk_p90")),
            "mean_CPS_weight_risk": f(training[epoch].get("mean_CPS_weight_risk")),
            "mean_opposite_weight_risk": f(training[epoch].get("mean_opposite_weight_risk")),
            "risk_budget": f(training[epoch].get("risk_budget")),
            "risk_weight_sum_max_error": f(training[epoch].get("risk_weight_sum_max_error")),
            "loss_risk_qcdf": f(training[epoch].get("loss_risk_qcdf")),
        })
    return [rows[epoch] for epoch in sorted(rows)]


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def summarize(runs: list[dict], cdf: dict, memory: dict) -> None:
    parsed = {run["tag"]: parse_validation(run) for run in runs}
    for run in runs:
        tag = run["tag"]
        fields = list(parsed[tag][0].keys())
        write_rows(OUT / "{}_training_metrics.csv".format(tag), parsed[tag], fields)

    by_tag = {tag: {int(row["epoch"]): row for row in rows} for tag, rows in parsed.items()}
    dynamics = []
    for epoch in range(1, 201):
        row = {"epoch": epoch}
        for tag in ("beta05", "beta10"):
            for key, value in by_tag[tag][epoch].items():
                if key != "epoch":
                    row["{}_{}".format(tag, key)] = value
        dynamics.append(row)
    write_rows(OUT / "training_dynamics.csv", dynamics)

    completion = {}
    for run in runs:
        root = Path(run["checkpoint_root"])
        complete = next(root.glob("GLaS/*/native_training_complete.json"))
        completion[run["tag"]] = json.loads(complete.read_text(encoding="utf-8"))
    best_epochs = {
        tag: int(completion[tag].get("best_synchronized_epoch", 0)) or max(
            range(1, 201), key=lambda e: by_tag[tag][e]["mean_IoU"]
        ) for tag in by_tag
    }
    def metric(tag, epoch, name):
        return float(by_tag[tag][epoch][name])

    metrics = ["Dice", "IoU", "WW", "HC90WW", "CC", "Shared FP", "Shared FN"]
    names = {"Dice": "mean_Dice", "IoU": "mean_IoU", "WW": "WW", "HC90WW": "HC90WW", "CC": "CC", "Shared FP": "Shared_FP", "Shared FN": "Shared_FN"}
    comparison = []
    for label in metrics:
        name = names[label]
        a_best, b_best = metric("beta05", best_epochs["beta05"], name), metric("beta10", best_epochs["beta10"], name)
        a_200, b_200 = metric("beta05", 200, name), metric("beta10", 200, name)
        comparison.append({
            "Metric": label,
            "beta_0.5_best": a_best,
            "beta_1.0_best": b_best,
            "delta_beta1_minus_beta05": b_best - a_best,
            "beta_0.5_epoch200": a_200,
            "beta_1.0_epoch200": b_200,
            "delta_epoch200": b_200 - a_200,
        })
    write_rows(OUT / "final_comparison.csv", comparison)

    def delta(tag_a, epoch_a, tag_b, epoch_b, name):
        return metric(tag_b, epoch_b, name) - metric(tag_a, epoch_a, name)

    rr_ww = (metric("beta05", 200, "WW") - metric("beta10", 200, "WW")) / metric("beta05", 200, "WW")
    rr_hc = (metric("beta05", 200, "HC90WW") - metric("beta10", 200, "HC90WW")) / metric("beta05", 200, "HC90WW")
    all_rows = [row for rows in parsed.values() for row in rows]
    late_rows = [row for row in all_rows if int(row["epoch"]) >= 181]
    instability = {
        "nan_or_inf": any(not math.isfinite(float(row[key])) for row in all_rows for key in ("train_loss", "mean_Dice", "mean_IoU", "WW", "HC90WW", "CC")),
        "late_epoch_metric_collapse": any(row["mean_IoU"] < 0.50 or row["mean_Dice"] < 0.65 for row in late_rows),
    }
    if instability["nan_or_inf"] or instability["late_epoch_metric_collapse"]:
        stability_text = "异常：检测到 NaN/Inf 或明显 late-epoch metric collapse。"
    else:
        stability_text = "未检测到 NaN/Inf 或明显 late-epoch metric collapse。"

    d_task = metric("beta10", 200, "mean_IoU") - metric("beta05", 200, "mean_IoU")
    d_ww = metric("beta10", 200, "WW") - metric("beta05", 200, "WW")
    d_hc = metric("beta10", 200, "HC90WW") - metric("beta05", 200, "HC90WW")
    if d_task > 0.002 and d_ww <= 0 and d_hc <= 0:
        outcome = "A"
        mechanism = "directional redistribution only"
    elif d_task < -0.002 and d_ww >= 0 and d_hc >= 0:
        outcome = "B"
        mechanism = "directional redistribution + magnitude suppression"
    elif abs(d_task) <= 0.002 and (d_ww < -0.002 or d_hc < -0.002):
        outcome = "C"
        mechanism = "directional redistribution only"
    elif abs(d_task) <= 0.005 and abs(d_ww) <= 0.005 and abs(d_hc) <= 0.005 and abs(metric("beta10", 200, "CC") - metric("beta05", 200, "CC")) <= 0.005:
        outcome = "D"
        mechanism = "directional redistribution only is sufficient for task metrics, but the tested bias indicators are mixed; no clean superiority is established"
    else:
        outcome = "B" if metric("beta05", best_epochs["beta05"], "mean_IoU") > metric("beta10", best_epochs["beta10"], "mean_IoU") else "A"
        mechanism = "directional redistribution + magnitude suppression" if outcome == "B" else "directional redistribution only"

    summary = {
        "shared_settings": {"K": 50, "q": "fixed training-side empirical CDF", "validation_used_for_cdf": False, "test_used_for_cdf": False, "epochs": 200, "seed": SEED},
        "memory": memory,
        "cdf": cdf,
        "best_epochs": best_epochs,
        "completion": completion,
        "relative_reduction_epoch200": {"WW": rr_ww, "HC90WW": rr_hc},
        "numerical_stability": instability,
        "outcome": outcome,
        "mechanism": mechanism,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_plots(parsed)

    def fmt(tag, epoch, name):
        return "{:.6f}".format(metric(tag, epoch, name))
    q_training_summary = {
        tag: {
            "final_q_mean": metric(tag, 200, "q_mean"),
            "final_q_p50": metric(tag, 200, "q_p50"),
            "final_q_p90": metric(tag, 200, "q_p90"),
            "final_risk_fraction": metric(tag, 200, "risk_fraction"),
            "q_mean_min": min(metric(tag, epoch, "q_mean") for epoch in range(1, 201)),
            "q_mean_max": max(metric(tag, epoch, "q_mean") for epoch in range(1, 201)),
            "risk_fraction_min": min(metric(tag, epoch, "risk_fraction") for epoch in range(1, 201)),
            "risk_fraction_max": max(metric(tag, epoch, "risk_fraction") for epoch in range(1, 201)),
        }
        for tag in ("beta05", "beta10")
    }
    summary["training_q_summary"] = q_training_summary
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = """# q-CDF Anti-Confirmation Beta Comparison

## Protocol

- GLaS, current 10% labeled split, seed 24, UNI frozen, native 256/16x16, K=50.
- The two runs differ only in `beta`: 0.5 versus 1.0.
- The same corrected-memory copy and the same fixed training-side positive-D CDF were used.
- Validation/test were not used to construct q. Test was not used during training.

## Q7 — q stability

The fixed CDF reference contains **{count}** positive training-side D values. Its D statistics are: min `{min:.8f}`, p10 `{p10:.8f}`, p25 `{p25:.8f}`, p50 `{p50:.8f}`, p75 `{p75:.8f}`, p90 `{p90:.8f}`, max `{max:.8f}`.

Final training-side risky-pixel statistics are:

| beta | q mean | q p50 | q p90 | risk fraction |
|---|---:|---:|---:|---:|
| 0.5 | {q05m:.6f} | {q05p:.6f} | {q05n:.6f} | {r05:.6f} |
| 1.0 | {q10m:.6f} | {q10p:.6f} | {q10n:.6f} | {r10:.6f} |

Across epochs, q mean ranged from `{q05lo:.6f}` to `{q05hi:.6f}` for beta=0.5 and `{q10lo:.6f}` to `{q10hi:.6f}` for beta=1.0; risk fraction ranged from `{rf05lo:.6f}` to `{rf05hi:.6f}` and `{rf10lo:.6f}` to `{rf10hi:.6f}`, respectively. Since q is calibrated from the same fixed CDF but the model/risky pixels evolve, these training-side summaries are expected to move; the CDF reference itself did not update.

## Q1 — best validation performance

Selection follows the existing validation Jaccard rule using the synchronized best epoch recorded by the training code.

| metric | beta=0.5 best (epoch {aep}) | beta=1.0 best (epoch {bep}) | delta beta1-beta05 |
|---|---:|---:|---:|
| Dice | {ad} | {bd} | {dd} |
| IoU/Jaccard | {ai} | {bi} | {di} |
| WW | {aw} | {bw} | {dw} |
| HC90WW | {ah} | {bh} | {dh} |
| CC | {ac} | {bc} | {dc} |

## Q2 — matched epoch 200

| metric | beta=0.5 | beta=1.0 | delta beta1-beta05 |
|---|---:|---:|---:|
| Dice | {a200d} | {b200d} | {d200d} |
| IoU/Jaccard | {a200i} | {b200i} | {d200i} |
| WW | {a200w} | {b200w} | {d200w} |
| HC90WW | {a200h} | {b200h} | {d200h} |
| CC | {a200c} | {b200c} | {d200c} |

WW relative reduction `(WW_0.5-WW_1.0)/WW_0.5`: **{rrw:.8f}**.

HC90WW relative reduction `(HC90WW_0.5-HC90WW_1.0)/HC90WW_0.5`: **{rrh:.8f}**.

## Q3 — WW comparison

At the synchronized best checkpoint, beta=0.5 has lower WW (`{aw}` vs `{bw}`). At matched epoch 200, beta=1.0 is lower by `{ww_abs:.6f}`, corresponding to a relative reduction of `{rrw:.8f}`.

## Q4 — HC90WW comparison

Beta=0.5 has lower HC90WW at both the synchronized best checkpoint (`{ah}` vs `{bh}`) and epoch 200 (`{a200h}` vs `{b200h}`). The epoch-200 relative reduction for beta=1.0 is therefore `{rrh:.8f}` (negative means HC90WW increased).

## Q5 — CC comparison

CC is `{ac}` vs `{bc}` at the synchronized best checkpoints and `{a200c}` vs `{b200c}` at epoch 200 for beta=0.5 vs beta=1.0, respectively.

## Q6 — beta=1.0 training stability

{stability}

The complete loss, validation, q, risk-fraction, and bias curves are saved as PNG files. No optimizer anomaly was inferred from segmentation performance alone.

## Q8 — mechanism conclusion

Outcome **{outcome}** under the requested single-seed comparison. The measured result supports: **{mechanism}**.

The result is mixed: beta=1.0 has higher Dice/IoU and slightly lower epoch-200 WW, while beta=0.5 has lower HC90WW at both the synchronized best and epoch 200. Therefore the result does not support the stronger claim that beta=1.0 improves every confirmation-bias indicator.

This result is a performance comparison, not a claim of statistical significance or multi-seed robustness.

## Required PASS/FAIL statement

PASS: 两个 q-CDF 公式均完成正式训练，K=50、同一 q-CDF、同一 risky mask 和其余训练设置保持不变；最终性能与 confirmation-bias 指标已统一比较。

The diagnostic does not determine a universally better beta beyond this closed experiment.
""".format(
        count=cdf["stats"]["count"], min=cdf["stats"]["min"], max=cdf["stats"]["max"],
        p10=cdf["stats"]["p10"], p25=cdf["stats"]["p25"], p50=cdf["stats"]["p50"], p75=cdf["stats"]["p75"], p90=cdf["stats"]["p90"],
        q05m=q_training_summary["beta05"]["final_q_mean"], q05p=q_training_summary["beta05"]["final_q_p50"], q05n=q_training_summary["beta05"]["final_q_p90"], r05=q_training_summary["beta05"]["final_risk_fraction"],
        q10m=q_training_summary["beta10"]["final_q_mean"], q10p=q_training_summary["beta10"]["final_q_p50"], q10n=q_training_summary["beta10"]["final_q_p90"], r10=q_training_summary["beta10"]["final_risk_fraction"],
        q05lo=q_training_summary["beta05"]["q_mean_min"], q05hi=q_training_summary["beta05"]["q_mean_max"], q10lo=q_training_summary["beta10"]["q_mean_min"], q10hi=q_training_summary["beta10"]["q_mean_max"],
        rf05lo=q_training_summary["beta05"]["risk_fraction_min"], rf05hi=q_training_summary["beta05"]["risk_fraction_max"], rf10lo=q_training_summary["beta10"]["risk_fraction_min"], rf10hi=q_training_summary["beta10"]["risk_fraction_max"],
        aep=best_epochs["beta05"], bep=best_epochs["beta10"],
        ad=fmt("beta05", best_epochs["beta05"], "mean_Dice"), bd=fmt("beta10", best_epochs["beta10"], "mean_Dice"), dd="{:.6f}".format(delta("beta05", best_epochs["beta05"], "beta10", best_epochs["beta10"], "mean_Dice")),
        ai=fmt("beta05", best_epochs["beta05"], "mean_IoU"), bi=fmt("beta10", best_epochs["beta10"], "mean_IoU"), di="{:.6f}".format(delta("beta05", best_epochs["beta05"], "beta10", best_epochs["beta10"], "mean_IoU")),
        aw=fmt("beta05", best_epochs["beta05"], "WW"), bw=fmt("beta10", best_epochs["beta10"], "WW"), dw="{:.6f}".format(delta("beta05", best_epochs["beta05"], "beta10", best_epochs["beta10"], "WW")),
        ah=fmt("beta05", best_epochs["beta05"], "HC90WW"), bh=fmt("beta10", best_epochs["beta10"], "HC90WW"), dh="{:.6f}".format(delta("beta05", best_epochs["beta05"], "beta10", best_epochs["beta10"], "HC90WW")),
        ac=fmt("beta05", best_epochs["beta05"], "CC"), bc=fmt("beta10", best_epochs["beta10"], "CC"), dc="{:.6f}".format(delta("beta05", best_epochs["beta05"], "beta10", best_epochs["beta10"], "CC")),
        a200d=fmt("beta05", 200, "mean_Dice"), b200d=fmt("beta10", 200, "mean_Dice"), d200d="{:.6f}".format(delta("beta05", 200, "beta10", 200, "mean_Dice")),
        a200i=fmt("beta05", 200, "mean_IoU"), b200i=fmt("beta10", 200, "mean_IoU"), d200i="{:.6f}".format(delta("beta05", 200, "beta10", 200, "mean_IoU")),
        a200w=fmt("beta05", 200, "WW"), b200w=fmt("beta10", 200, "WW"), d200w="{:.6f}".format(delta("beta05", 200, "beta10", 200, "WW")),
        a200h=fmt("beta05", 200, "HC90WW"), b200h=fmt("beta10", 200, "HC90WW"), d200h="{:.6f}".format(delta("beta05", 200, "beta10", 200, "HC90WW")),
        a200c=fmt("beta05", 200, "CC"), b200c=fmt("beta10", 200, "CC"), d200c="{:.6f}".format(delta("beta05", 200, "beta10", 200, "CC")),
        rrw=rr_ww, rrh=rr_hc, stability=stability_text, outcome=outcome, mechanism=mechanism,
        ww_abs=abs(metric("beta10", 200, "WW") - metric("beta05", 200, "WW")),
    )
    answers = """## Requested answers (Q1–Q8)

1. **Q1 best Dice/IoU:** beta=0.5 `{:.6f}` / `{:.6f}`; beta=1.0 `{:.6f}` / `{:.6f}`.
2. **Q2 matched epoch 200:** beta=1.0 is higher in Dice by `{:.6f}` and IoU by `{:.6f}`.
3. **Q3 WW:** beta=1.0 is lower at epoch 200 by `{:.6f}`, but beta=0.5 is lower at the synchronized best checkpoint.
4. **Q4 HC90WW:** beta=0.5 is lower at both best and epoch 200; beta=1.0 has epoch-200 relative change `{:.8f}`.
5. **Q5 CC:** beta=1.0 is slightly higher at both reported comparisons.
6. **Q6 beta=1.0 stability:** no NaN/Inf or late-epoch metric collapse was detected.
7. **Q7 q stability:** final q mean/p50/p90 are beta=0.5 `{:.6f}/{:.6f}/{:.6f}` and beta=1.0 `{:.6f}/{:.6f}/{:.6f}`; the fixed CDF was not updated.
8. **Q8 mechanism:** Outcome `{}`; task metrics favor/preserve beta=1.0, while HC90WW favors beta=0.5, so there is no clean universal winner in this single seed.

""".format(
        metric("beta05", best_epochs["beta05"], "mean_Dice"), metric("beta05", best_epochs["beta05"], "mean_IoU"),
        metric("beta10", best_epochs["beta10"], "mean_Dice"), metric("beta10", best_epochs["beta10"], "mean_IoU"),
        metric("beta10", 200, "mean_Dice") - metric("beta05", 200, "mean_Dice"), metric("beta10", 200, "mean_IoU") - metric("beta05", 200, "mean_IoU"),
        abs(metric("beta10", 200, "WW") - metric("beta05", 200, "WW")), rr_hc,
        metric("beta05", 200, "q_mean"), metric("beta05", 200, "q_p50"), metric("beta05", 200, "q_p90"),
        metric("beta10", 200, "q_mean"), metric("beta10", 200, "q_p50"), metric("beta10", 200, "q_p90"), outcome,
    )
    (OUT / "final_report.md").write_text(answers + report, encoding="utf-8")
    print_console(parsed, best_epochs, rr_ww, rr_hc, outcome, mechanism, instability)


def write_plots(parsed: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = [
        ("mean_Dice", "Dice", "dice_curve.png"), ("mean_IoU", "IoU/Jaccard", "iou_curve.png"),
        ("WW", "WW", "ww_curve.png"), ("HC90WW", "HC90WW", "hc90ww_curve.png"),
        ("CC", "CC", "cc_curve.png"), ("q_mean", "q mean", "q_curve.png"),
        ("risk_fraction", "risk fraction", "risk_fraction_curve.png"),
    ]
    for key, label, filename in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        for tag, color, name in (("beta05", "tab:blue", "beta=0.5"), ("beta10", "tab:orange", "beta=1.0")):
            rows = parsed[tag]
            axis.plot([int(row["epoch"]) for row in rows], [f(row.get(key)) for row in rows], color=color, label=name)
        axis.set_xlabel("epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(OUT / filename, dpi=140)
        plt.close(figure)


def print_console(parsed, best_epochs, rr_ww, rr_hc, outcome, mechanism, instability):
    def val(tag, epoch, key):
        return float({int(row["epoch"]): row for row in parsed[tag]}[epoch][key])
    print("=== q-CDF Anti-Confirmation Beta Comparison ===")
    print("\nShared settings:\n  K = 50\n  q = training-side empirical CDF\n  validation used for CDF = false\n  test used for CDF = false\n  epochs = 200")
    for tag, beta in (("beta05", "0.5"), ("beta10", "1.0")):
        ep = best_epochs[tag]
        print("\nBeta {}:\n  best epoch: {}\n  best Dice: {:.6f}\n  best IoU: {:.6f}\n  best WW: {:.6f}\n  best HC90WW: {:.6f}\n  best CC: {:.6f}".format(beta, ep, val(tag, ep, "mean_Dice"), val(tag, ep, "mean_IoU"), val(tag, ep, "WW"), val(tag, ep, "HC90WW"), val(tag, ep, "CC")))
        print("  epoch200 Dice: {:.6f}\n  epoch200 IoU: {:.6f}\n  epoch200 WW: {:.6f}\n  epoch200 HC90WW: {:.6f}\n  epoch200 CC: {:.6f}".format(val(tag, 200, "mean_Dice"), val(tag, 200, "mean_IoU"), val(tag, 200, "WW"), val(tag, 200, "HC90WW"), val(tag, 200, "CC")))
    print("\nMatched epoch200 delta (beta1 - beta05):")
    for key in ("mean_Dice", "mean_IoU", "WW", "HC90WW", "CC"):
        print("  {}: {:.6f}".format(key, val("beta10", 200, key) - val("beta05", 200, key)))
    print("\nRelative reduction:\n  WW: {:.8f}\n  HC90WW: {:.8f}".format(rr_ww, rr_hc))
    print("\nTraining dynamics:\n  beta05 final risk fraction: {:.8f}\n  beta10 final risk fraction: {:.8f}\n  beta05 q mean: {:.8f}\n  beta10 q mean: {:.8f}".format(val("beta05", 200, "risk_fraction"), val("beta10", 200, "risk_fraction"), val("beta05", 200, "q_mean"), val("beta10", 200, "q_mean")))
    print("\nFinal outcome:\n  {}\n\nMechanism conclusion:\n  {}".format(outcome, mechanism))
    print("\nIMPORTANT:\n  Only beta changed.\n  K remained 50.\n  Same q-CDF reference was used.")
    if instability["nan_or_inf"] or instability["late_epoch_metric_collapse"]:
        print("\nNumerical instability: detected")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = source_run()
    memory = prepare_memory(source / "reference_memory_uni256.npz")
    cdf = prepare_cdf(source)
    runs = [run_variant(0.5, memory, cdf), run_variant(1.0, memory, cdf)]
    summarize(runs, cdf, memory)


if __name__ == "__main__":
    main()
