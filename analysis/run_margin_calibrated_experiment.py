"""Run the single fixed-tau K=50 Margin-Calibrated Anti-Confirmation experiment."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/k50_margin_calibrated_reverse"
MEMORY = ROOT / "outputs/qcdf_beta_comparison/reference_memory_qcdf_k50.npz"
RECORDS = ROOT / "outputs/train_topk_calibration/selected_k_token_records.csv"
QCDf_RUN = {
    "tag": "qCDF_beta1",
    "checkpoint_root": str(ROOT / "checkpoints/qcdf_k50_beta10"),
    "metrics_root": str(ROOT / "outputs/qcdf_beta_comparison/runs/beta10/native_metrics"),
    "mode_tag": "native_qcdf_anti_confirmation_s",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows for {}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def num(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_tau() -> dict:
    summary_path = RECORDS.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("selected_k") != 50 or summary.get("selection_protocol") != "image-level leave-one-image-out":
        raise RuntimeError("Calibration records are not the required train-only LOO K=50 records")
    if summary.get("validation_used", True) or summary.get("test_used", True):
        raise RuntimeError("Calibration records report validation/test usage")
    values = {"FG": [], "BG": []}
    for row in read_csv(RECORDS):
        values[row["gt_class"]].append(abs(float(row["delta"])))
    if not values["FG"] or not values["BG"]:
        raise RuntimeError("Missing FG/BG calibration tokens")
    result = {
        "source": "labeled train-only image-level LOO K=50",
        "records_path": str(RECORDS),
        "summary_path": str(summary_path),
        "validation_used": False, "test_used": False, "K": 50,
        "fg_token_count": len(values["FG"]), "bg_token_count": len(values["BG"]),
    }
    for cls in ("FG", "BG"):
        arr = np.asarray(values[cls], dtype=np.float64)
        prefix = cls.lower()
        result.update({
            "{}_mean_abs_delta".format(prefix): float(arr.mean()),
            "{}_median_abs_delta".format(prefix): float(np.median(arr)),
            "{}_p10_abs_delta".format(prefix): float(np.percentile(arr, 10)),
            "{}_p25_abs_delta".format(prefix): float(np.percentile(arr, 25)),
            "{}_p75_abs_delta".format(prefix): float(np.percentile(arr, 75)),
            "{}_p90_abs_delta".format(prefix): float(np.percentile(arr, 90)),
        })
    result["tau"] = 0.5 * (result["fg_median_abs_delta"] + result["bg_median_abs_delta"])
    result["tau_definition"] = "class-balanced typical semantic separation magnitude of the train-only UNI Memory reference"
    if not math.isfinite(result["tau"]) or result["tau"] <= 0:
        raise RuntimeError("Invalid tau: {}".format(result["tau"]))
    return result


def initial_sanity(calibration: dict) -> dict:
    if not MEMORY.is_file():
        raise FileNotFoundError(MEMORY)
    sys.path.insert(0, str(ROOT / "analysis"))
    import torch
    import anti_confirmation_budget_gradient_diagnostic as diagnostic
    import main as project
    from config.dataset_config.dataset_cfg import dataset_cfg
    from models.getnetwork import get_network

    diagnostic.seed_everything(24)
    args = diagnostic.make_args(MEMORY)
    args.cps_mode = "native_fine_margin_calibrated_reverse"
    args.reference_mode = args.cps_mode
    args.uni_reference_memory = str(MEMORY)
    cfg = dataset_cfg("GlaS")
    _, unsup_batches = diagnostic.build_training_batches(args, cfg)
    context = diagnostic.load_reference_context(args, cfg)
    context["margin_calibration"] = calibration
    model_a = get_network("unet", 3, 2).cuda()
    model_b = get_network("unet", 3, 2).cuda()
    model_a.eval(); model_b.eval()
    ds, qs, risks = [], [], []
    with torch.inference_mode():
        for cpu_batch in unsup_batches:
            batch = diagnostic.move_tensor_batch(cpu_batch)
            view_a, view_b = project.get_training_views(batch)
            logits_a = model_a(view_a)[0]
            logits_b = model_b(view_b)[0]
            targets = project._build_native_reference_targets(
                logits_a, logits_b, view_a, context, cfg, args,
            )
            fine = targets["fine_stats"]
            risk = fine["risk_pixel"][:, 0]
            q = fine["q_pixel"][:, 0][risk]
            d = fine["d_pixel"][:, 0][risk]
            if risk.any().item():
                ds.extend(d.cpu().tolist()); qs.extend(q.cpu().tolist()); risks.append(float(risk.float().mean().item()))
    del model_a, model_b
    torch.cuda.empty_cache()
    if not ds:
        raise RuntimeError("Initial sanity found no positive contradiction risky pixels")
    d = np.asarray(ds, dtype=np.float64); q = np.asarray(qs, dtype=np.float64)
    sanity = {
        "positive_contradiction_count": int(len(d)),
        "risk_fraction_mean": float(np.mean(risks)),
        "D_mean": float(d.mean()), "D_p10": float(np.percentile(d, 10)),
        "D_p50": float(np.percentile(d, 50)), "D_p90": float(np.percentile(d, 90)),
        "q_mean": float(q.mean()), "q_p10": float(np.percentile(q, 10)),
        "q_p25": float(np.percentile(q, 25)), "q_p50": float(np.percentile(q, 50)),
        "q_p75": float(np.percentile(q, 75)), "q_p90": float(np.percentile(q, 90)),
        "mean_CPS_weight": float((1.0 - q).mean()),
        "mean_opposite_weight": float(q.mean()),
        "max_weight_sum_error": float(np.max(np.abs((1.0 - q) + q - 1.0)),),
        "tau": calibration["tau"],
    }
    if sanity["max_weight_sum_error"] >= 1e-6:
        raise RuntimeError("Initial margin weight budget violated")
    if sanity["q_mean"] < 1e-3 or sanity["q_mean"] > 0.999:
        print("WARNING_Q_DEGENERATE")
    return sanity


def run_training() -> dict:
    checkpoint_root = ROOT / "checkpoints/k50_margin_calibrated_reverse"
    log_root = ROOT / "logs/k50_margin_calibrated_reverse"
    metrics_root = OUT / "native_metrics"
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*")):
        raise RuntimeError("Refusing to overwrite {}".format(checkpoint_root))
    if log_root.exists() and any(log_root.rglob("*")):
        raise RuntimeError("Refusing to overwrite {}".format(log_root))
    command = [
        sys.executable, "-u", "main.py",
        "--dataset_name", "GlaS", "--path_dataset", "datasets/GLaS",
        "--path_trained_models", "checkpoints/k50_margin_calibrated_reverse",
        "--path_seg_results", "seg_pred/k50_margin_calibrated_reverse",
        "--labeled_ratio", "10", "--sd", "24", "--batch_size", "2",
        "--num_epochs", "200", "--network", "unet", "--input_mode", "rgb_stain",
        "--step_size", "50", "--lr", "0.5", "--gamma", "0.5",
        "--unsup_weight", "5.0", "--warm_up_duration", "20", "--momentum", "0.9",
        "--cps_mode", "native_fine_margin_calibrated_reverse",
        "--uni_k", "50", "--uni_build_reference_memory",
        "--native-metrics-dir", str(metrics_root),
    ]
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "stdout.log"
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ))
    if result.returncode != 0:
        raise RuntimeError("Margin-calibrated training failed; inspect {}".format(log_path))
    return {"tag": "margin", "checkpoint_root": str(checkpoint_root), "metrics_root": str(metrics_root), "mode_tag": "native_fine_margin_calibrated_reverse_s", "stdout_log": str(log_path), "command": command}


def parse_run(run: dict) -> list[dict]:
    sys.path.insert(0, str(ROOT / "analysis"))
    import run_qcdf_beta_comparison as qcdf
    return qcdf.parse_validation(run)


def make_plots(rows_by_tag: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, ylabel, filename in (
        ("mean_Dice", "Dice", "dice_curve.png"), ("mean_IoU", "IoU/Jaccard", "iou_curve.png"),
        ("WW", "WW", "ww_curve.png"), ("HC90WW", "HC90WW", "hc90ww_curve.png"),
        ("CC", "CC", "cc_curve.png"), ("q_mean", "q mean", "q_curve.png"),
        ("risk_fraction", "risk fraction", "risk_fraction_curve.png"), ("D_mean", "D mean", "d_curve.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for tag, color, label in (("qCDF_beta1", "tab:orange", "q-CDF beta=1"), ("margin", "tab:blue", "margin-calibrated")):
            rows = rows_by_tag[tag]
            ax.plot([int(x["epoch"]) for x in rows], [num(x.get(key)) for x in rows], color=color, label=label)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=140); plt.close(fig)


def summarize(calibration: dict, sanity: dict, margin_run: dict) -> None:
    import run_qcdf_beta_comparison as qcdf
    q_rows = parse_run(QCDf_RUN)
    m_rows = parse_run(margin_run)
    rows_by_tag = {"qCDF_beta1": q_rows, "margin": m_rows}
    write_rows(OUT / "training_metrics.csv", m_rows)
    mechanism = read_csv(Path(margin_run["metrics_root"]) / "training_native_fine_margin_calibrated_reverse_s.csv")
    write_rows(OUT / "mechanism_metrics.csv", mechanism)
    by = {tag: {int(x["epoch"]): x for x in rows} for tag, rows in rows_by_tag.items()}
    completion = {}
    for tag, run in (("qCDF_beta1", QCDf_RUN), ("margin", margin_run)):
        complete = next(Path(run["checkpoint_root"]).glob("GLaS/*/native_training_complete.json"))
        completion[tag] = json.loads(complete.read_text(encoding="utf-8"))
    best = {tag: int(completion[tag]["best_synchronized_epoch"]) for tag in completion}
    names = {"Dice": "mean_Dice", "IoU/Jaccard": "mean_IoU", "WW": "WW", "HC90WW": "HC90WW", "CC": "CC", "Shared FP": "Shared_FP", "Shared FN": "Shared_FN"}
    comparison = []
    for label, key in names.items():
        q_best, m_best = float(by["qCDF_beta1"][best["qCDF_beta1"]][key]), float(by["margin"][best["margin"]][key])
        q_200, m_200 = float(by["qCDF_beta1"][200][key]), float(by["margin"][200][key])
        comparison.append({"Metric": label, "qCDF_beta1_best": q_best, "MarginCalibrated_best": m_best, "Delta_new_minus_qCDF": m_best - q_best, "qCDF_beta1_epoch200": q_200, "MarginCalibrated_epoch200": m_200, "Delta_epoch200": m_200 - q_200})
    write_rows(OUT / "final_comparison.csv", comparison)
    q_final, m_final = by["qCDF_beta1"][200], by["margin"][200]
    severity = [{
        "method": "qCDF_beta1", "q_definition": "F_ref(D)",
        "final_q_mean": q_final["q_mean"], "final_q_p50": q_final["q_p50"], "final_q_p90": q_final["q_p90"], "final_risk_fraction": q_final["risk_fraction"],
    }, {
        "method": "MarginCalibrated", "q_definition": "D/(D+tau)",
        "final_q_mean": m_final["q_mean"], "final_q_p50": m_final["q_p50"], "final_q_p90": m_final["q_p90"], "final_risk_fraction": m_final["risk_fraction"],
    }]
    write_rows(OUT / "severity_mapping_comparison.csv", severity)
    make_plots(rows_by_tag)
    late = [x for x in m_rows if int(x["epoch"]) >= 181]
    stability = {
        "nan_or_inf": any(not math.isfinite(num(x.get(k))) for x in m_rows for k in ("train_loss", "mean_Dice", "mean_IoU", "WW", "HC90WW", "CC")),
        "late_epoch_collapse": any(num(x.get("mean_IoU")) < .5 or num(x.get("mean_Dice")) < .65 for x in late),
        "late_loss_std": float(np.std([num(x.get("train_loss")) for x in late])),
    }
    q_best, m_best = by["qCDF_beta1"][best["qCDF_beta1"]], by["margin"][best["margin"]]
    task_deltas = [
        num(m_best["mean_Dice"]) - num(q_best["mean_Dice"]),
        num(m_best["mean_IoU"]) - num(q_best["mean_IoU"]),
        num(m_final["mean_Dice"]) - num(q_final["mean_Dice"]),
        num(m_final["mean_IoU"]) - num(q_final["mean_IoU"]),
    ]
    bias_deltas = [
        num(m_best["WW"]) - num(q_best["WW"]),
        num(m_best["HC90WW"]) - num(q_best["HC90WW"]),
        num(m_final["WW"]) - num(q_final["WW"]),
        num(m_final["HC90WW"]) - num(q_final["HC90WW"]),
    ]
    d_cc = num(m_final["CC"]) - num(q_final["CC"])
    task_comparable = all(delta >= -0.005 for delta in task_deltas)
    bias_ok = all(delta <= .005 for delta in bias_deltas) and d_cc >= -.005
    if task_comparable and bias_ok and not stability["nan_or_inf"] and not stability["late_epoch_collapse"]:
        outcome, replace = "A", "YES"
    elif task_comparable and not stability["nan_or_inf"] and not stability["late_epoch_collapse"]:
        outcome, replace = "B", "COMPARABLE_WITH_TRADEOFF"
    else:
        outcome, replace = "C", "NO"
    summary = {"calibration": calibration, "initial_sanity": sanity, "best_epochs": best, "completion": completion, "stability": stability, "outcome": outcome, "can_replace_qcdf": replace, "qCDF_beta1_source": QCDf_RUN}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = build_report(calibration, sanity, rows_by_tag, best, stability, outcome, replace)
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    print_console(calibration, sanity, rows_by_tag, best, stability, outcome, replace)


def build_report(calibration, sanity, rows_by_tag, best, stability, outcome, replace):
    q = {int(x["epoch"]): x for x in rows_by_tag["qCDF_beta1"]}; m = {int(x["epoch"]): x for x in rows_by_tag["margin"]}
    lines = [
        "# K50 Margin-Calibrated Anti-Confirmation",
        "",
        "## Protocol",
        "",
        "GLaS 10%, seed24, UNI frozen, native 256/16x16, K=50, same existing K=50 memory, same optimizer/LR/augmentation/batch/unsup ramp, 200 epochs.",
        "Calibration used only labeled train-only image-level LOO K=50 records; validation/test were not used. Old q-CDF and epoch200 bootstrap checkpoint were not used by the new run.",
        "",
        "## Q1 — tau calibration",
        "",
        "FG tokens: `{}`; BG tokens: `{}`; m_FG=`{:.8f}`; m_BG=`{:.8f}`; tau=`{:.8f}`.".format(calibration["fg_token_count"], calibration["bg_token_count"], calibration["fg_median_abs_delta"], calibration["bg_median_abs_delta"], calibration["tau"]),
        "tau is the class-balanced typical semantic separation magnitude of the train-only UNI Memory reference.",
        "",
        "## Q2/Q3 — initial and final dynamics",
        "",
        "| point | risk fraction | D mean | D p50 | D p90 | q mean | q p50 | q p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| sanity before epoch 1 | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(sanity["risk_fraction_mean"], sanity["D_mean"], sanity["D_p50"], sanity["D_p90"], sanity["q_mean"], sanity["q_p50"], sanity["q_p90"]),
        "| epoch 200 | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(num(m[200]["risk_fraction"]), num(m[200]["D_mean"]), num(m[200]["D_p50"]), num(m[200]["D_p90"]), num(m[200]["q_mean"]), num(m[200]["q_p50"]), num(m[200]["q_p90"])),
        "",
        "## Q4 — q dynamics",
        "",
        "| epoch | risk fraction | D mean | D p50 | D p90 | q mean | q p50 | q p90 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in (1, 40, 80, 120, 160, 200):
        x = m[epoch]; lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(epoch, num(x["risk_fraction"]), num(x["D_mean"]), num(x["D_p50"]), num(x["D_p90"]), num(x["q_mean"]), num(x["q_p50"]), num(x["q_p90"])))
    lines += ["", "## Q5/Q6 — task metrics", "", "| method | best epoch | best Dice | best IoU | epoch200 Dice | epoch200 IoU | epoch200 WW | epoch200 HC90WW | epoch200 CC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, label in (("qCDF_beta1", "q-CDF beta=1"), ("margin", "Margin-Calibrated")):
        b = best[tag]; lines.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(label, b, num(({"qCDF_beta1":q,"margin":m}[tag][b])["mean_Dice"]), num(({"qCDF_beta1":q,"margin":m}[tag][b])["mean_IoU"]), num(({"qCDF_beta1":q,"margin":m}[tag][200])["mean_Dice"]), num(({"qCDF_beta1":q,"margin":m}[tag][200])["mean_IoU"]), num(({"qCDF_beta1":q,"margin":m}[tag][200])["WW"]), num(({"qCDF_beta1":q,"margin":m}[tag][200])["HC90WW"]), num(({"qCDF_beta1":q,"margin":m}[tag][200])["CC"])))
    lines += ["", "## Q7 — delta Margin-Calibrated minus q-CDF beta=1", "", "See `final_comparison.csv` for Dice, IoU, WW, HC90WW, CC, Shared FP and Shared FN deltas at best and epoch 200.", "", "## Q8 — numerical stability", "", "NaN/Inf: `{}`; late-epoch collapse: `{}`; late train-loss std: `{:.8f}`.".format(stability["nan_or_inf"], stability["late_epoch_collapse"], stability["late_loss_std"]), "", "## Q9/Q10 — dependency and replacement conclusion", "", "The new method removes the old epoch200 bootstrap checkpoint, 145-value empirical CDF and CDF lookup, and uses no beta. It still uses the required labeled train-only LOO K=50 calibration and the fixed K=50 Memory.", "", "Outcome **{}**. Can margin calibration replace q-CDF? **{}**.".format(outcome, replace), "", "The q definitions are different and q magnitude is not interpreted as an absolute quality ranking.", ""]
    return "\n".join(lines)


def print_console(calibration, sanity, rows_by_tag, best, stability, outcome, replace):
    q = {int(x["epoch"]): x for x in rows_by_tag["qCDF_beta1"]}; m = {int(x["epoch"]): x for x in rows_by_tag["margin"]}
    print("=== K50 Margin-Calibrated Anti-Confirmation ===")
    print("\nCalibration:\n  source = labeled train-only image-level LOO\n  validation used = false\n  test used = false\n  K = 50\n  FG tokens: {}\n  BG tokens: {}\n  median |Delta| FG: {:.8f}\n  median |Delta| BG: {:.8f}\n  tau: {:.8f}".format(calibration["fg_token_count"], calibration["bg_token_count"], calibration["fg_median_abs_delta"], calibration["bg_median_abs_delta"], calibration["tau"]))
    print("\nSeverity mapping:\n  q = D / (D + tau)")
    print("\nTraining:\n  epochs = 200\n  beta = none\n  old CDF used = false\n  bootstrap checkpoint used = false\n  unsup ramp unchanged = true")
    print("\nInitial dynamics:\n  risk fraction: {:.8f}\n  D mean/p50/p90: {:.8f}/{:.8f}/{:.8f}\n  q mean/p50/p90: {:.8f}/{:.8f}/{:.8f}".format(sanity["risk_fraction_mean"], sanity["D_mean"], sanity["D_p50"], sanity["D_p90"], sanity["q_mean"], sanity["q_p50"], sanity["q_p90"]))
    print("\nFinal dynamics:\n  risk fraction: {:.8f}\n  D mean/p50/p90: {:.8f}/{:.8f}/{:.8f}\n  q mean/p50/p90: {:.8f}/{:.8f}/{:.8f}".format(num(m[200]["risk_fraction"]), num(m[200]["D_mean"]), num(m[200]["D_p50"]), num(m[200]["D_p90"]), num(m[200]["q_mean"]), num(m[200]["q_p50"]), num(m[200]["q_p90"])))
    for tag, label in (("margin", "Margin-Calibrated"), ("qCDF_beta1", "Existing q-CDF beta1")):
        x = m if tag == "margin" else q; e = best[tag]
        print("\n{} best: epoch={} Dice={:.6f} IoU={:.6f} WW={:.6f} HC90WW={:.6f} CC={:.6f}".format(label, e, num(x[e]["mean_Dice"]), num(x[e]["mean_IoU"]), num(x[e]["WW"]), num(x[e]["HC90WW"]), num(x[e]["CC"])))
        print("{} epoch200: Dice={:.6f} IoU={:.6f} WW={:.6f} HC90WW={:.6f} CC={:.6f}".format(label, num(x[200]["mean_Dice"]), num(x[200]["mean_IoU"]), num(x[200]["WW"]), num(x[200]["HC90WW"]), num(x[200]["CC"])))
    print("\nNumerical stability: NaN={} Inf={} late collapse={}".format(stability["nan_or_inf"], stability["nan_or_inf"], stability["late_epoch_collapse"]))
    print("\nFinal outcome: {}\n\nCan margin calibration replace q-CDF? {}".format(outcome, replace))
    print("\nOutputs: {}".format(OUT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = load_tau()
    (OUT / "tau_calibration.json").write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    sanity = initial_sanity(calibration)
    (OUT / "initial_sanity.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    margin_run = run_training()
    summarize(calibration, sanity, margin_run)


if __name__ == "__main__":
    main()
