#!/usr/bin/env python3
"""Offline WW -> disagreement -> CC diagnostic.

This script loads existing CPS checkpoints, performs deterministic validation
inference, and never updates model parameters or writes checkpoints.  UNI
evidence is computed once per validation image and then reused to derive the
D-Adaptive contradiction cohort at each epoch.
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/data/mission/models/Exp/DUAL")
sys.path.insert(0, str(ROOT))
import main as project
from config.dataset_config.dataset_cfg import dataset_cfg
from models.getnetwork import get_network


OUT = ROOT / "outputs/ww_recovery_dynamic_chain"
PRED = OUT / "predictions"
PLOTS = OUT / "plots"
COMMON_CPS = ROOT / "checkpoints/semi_10_seed_24_rgbstain/GLaS" / (
    "CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-"
    "image-input_mode=rgb_stain"
) / "milestones"
ADAPTIVE_CPS = ROOT / "checkpoints/native_d_adaptive_s_seed24" / "GLaS" / (
    "CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-"
    "image-input_mode=rgb_stain-cps_mode=standard-"
    "reference_mode=native_suppress_d_adaptive"
) / "milestones"
MEMORY = ROOT / "checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz"
EPOCH_RE = re.compile(r"epoch_(\d+)\.pth$")
METHODS = {"Original_CPS": COMMON_CPS, "D_Adaptive": ADAPTIVE_CPS}
STATE_NAMES = {0: "WW", 1: "DIS", 2: "CC"}
STATE_CODE = {v: k for k, v in STATE_NAMES.items()}


def common_epochs() -> list[int]:
    found = {}
    for method, directory in METHODS.items():
        values = {}
        for path in directory.glob("epoch_*.pth"):
            m = EPOCH_RE.search(path.name)
            if m:
                values[int(m.group(1))] = path
        found[method] = values
    common = sorted(set(found["Original_CPS"]) & set(found["D_Adaptive"]))
    if not common:
        raise RuntimeError(f"No common epochs: {found}")
    required = [40, 80, 120, 160, 200]
    missing = [e for e in required if e not in common]
    if missing:
        raise RuntimeError(f"Required common epochs missing: {missing}; found={common}")
    return common


def checkpoint_for(method: str, epoch: int) -> Path:
    path = METHODS[method] / f"epoch_{epoch:03d}.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def make_args() -> SimpleNamespace:
    adaptive_path = checkpoint_for("D_Adaptive", 40)
    payload = torch.load(adaptive_path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**payload["args"])
    args.path_dataset = str(ROOT / args.path_dataset)
    args.uni_reference_memory = str(MEMORY)
    args.reference_mode = "native_suppress_d_adaptive"
    args.cps_mode = "standard"
    args.reference_resolution = 256
    args.uni_input_size = 256
    args.uni_k = 5
    args.native_hc_thresh = 0.9
    args.native_suppress_gamma = 0.5
    return args


def load_validation_batches(args, cfg):
    datasets, _manifest, _use_manifests = project.build_datasets(args, cfg)
    loader = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    records = []
    with torch.inference_mode():
        for batch in loader:
            rgb, stain = project.get_training_views(batch)
            records.append({
                "rgb": rgb.cpu().contiguous(),
                "stain": stain.cpu().contiguous(),
                "gt": batch["mask"].cpu().contiguous().long(),
                "ids": list(batch["ID"]),
            })
    if sum(x["gt"].shape[0] for x in records) != 17:
        raise AssertionError("Expected exactly 17 validation images")
    return records


def load_reference_context(args, cfg):
    logger = logging.getLogger("ww_recovery_reference")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    context = project._load_reference(args, cfg, logger, manifest_dir=None, use_manifests=False)
    context["cfg"] = cfg
    return context


def precompute_reference_scores(records, context, cfg, args):
    """Cache S_FG and S_BG once; they are image-only and epoch-independent."""
    result = []
    with torch.inference_mode():
        for rec in records:
            rgb = rec["rgb"].cuda(non_blocking=True)
            real_rgb = project._reconstruct_rgb(rgb, cfg)
            tokens = project._uni_dense_features(context["uni"], real_rgb, 256)
            fg, bg = project._reference_scores(tokens, context, args.uni_k)
            result.append({
                "fg": fg.reshape(-1, 16, 16).cpu(),
                "bg": bg.reshape(-1, 16, 16).cpu(),
            })
    return result


def build_models(cfg, payload):
    m1 = get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    m2 = get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    m1.load_state_dict(payload["appearance_state_dict"], strict=True)
    m2.load_state_dict(payload["structure_state_dict"], strict=True)
    m1.eval(); m2.eval()
    for model in (m1, m2):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return m1, m2


def state_codes(pred_a, pred_b, gt):
    valid = gt.ge(0)
    correct_a = pred_a.eq(gt)
    correct_b = pred_b.eq(gt)
    agree = pred_a.eq(pred_b)
    cc = valid & agree & correct_a & correct_b
    ww = valid & agree & ~correct_a & ~correct_b
    dis = valid & ~agree
    if not torch.equal(cc | ww | dis, valid):
        raise AssertionError("CC/WW/DIS do not partition valid pixels")
    state = torch.full_like(gt, fill_value=-1, dtype=torch.uint8)
    state[ww] = STATE_CODE["WW"]
    state[dis] = STATE_CODE["DIS"]
    state[cc] = STATE_CODE["CC"]
    return state, correct_a, correct_b


def infer_epoch(method, epoch, records, ref_scores, args, cfg):
    path = checkpoint_for(method, epoch)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    models = build_models(cfg, payload)
    pred_a, pred_b, gts, d_pixels, c_pixels, ids = [], [], [], [], [], []
    with torch.inference_mode():
        for index, rec in enumerate(records):
            rgb = rec["rgb"].cuda(non_blocking=True)
            stain = rec["stain"].cuda(non_blocking=True)
            gt = rec["gt"].cuda(non_blocking=True)
            logits_a = models[0](rgb)[0]
            logits_b = models[1](stain)[0]
            pa = project._foreground_probability(logits_a)[:, 0]
            pb = project._foreground_probability(logits_b)[:, 0]
            hard_a = pa.ge(0.5)
            hard_b = pb.ge(0.5)
            pred_a.append(hard_a.cpu().numpy().astype(np.uint8))
            pred_b.append(hard_b.cpu().numpy().astype(np.uint8))
            gts.append(gt.cpu().numpy().astype(np.uint8))
            ids.extend(rec["ids"])

            if method == "D_Adaptive":
                fg = ref_scores[index]["fg"].cuda(non_blocking=True)
                bg = ref_scores[index]["bg"].cuda(non_blocking=True)
                pa_tok = F.adaptive_avg_pool2d(pa.unsqueeze(1), (16, 16))[:, 0]
                pb_tok = F.adaptive_avg_pool2d(pb.unsqueeze(1), (16, 16))[:, 0]
                ha_tok = pa_tok.ge(0.5)
                hb_tok = pb_tok.ge(0.5)
                agree = ha_tok.eq(hb_tok)
                s_cur = torch.where(ha_tok, fg, bg)
                s_opp = torch.where(ha_tok, bg, fg)
                d = (s_opp - s_cur).clamp_min(0.0)
                contradiction = agree & d.gt(0.0)
                dp = F.interpolate(d.unsqueeze(1), size=gt.shape[-2:], mode="nearest")[:, 0]
                cp = F.interpolate(contradiction.float().unsqueeze(1), size=gt.shape[-2:], mode="nearest")[:, 0].bool()
                d_pixels.append(dp.cpu().numpy().astype(np.float32))
                c_pixels.append(cp.cpu().numpy().astype(np.uint8))
            else:
                d_pixels.append(np.zeros_like(pred_a[-1], dtype=np.float32))
                c_pixels.append(np.zeros_like(pred_a[-1], dtype=np.uint8))
    del models
    torch.cuda.empty_cache()
    arrays = {
        "pred_a": np.concatenate(pred_a), "pred_b": np.concatenate(pred_b),
        "gt": np.concatenate(gts), "D_pixel": np.concatenate(d_pixels),
        "contradiction_pixel": np.concatenate(c_pixels), "image_ids": np.asarray(ids),
    }
    return arrays


def save_predictions(method, epoch, arrays):
    directory = PRED / method
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"epoch_{epoch:03d}.npz", **arrays)


def global_row(method, epoch, arrays):
    pa, pb, gt = arrays["pred_a"].astype(bool), arrays["pred_b"].astype(bool), arrays["gt"].astype(bool)
    state, ca, cb = state_codes(torch.from_numpy(pa), torch.from_numpy(pb), torch.from_numpy(gt))
    valid = state.ge(0)
    n = int(valid.sum())
    row = {"method": method, "epoch": epoch, "valid_pixels": n}
    for name, code in STATE_CODE.items():
        row[f"{name}_rate"] = float(((state == code) & valid).sum().item() / n)
    dis = state == STATE_CODE["DIS"]
    row["A_accuracy"] = float((ca & valid).sum().item() / n)
    row["B_accuracy"] = float((cb & valid).sum().item() / n)
    row["DIS_A_correct_fraction"] = float((dis & ca).sum().item() / max(1, dis.sum().item()))
    row["DIS_B_correct_fraction"] = float((dis & cb).sum().item() / max(1, dis.sum().item()))
    if abs(sum(row[f"{x}_rate"] for x in STATE_NAMES.values()) - 1.0) > 1e-7:
        raise AssertionError("CC+WW+DIS != 1")
    arrays["state"] = state.numpy()
    arrays["correct_a"] = ca.numpy()
    arrays["correct_b"] = cb.numpy()
    return row


def conditional_transition(src, dst, source_code):
    mask = src == source_code
    count = int(mask.sum())
    out = {"source_count": count}
    for name, code in STATE_CODE.items():
        out[name] = float(((mask & (dst == code)).sum() / max(1, count)))
        out[f"{name}_count"] = int((mask & (dst == code)).sum())
    return out


def transition_rows(method, epochs, arrays_by_epoch):
    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        src, dst = arrays_by_epoch[e0]["state"], arrays_by_epoch[e1]["state"]
        row = {"method": method, "epoch_from": e0, "epoch_to": e1}
        for source_name, source_code in STATE_CODE.items():
            vals = conditional_transition(src, dst, source_code)
            row[f"{source_name}_source_count"] = vals["source_count"]
            for target_name in STATE_CODE:
                row[f"{source_name}_to_{target_name}"] = vals[target_name]
                row[f"{source_name}_to_{target_name}_count"] = vals[f"{target_name}_count"]
        rows.append(row)
    return rows


def chain_rows(method, epochs, arrays_by_epoch):
    rows = []
    for e0, e1, e2 in zip(epochs[:-2], epochs[1:-1], epochs[2:]):
        s0, s1, s2 = (arrays_by_epoch[e]["state"] for e in (e0, e1, e2))
        base = s0 == STATE_CODE["WW"]
        row = {"method": method, "epoch_t": e0, "epoch_t1": e1, "epoch_t2": e2,
               "WW_source_count": int(base.sum())}
        for name, cond in {
            "WW_DIS_CC": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["CC"]),
            "WW_DIS_DIS": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["DIS"]),
            "WW_DIS_WW": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["WW"]),
        }.items():
            row[f"{name}_count"] = int((base & cond).sum())
            row[f"{name}_rate"] = float((base & cond).sum() / max(1, base.sum()))
        rows.append(row)
    return rows


def targeted_rows(epochs, adaptive, cps):
    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        target = (adaptive[e0]["state"] == STATE_CODE["WW"]) & adaptive[e0]["contradiction_pixel"].astype(bool)
        for method, arrays in [("Adaptive", adaptive), ("CPS", cps)]:
            dst = arrays[e1]["state"]
            count = int(target.sum())
            vals = {"method": method, "epoch_from": e0, "epoch_to": e1,
                    "TargetWW_source_count": count}
            for name, code in STATE_CODE.items():
                vals[f"TargetWW_to_{name}"] = float((target & (dst == code)).sum() / max(1, count))
                vals[f"TargetWW_to_{name}_count"] = int((target & (dst == code)).sum())
            rows.append(vals)
    return rows


def matched_rows(epochs, adaptive, cps):
    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        target = ((adaptive[e0]["state"] == STATE_CODE["WW"]) &
                  (cps[e0]["state"] == STATE_CODE["WW"]) &
                  adaptive[e0]["contradiction_pixel"].astype(bool))
        row = {"epoch_from": e0, "epoch_to": e1, "N_matched_WW": int(target.sum())}
        for prefix, arrays in [("CPS", cps), ("Adaptive", adaptive)]:
            dst = arrays[e1]["state"]
            for name, code in STATE_CODE.items():
                row[f"{prefix}_WW_to_{name}"] = float((target & (dst == code)).sum() / max(1, target.sum()))
                row[f"{prefix}_WW_to_{name}_count"] = int((target & (dst == code)).sum())
        for name in STATE_NAMES.values():
            row[f"Delta_WW_to_{name}"] = row[f"Adaptive_WW_to_{name}"] - row[f"CPS_WW_to_{name}"]
        rows.append(row)
    # Three-step matched chains use the same starting cohort.
    for e0, e1, e2 in zip(epochs[:-2], epochs[1:-1], epochs[2:]):
        target = ((adaptive[e0]["state"] == STATE_CODE["WW"]) &
                  (cps[e0]["state"] == STATE_CODE["WW"]) &
                  adaptive[e0]["contradiction_pixel"].astype(bool))
        row = {"epoch_from": e0, "epoch_to": e1, "epoch_to2": e2,
               "N_matched_WW": int(target.sum())}
        for prefix, arrays in [("CPS", cps), ("Adaptive", adaptive)]:
            s1, s2 = arrays[e1]["state"], arrays[e2]["state"]
            for name, cond in {
                "WW_DIS_CC": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["CC"]),
                "WW_DIS_DIS": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["DIS"]),
                "WW_DIS_WW": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["WW"]),
            }.items():
                row[f"{prefix}_{name}"] = float((target & cond).sum() / max(1, target.sum()))
                row[f"{prefix}_{name}_count"] = int((target & cond).sum())
        for name in ("WW_DIS_CC", "WW_DIS_DIS", "WW_DIS_WW"):
            row[f"Delta_{name}"] = row[f"Adaptive_{name}"] - row[f"CPS_{name}"]
        rows.append(row)
    return rows


def targeted_chain_rows(epochs, adaptive):
    rows = []
    for e0, e1, e2 in zip(epochs[:-2], epochs[1:-1], epochs[2:]):
        target = ((adaptive[e0]["state"] == STATE_CODE["WW"]) &
                  adaptive[e0]["contradiction_pixel"].astype(bool))
        s1, s2 = adaptive[e1]["state"], adaptive[e2]["state"]
        row = {"epoch_from": e0, "epoch_to": e1, "epoch_to2": e2,
               "TargetWW_source_count": int(target.sum())}
        for name, cond in {
            "WW_DIS_CC": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["CC"]),
            "WW_DIS_DIS": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["DIS"]),
            "WW_DIS_WW": (s1 == STATE_CODE["DIS"]) & (s2 == STATE_CODE["WW"]),
        }.items():
            row[f"{name}"] = float((target & cond).sum() / max(1, target.sum()))
            row[f"{name}_count"] = int((target & cond).sum())
        rows.append(row)
    return rows


def d_quartile_rows(epochs, adaptive):
    rows = []
    for e0, e1, e2 in zip(epochs[:-2], epochs[1:-1], epochs[2:]):
        d = adaptive[e0]["D_pixel"]
        target = (adaptive[e0]["state"] == STATE_CODE["WW"]) & adaptive[e0]["contradiction_pixel"].astype(bool)
        values = d[target]
        if not values.size:
            continue
        edges = np.quantile(values, [0.25, .5, .75])
        groups = [values <= edges[0], (values > edges[0]) & (values <= edges[1]),
                  (values > edges[1]) & (values <= edges[2]), values > edges[2]]
        s1, s2 = adaptive[e1]["state"], adaptive[e2]["state"]
        target_flat = np.flatnonzero(target)
        for q, group in enumerate(groups, 1):
            ids = target_flat[group]
            if not len(ids):
                continue
            row = {"epoch_from": e0, "epoch_to": e1, "epoch_to2": e2,
                   "quartile": f"Q{q}", "N": int(len(ids)),
                   "mean_D": float(d.reshape(-1)[ids].mean()),
                   "mean_weight": float((0.5 - d.reshape(-1)[ids]).clip(min=0).mean())}
            st1, st2 = s1.reshape(-1)[ids], s2.reshape(-1)[ids]
            for name, cond in {"WW_DIS_CC": (st1 == STATE_CODE["DIS"]) & (st2 == STATE_CODE["CC"]),
                               "WW_DIS_DIS": (st1 == STATE_CODE["DIS"]) & (st2 == STATE_CODE["DIS"]),
                               "WW_DIS_WW": (st1 == STATE_CODE["DIS"]) & (st2 == STATE_CODE["WW"]),
                               "WW_to_WW": st1 == STATE_CODE["WW"],
                               "WW_to_DIS": st1 == STATE_CODE["DIS"],
                               "WW_to_CC": st1 == STATE_CODE["CC"]}.items():
                row[name] = float(cond.sum() / len(ids))
            rows.append(row)
    return rows


def disagreement_balance(method, epochs, arrays):
    rows = []
    for e in epochs:
        a = arrays[e]
        dis = a["state"] == STATE_CODE["DIS"]
        n = int(dis.sum())
        rows.append({"method": method, "epoch": e, "DIS_count": n,
                     "A_correct_B_wrong": int((dis & a["correct_a"]).sum()),
                     "B_correct_A_wrong": int((dis & a["correct_b"]).sum()),
                     "R_A": float((dis & a["correct_a"]).sum() / max(1, n)),
                     "R_B": float((dis & a["correct_b"]).sum() / max(1, n))})
    return rows


def aggregate_dict(df, method, column):
    part = df[df.method == method]
    return {str(int(e)): float(part.loc[part.epoch == e, column].iloc[0]) for e in part.epoch}


def make_plots(global_df, transition_df, chain_df):
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for method, g in global_df.groupby("method"):
        plt.plot(g.epoch, g.DIS_rate, marker="o", label=method)
    plt.xlabel("epoch"); plt.ylabel("DIS rate"); plt.grid(alpha=.25); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / "disagreement_rate.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    for method, g in global_df.groupby("method"):
        plt.plot(g.epoch, g.WW_rate, marker="o", label=method)
    plt.xlabel("epoch"); plt.ylabel("WW rate"); plt.grid(alpha=.25); plt.legend()
    plt.tight_layout(); plt.savefig(PLOTS / "ww_rate.png", dpi=160); plt.close()

    plt.figure(figsize=(10, 5))
    for method, g in transition_df.groupby("method"):
        plt.plot(g.epoch_from, g.WW_to_WW, marker="o", label=f"{method} WW→WW")
        plt.plot(g.epoch_from, g.WW_to_DIS, marker="o", linestyle="--", label=f"{method} WW→DIS")
        plt.plot(g.epoch_from, g.WW_to_CC, marker="o", linestyle=":", label=f"{method} WW→CC")
    plt.xlabel("epoch from"); plt.ylabel("conditional rate"); plt.grid(alpha=.25); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(PLOTS / "ww_transition.png", dpi=160); plt.close()

    plt.figure(figsize=(10, 5))
    for method, g in chain_df.groupby("method"):
        for col in ["WW_DIS_CC_rate", "WW_DIS_DIS_rate", "WW_DIS_WW_rate"]:
            plt.plot(g.epoch_t, g[col], marker="o", label=f"{method} {col.replace('_rate','')}")
    plt.xlabel("epoch t"); plt.ylabel("rate / WW_t"); plt.grid(alpha=.25); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(PLOTS / "three_step_chain.png", dpi=160); plt.close()


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    project.init_seeds(24)
    epochs = common_epochs()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    cfg = dataset_cfg("GlaS")
    args = make_args()
    records = load_validation_batches(args, cfg)
    context = load_reference_context(args, cfg)
    ref_scores = precompute_reference_scores(records, context, cfg, args)
    context["uni"] = context["uni"].cpu() if False else context["uni"]

    arrays = {"Original_CPS": {}, "D_Adaptive": {}}
    global_rows = []
    for method in METHODS:
        for epoch in epochs:
            archive = PRED / method / f"epoch_{epoch:03d}.npz"
            if archive.is_file():
                with np.load(archive, allow_pickle=False) as cached:
                    arr = {key: cached[key] for key in cached.files}
                print(f"Load cached {method} epoch {epoch}", flush=True)
            else:
                print(f"Infer {method} epoch {epoch} ...", flush=True)
                arr = infer_epoch(method, epoch, records, ref_scores, args, cfg)
                save_predictions(method, epoch, arr)
            global_rows.append(global_row(method, epoch, arr))
            arrays[method][epoch] = arr
    # State labels and correct masks are not needed in per-prediction archives;
    # keep them in memory for the transition analysis.
    for method in arrays:
        for epoch in epochs:
            global_row(method, epoch, arrays[method][epoch])

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(OUT / "global_state_rates.csv", index=False)
    transition_rows_all = []
    chain_rows_all = []
    for method in METHODS:
        transition_rows_all.extend(transition_rows(method, epochs, arrays[method]))
        chain_rows_all.extend(chain_rows(method, epochs, arrays[method]))
    transition_df = pd.DataFrame(transition_rows_all)
    chain_df = pd.DataFrame(chain_rows_all)
    transition_df.to_csv(OUT / "transition_matrix.csv", index=False)
    chain_df.to_csv(OUT / "three_step_chains.csv", index=False)

    # Also materialize the milestone-spaced chains explicitly requested in the
    # diagnostic narrative, while retaining the denser 10-epoch chains above.
    selected_epochs = [40, 80, 120, 160, 200]
    selected_transitions = []
    selected_chains = []
    for method in METHODS:
        selected_transitions.extend(transition_rows(method, selected_epochs, arrays[method]))
        selected_chains.extend(chain_rows(method, selected_epochs, arrays[method]))
    pd.DataFrame(selected_transitions).to_csv(OUT / "selected_transition_matrix.csv", index=False)
    pd.DataFrame(selected_chains).to_csv(OUT / "selected_three_step_chains.csv", index=False)

    cps, adaptive = arrays["Original_CPS"], arrays["D_Adaptive"]
    targeted_df = pd.DataFrame(targeted_rows(epochs, adaptive, cps))
    targeted_df.to_csv(OUT / "targeted_ww_transitions.csv", index=False)
    matched_data = matched_rows(epochs, adaptive, cps)
    matched_one_step = [row for row in matched_data if "epoch_to2" not in row]
    matched_chains = [row for row in matched_data if "epoch_to2" in row]
    pd.DataFrame(matched_one_step).to_csv(OUT / "matched_ww_transitions.csv", index=False)
    pd.DataFrame(matched_chains).to_csv(OUT / "matched_ww_chains.csv", index=False)
    pd.DataFrame(targeted_chain_rows(epochs, adaptive)).to_csv(OUT / "targeted_ww_chains.csv", index=False)
    selected_matched = matched_rows(selected_epochs, adaptive, cps)
    pd.DataFrame([r for r in selected_matched if "epoch_to2" in r]).to_csv(
        OUT / "selected_matched_ww_chains.csv", index=False
    )
    pd.DataFrame(targeted_chain_rows(selected_epochs, adaptive)).to_csv(
        OUT / "selected_targeted_ww_chains.csv", index=False
    )
    dquart_df = pd.DataFrame(d_quartile_rows(epochs, adaptive))
    dquart_df.to_csv(OUT / "adaptive_d_quartile_recovery.csv", index=False)
    balance_df = pd.DataFrame(
        disagreement_balance("Original_CPS", epochs, cps) + disagreement_balance("D_Adaptive", epochs, adaptive)
    )
    balance_df.to_csv(OUT / "disagreement_balance.csv", index=False)

    make_plots(global_df, transition_df, chain_df)

    g_cps = global_df[global_df.method == "Original_CPS"].set_index("epoch")
    g_ad = global_df[global_df.method == "D_Adaptive"].set_index("epoch")
    t_cps = transition_df[transition_df.method == "Original_CPS"].set_index("epoch_from")
    t_ad = transition_df[transition_df.method == "D_Adaptive"].set_index("epoch_from")
    ch_cps = chain_df[chain_df.method == "Original_CPS"].set_index("epoch_t")
    ch_ad = chain_df[chain_df.method == "D_Adaptive"].set_index("epoch_t")

    def transition_summary(table, col):
        return {str(int(e)): float(table.loc[e, col]) for e in table.index}

    dis_delta = {str(e): float(g_ad.loc[e, "DIS_rate"] - g_cps.loc[e, "DIS_rate"]) for e in epochs}
    late = {str(e): {"cps": float(g_cps.loc[e, "DIS_rate"]), "adaptive": float(g_ad.loc[e, "DIS_rate"]),
                     "delta": dis_delta[str(e)]} for e in (160, 200)}
    delta_ww_dis = t_ad.WW_to_DIS - t_cps.WW_to_DIS
    delta_ww_cc = t_ad.WW_to_CC - t_cps.WW_to_CC
    delta_chain_cc = ch_ad.WW_DIS_CC_rate - ch_cps.WW_DIS_CC_rate
    delta_persist_dis = t_ad.DIS_to_DIS - t_cps.DIS_to_DIS
    delta_cc_damage = t_ad.CC_to_DIS - t_cps.CC_to_DIS

    # Qualitative registered outcome: first verify escape and full recovery,
    # then distinguish productive from persistent disagreement.
    escape_gain = float(delta_ww_dis.mean())
    chain_gain = float(delta_chain_cc.mean())
    persistent_gain = float(delta_persist_dis.mean())
    late_delta = max(late["160"]["delta"], late["200"]["delta"])
    cc_damage_gain = float(delta_cc_damage.mean())
    if escape_gain > 0 and chain_gain > 0 and persistent_gain <= 0.02 and late_delta <= 0.03 and cc_damage_gain <= 0.02:
        outcome = "Outcome A"
        mechanism = "Adaptive suppression creates productive transient disagreement."
    elif escape_gain > 0 and (chain_gain <= 0 or persistent_gain > 0.02 or late_delta > 0.03):
        outcome = "Outcome B"
        mechanism = "Adaptive suppression breaks wrong consensus but produces excessive persistent disagreement."
    else:
        outcome = "Outcome C"
        mechanism = "Adaptive suppression does not produce a beneficial WW recovery pathway."

    matched_summary = [{k: v for k, v in row.items() if not k.endswith("_count")}
                       for row in matched_one_step]
    matched_chain_summary = [{k: v for k, v in row.items() if not k.endswith("_count")}
                             for row in matched_chains]
    targeted_chain_summary = [{k: v for k, v in row.items() if not k.endswith("_count")}
                              for row in targeted_chain_rows(epochs, adaptive)]
    summary = {
        "common_epochs": epochs,
        "validation_images": 17,
        "global_disagreement": {
            "cps": {str(e): float(g_cps.loc[e, "DIS_rate"]) for e in epochs},
            "adaptive": {str(e): float(g_ad.loc[e, "DIS_rate"]) for e in epochs},
            "adaptive_minus_cps": dis_delta,
        },
        "ww_transition": {
            "cps": {k: transition_summary(t_cps, k) for k in ["WW_to_WW", "WW_to_DIS", "WW_to_CC"]},
            "adaptive": {k: transition_summary(t_ad, k) for k in ["WW_to_WW", "WW_to_DIS", "WW_to_CC"]},
        },
        "three_step_chain": {
            "cps_WW_DIS_CC": transition_summary(ch_cps, "WW_DIS_CC_rate"),
            "adaptive_WW_DIS_CC": transition_summary(ch_ad, "WW_DIS_CC_rate"),
            "cps_WW_DIS_DIS": transition_summary(ch_cps, "WW_DIS_DIS_rate"),
            "adaptive_WW_DIS_DIS": transition_summary(ch_ad, "WW_DIS_DIS_rate"),
        },
        "matched_ww": matched_summary,
        "matched_ww_chain": matched_chain_summary,
        "targeted_ww_chain": targeted_chain_summary,
        "late_stage": late,
        "cc_damage": {
            "cps_CC_to_DIS": transition_summary(t_cps, "CC_to_DIS"),
            "adaptive_CC_to_DIS": transition_summary(t_ad, "CC_to_DIS"),
        },
        "disagreement_resolution": {
            "cps_DIS_to_CC": transition_summary(t_cps, "DIS_to_CC"),
            "adaptive_DIS_to_CC": transition_summary(t_ad, "DIS_to_CC"),
            "cps_DIS_to_DIS": transition_summary(t_cps, "DIS_to_DIS"),
            "adaptive_DIS_to_DIS": transition_summary(t_ad, "DIS_to_DIS"),
        },
        "aggregate_deltas": {
            "mean_WW_to_DIS": escape_gain, "mean_WW_to_CC": float(delta_ww_cc.mean()),
            "mean_WW_DIS_CC": chain_gain, "mean_DIS_to_DIS": persistent_gain,
            "mean_CC_to_DIS": cc_damage_gain,
        },
        "mechanism_outcome": outcome,
        "mechanism_conclusion": mechanism,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "config.json").write_text(json.dumps({
        "methods": {"Original_CPS": str(COMMON_CPS), "D_Adaptive": str(ADAPTIVE_CPS)},
        "memory": str(MEMORY), "common_epochs": epochs, "validation_images": 17,
        "state_definition": "CC=A=B=GT; WW=A=B!=GT; DIS=A!=B; threshold p>=0.5",
        "contradiction_definition": "agreement token and D=S_opp-S_cur>0; no GT",
        "deterministic": True, "shuffle": False, "random_augmentation": False,
    }, indent=2) + "\n")
    readme = f"""# WW Recovery Dynamic Chain Diagnostic

Offline deterministic validation diagnostic; no optimizer, backward pass, or checkpoint modification.

Common epochs: `{epochs}`. State definition: `CC=A=B=GT`, `WW=A=B!=GT`, `DIS=A!=B`, with `p>=0.5`.
Adaptive contradiction is recomputed from frozen UNI-256 reference evidence and is never defined from GT.

Outcome: **{outcome}** — {mechanism}

Prediction archives are under `predictions/`; CSVs contain global rates, conditional transitions, three-step chains, targeted/matched cohorts, D quartiles, and disagreement balance.
"""
    (OUT / "README.md").write_text(readme)
    print("=== WW Recovery Dynamic Chain Diagnostic ===")
    print("Methods:")
    print("  Original CPS:", COMMON_CPS)
    print("  D-Adaptive Native-S:", ADAPTIVE_CPS)
    print("Common epochs:", epochs)
    print("Global disagreement:")
    for e in epochs:
        print(f"  {e}: CPS={g_cps.loc[e, 'DIS_rate']:.6f} Adaptive={g_ad.loc[e, 'DIS_rate']:.6f} Delta={dis_delta[str(e)]:+.6f}")
    print("Late disagreement:", late)
    for e in epochs[:-1]:
        print(f"WW transition {e}->{epochs[epochs.index(e)+1]} | CPS WW/ DIS/ CC = {t_cps.loc[e,'WW_to_WW']:.4f}/{t_cps.loc[e,'WW_to_DIS']:.4f}/{t_cps.loc[e,'WW_to_CC']:.4f} | Adaptive = {t_ad.loc[e,'WW_to_WW']:.4f}/{t_ad.loc[e,'WW_to_DIS']:.4f}/{t_ad.loc[e,'WW_to_CC']:.4f}")
    print("Mechanism outcome:", outcome, "-", mechanism)
    print("Output:", OUT)


if __name__ == "__main__":
    main()
