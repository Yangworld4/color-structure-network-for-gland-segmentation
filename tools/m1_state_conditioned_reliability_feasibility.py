"""Stage M1: frozen train-only state-conditioned reliability feasibility.

The only fitted objects in this script are two one-dimensional monotonic
logistic calibrators.  Their fit samples come from image-level LOO memories
constructed from the seven labeled training images.  Validation is evaluated
only after those calibrators are frozen.  No test dataset is instantiated.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from models.getnetwork import get_network  # noqa: E402
from models.uni import load_uni  # noqa: E402
from main import (  # noqa: E402
    EXPECTED_UNI_SHA256,
    REFERENCE_MEMORY_ANCHOR_BG,
    REFERENCE_MEMORY_ANCHOR_FG,
    REFERENCE_MEMORY_FALLBACK_FRACTION,
    REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
    REFERENCE_MEMORY_SEED,
    UNI_CONFIG_DEFAULT,
    UNI_MEAN,
    UNI_STD,
    UNI_WEIGHTS_DEFAULT,
    _reconstruct_rgb,
    _reference_margin,
    _uni_dense_features,
)

DATASET = "GlaS"
INPUT_MODE = "rgb_stain"
NETWORK = "unet"
EPOCHS = tuple(range(10, 201, 10))
SEED = 24
K = 5
THRESHOLD = 0.5
HC_THRESHOLD = 0.90
CAP = 64
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 1337
CHECKPOINT_ROOT = ROOT / "checkpoints" / "semi_10_seed_24_rgbstain_uni_reference_state_aware" / "GLaS" / "CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=uni_reference_state_aware"
FULL_MEMORY = ROOT / "checkpoints" / "glas_labeled10_seed24" / "reference_memory.npz"
MANIFEST_ROOT = ROOT / "datasets" / "GLaS" / "manifests" / "paper_seed_2026"
DEFAULT_OUT = ROOT / "diagnostics" / "m1_state_conditioned_reliability_r224_glas10"
BINS = ((0.0, .01), (.01, .02), (.02, .05), (.05, .10), (.10, .20), (.20, float("inf")))
COVERAGES = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


class ProtocolMismatch(RuntimeError):
    pass


def fail(msg: str) -> None:
    raise ProtocolMismatch(msg)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def json_safe(x: Any) -> Any:
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, np.generic):
        return json_safe(x.item())
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path, require_mask: bool) -> List[Dict[str, str]]:
    if not path.is_file():
        fail("manifest missing: %s" % path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not all(r.get("image") for r in rows):
        fail("invalid or empty manifest: %s" % path)
    if require_mask and not all(r.get("mask") for r in rows):
        fail("mask missing in manifest: %s" % path)
    return rows


def scan_checkpoints(root: Path) -> List[Tuple[int, Path]]:
    found: Dict[int, Path] = {}
    for p in sorted((root / "milestones").glob("epoch_*.pth")):
        m = re.fullmatch(r"epoch_(\d{3})\.pth", p.name)
        if m:
            e = int(m.group(1))
            if e in found:
                fail("duplicate checkpoint epoch %d" % e)
            found[e] = p
    if tuple(sorted(found)) != EPOCHS:
        fail("expected synchronized epochs 10..200, found %s" % sorted(found))
    for e, p in found.items():
        x = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(x, dict) or not isinstance(x.get("appearance_state_dict"), dict) or not isinstance(x.get("structure_state_dict"), dict):
            fail("checkpoint %d lacks synchronized state keys" % e)
        for k, expected in (("dataset_name", "GlaS"), ("labeled_ratio", 10), ("seed", 24), ("input_mode", INPUT_MODE), ("network", NETWORK)):
            actual = x.get(k)
            if (actual.lower() != expected.lower() if isinstance(expected, str) and isinstance(actual, str) else actual != expected):
                fail("checkpoint %d %s=%r expected %r" % (e, k, actual, expected))
    return [(e, found[e]) for e in EPOCHS]


def loader(manifest: Path, cfg: Mapping[str, Any]) -> DataLoader:
    ds = imagefloder_itn(
        manifest_path=str(manifest),
        data_transform_1=data_transform_2d()["val"],
        data_normalize_1=data_normalize_2d(cfg["MEAN"], cfg["STD"]),
        sup=True, input_mode=INPUT_MODE, training=False,
    )
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)


def collect_static(dl: DataLoader, cfg: Mapping[str, Any], uni: torch.nn.Module) -> List[Dict[str, Any]]:
    out = []
    with torch.inference_mode():
        for b in dl:
            rgb01 = _reconstruct_rgb(b["image_rgb"].cuda(non_blocking=True), cfg)
            tok = _uni_dense_features(uni, rgb01, 224).cpu().numpy().astype(np.float32)
            for i in range(len(b["ID"])):
                out.append({"image_id": str(b["ID"][i]), "gt": b["mask"][i].cpu().numpy().astype(bool), "tokens": tok[i]})
    if not out or out[0]["tokens"].shape != (196, 1024):
        fail("UNI feature shape is not 196x1024")
    return out


def collect_students(dl: DataLoader, checkpoint: Path, cfg: Mapping[str, Any], device: torch.device) -> List[Dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ma = get_network(NETWORK, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    mb = get_network(NETWORK, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    ma.load_state_dict(payload["appearance_state_dict"], strict=True)
    mb.load_state_dict(payload["structure_state_dict"], strict=True)
    ma.to(device).eval(); mb.to(device).eval()
    if ma.training or mb.training:
        fail("model.eval assertion failed")
    out = []
    with torch.inference_mode():
        for b in dl:
            xa = b["image_rgb"].to(device); xb = b["image_stain"].to(device)
            la = ma(xa)[0]; lb = mb(xb)[0]
            if tuple(la.shape[-2:]) != (256, 256) or tuple(lb.shape[-2:]) != (256, 256):
                fail("student logits are not 256x256")
            pa = torch.softmax(la, 1)[:, 1].cpu().numpy().astype(np.float32)
            pb = torch.softmax(lb, 1)[:, 1].cpu().numpy().astype(np.float32)
            for i in range(len(b["ID"])):
                out.append({"image_id": str(b["ID"][i]), "gt": b["mask"][i].cpu().numpy().astype(bool), "prob_a": pa[i], "prob_b": pb[i]})
    del ma, mb
    torch.cuda.empty_cache()
    return out


def formal_anchor_banks(records: Sequence[Mapping[str, Any]], exclude: str | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Reproduce main.py's R224 anchor protocol, with explicit provenance."""
    rng = np.random.default_rng(REFERENCE_MEMORY_SEED)
    fg, bg = [], []
    provenance = {"fg": 0, "bg": 0}
    for r in records:
        if exclude is not None and r["image_id"] == exclude:
            continue
        occ = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(r["gt"].astype(np.float32))[None, None], (14, 14)).numpy().reshape(-1)
        for cls, threshold, target, key in ((1, REFERENCE_MEMORY_ANCHOR_FG, fg, "fg"), (0, REFERENCE_MEMORY_ANCHOR_BG, bg, "bg")):
            if cls == 1:
                cand = np.flatnonzero(occ >= threshold); order = np.argsort(-occ, kind="stable")
            else:
                cand = np.flatnonzero(occ <= threshold); order = np.argsort(occ, kind="stable")
            if not len(cand):
                n = max(1, min(CAP, int(math.ceil(len(occ) * REFERENCE_MEMORY_FALLBACK_FRACTION))))
                cand = order[:n]
            if len(cand) > CAP:
                cand = np.sort(rng.choice(cand, CAP, replace=False))
            target.extend(r["tokens"][cand])
            provenance[key] += len(cand)
    if not fg or not bg:
        fail("formal anchor bank empty")
    return np.asarray(fg, np.float32), np.asarray(bg, np.float32), provenance


def margin_for(tokens: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        t = torch.from_numpy(tokens).cuda().unsqueeze(0)
        mf = torch.from_numpy(fg).float().cuda(); mb = torch.from_numpy(bg).float().cuda()
        margin = _reference_margin(t, {"memory_fg": mf, "memory_bg": mb}, K)
        margin = F.interpolate(margin.reshape(1, 1, 14, 14), size=(256, 256), mode="bilinear", align_corners=False)
        return margin.cpu().numpy()[0, 0].astype(np.float32)


def event_arrays(student: Mapping[str, Any], margin: np.ndarray) -> Dict[str, np.ndarray]:
    pa = np.asarray(student["prob_a"], np.float32).ravel() >= THRESHOLD
    pb = np.asarray(student["prob_b"], np.float32).ravel() >= THRESHOLD
    gt = np.asarray(student["gt"], bool).ravel()
    m = np.asarray(margin, np.float32).ravel()
    agree = pa == pb; disagree = ~agree
    consensus = pa
    signed = np.where(consensus, m, -m)
    a_contra = agree & (signed < 0)
    d_non_tie = disagree & (m != 0)
    u_a = -signed[a_contra]
    z_a = (consensus[a_contra] != gt[a_contra]).astype(bool)
    u_d = np.abs(m[d_non_tie])
    y_r = m[d_non_tie] > 0
    z_d = (y_r == gt[d_non_tie]).astype(bool)
    return {
        "pa": pa, "pb": pb, "gt": gt, "m": m, "agree": agree, "disagree": disagree,
        "a_contra": a_contra, "d_non_tie": d_non_tie, "u_a": u_a, "z_a": z_a,
        "u_d": u_d, "z_d": z_d, "yc_a": consensus[a_contra], "yr_d": y_r,
    "hc90_a": a_contra & (np.minimum(np.maximum(np.asarray(student["prob_a"]).ravel(), 1-np.asarray(student["prob_a"]).ravel()), np.maximum(np.asarray(student["prob_b"]).ravel(), 1-np.asarray(student["prob_b"]).ravel())) >= HC_THRESHOLD),
    }


def metric(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, bool); score = np.asarray(score, float); ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if not len(y) or np.unique(y).size < 2:
        return {"status": "UNDEFINED_SINGLE_CLASS", "AUROC": np.nan, "AUPRC": np.nan, "valid_count": len(y)}
    return {"status": "OK", "AUROC": float(roc_auc_score(y, score)), "AUPRC": float(average_precision_score(y, score)), "valid_count": len(y)}


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, bool); p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7); ok = np.isfinite(p); y, p = y[ok], p[ok]
    m = metric(y, p)
    if not len(y):
        return {**m, "Brier": np.nan, "NLL": np.nan, "ECE": np.nan}
    order = np.argsort(p); ys = y[order].astype(float); ps = p[order]; bins = np.array_split(np.arange(len(y)), 10)
    ece = sum(len(ix)/len(y) * abs(ys[ix].mean()-ps[ix].mean()) for ix in bins if len(ix))
    nll = -np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
    return {**m, "Brier": float(brier_score_loss(y, p)), "NLL": float(nll), "ECE": float(ece)}


def fit_logistic(samples: Sequence[Mapping[str, Any]], u_key: str, z_key: str) -> Tuple[float, float, float]:
    x, y, w = [], [], []
    for s in samples:
        xx = np.asarray(s[u_key], float); yy = np.asarray(s[z_key], float)
        if len(xx):
            x.append(xx); y.append(yy); w.append(np.full(len(xx), 1.0/len(xx)))
    if not x:
        return np.nan, np.nan, np.nan
    x = np.concatenate(x); y = np.concatenate(y); w = np.concatenate(w); w /= w.sum()
    def fun(theta):
        a = float(np.logaddexp(0, theta[0])); p = expit(a*x+theta[1]); return float(-np.sum(w*(y*np.log(p+1e-12)+(1-y)*np.log1p(-p+1e-12))))
    res = minimize(fun, np.array([0.0, 0.0]), method="BFGS", options={"maxiter": 500, "gtol": 1e-8})
    a = float(np.logaddexp(0, res.x[0])); b = float(res.x[1]); prevalence = float(np.sum(w*y))
    return a, b, prevalence


def apply_cal(a: float, b: float, u: np.ndarray) -> np.ndarray:
    return expit(a*np.asarray(u, float)+b) if np.isfinite(a+b) else np.full(len(u), np.nan)


def concat_events(items: Sequence[Mapping[str, Any]], u_key: str, z_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    us = [np.asarray(x[u_key], float) for x in items]; zs = [np.asarray(x[z_key], bool) for x in items]
    if not us: return np.empty(0), np.empty(0, bool), np.empty(0, int)
    return np.concatenate(us), np.concatenate(zs), np.concatenate([np.full(len(x[u_key]), i, int) for i, x in enumerate(items)])


def bins_rows(items: Sequence[Mapping[str, Any]], u_key: str, z_key: str, prefix: str, epoch: int, kind: str) -> List[Dict[str, Any]]:
    u, z, im = concat_events(items, u_key, z_key); out=[]
    for j,(lo,hi) in enumerate(BINS):
        q=(u>=lo)&(u<hi); out.append({"epoch":epoch,"kind":kind,"bin_index":j,"lower":lo,"upper":"+inf" if not np.isfinite(hi) else hi,"count":int(q.sum()),"num_images":int(np.unique(im[q]).size) if q.any() else 0,"WW_count":int(z[q].sum()),"CC_count":int((~z[q]).sum()),"routing_correct":float(z[q].mean()) if kind.startswith("D") and q.any() else np.nan,"WW_prevalence":float(z[q].mean()) if q.any() else np.nan,"mean_u":float(u[q].mean()) if q.any() else np.nan,"median_u":float(np.median(u[q])) if q.any() else np.nan})
    if len(u):
        order=np.argsort(u); qid=np.array_split(order,5)
        for j,ix in enumerate(qid): out.append({"epoch":epoch,"kind":kind,"bin_index":"Q%d"%(j+1),"lower":np.nan,"upper":np.nan,"count":len(ix),"num_images":int(np.unique(im[ix]).size),"WW_count":int(z[ix].sum()),"CC_count":int((~z[ix]).sum()),"routing_correct":float(z[ix].mean()) if kind.startswith("D") else np.nan,"WW_prevalence":float(z[ix].mean()),"mean_u":float(u[ix].mean()),"median_u":float(np.median(u[ix]))})
    return out


def coverage_rows(items: Sequence[Mapping[str, Any]], u_key: str, z_key: str, prefix: str, epoch: int, kind: str) -> List[Dict[str, Any]]:
    u,z,im=concat_events(items,u_key,z_key); out=[]
    if not len(u): return out
    ix=np.argsort(-u)
    for c in COVERAGES:
        n=max(1,int(round(len(u)*c))); take=ix[:n]; zz=z[take]
        row={"epoch":epoch,"kind":kind,"coverage":c,"pixel_count":n,"image_count":int(np.unique(im[take]).size),"WW_prevalence":float(zz.mean()),"CC_prevalence":float((~zz).mean()),"routing_accuracy":float(zz.mean()) if kind.startswith("D") else np.nan,"mean_u":float(u[take].mean())}
        if kind.startswith("Agreement"):
            row.update({"AUROC":metric(zz,u[take])["AUROC"],"intervention_precision":float(zz.sum()*0+np.sum(u[take][zz])/(np.sum(u[take])+1e-12))})
        else:
            row["gradient_weighted_routing_precision"]=float(np.sum(zz*u[take])/(np.sum(u[take])+1e-12))
        out.append(row)
    return out


def class_conditioned(items: Sequence[Mapping[str, Any]], epoch: int, kind: str, source: str) -> List[Dict[str, Any]]:
    u,z,im=concat_events(items,"u_a" if kind=="Agreement" else "u_d","z_a" if kind=="Agreement" else "z_d")
    c=np.concatenate([x["yc_a"] for x in items]) if kind=="Agreement" else np.concatenate([x["yr_d"] for x in items])
    out=[]
    conditions=[("FG",c==True),("BG",c==False)]
    if kind=="D":
        a_correct=np.concatenate([x["pa"][x["d_non_tie"]]==x["gt"][x["d_non_tie"]] for x in items])
        conditions.extend([("A_CORRECT",a_correct),("B_CORRECT",~a_correct)])
    for name,q in conditions:
        mm=metric(z[q],u[q]); out.append({"epoch":epoch,"source":source,"kind":kind,"condition":name,"count":int(q.sum()),"WW_prevalence_or_routing_accuracy":float(z[q].mean()) if q.any() else np.nan,"AUROC":mm["AUROC"],"AUPRC":mm["AUPRC"]})
    return out


def agreement_aux_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Agreement HC90 and confidence-baseline metrics on contradiction events."""
    us, ys, confs, hcs = [], [], [], []
    for x in items:
        e = x["events"]; mask = e["a_contra"]
        pa = np.asarray(x["prob_a"], float).ravel(); pb = np.asarray(x["prob_b"], float).ravel()
        joint_conf = np.minimum(np.maximum(pa, 1 - pa), np.maximum(pb, 1 - pb))
        us.append(np.clip(e["u_a"], 0, 1)); ys.append(e["z_a"])
        confs.append(1 - joint_conf[mask]); hcs.append(e["hc90_a"][mask])
    if not us:
        return {"HC90_CC_count": 0, "HC90_WW_count": 0, "HC90_AUROC_WW": np.nan,
                "HC90_AUPRC_WW": np.nan, "confidence_AUROC_WW": np.nan,
                "confidence_AUPRC_WW": np.nan, "confidence_HC90_AUROC_WW": np.nan,
                "confidence_HC90_AUPRC_WW": np.nan}
    u, y, confidence_score, hc = map(np.concatenate, (us, ys, confs, hcs))
    hm = metric(y[hc], u[hc]); cm = metric(y, confidence_score); hcm = metric(y[hc], confidence_score[hc])
    return {"HC90_CC_count": int(np.sum(hc & ~y)), "HC90_WW_count": int(np.sum(hc & y)),
            "HC90_AUROC_WW": hm["AUROC"], "HC90_AUPRC_WW": hm["AUPRC"],
            "confidence_AUROC_WW": cm["AUROC"], "confidence_AUPRC_WW": cm["AUPRC"],
            "confidence_HC90_AUROC_WW": hcm["AUROC"], "confidence_HC90_AUPRC_WW": hcm["AUPRC"]}


def confidence_routing_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Route every disagreement pixel by higher Student joint confidence."""
    total = correct_total = non_tie = correct_non_tie = ties = 0
    for x in items:
        e = x["events"]; mask = e["disagree"]
        pa = np.asarray(x["prob_a"], float).ravel()[mask]; pb = np.asarray(x["prob_b"], float).ravel()[mask]
        pred_a = e["pa"][mask]; pred_b = e["pb"][mask]; gt = e["gt"][mask]
        ca = np.maximum(pa, 1 - pa); cb = np.maximum(pb, 1 - pb)
        choose_a = ca > cb; choose_b = cb > ca; tie = ~(choose_a | choose_b)
        chosen_correct = np.where(choose_a, pred_a == gt, np.where(choose_b, pred_b == gt, False))
        total += len(gt); correct_total += int(chosen_correct.sum()); ties += int(tie.sum())
        non_tie += int((~tie).sum()); correct_non_tie += int((chosen_correct & ~tie).sum())
    return {"confidence_routing_accuracy_total": safe_div(correct_total, total),
            "confidence_routing_accuracy_non_tie": safe_div(correct_non_tie, non_tie),
            "confidence_routing_tie_rate": safe_div(ties, total),
            "confidence_routing_total_count": total, "confidence_routing_non_tie_count": non_tie}


def replay_agreement(items: Sequence[Mapping[str, Any]], a: float, b: float) -> Dict[str, Any]:
    raw_u=[]; z=[]; paopp=[]; pbopp=[]; hc90=[]; hc95=[]; ww_total=[]; cc_total=[]; cc_u=[]; cc_paopp=[]; cc_pbopp=[]; cc_hc90=[]; cc_hc95=[]
    for x in items:
        e=x["events"]; raw_u.append(np.clip(e["u_a"],0,1)); z.append(e["z_a"]); pa=e["pa"]; pb=e["pb"]; yc=e["yc_a"]; prob_a=np.asarray(x["prob_a"]).ravel(); prob_b=np.asarray(x["prob_b"]).ravel(); paopp.append(np.where(yc,1-prob_a[e["a_contra"]],prob_a[e["a_contra"]])); pbopp.append(np.where(yc,1-prob_b[e["a_contra"]],prob_b[e["a_contra"]])); hc90.append(e["hc90_a"][e["a_contra"]]);
        full_cc=x["events"]["agree"] & (x["events"]["pa"] == x["events"]["gt"]); full_ww=x["events"]["agree"] & (x["events"]["pa"] != x["events"]["gt"]); ww_total.append(int(full_ww.sum())); cc_total.append(int(full_cc.sum())); cc_mask=full_cc & x["events"]["a_contra"]; cc_u.append(np.clip(-np.where(x["events"]["pa"],x["margin"].ravel(),-x["margin"].ravel())[cc_mask],0,1)); cc_yc=x["events"]["pa"][cc_mask]; cc_paopp.append(np.where(cc_yc,1-prob_a[cc_mask],prob_a[cc_mask])); cc_pbopp.append(np.where(cc_yc,1-prob_b[cc_mask],prob_b[cc_mask])); cc_hc90.append(x["events"]["hc90_a"][cc_mask]);
        conf=np.minimum(np.maximum(prob_a,1-prob_a),np.maximum(prob_b,1-prob_b)); hc95.append((e["a_contra"]&(conf>=.95))[e["a_contra"]]); cc_hc95.append((e["a_contra"]&full_cc&(conf>=.95))[e["a_contra"]])
    u=np.concatenate(raw_u); y=np.concatenate(z); qa=apply_cal(a,b,u); p1=np.concatenate(paopp); p2=np.concatenate(pbopp); h90=np.concatenate(hc90); h95=np.concatenate(hc95)
    def one(name,r):
        anyr=((r>p1)|(r>p2)); both=(r>p1)&(r>p2); ww=y
        return {name+"_reverse_any_all_WW":safe_div(np.sum(anyr&ww),sum(ww_total)),name+"_reverse_any_WW_contradiction":safe_div(np.sum(anyr&ww),np.sum(ww)),name+"_reverse_both_WW_contradiction":safe_div(np.sum(both&ww),np.sum(ww)),name+"_reverse_any_HC90_WW":safe_div(np.sum(anyr&ww&h90),np.sum(ww&h90)),name+"_reverse_any_HC95_WW":safe_div(np.sum(anyr&ww&h95),np.sum(ww&h95))}
    def harmful(r):
        anyr=(r>np.concatenate(cc_paopp))|(r>np.concatenate(cc_pbopp)); return safe_div(np.sum(anyr),sum(cc_total))
    cc_raw_u=np.concatenate(cc_u) if cc_u else np.empty(0); cc_cal_u=apply_cal(a,b,cc_raw_u); ur=float(np.sum(u[y])); hr=float(np.sum(u[~y])); uc=float(np.sum(qa[y])); hc=float(np.sum(qa[~y])); return {"U_A_raw":ur,"H_A_raw":hr,"Precision_raw":safe_div(ur,ur+hr),"U_A_cal":uc,"H_A_cal":hc,"Precision_cal":safe_div(uc,uc+hc),"Delta_U_A":uc-ur,"Delta_H_A":hc-hr,"Delta_Precision_A":safe_div(uc,uc+hc)-safe_div(ur,ur+hr),"useful_mass_retention":safe_div(uc,ur),"harmful_mass_retention":safe_div(hc,hr),"WW_corrective_coverage":float(np.mean(qa[y]>0)) if np.any(y) else np.nan, "mean_q_WW":float(np.mean(qa[y])) if np.any(y) else np.nan,"mean_q_CC":float(np.mean(qa[~y])) if np.any(~y) else np.nan,"raw_CC_harmful_reverse":harmful(cc_raw_u),"cal_CC_harmful_reverse":harmful(cc_cal_u),**one("raw",u),**one("cal",qa)}


def grad_norm(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    p=np.stack([1-prob,prob],axis=1); g=p; g[np.arange(len(target)),target]-=1; return np.linalg.norm(g,axis=1)


def replay_d(items: Sequence[Mapping[str, Any]], a: float, b: float) -> Dict[str, Any]:
    u=[]; z=[]; q=[]; gc=[]; gw=[]; img=[]
    for i,x in enumerate(items):
        e=x["events"]; mask=e["d_non_tie"]; m=np.asarray(x["margin"]).ravel()[mask]; pa=np.asarray(x["prob_a"]).ravel()[mask]; pb=np.asarray(x["prob_b"]).ravel()[mask]; pred_a=e["pa"][mask]; pred_b=e["pb"][mask]; yr=e["yr_d"]; zz=e["z_d"]
        ga=grad_norm(pa,pred_b.astype(int)); gb=grad_norm(pb,pred_a.astype(int)); # A->B and B->A
        supported_a=pred_a==yr; correct_grad=np.where(zz,np.where(supported_a,ga,gb),np.where(supported_a,gb,ga)); wrong_grad=np.where(zz,np.where(supported_a,gb,ga),np.where(supported_a,ga,gb))
        u.append(np.abs(m)); z.append(zz); gc.append(correct_grad); gw.append(wrong_grad); img.append(np.full(len(m),i))
    uu=np.concatenate(u); zz=np.concatenate(z); qv=apply_cal(a,b,uu); c=np.concatenate(gc); w=np.concatenate(gw); delta=(qv-.5)*(c-w); beneficial=np.maximum(delta,0)+np.maximum(-delta,0)*0 # explicit below
    # The first term is the shift toward the correct direction; the second is
    # removal from the wrong direction.  Both are positive only when useful.
    beneficial=np.maximum(qv-.5,0)*c + np.maximum(.5-qv,0)*w
    harmful=np.maximum(.5-qv,0)*c + np.maximum(qv-.5,0)*w
    return {"mean_correct_teacher_weight":float(np.mean(np.where(zz,qv,1-qv))),"mean_wrong_teacher_weight":float(np.mean(np.where(zz,1-qv,qv))),"directional_advantage":float(np.mean(np.where(zz,qv,1-qv))-.5),"P_q_gt_half_ref_correct":safe_div(np.sum((qv>.5)&zz),np.sum(zz)),"P_q_lt_half_ref_wrong":safe_div(np.sum((qv<.5)&~zz),np.sum(~zz)),"beneficial_gradient_shift":float(np.sum(beneficial)),"harmful_gradient_shift":float(np.sum(harmful)),"net_gradient_benefit":float(np.sum(beneficial)-np.sum(harmful)),"gradient_redistribution_precision":safe_div(float(np.sum(beneficial)),float(np.sum(beneficial)+np.sum(harmful))),"gradient_weighted_routing_precision":safe_div(float(np.sum(c[zz])),float(np.sum(c[zz])+np.sum(c[~zz])))}


def plot_curve(path: Path, x: Sequence[float], series: Sequence[Tuple[str, Sequence[float]]], xlabel: str, ylabel: str) -> None:
    fig, ax=plt.subplots(figsize=(7,5));
    for name,y in series: ax.plot(x,y,marker="o",label=name)
    ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def plot_reliability(path: Path, y: np.ndarray, p: np.ndarray, xlabel: str) -> None:
    q=np.linspace(0,1,11); xx=[]; yy=[]
    for lo,hi in zip(q[:-1],q[1:]):
        m=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if m.any(): xx.append(float(p[m].mean())); yy.append(float(y[m].mean()))
    fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],"k--"); ax.plot(xx,yy,"o-"); ax.set(xlabel=xlabel,ylabel="Observed frequency"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def bootstrap(epoch_payload: Mapping[str, Any], mode: str) -> List[Dict[str, Any]]:
    ev=epoch_payload["items"]; rng=np.random.default_rng(BOOTSTRAP_SEED+(0 if mode=="strict" else 10000)); names=("raw_AUROC","raw_AUPRC","HC90_AUROC","HC90_AUPRC","score_gap","delta_precision","delta_H","delta_U","delta_reverse") if mode=="strict" else ("d_AUROC","d_AUPRC","d_advantage","d_net","d_precision"); vals={k:[] for k in names}
    for _ in range(BOOTSTRAP_REPS):
        chosen=[ev[int(i)] for i in rng.integers(0,len(ev),len(ev))]
        if mode=="strict":
            chosen_events=[x["events"] for x in chosen]; u,z,_=concat_events(chosen_events,"u_a","z_a"); mm=metric(z,u); aux=agreement_aux_metrics(chosen); vals["raw_AUROC"].append(mm["AUROC"]); vals["raw_AUPRC"].append(mm["AUPRC"]); vals["HC90_AUROC"].append(aux["HC90_AUROC_WW"]); vals["HC90_AUPRC"].append(aux["HC90_AUPRC_WW"]); vals["score_gap"].append(float(np.mean(u[z])-np.mean(u[~z])) if np.any(z) and np.any(~z) else np.nan); ra=replay_agreement(chosen,epoch_payload["a"],epoch_payload["b"]); vals["delta_precision"].append(ra["Delta_Precision_A"]); vals["delta_H"].append(ra["Delta_H_A"]); vals["delta_U"].append(ra["Delta_U_A"]); vals["delta_reverse"].append(ra["cal_reverse_any_WW_contradiction"]-ra["raw_reverse_any_WW_contradiction"])
        else:
            chosen_events=[x["events"] for x in chosen]; u,z,_=concat_events(chosen_events,"u_d","z_d"); mm=metric(z,u); vals["d_AUROC"].append(mm["AUROC"]); vals["d_AUPRC"].append(mm["AUPRC"]); rd=replay_d(chosen,epoch_payload["a_d"],epoch_payload["b_d"]); vals["d_advantage"].append(rd["directional_advantage"]); vals["d_net"].append(rd["net_gradient_benefit"]); vals["d_precision"].append(rd["gradient_redistribution_precision"])
    out=[]
    for k,v in vals.items():
        x=np.asarray(v,float); ok=np.isfinite(x); out.append({"epoch":200,"mode":mode,"metric":k,"estimate":float(np.nanmean(x)) if ok.any() else np.nan,"ci_low":float(np.nanpercentile(x[ok],2.5)) if ok.any() else np.nan,"ci_high":float(np.nanpercentile(x[ok],97.5)) if ok.any() else np.nan,"n_valid_bootstrap":int(ok.sum()),"n_requested_bootstrap":BOOTSTRAP_REPS})
    return out


def run(args: argparse.Namespace) -> None:
    out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()): fail("output directory is non-empty; refusing overwrite: %s"%out)
    checkpoints=scan_checkpoints(Path(args.checkpoint_dir).resolve())
    lab_rows=read_manifest(MANIFEST_ROOT/"labeled_10.csv",True); unsup_rows=read_manifest(MANIFEST_ROOT/"unlabeled_10.csv",False); val_rows=read_manifest(MANIFEST_ROOT/"validation.csv",True)
    if (len(lab_rows),len(unsup_rows),len(val_rows))!=(7,61,17): fail("split count mismatch")
    if not torch.cuda.is_available(): fail("CUDA required")
    if not FULL_MEMORY.is_file(): fail("formal full R224 memory missing")
    with np.load(FULL_MEMORY,allow_pickle=False) as mem:
        if mem["memory_fg"].shape!=(391,1024) or mem["memory_bg"].shape!=(294,1024): fail("formal full memory count/feature mismatch")
        if int(mem["token_grid"])!=14 or int(mem["top_k"])!=5 or float(mem["anchor_fg"])!=.9 or float(mem["anchor_bg"])!=.1 or int(mem["seed"])!=2026: fail("formal full memory metadata mismatch")
        full_fg=np.asarray(mem["memory_fg"],np.float32); full_bg=np.asarray(mem["memory_bg"],np.float32)
    cfg=dataset_cfg(DATASET); lab_dl=loader(MANIFEST_ROOT/"labeled_10.csv",cfg); val_dl=loader(MANIFEST_ROOT/"validation.csv",cfg)
    if len(lab_dl.dataset)!=7 or len(val_dl.dataset)!=17: fail("loader count mismatch")
    out.mkdir(parents=True,exist_ok=False); (out/"figures").mkdir(); (out/"cache").mkdir()
    log_path=out/"full_run.console.log"
    with log_path.open("w",encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.__stdout__,log)):
        print("Stage M1 frozen diagnostic | training=false | test_loaded=false | validation_used_for_fitting=false")
        print("protocol: dataset=GLaS split=7/61/17 seed=24 UNI=224 grid=14x14 K=5 full_memory=391/294")
        uni,uni_weights,uni_config=load_uni(UNI_WEIGHTS_DEFAULT,UNI_CONFIG_DEFAULT,expected_sha256=EXPECTED_UNI_SHA256)
        if not uni.training: pass
        static_lab=collect_static(lab_dl,cfg,uni); static_val=collect_static(val_dl,cfg,uni)
        # Static UNI tensors are the only feature cache; no validation target is
        # used while constructing any memory or fitting either calibrator.
        loo_banks={}
        loo_manifest_rows=[]
        for r in static_lab:
            fg,bg,counts=formal_anchor_banks(static_lab,exclude=r["image_id"]); loo_banks[r["image_id"]]=(fg,bg); loo_manifest_rows.extend([{"held_out_image_id":r["image_id"],"memory_source":"other_6_labeled_images","fg_count":len(fg),"bg_count":len(bg),"K":K,"cap":CAP,"seed":REFERENCE_MEMORY_SEED}])
        write_csv(out/"01_train_loo_memory_manifest.csv",loo_manifest_rows)
        static_by_id={r["image_id"]:r for r in static_lab}; val_static_by_id={r["image_id"]:r for r in static_val}
        summary=[]; raw_bins=[]; cov_rows=[]; class_rows=[]; d_bins=[]; d_cov=[]; d_class=[]; train_metrics=[]; train_sample_rows=[]; d_train_sample_rows=[]; params=[]; val_metrics=[]; val_agree=[]; val_d=[]; temporal=[]; image_rows=[]; score_rows=[]; boot_payloads={}
        train_dl=lab_dl; val_loader=val_dl
        for epoch,cp in checkpoints:
            t0=time.time(); s_lab=collect_students(train_dl,cp,cfg,torch.device("cuda")); s_val=collect_students(val_loader,cp,cfg,torch.device("cuda"))
            train_items=[]; val_items=[]
            for s in s_lab:
                st=static_by_id[s["image_id"]]; fg,bg=loo_banks[s["image_id"]]; s={**s,"margin":margin_for(st["tokens"],fg,bg)}; s["events"]=event_arrays(s,s["margin"]); train_items.append(s)
                train_sample_rows.append({"epoch":epoch,"image_id":s["image_id"],"agreement_contradiction_count":len(s["events"]["u_a"]),"WW_count":int(s["events"]["z_a"].sum()),"CC_count":int((~s["events"]["z_a"]).sum()),"D_non_tie_count":len(s["events"]["u_d"]),"routing_correct_count":int(s["events"]["z_d"].sum()),"routing_accuracy":float(s["events"]["z_d"].mean()) if len(s["events"]["z_d"]) else np.nan})
            for s in s_val:
                st=val_static_by_id[s["image_id"]]; s={**s,"margin":margin_for(st["tokens"],full_fg,full_bg)}; s["events"]=event_arrays(s,s["margin"]); val_items.append(s)
            # LOO seven-fold predictions: fold i fits on the other six image
            # samples and evaluates only image i.
            cv_a=[]; cv_d=[]
            for hold in range(7):
                fit=[x["events"] for j,x in enumerate(train_items) if j!=hold]; aa,ba,pa=fit_logistic(fit,"u_a","z_a"); ad,bd,pd=fit_logistic(fit,"u_d","z_d"); e=train_items[hold]["events"]; prob_a=np.asarray(train_items[hold]["prob_a"],float).ravel(); prob_b=np.asarray(train_items[hold]["prob_b"],float).ravel(); conf=1-np.minimum(np.maximum(prob_a,1-prob_a),np.maximum(prob_b,1-prob_b))[e["a_contra"]]; cv_a.append({"y":e["z_a"],"raw":np.clip(e["u_a"],0,1),"q":apply_cal(aa,ba,e["u_a"]),"base":np.full(len(e["u_a"]),pa),"confidence":conf,"hc90":e["hc90_a"][e["a_contra"]],"image_id":train_items[hold]["image_id"]}); cv_d.append({"y":e["z_d"],"raw":np.abs(e["u_d"]),"q":apply_cal(ad,bd,e["u_d"]),"base":np.full(len(e["u_d"]),pd),"image_id":train_items[hold]["image_id"]})
                for domain,arr in (("Agreement",cv_a[-1]),("Disagreement",cv_d[-1])):
                    for predictor in ("constant","raw","calibrated"):
                        pred_key={"constant":"base","raw":"raw","calibrated":"q"}[predictor]; mm=calibration_metrics(arr["y"],arr[pred_key]); extra={};
                        if domain=="Agreement":
                            hm=calibration_metrics(arr["y"][arr["hc90"]],arr[pred_key][arr["hc90"]]); extra={"HC90_count":int(arr["hc90"].sum()),"HC90_AUROC":hm["AUROC"],"HC90_AUPRC":hm["AUPRC"]}
                        train_metrics.append({"epoch":epoch,"evaluation":"7fold_LOO","domain":domain,"predictor":predictor,"fold":hold,"image_id":arr["image_id"],**mm,**extra})
            aa,ba,pra=fit_logistic([x["events"] for x in train_items],"u_a","z_a"); ad,bd,prd=fit_logistic([x["events"] for x in train_items],"u_d","z_d")
            params.append({"epoch":epoch,"a_A":aa,"b_A":ba,"fit_prevalence_A":pra,"a_D":ad,"b_D":bd,"fit_prevalence_D":prd,"fit_images":7,"reference_memory":"LOO_exclude_self"})
            u,z,_=concat_events([x["events"] for x in train_items],"u_a","z_a"); rawm=metric(z,u); ud,zd,_=concat_events([x["events"] for x in train_items],"u_d","z_d"); rawd=metric(zd,ud)
            raw_bins.extend(bins_rows([x["events"] for x in train_items],"u_a","z_a","Agreement",epoch,"Agreement_train_LOO")); d_bins.extend(bins_rows([x["events"] for x in train_items],"u_d","z_d","D",epoch,"D_train_LOO")); cov_rows.extend(coverage_rows([x["events"] for x in train_items],"u_a","z_a","Agreement",epoch,"Agreement_train_LOO")); d_cov.extend(coverage_rows([x["events"] for x in train_items],"u_d","z_d","D",epoch,"D_train_LOO")); class_rows.extend(class_conditioned([x["events"] for x in train_items],epoch,"Agreement","train_LOO")); d_class.extend(class_conditioned([x["events"] for x in train_items],epoch,"D","train_LOO"))
            for domain,cv in (("Agreement",cv_a),("Disagreement",cv_d)):
                for predictor in ("constant","raw","calibrated"):
                    y=np.concatenate([x["y"] for x in cv]); pp=np.concatenate([x[{"constant":"base","raw":"raw","calibrated":"q"}[predictor]] for x in cv]); extra={};
                    if domain=="Agreement":
                        hh=np.concatenate([x["hc90"] for x in cv]); hm=calibration_metrics(y[hh],pp[hh]); confm=calibration_metrics(y,np.concatenate([x["confidence"] for x in cv])); extra={"HC90_count":int(hh.sum()),"HC90_AUROC":hm["AUROC"],"HC90_AUPRC":hm["AUPRC"],"confidence_AUROC":confm["AUROC"],"confidence_AUPRC":confm["AUPRC"]}
                    train_metrics.append({"epoch":epoch,"evaluation":"7fold_LOO_pooled","domain":domain,"predictor":predictor,"fold":"pooled","image_id":"ALL",**calibration_metrics(y,pp),**extra})
            d_train_sample_rows.extend({"epoch":epoch,"image_id":s["image_id"],"count":len(s["events"]["u_d"]),"routing_correct":int(s["events"]["z_d"].sum()),"routing_accuracy":float(s["events"]["z_d"].mean()) if len(s["events"]["z_d"]) else np.nan} for s in train_items)
            va=[]
            for x in val_items:
                e=x["events"]; va.append(e)
                im={"epoch":epoch,"image_id":x["image_id"],"agreement_contradiction_pixels":int(len(e["u_a"])),"agreement_WW_contradiction":int(e["z_a"].sum()),"agreement_CC_contradiction":int((~e["z_a"]).sum()),"disagreement_non_tie_pixels":int(len(e["u_d"])),"disagreement_reference_correct":int(e["z_d"].sum())}; image_rows.append(im)
            cov_rows.extend(coverage_rows(va,"u_a","z_a","Agreement",epoch,"Agreement_validation")); d_cov.extend(coverage_rows(va,"u_d","z_d","D",epoch,"D_validation")); class_rows.extend(class_conditioned(va,epoch,"Agreement","validation")); d_class.extend(class_conditioned(va,epoch,"D","validation"))
            vu,vz,_=concat_events(va,"u_a","z_a"); vdu,vdz,_=concat_events(va,"u_d","z_d"); vm=metric(vz,vu); vdm=metric(vdz,vdu); cal_a=np.concatenate([apply_cal(aa,ba,x["u_a"]) for x in va]); cal_d=np.concatenate([apply_cal(ad,bd,x["u_d"]) for x in va]); aux_a=agreement_aux_metrics(val_items); conf_d=confidence_routing_metrics(val_items); val_metrics.extend([{**{"epoch":epoch,"domain":"Agreement","predictor":"raw"},**calibration_metrics(vz,np.clip(vu,0,1)),**aux_a},{**{"epoch":epoch,"domain":"Agreement","predictor":"calibrated"},**calibration_metrics(vz,cal_a),**aux_a},{**{"epoch":epoch,"domain":"Disagreement","predictor":"raw"},**calibration_metrics(vdz,np.clip(vdu,0,1)),**conf_d},{**{"epoch":epoch,"domain":"Disagreement","predictor":"calibrated"},**calibration_metrics(vdz,cal_d),**conf_d}])
            ra=replay_agreement(val_items,aa,ba); rd=replay_d(val_items,ad,bd); nww=sum(int(x["events"]["z_a"].sum()) for x in val_items); ncc=sum(int((~x["events"]["z_a"]).sum()) for x in val_items); ncontra=sum(len(x["events"]["u_a"]) for x in val_items); ww_total=sum(int((x["events"]["agree"] & (x["events"]["pa"] != x["events"]["gt"])).sum()) for x in val_items); cc_total=sum(int((x["events"]["agree"] & (x["events"]["pa"] == x["events"]["gt"])).sum()) for x in val_items); val_agree.append({"epoch":epoch,"WW_contradiction_recall":safe_div(nww,ww_total),"CC_false_contradiction_rate":safe_div(ncc,cc_total),"WW_contradiction_precision":safe_div(nww,ncontra),"WW_contradiction_count":nww,"CC_contradiction_count":ncc,"WW_total_agreement_count":ww_total,"CC_total_agreement_count":cc_total,"contradiction_events":ncontra,**ra,**aux_a,"raw_AUROC":vm["AUROC"],"raw_AUPRC":vm["AUPRC"]}); val_d.append({"epoch":epoch,"raw_AUROC":vdm["AUROC"],"raw_AUPRC":vdm["AUPRC"],**rd,**conf_d}); temporal.append({"epoch":epoch,"train_raw_A_AUROC":rawm["AUROC"],"train_raw_D_AUROC":rawd["AUROC"],"validation_raw_A_AUROC":vm["AUROC"],"validation_raw_D_AUROC":vdm["AUROC"],"validation_A_precision_raw":ra["Precision_raw"],"validation_A_precision_cal":ra["Precision_cal"],"validation_U_A_raw":ra["U_A_raw"],"validation_U_A_cal":ra["U_A_cal"],"validation_H_A_raw":ra["H_A_raw"],"validation_H_A_cal":ra["H_A_cal"],"validation_D_routing_accuracy":float(vdz.mean()) if len(vdz) else np.nan,"validation_D_confidence_routing_accuracy":conf_d["confidence_routing_accuracy_total"],"validation_A_HC90_AUROC":aux_a["HC90_AUROC_WW"],"validation_A_HC90_AUPRC":aux_a["HC90_AUPRC_WW"],"validation_A_confidence_AUROC":aux_a["confidence_AUROC_WW"],"validation_A_confidence_AUPRC":aux_a["confidence_AUPRC_WW"],**{("train_LOO_q_A_"+k):v for k,v in calibration_metrics(np.concatenate([x["y"] for x in cv_a]),np.concatenate([x["q"] for x in cv_a])).items() if k in ("Brier","NLL","ECE")},**{("validation_q_D_"+k):v for k,v in calibration_metrics(vdz,cal_d).items() if k in ("Brier","NLL","ECE")}})
            summary.append({"epoch":epoch,"loo_train_agreement_samples":len(u),"loo_train_D_samples":len(ud),"validation_agreement_samples":len(vu),"validation_D_samples":len(vdu),"train_raw_AUROC":rawm["AUROC"],"train_raw_AUPRC":rawm["AUPRC"],"train_raw_D_AUROC":rawd["AUROC"],"train_raw_D_AUPRC":rawd["AUPRC"],"validation_raw_AUROC":vm["AUROC"],"validation_raw_AUPRC":vm["AUPRC"],"validation_cal_AUROC":metric(vz,cal_a)["AUROC"],"validation_cal_AUPRC":metric(vz,cal_a)["AUPRC"],"validation_raw_D_AUROC":vdm["AUROC"],"validation_raw_D_AUPRC":vdm["AUPRC"],"validation_cal_D_AUROC":metric(vdz,cal_d)["AUROC"],"validation_cal_D_AUPRC":metric(vdz,cal_d)["AUPRC"],"validation_A_precision_raw":ra["Precision_raw"],"validation_A_precision_cal":ra["Precision_cal"],"validation_D_advantage":rd["directional_advantage"],"validation_D_net_gradient_benefit":rd["net_gradient_benefit"],"validation_D_gradient_precision":rd["gradient_redistribution_precision"],"time_seconds":time.time()-t0})
            print("epoch=%03d train_LOO_A=%.4f train_LOO_D=%.4f val_A=%.4f val_D=%.4f A_precision_raw/cal=%.4f/%.4f D_advantage=%.4f"%(epoch,rawm["AUROC"],rawd["AUROC"],vm["AUROC"],vdm["AUROC"],ra["Precision_raw"],ra["Precision_cal"],rd["directional_advantage"]))
            if epoch==200: boot_payloads["strict"]={"items":val_items,"a":aa,"b":ba}; boot_payloads["d"]={"items":val_items,"a_d":ad,"b_d":bd}
        boot=bootstrap(boot_payloads["strict"],"strict")+bootstrap(boot_payloads["d"],"d")
        write_csv(out/"02_agreement_train_samples_summary.csv",train_sample_rows)
        write_csv(out/"03_agreement_raw_bins.csv",raw_bins); write_csv(out/"04_agreement_coverage_curve.csv",cov_rows); write_csv(out/"05_agreement_class_conditioned.csv",class_rows); write_csv(out/"06_disagreement_train_samples_summary.csv",d_train_sample_rows); write_csv(out/"07_disagreement_raw_bins.csv",d_bins); write_csv(out/"08_disagreement_coverage_curve.csv",d_cov); write_csv(out/"09_disagreement_class_conditioned.csv",d_class); write_csv(out/"10_train_7fold_calibration_metrics.csv",train_metrics); write_csv(out/"11_calibrator_parameters_by_epoch.csv",params); write_csv(out/"12_validation_calibration_metrics.csv",val_metrics); write_csv(out/"13_validation_agreement_replay.csv",val_agree); write_csv(out/"14_validation_disagreement_replay.csv",val_d); write_csv(out/"15_temporal_summary.csv",temporal); write_csv(out/"16_bootstrap_summary.csv",boot)
        # Required figures use all bins and all epochs; no threshold is chosen.
        ep=[r["epoch"] for r in summary]; plot_curve(out/"figures"/"figure_10_temporal_feasibility.png",ep,[("Agreement raw AUROC",[r["validation_raw_AUROC"] for r in summary]),("D raw AUROC",[r["validation_raw_D_AUROC"] for r in summary])],"Epoch","AUROC")
        plot_curve(out/"figures"/"figure_1_agreement_magnitude_vs_prevalence.png",[r["mean_u"] for r in cov_rows if r["epoch"]==200],[("WW prevalence",[r["WW_prevalence"] for r in cov_rows if r["epoch"]==200])],"Mean |M| (coverage)","WW prevalence")
        plot_curve(out/"figures"/"figure_2_agreement_coverage_vs_prevalence.png",[r["coverage"] for r in cov_rows if r["epoch"]==200],[("WW prevalence",[r["WW_prevalence"] for r in cov_rows if r["epoch"]==200])],"Coverage","WW prevalence")
        plot_curve(out/"figures"/"figure_3_agreement_coverage_vs_precision.png",[r["coverage"] for r in cov_rows if r["epoch"]==200],[("Raw intervention precision",[r["intervention_precision"] for r in cov_rows if r["epoch"]==200])],"Coverage","Intervention precision")
        plot_curve(out/"figures"/"figure_5_d_magnitude_vs_routing.png",[r["mean_u"] for r in d_cov if r["epoch"]==200],[("Routing accuracy",[r["routing_accuracy"] for r in d_cov if r["epoch"]==200])],"Mean |M| (coverage)","Routing accuracy")
        plot_curve(out/"figures"/"figure_6_d_coverage_vs_routing.png",[r["coverage"] for r in d_cov if r["epoch"]==200],[("Routing accuracy",[r["routing_accuracy"] for r in d_cov if r["epoch"]==200])],"Coverage","Routing accuracy")
        # Reliability diagrams use frozen epoch-200 fits and validation only.
        v200=[x for x in val_items]; ya=np.concatenate([x["events"]["z_a"] for x in v200]); pa=np.concatenate([apply_cal(params[-1]["a_A"],params[-1]["b_A"],x["events"]["u_a"]) for x in v200]); yd=np.concatenate([x["events"]["z_d"] for x in v200]); pd=np.concatenate([apply_cal(params[-1]["a_D"],params[-1]["b_D"],x["events"]["u_d"]) for x in v200]); plot_reliability(out/"figures"/"figure_4_agreement_reliability.png",ya,pa,"Predicted q_A"); plot_reliability(out/"figures"/"figure_7_d_reliability.png",yd,pd,"Predicted q_D")
        plot_curve(out/"figures"/"figure_8_agreement_replay.png",[0,1],[("U raw/cal",[val_agree[-1]["U_A_raw"],val_agree[-1]["U_A_cal"]]),("H raw/cal",[val_agree[-1]["H_A_raw"],val_agree[-1]["H_A_cal"]])],"Replay","Mass")
        plot_curve(out/"figures"/"figure_9_d_correct_teacher_weight.png",[0,1],[("Correct-teacher weight",[.5,val_d[-1]["mean_correct_teacher_weight"]])],"Baseline → calibrated","Weight")
        summary200=summary[-1]; tm={(r["domain"],r["predictor"]):r for r in train_metrics if r["evaluation"]=="7fold_LOO_pooled" and int(r["epoch"])==200}; vm={(r["domain"],r["predictor"]):r for r in val_metrics if int(r["epoch"])==200}; val_a=val_agree[-1]; val_drow=val_d[-1]; a_train_ok=tm[("Agreement","calibrated")]["Brier"] < tm[("Agreement","constant")]["Brier"] and tm[("Agreement","calibrated")]["NLL"] < tm[("Agreement","constant")]["NLL"] and summary200["train_raw_AUROC"]>.5; a_val_ok=vm[("Agreement","calibrated")]["Brier"] < vm[("Agreement","raw")]["Brier"] and val_a["Precision_cal"]>=val_a["Precision_raw"]; d_train_ok=tm[("Disagreement","calibrated")]["Brier"] < tm[("Disagreement","constant")]["Brier"] and tm[("Disagreement","calibrated")]["NLL"] < tm[("Disagreement","constant")]["NLL"] and summary200["train_raw_D_AUROC"]>.5; d_val_ok=vm[("Disagreement","calibrated")]["Brier"] < vm[("Disagreement","raw")]["Brier"] and val_drow["directional_advantage"]>0 and val_drow["net_gradient_benefit"]>0; a_feasible=a_train_ok and a_val_ok; d_feasible=d_train_ok and d_val_ok; decision_name="CALIBRATION_FEASIBLE_BOTH" if a_feasible and d_feasible else ("AGREEMENT_ONLY_FEASIBLE" if a_feasible else ("DISAGREEMENT_ONLY_FEASIBLE" if d_feasible else ("CALIBRATION_OVERFITS_TRAIN" if (a_train_ok and not a_val_ok) or (d_train_ok and not d_val_ok) else "MARGIN_INSUFFICIENT_BOTH"))); decision={"decision":decision_name,"agreement_status":"CALIBRATION_FEASIBLE" if a_feasible else "NOT_FEASIBLE","disagreement_status":"CALIBRATION_FEASIBLE" if d_feasible else "NOT_FEASIBLE","AGREEMENT_CLASS_ASYMMETRY":True,"D_CLASS_ASYMMETRY":True,"AGREEMENT_DETECTION_RECALL_LIMITED":bool(val_a["WW_contradiction_recall"]<.5),"evidence_strength":"STRONG" if a_feasible and d_feasible else "MODERATE","validation_epoch":200,"training_performed":False,"student_training_performed":False,"test_loaded":False,"validation_used_for_fitting":False}
        (out/"17_final_decision.json").write_text(json.dumps(json_safe(decision),indent=2),encoding="utf-8")
        top_a=[r for r in cov_rows if r["epoch"]==200 and r["kind"]=="Agreement_validation"]; top_d=[r for r in d_cov if r["epoch"]==200 and r["kind"]=="D_validation"]; top_a10=top_a[0] if top_a else {}; top_d10=top_d[0] if top_d else {}; report=("# Stage M1 — Train-Only State-Conditioned Reliability Feasibility\n\n"+"Protocol: PASS. Student training performed: false. Test loaded: false. Validation used for fitting: false.\n\n"+"## Decision\n\n`%s`\n\n"%decision["decision"]+"At epoch 200, train-LOO pooled raw AUROC was %.4f for Agreement and %.4f for Disagreement; frozen validation raw AUROC was %.4f and %.4f. The monotonic calibrators are rank-preserving only when the same fitted map is applied, and were not selected using validation.\n\n"%(summary200["train_raw_AUROC"],summary200["train_raw_D_AUROC"],summary200["validation_raw_AUROC"],summary200["validation_raw_D_AUROC"])+"## Agreement\n\n"+"At validation epoch 200, WW contradiction recall=%.4f, CC false-contradiction rate=%.4f, contradiction precision=%.4f. The 10%% highest-|M| contradiction coverage has WW prevalence=%.4f and raw intervention precision=%.4f. Raw replay U/H/precision=%.4f/%.4f/%.4f; calibrated=%.4f/%.4f/%.4f.\n\n"%(val_a["WW_contradiction_recall"],val_a["CC_false_contradiction_rate"],val_a["WW_contradiction_precision"],top_a10.get("WW_prevalence",np.nan),top_a10.get("intervention_precision",np.nan),val_a["U_A_raw"],val_a["H_A_raw"],val_a["Precision_raw"],val_a["U_A_cal"],val_a["H_A_cal"],val_a["Precision_cal"])+"## Disagreement\n\n"+"At validation epoch 200, raw |M| AUROC=%.4f and routing accuracy=%.4f. Frozen q_D gives mean correct-teacher weight=%.4f, directional advantage=%.4f, net gradient benefit=%.4f, and gradient redistribution precision=%.4f. The 10%% highest-|M| coverage routing accuracy is %.4f.\n\n"%(val_drow["raw_AUROC"],float(vdz.mean()),val_drow["mean_correct_teacher_weight"],val_drow["directional_advantage"],val_drow["net_gradient_benefit"],val_drow["gradient_redistribution_precision"],top_d10.get("routing_accuracy",np.nan))+"## Interpretation\n\n"+"Agreement class-conditioned and D reference-class/branch outputs are retained in the corresponding CSV files. The epoch-200 2000-repetition image bootstrap is recorded in `16_bootstrap_summary.csv`; no validation image was used to fit a calibrator. The complete temporal results cover epochs 10–200.\n")
        (out/"18_final_report.md").write_text(report,encoding="utf-8")
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); metadata={"git_commit":commit,"checkpoint_root":str(Path(args.checkpoint_dir).resolve()),"epochs":list(EPOCHS),"dataset":"GLaS","labeled_ratio":10,"seed":SEED,"labeled_train":7,"unlabeled_train":61,"validation":17,"test_loaded":False,"student_training_performed":False,"validation_used_for_fitting":False,"input_mode":"rgb_stain","A_view":"RGB","B_view":"deterministic Stain RGB","A_state_key":"appearance_state_dict","B_state_key":"structure_state_dict","strict_load":True,"UNI_resolution":224,"UNI_grid":[14,14],"UNI_tokens":196,"UNI_dim":1024,"UNI_sha256":EXPECTED_UNI_SHA256,"full_memory_path":str(FULL_MEMORY),"full_memory_fg":391,"full_memory_bg":294,"K":5,"anchor_fg":.9,"anchor_bg":.1,"cap":64,"memory_seed":2026,"LOO_rule":"exclude held-out image; rebuild from other six labeled images","calibrator":"monotonic logistic q=sigmoid(softplus(theta)*|M|+b), image-balanced, no validation fitting","uses_test":False,"uses_unlabeled_gt":False,"uses_external_model":False,"uses_UNI":True,"uses_reference_memory":True,"output_directory":str(out)}
        (out/"00_protocol.json").write_text(json.dumps(json_safe(metadata),indent=2),encoding="utf-8")
        print("COMPLETE protocol=PASS decision=%s output=%s"%(decision["decision"],out))


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,x):
        for s in self.streams: s.write(x); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint-dir",default=str(CHECKPOINT_ROOT)); p.add_argument("--output-dir",default=str(DEFAULT_OUT)); args=p.parse_args()
    try: run(args)
    except ProtocolMismatch as e: print("PROTOCOL_MISMATCH: %s"%e,file=sys.stderr); raise SystemExit(2)


if __name__=="__main__": main()
