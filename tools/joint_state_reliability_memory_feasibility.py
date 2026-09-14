"""Frozen Joint-State Reliability Memory feasibility diagnostic.

This diagnostic deliberately uses only the original RGB+Stain U-Net pair.
It builds a fresh, state-conditioned memory from the seven labeled training
images for every synchronized checkpoint and evaluates it on validation only.
No UNI, reference memory, State-Aware code, test loader, optimizer, or gradient
operation is imported or used.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from models.getnetwork import get_network  # noqa: E402
from tools.joint_state_error_diagnosis import joint_high_confidence_mask  # noqa: E402


DATASET = "GlaS"
INPUT_MODE = "rgb_stain"
NETWORK = "unet"
EPOCHS = tuple(range(10, 201, 10))
PIXEL_SIZE = (256, 256)
SPLIT_SEED = 24
MEMORY_SEED = 1337
CAP_DEFAULT = 256
K_DEFAULT = 5
HC_THRESHOLD = 0.90
THRESHOLD = 0.5
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 1337
BOOTSTRAP_EPOCHS = (200,)
CHECKPOINT_ROOT = ROOT / "checkpoints" / "semi_10_seed_24_rgbstain" / "GLaS" / "CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image-input_mode=rgb_stain"
DEFAULT_OUT = ROOT / "diagnostics" / "joint_state_reliability_memory_feasibility"
MANIFEST_ROOT = ROOT / "datasets" / "GLaS" / "manifests" / "paper_seed_2026"

STRICT_AGREEMENT_STATES = ("CC_pred0", "WW_pred0", "CC_pred1", "WW_pred1")
STRICT_DISAGREEMENT_STATES = ("Acorrect_10", "Bcorrect_10", "Acorrect_01", "Bcorrect_01")
HARD_AGREEMENT_STATES = ("Reliable_pred0", "Confusable_pred0", "Reliable_pred1", "Confusable_pred1")
ALL_STATES = STRICT_AGREEMENT_STATES + STRICT_DISAGREEMENT_STATES
ALL_HARD_STATES = HARD_AGREEMENT_STATES + STRICT_DISAGREEMENT_STATES


class ProtocolMismatch(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ProtocolMismatch(message)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {k: float("nan") for k in ("mean", "std", "median", "q05", "q25", "q75", "q95")}
    return dict(zip(("mean", "std", "median", "q05", "q25", "q75", "q95"),
                    (float(x.mean()), float(x.std()), float(np.median(x)), *map(float, np.percentile(x, [5, 25, 75, 95])))))


def binary_metric(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=bool).ravel(); score = np.asarray(score, dtype=float).ravel()
    valid = np.isfinite(score)
    y, score = y[valid], score[valid]
    if not y.size or np.unique(y).size < 2:
        return {"status": "UNDEFINED_SINGLE_CLASS", "valid_count": int(y.size), "auroc": float("nan"), "auprc": float("nan")}
    return {"status": "OK", "valid_count": int(y.size), "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score))}


def metric_fields(prefix: str, y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    m = binary_metric(y, score)
    return {prefix + "_status": m["status"], prefix + "_AUROC": m["auroc"], prefix + "_AUPRC": m["auprc"], prefix + "_valid_count": m["valid_count"]}


def read_manifest_count(path: Path, required_mask: bool) -> int:
    if not path.is_file():
        fail("manifest missing: %s" % path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not all(row.get("image") for row in rows):
        fail("manifest has an invalid image row: %s" % path)
    if required_mask and not all(row.get("mask") for row in rows):
        fail("validation/labeled manifest lacks mask: %s" % path)
    return len(rows)


def scan_checkpoints(root: Path) -> List[Tuple[int, Path]]:
    if not root.is_dir():
        fail("checkpoint root missing: %s" % root)
    found: Dict[int, Path] = {}
    for path in sorted((root / "milestones").glob("epoch_*.pth")):
        match = re.fullmatch(r"epoch_(\d{3})\.pth", path.name)
        if match:
            epoch = int(match.group(1))
            if epoch in found:
                fail("duplicate checkpoint epoch %d" % epoch)
            found[epoch] = path
    if tuple(sorted(found)) != EPOCHS:
        fail("expected synchronized epochs 10..200 step10, found %s" % sorted(found))
    for epoch, path in found.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict): fail("checkpoint %d payload is not mapping" % epoch)
        if not isinstance(payload.get("appearance_state_dict"), dict) or not isinstance(payload.get("structure_state_dict"), dict):
            fail("checkpoint %d lacks appearance_state_dict/structure_state_dict" % epoch)
        for key, expected in (("dataset_name", "GlaS"), ("labeled_ratio", 10), ("seed", 24), ("input_mode", "rgb_stain"), ("network", "unet")):
            actual = payload.get(key)
            ok = str(actual).lower() == expected.lower() if isinstance(expected, str) else actual == expected
            if not ok: fail("checkpoint %d %s=%r expected %r" % (epoch, key, actual, expected))
    return [(e, found[e]) for e in EPOCHS]


def build_loader(manifest: Path, cfg: Mapping[str, Any], sup: bool) -> DataLoader:
    normalize = data_normalize_2d(cfg["MEAN"], cfg["STD"])
    ds = imagefloder_itn(manifest_path=str(manifest), data_transform_1=data_transform_2d()["val"], data_normalize_1=normalize, sup=sup, input_mode=INPUT_MODE, training=False)
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)


def state_masks(pa: np.ndarray, pb: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    pa = np.asarray(pa, dtype=bool); pb = np.asarray(pb, dtype=bool); gt = np.asarray(gt, dtype=bool)
    if pa.shape != pb.shape or pa.shape != gt.shape: fail("A/B/GT shape mismatch")
    cc = (pa == pb) & (pa == gt); ww = (pa == pb) & (pa != gt)
    ac = (pa != pb) & (pa == gt); bc = (pa != pb) & (pb == gt)
    if not np.array_equal(cc | ww | ac | bc, np.ones_like(gt, dtype=bool)) or np.any((cc|ww) & (ac|bc)):
        fail("joint state partition is invalid")
    return {"cc": cc, "ww": ww, "ac": ac, "bc": bc, "agreement": cc | ww, "disagreement": ac | bc}


def capture_model(model: torch.nn.Module, x: torch.Tensor, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    module = dict(model.named_modules()).get("Conv_1x1")
    if module is None: fail("feature layer Conv_1x1 not found")
    def hook(_module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        captured["feature"] = inputs[0]
    handle = module.register_forward_pre_hook(hook)
    try:
        output = model(x)
    finally:
        handle.remove()
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if "feature" not in captured: fail("feature hook did not capture %s" % layer_name)
    feature = captured["feature"]
    if logits.ndim != 4 or feature.ndim != 4 or logits.shape[-2:] != feature.shape[-2:]:
        fail("feature/logit shape cannot be aligned: %s vs %s" % (tuple(feature.shape), tuple(logits.shape)))
    return logits, feature


def collect_pair(model_a: torch.nn.Module, model_b: torch.nn.Module, loader: DataLoader, device: torch.device, layer_name: str) -> List[Dict[str, Any]]:
    rows=[]
    with torch.inference_mode():
        for batch in loader:
            if set(("image_rgb", "image_stain", "mask", "ID")) - set(batch): fail("RGB/Stain loader fields incomplete")
            xa=batch["image_rgb"].to(device); xb=batch["image_stain"].to(device)
            logits_a, feat_a=capture_model(model_a,xa,layer_name); logits_b, feat_b=capture_model(model_b,xb,layer_name)
            if tuple(logits_a.shape[-2:]) != PIXEL_SIZE or tuple(logits_b.shape[-2:]) != PIXEL_SIZE: fail("logits are not 256x256")
            pa=torch.softmax(logits_a,1)[:,1].cpu().numpy().astype(np.float32); pb=torch.softmax(logits_b,1)[:,1].cpu().numpy().astype(np.float32)
            for i in range(int(logits_a.shape[0])):
                rows.append({"image_id":str(batch["ID"][i]),"gt":batch["mask"][i].cpu().numpy().astype(bool),"prob_a":pa[i],"prob_b":pb[i],"feature_a":feat_a[i].cpu().numpy().astype(np.float32),"feature_b":feat_b[i].cpu().numpy().astype(np.float32)})
    return rows


def normalized_joint(row: Mapping[str, Any]) -> np.ndarray:
    fa=np.asarray(row["feature_a"],dtype=np.float32); fb=np.asarray(row["feature_b"],dtype=np.float32)
    if fa.shape != fb.shape or fa.shape[-2:] != PIXEL_SIZE: fail("A/B feature mismatch %s/%s"%(fa.shape,fb.shape))
    za=fa/(np.linalg.norm(fa,axis=0,keepdims=True)+1e-12); zb=fb/(np.linalg.norm(fb,axis=0,keepdims=True)+1e-12)
    return np.concatenate((za,zb),axis=0).transpose(1,2,0).reshape(-1,fa.shape[0]*2)/math.sqrt(2.0)


def stable_rng(epoch: int, image_id: str, state: str) -> np.random.Generator:
    digest=hashlib.sha256(("%d|%s|%s|%d"%(epoch,image_id,state,MEMORY_SEED)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],"little"))


def memory_for_epoch(rows: Sequence[Mapping[str, Any]], epoch: int, cap: int, mode: str) -> Tuple[Dict[str,np.ndarray],List[Dict[str,Any]],Dict[str,int]]:
    banks: Dict[str,List[np.ndarray]]={s:[] for s in (ALL_STATES if mode=="strict_state" else ALL_HARD_STATES)}
    counts=[]; raw_counts={s:0 for s in banks}
    for row in rows:
        pa=row["prob_a"]>=THRESHOLD; pb=row["prob_b"]>=THRESHOLD; gt=row["gt"]; st=state_masks(pa,pb,gt); joint=normalized_joint(row); n=gt.size
        masks: Dict[str,np.ndarray]={"CC_pred0":st["cc"]&~pa,"WW_pred0":st["ww"]&~pa,"CC_pred1":st["cc"]&pa,"WW_pred1":st["ww"]&pa,"Acorrect_10":st["ac"]&pa&~pb,"Bcorrect_10":st["bc"]&pa&~pb,"Acorrect_01":st["ac"]&~pa&pb,"Bcorrect_01":st["bc"]&~pa&pb}
        if mode=="hard_confusable":
            fg=(row["prob_a"]+row["prob_b"])/2; bg=1-fg
            masks={"Reliable_pred0":st["cc"]&~pa,"Reliable_pred1":st["cc"]&pa,"Confusable_pred0":gt & bg.astype(bool) if False else (gt & ~np.zeros_like(gt,dtype=bool)),"Confusable_pred1":~gt}
            # Confusable anchors are selected from the opposite GT class by
            # the stated probability ordering, not by a hidden state fallback.
            masks["Confusable_pred0"]=gt; masks["Confusable_pred1"]=~gt
            masks.update({"Acorrect_10":st["ac"]&pa&~pb,"Bcorrect_10":st["bc"]&pa&~pb,"Acorrect_01":st["ac"]&~pa&pb,"Bcorrect_01":st["bc"]&~pa&pb})
        for state, mask in masks.items():
            # For hard-confusable banks, selection order is hard-error order;
            # for strict banks, deterministic random subsampling is used.
            idx=np.flatnonzero(mask.ravel()); raw_counts[state]+=int(idx.size)
            if mode=="hard_confusable" and state=="Confusable_pred1": idx=idx[np.argsort(-((row["prob_a"]+row["prob_b"])/2).ravel()[idx],kind="stable")]
            elif mode=="hard_confusable" and state=="Confusable_pred0": idx=idx[np.argsort(-((1-(row["prob_a"]+row["prob_b"])/2)).ravel()[idx],kind="stable")]
            elif idx.size>cap: idx=stable_rng(epoch,row["image_id"],state).choice(idx,size=cap,replace=False)
            if idx.size>cap: idx=idx[:cap]
            banks[state].append(joint[idx])
            counts.append({"epoch":epoch,"image_id":row["image_id"],"state_name":state,"raw_count":int(raw_counts[state]-sum(x["raw_count"] for x in counts if x["image_id"]==row["image_id"] and x["state_name"]==state)),"kept_count":int(idx.size),"mode":mode})
    final={s:(np.concatenate(v,axis=0) if v and sum(x.shape[0] for x in v) else np.empty((0,128),dtype=np.float32)) for s,v in banks.items()}
    bank_counts={s:int(v.shape[0]) for s,v in final.items()}
    return final,counts,bank_counts


def topk_score(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    if query.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if bank.shape[0] < k: return np.full(query.shape[0],np.nan,dtype=np.float32)
    # Similarity is evaluated on CUDA in chunks.  This is the same joint
    # cosine/top-K operation, but avoids an unbounded CPU matrix multiply for
    # every 256x256 validation image.
    if torch.cuda.is_available():
        q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to("cuda")
        b = torch.from_numpy(np.asarray(bank, dtype=np.float32)).to("cuda")
        chunks = []
        with torch.inference_mode():
            for start in range(0, q.shape[0], 8192):
                chunks.append((q[start:start+8192] @ b.t()).topk(k, dim=1).values.mean(dim=1).cpu())
        del q, b
        return torch.cat(chunks).numpy().astype(np.float32)
    out=[]
    for start in range(0,query.shape[0],8192):
        sim=query[start:start+8192]@bank.T; part=np.partition(sim,-k,axis=1)[:,-k:]; out.append(part.mean(axis=1))
    return np.concatenate(out).astype(np.float32)


def score_query(query: np.ndarray, banks: Mapping[str,np.ndarray], mode: str, k: int) -> Dict[str,np.ndarray]:
    if mode=="strict_state": pairs={"0":("CC_pred0","WW_pred0"),"1":("CC_pred1","WW_pred1"),"10":("Acorrect_10","Bcorrect_10"),"01":("Acorrect_01","Bcorrect_01")}
    else: pairs={"0":("Reliable_pred0","Confusable_pred0"),"1":("Reliable_pred1","Confusable_pred1"),"10":("Acorrect_10","Bcorrect_10"),"01":("Acorrect_01","Bcorrect_01")}
    out={}
    for key,(positive,negative) in pairs.items():
        # Agreement uses error minus reliable; disagreement uses A-correct minus B-correct.
        if key in ("0","1"): out[key]=topk_score(query,banks[negative],k)-topk_score(query,banks[positive],k)
        else: out[key]=topk_score(query,banks[positive],k)-topk_score(query,banks[negative],k)
    return out


def score_one(query: np.ndarray, banks: Mapping[str, np.ndarray], mode: str, key: str, k: int) -> np.ndarray:
    if mode == "strict_state":
        pairs = {"0": ("CC_pred0", "WW_pred0"), "1": ("CC_pred1", "WW_pred1"), "10": ("Acorrect_10", "Bcorrect_10"), "01": ("Acorrect_01", "Bcorrect_01")}
    else:
        pairs = {"0": ("Reliable_pred0", "Confusable_pred0"), "1": ("Reliable_pred1", "Confusable_pred1"), "10": ("Acorrect_10", "Bcorrect_10"), "01": ("Acorrect_01", "Bcorrect_01")}
    positive, negative = pairs[key]
    if key in ("0", "1"):
        return topk_score(query, banks[negative], k) - topk_score(query, banks[positive], k)
    return topk_score(query, banks[positive], k) - topk_score(query, banks[negative], k)


def evaluate_image(row: Mapping[str,Any], banks: Mapping[str,np.ndarray], mode: str, k: int) -> Dict[str,Any]:
    pa=(row["prob_a"]>=THRESHOLD).ravel(); pb=(row["prob_b"]>=THRESHOLD).ravel(); gt=np.asarray(row["gt"],dtype=bool).ravel(); st=state_masks(pa,pb,gt); joint=normalized_joint(row)
    # Keep every evaluation object in the same flattened pixel coordinate
    # system as the joint feature query.  This is only a reshape; the feature
    # and GT spatial mapping remains the validation 256x256 identity.
    row_prob_a=np.asarray(row["prob_a"],dtype=np.float32).ravel(); row_prob_b=np.asarray(row["prob_b"],dtype=np.float32).ravel()
    conf_a=np.maximum(row_prob_a,1-row_prob_a); conf_b=np.maximum(row_prob_b,1-row_prob_b); jconf=np.minimum(conf_a,conf_b); hc=joint_high_confidence_mask(row_prob_a,row_prob_b,HC_THRESHOLD).ravel()
    agreement_score=np.full(gt.size,np.nan,dtype=np.float32); mask0=st["agreement"]&~pa; mask1=st["agreement"]&pa; agreement_score[mask0]=score_one(joint[mask0],banks,mode,"0",k); agreement_score[mask1]=score_one(joint[mask1],banks,mode,"1",k)
    hard_score=agreement_score.copy()
    # Query-bank selection is prediction-pattern-only.  Validation GT is not
    # consulted until the subsequent routing correctness evaluation.
    routing_score=np.full(gt.size,np.nan,dtype=np.float32); mask10=pa&~pb; mask01=~pa&pb; routing_score[mask10]=score_one(joint[mask10],banks,mode,"10",k); routing_score[mask01]=score_one(joint[mask01],banks,mode,"01",k)
    # The hard-confusable score is identical on D by protocol, but retain it
    # separately to keep primary and secondary result namespaces independent.
    d=st["disagreement"]; route_pred=np.where(routing_score>0,pa,~pa); route_tie=d & (routing_score==0); route_correct=d & ~route_tie & (route_pred==pa==gt) if False else (d & ~route_tie & ((routing_score>0)&st["ac"] | (routing_score<0)&st["bc"]))
    conf_score=conf_a-conf_b; conf_tie=d&(conf_score==0); conf_correct=d&~conf_tie&(((conf_score>0)&st["ac"])|((conf_score<0)&st["bc"]))
    return {"image_id":row["image_id"],"gt":gt,"state":st,"agreement_score":agreement_score,"hard_score":hard_score,"routing_score":routing_score,"confidence_score":1-jconf,"joint_conf":jconf,"hc":hc,"route_tie":route_tie,"route_correct":route_correct,"confidence_tie":conf_tie,"confidence_correct":conf_correct,"pa":pa,"pb":pb,"pred_pattern10":pa&~pb,"pred_pattern01":~pa&pb}


def aggregate_eval(evals: Sequence[Mapping[str,Any]], mode: str) -> Tuple[Dict[str,Any],List[Dict[str,Any]],Dict[str,Any]]:
    gt=np.concatenate([e["gt"] for e in evals]); st={k:np.concatenate([e["state"][k] for e in evals]) for k in ("cc","ww","ac","bc","agreement","disagreement")}; score=np.concatenate([e["agreement_score"] if mode=="strict_state" else e["hard_score"] for e in evals]); conf=np.concatenate([e["confidence_score"] for e in evals]); hc=np.concatenate([e["hc"] for e in evals]); jconf=np.concatenate([e["joint_conf"] for e in evals]); route=np.concatenate([e["routing_score"] for e in evals]); rc=np.concatenate([e["route_correct"] for e in evals]); rt=np.concatenate([e["route_tie"] for e in evals]); cscore=np.concatenate([e["confidence_score"] for e in evals]); ccorr=np.concatenate([e["confidence_correct"] for e in evals]); ctie=np.concatenate([e["confidence_tie"] for e in evals])
    ag=st["agreement"]; y=st["ww"]; hmask=ag&hc; pred0=ag&~np.concatenate([e["pa"] for e in evals]); pred1=ag&np.concatenate([e["pa"] for e in evals]);
    row={"mode":mode,"agreement_pixels":int(ag.sum()),"cc_pixels":int(st["cc"].sum()),"ww_pixels":int(st["ww"].sum()),"disagreement_pixels":int(st["disagreement"].sum())}
    row.update(metric_fields("agreement",y[ag],score[ag])); row.update(metric_fields("pred0",y[pred0],score[pred0])); row.update(metric_fields("pred1",y[pred1],score[pred1])); row.update(metric_fields("HC90",y[hmask],score[hmask])); row.update(metric_fields("confidence",y[ag],conf[ag])); row.update(metric_fields("confidence_HC90",y[hmask],conf[hmask]))
    for suffix,mask in (("",hmask&~pred1),("_pred1",hmask&pred1)):
        row["HC90_CC_count"+suffix]=int((mask&st["cc"]).sum()); row["HC90_WW_count"+suffix]=int((mask&st["ww"]).sum()); row["HC90_score_gap"+suffix]=float(np.nanmean(score[mask&st["ww"]])-np.nanmean(score[mask&st["cc"]])) if (mask&st["ww"]).any() and (mask&st["cc"]).any() else float("nan")
    row["score_gap"]=float(np.nanmean(score[ag&st["ww"]])-np.nanmean(score[ag&st["cc"]])) if (ag&st["ww"]).any() and (ag&st["cc"]).any() else float("nan")
    row["mean_score_CC"]=float(np.nanmean(score[ag&st["cc"]])) if (ag&st["cc"]).any() else float("nan"); row["mean_score_WW"]=float(np.nanmean(score[ag&st["ww"]])) if (ag&st["ww"]).any() else float("nan"); row["median_score_CC"]=float(np.nanmedian(score[ag&st["cc"]])) if np.isfinite(score[ag&st["cc"]]).any() else float("nan"); row["median_score_WW"]=float(np.nanmedian(score[ag&st["ww"]])) if np.isfinite(score[ag&st["ww"]]).any() else float("nan")
    route_total=int(st["disagreement"].sum()); route_non=int((st["disagreement"]&~rt).sum()); conf_non=int((st["disagreement"]&~ctie).sum())
    row.update({"routing_accuracy_total":safe_div(float(rc.sum()),route_total),"routing_accuracy_non_tie":safe_div(float(rc.sum()),route_non),"routing_tie_rate":safe_div(float(rt.sum()),route_total),"confidence_routing_accuracy_total":safe_div(float(ccorr.sum()),route_total),"confidence_routing_accuracy_non_tie":safe_div(float(ccorr.sum()),conf_non),"confidence_routing_tie_rate":safe_div(float(ctie.sum()),route_total)})
    for name,mask in (("gt_fg",st["disagreement"]&gt),("gt_bg",st["disagreement"]&~gt),("a_correct",st["ac"]),("b_correct",st["bc"]),("pattern10",st["disagreement"]&np.concatenate([e["pred_pattern10"] for e in evals])),("pattern01",st["disagreement"]&np.concatenate([e["pred_pattern01"] for e in evals]))): row["routing_accuracy_"+name]=safe_div(float(rc[mask].sum()),float(mask.sum()))
    row["routing_tie_count"]=int(rt.sum()); row["confidence_routing_tie_count"]=int(ctie.sum()); row["strict_state_status"]="OK" if row.get("agreement_status")=="OK" and all(np.isfinite(x) for x in [row.get("HC90_AUROC",np.nan)] if x is not None) else row.get("agreement_status")
    image_rows=[]
    for e in evals:
        agi=e["state"]["agreement"]; hci=agi&e["hc"]; si=e["agreement_score"] if mode=="strict_state" else e["hard_score"]
        ir={"image_id":e["image_id"],"agreement_pixels":int(agi.sum()),"cc_pixels":int(e["state"]["cc"].sum()),"ww_pixels":int(e["state"]["ww"].sum()),"hc90_cc_pixels":int((hci&e["state"]["cc"]).sum()),"hc90_ww_pixels":int((hci&e["state"]["ww"]).sum()),"disagreement_pixels":int(e["state"]["disagreement"].sum()),"routing_correct":int(e["route_correct"].sum()),"routing_total":int(e["state"]["disagreement"].sum()),"confidence_routing_correct":int(e["confidence_correct"].sum()),"confidence_routing_total":int(e["state"]["disagreement"].sum())}
        ir.update(metric_fields("agreement",e["state"]["ww"][agi],si[agi])); ir.update(metric_fields("hc90",e["state"]["ww"][hci],si[hci])); image_rows.append(ir)
    payload={"gt":gt,"state":st,"score":score,"conf":conf,"hc":hc,"route_correct":rc,"route_tie":rt,"confidence_correct":ccorr,"confidence_tie":ctie,"image_evals":evals}
    return row,image_rows,payload


def bootstrap_payload(payload: Mapping[str,Any], epoch: int, mode: str) -> List[Dict[str,Any]]:
    ev=payload["image_evals"]; rng=np.random.default_rng(BOOTSTRAP_SEED+epoch+(0 if mode=="strict_state" else 10000)); metrics=("AUROC","AUPRC","HC90_AUROC","HC90_AUPRC","score_gap","HC90_score_gap","routing_accuracy_total","confidence_routing_accuracy_total"); values={m:[] for m in metrics}
    for _ in range(BOOTSTRAP_REPS):
        chosen=[ev[int(i)] for i in rng.integers(0,len(ev),size=len(ev))]
        den=sum(int(x["state"]["disagreement"].sum()) for x in chosen); values["routing_accuracy_total"].append(safe_div(sum(int(x["route_correct"].sum()) for x in chosen),den)); values["confidence_routing_accuracy_total"].append(safe_div(sum(int(x["confidence_correct"].sum()) for x in chosen),den))
        ss=np.concatenate([x["agreement_score"] if mode=="strict_state" else x["hard_score"] for x in chosen]); yy=np.concatenate([x["state"]["ww"] for x in chosen]); hs=np.concatenate([x["hc"] for x in chosen]); finite=np.isfinite(ss)
        for metric,mask in (("",finite),("HC90_",finite&hs)):
            mm=binary_metric(yy[mask],ss[mask]); values[metric+"AUROC"].append(mm["auroc"]); values[metric+"AUPRC"].append(mm["auprc"])
            values[metric+"score_gap"].append(float(np.nanmean(ss[mask&yy])-np.nanmean(ss[mask&~yy])) if (mask&yy).any() and (mask&~yy).any() else np.nan)
    out=[]
    for metric in metrics:
        vals=np.asarray(values[metric],dtype=float); valid=np.isfinite(vals); out.append({"epoch":epoch,"mode":mode,"metric":metric,"estimate":float(np.nanmean(vals)) if valid.any() else float("nan"),"ci_low":float(np.nanpercentile(vals[valid],2.5)) if valid.any() else float("nan"),"ci_high":float(np.nanpercentile(vals[valid],97.5)) if valid.any() else float("nan"),"n_valid_bootstrap":int(valid.sum()),"n_requested_bootstrap":BOOTSTRAP_REPS})
    return out


def score_distribution_rows(evals: Sequence[Mapping[str,Any]], epoch: int, mode: str) -> List[Dict[str,Any]]:
    out=[]
    for subgroup in ("agreement_all","agreement_pred0","agreement_pred1","agreement_hc90","agreement_hc90_pred0","agreement_hc90_pred1"):
        for state in ("CC","WW"):
            vals=[]
            for e in evals:
                ag=e["state"]["agreement"]; pred=e["pa"]; mask=ag
                if subgroup=="agreement_pred0": mask&=~pred
                if subgroup=="agreement_pred1": mask&=pred
                if subgroup=="agreement_hc90": mask&=e["hc"]
                if subgroup=="agreement_hc90_pred0": mask&=e["hc"]&~pred
                if subgroup=="agreement_hc90_pred1": mask&=e["hc"]&pred
                mask &= e["state"]["cc"] if state=="CC" else e["state"]["ww"]
                score=e["agreement_score"] if mode=="strict_state" else e["hard_score"]; vals.append(score[mask])
            x=np.concatenate(vals) if vals else np.empty(0); out.append({"epoch":epoch,"mode":mode,"subgroup":subgroup,"state":state,"count":int(np.isfinite(x).sum()),**stats(x)})
    return out


class Tee:
    def __init__(self, *streams): self.streams=streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


def run(args: argparse.Namespace) -> None:
    out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()): fail("output directory is non-empty; refusing overwrite: %s"%out)
    checkpoints=scan_checkpoints(Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else CHECKPOINT_ROOT)
    labeled=MANIFEST_ROOT/"labeled_10.csv"; unlabeled=MANIFEST_ROOT/"unlabeled_10.csv"; validation=MANIFEST_ROOT/"validation.csv"
    n_lab=read_manifest_count(labeled,True); n_unsup=read_manifest_count(unlabeled,False); n_val=read_manifest_count(validation,True)
    if (n_lab,n_unsup,n_val)!=(7,61,17): fail("split counts %s/%s/%s != 7/61/17"%(n_lab,n_unsup,n_val))
    if not torch.cuda.is_available(): fail("CUDA is required")
    device=torch.device("cuda"); cfg=dataset_cfg(DATASET); lab_loader=build_loader(labeled,cfg,True); val_loader=build_loader(validation,cfg,True)
    if len(lab_loader.dataset)!=7 or len(val_loader.dataset)!=17: fail("loader counts differ from manifest")
    out.mkdir(parents=True,exist_ok=False); (out/"cache").mkdir(); log_path=out/"full_run.console.log"
    rows_summary=[]; hard_summary=[]; bank_rows=[]; image_rows=[]; bootstrap_rows=[]; distribution_rows=[]; key_rows=[]; feature_dim_a=feature_dim_b=feature_h=feature_w=None
    with log_path.open("w",encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.__stdout__,log)):
        print("Joint-State Reliability Memory Feasibility Diagnostic")
        print("protocol: frozen | uses_UNI=false | uses_reference_memory=false | uses_State_Aware=false | uses_test=false")
        print("split: labeled=%d unlabeled=%d validation=%d seed=%d"%(n_lab,n_unsup,n_val,SPLIT_SEED))
        for epoch,path in checkpoints:
            started=time.time(); payload=torch.load(path,map_location="cpu",weights_only=False); ma=get_network(NETWORK,cfg["IN_CHANNELS"],cfg["NUM_CLASSES"]); mb=get_network(NETWORK,cfg["IN_CHANNELS"],cfg["NUM_CLASSES"]); ma.load_state_dict(payload["appearance_state_dict"],strict=True); mb.load_state_dict(payload["structure_state_dict"],strict=True); ma.to(device).eval(); mb.to(device).eval()
            if ma.training or mb.training: fail("model.eval assertion failed")
            layer_name="Conv_1x1.input"; lab=collect_pair(ma,mb,lab_loader,device,layer_name); val=collect_pair(ma,mb,val_loader,device,layer_name)
            feature_dim_a=int(lab[0]["feature_a"].shape[0]); feature_dim_b=int(lab[0]["feature_b"].shape[0]); feature_h=int(lab[0]["feature_a"].shape[1]); feature_w=int(lab[0]["feature_a"].shape[2]);
            if feature_dim_a!=feature_dim_b or (feature_h,feature_w)!=PIXEL_SIZE: fail("feature shape mismatch")
            strict_bank,strict_counts,strict_sizes=memory_for_epoch(lab,epoch,args.per_image_state_cap,"strict_state"); hard_bank,hard_counts,hard_sizes=memory_for_epoch(lab,epoch,args.per_image_state_cap,"hard_confusable"); bank_rows.extend(strict_counts); bank_rows.extend(hard_counts)
            for state,size in {**strict_sizes,**hard_sizes}.items():
                if state in ("CC_pred0","WW_pred0","CC_pred1","WW_pred1","Acorrect_10","Bcorrect_10","Acorrect_01","Bcorrect_01") and size<K_DEFAULT: print("WARNING insufficient-memory epoch=%d state=%s count=%d K=%d"%(epoch,state,size,K_DEFAULT))
            evs_strict=[]; evs_hard=[]
            for row in val:
                es=evaluate_image(row,strict_bank,"strict_state",args.k); eh=evaluate_image(row,hard_bank,"hard_confusable",args.k); es["pa"]=(row["prob_a"]>=THRESHOLD).ravel(); eh["pa"]=es["pa"]; es["hc"]=eh["hc"]; es["hard_score"]=eh["hard_score"]; es["state"]=es["state"]; evs_strict.append(es); evs_hard.append(eh)
            agg_s,img_s,pay_s=aggregate_eval(evs_strict,"strict_state"); agg_h,img_h,pay_h=aggregate_eval(evs_hard,"hard_confusable");
            agg_s.update({"epoch":epoch,"feature_dim_a":feature_dim_a,"feature_dim_b":feature_dim_b,"feature_h":feature_h,"feature_w":feature_w}); agg_h.update({"epoch":epoch,"feature_dim_a":feature_dim_a,"feature_dim_b":feature_dim_b,"feature_h":feature_h,"feature_w":feature_w})
            for state,size in strict_sizes.items(): agg_s["memory_"+state+"_kept_count"]=size
            for state,size in hard_sizes.items(): agg_h["memory_"+state+"_kept_count"]=size
            rows_summary.append(agg_s); hard_summary.append(agg_h)
            for ir_s,ir_h in zip(img_s,img_h): image_rows.append({"epoch":epoch,"image_id":ir_s.pop("image_id"),**{("strict_"+k):v for k,v in ir_s.items()},**{("hard_"+k):v for k,v in ir_h.items()}})
            if epoch in BOOTSTRAP_EPOCHS:
                bootstrap_rows.extend(bootstrap_payload(pay_s,epoch,"strict_state")); bootstrap_rows.extend(bootstrap_payload(pay_h,epoch,"hard_confusable"))
            distribution_rows.extend(score_distribution_rows(evs_strict,epoch,"strict_state")); distribution_rows.extend(score_distribution_rows(evs_hard,epoch,"hard_confusable"))
            print("epoch=%03d memory_build_time=%.2fs validation_images_processed=%d"%(epoch,time.time()-started,len(val)))
            print("  strict banks=%s"%strict_sizes); print("  Agreement WW/CC=%d/%d AUROC/AUPRC=%.6g/%.6g HC90=%.6g/%.6g confidence=%.6g/%.6g"%(agg_s["ww_pixels"],agg_s["cc_pixels"],agg_s["agreement_AUROC"],agg_s["agreement_AUPRC"],agg_s["HC90_AUROC"],agg_s["HC90_AUPRC"],agg_s["confidence_AUROC"],agg_s["confidence_AUPRC"])); print("  Disagreement routing=%.6g confidence=%.6g"%(agg_s["routing_accuracy_total"],agg_s["confidence_routing_accuracy_total"]))
            del ma,mb; torch.cuda.empty_cache()
        write_csv(out/"joint_state_memory_epoch_summary.csv",rows_summary); write_csv(out/"joint_state_memory_hard_confusable_summary.csv",hard_summary); write_csv(out/"joint_state_memory_bank_counts.csv",bank_rows); write_csv(out/"joint_state_memory_image_metrics.csv",image_rows); write_csv(out/"joint_state_memory_bootstrap_ci.csv",bootstrap_rows); write_csv(out/"joint_state_memory_score_distribution.csv",distribution_rows)
        by_epoch={int(r["epoch"]):r for r in rows_summary}
        for epoch in (20,40,80,120,160,200): key_rows.append(by_epoch[epoch])
        write_csv(out/"joint_state_memory_key_epochs.csv",key_rows)
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        metadata={"git_commit":commit,"checkpoint_root":str(Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else CHECKPOINT_ROOT),"epochs":list(EPOCHS),"dataset":"GLaS","seed":SPLIT_SEED,"labeled_count":n_lab,"unlabeled_count":n_unsup,"validation_count":n_val,"input_mode":"rgb_stain","A_view":"RGB","B_view":"deterministic Stain RGB","A_state_key":"appearance_state_dict","B_state_key":"structure_state_dict","strict_load":True,"feature_layer":"Conv_1x1 input (pre-segmentation classifier)","feature_dim_a":feature_dim_a,"feature_dim_b":feature_dim_b,"feature_resolution":[feature_h,feature_w],"joint_feature":"concat(L2_normalize(f_A), L2_normalize(f_B))/sqrt(2); branch-aligned cosine","K":args.k,"state_threshold":THRESHOLD,"HC_threshold":HC_THRESHOLD,"per_image_state_cap":args.per_image_state_cap,"sampling_seed":MEMORY_SEED,"memory_source":"labeled_train_gt_only","evaluation_source":"validation_gt_only","uses_UNI":False,"uses_test":False,"uses_unlabeled_gt":False,"uses_external_model":False,"uses_training":False,"training_performed":False,"test_loaded":False,"optimizer_created":False,"backward_called":False,"checkpoint_selection":False,"validation_gt_used_after_score_only":True,"primary_mode":"strict_state","secondary_mode":"hard_confusable","bootstrap_epochs":list(BOOTSTRAP_EPOCHS),"bootstrap_repetitions":BOOTSTRAP_REPS,"output_directory":str(out)}
        (out/"run_metadata.json").write_text(json.dumps(json_safe(metadata),indent=2,ensure_ascii=False),encoding="utf-8")
        print("COMPLETE protocol=PASS checkpoints=%d labeled=%d unlabeled=%d validation=%d output=%s"%(len(checkpoints),n_lab,n_unsup,n_val,out))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint-dir",default=str(CHECKPOINT_ROOT)); p.add_argument("--output-dir",default=str(DEFAULT_OUT)); p.add_argument("--per-image-state-cap",type=int,default=CAP_DEFAULT); p.add_argument("--k",type=int,default=K_DEFAULT); return p.parse_args()


def main() -> None:
    try:
        args=parse_args()
        if args.k!=K_DEFAULT or args.per_image_state_cap<=0: fail("this run requires fixed K=5 and positive cap")
        run(args)
    except ProtocolMismatch as exc:
        print("PROTOCOL_MISMATCH: %s"%exc,file=sys.stderr); raise SystemExit(2)


if __name__=="__main__": main()
