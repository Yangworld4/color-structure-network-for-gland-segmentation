"""Stage M0: offline UNI-R224 margin-to-intervention diagnosis.

This is an intentionally closed validation-only replay.  It loads the formal
GLaS 10% synchronized milestones and the memory made by the formal run,
computes UNI evidence once, and replays the unchanged State-Aware quantities.
It does not construct an optimizer, call backward, enter train mode, or read
the test manifest/images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from main import (  # noqa: E402
    EXPECTED_UNI_SHA256,
    UNI_CONFIG_DEFAULT,
    UNI_MEAN,
    UNI_STD,
    UNI_WEIGHTS_DEFAULT,
    _build_state_aware_reference_targets_from_margin,
    _reference_weights_from_margin,
    _reconstruct_rgb,
    _topk_scores,
    _uni_dense_features,
)
from models.getnetwork import get_network  # noqa: E402
from models.uni import load_uni  # noqa: E402


DATASET = "GLaS"
LABELED_RATIO = 10
SPLIT_SEED = 24
MEMORY_SEED = 2026
K = 5
ANCHOR_FG = 0.90
ANCHOR_BG = 0.10
CAP = 64
GRID = 14
TOKENS = 196
FEATURE_DIM = 1024
REFERENCE_RESOLUTION = 224
PIXELS = (256, 256)
INPUT_MODE = "rgb_stain"
CPS_MODE = "uni_reference_state_aware"
EXPECTED_MEMORY_FG = 391
EXPECTED_MEMORY_BG = 294
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 1337
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
EPOCHS = tuple(range(10, 201, 10))
BIN_EDGES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, float("inf"))
BIN_LABELS = ("[0,0.01)", "[0.01,0.02)", "[0.02,0.05)",
              "[0.05,0.10)", "[0.10,0.20)", "[0.20,+inf)")


class ProtocolMismatch(RuntimeError):
    pass


def mismatch(message: str) -> None:
    raise ProtocolMismatch(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def div(num: float, den: float) -> float:
    return finite(float(num) / float(den)) if den else float("nan")


def json_safe(value: Any) -> Any:
    """Represent empty subsets as JSON null instead of non-standard NaN."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def mean(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = x if mask is None else x[mask]
    return finite(float(a.mean())) if a.size else float("nan")


def median(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = x if mask is None else x[mask]
    return finite(float(np.median(a))) if a.size else float("nan")


def total(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = x if mask is None else x[mask]
    return float(a.sum()) if a.size else 0.0


def dist(x: np.ndarray, prefix: str) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {prefix + k: float("nan") for k in
                ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p95", "p99")}
    q = np.percentile(x, [10, 25, 50, 75, 90, 95, 99])
    vals = [x.mean(), x.std(), x.min(), x.max(), *q]
    return dict(zip((prefix + k for k in
                     ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p95", "p99")),
                    [finite(float(v)) for v in vals]))


def corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def metrics_for_score(gt: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=bool).ravel()
    score = np.asarray(score, dtype=float).ravel()
    valid = np.isfinite(score)
    gt, score = gt[valid], score[valid]
    pred = score > 0
    fg = gt
    bg = ~gt
    fg_acc = div(float((pred[fg]).sum()), float(fg.sum()))
    bg_acc = div(float((~pred[bg]).sum()), float(bg.sum()))
    tie = score == 0
    try:
        auc = float(roc_auc_score(gt, score))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(gt, score))
    except ValueError:
        ap = float("nan")
    non_tie = ~tie
    return {
        "pixel_count": int(score.size),
        "accuracy": div(float((pred == gt).sum()), float(score.size)),
        "balanced_accuracy": finite(np.nanmean([fg_acc, bg_acc])),
        "fg_accuracy": fg_acc,
        "bg_accuracy": bg_acc,
        "auroc": auc,
        "auprc": ap,
        "tie_count": int(tie.sum()),
        "tie_rate": div(float(tie.sum()), float(score.size)),
        "accuracy_non_tie": div(float((pred[non_tie] == gt[non_tie]).sum()), float(non_tie.sum())),
    }


def load_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    if not directory.is_dir():
        mismatch("checkpoint directory missing: %s" % directory)
    found: Dict[int, Path] = {}
    for path in sorted(directory.glob("epoch_*.pth")):
        m = re.fullmatch(r"epoch_(\d{3})\.pth", path.name)
        if m:
            epoch = int(m.group(1))
            if epoch in found:
                mismatch("duplicate checkpoint epoch %d" % epoch)
            found[epoch] = path
    if tuple(sorted(found)) != EPOCHS:
        mismatch("milestones are not exactly epoch_010..epoch_200 step10: %s" % sorted(found))
    return [(e, found[e]) for e in EPOCHS]


def checkpoint_metadata_assertions(checkpoints: Sequence[Tuple[int, Path]], memory_sha: str) -> str:
    network = None
    for epoch, path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            mismatch("checkpoint %d is not a mapping" % epoch)
        for key in ("appearance_state_dict", "structure_state_dict"):
            if not isinstance(payload.get(key), dict):
                mismatch("checkpoint %d missing %s" % (epoch, key))
        checks = {
            "dataset_name": DATASET,
            "labeled_ratio": LABELED_RATIO,
            "seed": SPLIT_SEED,
            "input_mode": INPUT_MODE,
            "cps_mode": CPS_MODE,
            "selection_split": "validation",
            "uni_k": K,
            "reference_memory_sha256": memory_sha,
            "uni_sha256": EXPECTED_UNI_SHA256,
        }
        for key, expected in checks.items():
            actual = payload.get(key)
            if key == "dataset_name":
                ok = str(actual).lower() == expected.lower()
            else:
                ok = actual == expected
            if not ok:
                mismatch("checkpoint %d metadata %s=%r expected %r" % (epoch, key, actual, expected))
        # These older synchronized milestones predate the explicit resolution
        # metadata fields.  The protocol is still asserted independently by
        # the exact UNI architecture/hash, 14x14 token shape, and memory
        # metadata.  If either field is present, it must agree with R224.
        for key, expected in (("reference_resolution", REFERENCE_RESOLUTION), ("reference_grid", GRID)):
            actual = payload.get(key)
            if actual is not None and actual != expected:
                mismatch("checkpoint %d metadata %s=%r expected %r" % (epoch, key, actual, expected))
        for key in ("hard_reference_threshold_used", "hard_routing_used", "student_confidence_used", "direction_specific_scaling_used"):
            if payload.get(key, False) not in (False, None):
                mismatch("forbidden checkpoint mechanism %s at epoch %d" % (key, epoch))
        network = network or str(payload.get("network") or "unet")
        if str(payload.get("network") or network) != network:
            mismatch("network differs across checkpoints")
    return network or "unet"


def load_memory(path: Path, expected_memory_sha: str | None) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    if not path.is_file():
        mismatch("reference memory missing: %s" % path)
    memory_sha = sha256_file(path)
    if expected_memory_sha and memory_sha != expected_memory_sha:
        mismatch("reference memory SHA mismatch: %s != %s" % (memory_sha, expected_memory_sha))
    required = {"memory_fg", "memory_bg", "memory_fg_count", "memory_bg_count", "feature_dim", "token_grid", "top_k", "anchor_fg", "anchor_bg", "memory_per_image_class", "source_labeled_ratio", "seed", "source_dataset"}
    with np.load(path, allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            mismatch("reference memory metadata missing: %s" % sorted(required - set(z.files)))
        fg = np.asarray(z["memory_fg"], dtype=np.float32)
        bg = np.asarray(z["memory_bg"], dtype=np.float32)
        meta = {k: z[k].item() for k in z.files if z[k].shape == ()}
    if fg.shape != (EXPECTED_MEMORY_FG, FEATURE_DIM) or bg.shape != (EXPECTED_MEMORY_BG, FEATURE_DIM):
        mismatch("reference memory shape is %s/%s" % (fg.shape, bg.shape))
    checks = [("memory_fg_count", EXPECTED_MEMORY_FG), ("memory_bg_count", EXPECTED_MEMORY_BG),
              ("feature_dim", FEATURE_DIM), ("token_grid", GRID), ("top_k", K),
              ("memory_per_image_class", CAP), ("source_labeled_ratio", LABELED_RATIO),
              ("seed", MEMORY_SEED)]
    for key, expected in checks:
        if int(meta[key]) != expected:
            mismatch("memory metadata %s=%r expected %r" % (key, meta[key], expected))
    if float(meta["anchor_fg"]) != ANCHOR_FG or float(meta["anchor_bg"]) != ANCHOR_BG or str(meta["source_dataset"]).lower() != "glas":
        mismatch("reference memory anchor/source metadata mismatch")
    if not np.allclose(np.linalg.norm(fg, axis=1), 1, atol=1e-4) or not np.allclose(np.linalg.norm(bg, axis=1), 1, atol=1e-4):
        mismatch("reference memory is not normalized")
    return {"fg": fg, "bg": bg}, memory_sha, meta


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output) or output.ndim != 4:
        raise TypeError("expected [N,C,H,W] logits")
    return output


def build_validation(cfg: Mapping[str, Any]) -> Tuple[Any, DataLoader, Path]:
    manifest = ROOT / "datasets" / "GLaS" / "manifests" / "paper_seed_2026" / "validation.csv"
    if not manifest.is_file() or "test" in str(manifest).lower():
        mismatch("validation manifest is missing or is a test path")
    normalize = data_normalize_2d(cfg["MEAN"], cfg["STD"])
    dataset = imagefloder_itn(
        manifest_path=str(manifest), data_transform_1=data_transform_2d()["val"],
        data_normalize_1=normalize, sup=True, input_mode=INPUT_MODE, training=False,
    )
    if len(dataset) != 17:
        mismatch("validation image count is %d, expected 17" % len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    return dataset, loader, manifest


def build_reference_cache(out: Path, loader: DataLoader, cfg: Mapping[str, Any], memory: Mapping[str, Any], uni: torch.nn.Module, device: torch.device) -> List[Dict[str, Any]]:
    cache_dir = out / "cache" / "reference"
    cache_dir.mkdir(parents=True, exist_ok=False)
    fg = torch.as_tensor(memory["fg"], device=device)
    bg = torch.as_tensor(memory["bg"], device=device)
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if set(("image_rgb", "image_stain", "mask", "ID")) - set(batch):
                mismatch("formal RGB/Stain validation fields are incomplete")
            rgb_norm = batch["image_rgb"].to(device)
            gt = batch["mask"][0].cpu().numpy().astype(np.uint8)
            rgb_01 = _reconstruct_rgb(rgb_norm, cfg)
            tokens = _uni_dense_features(uni, rgb_01, REFERENCE_RESOLUTION)
            if tuple(tokens.shape) != (1, TOKENS, FEATURE_DIM):
                mismatch("UNI token shape is %s" % (tuple(tokens.shape),))
            sf14 = _topk_scores(tokens.reshape(-1, FEATURE_DIM), fg, K).reshape(1, GRID, GRID)[0]
            sb14 = _topk_scores(tokens.reshape(-1, FEATURE_DIM), bg, K).reshape(1, GRID, GRID)[0]
            m14 = sf14 - sb14
            def up(x: torch.Tensor) -> np.ndarray:
                return F.interpolate(x[None, None], size=PIXELS, mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
            image_id = str(batch["ID"][0])
            row = {"image_id": image_id, "gt": gt, "S_FG": up(sf14), "S_BG": up(sb14), "M": up(m14),
                   "S_FG_14": sf14.cpu().numpy().astype(np.float32), "S_BG_14": sb14.cpu().numpy().astype(np.float32), "M_14": m14.cpu().numpy().astype(np.float32)}
            np.savez_compressed(cache_dir / ("%03d.npz" % index), image_id=np.asarray(image_id), gt=gt, S_FG=row["S_FG"], S_BG=row["S_BG"], M=row["M"], S_FG_14=row["S_FG_14"], S_BG_14=row["S_BG_14"], M_14=row["M_14"])
            rows.append(row)
    if len(rows) != 17:
        mismatch("reference cache count is not 17")
    return rows


def load_reference_cache(out: Path) -> List[Dict[str, Any]]:
    files = sorted((out / "cache" / "reference").glob("*.npz"))
    if len(files) != 17:
        mismatch("reference cache must contain 17 images")
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            rows.append({"image_id": str(z["image_id"].item()), "gt": z["gt"].astype(bool), "S_FG": z["S_FG"], "S_BG": z["S_BG"], "M": z["M"], "S_FG_14": z["S_FG_14"], "S_BG_14": z["S_BG_14"], "M_14": z["M_14"]})
    return rows


def model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def infer_all(out: Path, checkpoints: Sequence[Tuple[int, Path]], network: str, cfg: Mapping[str, Any], loader: DataLoader, device: torch.device) -> None:
    pred_dir = out / "cache" / "student_predictions"
    pred_dir.mkdir(parents=True, exist_ok=False)
    for epoch, path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        a = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
        b = get_network(network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
        a.load_state_dict(payload["appearance_state_dict"], strict=True)
        b.load_state_dict(payload["structure_state_dict"], strict=True)
        a.to(device).eval(); b.to(device).eval()
        if a.training or b.training:
            mismatch("student model entered train mode")
        before = (model_hash(a), model_hash(b))
        la, lb = [], []
        with torch.inference_mode():
            for batch in loader:
                xa = batch["image_rgb"].to(device)
                xb = batch["image_stain"].to(device)
                logits_a, logits_b = extract_logits(a(xa)), extract_logits(b(xb))
                if tuple(logits_a.shape[-2:]) != PIXELS or tuple(logits_b.shape[-2:]) != PIXELS or logits_a.shape[1] != 2 or logits_b.shape[1] != 2:
                    mismatch("student output shape mismatch at epoch %d" % epoch)
                la.append(logits_a.cpu().numpy().astype(np.float32)); lb.append(logits_b.cpu().numpy().astype(np.float32))
        if before != (model_hash(a), model_hash(b)):
            mismatch("checkpoint model changed during inference")
        np.savez_compressed(pred_dir / ("epoch_%03d.npz" % epoch), logits_a=np.concatenate(la), logits_b=np.concatenate(lb))
        del a, b
        torch.cuda.empty_cache()


def state_from_logits(logits_a: np.ndarray, logits_b: np.ndarray, gt: np.ndarray, margin: np.ndarray) -> Dict[str, np.ndarray]:
    ta = torch.from_numpy(logits_a[None] if logits_a.ndim == 3 else logits_a)
    tb = torch.from_numpy(logits_b[None] if logits_b.ndim == 3 else logits_b)
    tm = torch.from_numpy(margin[None] if margin.ndim == 2 else margin.astype(np.float32))
    formal = _build_state_aware_reference_targets_from_margin(ta, tb, tm)
    pa = ta.argmax(1)[0].numpy().astype(bool); pb = tb.argmax(1)[0].numpy().astype(bool)
    probs_a = torch.softmax(ta, 1)[:, 1][0].numpy(); probs_b = torch.softmax(tb, 1)[:, 1][0].numpy()
    agreement = formal["agreement"][0].numpy(); disagreement = formal["disagreement"][0].numpy()
    a_correct, b_correct = pa == gt, pb == gt
    cc = agreement & a_correct; ww = agreement & ~a_correct
    ac = disagreement & a_correct; bc = disagreement & b_correct
    r = formal["conflict"][0].numpy(); ra = formal["r_a"][0].numpy(); rb = formal["r_b"][0].numpy()
    wa = formal["w_a"][0].numpy(); wb = formal["w_b"][0].numpy()
    sa = formal["signed_support_a"][0].numpy(); sb = formal["signed_support_b"][0].numpy()
    tie = disagreement & (np.abs(sa - sb) <= 1e-8)
    support_a, support_b = sa > sb, sb > sa
    ref_correct = ((margin > 0) & gt) | ((margin < 0) & ~gt)
    p_opp_a = np.where(pa, 1 - probs_a, probs_a); p_opp_b = np.where(pb, 1 - probs_b, probs_b)
    rev_a, rev_b = agreement & (r > p_opp_a), agreement & (r > p_opp_b)
    joint = np.minimum(np.maximum(probs_a, 1 - probs_a), np.maximum(probs_b, 1 - probs_b))
    # Exact binary CE logit-gradient norms for each D teacher->receiver pair.
    grad_a_to_b = np.sqrt((probs_b - pa.astype(float)) ** 2 + ((1 - probs_b) - (~pa).astype(float)) ** 2)
    grad_b_to_a = np.sqrt((probs_a - pb.astype(float)) ** 2 + ((1 - probs_a) - (~pb).astype(float)) ** 2)
    # The expression above is algebraically the two-class softmax gradient;
    # keep the explicit vector form as the authoritative numerical quantity.
    grad_a_to_b = np.sqrt((probs_b - pa.astype(float)) ** 2 + ((1 - probs_b) - (1 - pa.astype(float))) ** 2)
    grad_b_to_a = np.sqrt((probs_a - pb.astype(float)) ** 2 + ((1 - probs_a) - (1 - pb.astype(float))) ** 2)
    route_correct = disagreement & ~tie & ((support_a & ac) | (support_b & bc))
    route_wrong = disagreement & ~tie & ((support_a & bc) | (support_b & ac))
    return {**locals(), "M": margin}


def aggregate_state(ref: Mapping[str, Any], pred_a: np.ndarray, pred_b: np.ndarray, epoch: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    image_rows: List[Dict[str, Any]] = []
    bin_acc: Dict[str, Dict[str, float]] = {label: {"count_cc": 0, "count_ww": 0, "refcorrect_ww": 0, "r_ww": 0, "r_cc": 0, "reverse_ww": 0, "ww_refcorrect": 0, "u": 0, "h": 0, "d": 0, "route_correct": 0, "r_d": 0, "w_wrong": 0, "w_correct": 0, "u_d": 0, "h_d": 0, "orig_wrong": 0, "orig_correct": 0} for label in BIN_LABELS}
    alpha_rows: List[Dict[str, Any]] = []
    popp_values: List[np.ndarray] = []
    aggregate = {"epoch": epoch}
    totals = {k: 0.0 for k in ("pixels", "agreement", "cc", "ww", "d", "ww_ref", "ww_ref_wrong", "ww_ref_no_reverse", "ww_ref_reverse", "ww_rev_a", "ww_rev_b", "ww_rev_any", "ww_rev_both", "hc90", "hc90_ref", "hc90_rev", "hc90_both", "hc95", "hc95_ref", "hc95_rev", "hc95_both", "u_a", "h_a", "u_d", "h_d", "orig_wrong", "orig_correct", "route_correct", "route_total", "d_ties", "sum_r_ww", "sum_r_cc", "sum_popp_ww", "sum_r_wrong", "sum_r_correct", "sum_w_wrong", "sum_w_correct", "sum_gap", "sum_d_weight", "shift_l1_ww", "shift_l2_ww", "shift_l1_cc", "shift_l2_cc", "cc_rpos", "cc_reverse" )}
    for index, (refrow, la, lb) in enumerate(zip(ref["rows"], pred_a, pred_b)):
        gt, margin = refrow["gt"], refrow["M"]
        s = state_from_logits(la, lb, gt, margin)
        agreement, cc, ww, d = s["agreement"], s["cc"], s["ww"], s["disagreement"]
        refc = s["ref_correct"]; rev_any = s["rev_a"] | s["rev_b"]; rev_both = s["rev_a"] & s["rev_b"]
        popp_values.extend((s["p_opp_a"][ww], s["p_opp_b"][ww]))
        wrong_teacher_a = d & ~s["a_correct"]; wrong_teacher_b = d & ~s["b_correct"]
        correct_teacher_a = d & s["a_correct"]; correct_teacher_b = d & s["b_correct"]
        removed_a, removed_b = s["ra"] * s["grad_a_to_b"], s["rb"] * s["grad_b_to_a"]
        u_d = total(removed_a, wrong_teacher_a) + total(removed_b, wrong_teacher_b)
        h_d = total(removed_a, correct_teacher_a) + total(removed_b, correct_teacher_b)
        ow = total(s["grad_a_to_b"], wrong_teacher_a) + total(s["grad_b_to_a"], wrong_teacher_b)
        oc = total(s["grad_a_to_b"], correct_teacher_a) + total(s["grad_b_to_a"], correct_teacher_b)
        non_tie = d & ~s["tie"]
        route = s["route_correct"]
        hc90, hc95 = ww & (s["joint"] >= .90), ww & (s["joint"] >= .95)
        def imetric(mask: np.ndarray, x: np.ndarray) -> float: return mean(x, mask)
        row = {"epoch": epoch, "image_id": refrow["image_id"], "pixel_count": int(gt.size),
               "agreement_count": int(agreement.sum()), "cc_count": int(cc.sum()), "ww_count": int(ww.sum()), "d_count": int(d.sum()),
               "agreement_rate": float(agreement.mean()), "cc_rate": float(cc.mean()), "ww_rate": float(ww.mean()), "d_rate": float(d.mean()),
               "ww_refcorrect_count": int((ww & refc).sum()), "ww_reference_wrong_count": int((ww & ~refc).sum()), "ww_refcorrect_no_reverse_count": int((ww & refc & ~rev_any).sum()), "ww_refcorrect_reverse_count": int((ww & refc & rev_any).sum()), "ww_reverse_A_count": int((ww & refc & s["rev_a"]).sum()), "ww_reverse_B_count": int((ww & refc & s["rev_b"]).sum()), "ww_reverse_any_count": int((ww & refc & rev_any).sum()), "ww_reverse_both_count": int((ww & refc & rev_both).sum()),
               "hc90_count": int(hc90.sum()), "hc90_refcorrect_count": int((hc90 & refc).sum()), "hc90_reverse_count": int((hc90 & refc & rev_any).sum()),
               "hc95_count": int(hc95.sum()), "hc95_refcorrect_count": int((hc95 & refc).sum()), "hc95_reverse_count": int((hc95 & refc & rev_any).sum()),
               "sum_r_ww": total(s["r"], ww), "sum_r_cc": total(s["r"], cc), "mean_r_ww": imetric(ww, s["r"]), "median_r_ww": median(s["r"], ww), "mean_r_cc": imetric(cc, s["r"]), "median_r_cc": median(s["r"], cc),
               "u_a": total(s["r"], ww), "h_a": total(s["r"], cc), "ww_refcorrect_mean_abs_m": imetric(ww & refc, np.abs(margin)),
               "ww_refcorrect_mean_r": imetric(ww & refc, s["r"]), "mean_p_opp_WW": mean(np.r_[s["p_opp_a"][ww], s["p_opp_b"][ww]]), "median_p_opp_WW": median(np.r_[s["p_opp_a"][ww], s["p_opp_b"][ww]]), "ww_refcorrect_reverse_rate": div(float((ww & refc & rev_any).sum()), float((ww & refc).sum())),
               "ww_refcorrect_reverse_both_rate": div(float((ww & refc & rev_both).sum()), float((ww & refc).sum())),
               "cc_intervention_rate": div(float((cc & (s["r"] > 0)).sum()), float(cc.sum())), "cc_harmful_reverse_rate": div(float((cc & ((s["r"] > s["p_opp_a"]) | (s["r"] > s["p_opp_b"]))).sum()), float(cc.sum())),
               "u_d": u_d, "h_d": h_d, "original_wrong_grad": ow, "original_correct_grad": oc,
               "route_correct_count": int(route.sum()), "route_total_count": int(non_tie.sum()), "routing_tie_count": int(s["tie"].sum()),
               "routing_accuracy": div(float(route.sum()), float(non_tie.sum())), "mean_r_wrong_teacher": div(total(s["ra"], wrong_teacher_a) + total(s["rb"], wrong_teacher_b), float(wrong_teacher_a.sum() + wrong_teacher_b.sum())),
               "mean_r_correct_teacher": div(total(s["ra"], correct_teacher_a) + total(s["rb"], correct_teacher_b), float(correct_teacher_a.sum() + correct_teacher_b.sum())),
               "mean_w_wrong_teacher": div(total(s["wa"], wrong_teacher_a) + total(s["wb"], wrong_teacher_b), float(wrong_teacher_a.sum() + wrong_teacher_b.sum())),
               "mean_w_correct_teacher": div(total(s["wa"], correct_teacher_a) + total(s["wb"], correct_teacher_b), float(correct_teacher_a.sum() + correct_teacher_b.sum())),
               "mean_directional_gap": mean(np.abs(s["wa"] - s["wb"]), d), "mean_total_D_weight": mean(s["wa"] + s["wb"], d), "relative_D_mass": mean((s["wa"] + s["wb"]) / 2, d),
               "useful_removed_fraction": div(u_d, ow), "harmful_removed_fraction": div(h_d, oc), "gradient_intervention_precision": div(u_d, u_d + h_d), "u_d_over_h_d": div(u_d, h_d),
               "agreement_mass_ratio": div(total(s["r"], ww), total(s["r"], cc)), "agreement_intervention_precision": div(total(s["r"], ww), total(s["r"], ww) + total(s["r"], cc)),
               "branch_macro_reverse_rate": mean((s["rev_a"].astype(float) + s["rev_b"].astype(float)) / 2, ww & refc),
               "agreement_gradient_shift_l1_ww": 2 * total(s["r"], ww), "agreement_gradient_shift_l2_ww": math.sqrt(2) * total(s["r"], ww), "agreement_gradient_shift_l1_cc": 2 * total(s["r"], cc), "agreement_gradient_shift_l2_cc": math.sqrt(2) * total(s["r"], cc)}
        image_rows.append(row)
        for key, value in (("pixels", gt.size), ("agreement", agreement.sum()), ("cc", cc.sum()), ("ww", ww.sum()), ("d", d.sum()), ("ww_ref", (ww & refc).sum()), ("ww_ref_wrong", (ww & ~refc).sum()), ("ww_ref_no_reverse", (ww & refc & ~rev_any).sum()), ("ww_ref_reverse", (ww & refc & rev_any).sum()), ("ww_rev_a", (ww & refc & s["rev_a"]).sum()), ("ww_rev_b", (ww & refc & s["rev_b"]).sum()), ("ww_rev_any", (ww & refc & rev_any).sum()), ("ww_rev_both", (ww & refc & rev_both).sum()), ("hc90", hc90.sum()), ("hc90_ref", (hc90 & refc).sum()), ("hc90_rev", (hc90 & refc & rev_any).sum()), ("hc90_both", (hc90 & refc & s["rev_a"] & s["rev_b"]).sum()), ("hc95", hc95.sum()), ("hc95_ref", (hc95 & refc).sum()), ("hc95_rev", (hc95 & refc & rev_any).sum()), ("hc95_both", (hc95 & refc & s["rev_a"] & s["rev_b"]).sum()), ("u_a", total(s["r"], ww)), ("h_a", total(s["r"], cc)), ("u_d", u_d), ("h_d", h_d), ("orig_wrong", ow), ("orig_correct", oc), ("route_correct", route.sum()), ("route_total", non_tie.sum()), ("d_ties", s["tie"].sum()), ("sum_r_ww", total(s["r"], ww)), ("sum_r_cc", total(s["r"], cc)), ("sum_popp_ww", total(s["p_opp_a"], ww) + total(s["p_opp_b"], ww)), ("sum_r_wrong", total(s["ra"], wrong_teacher_a) + total(s["rb"], wrong_teacher_b)), ("sum_r_correct", total(s["ra"], correct_teacher_a) + total(s["rb"], correct_teacher_b)), ("sum_w_wrong", total(s["wa"], wrong_teacher_a) + total(s["wb"], wrong_teacher_b)), ("sum_w_correct", total(s["wa"], correct_teacher_a) + total(s["wb"], correct_teacher_b)), ("sum_gap", total(np.abs(s["wa"] - s["wb"]), d)), ("sum_d_weight", total(s["wa"] + s["wb"], d)), ("shift_l1_ww", 2 * total(s["r"], ww)), ("shift_l2_ww", math.sqrt(2) * total(s["r"], ww)), ("shift_l1_cc", 2 * total(s["r"], cc)), ("shift_l2_cc", math.sqrt(2) * total(s["r"], cc)), ("cc_rpos", (cc & (s["r"] > 0)).sum()), ("cc_reverse", (cc & ((s["r"] > s["p_opp_a"]) | (s["r"] > s["p_opp_b"]))).sum())): totals[key] += float(value)
        for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
            bm = (np.abs(margin) >= lo) & (np.abs(margin) < hi)
            ba = bin_acc[label]
            for key, value in (("count_cc", (bm & cc).sum()), ("count_ww", (bm & ww).sum()), ("refcorrect_ww", (bm & ww & refc).sum()), ("r_ww", total(s["r"], bm & ww)), ("r_cc", total(s["r"], bm & cc)), ("reverse_ww", (bm & ww & refc & rev_any).sum()), ("ww_refcorrect", (bm & ww & refc).sum()), ("u", total(s["r"], bm & ww)), ("h", total(s["r"], bm & cc)), ("d", (bm & d).sum()), ("route_correct", (bm & route).sum()), ("r_d", total(s["r"], bm & d)), ("w_wrong", total(np.where(s["a_correct"], s["wb"], s["wa"]), bm & d)), ("w_correct", total(np.where(s["a_correct"], s["wa"], s["wb"]), bm & d)), ("u_d", total(removed_a, bm & wrong_teacher_a) + total(removed_b, bm & wrong_teacher_b)), ("h_d", total(removed_a, bm & correct_teacher_a) + total(removed_b, bm & correct_teacher_b)), ("orig_wrong", total(s["grad_a_to_b"], bm & wrong_teacher_a) + total(s["grad_b_to_a"], bm & wrong_teacher_b)), ("orig_correct", total(s["grad_a_to_b"], bm & correct_teacher_a) + total(s["grad_b_to_a"], bm & correct_teacher_b))): ba[key] += float(value)
        for alpha in ALPHAS:
            ra, rb = np.clip(alpha * s["ra"], 0, 1), np.clip(alpha * s["rb"], 0, 1)
            rev = (ra > s["p_opp_a"]) | (rb > s["p_opp_b"])
            hc90r, hc95r = hc90 & refc & rev, hc95 & refc & rev
            au = total(ra, ww)  # agreement R is shared; ra==rb on agreement
            ah = total(ra, cc)
            ud = total(ra * s["grad_a_to_b"], wrong_teacher_a) + total(rb * s["grad_b_to_a"], wrong_teacher_b)
            hd = total(ra * s["grad_a_to_b"], correct_teacher_a) + total(rb * s["grad_b_to_a"], correct_teacher_b)
            alpha_rows.append({"epoch": epoch, "alpha": alpha, "U_A": au, "H_A": ah, "U_A_over_H_A": div(au, ah), "agreement_intervention_precision": div(au, au + ah), "ww_refcorrect_count": int((ww & refc).sum()), "ww_reverse_count": int((ww & refc & rev).sum()), "cc_count": int(cc.sum()), "cc_harmful_reverse_count": int((cc & (ra > s["p_opp_a"])).sum()), "hc90_count": int(hc90.sum()), "hc90_reverse_count": int(hc90r.sum()), "hc95_count": int(hc95.sum()), "hc95_reverse_count": int(hc95r.sum()), "d_count": int(d.sum()), "original_wrong_grad": ow, "original_correct_grad": oc, "WW_reverse_rate_refcorrect": div(float((ww & refc & rev).sum()), float((ww & refc).sum())), "CC_harmful_reverse_rate": div(float((cc & (ra > s["p_opp_a"])).sum()), float(cc.sum())), "HC90_reverse": div(float(hc90r.sum()), float(hc90.sum())), "HC95_reverse": div(float(hc95r.sum()), float(hc95.sum())), "U_D": ud, "H_D": hd, "U_D_over_H_D": div(ud, hd), "gradient_intervention_precision": div(ud, ud + hd), "useful_removed_fraction": div(ud, ow), "harmful_removed_fraction": div(hd, oc), "mean_total_D_weight": mean((1 - ra) + (1 - rb), d)})
    raw_alpha_rows = alpha_rows
    alpha_rows = []
    for alpha in ALPHAS:
        q = [x for x in raw_alpha_rows if x["alpha"] == alpha]
        sa, sh, sud, shd = sum(x["U_A"] for x in q), sum(x["H_A"] for x in q), sum(x["U_D"] for x in q), sum(x["H_D"] for x in q)
        alpha_rows.append({"epoch": epoch, "alpha": alpha, "scope": "aggregate", "U_A": sa, "H_A": sh, "U_A_over_H_A": div(sa, sh), "agreement_intervention_precision": div(sa, sa + sh), "WW_reverse_rate_refcorrect": div(sum(x["ww_reverse_count"] for x in q), sum(x["ww_refcorrect_count"] for x in q)), "CC_harmful_reverse_rate": div(sum(x["cc_harmful_reverse_count"] for x in q), sum(x["cc_count"] for x in q)), "HC90_reverse": div(sum(x["hc90_reverse_count"] for x in q), sum(x["hc90_count"] for x in q)), "HC95_reverse": div(sum(x["hc95_reverse_count"] for x in q), sum(x["hc95_count"] for x in q)), "U_D": sud, "H_D": shd, "U_D_over_H_D": div(sud, shd), "gradient_intervention_precision": div(sud, sud + shd), "useful_removed_fraction": div(sud, sum(x["original_wrong_grad"] for x in q)), "harmful_removed_fraction": div(shd, sum(x["original_correct_grad"] for x in q)), "mean_total_D_weight": div(sum(x["mean_total_D_weight"] * x["d_count"] for x in q), sum(x["d_count"] for x in q))})
    def put(key: str, value: Any) -> None: aggregate[key] = value
    put("Agreement_rate", div(totals["agreement"], totals["pixels"])); put("CC_rate", div(totals["cc"], totals["pixels"])); put("WW_rate", div(totals["ww"], totals["pixels"])); put("D_rate", div(totals["d"], totals["pixels"]))
    for k, label in (("ww", "WW"), ("cc", "CC")):
        put("mean_R_" + label, div(totals["sum_r_" + k], totals[k])); put("total_R_" + label, totals["sum_r_" + k])
    put("median_R_WW", float(np.nanmedian([r["median_r_ww"] for r in image_rows]))); put("median_R_CC", float(np.nanmedian([r["median_r_cc"] for r in image_rows])))
    put("agreement_mass_ratio", div(totals["u_a"], totals["h_a"])); put("agreement_intervention_precision", div(totals["u_a"], totals["u_a"] + totals["h_a"])); put("selectivity_mean", div(totals["u_a"] / totals["ww"], totals["h_a"] / totals["cc"]))
    put("WW_ref_correct", int(totals["ww_ref"])); put("WW_ref_correct_rate", div(totals["ww_ref"], totals["ww"])); put("WW_reference_direction_wrong", int(totals["ww_ref_wrong"])); put("WW_reference_correct_but_no_reverse", int(totals["ww_ref_no_reverse"])); put("WW_reference_correct_and_reverse", int(totals["ww_ref_reverse"])); put("WW_reverse_A", int(totals["ww_rev_a"])); put("WW_reverse_B", int(totals["ww_rev_b"])); put("WW_reverse_any", int(totals["ww_rev_any"])); put("WW_reverse_both", int(totals["ww_rev_both"])); put("WW_reverse_A_rate", div(totals["ww_rev_a"], totals["ww_ref"])); put("WW_reverse_B_rate", div(totals["ww_rev_b"], totals["ww_ref"])); put("WW_reverse_any_rate", div(totals["ww_rev_any"], totals["ww_ref"])); put("WW_reverse_both_rate", div(totals["ww_rev_both"], totals["ww_ref"]))
    put("P_R_gt_popp_WW", div(totals["ww_rev_any"], totals["ww"])); put("P_R_gt_popp_WW_RefCorrect", div(totals["ww_rev_any"], totals["ww_ref"])); put("mean_p_opp_WW", div(totals["sum_popp_ww"], 2 * totals["ww"])); put("median_p_opp_WW", float(np.median(np.concatenate(popp_values))) if popp_values else float("nan")); put("mean_R_minus_popp_WW", div(2 * totals["sum_r_ww"] - totals["sum_popp_ww"], 2 * totals["ww"]))
    put("CC_intervention_rate", div(totals["cc_rpos"], totals["cc"])); put("CC_harmful_reverse_rate", div(totals["cc_reverse"], totals["cc"]))
    put("HC90_count", int(totals["hc90"])); put("HC90_ref_correct", int(totals["hc90_ref"])); put("HC90_reverse", int(totals["hc90_rev"])); put("HC90_refcorrect_rate", div(totals["hc90_ref"], totals["hc90"])); put("HC90_reverse_rate", div(totals["hc90_rev"], totals["hc90_ref"])); put("HC90_reverse_both", int(totals["hc90_both"])); put("HC90_reverse_both_rate", div(totals["hc90_both"], totals["hc90_ref"]))
    put("HC95_count", int(totals["hc95"])); put("HC95_ref_correct", int(totals["hc95_ref"])); put("HC95_reverse", int(totals["hc95_rev"])); put("HC95_refcorrect_rate", div(totals["hc95_ref"], totals["hc95"])); put("HC95_reverse_rate", div(totals["hc95_rev"], totals["hc95_ref"])); put("HC95_reverse_both", int(totals["hc95_both"])); put("HC95_reverse_both_rate", div(totals["hc95_both"], totals["hc95_ref"]))
    put("routing_accuracy", div(totals["route_correct"], totals["route_total"])); put("routing_tie_rate", div(totals["d_ties"], totals["d"])); put("mean_R_wrong_teacher", div(totals["sum_r_wrong"], totals["orig_wrong"] * 0 + sum((r["d_count"] for r in image_rows)) / 2 if False else sum((r["d_count"] for r in image_rows))))
    wrong_dir = sum((r["d_count"] for r in image_rows)) # one wrong teacher per D pixel
    put("mean_R_wrong_teacher", div(totals["sum_r_wrong"], wrong_dir)); put("mean_R_correct_teacher", div(totals["sum_r_correct"], wrong_dir)); put("mean_w_wrong_teacher", div(totals["sum_w_wrong"], wrong_dir)); put("mean_w_correct_teacher", div(totals["sum_w_correct"], wrong_dir)); put("mean_directional_gap", div(totals["sum_gap"], totals["d"])); put("mean_total_D_weight", div(totals["sum_d_weight"], totals["d"])); put("relative_D_mass", div(totals["sum_d_weight"], 2 * totals["d"])); put("U_D", totals["u_d"]); put("H_D", totals["h_d"]); put("U_D_over_H_D", div(totals["u_d"], totals["h_d"])); put("useful_removed_fraction", div(totals["u_d"], totals["orig_wrong"])); put("harmful_removed_fraction", div(totals["h_d"], totals["orig_correct"])); put("gradient_intervention_precision", div(totals["u_d"], totals["u_d"] + totals["h_d"]))
    put("agreement_gradient_shift_l1_WW", totals["shift_l1_ww"]); put("agreement_gradient_shift_l2_WW", totals["shift_l2_ww"]); put("agreement_gradient_shift_l1_CC", totals["shift_l1_cc"]); put("agreement_gradient_shift_l2_CC", totals["shift_l2_cc"]); put("branch_macro_reverse_rate", div(totals["ww_rev_any"], totals["ww_ref"]))
    for r in image_rows:
        r["epoch"] = epoch
    return aggregate, image_rows, {"bins": bin_acc, "alpha": alpha_rows}


def make_bin_rows(epoch: int, ref: Mapping[str, Any], analysis: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    b = analysis["extra"]["bins"]; rows_a, rows_d = [], []
    all_m = np.concatenate([x["M"].ravel() for x in ref["rows"]]); abs_m = np.abs(all_m)
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        bm = (abs_m >= lo) & (abs_m < hi)
        q = b[label]
        rows_a.append({"epoch": epoch, "margin_bin": label, "lo": lo, "hi": hi, "count_CC": int(q["count_cc"]), "count_WW": int(q["count_ww"]), "RefCorrect_WW": int(q["refcorrect_ww"]), "mean_R_WW": div(q["r_ww"], q["count_ww"]), "mean_R_CC": div(q["r_cc"], q["count_cc"]), "reverse_rate_WW": div(q["reverse_ww"], q["refcorrect_ww"]), "U_A": q["u"], "H_A": q["h"], "intervention_precision": div(q["u"], q["u"] + q["h"]), "pixel_count": int(bm.sum()), "mean_abs_M": mean(abs_m[bm])})
        rows_d.append({"epoch": epoch, "margin_bin": label, "lo": lo, "hi": hi, "count_D": int(q["d"]), "routing_accuracy": div(q["route_correct"], q["d"]), "mean_R": div(q["r_d"], q["d"]), "mean_w_wrong": div(q["w_wrong"], q["d"]), "mean_w_correct": div(q["w_correct"], q["d"]), "U_D": q["u_d"], "H_D": q["h_d"], "useful_removed_fraction": div(q["u_d"], q["orig_wrong"]), "harmful_removed_fraction": div(q["h_d"], q["orig_correct"]), "gradient_intervention_precision": div(q["u_d"], q["u_d"] + q["h_d"]), "pixel_count": int(bm.sum()), "mean_abs_M": mean(abs_m[bm])})
    return rows_a, rows_d


def reference_outputs(out: Path, ref: Mapping[str, Any]) -> Dict[str, Any]:
    rows = ref["rows"]
    sf = np.concatenate([r["S_FG"].ravel() for r in rows]); sb = np.concatenate([r["S_BG"].ravel() for r in rows]); m = np.concatenate([r["M"].ravel() for r in rows]); gt = np.concatenate([r["gt"].ravel() for r in rows]).astype(bool); am = np.abs(m)
    pc, sc = corr(sf, sb)
    margin_row = {"epoch": "reference_static", "category": "All", "pixel_count": int(m.size), "image_count": len(rows), **dist(sf, "S_FG_"), **dist(sb, "S_BG_"), **dist(m, "M_"), **dist(am, "abs_M_"), "pearson_S_FG_S_BG": pc, "spearman_S_FG_S_BG": sc}
    write_csv(out / "01_margin_distribution.csv", [margin_row])
    ref_metric = metrics_for_score(gt, m)
    write_csv(out / "03_reference_summary.csv", [{"scope": "All_validation_pixels", **ref_metric, "mean_abs_M": float(am.mean()), "median_abs_M": float(np.median(am)), "reference_tie_rate": float((m == 0).mean()), "pearson_S_FG_S_BG": pc, "spearman_S_FG_S_BG": sc}])
    rows_b = []
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (am >= lo) & (am < hi); mm = metrics_for_score(gt[mask], m[mask]) if mask.any() else {k: float("nan") for k in ("accuracy", "fg_accuracy", "bg_accuracy", "auroc", "auprc", "balanced_accuracy", "tie_rate", "accuracy_non_tie")}
        rows_b.append({"partition_type": "fixed_abs_M_bin", "partition": label, "lo": lo, "hi": hi, "pixel_count": int(mask.sum()), "image_count": int(sum(bool(np.any((np.abs(r["M"]) >= lo) & (np.abs(r["M"]) < hi))) for r in rows)), "mean_abs_M": mean(am[mask]), "median_abs_M": median(am[mask]), "reference_accuracy": mm.get("accuracy"), "FG_accuracy": mm.get("fg_accuracy"), "BG_accuracy": mm.get("bg_accuracy"), "balanced_accuracy": mm.get("balanced_accuracy"), "AUROC": mm.get("auroc"), "AUPRC": mm.get("auprc"), "tie_rate": mm.get("tie_rate")})
    qs = np.quantile(am, np.linspace(0, 1, 6))
    for i in range(5):
        mask = (am >= qs[i]) & ((am <= qs[i+1]) if i == 4 else (am < qs[i+1])); mm = metrics_for_score(gt[mask], m[mask])
        rows_b.append({"partition_type": "abs_M_quintile", "partition": "Q%d" % (i + 1), "lo": float(qs[i]), "hi": float(qs[i+1]), "pixel_count": int(mask.sum()), "image_count": int(sum(bool(np.any(((np.abs(r["M"]) >= qs[i]) & ((np.abs(r["M"]) <= qs[i+1]) if i == 4 else (np.abs(r["M"]) < qs[i+1]))))) for r in rows)), "mean_abs_M": mean(am[mask]), "median_abs_M": median(am[mask]), "reference_accuracy": mm["accuracy"], "FG_accuracy": mm["fg_accuracy"], "BG_accuracy": mm["bg_accuracy"], "balanced_accuracy": mm["balanced_accuracy"], "AUROC": mm["auroc"], "AUPRC": mm["auprc"], "tie_rate": mm["tie_rate"]})
    write_csv(out / "02_margin_reliability_bins.csv", rows_b)
    return {"sf": sf, "sb": sb, "m": m, "gt": gt, "abs_m": am, "summary": ref_metric, "bins": rows_b, "distribution_row": margin_row}


def margin_distribution_dynamic(ref: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for a in analyses:
        epoch = int(a["epoch"])
        for category in ("All", "CC", "WW", "D"):
            vals_sf=[]; vals_sb=[]; vals_m=[]
            for r, s in zip(ref["rows"], a["states"]):
                mask = {"All": np.ones_like(s["gt"], bool), "CC": s["cc"], "WW": s["ww"], "D": s["disagreement"]}[category]
                vals_sf.append(r["S_FG"][mask]); vals_sb.append(r["S_BG"][mask]); vals_m.append(r["M"][mask])
            sf, sb, m = np.concatenate(vals_sf), np.concatenate(vals_sb), np.concatenate(vals_m); p = np.abs(m); pc, sc = corr(sf, sb)
            row = {"epoch": epoch, "category": category, "pixel_count": int(m.size), "image_count": int(sum(np.any({"All": np.ones_like(s["gt"], bool), "CC": s["cc"], "WW": s["ww"], "D": s["disagreement"]}[category]) for s in a["states"])), **dist(sf, "S_FG_"), **dist(sb, "S_BG_"), **dist(m, "M_"), **dist(p, "abs_M_"), "pearson_S_FG_S_BG": pc, "spearman_S_FG_S_BG": sc}
            for threshold in (.01, .02, .05, .10, .20): row["P_abs_M_lt_%g" % threshold] = float((p < threshold).mean())
            row["P_abs_M_ge_0.20"] = float((p >= .20).mean())
            out.append(row)
    return out


def hc_rows(analyses: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows=[]
    for a in analyses:
        for threshold, prefix in ((.90, "HC90"), (.95, "HC95")):
            rows.append({"epoch": a["epoch"], "subset": prefix + "-WW", "count": a[prefix + "_count"], "RefCorrect": a[prefix + "_refcorrect_rate"], "mean_abs_M": a[prefix + "mean_abs_M"], "mean_R": a[prefix + "mean_R"], "P_R_gt_popp": a[prefix + "reverse_rate"], "reverse_any": a[prefix + "reverse_rate"], "reverse_both": a[prefix + "reverse_both_rate"]})
    return rows


def add_hc_fields(ref: Mapping[str, Any], a: Dict[str, Any], pred_a: np.ndarray, pred_b: np.ndarray) -> None:
    counts = {"HC90": [], "HC95": []}
    for r, la, lb in zip(ref["rows"], pred_a, pred_b):
        s=state_from_logits(la,lb,r["gt"],r["M"]); ww=s["ww"]; rc=s["ref_correct"]; rev=s["rev_a"]|s["rev_b"]; both=s["rev_a"]&s["rev_b"]
        for t,p in ((.90,"HC90"),(.95,"HC95")):
            mask=ww&(s["joint"]>=t); counts[p].append((mask,rc,rev,both,np.abs(r["M"]),s["r"]))
    for t,p in ((.90,"HC90"),(.95,"HC95")):
        mask=np.concatenate([x[0] for x in counts[p]]); rc=np.concatenate([x[1] for x in counts[p]]); rev=np.concatenate([x[2] for x in counts[p]]); both=np.concatenate([x[3] for x in counts[p]]); am=np.concatenate([x[4] for x in counts[p]]); rr=np.concatenate([x[5] for x in counts[p]])
        a[p+"_count"]=int(mask.sum()); a[p+"_refcorrect_rate"]=div(float((mask&rc).sum()),float(mask.sum())); a[p+"_mean_abs_M"]=mean(am,mask); a[p+"_mean_R"]=mean(rr,mask); a[p+"_reverse_rate"]=div(float((mask&rc&rev).sum()),float((mask&rc).sum())); a[p+"_reverse_both_rate"]=div(float((mask&rc&both).sum()),float((mask&rc).sum()))


def bootstrap(out: Path, image_rows: Sequence[Mapping[str, Any]]) -> None:
    rows=list(image_rows); n=len(rows); rng=np.random.default_rng(BOOTSTRAP_SEED); draws=rng.integers(0,n,size=(BOOTSTRAP_REPS,n))
    def val(metric: str, ix: np.ndarray) -> float:
        x=[rows[int(i)] for i in ix]
        if metric=="agreement_mean_R_WW_minus_CC": return div(sum(r["u_a"] for r in x),sum(r["ww_count"] for r in x))-div(sum(r["h_a"] for r in x),sum(r["cc_count"] for r in x))
        if metric=="agreement_U_A_over_H_A": return div(sum(r["u_a"] for r in x),sum(r["h_a"] for r in x))
        if metric=="agreement_intervention_precision": return div(sum(r["u_a"] for r in x),sum(r["u_a"]+r["h_a"] for r in x))
        if metric=="WW_reverse_rate": return div(sum(r["ww_reverse_any_count"] for r in x),sum(r["ww_refcorrect_count"] for r in x))
        if metric=="routing_accuracy": return div(sum(r["route_correct_count"] for r in x),sum(r["route_total_count"] for r in x))
        if metric=="mean_R_wrong_minus_correct": return div(sum(r["u_d"]*0 + r["mean_r_wrong_teacher"]*r["d_count"] for r in x),sum(r["d_count"] for r in x))-div(sum(r["mean_r_correct_teacher"]*r["d_count"] for r in x),sum(r["d_count"] for r in x))
        if metric=="U_D_over_H_D": return div(sum(r["u_d"] for r in x),sum(r["h_d"] for r in x))
        if metric=="gradient_intervention_precision": return div(sum(r["u_d"] for r in x),sum(r["u_d"]+r["h_d"] for r in x))
        if metric=="useful_removed_fraction": return div(sum(r["u_d"] for r in x),sum(r["original_wrong_grad"] for r in x))
        raise KeyError(metric)
    metrics=("agreement_mean_R_WW_minus_CC","agreement_U_A_over_H_A","agreement_intervention_precision","WW_reverse_rate","routing_accuracy","mean_R_wrong_minus_correct","U_D_over_H_D","gradient_intervention_precision","useful_removed_fraction")
    output=[]
    for metric in metrics:
        point=val(metric,np.arange(n)); boot=np.asarray([val(metric,ix) for ix in draws]); low,high=np.nanpercentile(boot,[2.5,97.5]); numerator=denominator=float("nan")
        if metric=="agreement_U_A_over_H_A": numerator=sum(r["u_a"] for r in rows); denominator=sum(r["h_a"] for r in rows)
        if metric=="U_D_over_H_D": numerator=sum(r["u_d"] for r in rows); denominator=sum(r["h_d"] for r in rows)
        output.append({"epoch":200,"metric":metric,"point_estimate":point,"ci_lower_95":low,"ci_upper_95":high,"bootstrap_std":float(np.nanstd(boot)),"bootstrap_repetitions":BOOTSTRAP_REPS,"bootstrap_seed":BOOTSTRAP_SEED,"numerator_point":numerator,"denominator_point":denominator,"ratio_ci_note":"ratio CI is image-bootstrap ratio; numerator/denominator are also reported" if np.isfinite(numerator) else "not_a_ratio"})
    write_csv(out / "10_bootstrap_summary.csv",output)


def plot_save(path: Path, x: Sequence[Any], ys: Mapping[str, Sequence[float]], xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(7,4.5))
    for label,y in ys.items(): plt.plot(x,y,marker="o",label=label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()


def make_figures(out: Path, ref_summary: Mapping[str, Any], agreement_bins: Sequence[Mapping[str, Any]], disagreement_bins: Sequence[Mapping[str, Any]], temporal: Sequence[Mapping[str, Any]]) -> None:
    f=out/"figures"; f.mkdir(exist_ok=True)
    labels=list(BIN_LABELS); x=np.arange(len(labels));
    plot_save(f/"figure_1_margin_reliability.png",labels,{"Reference correctness":[r["reference_accuracy"] for r in ref_summary["bins"] if r["partition_type"]=="fixed_abs_M_bin"]},"|M| bin","Reference correctness")
    a=[r for r in agreement_bins if int(r["epoch"])==200]; d=[r for r in disagreement_bins if int(r["epoch"])==200]
    plot_save(f/"figure_2_agreement_ww_reverse.png",labels,{"WW reverse rate":[r["reverse_rate_WW"] for r in a]},"|M| bin","WW reverse rate")
    plot_save(f/"figure_3_agreement_useful_harmful_mass.png",labels,{"U_A": [r["U_A"] for r in a],"H_A":[r["H_A"] for r in a]},"|M| bin","intervention mass")
    plot_save(f/"figure_4_disagreement_routing_accuracy.png",labels,{"routing accuracy":[r["routing_accuracy"] for r in d]},"|M| bin","routing accuracy")
    plot_save(f/"figure_5_disagreement_gradient_mass.png",labels,{"U_D":[r["U_D"] for r in d],"H_D":[r["H_D"] for r in d]},"|M| bin","removed gradient mass")
    plot_save(f/"figure_6_disagreement_gradient_precision.png",labels,{"gradient precision":[r["gradient_intervention_precision"] for r in d]},"|M| bin","gradient intervention precision")
    epochs=[r["epoch"] for r in temporal]
    plot_save(f/"figure_7_temporal_agreement.png",epochs,{"WW reverse": [r["WW_reverse_any_rate"] for r in temporal],"agreement precision":[r["agreement_intervention_precision"] for r in temporal]},"epoch","rate")
    plot_save(f/"figure_8_temporal_disagreement.png",epochs,{"routing accuracy":[r["routing_accuracy"] for r in temporal],"useful removed fraction":[r["useful_removed_fraction"] for r in temporal]},"epoch","rate")


def auc_safe(y: np.ndarray, score: np.ndarray) -> float:
    try: return float(roc_auc_score(y.astype(int),score))
    except ValueError: return float("nan")


def final_diagnosis(out: Path, ref_summary: Mapping[str, Any], analyses: Sequence[Mapping[str, Any]], alpha_rows: Sequence[Mapping[str, Any]], bootstrap_path: Path) -> Dict[str, Any]:
    e200=next(a for a in analyses if a["epoch"]==200); e40=next(a for a in analyses if a["epoch"]==40)
    m=ref_summary["m"]; gt=ref_summary["gt"]; am=ref_summary["abs_m"]
    small=(am<.05); small_rel=metrics_for_score(gt[small],m[small]) if small.any() else {}
    # Pixel-level separation diagnostics are explicitly unweighted; mass
    # effects remain in the reported binned/alpha columns.
    useful=[]; harmful=[]; route_y=[]; route_s=[]; grad_y=[]; grad_s=[]
    for r in analyses[-1]["states"]:
        useful.append(np.abs(r["M"])[r["ww"] & r["ref_correct"]].ravel()); harmful.append(np.abs(r["M"])[r["cc"]].ravel());
        non=r["disagreement"] & ~r["tie"]; route_y.append(((r["support_a"]&r["ac"])|(r["support_b"]&r["bc"]))[non].ravel()); route_s.append(np.abs(r["M"])[non].ravel())
        grad_y.append(np.r_[np.ones(int((r["a_correct"] & r["disagreement"]).sum())),np.zeros(int((~r["a_correct"] & r["disagreement"]).sum()))].ravel()); grad_s.append(np.r_[np.abs(r["M"])[r["a_correct"]&r["disagreement"]].ravel(),np.abs(r["M"])[~r["a_correct"]&r["disagreement"]].ravel()])
    useful_f=np.concatenate([x for x in useful if x.size]); harmful_f=np.concatenate([x for x in harmful if x.size]); route_yf=np.concatenate(route_y); route_sf=np.concatenate(route_s); grad_yf=np.concatenate(grad_y); grad_sf=np.concatenate(grad_s)
    sep_agr=auc_safe(np.r_[np.ones(useful_f.size),np.zeros(harmful_f.size)],np.r_[useful_f,harmful_f]); sep_route=auc_safe(route_yf,route_sf); sep_grad=auc_safe(grad_yf,grad_sf)
    a200=[r for r in alpha_rows if r["epoch"]==200]; alpha_one=next(r for r in a200 if r["alpha"]==1.0); alpha_eight=next(r for r in a200 if r["alpha"]==8.0)
    # Transparent, fixed interpretive rubric, applied after the prespecified
    # replay; no validation choice is made by this script.
    direction_ok=ref_summary["summary"]["auroc"] >= .5 and ref_summary["summary"]["balanced_accuracy"] >= .5
    agreement_effective=e200["WW_reverse_any_rate"] >= .25 and e200["agreement_intervention_precision"] >= .5
    d_under=e200["routing_accuracy"] >= .60 and e200["useful_removed_fraction"] < .25
    both_under=not agreement_effective and d_under
    selectivity_limited=(np.isfinite(sep_agr) and abs(sep_agr-.5)<.10 and np.isfinite(sep_route) and abs(sep_route-.5)<.10 and alpha_eight["gradient_intervention_precision"] <= alpha_one["gradient_intervention_precision"])
    if not direction_ok: decision="RAW_MAPPING_NOT_SUPPORTED"
    elif selectivity_limited: decision="SELECTIVITY_LIMITED_NOT_SCALE_LIMITED"
    elif both_under: decision="GLOBAL_UNDERUTILIZATION"
    elif agreement_effective and d_under: decision="AGREEMENT_OK_DISAGREEMENT_UNDERUTILIZED"
    else: decision="RAW_MAPPING_SUPPORTED"
    evidence="STRONG" if direction_ok and np.isfinite(ref_summary["summary"]["auroc"]) and (e200["routing_accuracy"] > .65 or e200["WW_reverse_any_rate"] > .4) else "MODERATE"
    e200_json={k:v for k,v in e200.items() if k not in ("states","extra")}
    diagnosis={"final_decision":decision,"evidence_strength":evidence,"fixed_interpretive_rubric":{"direction_ok":direction_ok,"agreement_effective":agreement_effective,"disagreement_underutilized":d_under,"selectivity_limited":selectivity_limited},"reference":{"AUROC":ref_summary["summary"]["auroc"],"AUPRC":ref_summary["summary"]["auprc"],"balanced_accuracy":ref_summary["summary"]["balanced_accuracy"],"accuracy":ref_summary["summary"]["accuracy"],"tie_rate":ref_summary["summary"]["tie_rate"],"pearson_S_FG_S_BG":ref_summary["distribution_row"]["pearson_S_FG_S_BG"],"spearman_S_FG_S_BG":ref_summary["distribution_row"]["spearman_S_FG_S_BG"],"mean_abs_M":float(am.mean()),"abs_M_p50":float(np.percentile(am,50)),"abs_M_p95":float(np.percentile(am,95)),"small_abs_M_lt_005_fraction":float(small.mean()),"small_abs_M_lt_005_reference_accuracy":small_rel.get("accuracy")},"epoch_200":e200_json,"separation":{"AUROC_abs_M_useful_WW_vs_harmful_CC":sep_agr,"AUROC_abs_M_routing_correct":sep_route,"AUROC_abs_M_useful_vs_harmful_removed_gradient":sep_grad},"alpha_epoch_200":{"alpha_1":alpha_one,"alpha_8":alpha_eight},"training_performed":False,"test_loaded":False,"validation_images":17,"bootstrap":{"repetitions":BOOTSTRAP_REPS,"seed":BOOTSTRAP_SEED,"file":str(bootstrap_path)}}
    (out/"12_final_diagnosis.json").write_text(json.dumps(json_safe(diagnosis),indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return diagnosis


def report(out: Path, diagnosis: Mapping[str, Any], temporal: Sequence[Mapping[str, Any]], alpha_rows: Sequence[Mapping[str, Any]]) -> None:
    d=diagnosis; e=d["epoch_200"]; r=d["reference"]; sep=d["separation"]
    key={x["epoch"]:x for x in temporal}
    lines=["# Stage M0 — UNI-R224 Margin-to-Intervention Validity Diagnosis", "", "## Protocol", "", "Protocol status: PASS. This report is validation-only retrospective diagnosis; no training, backward, optimizer, scheduler, test loader, test image, test metric, threshold/temperature tuning, or formal State-Aware modification was performed.", "", "- Dataset/split: GLaS, labeled 10%, seed 24; 7 supervised / 61 unsupervised / 17 validation.", "- UNI: 224×224, 14×14, 196 tokens, dim 1024, K=5; formal RGB/Stain student views; formal RGB-only UNI reference preprocessing.", "- HC90/HC95: exact existing joint confidence `min(max(pA,1-pA), max(pB,1-pB)) >= threshold`.", "- `training_performed = false`; `test_loaded = false`.", "", "## Reference margin and reliability", "", f"Q1/Q2: mean |M|={r['mean_abs_M']:.6g}, median={r['abs_M_p50']:.6g}, P95={r['abs_M_p95']:.6g}; |M|<0.05 accounts for {r['small_abs_M_lt_005_fraction']:.3%} of pixels and has reference accuracy {r['small_abs_M_lt_005_reference_accuracy']:.3%}. Pearson(S_FG,S_BG)={r['pearson_S_FG_S_BG']:.4f}, Spearman={r['spearman_S_FG_S_BG']:.4f}. Correlation alone is not treated as causal proof of residual compression; the direct S_FG/S_BG/M distributions and fixed-bin curves are the evidence.", f"Reference direction: AUROC={r['AUROC']:.4f}, AUPRC={r['AUPRC']:.4f}, balanced accuracy={r['balanced_accuracy']:.4f}, accuracy={r['accuracy']:.4f}, tie rate={r['tie_rate']:.6g}.", "", "## Agreement", "", f"Q3/Q4: epoch 200 WW RefCorrect={e['WW_ref_correct_rate']:.3%}; P(R>p_opp | WW & RefCorrect)={e['P_R_gt_popp_WW_RefCorrect']:.3%}; reverse-any={e['WW_reverse_any_rate']:.3%}; reverse-both={e['WW_reverse_both_rate']:.3%}; HC90 reverse={e['HC90_reverse_rate']:.3%}; HC95 reverse={e['HC95_reverse_rate']:.3%}.", f"WW useful mass U_A={e['total_R_WW']:.6g}, CC harmful mass H_A={e['total_R_CC']:.6g}; U_A/H_A={e['agreement_mass_ratio']:.6g}; mean_R_WW={e['mean_R_WW']:.6g}, mean_R_CC={e['mean_R_CC']:.6g}; intervention precision={e['agreement_intervention_precision']:.3%}. The report keeps mean-level and total-mass conclusions separate because CC is usually much more numerous.", "The binary gradient identity is numerically replayed from the formal tensors: agreement gradient shift is -R on the opposite-class logit, with reported L1/L2 totals equal to 2R and sqrt(2)R per receiver.", "", "## Disagreement", "", f"Q5/Q6: epoch 200 routing accuracy={e['routing_accuracy']:.3%}; mean R wrong teacher={e['mean_R_wrong_teacher']:.6g}, correct teacher={e['mean_R_correct_teacher']:.6g}; useful removed fraction={e['useful_removed_fraction']:.3%}; harmful removed fraction={e['harmful_removed_fraction']:.3%}; gradient precision={e['gradient_intervention_precision']:.3%}; relative D mass={e['relative_D_mass']:.6g}.", "Thus routing correctness and actual attenuation are reported as separate quantities; high routing accuracy does not by itself imply useful gradient deletion.", "", "## |M| selectivity and alpha replay", "", f"Q7: AUROC(|M|→useful WW vs harmful CC)={sep['AUROC_abs_M_useful_WW_vs_harmful_CC']:.4f}; AUROC(|M|→routing correct)={sep['AUROC_abs_M_routing_correct']:.4f}; AUROC(|M|→useful vs harmful removed gradient)={sep['AUROC_abs_M_useful_vs_harmful_removed_gradient']:.4f}. These are unweighted pixel/direction diagnostics; fixed-bin gradient masses provide the weighted view.", f"Q8/Q9: alpha is counterfactual only. At epoch 200, alpha=1 gives Agreement precision={next(x for x in alpha_rows if x['epoch']==200 and x['alpha']==1.0)['agreement_intervention_precision']:.3%}, D precision={next(x for x in alpha_rows if x['epoch']==200 and x['alpha']==1.0)['gradient_intervention_precision']:.3%}; alpha=8 gives Agreement precision={next(x for x in alpha_rows if x['epoch']==200 and x['alpha']==8.0)['agreement_intervention_precision']:.3%}, D precision={next(x for x in alpha_rows if x['epoch']==200 and x['alpha']==8.0)['gradient_intervention_precision']:.3%}. No alpha was selected or applied to the formal model.", "", "## Temporal checkpoints", "", "| epoch | WW reverse | Agreement precision | routing accuracy | useful removed fraction | D precision |", "|---:|---:|---:|---:|---:|---:|"]
    for epoch in (40,80,120,160,200):
        x=key[epoch]; lines.append(f"| {epoch} | {x['WW_reverse_any_rate']:.3%} | {x['agreement_intervention_precision']:.3%} | {x['routing_accuracy']:.3%} | {x['useful_removed_fraction']:.3%} | {x['gradient_intervention_precision']:.3%} |")
    lines += ["", "The complete 20-checkpoint temporal CSV and figures are retained; the five rows above are checkpoints requested for interpretation, not cherry-picking.", "", "## Bootstrap", "", "The epoch-200 image-level bootstrap uses 17 validation images, 2000 repetitions, seed 1337. Ratio metrics include numerator/denominator columns in `10_bootstrap_summary.csv`; image resampling is the unit.", "", "## Final decision", "", f"**{d['final_decision']}**", "", f"evidence_strength = **{d['evidence_strength']}**", "", "All requested artifacts are in `diagnostics/m0_margin_intervention_r224_glas10/`. The script stops here and does not enter any later stage."]
    (out/"13_final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out=ROOT/"diagnostics"/"m0_margin_intervention_r224_glas10"
    if out.exists() and any(out.iterdir()): mismatch("M0 output directory is non-empty; refusing overwrite: %s" % out)
    ckpt_dir=ROOT/"checkpoints"/"semi_10_seed_24_rgbstain_uni_reference_state_aware"/"GLaS"/"CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=uni_reference_state_aware"/"milestones"
    checkpoints=load_checkpoints(ckpt_dir)
    first_payload=torch.load(checkpoints[0][1],map_location="cpu",weights_only=False)
    memory_path=Path(first_payload.get("reference_memory_path",ROOT/"checkpoints/glas_labeled10_seed24/reference_memory.npz"))
    memory, memory_sha, memory_meta=load_memory(memory_path,first_payload.get("reference_memory_sha256"))
    network=checkpoint_metadata_assertions(checkpoints,memory_sha)
    if not torch.cuda.is_available(): mismatch("CUDA is required for the formal UNI implementation")
    device=torch.device("cuda")
    # The repository's config key is historically ``GlaS`` while the formal
    # protocol/report name is ``GLaS``.
    cfg=dataset_cfg("GlaS"); dataset,loader,manifest=build_validation(cfg)
    if len(loader.dataset)!=17: mismatch("loader does not contain 17 validation images")
    out.mkdir(parents=True,exist_ok=False); (out/"figures").mkdir()
    reuse_source = getattr(args, "reuse_cache_source", None)
    if reuse_source:
        source = Path(reuse_source).resolve()
        if not (source / "cache").is_dir(): mismatch("reuse cache source missing cache: %s" % source)
        shutil.copytree(source / "cache", out / "cache")
        ref_rows=load_reference_cache(out)
        uni_weights=Path(UNI_WEIGHTS_DEFAULT); uni_config=Path(UNI_CONFIG_DEFAULT)
        if sha256_file(uni_weights) != EXPECTED_UNI_SHA256: mismatch("UNI SHA mismatch while reusing cache")
    else:
        uni,uni_weights,uni_config=load_uni(UNI_WEIGHTS_DEFAULT,UNI_CONFIG_DEFAULT,expected_sha256=EXPECTED_UNI_SHA256)
        if uni.training or any(p.requires_grad for p in uni.parameters()): mismatch("UNI is not frozen/eval")
        ref_rows=build_reference_cache(out,loader,cfg,memory,uni,device); del uni; torch.cuda.empty_cache()
    ref={"rows":ref_rows}; ref_summary=reference_outputs(out,ref)
    if not reuse_source:
        infer_all(out,checkpoints,network,cfg,loader,device)
    analyses=[]; all_image=[]; agreement_bins=[]; disagreement_bins=[]; alpha_rows=[]
    for epoch,_ in checkpoints:
        with np.load(out/"cache"/"student_predictions"/("epoch_%03d.npz"%epoch),allow_pickle=False) as z: la,lb=z["logits_a"],z["logits_b"]
        states=[]
        for rr,aa,bb in zip(ref_rows,la,lb): states.append(state_from_logits(aa,bb,rr["gt"],rr["M"]))
        arows, irows, extra=aggregate_state({"rows":ref_rows},la,lb,epoch)
        arows["states"]=states; add_hc_fields({"rows":ref_rows},arows,la,lb); arows.pop("states",None)
        # Keep states only in memory for bin analysis; it is never serialized.
        analysis={**arows,"states":states,"extra":extra}; analyses.append(analysis); all_image.extend(irows); ab,db=make_bin_rows(epoch,ref,analysis); agreement_bins.extend(ab); disagreement_bins.extend(db); alpha_rows.extend(extra["alpha"])
    write_csv(out/"01_margin_distribution.csv",margin_distribution_dynamic(ref,analyses)+[ref_summary["distribution_row"]])
    write_csv(out/"04_agreement_checkpoint_summary.csv",[{k:v for k,v in a.items() if k not in ("states","extra")} for a in analyses])
    write_csv(out/"05_agreement_margin_bins.csv",agreement_bins); write_csv(out/"06_disagreement_checkpoint_summary.csv",[{k:v for k,v in a.items() if k not in ("states","extra")} for a in analyses]); write_csv(out/"07_disagreement_margin_bins.csv",disagreement_bins); write_csv(out/"08_alpha_scale_sensitivity.csv",alpha_rows); write_csv(out/"09_per_image_metrics.csv",all_image)
    temporal=[{k:v for k,v in a.items() if k not in ("states","extra")} for a in analyses]; write_csv(out/"11_temporal_summary.csv",temporal); bootstrap(out,[r for r in all_image if int(r["epoch"])==200]); make_figures(out,ref_summary,agreement_bins,disagreement_bins,temporal)
    diagnosis=final_diagnosis(out,ref_summary,analyses,alpha_rows,out/"10_bootstrap_summary.csv"); report(out,diagnosis,temporal,alpha_rows)
    protocol={"protocol_status":"PASS","stage":"M0","dataset":DATASET,"split":"validation","labeled_ratio":LABELED_RATIO,"split_seed":SPLIT_SEED,"validation_images":17,"checkpoints":list(EPOCHS),"checkpoint_count":20,"reference_resolution":224,"uni_grid":[14,14],"uni_tokens":196,"uni_feature_dim":1024,"uni_k":K,"anchor_fg":ANCHOR_FG,"anchor_bg":ANCHOR_BG,"memory_cap_per_image_class":CAP,"memory_seed":MEMORY_SEED,"memory_fg":EXPECTED_MEMORY_FG,"memory_bg":EXPECTED_MEMORY_BG,"memory_path":str(memory_path),"memory_sha256":memory_sha,"memory_metadata":memory_meta,"uni_weights":str(uni_weights),"uni_config":str(uni_config),"uni_sha256":EXPECTED_UNI_SHA256,"input_mode":INPUT_MODE,"student_views":{"A":"image_rgb","B":"image_stain"},"reference_view":"reconstructed RGB from image_rgb","interpolation":"bilinear","align_corners":False,"validation_transform":"data_transform_2d()['val']","strict_checkpoint_load":True,"model_eval":True,"torch_inference_mode":True,"training_performed":False,"test_loaded":False,"test_dataset_loaded":False,"test_loader_created":False,"backward_called":False,"optimizer_created":False,"scheduler_created":False,"validation_gt_used_for_diagnostics_only":True,"threshold_tuning":False,"temperature_tuning":False,"R_formal_unchanged":True,"alpha_counterfactual_only":True,"output_overwrite_refused":True,"output_directory":str(out)}
    (out/"00_protocol.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print("M0_COMPLETE",json.dumps({"protocol":"PASS","checkpoints":20,"validation_images":17,"reference_mean_abs_M":diagnosis["reference"]["mean_abs_M"],"reference_p50_abs_M":diagnosis["reference"]["abs_M_p50"],"reference_auroc":diagnosis["reference"]["AUROC"],"epoch200_agreement_precision":diagnosis["epoch_200"]["agreement_intervention_precision"],"epoch200_ww_reverse":diagnosis["epoch_200"]["WW_reverse_any_rate"],"epoch200_routing_accuracy":diagnosis["epoch_200"]["routing_accuracy"],"epoch200_useful_removed_fraction":diagnosis["epoch_200"]["useful_removed_fraction"],"decision":diagnosis["final_decision"],"evidence":diagnosis["evidence_strength"],"output":str(out)},ensure_ascii=False))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint-dir",default=None); parser.add_argument("--reuse-cache-source",default=None); args=parser.parse_args()
    try: run(args)
    except ProtocolMismatch as exc:
        print("PROTOCOL_MISMATCH: %s" % exc, file=sys.stderr); raise SystemExit(2)


if __name__ == "__main__": main()
