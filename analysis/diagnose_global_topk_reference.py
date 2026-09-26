#!/usr/bin/env python3
"""Controlled comparison of class-wise and global UNI Top-K retrieval.

This is an inference-only GlaS validation diagnostic. It reuses the existing
native UNI256 preprocessing, the saved memory, the synchronized CPS checkpoint,
and the token-level CC/WW definitions from diagnose_contradiction_enrichment.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project
from config.dataset_config.dataset_cfg import dataset_cfg
from config.uni_config.uni_config import UNI_CONFIG_DEFAULT, UNI_WEIGHTS_DEFAULT
from analysis.diagnose_contradiction_enrichment import select_checkpoints
from models.getnetwork import get_network
from models.uni import load_uni
from training.cps import get_training_views
from utils.native_reference import (
    _get_rgb_view,
    _reconstruct_rgb,
    _reference_scores,
    _uni_dense_features,
)
from utils.training_diagnostics import foreground_probability


DATASET = "GlaS"
SEED = 1337
GRID = 16
K = 50
TAU = 0.1
HC_THRESHOLD = 0.90
MEMORY_PATH = ROOT / "checkpoints/semi_10_seed_1337_rgbstain_coarse2fine/GlaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=native_fine_margin_calibrated_reverse/reference_memory_uni256.npz"
CHECKPOINT_ROOT = ROOT / "checkpoints/semi_10_seed_1337_rgb/GlaS"
OUTPUT_DIR = ROOT / "outputs/global_topk_k50_glas10_seed1337_rawcps"
REGIONS = ["ALL", "CC", "WW", "HC90WW", "Shared FP", "Shared FN"]
REGION_FLAGS = {
    "ALL": None,
    "CC": "is_cc",
    "WW": "is_ww",
    "HC90WW": "is_hc90ww",
    "Shared FP": "is_shared_fp",
    "Shared FN": "is_shared_fn",
}
CSV_FIELDS = [
    "image_id", "epoch", "checkpoint_id", "token_y", "token_x", "gt",
    "student1_pred", "student2_pred", "student1_conf", "student2_conf",
    "region_type", "is_cc", "is_ww", "is_hc90ww", "is_shared_fp", "is_shared_fn",
    "current_fg_score", "current_bg_score", "current_margin", "current_ref_pred", "current_correct",
    "global_fg_count", "global_bg_count", "majority_ratio", "majority_ref_pred", "majority_correct",
    "weighted_p_fg", "weighted_p_bg", "weighted_ref_pred", "weighted_correct", "weighted_conf_prob",
    "c_vote", "top50_mean_sim", "top50_max_sim", "weighted_s_mean",
]


def _fmt(value: float | int | None, percent: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{100 * float(value):.4f}%" if percent else f"{float(value):.6f}"


def _prediction(value: float) -> int | None:
    if value > 0.5:
        return 1
    if value < 0.5:
        return 0
    return None


def _direction_from_margin(value: float) -> int | None:
    if value > 0:
        return 1
    if value < 0:
        return 0
    return None


def load_memory(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as bank:
        print("Memory fields:", ", ".join(bank.files))
        required = {"memory_fg", "memory_bg", "memory_fg_count", "memory_bg_count", "feature_dim", "token_grid"}
        missing = required - set(bank.files)
        if missing:
            raise RuntimeError(f"Memory missing fields: {sorted(missing)}")
        fg_np = np.asarray(bank["memory_fg"], dtype=np.float32)
        bg_np = np.asarray(bank["memory_bg"], dtype=np.float32)
        metadata = {key: bank[key].item() for key in bank.files if bank[key].shape == ()}
    if metadata["memory_fg_count"] != len(fg_np) or metadata["memory_bg_count"] != len(bg_np):
        raise RuntimeError("Memory metadata counts do not match feature arrays")
    if metadata["feature_dim"] != 1024 or metadata["token_grid"] != GRID:
        raise RuntimeError(f"Unexpected memory metadata: {metadata}")
    fg = torch.from_numpy(fg_np).to(device)
    bg = torch.from_numpy(bg_np).to(device)
    fg = F.normalize(fg, dim=-1) if not torch.allclose(fg.norm(dim=-1), torch.ones(len(fg), device=device), atol=1e-4) else fg
    bg = F.normalize(bg, dim=-1) if not torch.allclose(bg.norm(dim=-1), torch.ones(len(bg), device=device), atol=1e-4) else bg
    memory = torch.cat([fg, bg], dim=0)
    labels = torch.cat([
        torch.ones(len(fg), device=device, dtype=torch.long),
        torch.zeros(len(bg), device=device, dtype=torch.long),
    ])
    # Sanity check 1: global labeled memory is exactly the two immutable banks.
    if len(memory) != len(fg) + len(bg) or len(labels) != len(memory):
        raise RuntimeError("Global memory size != FG size + BG size")
    if len(memory) < K:
        raise RuntimeError(f"Global memory size {len(memory)} is below K={K}")
    return fg, bg, memory, labels, metadata


def _summaries(records: list[dict]):
    result = {}
    for region in REGIONS:
        flag = REGION_FLAGS[region]
        subset = records if flag is None else [r for r in records if r[flag]]
        entry = {"n": len(subset)}
        for method in ("current", "majority", "weighted"):
            valid = [r for r in subset if r[f"{method}_ref_pred"] is not None]
            correct = sum(r[f"{method}_correct"] == 1 for r in valid)
            entry[method] = {
                "valid": len(valid), "uncertain": len(subset) - len(valid),
                "correct": correct, "accuracy": correct / len(valid) if valid else None,
            }
        for field in ("fg_ratio", "bg_ratio", "majority_ratio", "top50_mean_sim", "top50_max_sim"):
            entry[field] = float(np.mean([r[field] for r in subset])) if subset else None
        weighted_fields = {
            "mean_p_fg": "weighted_p_fg", "mean_c_prob": "weighted_conf_prob",
            "median_c_prob": "weighted_conf_prob", "mean_c_vote": "c_vote",
            "mean_s_mean": "weighted_s_mean",
        }
        for output, source in weighted_fields.items():
            vals = [r[source] for r in subset]
            if vals:
                entry[output] = float(np.median(vals) if output == "median_c_prob" else np.mean(vals))
            else:
                entry[output] = None
        for correctness, name in ((1, "correct"), (0, "wrong")):
            vals = [r["weighted_conf_prob"] for r in subset if r["weighted_correct"] == correctness]
            entry[f"{name}_mean_c_prob"] = float(np.mean(vals)) if vals else None
            entry[f"{name}_count"] = len(vals)
        result[region] = entry
    return result


def _validate_existing_classwise_baseline(records: list[dict]) -> str:
    """Compare Method A and masks against the saved raw CPS diagnostic."""
    baseline_path = ROOT / "outputs/contradiction_enrichment_glas10_seed1337/contradiction_token_records.csv"
    if not baseline_path.is_file():
        raise RuntimeError(f"Existing Class-wise Top-K baseline records not found: {baseline_path}")
    with baseline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))

    def key(row):
        return (str(row["image_id"]), int(row["token_y"]), int(row["token_x"]))

    current = {key(row): row for row in records}
    previous = {key(row): row for row in baseline_rows}
    if current.keys() != previous.keys():
        raise RuntimeError(
            "STOP: Method A population differs from the existing diagnostic: "
            f"new={len(current)} prior={len(previous)} shared={len(current.keys() & previous.keys())}"
        )
    max_fg_diff = max(abs(current[k]["current_fg_score"] - float(previous[k]["s_fg"])) for k in current)
    max_bg_diff = max(abs(current[k]["current_bg_score"] - float(previous[k]["s_bg"])) for k in current)
    if max(max_fg_diff, max_bg_diff) > 2e-5:
        raise RuntimeError(
            "STOP: Class-wise Top-K differs from the existing saved baseline: "
            f"max |FG score diff|={max_fg_diff:.8g}, max |BG score diff|={max_bg_diff:.8g}"
        )
    def truth(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    for flag in ("is_cc", "is_ww", "is_hc90ww", "is_shared_fp", "is_shared_fn"):
        mismatch = sum(bool(current[k][flag]) != truth(previous[k][flag]) for k in current)
        if mismatch:
            raise RuntimeError(f"STOP: existing CPS region mask {flag} differs on {mismatch} tokens")
    baseline_acc = sum(
        (float(previous[k]["s_fg"]) > float(previous[k]["s_bg"])) == (previous[k]["gt_class"] == "FG")
        for k in previous
    ) / len(previous)
    current_acc = _summaries(records)["ALL"]["current"]["accuracy"]
    if abs(baseline_acc - current_acc) > 1e-12:
        raise RuntimeError(
            f"STOP: current ALL Class-wise direction accuracy {current_acc:.12f} "
            f"does not reproduce saved baseline {baseline_acc:.12f}"
        )
    return (
        f"PASS against {baseline_path}: population/masks exact; max FG score diff={max_fg_diff:.3g}; "
        f"max BG score diff={max_bg_diff:.3g}; ALL direction accuracy={current_acc:.6f}"
    )


def _collect(model_a, model_b, uni, loader, cfg, memory_context, epoch: int, checkpoint_path: Path):
    records = []
    memory_fg, memory_bg = memory_context["fg"], memory_context["bg"]
    global_memory, labels = memory_context["memory"], memory_context["labels"]
    with torch.inference_mode():
        for batch in loader:
            input_a, input_b = get_training_views(batch)
            logits_a = model_a(input_a.cuda(non_blocking=True))[0]
            logits_b = model_b(input_b.cuda(non_blocking=True))[0]
            prob_a = F.adaptive_avg_pool2d(foreground_probability(logits_a)[:, :1], (GRID, GRID))[:, 0]
            prob_b = F.adaptive_avg_pool2d(foreground_probability(logits_b)[:, :1], (GRID, GRID))[:, 0]
            class_a, class_b = prob_a.ge(0.5), prob_b.ge(0.5)
            conf_a, conf_b = torch.maximum(prob_a, 1 - prob_a), torch.maximum(prob_b, 1 - prob_b)
            agreement = class_a.eq(class_b)
            gt_mask = batch["mask"].cuda(non_blocking=True)
            if gt_mask.dim() == 3:
                gt_mask = gt_mask[:, None]
            gt_occ = F.adaptive_avg_pool2d(gt_mask.float(), (GRID, GRID))[:, 0]
            gt_class = gt_occ.ge(0.5)

            # Reuse the current native UNI256 query extraction and pixel/token mapping.
            rgb = _reconstruct_rgb(_get_rgb_view(batch).cuda(non_blocking=True), cfg)
            tokens = _uni_dense_features(uni, rgb, 256)
            if tuple(tokens.shape[1:]) != (GRID * GRID, 1024):
                raise RuntimeError(f"Unexpected query UNI token shape {tuple(tokens.shape)}")
            # Method A is the project's current implementation, untouched.
            score_fg, score_bg = _reference_scores(tokens, memory_context, K)
            q = F.normalize(tokens.reshape(-1, tokens.shape[-1]), dim=-1)
            # Independent vectorized check for Method A: same classwise cosine Top-K.
            direct_fg = torch.topk(q @ memory_fg.T, K, dim=-1).values.mean(-1)
            direct_bg = torch.topk(q @ memory_bg.T, K, dim=-1).values.mean(-1)
            ref_query_norm = tokens.reshape(-1, 1024).norm(dim=-1)
            baseline_fg = score_fg.reshape(-1)
            baseline_bg = score_bg.reshape(-1)
            if not torch.allclose(direct_fg * ref_query_norm, baseline_fg, atol=2e-5, rtol=2e-5):
                raise RuntimeError("Method A FG does not match current _reference_scores")
            if not torch.allclose(direct_bg * ref_query_norm, baseline_bg, atol=2e-5, rtol=2e-5):
                raise RuntimeError("Method A BG does not match current _reference_scores")

            # Methods B/C share one global Top-50 retrieval over FG union BG.
            sims = q @ global_memory.T
            values, indices = torch.topk(sims, K, dim=-1, largest=True, sorted=True)
            neighbor_labels = labels[indices]
            fg_count = neighbor_labels.sum(dim=-1)
            bg_count = K - fg_count
            # Sanity check 2: every query retrieves exactly 50 labeled neighbors.
            if values.shape[-1] != K or indices.shape[-1] != K:
                raise RuntimeError("Global Top-K neighbor count is not exactly 50")
            c_vote = torch.maximum(fg_count, bg_count).float() / K
            p_majority = fg_count.float() / K
            weights = torch.softmax(values / TAU, dim=-1)
            if float((weights.sum(dim=-1) - 1).abs().max()) >= 1e-5:
                raise RuntimeError("Weighted neighbor weights do not sum to 1")
            p_fg = (weights * neighbor_labels.float()).sum(dim=-1)
            p_bg = 1 - p_fg
            if float((p_fg + p_bg - 1).abs().max()) >= 1e-5:
                raise RuntimeError("Weighted P(FG)+P(BG) != 1")
            c_prob = (2 * p_fg - 1).abs()
            flat_a, flat_b = prob_a.flatten(1), prob_b.flatten(1)
            flat_ca, flat_cb = class_a.flatten(1), class_b.flatten(1)
            flat_gt = gt_class.flatten(1)
            flat_agree = agreement.flatten(1)
            flat_ww = flat_agree & flat_ca.ne(flat_gt)
            flat_cc = flat_agree & flat_ca.eq(flat_gt)
            flat_hc = flat_ww & conf_a.flatten(1).ge(HC_THRESHOLD) & conf_b.flatten(1).ge(HC_THRESHOLD)
            flat_fp = flat_ww & flat_ca & ~flat_gt
            flat_fn = flat_ww & ~flat_ca & flat_gt
            flat_fg, flat_bg = score_fg, score_bg
            image_ids = [str(x) for x in batch["ID"]]
            for b, image_id in enumerate(image_ids):
                for token_id in torch.nonzero(flat_agree[b], as_tuple=False).flatten().tolist():
                    a, c = divmod(token_id, GRID)
                    gt = int(flat_gt[b, token_id])
                    margin = float(flat_fg[b, token_id] - flat_bg[b, token_id])
                    current_pred = _direction_from_margin(margin)
                    global_index = b * GRID * GRID + token_id
                    fg_n = int(fg_count[global_index])
                    bg_n = int(bg_count[global_index])
                    majority_p = float(p_majority[global_index])
                    majority_pred = None if fg_n == bg_n else int(fg_n > bg_n)
                    weighted_p = float(p_fg[global_index])
                    weighted_pred = _prediction(weighted_p)
                    flags = {
                        "is_cc": bool(flat_cc[b, token_id]), "is_ww": bool(flat_ww[b, token_id]),
                        "is_hc90ww": bool(flat_hc[b, token_id]), "is_shared_fp": bool(flat_fp[b, token_id]),
                        "is_shared_fn": bool(flat_fn[b, token_id]),
                    }
                    active = [name for name, key in (("CC", "is_cc"), ("WW", "is_ww"), ("HC90WW", "is_hc90ww"), ("Shared FP", "is_shared_fp"), ("Shared FN", "is_shared_fn")) if flags[key]]
                    row = {
                        "image_id": image_id, "epoch": epoch, "checkpoint_id": checkpoint_path.name,
                        "token_y": a, "token_x": c, "gt": gt,
                        "student1_pred": int(flat_ca[b, token_id]), "student2_pred": int(flat_cb[b, token_id]),
                        "student1_conf": float(conf_a[b, a, c]), "student2_conf": float(conf_b[b, a, c]),
                        "region_type": "+".join(active) if active else "AGREEMENT_OTHER",
                        **{name: int(value) for name, value in flags.items()},
                        "current_fg_score": float(flat_fg[b, token_id]), "current_bg_score": float(flat_bg[b, token_id]),
                        "current_margin": margin, "current_ref_pred": current_pred,
                        "current_correct": int(current_pred == gt) if current_pred is not None else None,
                        "global_fg_count": fg_n, "global_bg_count": bg_n,
                        "majority_ratio": float(c_vote[global_index]), "majority_ref_pred": majority_pred,
                        "majority_correct": int(majority_pred == gt) if majority_pred is not None else None,
                        "weighted_p_fg": weighted_p, "weighted_p_bg": float(p_bg[global_index]),
                        "weighted_ref_pred": weighted_pred,
                        "weighted_correct": int(weighted_pred == gt) if weighted_pred is not None else None,
                        "weighted_conf_prob": float(c_prob[global_index]), "c_vote": float(c_vote[global_index]),
                        "top50_mean_sim": float(values[global_index].mean()), "top50_max_sim": float(values[global_index].max()),
                        "weighted_s_mean": float(values[global_index].mean()),
                        "fg_ratio": fg_n / K, "bg_ratio": bg_n / K,
                    }
                    records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=MEMORY_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing UNI implementation")
    args.memory = args.memory.resolve()
    if not args.memory.is_file():
        raise FileNotFoundError(f"Required memory missing: {args.memory}")
    # Reuse the original CPS validation-best synchronized checkpoint selection
    # and state verification used by the existing error-region diagnostic.
    paired_path, student_a_path, student_b_path, selected_checkpoint, selected_epoch = select_checkpoints()
    args.checkpoint = (args.checkpoint or paired_path).resolve()
    if args.checkpoint != paired_path.resolve():
        raise RuntimeError(f"Checkpoint override differs from current diagnostic selection: {args.checkpoint} != {paired_path}")
    project.init_seeds(SEED)
    device = torch.device("cuda")
    fg, bg, global_memory, labels, metadata = load_memory(args.memory, device)
    if len(global_memory) != len(fg) + len(bg):
        raise RuntimeError("Check 1 failed")
    print(f"Sanity Check 1 PASS: global={len(global_memory)} = FG {len(fg)} + BG {len(bg)}")
    data_args = SimpleNamespace(
        dataset_name=DATASET, path_dataset=str(ROOT / "datasets/GlaS"), manifest_dir=None,
        labeled_ratio=10, input1="image", input_mode="rgb_rgb", sup_mark="7", unsup_mark="61",
    )
    cfg = dataset_cfg(DATASET)
    datasets, manifest_dir, use_manifests = project.build_datasets(data_args, cfg)
    if not use_manifests:
        raise RuntimeError("Expected existing fixed GlaS validation manifest")
    loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "appearance_state_dict" not in checkpoint or "structure_state_dict" not in checkpoint:
        raise KeyError("Expected synchronized CPS checkpoint with both student states")
    model_a = get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).to(device)
    model_b = get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).to(device)
    model_a.load_state_dict(checkpoint["appearance_state_dict"], strict=True)
    model_b.load_state_dict(checkpoint["structure_state_dict"], strict=True)
    model_a.eval(); model_b.eval()
    uni, uni_weights, uni_config = load_uni(UNI_WEIGHTS_DEFAULT, UNI_CONFIG_DEFAULT)
    context = {"memory_fg": fg, "memory_bg": bg, "fg": fg, "bg": bg,
               "memory": global_memory, "labels": labels}
    records = _collect(model_a, model_b, uni, loader, cfg, context, int(checkpoint["epoch"]), args.checkpoint)
    if len(records) == 0:
        raise RuntimeError("No CPS agreement tokens were recorded")
    baseline_check = _validate_existing_classwise_baseline(records)
    # Check 3/4 are checked for every query in _collect; check 2 is also checked there.
    print("Sanity Checks 2-4 PASS: all Global Top-50 sets have 50 labels; softmax sums and P(FG)+P(BG) are within 1e-5")
    summary = _summaries(records)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    token_path = outdir / "global_topk_k50_token_records.csv"
    with token_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary_path = outdir / "global_topk_k50_summary.txt"
    lines = [
        "==============================================================",
        "UNI Reference Retrieval Diagnostic", f"K = {K}", f"tau = {TAU}",
        "==============================================================", "",
        f"Memory path: {args.memory}", f"Memory FG tokens: {len(fg)}", f"Memory BG tokens: {len(bg)}",
        f"Total memory size: {len(global_memory)}", f"Checkpoint: {args.checkpoint}",
        f"Checkpoint epoch: {checkpoint['epoch']}", f"Selected synchronized raw CPS checkpoint: {paired_path}",
        f"Student A validation-best branch: {student_a_path}", f"Student B validation-best branch: {student_b_path}",
        f"Validation manifest: {Path(manifest_dir) / 'validation.csv'}",
        f"Validation images: {len(datasets['val'])}", f"Agreement tokens recorded: {len(records)}",
        f"UNI weights: {uni_weights}", f"UNI config: {uni_config}",
        "Region masks: reused token-level definitions from diagnose_contradiction_enrichment.py; agreement tokens form ALL.",
        "CC = student agreement and token prediction equals GT; WW = agreement and token prediction differs from GT; HC90WW = WW and both pooled student confidences >= 0.90.",
        "Shared FP/FN use the same WW mask and the agreed prediction relative to GT.", "",
        "[Direction accuracy: correct / valid (uncertain ties excluded; counts shown)]",
        "Metric             ClassTopK   GlobalMajority   GlobalWeighted",
    ]
    for region in REGIONS:
        e = summary[region]
        vals = [e[m]["accuracy"] for m in ("current", "majority", "weighted")]
        lines.append(f"{region:<18} {_fmt(vals[0], True):>10} {_fmt(vals[1], True):>16} {_fmt(vals[2], True):>16} | N={e['n']} valid={e['current']['valid']}/{e['majority']['valid']}/{e['weighted']['valid']} uncertain={e['current']['uncertain']}/{e['majority']['uncertain']}/{e['weighted']['uncertain']}")
    lines += ["", "[Delta vs Current Class-wise Top-K, percentage points]"]
    for region in REGIONS:
        e = summary[region]
        current = e["current"]["accuracy"]
        deltas = []
        for m in ("majority", "weighted"):
            val = e[m]["accuracy"]
            deltas.append(None if current is None or val is None else (val - current) * 100)
        lines.append(f"{region:<18} Global Majority {('n/a' if deltas[0] is None else f'{deltas[0]:+.2f} pp')} | Global Weighted {('n/a' if deltas[1] is None else f'{deltas[1]:+.2f} pp')}")
    lines += ["", "[Global Top-K retrieval statistics; Methods B and C share retrieved neighbors]",
              "Region | mean FG neighbor ratio | mean BG neighbor ratio | mean majority ratio | mean Top50 cosine | mean max cosine"]
    for region in REGIONS[1:]:
        e = summary[region]
        lines.append(f"{region} | {_fmt(e['fg_ratio'], True)} | {_fmt(e['bg_ratio'], True)} | {_fmt(e['majority_ratio'], True)} | {_fmt(e['top50_mean_sim'])} | {_fmt(e['top50_max_sim'])}")
    lines += ["", "[Weighted confidence statistics]",
              "Region | mean P_ref(FG) | mean C_prob | median C_prob | mean C_vote | mean S_mean | correct mean C_prob (N) | wrong mean C_prob (N)"]
    for region in REGIONS[1:]:
        e = summary[region]
        lines.append(f"{region} | {_fmt(e['mean_p_fg'])} | {_fmt(e['mean_c_prob'])} | {_fmt(e['median_c_prob'])} | {_fmt(e['mean_c_vote'])} | {_fmt(e['mean_s_mean'])} | {_fmt(e['correct_mean_c_prob'])} ({e['correct_count']}) | {_fmt(e['wrong_mean_c_prob'])} ({e['wrong_count']})")
    lines += ["", "[Sanity checks]",
              f"1. Global memory = FG + BG: PASS ({len(global_memory)} = {len(fg)} + {len(bg)})",
              f"2. Every global retrieval has K={K}: PASS", "3. softmax weights sum to one within 1e-5: PASS",
              "4. weighted P(FG)+P(BG)=1 within 1e-5: PASS",
              "5. Classwise Top-K reproduces existing raw CPS diagnostic: " + baseline_check,
              "Method A scores also matched utils.native_reference._reference_scores tokenwise (score comparison atol=2e-5, rtol=2e-5).",
              "", f"CSV: {token_path}", f"Summary: {summary_path}"]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"CSV: {token_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
