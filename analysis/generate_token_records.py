#!/usr/bin/env python3
"""Generate native UNI validation token records for offline diagnostics.

This is a diagnostic-only script. It performs one frozen UNI forward over the
existing GLaS validation split, reads the existing immutable memory, and writes
token-level retrieval scores. It never trains, updates UNI, rebuilds memory, or
runs CPS.
"""

from __future__ import annotations

import argparse
import csv
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
from config.uni_config.uni_config import (
    EXPECTED_UNI_SHA256,
    UNI_CONFIG_DEFAULT,
    UNI_WEIGHTS_DEFAULT,
)
from models.uni import load_uni
from utils.native_reference import (
    _get_rgb_view,
    _reconstruct_rgb,
    _reference_scores,
    _uni_dense_features,
)


DEFAULT_MEMORY = ROOT / "/data/mission/models/Exp/DUAL/checkpoints/d/GlaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=native_fine_margin_calibrated_reverse/reference_memory_uni2h224.npz"
DEFAULT_OUTPUT = ROOT / "outputs/native_scale_reference/glas_standard_cps_uni2224/token_records.csv"


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="GlaS",
        path_dataset=str(ROOT / "datasets/GlaS"),
        manifest_dir=None,
        labeled_ratio=10,
        input1="image",
        input_mode="rgb_rgb",
        sup_mark="9",
        unsup_mark="76",
    )


def _load_memory(path: Path):
    with np.load(str(path), allow_pickle=False) as memory:
        required = {
            "memory_fg", "memory_bg", "top_k", "token_grid", "feature_dim",
        }
        missing = sorted(required.difference(memory.files))
        if missing:
            raise RuntimeError(f"Memory missing fields: {missing}")
        memory_fg = np.asarray(memory["memory_fg"], dtype=np.float32)
        memory_bg = np.asarray(memory["memory_bg"], dtype=np.float32)
        top_k = int(memory["top_k"])
        token_grid = int(memory["token_grid"])
        feature_dim = int(memory["feature_dim"])
        reference_resolution = (
            int(memory["reference_resolution"])
            if "reference_resolution" in memory.files else 256
        )
        encoder_id = (
            str(memory["encoder_id"].item())
            if "encoder_id" in memory.files else ""
        )
    if memory_fg.ndim != 2 or memory_bg.ndim != 2:
        raise RuntimeError("FG/BG memory arrays must have shape [N, feature_dim]")
    if memory_fg.shape[1] != feature_dim or memory_bg.shape[1] != feature_dim:
        raise RuntimeError(
            "Memory feature_dim metadata does not match FG/BG arrays: "
            f"metadata={feature_dim}, FG={memory_fg.shape}, BG={memory_bg.shape}"
        )
    if not np.allclose(np.linalg.norm(memory_fg, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("FG memory is not L2-normalized")
    if not np.allclose(np.linalg.norm(memory_bg, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("BG memory is not L2-normalized")
    return {
        "memory_fg": torch.from_numpy(memory_fg).cuda(non_blocking=True),
        "memory_bg": torch.from_numpy(memory_bg).cuda(non_blocking=True),
        "top_k": top_k,
        "token_grid": token_grid,
        "feature_dim": feature_dim,
        "reference_resolution": reference_resolution,
        "encoder_id": encoder_id,
    }


def _infer_backbone(memory):
    encoder_id = memory["encoder_id"].lower()
    if "uni2-h" in encoder_id or memory["feature_dim"] == 1536:
        backbone = "uni2-h"
    elif "kaiko" in encoder_id or memory["feature_dim"] == 768:
        backbone = "kaiko-vitb8"
    elif "uni" in encoder_id or memory["feature_dim"] == 1024:
        backbone = "uni"
    else:
        raise RuntimeError(
            "Cannot identify encoder for reference memory: "
            f"encoder_id={memory['encoder_id']!r}, feature_dim={memory['feature_dim']}"
        )
    expected = {
        "uni": (16, 1024, 16),
        "uni2-h": (14, 1536, 16),
        "kaiko-vitb8": (8, 768, 28),
    }[backbone]
    patch_size, feature_dim, token_grid = expected
    if memory["feature_dim"] != feature_dim or memory["token_grid"] != token_grid:
        raise RuntimeError(
            f"Memory metadata is inconsistent with {backbone}: "
            f"grid={memory['token_grid']} dim={memory['feature_dim']}"
        )
    if memory["reference_resolution"] != patch_size * token_grid:
        raise RuntimeError(
            f"Memory input resolution={memory['reference_resolution']} does not "
            f"match {backbone} grid={token_grid} and patch={patch_size}"
        )
    return backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen UNI validation token records")
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override memory metadata top-K for this diagnostic; does not modify the memory file.",
    )
    parser.add_argument(
        "--include-mixed", action="store_true",
        help="Write all 16x16 tokens, labeling 0.1<fg_ratio<0.9 as MIXED.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the existing UNI implementation")
    memory_path = args.memory.expanduser().resolve()
    if not memory_path.is_file():
        raise FileNotFoundError(f"Reference memory not found: {memory_path}")

    project.init_seeds(24)
    cfg = dataset_cfg("GlaS")
    data_args = _build_args()
    datasets, manifest_dir, use_manifests = project.build_datasets(data_args, cfg)
    validation_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    memory = _load_memory(memory_path)
    backbone = _infer_backbone(memory)
    loader = "hf-hub" if backbone == "uni2-h" else "local"
    hub_model_id = "MahmoodLab/UNI2-h" if backbone == "uni2-h" else "MahmoodLab/UNI"
    uni, weights_path, config_path = load_uni(
        UNI_WEIGHTS_DEFAULT,
        UNI_CONFIG_DEFAULT,
        expected_sha256=EXPECTED_UNI_SHA256 if backbone == "uni" else None,
        loader=loader,
        hub_model_id=hub_model_id,
        backbone=backbone,
    )
    if uni.feature_dim != memory["feature_dim"] or uni.patch_size * memory["token_grid"] != memory["reference_resolution"]:
        raise RuntimeError(
            "Loaded encoder does not match memory metadata: "
            f"encoder={backbone} dim={uni.feature_dim} patch={uni.patch_size}, "
            f"memory_dim={memory['feature_dim']} grid={memory['token_grid']} "
            f"resolution={memory['reference_resolution']}"
        )
    memory_fg, memory_bg = memory["memory_fg"], memory["memory_bg"]
    memory_top_k = memory["top_k"]
    token_grid = memory["token_grid"]
    resolution = memory["reference_resolution"]
    top_k = memory_top_k if args.top_k is None else int(args.top_k)
    if top_k < 1 or top_k > min(memory_fg.shape[0], memory_bg.shape[0]):
        raise ValueError(
            f"top-k={top_k} is invalid for memory sizes "
            f"FG={memory_fg.shape[0]}, BG={memory_bg.shape[0]}"
        )

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id", "token_row", "token_col", "S_FG", "S_BG", "Delta",
        "fg_ratio", "gt_class", "purity", "top_k",
    ]
    total_images = 0
    total_records = 0
    fg_records = 0
    bg_records = 0
    with args.output.expanduser().resolve().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for batch in validation_loader:
                rgb = _reconstruct_rgb(_get_rgb_view(batch).cuda(non_blocking=True), cfg)
                tokens = _uni_dense_features(uni, rgb, resolution)
                score_fg, score_bg = _reference_scores(
                    tokens,
                    {"memory_fg": memory_fg, "memory_bg": memory_bg},
                    top_k,
                )
                gt = batch["mask"].cuda(non_blocking=True).float().unsqueeze(1)
                occupancy = F.adaptive_avg_pool2d(gt, (token_grid, token_grid))[:, 0]
                score_fg = score_fg.detach().cpu().numpy()
                score_bg = score_bg.detach().cpu().numpy()
                occupancy = occupancy.detach().cpu().numpy()
                ids = [str(value) for value in batch["ID"]]
                total_images += len(ids)
                for image_index, image_id in enumerate(ids):
                    for row in range(token_grid):
                        for col in range(token_grid):
                            fg_ratio = float(occupancy[image_index, row, col])
                            if fg_ratio >= 0.90:
                                gt_class = "FG"
                            elif fg_ratio <= 0.10:
                                gt_class = "BG"
                            elif not args.include_mixed:
                                continue
                            else:
                                gt_class = "MIXED"
                            token_index = row * token_grid + col
                            s_fg = float(score_fg[image_index, token_index])
                            s_bg = float(score_bg[image_index, token_index])
                            writer.writerow({
                                "image_id": image_id,
                                "token_row": row,
                                "token_col": col,
                                "S_FG": s_fg,
                                "S_BG": s_bg,
                                "Delta": s_fg - s_bg,
                                "fg_ratio": fg_ratio,
                                "gt_class": gt_class,
                                "purity": "PURE",
                                "top_k": top_k,
                            })
                            total_records += 1
                            if gt_class == "FG":
                                fg_records += 1
                            elif gt_class == "BG":
                                bg_records += 1

    print("=== Token Record Generation ===")
    print(f"Validation images: {total_images}")
    print(f"Token records: {total_records}")
    print(f"FG records: {fg_records}")
    print(f"BG records: {bg_records}")
    print(f"Memory: {memory_path}")
    print(f"Backbone: {backbone} | input={resolution} | grid={token_grid}x{token_grid} | dim={uni.feature_dim}")
    print(f"Top-K used for retrieval: {top_k} (memory metadata: {memory_top_k})")
    print(f"UNI weights: {weights_path}")
    print(f"UNI config: {config_path}")
    print(f"Manifest: {manifest_dir} (use_manifests={use_manifests})")
    print(f"Output: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
