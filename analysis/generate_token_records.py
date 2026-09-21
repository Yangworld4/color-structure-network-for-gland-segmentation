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


DEFAULT_MEMORY = ROOT / "checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz"
DEFAULT_OUTPUT = ROOT / "outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv"


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="GlaS",
        path_dataset=str(ROOT / "datasets/GLaS"),
        manifest_dir=None,
        labeled_ratio=10,
        input1="image",
        input_mode="rgb_rgb",
        sup_mark="9",
        unsup_mark="76",
    )


def _load_memory(path: Path):
    with np.load(str(path), allow_pickle=False) as memory:
        required = {"memory_fg", "memory_bg", "top_k", "token_grid", "feature_dim"}
        missing = sorted(required.difference(memory.files))
        if missing:
            raise RuntimeError(f"Memory missing fields: {missing}")
        memory_fg = np.asarray(memory["memory_fg"], dtype=np.float32)
        memory_bg = np.asarray(memory["memory_bg"], dtype=np.float32)
        top_k = int(memory["top_k"])
        token_grid = int(memory["token_grid"])
        feature_dim = int(memory["feature_dim"])
    if token_grid != 16 or feature_dim != 1024:
        raise RuntimeError(
            f"Expected UNI256 memory grid=16 dim=1024, got grid={token_grid} dim={feature_dim}"
        )
    if not np.allclose(np.linalg.norm(memory_fg, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("FG memory is not L2-normalized")
    if not np.allclose(np.linalg.norm(memory_bg, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("BG memory is not L2-normalized")
    return (
        torch.from_numpy(memory_fg).cuda(non_blocking=True),
        torch.from_numpy(memory_bg).cuda(non_blocking=True),
        top_k,
    )


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

    uni, weights_path, config_path = project.load_uni(
        UNI_WEIGHTS_DEFAULT,
        UNI_CONFIG_DEFAULT,
        expected_sha256=EXPECTED_UNI_SHA256,
    )
    memory_fg, memory_bg, memory_top_k = _load_memory(memory_path)
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
                rgb = project._reconstruct_rgb(
                    project._get_rgb_view(batch).cuda(non_blocking=True), cfg
                )
                tokens = project._uni_dense_features(uni, rgb, 256)
                score_fg, score_bg = project._reference_scores(
                    tokens,
                    {"memory_fg": memory_fg, "memory_bg": memory_bg},
                    top_k,
                )
                gt = batch["mask"].cuda(non_blocking=True).float().unsqueeze(1)
                occupancy = F.adaptive_avg_pool2d(gt, (16, 16))[:, 0]
                score_fg = score_fg.detach().cpu().numpy()
                score_bg = score_bg.detach().cpu().numpy()
                occupancy = occupancy.detach().cpu().numpy()
                ids = [str(value) for value in batch["ID"]]
                total_images += len(ids)
                for image_index, image_id in enumerate(ids):
                    for row in range(16):
                        for col in range(16):
                            fg_ratio = float(occupancy[image_index, row, col])
                            if fg_ratio >= 0.90:
                                gt_class = "FG"
                            elif fg_ratio <= 0.10:
                                gt_class = "BG"
                            elif not args.include_mixed:
                                continue
                            else:
                                gt_class = "MIXED"
                            s_fg = float(score_fg[image_index, row * 16 + col])
                            s_bg = float(score_bg[image_index, row * 16 + col])
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
    print(f"Top-K used for retrieval: {top_k} (memory metadata: {memory_top_k})")
    print(f"UNI weights: {weights_path}")
    print(f"UNI config: {config_path}")
    print(f"Manifest: {manifest_dir} (use_manifests={use_manifests})")
    print(f"Output: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
