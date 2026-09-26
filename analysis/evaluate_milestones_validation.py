#!/usr/bin/env python3
"""Evaluate selected CPS milestone checkpoints on the fixed validation split.

The script reuses the project's dataset, model, validation forward, and Dice
evaluator.  It does not train or modify checkpoints.

Example:
  python analysis/evaluate_milestones_validation.py \
    --checkpoint-root /data/mission/models/Exp/DUAL/checkpoints/native_d_adaptive_fine_seed24/GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-cps_mode=standard-reference_mode=native_suppress_d_adaptive_fine/milestones \
    --epochs 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 \
    --input-mode rgb_stain \
    --output outputs/coarse2fine_noreverse_cps_rgbstain.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# Allow the script to be invoked directly as
# ``python analysis/evaluate_milestones_validation.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as project
from config.eval_config.eval import evaluate


DEFAULT_EPOCHS = [10,20,30,40,50,60,70,80,90,100,110, 120,130,140,150, 160,170,180,190, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/native_d_adaptive_fine_reverse_seed24",
        help="Root containing GLaS/<experiment>/milestones/.",
    )
    parser.add_argument(
        "--milestones-dir",
        default=None,
        help="Exact milestones directory; overrides --checkpoint-root discovery.",
    )
    parser.add_argument("--path-dataset", default="datasets/GlaS")
    parser.add_argument("--dataset-name", default="GlaS")
    parser.add_argument("--labeled-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--input-mode", default="rgb_stain")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output",
        default="outputs/milestone_validation_metrics.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def find_milestones(args: argparse.Namespace) -> Path:
    if args.milestones_dir:
        path = Path(args.milestones_dir)
    else:
        root = Path(args.checkpoint_root)
        # Accept both an experiment root and an already-resolved milestones
        # directory, e.g. ``.../reference_mode=.../milestones``.
        if root.name == "milestones":
            path = root
        else:
            matches = sorted(root.glob("**/milestones"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Expected exactly one milestones directory under {} but found: {}".format(
                        args.checkpoint_root, matches
                    )
                )
            path = matches[0]
    if not path.is_dir():
        raise FileNotFoundError(f"Milestones directory does not exist: {path}")
    return path


def checkpoint_for_epoch(milestones: Path, epoch: int) -> Path:
    exact = [milestones / f"epoch_{epoch:03d}.pth", milestones / f"epoch_{epoch}.pth"]
    for path in exact:
        if path.is_file():
            return path
    available = sorted(p.name for p in milestones.glob("epoch_*.pth"))
    raise FileNotFoundError(
        f"Missing exact milestone for epoch {epoch} in {milestones}. "
        f"Available milestone files: {available}"
    )


def make_project_args(args: argparse.Namespace) -> SimpleNamespace:
    # These fields are the same fixed validation configuration used by main.py.
    return SimpleNamespace(
        manifest_dir=None,
        path_dataset=args.path_dataset,
        input1="image",
        input_mode=args.input_mode,
        labeled_ratio=args.labeled_ratio,
        sup_mark="7",
        unsup_mark="61",
        network="unet",
        cps_mode="standard",
        lr=0.5,
        momentum=0.9,
        wd=-5,
        step_size=50,
        gamma=0.5,
        warm_up_duration=20,
    )


def build_validation(args: argparse.Namespace):
    project_args = make_project_args(args)
    cfg = project.dataset_cfg(args.dataset_name)
    datasets, manifest_dir, use_manifests = project.build_datasets(project_args, cfg)
    loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return project_args, cfg, loader, manifest_dir, use_manifests


def build_models(cfg):
    model_a = project.get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    model_b = project.get_network("unet", cfg["IN_CHANNELS"], cfg["NUM_CLASSES"]).cuda()
    losses = {"ce": CrossEntropyLoss().cuda(), "dice": project.DiceLoss().cuda()}
    return (model_a, model_b), losses


@torch.no_grad()
def compute_shared_rates(validation: dict) -> dict:
    logits_a = validation["scores1"].cuda(non_blocking=True)
    logits_b = validation["scores2"].cuda(non_blocking=True)
    gt = validation["masks"].cuda(non_blocking=True)
    valid = gt.ne(-1)

    prob_a = project._foreground_probability(logits_a)[:, 0]
    prob_b = project._foreground_probability(logits_b)[:, 0]
    hard_a = prob_a.ge(0.5)
    hard_b = prob_b.ge(0.5)
    gt_fg = gt.eq(1)
    correct_a = hard_a.eq(gt_fg)
    correct_b = hard_b.eq(gt_fg)
    agreement = hard_a.eq(hard_b)
    ww = (~correct_a) & (~correct_b) & agreement & valid
    confidence_a = torch.maximum(prob_a, 1.0 - prob_a)
    confidence_b = torch.maximum(prob_b, 1.0 - prob_b)
    hc90ww = ww & confidence_a.ge(0.90) & confidence_b.ge(0.90)

    valid_count = valid.sum().item()
    def rate(mask: torch.Tensor) -> float:
        return float(mask.sum().item() / valid_count) if valid_count else float("nan")

    dice_a = evaluate(validation["scores1"].cpu(), validation["masks"].cpu())[2]
    dice_b = evaluate(validation["scores2"].cpu(), validation["masks"].cpu())[2]
    return {
        "dice_student1": float(dice_a),
        "dice_student2": float(dice_b),
        "dice_mean": 0.5 * (float(dice_a) + float(dice_b)),
        "agreement_rate": rate(agreement & valid),
        "WW_rate": rate(ww),
        "HC90WW_rate": rate(hc90ww),
        "valid_pixels": int(valid_count),
        "agreement_pixels": int((agreement & valid).sum().item()),
        "WW_pixels": int(ww.sum().item()),
        "HC90WW_pixels": int(hc90ww.sum().item()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This project evaluator expects CUDA; no CUDA device is available.")

    milestones = find_milestones(args)
    project_args, cfg, val_loader, manifest_dir, use_manifests = build_validation(args)
    models, losses = build_models(cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for epoch in args.epochs:
        checkpoint = checkpoint_for_epoch(milestones, epoch)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"appearance_state_dict", "structure_state_dict"}
        missing = required.difference(payload)
        if missing:
            raise KeyError(f"Checkpoint {checkpoint} is missing keys: {sorted(missing)}")
        models[0].load_state_dict(payload["appearance_state_dict"], strict=True)
        models[1].load_state_dict(payload["structure_state_dict"], strict=True)
        validation = project.validate(models, losses, val_loader)
        metrics = compute_shared_rates(validation)
        rows.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "manifest_dir": str(manifest_dir),
            "use_manifests": bool(use_manifests),
            **metrics,
        })
        print(
            "epoch={:03d} | Dice={:.6f} | Agreement={:.6f} | WW={:.6f} | HC90WW={:.6f}".format(
                epoch, metrics["dice_mean"], metrics["agreement_rate"],
                metrics["WW_rate"], metrics["HC90WW_rate"]
            )
        )

    fields = list(rows[0].keys()) if rows else ["epoch", "checkpoint"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
