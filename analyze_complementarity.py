#!/usr/bin/env python3
"""Offline pixel-level complementarity analysis for the two CPS branches.

This script is analysis-only. It never trains, updates, or saves a model.
The oracle uses ground truth and is not deployable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.getnetwork import get_network


EPS = 1e-8
ORACLE_NOTICE = "Oracle is analysis-only upper bound and is not deployable."
REGION_NAMES = ("all_pixels", "foreground", "background", "boundary", "interior")
CONFIDENCE_MARGINS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
CONFIDENCE_BINS = (
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.50),
    (0.50, 1.000001),
)


@dataclass
class ComplementarityCounts:
    both_correct: int = 0
    a_only_correct: int = 0
    s_only_correct: int = 0
    both_wrong: int = 0

    @property
    def total(self) -> int:
        return (
            self.both_correct
            + self.a_only_correct
            + self.s_only_correct
            + self.both_wrong
        )

    def add(self, other: "ComplementarityCounts") -> None:
        self.both_correct += other.both_correct
        self.a_only_correct += other.a_only_correct
        self.s_only_correct += other.s_only_correct
        self.both_wrong += other.both_wrong


@dataclass
class BinaryCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def add(self, other: "BinaryCounts") -> None:
        self.tp += other.tp
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn


@dataclass
class WinnerCounts:
    disagreement_pixels: int = 0
    a_wins: int = 0
    s_wins: int = 0

    def add(self, other: "WinnerCounts") -> None:
        self.disagreement_pixels += other.disagreement_pixels
        self.a_wins += other.a_wins
        self.s_wins += other.s_wins


@dataclass
class PredictionPairCounts:
    count: int = 0
    gt_foreground: int = 0
    gt_background: int = 0
    appearance_correct: int = 0
    structure_correct: int = 0

    def add(self, other: "PredictionPairCounts") -> None:
        self.count += other.count
        self.gt_foreground += other.gt_foreground
        self.gt_background += other.gt_background
        self.appearance_correct += other.appearance_correct
        self.structure_correct += other.structure_correct


@dataclass
class ConfidenceCounts:
    total_disagreement_pixels: int = 0
    selected_pixels: int = 0
    correct_selections: int = 0

    def add(self, other: "ConfidenceCounts") -> None:
        self.total_disagreement_pixels += other.total_disagreement_pixels
        self.selected_pixels += other.selected_pixels
        self.correct_selections += other.correct_selections


@dataclass
class ConfidenceBinCounts:
    count: int = 0
    higher_confidence_correct: int = 0
    a_wins: int = 0
    s_wins: int = 0

    def add(self, other: "ConfidenceBinCounts") -> None:
        self.count += other.count
        self.higher_confidence_correct += other.higher_confidence_correct
        self.a_wins += other.a_wins
        self.s_wins += other.s_wins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="one checkpoint containing both branch state_dict objects",
    )
    checkpoint_group.add_argument(
        "--appearance-checkpoint",
        type=Path,
        help="Appearance branch checkpoint; requires --structure-checkpoint",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        help="Structure branch checkpoint used with --appearance-checkpoint",
    )
    parser.add_argument(
        "--dataset-root",
        "--path-dataset",
        dest="dataset_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        choices=("GlaS", "CRAG", "MoNuSeg"),
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("val", "validation", "test"),
        default="val",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional manifest override",
    )
    parser.add_argument("--network", default="xnet_sb")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA index, 'cpu', or 'auto' (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("complementarity_results/deep_analysis"),
    )
    parser.add_argument(
        "--boundary-width",
        type=int,
        default=3,
        help="foreground erosion radius in pixels (default: 3)",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="label excluded from every statistic (project manifests normally use none)",
    )
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--max-vis", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("complementarity_results/summary.json"),
        help="previous basic-analysis summary used only for regression checking",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="compute JSON/CSV statistics without generating paper figures",
    )
    parser.add_argument("--qualitative-samples", type=int, default=6)
    args = parser.parse_args()

    if args.appearance_checkpoint is not None and args.structure_checkpoint is None:
        parser.error("--appearance-checkpoint requires --structure-checkpoint")
    if args.checkpoint is not None and args.structure_checkpoint is not None:
        parser.error("do not combine --checkpoint with --structure-checkpoint")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    if args.boundary_width < 1:
        parser.error("--boundary-width must be at least 1")
    if args.max_vis < 0:
        parser.error("--max-vis must be non-negative")
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive")
    if not 4 <= args.qualitative_samples <= 6:
        parser.error("--qualitative-samples must be between 4 and 6")
    return args


def resolve_device(gpu: str) -> torch.device:
    value = str(gpu).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not re.fullmatch(r"\d+", value):
        raise ValueError("--gpu must be a CUDA index, 'cpu', or 'auto'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError(
            "CUDA device {} does not exist; available devices: {}".format(
                index, torch.cuda.device_count()
            )
        )
    return torch.device("cuda:{}".format(index))


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def extract_branch_state_dict(
    payload: Any,
    branch: str,
    allow_direct: bool = True,
) -> dict:
    """Extract a direct or nested state_dict without changing its tensors."""
    if _is_state_dict(payload) and allow_direct:
        return _strip_prefix(payload, "module.")
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if _is_state_dict(payload):
        payload = _strip_prefix(payload, "module.")

    aliases = {
        "appearance": (
            "appearance", "model1", "appearance_state_dict", "model1_state_dict"
        ),
        "structure": (
            "structure", "model2", "structure_state_dict", "model2_state_dict"
        ),
    }[branch]
    for key in aliases:
        candidate = payload.get(key)
        if _is_state_dict(candidate):
            return _strip_prefix(candidate, "module.")

    prefix_aliases = tuple(alias + "." for alias in aliases)
    for prefix in prefix_aliases:
        selected = {
            key[len(prefix):]: value
            for key, value in payload.items()
            if isinstance(key, str)
            and key.startswith(prefix)
            and torch.is_tensor(value)
        }
        if selected:
            return _strip_prefix(selected, "module.")

    for container_key in ("state_dict", "model_state_dict", "models"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in aliases:
            candidate = container.get(key)
            if _is_state_dict(candidate):
                return _strip_prefix(candidate, "module.")
        for prefix in prefix_aliases:
            selected = {
                key[len(prefix):]: value
                for key, value in container.items()
                if isinstance(key, str)
                and key.startswith(prefix)
                and torch.is_tensor(value)
            }
            if selected:
                return _strip_prefix(selected, "module.")

    raise KeyError(
        "could not find the {} branch in the combined checkpoint; expected "
        "one of keys {}".format(branch, aliases)
    )


def load_models(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Path, Path]:
    if args.checkpoint is not None:
        combined_path = args.checkpoint.expanduser().resolve()
        if not combined_path.is_file():
            raise FileNotFoundError(combined_path)
        payload = torch.load(combined_path, map_location="cpu")
        state_a = extract_branch_state_dict(
            payload, "appearance", allow_direct=False
        )
        state_s = extract_branch_state_dict(
            payload, "structure", allow_direct=False
        )
        path_a = path_s = combined_path
    else:
        path_a = args.appearance_checkpoint.expanduser().resolve()
        path_s = args.structure_checkpoint.expanduser().resolve()
        if not path_a.is_file():
            raise FileNotFoundError(path_a)
        if not path_s.is_file():
            raise FileNotFoundError(path_s)
        state_a = extract_branch_state_dict(
            torch.load(path_a, map_location="cpu"), "appearance"
        )
        state_s = extract_branch_state_dict(
            torch.load(path_s, map_location="cpu"), "structure"
        )

    model_a = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_s = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    model_a.load_state_dict(state_a, strict=True)
    model_s.load_state_dict(state_s, strict=True)
    model_a.to(device)
    model_s.to(device)
    model_a.eval()
    model_s.eval()
    return model_a, model_s, path_a, path_s


def build_validation_loader(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[DataLoader, Path, str]:
    canonical_split = "validation" if args.split in ("val", "validation") else "test"
    manifest = args.manifest or (
        args.dataset_root
        / "manifests"
        / "paper_seed_2026"
        / "{}.csv".format(canonical_split)
    )
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg["MEAN"], cfg["STD"])
    dual_view_cfg = copy.deepcopy(cfg["dual_view"])
    dual_view_cfg["enabled"] = True
    dataset = imagefloder_itn(
        manifest_path=str(manifest),
        data_transform_1=transforms[
            "val" if canonical_split == "validation" else "test"
        ],
        data_normalize_1=normalize,
        sup=True,
        dual_view_config=dual_view_cfg,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, manifest, canonical_split


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    elif isinstance(output, Mapping):
        logits = next(
            (output[key] for key in ("logits", "out", "prediction", "output") if key in output),
            None,
        )
    else:
        logits = None
    if not torch.is_tensor(logits) or logits.ndim != 4:
        raise TypeError("could not extract [N,C,H,W] logits from model output")
    return logits


def get_predictions(
    model_a: torch.nn.Module,
    model_s: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    required = ("image_appearance", "image_structure", "mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError("validation batch is missing fields: {}".format(missing))
    x_rgb = batch["image_appearance"].to(device, non_blocking=True)
    x_structure = batch["image_structure"].to(device, non_blocking=True)
    gt = batch["mask"].to(device, non_blocking=True).long()
    if x_rgb.shape[-2:] != x_structure.shape[-2:]:
        raise AssertionError("Appearance and Structure inputs are not spatially aligned")

    logits_a = extract_logits(model_a(x_rgb))
    logits_s = extract_logits(model_s(x_structure))
    if logits_a.shape[-2:] != gt.shape[-2:]:
        logits_a = F.interpolate(
            logits_a, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    if logits_s.shape[-2:] != gt.shape[-2:]:
        logits_s = F.interpolate(
            logits_s, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    pred_a = logits_a.argmax(dim=1)
    pred_s = logits_s.argmax(dim=1)
    conf_a = torch.softmax(logits_a, dim=1).amax(dim=1)
    conf_s = torch.softmax(logits_s, dim=1).amax(dim=1)
    if pred_a.shape != gt.shape or pred_s.shape != gt.shape:
        raise AssertionError(
            "prediction/GT shape mismatch: A={}, S={}, GT={}".format(
                tuple(pred_a.shape), tuple(pred_s.shape), tuple(gt.shape)
            )
        )
    return pred_a, pred_s, gt, conf_a, conf_s


def compute_region_masks(
    gt: torch.Tensor,
    valid: torch.Tensor,
    boundary_width: int,
) -> dict[str, torch.Tensor]:
    foreground = (gt == 1) & valid
    background = (gt == 0) & valid
    fg_float = foreground.float().unsqueeze(1)
    radius = int(boundary_width)
    kernel_size = 2 * radius + 1
    # Explicit zero padding treats pixels outside the image as background.
    padded = F.pad(fg_float, (radius, radius, radius, radius), value=0.0)
    eroded = -F.max_pool2d(-padded, kernel_size=kernel_size, stride=1)
    eroded = eroded.squeeze(1) > 0.5
    boundary = foreground & ~eroded
    interior = foreground & ~boundary

    if torch.any(boundary & interior):
        raise AssertionError("boundary and interior overlap")
    if not torch.equal(boundary | interior, foreground):
        raise AssertionError("boundary and interior do not cover foreground")
    if torch.any(foreground & background):
        raise AssertionError("foreground and background overlap")
    if not torch.equal(foreground | background, valid):
        raise AssertionError("foreground and background do not cover valid pixels")
    return {
        "all_pixels": valid,
        "foreground": foreground,
        "background": background,
        "boundary": boundary,
        "interior": interior,
    }


def update_complementarity_counts(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    region: torch.Tensor,
) -> ComplementarityCounts:
    correct_a = pred_a == gt
    correct_s = pred_s == gt
    counts = ComplementarityCounts(
        both_correct=int((region & correct_a & correct_s).sum().item()),
        a_only_correct=int((region & correct_a & ~correct_s).sum().item()),
        s_only_correct=int((region & ~correct_a & correct_s).sum().item()),
        both_wrong=int((region & ~correct_a & ~correct_s).sum().item()),
    )
    expected = int(region.sum().item())
    if counts.total != expected:
        raise AssertionError(
            "complementarity partition {} != valid region {}".format(
                counts.total, expected
            )
        )
    return counts


def compute_complementarity_metrics(
    counts: ComplementarityCounts,
) -> dict[str, Any]:
    total = counts.total
    denominator = float(total) + EPS
    metrics = {
        "valid_pixels": total,
        "both_correct": counts.both_correct,
        "a_only_correct": counts.a_only_correct,
        "s_only_correct": counts.s_only_correct,
        "both_wrong": counts.both_wrong,
        "both_correct_ratio": counts.both_correct / denominator,
        "a_only_correct_ratio": counts.a_only_correct / denominator,
        "s_only_correct_ratio": counts.s_only_correct / denominator,
        "both_wrong_ratio": counts.both_wrong / denominator,
        "A_corrects_S": counts.a_only_correct
        / (counts.a_only_correct + counts.both_wrong + EPS),
        "S_corrects_A": counts.s_only_correct
        / (counts.s_only_correct + counts.both_wrong + EPS),
    }
    for key, value in metrics.items():
        if key.endswith("_ratio") or key in ("A_corrects_S", "S_corrects_A"):
            if not 0.0 <= value <= 1.0:
                raise AssertionError("{} is outside [0,1]: {}".format(key, value))
    return metrics


def disagreement_winner_counts(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    region: torch.Tensor,
) -> WinnerCounts:
    disagreement = region & (pred_a != pred_s)
    correct_a = pred_a == gt
    correct_s = pred_s == gt
    both_correct = int((disagreement & correct_a & correct_s).sum().item())
    both_wrong = int((disagreement & ~correct_a & ~correct_s).sum().item())
    if both_correct != 0 or both_wrong != 0:
        print(
            "WARNING: binary disagreement contains both-correct={} or "
            "both-wrong={} pixels; check labels.".format(both_correct, both_wrong)
        )
    counts = WinnerCounts(
        disagreement_pixels=int(disagreement.sum().item()),
        a_wins=int((disagreement & correct_a & ~correct_s).sum().item()),
        s_wins=int((disagreement & ~correct_a & correct_s).sum().item()),
    )
    if counts.a_wins + counts.s_wins != counts.disagreement_pixels:
        raise AssertionError("disagreement winners do not partition disagreement")
    return counts


def winner_metrics(counts: WinnerCounts) -> dict[str, Any]:
    denominator = counts.a_wins + counts.s_wins
    return {
        "disagreement_pixels": counts.disagreement_pixels,
        "A_wins": counts.a_wins,
        "S_wins": counts.s_wins,
        "A_win_rate": counts.a_wins / (denominator + EPS),
        "S_win_rate": counts.s_wins / (denominator + EPS),
    }


def prediction_pair_counts(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    appearance_class: int,
) -> PredictionPairCounts:
    structure_class = 1 - appearance_class
    case = valid & (pred_a == appearance_class) & (pred_s == structure_class)
    return PredictionPairCounts(
        count=int(case.sum().item()),
        gt_foreground=int((case & (gt == 1)).sum().item()),
        gt_background=int((case & (gt == 0)).sum().item()),
        appearance_correct=int((case & (pred_a == gt)).sum().item()),
        structure_correct=int((case & (pred_s == gt)).sum().item()),
    )


def prediction_pair_metrics(counts: PredictionPairCounts) -> dict[str, Any]:
    return {
        "count": counts.count,
        "gt_foreground_count": counts.gt_foreground,
        "gt_background_count": counts.gt_background,
        "appearance_correct_count": counts.appearance_correct,
        "structure_correct_count": counts.structure_correct,
        "gt_foreground_ratio": counts.gt_foreground / (counts.count + EPS),
        "gt_background_ratio": counts.gt_background / (counts.count + EPS),
        "appearance_accuracy": counts.appearance_correct / (counts.count + EPS),
        "structure_accuracy": counts.structure_correct / (counts.count + EPS),
    }


def confidence_counts(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    conf_a: torch.Tensor,
    conf_s: torch.Tensor,
    region: torch.Tensor,
    margin: float,
) -> ConfidenceCounts:
    disagreement = region & (pred_a != pred_s)
    delta = (conf_a - conf_s).abs()
    selected = disagreement & (delta >= margin)
    # The selection rule uses predictions/confidences only. GT is consulted
    # below solely to evaluate whether that deployable rule selected correctly.
    choose_a = conf_a > conf_s
    selected_prediction = torch.where(choose_a, pred_a, pred_s)
    return ConfidenceCounts(
        total_disagreement_pixels=int(disagreement.sum().item()),
        selected_pixels=int(selected.sum().item()),
        correct_selections=int((selected & (selected_prediction == gt)).sum().item()),
    )


def confidence_metrics(counts: ConfidenceCounts) -> dict[str, Any]:
    return {
        "total_disagreement_pixels": counts.total_disagreement_pixels,
        "selected_pixels": counts.selected_pixels,
        "coverage": counts.selected_pixels
        / (counts.total_disagreement_pixels + EPS),
        "selection_accuracy": counts.correct_selections
        / (counts.selected_pixels + EPS),
    }


def confidence_bin_counts(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    conf_a: torch.Tensor,
    conf_s: torch.Tensor,
    valid: torch.Tensor,
    lower: float,
    upper: float,
) -> ConfidenceBinCounts:
    disagreement = valid & (pred_a != pred_s)
    delta = (conf_a - conf_s).abs()
    selected = disagreement & (delta >= lower) & (delta < upper)
    choose_a = conf_a > conf_s
    high_prediction = torch.where(choose_a, pred_a, pred_s)
    return ConfidenceBinCounts(
        count=int(selected.sum().item()),
        higher_confidence_correct=int(
            (selected & (high_prediction == gt)).sum().item()
        ),
        a_wins=int((selected & (pred_a == gt) & (pred_s != gt)).sum().item()),
        s_wins=int((selected & (pred_s == gt) & (pred_a != gt)).sum().item()),
    )


def binary_counts(
    prediction: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> BinaryCounts:
    pred_fg = prediction == 1
    gt_fg = gt == 1
    return BinaryCounts(
        tp=int((valid & pred_fg & gt_fg).sum().item()),
        tn=int((valid & ~pred_fg & ~gt_fg).sum().item()),
        fp=int((valid & pred_fg & ~gt_fg).sum().item()),
        fn=int((valid & ~pred_fg & gt_fg).sum().item()),
    )


def compute_binary_dice(counts: BinaryCounts) -> float:
    return (2.0 * counts.tp) / (2.0 * counts.tp + counts.fp + counts.fn + EPS)


def compute_binary_iou(counts: BinaryCounts) -> float:
    return counts.tp / (counts.tp + counts.fp + counts.fn + EPS)


def pixel_accuracy(counts: BinaryCounts) -> float:
    return (counts.tp + counts.tn) / (counts.total + EPS)


def compute_class_metrics(counts: BinaryCounts) -> dict[str, Any]:
    return {
        "tp": counts.tp,
        "fn": counts.fn,
        "tn": counts.tn,
        "fp": counts.fp,
        "precision": counts.tp / (counts.tp + counts.fp + EPS),
        "recall": counts.tp / (counts.tp + counts.fn + EPS),
        "sensitivity": counts.tp / (counts.tp + counts.fn + EPS),
        "specificity": counts.tn / (counts.tn + counts.fp + EPS),
        "fpr": counts.fp / (counts.tn + counts.fp + EPS),
        "fnr": counts.fn / (counts.tp + counts.fn + EPS),
    }


def build_oracle_prediction(
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Build the GT-selected, analysis-only and non-deployable upper bound."""
    oracle = pred_s.clone()
    use_a = valid & (pred_s != gt) & (pred_a == gt)
    oracle[use_a] = pred_a[use_a]
    return oracle


def _safe_name(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "sample"


def build_relation_map(
    gt: torch.Tensor,
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    valid: torch.Tensor,
) -> np.ndarray:
    gt_np = gt.cpu().numpy()
    pred_a_np = pred_a.cpu().numpy()
    pred_s_np = pred_s.cpu().numpy()
    valid_np = valid.cpu().numpy().astype(bool)
    correct_a = pred_a_np == gt_np
    correct_s = pred_s_np == gt_np
    relation = np.full(gt_np.shape, 4, dtype=np.uint8)
    relation[valid_np & correct_a & correct_s] = 0
    relation[valid_np & correct_a & ~correct_s] = 1
    relation[valid_np & ~correct_a & correct_s] = 2
    relation[valid_np & ~correct_a & ~correct_s] = 3
    return relation


def denormalize_rgb(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype).view(3, 1, 1)
    rgb = (image.cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0).numpy()


def save_visualization(
    output_dir: Path,
    image_id: str,
    gt: torch.Tensor,
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    sample_name = _safe_name(image_id)
    gt_np = gt.cpu().numpy().astype(np.uint8)
    pred_a_np = pred_a.cpu().numpy().astype(np.uint8)
    pred_s_np = pred_s.cpu().numpy().astype(np.uint8)
    # Indexed mask values: 0=both correct, 1=A-only, 2=S-only,
    # 3=both wrong, 4=ignored. A palette makes the categories easy to inspect
    # without changing the stored class indices.
    relation = build_relation_map(gt, pred_a, pred_s, valid)
    Image.fromarray(gt_np * 255, mode="L").save(output_dir / f"{sample_name}_gt.png")
    Image.fromarray(pred_a_np * 255, mode="L").save(
        output_dir / f"{sample_name}_pred_appearance.png"
    )
    Image.fromarray(pred_s_np * 255, mode="L").save(
        output_dir / f"{sample_name}_pred_structure.png"
    )
    relation_image = Image.fromarray(relation, mode="P")
    palette = [60, 180, 75, 255, 165, 0, 0, 200, 220, 230, 50, 50, 0, 0, 0]
    relation_image.putpalette(palette + [0] * (768 - len(palette)))
    relation_image.save(output_dir / f"{sample_name}_complementarity_map.png")


def _per_image_row(
    image_id: str,
    pred_a: torch.Tensor,
    pred_s: torch.Tensor,
    oracle: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    conf_a: torch.Tensor,
    conf_s: torch.Tensor,
) -> dict[str, Any]:
    counts_a = binary_counts(pred_a, gt, valid)
    counts_s = binary_counts(pred_s, gt, valid)
    counts_o = binary_counts(oracle, gt, valid)
    row: dict[str, Any] = {
        "image_id": image_id,
        "appearance_dice": compute_binary_dice(counts_a),
        "appearance_iou": compute_binary_iou(counts_a),
        "structure_dice": compute_binary_dice(counts_s),
        "structure_iou": compute_binary_iou(counts_s),
        "oracle_dice": compute_binary_dice(counts_o),
        "oracle_iou": compute_binary_iou(counts_o),
        "oracle_gain_over_structure": (
            compute_binary_dice(counts_o) - compute_binary_dice(counts_s)
        ),
        "oracle_gain": compute_binary_dice(counts_o) - compute_binary_dice(counts_s),
        "structure_minus_appearance_dice": (
            compute_binary_dice(counts_s) - compute_binary_dice(counts_a)
        ),
    }
    for region_name, region in regions.items():
        metrics = compute_complementarity_metrics(
            update_complementarity_counts(pred_a, pred_s, gt, region)
        )
        for key, value in metrics.items():
            row["{}_{}".format(region_name, key)] = value
        winners = winner_metrics(
            disagreement_winner_counts(pred_a, pred_s, gt, region)
        )
        row["A_win_rate_{}".format(region_name)] = winners["A_win_rate"]
        row["S_win_rate_{}".format(region_name)] = winners["S_win_rate"]
        row["{}_disagreement_pixels".format(region_name)] = winners[
            "disagreement_pixels"
        ]
    all_valid = regions["all_pixels"]
    all_confidence = confidence_metrics(
        confidence_counts(pred_a, pred_s, gt, conf_a, conf_s, all_valid, 0.0)
    )
    row["disagreement_ratio"] = row["all_pixels_disagreement_pixels"] / (
        row["all_pixels_valid_pixels"] + EPS
    )
    row["A_win_rate_all"] = row["A_win_rate_all_pixels"]
    row["S_win_rate_all"] = row["S_win_rate_all_pixels"]
    row["confidence_selection_accuracy"] = all_confidence[
        "selection_accuracy"
    ]
    return row


def bootstrap_confidence_intervals(
    per_image_rows: Sequence[Mapping[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    metric_columns = {
        "appearance_dice": "appearance_dice",
        "structure_dice": "structure_dice",
        "structure_minus_appearance_dice": "structure_minus_appearance_dice",
        "oracle_dice": "oracle_dice",
        "oracle_gain": "oracle_gain",
        "A_corrects_S_errors": "all_pixels_A_corrects_S",
        "S_corrects_A_errors": "all_pixels_S_corrects_A",
        "A_disagreement_win_rate": "A_win_rate_all_pixels",
        "S_disagreement_win_rate": "S_win_rate_all_pixels",
        "foreground_A_win_rate": "A_win_rate_foreground",
        "foreground_S_win_rate": "S_win_rate_foreground",
        "background_A_win_rate": "A_win_rate_background",
        "background_S_win_rate": "S_win_rate_background",
        "boundary_A_win_rate": "A_win_rate_boundary",
        "boundary_S_win_rate": "S_win_rate_boundary",
    }
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {
        "metadata": {
            "sampling_unit": "image",
            "iterations": iterations,
            "seed": seed,
            "confidence_level": 0.95,
        }
    }
    for output_name, column in metric_columns.items():
        values = np.asarray([float(row[column]) for row in per_image_rows])
        values = values[np.isfinite(values)]
        if values.size == 0:
            output[output_name] = {
                "mean": None,
                "ci95_low": None,
                "ci95_high": None,
                "images": 0,
            }
            continue
        sampled_indices = rng.integers(
            0, values.size, size=(iterations, values.size)
        )
        bootstrap_means = values[sampled_indices].mean(axis=1)
        low, high = np.percentile(bootstrap_means, (2.5, 97.5))
        output[output_name] = {
            "mean": float(values.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "images": int(values.size),
        }
    return output


def select_qualitative_samples(
    per_image_rows: Sequence[Mapping[str, Any]],
    count: int,
) -> list[str]:
    selectors = (
        "oracle_gain",
        "all_pixels_a_only_correct_ratio",
        "all_pixels_s_only_correct_ratio",
        "boundary_disagreement_pixels",
    )
    selected: list[str] = []
    for metric in selectors:
        candidate = max(per_image_rows, key=lambda row: float(row[metric]))
        image_id = str(candidate["image_id"])
        if image_id not in selected:
            selected.append(image_id)
    remaining = sorted(
        per_image_rows,
        key=lambda row: (
            float(row["oracle_gain"]),
            float(row["disagreement_ratio"]),
        ),
        reverse=True,
    )
    for row in remaining:
        image_id = str(row["image_id"])
        if image_id not in selected:
            selected.append(image_id)
        if len(selected) >= count:
            break
    return selected[:count]


def regression_check(
    reference: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not reference:
        return {"status": "not_run", "reason": "reference summary not found"}
    reference_meta = reference.get("metadata", {})
    current_meta = summary.get("metadata", {})
    same_inputs = all(
        str(reference_meta.get(key)) == str(current_meta.get(key))
        for key in (
            "manifest",
            "appearance_checkpoint",
            "structure_checkpoint",
            "boundary_width",
        )
    )
    if not same_inputs:
        return {
            "status": "not_run",
            "reason": "reference uses different manifest/checkpoints/settings",
        }
    comparisons = {
        "appearance_dice": (reference["appearance"]["dice"], summary["appearance"]["dice"]),
        "structure_dice": (reference["structure"]["dice"], summary["structure"]["dice"]),
        "oracle_dice": (reference["oracle"]["dice"], summary["oracle"]["dice"]),
        "oracle_gain": (
            reference["oracle"]["gain_over_structure"],
            summary["oracle"]["gain_over_structure"],
        ),
        "a_only_correct": (
            reference["all_pixels"]["a_only_correct"],
            summary["all_pixels"]["a_only_correct"],
        ),
        "s_only_correct": (
            reference["all_pixels"]["s_only_correct"],
            summary["all_pixels"]["s_only_correct"],
        ),
    }
    differences = {
        key: abs(float(expected) - float(actual))
        for key, (expected, actual) in comparisons.items()
    }
    passed = all(
        difference <= (0.0005 if "correct" not in key else 0.0)
        for key, difference in differences.items()
    )
    return {
        "status": "passed" if passed else "failed",
        "differences": differences,
    }


def save_summary(
    output_dir: Path,
    summary: Mapping[str, Any],
    per_image_rows: list[dict[str, Any]],
) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV: {}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Mapping[str, Any]) -> None:
    print("=" * 50)
    print("Complementarity Analysis")
    print("=" * 50)
    print("Appearance Dice : {:.4f}".format(summary["appearance"]["dice"]))
    print("Structure Dice  : {:.4f}".format(summary["structure"]["dice"]))
    print("Oracle Dice     : {:.4f}".format(summary["oracle"]["dice"]))
    print("Oracle Gain     : {:+.4f}".format(summary["oracle"]["gain_over_structure"]))
    for region_name in REGION_NAMES:
        values = summary[region_name]
        print("\n" + "-" * 50)
        print(region_name.replace("_", " ").title())
        print("-" * 50)
        print("Both Correct         : {:6.2f} %".format(100 * values["both_correct_ratio"]))
        print("A Correct / S Wrong  : {:6.2f} %".format(100 * values["a_only_correct_ratio"]))
        print("S Correct / A Wrong  : {:6.2f} %".format(100 * values["s_only_correct_ratio"]))
        print("Both Wrong           : {:6.2f} %".format(100 * values["both_wrong_ratio"]))
        print("\nA corrects S errors  : {:6.2f} %".format(100 * values["A_corrects_S"]))
        print("S corrects A errors  : {:6.2f} %".format(100 * values["S_corrects_A"]))
    winners = summary["disagreement_winner"]
    confidence = summary["confidence_selection"]
    best_margin = summary["best_confidence_margin"]
    print("\n" + "=" * 50)
    print("Descriptive reliability findings")
    print("=" * 50)
    print(
        "Overall disagreements: Appearance wins {:.2f} %, Structure wins {:.2f} %.".format(
            100 * winners["all_pixels"]["A_win_rate"],
            100 * winners["all_pixels"]["S_win_rate"],
        )
    )
    for region in ("foreground", "background", "boundary"):
        print(
            "{} disagreements: Appearance wins {:.2f} %, Structure wins {:.2f} %.".format(
                region.title(),
                100 * winners[region]["A_win_rate"],
                100 * winners[region]["S_win_rate"],
            )
        )
    print(
        "Higher-confidence selection accuracy: {:.2f} % at {:.2f} % coverage.".format(
            100 * confidence["all_pixels"]["selection_accuracy"],
            100 * confidence["all_pixels"]["coverage"],
        )
    )
    print(
        "Best observed margin {:.2f}: {:.2f} % accuracy at {:.2f} % coverage.".format(
            best_margin["margin"],
            100 * best_margin["selection_accuracy"],
            100 * best_margin["coverage"],
        )
    )
    oracle_ci = summary["bootstrap_ci"]["oracle_gain"]
    print(
        "Oracle gain image-level mean: {:+.4f} (95% CI {:+.4f}, {:+.4f}).".format(
            oracle_ci["mean"], oracle_ci["ci95_low"], oracle_ci["ci95_high"]
        )
    )
    print("\n" + ORACLE_NOTICE)
    print("=" * 50)


def _prepare_output_dir(requested: Path) -> Path:
    project_root = Path(__file__).resolve().parent
    allowed_root = (project_root / "complementarity_results").resolve()
    output_dir = requested.expanduser().resolve()
    if output_dir == allowed_root or allowed_root not in output_dir.parents:
        raise ValueError(
            "--output-dir must be a new subdirectory inside complementarity_results/"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    device = resolve_device(args.gpu)
    cfg = dataset_cfg(args.dataset_name)
    if cfg["NUM_CLASSES"] != 2:
        raise ValueError("this analysis currently requires binary segmentation")
    loader, manifest, canonical_split = build_validation_loader(args, cfg, device)
    model_a, model_s, path_a, path_s = load_models(args, cfg, device)
    output_dir = _prepare_output_dir(args.output_dir)
    reference = None
    reference_path = args.reference_summary.expanduser().resolve()
    if reference_path.is_file():
        with reference_path.open(encoding="utf-8") as handle:
            reference = json.load(handle)

    print("Device                : {}".format(device))
    print("Dataset split         : {}".format(canonical_split))
    print("Dataset size          : {}".format(len(loader.dataset)))
    print("Appearance checkpoint : {}".format(path_a))
    print("Structure checkpoint  : {}".format(path_s))
    print("Manifest              : {}".format(manifest))
    print("Output directory      : {}".format(output_dir))
    print(ORACLE_NOTICE)

    aggregate_regions = {
        name: ComplementarityCounts() for name in REGION_NAMES
    }
    aggregate_winners = {name: WinnerCounts() for name in REGION_NAMES}
    aggregate_confidence = {
        name: {margin: ConfidenceCounts() for margin in CONFIDENCE_MARGINS}
        for name in REGION_NAMES
    }
    aggregate_bins = {
        bounds: ConfidenceBinCounts() for bounds in CONFIDENCE_BINS
    }
    aggregate_pairs = {
        "A1_S0": PredictionPairCounts(),
        "A0_S1": PredictionPairCounts(),
    }
    aggregate_a = BinaryCounts()
    aggregate_s = BinaryCounts()
    aggregate_oracle = BinaryCounts()
    per_image_rows: list[dict[str, Any]] = []
    image_records: dict[str, dict[str, np.ndarray]] = {}
    visualizations_saved = 0
    visualization_dir = output_dir / "visualizations"
    if args.save_vis:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    model_a.eval()
    model_s.eval()
    with torch.no_grad():
        for batch in loader:
            pred_a, pred_s, gt, conf_a, conf_s = get_predictions(
                model_a, model_s, batch, device
            )
            valid = torch.ones_like(gt, dtype=torch.bool)
            if args.ignore_index is not None:
                valid &= gt != args.ignore_index
            regions = compute_region_masks(gt, valid, args.boundary_width)
            oracle = build_oracle_prediction(pred_a, pred_s, gt, valid)

            aggregate_a.add(binary_counts(pred_a, gt, valid))
            aggregate_s.add(binary_counts(pred_s, gt, valid))
            aggregate_oracle.add(binary_counts(oracle, gt, valid))
            for region_name, region in regions.items():
                aggregate_regions[region_name].add(
                    update_complementarity_counts(pred_a, pred_s, gt, region)
                )
                aggregate_winners[region_name].add(
                    disagreement_winner_counts(pred_a, pred_s, gt, region)
                )
                for margin in CONFIDENCE_MARGINS:
                    aggregate_confidence[region_name][margin].add(
                        confidence_counts(
                            pred_a,
                            pred_s,
                            gt,
                            conf_a,
                            conf_s,
                            region,
                            margin,
                        )
                    )
            aggregate_pairs["A1_S0"].add(
                prediction_pair_counts(pred_a, pred_s, gt, valid, 1)
            )
            aggregate_pairs["A0_S1"].add(
                prediction_pair_counts(pred_a, pred_s, gt, valid, 0)
            )
            for bounds in CONFIDENCE_BINS:
                aggregate_bins[bounds].add(
                    confidence_bin_counts(
                        pred_a,
                        pred_s,
                        gt,
                        conf_a,
                        conf_s,
                        valid,
                        *bounds,
                    )
                )

            image_ids = list(batch["ID"])
            for index, image_id in enumerate(image_ids):
                image_regions = {
                    name: mask[index] for name, mask in regions.items()
                }
                per_image_rows.append(
                    _per_image_row(
                        str(image_id),
                        pred_a[index],
                        pred_s[index],
                        oracle[index],
                        gt[index],
                        valid[index],
                        image_regions,
                        conf_a[index],
                        conf_s[index],
                    )
                )
                image_records[str(image_id)] = {
                    "rgb": denormalize_rgb(
                        batch["image_appearance"][index], cfg["MEAN"], cfg["STD"]
                    ),
                    "gt": gt[index].cpu().numpy().astype(np.uint8),
                    "pred_a": pred_a[index].cpu().numpy().astype(np.uint8),
                    "pred_s": pred_s[index].cpu().numpy().astype(np.uint8),
                    "relation": build_relation_map(
                        gt[index], pred_a[index], pred_s[index], valid[index]
                    ),
                }
                if args.save_vis and visualizations_saved < args.max_vis:
                    save_visualization(
                        visualization_dir,
                        str(image_id),
                        gt[index],
                        pred_a[index],
                        pred_s[index],
                        valid[index],
                    )
                    visualizations_saved += 1

    if not per_image_rows:
        raise RuntimeError("the selected dataset contains no images")

    dice_a = compute_binary_dice(aggregate_a)
    dice_s = compute_binary_dice(aggregate_s)
    dice_oracle = compute_binary_dice(aggregate_oracle)
    accuracy_s = pixel_accuracy(aggregate_s)
    accuracy_oracle = pixel_accuracy(aggregate_oracle)
    if accuracy_oracle + EPS < accuracy_s:
        raise AssertionError("Oracle pixel accuracy is below Structure accuracy")
    if dice_oracle + 1e-6 < dice_s:
        print("WARNING: Oracle Dice is unexpectedly below Structure Dice; check labels.")

    disagreement_summary = {
        name: winner_metrics(counts)
        for name, counts in aggregate_winners.items()
    }
    confidence_summary = {
        name: confidence_metrics(aggregate_confidence[name][0.0])
        for name in REGION_NAMES
    }
    confidence_curve_rows: list[dict[str, Any]] = []
    for region_name in REGION_NAMES:
        for margin in CONFIDENCE_MARGINS:
            values = confidence_metrics(
                aggregate_confidence[region_name][margin]
            )
            confidence_curve_rows.append(
                {
                    "region": region_name,
                    "margin": margin,
                    "selected_pixels": values["selected_pixels"],
                    "total_disagreement_pixels": values[
                        "total_disagreement_pixels"
                    ],
                    "coverage": values["coverage"],
                    "selection_accuracy": values["selection_accuracy"],
                }
            )
    all_curve = [
        row
        for row in confidence_curve_rows
        if row["region"] == "all_pixels" and row["selected_pixels"] > 0
    ]
    best_margin = max(
        all_curve,
        key=lambda row: (row["selection_accuracy"], row["coverage"]),
    )

    total_disagreements = aggregate_winners["all_pixels"].disagreement_pixels
    if sum(counts.count for counts in aggregate_pairs.values()) != total_disagreements:
        raise AssertionError("prediction-pair cases do not cover disagreement")
    if sum(counts.count for counts in aggregate_bins.values()) != total_disagreements:
        raise AssertionError("confidence bins do not cover disagreement")
    for region_name in REGION_NAMES:
        expected_disagreement = (
            aggregate_regions[region_name].a_only_correct
            + aggregate_regions[region_name].s_only_correct
        )
        if aggregate_winners[region_name].disagreement_pixels != expected_disagreement:
            raise AssertionError(
                "{} winner counts do not match complementarity counts".format(
                    region_name
                )
            )
        selected_counts = [
            aggregate_confidence[region_name][margin].selected_pixels
            for margin in CONFIDENCE_MARGINS
        ]
        if any(
            later > earlier
            for earlier, later in zip(selected_counts, selected_counts[1:])
        ):
            raise AssertionError("confidence-margin coverage is not monotonic")
    confidence_bin_rows: list[dict[str, Any]] = []
    for lower, upper in CONFIDENCE_BINS:
        counts = aggregate_bins[(lower, upper)]
        confidence_bin_rows.append(
            {
                "bin_lower": lower,
                "bin_upper": min(upper, 1.0),
                "bin_label": (
                    "[{:.2f}, {:.2f}]".format(lower, 1.0)
                    if upper > 1.0
                    else "[{:.2f}, {:.2f})".format(lower, upper)
                ),
                "pixel_count": counts.count,
                "ratio": counts.count / (total_disagreements + EPS),
                "higher_confidence_branch_accuracy": (
                    counts.higher_confidence_correct / (counts.count + EPS)
                ),
                "Appearance_win_rate": counts.a_wins / (counts.count + EPS),
                "Structure_win_rate": counts.s_wins / (counts.count + EPS),
            }
        )

    bootstrap_ci = bootstrap_confidence_intervals(
        per_image_rows,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )

    summary: dict[str, Any] = {
        "metadata": {
            "dataset": args.dataset_name,
            "split": canonical_split,
            "dataset_size": len(loader.dataset),
            "manifest": str(manifest),
            "appearance_checkpoint": str(path_a),
            "structure_checkpoint": str(path_s),
            "boundary_width": args.boundary_width,
            "ignore_index": args.ignore_index,
            "oracle_notice": ORACLE_NOTICE,
        },
        "appearance": {
            "dice": dice_a,
            "iou": compute_binary_iou(aggregate_a),
            "pixel_accuracy": pixel_accuracy(aggregate_a),
        },
        "structure": {
            "dice": dice_s,
            "iou": compute_binary_iou(aggregate_s),
            "pixel_accuracy": accuracy_s,
        },
        "oracle": {
            "dice": dice_oracle,
            "iou": compute_binary_iou(aggregate_oracle),
            "pixel_accuracy": accuracy_oracle,
            "gain_over_structure": dice_oracle - dice_s,
            "notice": ORACLE_NOTICE,
        },
        "appearance_class_metrics": compute_class_metrics(aggregate_a),
        "structure_class_metrics": compute_class_metrics(aggregate_s),
        "disagreement_winner": disagreement_summary,
        "prediction_pair_analysis": {
            name: prediction_pair_metrics(counts)
            for name, counts in aggregate_pairs.items()
        },
        "confidence_selection": confidence_summary,
        "best_confidence_margin": dict(best_margin),
        "bootstrap_ci": bootstrap_ci,
    }
    summary.update(
        {
            name: compute_complementarity_metrics(counts)
            for name, counts in aggregate_regions.items()
        }
    )
    summary["regression_check"] = regression_check(reference, summary)
    if summary["regression_check"]["status"] == "failed":
        print(
            "WARNING: upgraded basic statistics differ from the reference; "
            "figures will not be generated. Differences: {}".format(
                summary["regression_check"]["differences"]
            )
        )
    save_summary(output_dir, summary, per_image_rows)
    save_json(output_dir / "bootstrap_ci.json", bootstrap_ci)
    save_csv(output_dir / "confidence_margin_curve.csv", confidence_curve_rows)
    save_csv(output_dir / "confidence_bins.csv", confidence_bin_rows)

    if not args.no_figures and summary["regression_check"]["status"] != "failed":
        from analysis.complementarity_plots import (
            plot_class_error_profile,
            plot_confidence_coverage_accuracy,
            plot_disagreement_winner,
            plot_per_image_metrics,
            plot_qualitative_panel,
        )

        figures_dir = output_dir / "figures"
        plot_disagreement_winner(disagreement_summary, figures_dir)
        plot_confidence_coverage_accuracy(confidence_curve_rows, figures_dir)
        plot_per_image_metrics(per_image_rows, figures_dir)
        plot_class_error_profile(
            summary["appearance_class_metrics"],
            summary["structure_class_metrics"],
            figures_dir,
        )
        selected_ids = select_qualitative_samples(
            per_image_rows, args.qualitative_samples
        )
        summary["metadata"]["qualitative_image_ids"] = selected_ids
        plot_qualitative_panel(selected_ids, image_records, figures_dir)
        # Persist the selected image IDs added after figure generation.
        save_summary(output_dir, summary, per_image_rows)
    print_summary(summary)


if __name__ == "__main__":
    main()
