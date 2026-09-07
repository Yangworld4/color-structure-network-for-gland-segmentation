"""Closed E0: fixed radial frequency-band sensitivity diagnostic for GLaS."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from models.getnetwork import get_network  # noqa: E402
from test import (  # noqa: E402
    extract_logits,
    load_checkpoint,
    metrics_from_counts,
    resolve_input_view,
    segmentation_counts,
)

from frequency_filter import BANDS, attenuate_radial_band, check_zero_attenuation  # noqa: E402
from gland_regions import make_region_masks  # noqa: E402
from structural_metrics import merge_split_counts, regional_error  # noqa: E402


CSV_COLUMNS = [
    "image_id", "band_id", "freq_low", "freq_high", "attenuation",
    "dice_base", "dice_band", "delta_dice", "iou_base", "iou_band", "delta_iou",
    "sens_interior", "sens_boundary", "sens_separator",
    "err_interior_base", "err_interior_band", "delta_err_interior",
    "err_boundary_base", "err_boundary_band", "delta_err_boundary",
    "err_separator_base", "err_separator_band", "delta_err_separator",
    "merge_count_base", "merge_count_band", "delta_merge_count",
    "split_count_base", "split_count_band", "delta_split_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="./outputs/frequency_e0")
    parser.add_argument("--attenuation", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="test")
    parser.add_argument("--network", default="unet")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args()


def _identity_normalize() -> A.Compose:
    # Existing resize is retained; this identity normalization exposes the
    # resized RGB tensor before the project's GLaS normalization.
    return A.Compose([A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2()])


def _evaluation_transform() -> A.Compose:
    """The project's val/test transform, isolated from its train-only A.Flip."""
    return A.Compose(
        [A.Resize(256, 256, p=1)],
        additional_targets={"image2": "image", "mask2": "mask"},
    )


def _model_normalize(image: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_tensor = torch.as_tensor(list(mean), dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_tensor = torch.as_tensor(list(std), dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean_tensor) / std_tensor


def _read_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")
    for required in ("image", "mask"):
        if required not in rows[0]:
            raise ValueError(f"manifest lacks {required}: {manifest_path}")
    return rows


def _load_instance(raw_path: Path, transform: A.Compose) -> np.ndarray:
    with Image.open(raw_path) as handle:
        instance = np.asarray(handle.convert("L"))
    # The image payload is only a shape anchor; the evaluation transform is
    # deterministic resize and does not inspect image values.
    anchor = np.zeros((*instance.shape, 3), dtype=np.uint8)
    transformed = transform(image=anchor, mask=instance)
    return transformed["mask"].cpu().numpy() if torch.is_tensor(transformed["mask"]) else np.asarray(transformed["mask"])


def _scalar(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.10g}"


def _save_gray(array: np.ndarray, path: Path, clip: bool = True) -> None:
    value = np.asarray(array)
    if value.dtype == bool:
        value = value.astype(np.uint8) * 255
    else:
        if clip:
            value = np.clip(value, 0.0, 1.0)
        value = (value * 255.0).round().astype(np.uint8) if value.max(initial=0) <= 1.0 else value.astype(np.uint8)
    Image.fromarray(value).save(path)


def _save_rgb(array: np.ndarray, path: Path) -> None:
    value = np.clip(np.asarray(array), 0.0, 1.0)
    Image.fromarray((value * 255.0).round().astype(np.uint8)).save(path)


def _save_visualization(
    directory: Path,
    raw_rgb: np.ndarray,
    gt: np.ndarray,
    pred_base: np.ndarray,
    regions: Dict[str, np.ndarray],
    bands: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _save_rgb(raw_rgb.transpose(1, 2, 0), directory / "original.png")
    _save_gray(gt, directory / "gt.png")
    _save_gray(pred_base, directory / "pred_base.png")
    _save_gray(regions["interior"], directory / "mask_interior.png")
    _save_gray(regions["boundary"], directory / "mask_boundary.png")
    _save_gray(regions["separator"], directory / "mask_separator.png")
    for index, (band_image, prediction) in enumerate(bands, start=1):
        _save_rgb(band_image.transpose(1, 2, 0), directory / f"band_B{index}.png")
        _save_gray(prediction, directory / f"pred_B{index}.png")
        _save_gray(prediction != gt, directory / f"error_B{index}.png")


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).any():
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def _summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for band_id, (low, high) in enumerate(BANDS, start=1):
        selected = [row for row in rows if row["band_id"] == f"B{band_id}"]
        output: Dict[str, object] = {"band_id": f"B{band_id}", "freq_low": low, "freq_high": high}
        fields = [
            "delta_dice", "delta_iou", "sens_interior", "sens_boundary", "sens_separator",
            "delta_err_interior", "delta_err_boundary", "delta_err_separator",
            "delta_merge_count", "delta_split_count",
        ]
        for field in fields:
            mean, std = _mean_std(float(row[field]) for row in selected)
            output[f"{field}_mean"] = mean
            output[f"{field}_std"] = std
        output["structure_selectivity"] = (
            (output["delta_err_boundary_mean"] + output["delta_err_separator_mean"]) / 2.0
            - output["delta_err_interior_mean"]
        )
        output["region_selectivity"] = (
            output["delta_err_interior_mean"]
            - (output["delta_err_boundary_mean"] + output["delta_err_separator_mean"]) / 2.0
        )
        summary.append(output)
    return summary


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _scalar(row[column]) if isinstance(row.get(column), float) else row.get(column, "") for column in columns})


def _plot_summary(summary: List[Dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"B{i} [{low:g},{high:g}]" for i, (low, high) in enumerate(BANDS, start=1)]
    x = np.arange(1, len(BANDS) + 1)
    specs = [
        ("prediction_sensitivity.png", "Frequency band vs prediction sensitivity", ["sens_interior_mean", "sens_boundary_mean", "sens_separator_mean"], ["interior", "boundary", "separator"], None),
        ("delta_regional_error.png", "Frequency band vs delta regional GT error", ["delta_err_interior_mean", "delta_err_boundary_mean", "delta_err_separator_mean"], ["delta_err_interior", "delta_err_boundary", "delta_err_separator"], None),
        ("structural_errors.png", "Frequency band vs structural errors", ["delta_merge_count_mean", "delta_split_count_mean"], ["delta_merge_count", "delta_split_count"], None),
        ("delta_dice.png", "Frequency band vs delta Dice", ["delta_dice_mean"], ["delta Dice"], None),
    ]
    for filename, title, fields, names, _ in specs:
        plt.figure(figsize=(10, 5.5))
        for field, name in zip(fields, names):
            plt.plot(x, [row[field] for row in summary], marker="o", linewidth=2, label=name)
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.xlabel("Radial frequency band")
        plt.ylabel("Metric")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=160)
        plt.close()


def _print_table(summary: List[Dict[str, object]]) -> None:
    print("\nBand | ΔDice | Sens_Int | Sens_Bd | Sens_Sep | ΔErr_Int | ΔErr_Bd | ΔErr_Sep | ΔMerge | ΔSplit")
    print("-----|-------|----------|---------|----------|----------|---------|----------|--------|-------")
    for row in summary:
        print(
            f"{row['band_id']} | {_scalar(row['delta_dice_mean'])} | {_scalar(row['sens_interior_mean'])} | "
            f"{_scalar(row['sens_boundary_mean'])} | {_scalar(row['sens_separator_mean'])} | "
            f"{_scalar(row['delta_err_interior_mean'])} | {_scalar(row['delta_err_boundary_mean'])} | "
            f"{_scalar(row['delta_err_separator_mean'])} | {_scalar(row['delta_merge_count_mean'])} | "
            f"{_scalar(row['delta_split_count_mean'])}"
        )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.attenuation <= 1.0:
        raise ValueError("--attenuation must be in [0, 1]")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    input_view = resolve_input_view("auto", checkpoint_path)
    input_mode = "rgb_rgb"
    cfg = dataset_cfg("GlaS")
    model = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
    load_checkpoint(model, checkpoint_path)
    model.to(device).eval()
    print(f"Successfully loaded checkpoint: {checkpoint_path}")
    print(f"CPS branch used: Model 2 / {input_view} (checkpoint best_result2; input_mode={input_mode}, RGB input)")

    evaluation_transform = _evaluation_transform()
    identity_normalize = _identity_normalize()
    mean, std = cfg["MEAN"], cfg["STD"]
    selected_splits = ["validation", "test"] if args.split == "both" else [args.split]
    all_rows: List[Dict[str, object]] = []
    visualization_samples = []

    for split in selected_splits:
        manifest_path = data_root / "manifests" / "paper_seed_2026" / f"{split}.csv"
        rows = _read_rows(manifest_path)
        dataset = imagefloder_itn(
            manifest_path=str(manifest_path),
            data_transform_1=evaluation_transform,
            data_normalize_1=identity_normalize,
            sup=True,
            input_mode=input_mode,
            training=False,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        print(f"Evaluation split: {split}, images: {len(dataset)}")
        for index, (batch, row) in enumerate(zip(loader, rows)):
            image_id = str(batch["ID"][0])
            raw_rgb = batch["image"][0].float().to(device)  # [0,1], resized before FFT
            gt = batch["mask"][0].cpu().numpy().astype(bool)
            reference_rgb = (raw_rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            instance_path = manifest_path.parent / row["instance_mask"] if row.get("instance_mask") else None
            instance = _load_instance(instance_path, evaluation_transform) if instance_path else None
            regions = make_region_masks(gt, instance)
            print(
                f"GT region pixels ({split}/{image_id}): "
                f"interior={regions['interior'].sum()}, "
                f"boundary={regions['boundary'].sum()}, "
                f"separator={regions['separator'].sum()}"
            )
            if index == 0:
                print(f"Input tensor range before FFT [{raw_rgb.min().item():.6f}, {raw_rgb.max().item():.6f}]")
                sanity_error = check_zero_attenuation(raw_rgb)
                print(f"FFT attenuation=0 reconstruction max_abs_error: {sanity_error:.3e}")
                if sanity_error >= 1e-5:
                    raise RuntimeError(f"FFT reconstruction sanity check failed: {sanity_error}")

            normalized_base = _model_normalize(raw_rgb, mean, std).unsqueeze(0)
            with torch.inference_mode():
                base_prob = torch.softmax(extract_logits(model(normalized_base)), dim=1)[0, 1].cpu().numpy()
            pred_base = base_prob > 0.5
            base_global = metrics_from_counts(*segmentation_counts(pred_base, gt))
            base_err = {name: regional_error(pred_base, gt, regions[name]) for name in ("interior", "boundary", "separator")}
            base_objects = merge_split_counts(pred_base, regions["instance_labels"])
            band_visuals = []
            for band_id, band in enumerate(BANDS, start=1):
                perturbed = attenuate_radial_band(raw_rgb, band, args.attenuation)
                if index == 0:
                    print(f"Reconstructed B{band_id} image range after FFT [{perturbed.min().item():.6f}, {perturbed.max().item():.6f}]")
                normalized = _model_normalize(perturbed, mean, std).unsqueeze(0)
                with torch.inference_mode():
                    probability = torch.softmax(extract_logits(model(normalized)), dim=1)[0, 1].cpu().numpy()
                prediction = probability > 0.5
                global_metrics = metrics_from_counts(*segmentation_counts(prediction, gt))
                errors = {name: regional_error(prediction, gt, regions[name]) for name in ("interior", "boundary", "separator")}
                objects = merge_split_counts(prediction, regions["instance_labels"])
                delta = lambda value, baseline: float(value - baseline) if np.isfinite(value) and np.isfinite(baseline) else float("nan")
                record: Dict[str, object] = {
                    "image_id": image_id, "band_id": f"B{band_id}", "freq_low": band[0], "freq_high": band[1], "attenuation": args.attenuation,
                    "dice_base": base_global["dice"], "dice_band": global_metrics["dice"], "delta_dice": global_metrics["dice"] - base_global["dice"],
                    "iou_base": base_global["jaccard"], "iou_band": global_metrics["jaccard"], "delta_iou": global_metrics["jaccard"] - base_global["jaccard"],
                    "sens_interior": float(np.abs(probability - base_prob)[regions["interior"]].mean()) if regions["interior"].any() else float("nan"),
                    "sens_boundary": float(np.abs(probability - base_prob)[regions["boundary"]].mean()) if regions["boundary"].any() else float("nan"),
                    "sens_separator": float(np.abs(probability - base_prob)[regions["separator"]].mean()) if regions["separator"].any() else float("nan"),
                }
                for name, suffix in (("interior", "interior"), ("boundary", "boundary"), ("separator", "separator")):
                    record[f"err_{suffix}_base"] = base_err[name]
                    record[f"err_{suffix}_band"] = errors[name]
                    record[f"delta_err_{suffix}"] = delta(errors[name], base_err[name])
                record.update({
                    "merge_count_base": base_objects["merge_count"], "merge_count_band": objects["merge_count"],
                    "delta_merge_count": objects["merge_count"] - base_objects["merge_count"],
                    "split_count_base": base_objects["split_count"], "split_count_band": objects["split_count"],
                    "delta_split_count": objects["split_count"] - base_objects["split_count"],
                })
                all_rows.append(record)
                if len(visualization_samples) < args.sample_count and index not in [sample[0] for sample in visualization_samples]:
                    band_visuals.append((perturbed.detach().cpu().numpy(), prediction))
            if len(visualization_samples) < args.sample_count:
                visualization_samples.append((index, split, image_id, raw_rgb.detach().cpu().numpy(), gt, pred_base, regions, band_visuals))

    summary = _summarize(all_rows)
    _write_csv(output_dir / "e0_frequency_per_image.csv", all_rows, CSV_COLUMNS)
    summary_columns = ["band_id", "freq_low", "freq_high"] + [
        field for field in (
            "delta_dice", "delta_iou", "sens_interior", "sens_boundary", "sens_separator",
            "delta_err_interior", "delta_err_boundary", "delta_err_separator",
            "delta_merge_count", "delta_split_count",
        ) for field in (f"{field}_mean", f"{field}_std")
    ] + ["structure_selectivity", "region_selectivity"]
    _write_csv(output_dir / "e0_frequency_summary.csv", summary, summary_columns)
    _plot_summary(summary, output_dir / "figures")
    for _, split, image_id, raw_rgb, gt, pred_base, regions, band_visuals in visualization_samples:
        safe_id = Path(image_id).stem
        _save_visualization(output_dir / "visualizations" / safe_id, raw_rgb, gt, pred_base, regions, band_visuals)
    _print_table(summary)
    base_rows = [row for row in all_rows if row["band_id"] == "B1"]
    print(f"Original model Dice / IoU: {_scalar(np.mean([row['dice_base'] for row in base_rows]))} / {_scalar(np.mean([row['iou_base'] for row in base_rows]))}")
    print(f"Wrote per-image CSV: {output_dir / 'e0_frequency_per_image.csv'}")
    print(f"Wrote summary CSV: {output_dir / 'e0_frequency_summary.csv'}")
    print(f"Wrote figures: {output_dir / 'figures'}")
    print(f"Wrote visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
