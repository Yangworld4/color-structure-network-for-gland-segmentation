#!/usr/bin/env python
"""Save FDCL labeled/unlabeled dual-view debug panels without training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn


PANEL_SIZE = (256, 256)
LABEL_HEIGHT = 24


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def denormalize_appearance(
    tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]
) -> np.ndarray:
    mean_tensor = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
    return to_uint8_rgb(tensor.detach().cpu() * std_tensor + mean_tensor)


def display_channel(channel: torch.Tensor) -> np.ndarray:
    channel = channel.detach().cpu().float()
    if float(channel.min()) < 0.0 or float(channel.max()) > 1.0:
        low, high = torch.quantile(channel, torch.tensor([0.01, 0.99]))
        channel = (channel - low) / max(float(high - low), 1e-6)
    channel = channel.clamp(0.0, 1.0).numpy()
    gray = np.round(channel * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def make_panel(image: Image.Image, label: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_SIZE[0], PANEL_SIZE[1] + LABEL_HEIGHT), "white")
    resized = image.convert("RGB").resize(PANEL_SIZE, Image.Resampling.BILINEAR)
    panel.paste(resized, (0, LABEL_HEIGHT))
    ImageDraw.Draw(panel).text((5, 5), label, fill="black")
    return panel


def save_sample_panel(
    sample: dict,
    original_path: str,
    output_path: Path,
    mean: list[float],
    std: list[float],
) -> None:
    with Image.open(original_path) as original_file:
        original = original_file.convert("RGB").copy()
    structure = sample["image_structure"]
    panels = [
        make_panel(original, "original RGB"),
        make_panel(
            Image.fromarray(to_uint8_rgb(sample["image_geometric"]), mode="RGB"),
            "shared geometry RGB",
        ),
        make_panel(
            Image.fromarray(
                denormalize_appearance(sample["image_appearance"], mean, std),
                mode="RGB",
            ),
            "appearance RGB",
        ),
        make_panel(Image.fromarray(display_channel(structure[0])), "Hematoxylin"),
        make_panel(Image.fromarray(display_channel(structure[1])), "Gray"),
        make_panel(Image.fromarray(display_channel(structure[2])), "Sobel edge"),
        make_panel(
            Image.fromarray(
                np.stack(
                    [
                        display_channel(structure[index])[..., 0]
                        for index in range(3)
                    ],
                    axis=2,
                ),
                mode="RGB",
            ),
            "structure as RGB",
        ),
    ]
    if "mask" in sample:
        mask = (sample["mask"].detach().cpu().numpy().astype(np.uint8) * 255)
        panels.append(make_panel(Image.fromarray(mask, mode="L"), "mask"))

    columns = 4
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * PANEL_SIZE[0], rows * (PANEL_SIZE[1] + LABEL_HEIGHT)),
        (230, 230, 230),
    )
    for index, panel in enumerate(panels):
        sheet.paste(
            panel,
            (
                (index % columns) * PANEL_SIZE[0],
                (index // columns) * (PANEL_SIZE[1] + LABEL_HEIGHT),
            ),
        )
    sheet.save(output_path)

    channel_std = structure.float().std(dim=(1, 2))
    assert torch.isfinite(structure).all()
    assert torch.all(channel_std > 1e-6), channel_std
    print(
        "{} | structure min/max={:.4f}/{:.4f} channel_std={}".format(
            output_path,
            float(structure.min()),
            float(structure.max()),
            [round(value, 6) for value in channel_std.tolist()],
        )
    )


def build_dataset(
    manifest: Path,
    supervised: bool,
    config: dict,
    mean: list[float],
    std: list[float],
):
    transforms = data_transform_2d()
    return imagefloder_itn(
        manifest_path=str(manifest),
        data_transform_1=transforms["train"],
        data_normalize_1=data_normalize_2d(mean, std),
        sup=supervised,
        dual_view_config=config,
        training=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="GlaS", choices=["GlaS", "CRAG"])
    parser.add_argument("--path_dataset", default=None)
    parser.add_argument("--labeled_ratio", default=10, type=int, choices=[10, 20])
    parser.add_argument("--num_samples", default=2, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--output_dir", default="outputs/dual_view_debug")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cfg = dataset_cfg(args.dataset_name)
    dual_config = cfg["dual_view"]
    dual_config["enabled"] = True
    dual_config["debug_visualization"] = True
    root = Path(args.path_dataset or cfg["PATH_DATASET"])
    manifests = root / "manifests" / "paper_seed_2026"
    datasets = {
        "labeled": build_dataset(
            manifests / "labeled_{}.csv".format(args.labeled_ratio),
            True,
            dual_config,
            cfg["MEAN"],
            cfg["STD"],
        ),
        "unlabeled": build_dataset(
            manifests / "unlabeled_{}.csv".format(args.labeled_ratio),
            False,
            dual_config,
            cfg["MEAN"],
            cfg["STD"],
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for group, dataset in datasets.items():
        for index in range(min(args.num_samples, len(dataset))):
            sample = dataset[index]
            safe_name = Path(sample["ID"]).stem.replace(" ", "_")
            output_path = output_dir / "{}_{}_{:03d}.png".format(
                group, safe_name, index
            )
            save_sample_panel(
                sample,
                dataset.img_paths_1[index],
                output_path,
                cfg["MEAN"],
                cfg["STD"],
            )

    print("Dual-view visualizations saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
