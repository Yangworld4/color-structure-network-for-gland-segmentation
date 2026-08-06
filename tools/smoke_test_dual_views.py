#!/usr/bin/env python
"""Smoke-test FDCL aligned dual-view datasets without starting training."""

from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from dataload.dual_view_transformer import DualViewTransform
from models.getnetwork import get_network


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_stats(name: str, tensor: torch.Tensor) -> str:
    values = tensor.float()
    return (
        "{}: dtype={} shape={} min={:.6f} max={:.6f} "
        "mean={:.6f} std={:.6f}".format(
            name,
            tensor.dtype,
            tuple(tensor.shape),
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(values.std()),
        )
    )


def assert_dual_batch(batch: Dict[str, Any], supervised: bool) -> None:
    expected = {"image", "image_appearance", "image_structure", "ID"}
    if supervised:
        expected.add("mask")
    assert expected.issubset(batch), (expected, batch.keys())
    appearance = batch["image_appearance"]
    structure = batch["image_structure"]
    assert appearance.dtype == torch.float32
    assert structure.dtype == torch.float32
    assert appearance.shape == structure.shape
    assert appearance.ndim == 4 and appearance.shape[1] == 3
    assert torch.equal(batch["image"], appearance)
    assert torch.isfinite(structure).all()
    channel_std = structure.float().std(dim=(0, 2, 3))
    assert torch.all(channel_std > 1e-6), channel_std
    if supervised:
        mask = batch["mask"]
        assert mask.dtype == torch.int64
        assert tuple(mask.shape[-2:]) == tuple(appearance.shape[-2:])
    else:
        assert "mask" not in batch


def build_dataset(
    manifest: Path,
    supervised: bool,
    split: str,
    config: Dict[str, Any],
    mean: list[float],
    std: list[float],
):
    transforms = data_transform_2d()
    normalize = data_normalize_2d(mean, std)
    return imagefloder_itn(
        manifest_path=str(manifest),
        data_transform_1=transforms[split],
        data_normalize_1=normalize,
        sup=supervised,
        dual_view_config=config,
        training=split == "train",
    )


def check_synthetic_alignment(
    dual_config: Dict[str, Any], mean: list[float], std: list[float], seed: int
) -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[31:177, 59:211] = 255
    mask = (image[..., 0] > 0).astype(np.uint8)
    config = deepcopy(dual_config)
    config["debug_visualization"] = True
    config["color_jitter"]["enabled"] = False
    transform = DualViewTransform(
        geometric_transform=data_transform_2d()["train"],
        appearance_normalize=data_normalize_2d(mean, std),
        config=config,
        training=True,
    )
    seed_everything(seed)
    sample = transform(image, mask)
    geometric_binary = sample["image_geometric"][0] > 0.5
    assert torch.equal(geometric_binary, sample["mask"].bool())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="GlaS", choices=["GlaS", "CRAG"])
    parser.add_argument("--path_dataset", default=None)
    parser.add_argument("--labeled_ratio", default=10, type=int, choices=[10, 20])
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument(
        "--check_forward",
        action="store_true",
        help="also run one CPU forward through the selected original FDCL model",
    )
    parser.add_argument("--network", default="xnet_sb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dataset_cfg(args.dataset_name)
    dataset_root = Path(args.path_dataset or cfg["PATH_DATASET"])
    manifest_root = dataset_root / "manifests" / "paper_seed_2026"
    labeled_manifest = manifest_root / "labeled_{}.csv".format(args.labeled_ratio)
    unlabeled_manifest = manifest_root / "unlabeled_{}.csv".format(args.labeled_ratio)
    validation_manifest = manifest_root / "validation.csv"
    dual_config = cfg["dual_view"]

    seed_everything(args.seed)
    labeled = build_dataset(
        labeled_manifest, True, "train", dual_config, cfg["MEAN"], cfg["STD"]
    )
    unlabeled = build_dataset(
        unlabeled_manifest, False, "train", dual_config, cfg["MEAN"], cfg["STD"]
    )
    validation = build_dataset(
        validation_manifest, True, "val", dual_config, cfg["MEAN"], cfg["STD"]
    )
    loaders = {
        "labeled": DataLoader(
            labeled,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "unlabeled": DataLoader(
            unlabeled,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }

    batches: Dict[str, Dict[str, Any]] = {}
    for name, loader in loaders.items():
        batch = next(iter(loader))
        batches[name] = batch
        assert_dual_batch(batch, supervised=name == "labeled")
        print("\n{} fields: {}".format(name, list(batch.keys())))
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(tensor_stats(key, value))
            else:
                print("{}: {}".format(key, value))
        if "image_structure" in batch:
            print(
                "structure channel std:",
                batch["image_structure"].float().std(dim=(0, 2, 3)).tolist(),
            )

    # Validation must not contain random geometry or photometric changes.
    validation_first = validation[0]
    validation_second = validation[0]
    for key in ("image", "image_appearance", "image_structure", "mask"):
        assert torch.equal(validation_first[key], validation_second[key]), key

    # All training randomness is reproducible after resetting Python/NumPy/Torch seeds.
    seed_everything(args.seed)
    training_first = labeled[0]
    seed_everything(args.seed)
    training_second = labeled[0]
    for key in ("image", "image_appearance", "image_structure", "mask"):
        assert torch.equal(training_first[key], training_second[key]), key

    # Disabled mode keeps the original return schema and transformation behavior.
    disabled_config = deepcopy(dual_config)
    disabled_config["enabled"] = False
    legacy = build_dataset(
        labeled_manifest, True, "train", disabled_config, cfg["MEAN"], cfg["STD"]
    )
    assert set(legacy[0]) == {"image", "mask", "ID"}

    # With photometric augmentation off, the appearance view equals the legacy path.
    compatible_config = deepcopy(dual_config)
    compatible_config["color_jitter"]["enabled"] = False
    compatible = build_dataset(
        labeled_manifest, True, "train", compatible_config, cfg["MEAN"], cfg["STD"]
    )
    seed_everything(args.seed)
    legacy_sample = legacy[0]
    seed_everything(args.seed)
    compatible_sample = compatible[0]
    assert torch.equal(legacy_sample["image"], compatible_sample["image_appearance"])
    assert torch.equal(legacy_sample["mask"], compatible_sample["mask"])

    check_synthetic_alignment(dual_config, cfg["MEAN"], cfg["STD"], args.seed)

    if args.check_forward:
        model1 = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
        model2 = get_network(args.network, cfg["IN_CHANNELS"], cfg["NUM_CLASSES"])
        model1.eval()
        model2.eval()
        with torch.inference_mode():
            output1 = model1(batches["labeled"]["image_appearance"][:1])
            output2 = model2(batches["labeled"]["image_structure"][:1])
        logits1 = output1[0] if isinstance(output1, (tuple, list)) else output1
        logits2 = output2[0] if isinstance(output2, (tuple, list)) else output2
        assert logits1.ndim == 4 and logits1.shape[1] == cfg["NUM_CLASSES"]
        assert logits2.ndim == 4 and logits2.shape[1] == cfg["NUM_CLASSES"]
        print("\nmodel1 appearance logits:", tuple(logits1.shape), logits1.dtype)
        print("model2 structure logits:", tuple(logits2.shape), logits2.dtype)

    print("\nDual-view smoke test: PASS")


if __name__ == "__main__":
    main()
