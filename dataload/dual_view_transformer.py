"""Spatially aligned appearance/structure views for histopathology images."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F


ArrayTransform = Callable[..., Mapping[str, Any]]


def _as_contiguous_uint8(image: np.ndarray) -> np.ndarray:
    """Return a contiguous RGB uint8 array without changing its value range."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image shaped [H,W,3], got {}".format(image.shape))
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class SharedGeometricTransform:
    """Apply the existing project geometry exactly once to image and optional mask."""

    def __init__(self, transform: Optional[ArrayTransform]) -> None:
        self.transform = transform

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        if self.transform is None:
            result: Dict[str, np.ndarray] = {"image": np.ascontiguousarray(image)}
            if mask is not None:
                result["mask"] = np.ascontiguousarray(mask)
            return result

        if mask is None:
            transformed = self.transform(image=image)
        else:
            transformed = self.transform(image=image, mask=mask)

        result = {"image": _as_contiguous_uint8(transformed["image"])}
        if mask is not None:
            result["mask"] = np.ascontiguousarray(transformed["mask"])
        return result


class AppearanceViewTransform:
    """Photometric-only augmentation followed by the project's RGB normalization."""

    def __init__(
        self,
        normalize: ArrayTransform,
        training: bool,
        color_jitter: Mapping[str, Any],
        stain_augmentation: Mapping[str, Any],
    ) -> None:
        self.normalize = normalize
        self.training = training
        self.stain_enabled = bool(stain_augmentation.get("enabled", False))
        self.stain_probability = float(stain_augmentation.get("probability", 0.5))
        self.color_jitter: Optional[A.ColorJitter] = None
        if training and bool(color_jitter.get("enabled", True)):
            self.color_jitter = A.ColorJitter(
                brightness=float(color_jitter.get("brightness", 0.10)),
                contrast=float(color_jitter.get("contrast", 0.10)),
                saturation=float(color_jitter.get("saturation", 0.08)),
                hue=float(color_jitter.get("hue", 0.02)),
                always_apply=True,
                p=1.0,
            )

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        appearance = _as_contiguous_uint8(image).copy()
        if self.training and self.stain_enabled and random.random() < self.stain_probability:
            appearance = self._perturb_stain(appearance)
        if self.color_jitter is not None:
            appearance = self.color_jitter(image=appearance)["image"]

        if mask is None:
            normalized = self.normalize(image=appearance)
        else:
            normalized = self.normalize(image=appearance, mask=np.ascontiguousarray(mask))

        output = {"image": normalized["image"].to(dtype=torch.float32)}
        if mask is not None:
            output["mask"] = normalized["mask"].long()
        return output

    @staticmethod
    def _perturb_stain(image: np.ndarray) -> np.ndarray:
        """Apply a mild H/E concentration perturbation without moving pixels."""
        rgb = np.clip(image.astype(np.float32) / 255.0, 1e-6, 1.0)
        optical_density = -np.log(rgb)
        concentrations = optical_density @ StructureViewGenerator.HED_FROM_RGB

        concentrations[..., 0] = (
            concentrations[..., 0] * random.uniform(0.90, 1.10)
            + random.uniform(-0.015, 0.015)
        )
        concentrations[..., 1] = (
            concentrations[..., 1] * random.uniform(0.90, 1.10)
            + random.uniform(-0.015, 0.015)
        )
        perturbed_od = concentrations @ StructureViewGenerator.RGB_FROM_HED
        perturbed = np.exp(-perturbed_od)
        return np.ascontiguousarray(
            np.round(np.clip(perturbed, 0.0, 1.0) * 255.0).astype(np.uint8)
        )


class StructureViewGenerator:
    """Generate configurable H/Gray/Sobel channels from an unjittered RGB image."""

    # Ruifrok HED matrix, matching the fixed matrix used by common HED
    # color-deconvolution implementations. NumPy is used to avoid a new dependency.
    RGB_FROM_HED = np.asarray(
        [
            [0.650, 0.700, 0.290],
            [0.070, 0.990, 0.110],
            [0.270, 0.570, 0.780],
        ],
        dtype=np.float32,
    )
    HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)

    def __init__(
        self,
        channels: Sequence[str] = ("hematoxylin", "gray", "sobel"),
        normalization: str = "minmax",
        eps: float = 1e-6,
    ) -> None:
        canonical = tuple(str(channel).lower() for channel in channels)
        supported = {"hematoxylin", "gray", "sobel"}
        if len(canonical) != 3 or set(canonical) != supported:
            raise ValueError(
                "structure_channels must contain hematoxylin, gray and sobel exactly once"
            )
        if normalization not in {"minmax", "standardize", "none"}:
            raise ValueError(
                "structure_normalization must be minmax, standardize or none"
            )
        self.channels = canonical
        self.normalization = normalization
        self.eps = float(eps)
        self.sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        rgb = np.clip(_as_contiguous_uint8(image).astype(np.float32) / 255.0, 1e-6, 1.0)
        gray = (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ).astype(np.float32)

        # Avoid log(0); the first deconvolved concentration is Hematoxylin.
        optical_density = -np.log(rgb)
        hematoxylin = (optical_density @ self.HED_FROM_RGB)[..., 0]
        hematoxylin = self._robust_minmax(hematoxylin)
        edge = self._robust_minmax(self._sobel(gray))

        channel_map = {
            "hematoxylin": hematoxylin,
            "gray": np.clip(gray, 0.0, 1.0),
            "sobel": edge,
        }
        structure = torch.from_numpy(
            np.ascontiguousarray(np.stack([channel_map[name] for name in self.channels]))
        ).to(dtype=torch.float32)

        if self.normalization == "standardize":
            means = structure.mean(dim=(1, 2), keepdim=True)
            stds = structure.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
            structure = (structure - means) / stds
        elif self.normalization == "minmax":
            structure = structure.clamp(0.0, 1.0)

        if not torch.isfinite(structure).all():
            raise FloatingPointError("Structure view contains NaN or Inf")
        return structure

    def _sobel(self, gray: np.ndarray) -> np.ndarray:
        gray_tensor = torch.from_numpy(np.ascontiguousarray(gray)).view(1, 1, *gray.shape)
        # Replicate padding keeps output centered and avoids an artificial dark border.
        padded = F.pad(gray_tensor, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, self.sobel_x)
        gradient_y = F.conv2d(padded, self.sobel_y)
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + self.eps)
        return magnitude[0, 0].numpy()

    def _robust_minmax(self, array: np.ndarray) -> np.ndarray:
        finite = np.asarray(array, dtype=np.float32)
        lower, upper = np.percentile(finite, (1.0, 99.0))
        scale = max(float(upper - lower), self.eps)
        return np.clip((finite - lower) / scale, 0.0, 1.0).astype(np.float32)


class DualViewTransform:
    """Compose shared geometry, appearance augmentation, and structure extraction."""

    def __init__(
        self,
        geometric_transform: Optional[ArrayTransform],
        appearance_normalize: ArrayTransform,
        config: Mapping[str, Any],
        training: bool,
    ) -> None:
        self.debug_visualization = bool(config.get("debug_visualization", False))
        self.shared_geometry = SharedGeometricTransform(geometric_transform)
        self.appearance = AppearanceViewTransform(
            normalize=appearance_normalize,
            training=training,
            color_jitter=config.get("color_jitter", {}),
            stain_augmentation=config.get("stain_augmentation", {}),
        )
        self.structure = StructureViewGenerator(
            channels=config.get(
                "structure_channels", ("hematoxylin", "gray", "sobel")
            ),
            normalization=str(config.get("structure_normalization", "minmax")),
        )

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        geometric = self.shared_geometry(image, mask)
        geometric_image = geometric["image"]
        geometric_mask = geometric.get("mask")
        appearance = self.appearance(geometric_image, geometric_mask)

        sample = {
            "image": appearance["image"],
            "image_appearance": appearance["image"],
            "image_structure": self.structure(geometric_image),
        }
        if geometric_mask is not None:
            sample["mask"] = appearance["mask"]
        if self.debug_visualization:
            sample["image_geometric"] = torch.from_numpy(
                np.ascontiguousarray(geometric_image.transpose(2, 0, 1))
            ).to(dtype=torch.float32).div_(255.0)
        return sample
