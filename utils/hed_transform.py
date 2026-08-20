"""RGB to HED transform for H&E pathology images."""

from __future__ import annotations

import numpy as np
import torch


class RGBToHEDTransform:
    """Convert RGB images to Ruifrok-Macenko HED representation."""

    RGB_FROM_HED = np.asarray(
        [
            [0.650, 0.700, 0.290],
            [0.070, 0.990, 0.110],
            [0.270, 0.570, 0.780],
        ],
        dtype=np.float32,
    )

    HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)

    def __call__(self, image):
        rgb = image.astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 1e-6, 1.0)
        optical_density = -np.log(rgb)
        hed = optical_density @ self.HED_FROM_RGB
        hed = np.ascontiguousarray(hed.transpose(2, 0, 1).astype(np.float32))
        return torch.from_numpy(hed).float()
