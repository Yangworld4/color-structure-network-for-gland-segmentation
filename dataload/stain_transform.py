"""Stain RGB transform for H&E pathology images."""

from __future__ import annotations

import random

import numpy as np


class StainRGBTransform:
    """Generate stain-diverse RGB view while preserving spatial structure."""

    RGB_FROM_HED = np.asarray(
        [
            [0.650, 0.700, 0.290],
            [0.070, 0.990, 0.110],
            [0.270, 0.570, 0.780],
        ],
        dtype=np.float32,
    )

    HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)

    def __init__(self, training=True):
        self.training = training

    def __call__(self, image):

        rgb = np.clip(
            image.astype(np.float32) / 255.0,
            1e-6,
            1.0
        )

        # RGB -> optical density
        concentrations = -np.log(rgb) @ self.HED_FROM_RGB


        if self.training:

            # Stronger H/E variation
            h_scale = random.uniform(0.75, 1.25)
            e_scale = random.uniform(0.75, 1.25)

            h_shift = random.uniform(-0.03, 0.03)
            e_shift = random.uniform(-0.03, 0.03)


            # additional stain concentration perturbation
            concentration_noise = random.uniform(0.95, 1.05)

        else:
            # deterministic validation transformation
            h_scale = 0.9
            e_scale = 1.1

            h_shift = 0.0
            e_shift = 0.0

            concentration_noise = 1.0


        # Hematoxylin
        concentrations[..., 0] = (
            concentrations[..., 0] * h_scale + h_shift
        )

        # Eosin
        concentrations[..., 1] = (
            concentrations[..., 1] * e_scale + e_shift
        )


        concentrations *= concentration_noise


        # HED -> RGB
        transformed = np.exp(
            -(concentrations @ self.RGB_FROM_HED)
        )


        return np.ascontiguousarray(
            np.round(
                np.clip(transformed, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        )