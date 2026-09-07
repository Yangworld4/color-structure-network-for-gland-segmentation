"""GT gland regions for the E0 structural diagnostic."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


INTERIOR_RADIUS = 5
BOUNDARY_RADIUS = 3
SEPARATOR_DISTANCE = 12


def _disk(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return (xx * xx + yy * yy <= radius * radius)


def _binary_instances(binary_mask: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(binary_mask, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32, copy=False)


def prepare_instance_labels(
    binary_mask: np.ndarray, instance_mask: np.ndarray | None
) -> np.ndarray:
    """Use supplied identity labels when meaningful, else connected components."""
    binary = np.asarray(binary_mask).astype(bool)
    if instance_mask is None:
        return _binary_instances(binary)
    raw = np.asarray(instance_mask)
    labels = np.zeros(binary.shape, dtype=np.int32)
    positive = raw > 0
    labels[positive & binary] = raw[positive & binary].astype(np.int32)
    unique = np.unique(labels[labels > 0])
    if unique.size >= 2:
        return labels
    return _binary_instances(binary)


def make_region_masks(
    binary_mask: np.ndarray,
    instance_mask: np.ndarray | None = None,
    interior_radius: int = INTERIOR_RADIUS,
    boundary_radius: int = BOUNDARY_RADIUS,
    separator_distance: int = SEPARATOR_DISTANCE,
) -> Dict[str, np.ndarray]:
    """Build interior, narrow boundary, and inter-gland separator masks."""
    binary = np.asarray(binary_mask).astype(bool)
    structure_interior = _disk(interior_radius)
    structure_boundary = _disk(boundary_radius)
    interior = ndimage.binary_erosion(binary, structure=structure_interior)
    eroded = ndimage.binary_erosion(binary, structure=structure_boundary)
    dilated = ndimage.binary_dilation(binary, structure=structure_boundary)
    boundary = dilated & ~eroded

    labels = prepare_instance_labels(binary, instance_mask)
    instance_ids = np.unique(labels[labels > 0])
    if instance_ids.size < 2:
        separator = np.zeros(binary.shape, dtype=bool)
    else:
        # Each distance transform is computed in compiled scipy code. The
        # stack is small for GLaS and avoids a Python pixel-by-gland loop.
        distances = np.stack(
            [ndimage.distance_transform_edt(labels != instance_id)
             for instance_id in instance_ids],
            axis=0,
        )
        nearest_two = np.partition(distances, kth=1, axis=0)[:2]
        separator = (
            ~binary
            & (nearest_two[1] <= float(separator_distance))
            & np.isfinite(nearest_two[1])
        )

    return {
        "interior": interior,
        "boundary": boundary,
        "separator": separator,
        "instance_labels": labels,
    }

