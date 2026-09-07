"""Pixel and object-level metrics for the E0 diagnostic."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


MIN_COMPONENT_AREA = 10
OBJECT_OVERLAP_THRESHOLD = 0.30


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = np.logical_and(prediction, target).sum(dtype=np.float64)
    fp = np.logical_and(prediction, ~target).sum(dtype=np.float64)
    fn = np.logical_and(~prediction, target).sum(dtype=np.float64)
    union = tp + fp + fn
    target_sum = tp + fn
    pred_sum = tp + fp
    return {
        "dice": float(2.0 * tp / (pred_sum + target_sum))
        if pred_sum + target_sum else 1.0,
        "iou": float(tp / union) if union else 1.0,
    }


def regional_error(
    prediction: np.ndarray, target: np.ndarray, region: np.ndarray
) -> float:
    region = np.asarray(region).astype(bool)
    if not region.any():
        return float("nan")
    mismatch = np.asarray(prediction).astype(bool) != np.asarray(target).astype(bool)
    return float(mismatch[region].mean())


def _large_components(mask: np.ndarray, min_area: int, preserve_labels: bool = False) -> np.ndarray:
    if preserve_labels:
        labels = np.asarray(mask).astype(np.int32, copy=False)
    else:
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    ids = np.unique(labels[labels > 0])
    result = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for component_id in ids:
        component = labels == component_id
        if component.sum() >= min_area:
            result[component] = next_id
            next_id += 1
    return result


def merge_split_counts(
    prediction: np.ndarray,
    gt_instance_labels: np.ndarray,
    min_component_area: int = MIN_COMPONENT_AREA,
    overlap_threshold: float = OBJECT_OVERLAP_THRESHOLD,
) -> Dict[str, int]:
    """Count prediction components merging GT glands and GT glands being split."""
    pred_labels = _large_components(np.asarray(prediction).astype(bool), min_component_area)
    gt_labels = _large_components(
        np.asarray(gt_instance_labels), min_component_area, preserve_labels=True
    )
    pred_ids = np.unique(pred_labels[pred_labels > 0])
    gt_ids = np.unique(gt_labels[gt_labels > 0])

    merge_count = 0
    for pred_id in pred_ids:
        pred_pixels = pred_labels == pred_id
        covered_gt = []
        for gt_id in gt_ids:
            gt_pixels = gt_labels == gt_id
            coverage = np.logical_and(pred_pixels, gt_pixels).sum() / max(
                1, gt_pixels.sum()
            )
            if coverage > overlap_threshold:
                covered_gt.append(gt_id)
        if len(covered_gt) >= 2:
            merge_count += 1

    split_count = 0
    for gt_id in gt_ids:
        gt_pixels = gt_labels == gt_id
        covered_pred = []
        for pred_id in pred_ids:
            pred_pixels = pred_labels == pred_id
            coverage = np.logical_and(pred_pixels, gt_pixels).sum() / max(
                1, gt_pixels.sum()
            )
            if coverage > overlap_threshold:
                covered_pred.append(pred_id)
        if len(covered_pred) >= 2:
            split_count += 1

    return {"merge_count": int(merge_count), "split_count": int(split_count)}
