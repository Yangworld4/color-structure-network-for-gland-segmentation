"""Non-learning RGB-guided projection operators for E2.

Both operators consume only an aligned RGB image and the immutable native UNI
margin.  No prediction or target mask is used to form their weights.
"""

from __future__ import annotations

import cv2
import numpy as np


def _box(a: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * int(radius) + 1
    return cv2.boxFilter(a.astype(np.float32), -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter_color(rgb: np.ndarray, p: np.ndarray, radius: int = 8,
                        eps: float = 1e-3) -> np.ndarray:
    """Color guided filter, with RGB as guidance and p as the filtered field."""
    I = np.asarray(rgb, np.float32).clip(0.0, 1.0)
    p = np.asarray(p, np.float32)
    if I.ndim != 3 or I.shape[-1] != 3 or p.shape != I.shape[:2]:
        raise ValueError(f"guided filter shape mismatch: rgb={I.shape}, p={p.shape}")
    r = int(radius)
    mean_I = np.stack([_box(I[..., c], r) for c in range(3)], -1)
    mean_p = _box(p, r)
    cov = np.empty((*p.shape, 3, 3), np.float32)
    cov_p = np.empty((*p.shape, 3), np.float32)
    for c in range(3):
        cov_p[..., c] = _box(I[..., c] * p, r) - mean_I[..., c] * mean_p
        for d in range(3):
            cov[..., c, d] = (_box(I[..., c] * I[..., d], r)
                              - mean_I[..., c] * mean_I[..., d])
    cov += np.eye(3, dtype=np.float32)[None, None] * float(eps)
    a = np.linalg.solve(cov.reshape(-1, 3, 3), cov_p.reshape(-1, 3)).reshape(*p.shape, 3)
    b = mean_p - np.sum(a * mean_I, axis=-1)
    q = sum(_box(a[..., c], r) * I[..., c] for c in range(3)) + _box(b, r)
    return q.astype(np.float32)


def _token_footprints(rgb: np.ndarray, grid: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    yy = np.minimum(np.arange(h) * grid // h, grid - 1)
    xx = np.minimum(np.arange(w) * grid // w, grid - 1)
    means = np.zeros((grid, grid, 3), np.float32)
    counts = np.zeros((grid, grid), np.int32)
    for y in range(h):
        for x in range(w):
            means[yy[y], xx[x]] += rgb[y, x]
            counts[yy[y], xx[x]] += 1
    return means / np.maximum(counts[..., None], 1)


def jbu(rgb: np.ndarray, token_margin: np.ndarray, radius: int = 2,
        sigma_spatial: float = 32.0, sigma_range: float = 0.12,
        eps: float = 1e-8) -> np.ndarray:
    """Joint bilateral upsampling from native tokens using RGB only.

    ``radius`` is measured in native-token grid cells; sigma_spatial is in
    target pixels and sigma_range in normalized RGB units.
    """
    I = np.asarray(rgb, np.float32).clip(0.0, 1.0)
    M = np.asarray(token_margin, np.float32)
    if I.ndim != 3 or I.shape[-1] != 3 or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"JBU shape mismatch: rgb={I.shape}, margin={M.shape}")
    h, w = I.shape[:2]; grid = M.shape[0]
    means = _token_footprints(I, grid)
    ty = np.minimum(np.arange(h) * grid // h, grid - 1)
    tx = np.minimum(np.arange(w) * grid // w, grid - 1)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    base_y = ty[:, None]; base_x = tx[None, :]
    numerator = np.zeros((h, w), np.float32)
    denominator = np.zeros((h, w), np.float32)
    ss = max(float(sigma_spatial), 1e-6)
    sr = max(float(sigma_range), 1e-6)
    for dy in range(-int(radius), int(radius) + 1):
        iy = base_y + dy
        valid_y = (iy >= 0) & (iy < grid)
        iyc = np.clip(iy, 0, grid - 1)
        cy = (iyc.astype(np.float32) + .5) * h / grid
        for dx in range(-int(radius), int(radius) + 1):
            ix = base_x + dx
            valid = valid_y & (ix >= 0) & (ix < grid)
            ixc = np.clip(ix, 0, grid - 1)
            cx = (ixc.astype(np.float32) + .5) * w / grid
            spatial = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * ss * ss))
            mu = means[iyc, ixc]
            appearance = np.sum((I - mu) ** 2, axis=-1)
            weight = spatial * np.exp(-appearance / (2 * sr * sr)) * valid.astype(np.float32)
            numerator += weight * M[iyc, ixc]
            denominator += weight
    return (numerator / (denominator + float(eps))).astype(np.float32)


def project(method: str, rgb: np.ndarray, p0: np.ndarray, native_margin: np.ndarray,
            guided_radius: int, guided_eps: float, jbu_radius: int,
            jbu_sigma_spatial: float, jbu_sigma_range: float) -> np.ndarray:
    if method == "bilinear":
        return p0.astype(np.float32)
    if method == "guided":
        return guided_filter_color(rgb, p0, guided_radius, guided_eps)
    if method == "jbu":
        return jbu(rgb, native_margin, jbu_radius, jbu_sigma_spatial, jbu_sigma_range)
    raise ValueError(method)
