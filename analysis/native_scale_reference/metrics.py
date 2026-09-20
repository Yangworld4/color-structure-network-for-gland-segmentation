"""Deterministic aggregation and image-level bootstrap helpers for E4."""

from __future__ import annotations

import numpy as np


def mean(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def median(values):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def bootstrap(values, seed=2026, samples=1000):
    a = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(a): return [float("nan")]*3
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(a, len(a), replace=True).mean() for _ in range(samples)])
    return [float(a.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
