"""Small deterministic aggregation helpers for E3."""

from __future__ import annotations

import numpy as np


def mean(v):
    a = np.asarray(v, float); a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def balanced(correct, gt_fg):
    correct = np.asarray(correct, bool); gt_fg = np.asarray(gt_fg, bool)
    return mean([mean(correct[gt_fg]), mean(correct[~gt_fg])])


def bootstrap(values, seed=2026, samples=1000):
    values = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(values): return [float("nan")]*3
    rng = np.random.default_rng(seed)
    b = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return [float(values.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
