"""E2 aggregation helpers; subset definitions are inherited from E1 records."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def mean_or_nan(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def event_rows(image_id, epoch, arrays, masks):
    rows=[]
    for name, mask in masks.items():
        labels, n = ndimage.label(mask.astype(bool), structure=np.ones((3,3), bool))
        for idx in range(1, n + 1):
            event = labels == idx
            acc = mean_or_nan(arrays["correct"][event])
            rows.append({"image_id":image_id,"epoch":epoch,"method":arrays["method"],
                         "event_type":name,"event_index":idx,"area_pixels":int(event.sum()),
                         "pixel_arbitration_accuracy":acc,
                         "event_arbitrated":bool(np.isfinite(acc) and acc > .5),
                         "mean_D":mean_or_nan(arrays["D"][event])})
    return rows


def bootstrap(values, seed=2026, samples=1000):
    values=np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(values): return [float("nan")]*3
    rng=np.random.default_rng(seed)
    b=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(samples)])
    return [float(values.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
