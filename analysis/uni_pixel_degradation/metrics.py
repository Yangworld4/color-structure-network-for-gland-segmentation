"""Aggregation, component, and image-level bootstrap utilities for E1."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def mean_or_nan(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def boundary_distance(gt: np.ndarray) -> np.ndarray:
    gt = gt.astype(bool); structure = np.ones((3, 3), bool)
    boundary = gt ^ ndimage.binary_erosion(gt, structure=structure)
    boundary |= (~gt) ^ ndimage.binary_erosion(~gt, structure=structure)
    return ndimage.distance_transform_edt(~boundary).astype(np.float32)


def component_size_map(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, number = ndimage.label(mask.astype(bool), structure=np.ones((3,3),bool))
    if not number: return np.zeros(mask.shape, np.int32), 0
    sizes = np.bincount(labels.ravel()); return sizes[labels].astype(np.int32), int(number)


def metric_row(name, arrays, mask, num_events=0):
    mask = mask.astype(bool); n = int(mask.sum())
    if not n:
        return {"subset":name,"num_images":0,"num_pixels":0,"num_events":num_events,"pixel_arbitration_accuracy":float("nan"),"mean_S_FG":float("nan"),"mean_S_BG":float("nan"),"mean_margin":float("nan"),"mean_S_cur":float("nan"),"mean_S_opp":float("nan"),"mean_D":float("nan"),"median_D":float("nan"),"D_positive_rate":float("nan")}
    get=lambda key: arrays[key][mask]
    return {"subset":name,"num_images":1,"num_pixels":n,"num_events":num_events,"pixel_arbitration_accuracy":mean_or_nan(get("uni_correct")),"mean_S_FG":mean_or_nan(get("S_FG")),"mean_S_BG":mean_or_nan(get("S_BG")),"mean_margin":mean_or_nan(get("margin")),"mean_S_cur":mean_or_nan(get("S_cur")),"mean_S_opp":mean_or_nan(get("S_opp")),"mean_D":mean_or_nan(get("D")),"median_D":float(np.nanmedian(get("D"))),"D_positive_rate":mean_or_nan(get("D")>0)}


def merge_rows(rows):
    if not rows: return None
    out={"subset":rows[0]["subset"],"num_images":sum(r["num_images"] for r in rows if r["num_pixels"]),"num_pixels":sum(r["num_pixels"] for r in rows),"num_events":sum(r["num_events"] for r in rows)}
    keys=("pixel_arbitration_accuracy","mean_S_FG","mean_S_BG","mean_margin","mean_S_cur","mean_S_opp","mean_D","D_positive_rate")
    for key in keys:
        valid=[r for r in rows if r["num_pixels"] and np.isfinite(r[key])]
        out[key]=float(np.average([r[key] for r in valid],weights=[r["num_pixels"] for r in valid])) if valid else float("nan")
    # Exact global median requires raw D; intentionally omitted from merges until
    # supplied by the caller, because a mean of image medians is not a median.
    out["median_D"]=float("nan")
    return out


def event_rows(image_id, epoch, arrays, masks):
    rows=[]
    for name,mask in masks.items():
        labels,n=ndimage.label(mask.astype(bool),structure=np.ones((3,3),bool))
        for i in range(1,n+1):
            event=labels==i; accuracy=mean_or_nan(arrays["uni_correct"][event])
            rows.append({"image_id":image_id,"epoch":epoch,"event_type":name,"event_index":i,"area_pixels":int(event.sum()),"pixel_arbitration_accuracy":accuracy,"mean_D":mean_or_nan(arrays["D"][event]),"event_arbitrated":bool(np.isfinite(accuracy) and accuracy>.5)})
    return rows


def bootstrap(values, seed=2026, samples=1000):
    values=np.asarray([x for x in values if np.isfinite(x)],float)
    if not len(values): return [float("nan")]*3
    rng=np.random.default_rng(seed); b=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(samples)])
    return [float(values.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]
