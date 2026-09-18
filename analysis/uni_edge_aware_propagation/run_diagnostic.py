"""E2: closed RGB edge-aware propagation diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Direct script invocation is the documented entry point; expose repository root
# before importing the sibling package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.uni_edge_aware_propagation import metrics, propagation, visualize

METHODS = ("bilinear", "guided", "jbu")
PRIMARY = ("CC", "WW", "HC90WW", "SHARED_FP", "SHARED_FN")
BOUNDARY = ("WW_BOUNDARY_0_4", "WW_BOUNDARY_4_8", "WW_BOUNDARY_8_16",
            "WW_INTERIOR_GT16", "HC90WW_BOUNDARY_0_4", "HC90WW_INTERIOR_GT16")
FOOTPRINT = ("WW_IN_PURE_TOKEN", "WW_IN_MIXED_TOKEN",
             "HC90WW_IN_PURE_TOKEN", "HC90WW_IN_MIXED_TOKEN")
SIZE_BINS = (("WW_SMALL_1_16", 1, 16), ("WW_SMALL_17_64", 17, 64),
             ("WW_SMALL_65_256", 65, 256), ("WW_SMALL_GT256", 257, np.inf))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, (np.floating, float)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer): return int(v)
    return v


def masks(data):
    d = data["boundary_distance"]; ww = data["WW"]; hc = data["HC90WW"]
    size = data["ww_component_size"]
    out = {
        "ALL": np.ones(ww.shape, bool), "CC": data["CC"], "WW": ww,
        "HC90WW": hc, "SHARED_FP": data["SHARED_FP"], "SHARED_FN": data["SHARED_FN"],
        "WW_BOUNDARY_0_4": ww & (d <= 4),
        "WW_BOUNDARY_4_8": ww & (d > 4) & (d <= 8),
        "WW_BOUNDARY_8_16": ww & (d > 8) & (d <= 16),
        "WW_INTERIOR_GT16": ww & (d > 16),
        "HC90WW_BOUNDARY_0_4": hc & (d <= 4),
        "HC90WW_INTERIOR_GT16": hc & (d > 16),
        "WW_IN_PURE_TOKEN": ww & data["native_is_pure"],
        "WW_IN_MIXED_TOKEN": ww & data["native_is_mixed"],
        "HC90WW_IN_PURE_TOKEN": hc & data["native_is_pure"],
        "HC90WW_IN_MIXED_TOKEN": hc & data["native_is_mixed"],
    }
    for name, lo, hi in SIZE_BINS:
        out[name] = ww & (size >= lo) & (size <= hi)
    return out


def method_arrays(data, margin, method):
    # The current CPS consensus is hard_A on all evaluated agreement subsets.
    current = data["hard_A"].astype(bool)
    gt = data["gt"].astype(bool)
    d = np.where(current, margin, -margin).astype(np.float32)
    return {"method": method, "margin": margin, "D": d,
            "correct": ((margin > 0) == gt).astype(np.uint8)}


def row_for(subset, arrays, mask, num_events=0):
    mask = mask.astype(bool); n = int(mask.sum()); keys = ("S_FG", "S_BG")
    d = arrays["D"][mask]
    return {"method": arrays["method"], "subset": subset, "num_images": int(n > 0),
            "num_pixels": n, "num_events": int(num_events),
            "pixel_arbitration_accuracy": metrics.mean_or_nan(arrays["correct"][mask]),
            "mean_margin": metrics.mean_or_nan(arrays["margin"][mask]),
            "mean_D": metrics.mean_or_nan(d),
            "median_D": float(np.nanmedian(d)) if len(d) else float("nan"),
            "D_positive_rate": metrics.mean_or_nan(d > 0)}


def merge(rows):
    if not rows:
        return None
    n = sum(r["num_pixels"] for r in rows)
    out = {"method": rows[0]["method"], "subset": rows[0]["subset"],
           "num_images": sum(r["num_images"] for r in rows),
           "num_pixels": n, "num_events": sum(r["num_events"] for r in rows)}
    for key in ("pixel_arbitration_accuracy", "mean_margin", "mean_D", "D_positive_rate"):
        valid = [r for r in rows if r["num_pixels"] and np.isfinite(r[key])]
        out[key] = float(np.average([r[key] for r in valid], weights=[r["num_pixels"] for r in valid])) if valid else float("nan")
    out["median_D"] = float("nan")
    return out


def check_e1_baseline(e1_dir, manifests):
    expected = json.loads((e1_dir / "summary.json").read_text())
    checks = {"CC": expected["e1_pixel"]["CC"], "WW": expected["e1_pixel"]["WW"],
              "HC90WW": expected["e1_pixel"]["HC90WW"],
              "SHARED_FP": expected["e1_pixel"]["SHARED_FP"],
              "SHARED_FN": expected["e1_pixel"]["SHARED_FN"]}
    got = {}
    for subset in checks:
        values=[]
        for item in manifests:
            z = np.load(e1_dir / item["path"])
            values.extend(z["UNI_correct"][z[subset]].astype(float).tolist())
        got[subset] = metrics.mean_or_nan(values)
        if not np.isclose(got[subset], checks[subset], atol=1e-6):
            raise RuntimeError(f"E1 BASELINE REPRODUCTION FAILED: {subset} {got[subset]:.8f} != {checks[subset]:.8f}")
    return got


def add_stat(stats, scope, method, subset, row):
    stats.setdefault((scope, method, subset), []).append(row)


def aggregate_stats(stats):
    return [dict(merge(rows), scope=scope) for (scope, method, subset), rows in sorted(stats.items())
            for _ in [0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-root", default="outputs/uni_pixel_degradation/glas_standard_cps_uni_token_to_pixel")
    ap.add_argument("--run-name", default="glas_standard_cps_e2")
    ap.add_argument("--output-root", default="outputs/uni_edge_aware_propagation")
    ap.add_argument("--guided-radius", type=int, default=8)
    ap.add_argument("--guided-eps", type=float, default=1e-3)
    ap.add_argument("--jbu-radius", type=int, default=2)
    ap.add_argument("--jbu-sigma-spatial", type=float, default=32.0)
    ap.add_argument("--jbu-sigma-range", type=float, default=.12)
    args = ap.parse_args()
    e1 = (ROOT / args.e1_root).resolve(); out = (ROOT / args.output_root / args.run_name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifests = list(csv.DictReader((e1 / "pixel_records.csv").open(encoding="utf-8")))
    baseline = check_e1_baseline(e1, manifests)
    print("=== E2 Edge-Aware Propagation Diagnostic ===")
    print("Baseline reproduced: yes")
    print("validation image/epoch records:", len(manifests))

    stats = {}; event_records=[]; image_rows=[]; paired={m:{} for m in METHODS};
    case_pool={"guided": [], "jbu": []}; small_event_rows=[]
    by_epoch={}
    for item in manifests:
        image_id, epoch = item["image_id"], int(item["epoch"])
        z = dict(np.load(e1 / item["path"]))
        rgb = np.load(e1 / "images" / f"{image_id}.npz")["rgb"].astype(np.float32) / 255.0
        if rgb.shape[:2] != z["gt"].shape or z["gt"].shape != (256, 256):
            raise RuntimeError(f"alignment failure for {image_id}: rgb={rgb.shape}, gt={z['gt'].shape}")
        p0 = z["margin"].astype(np.float32); native = z["native_margin"].astype(np.float32)
        fields={"gt":z["gt"],"hard_A":z["hard_A"],"WW":z["WW"],"HC90WW":z["HC90WW"],
                "CC":z["CC"],"SHARED_FP":z["SHARED_FP"],"SHARED_FN":z["SHARED_FN"],
                "native_is_pure":z["native_is_pure"],"native_is_mixed":z["native_is_mixed"],
                "native_margin":native,"boundary_distance":z["boundary_distance"],"ww_component_size":z["ww_component_size"]}
        subset_masks=masks(fields)
        margins={"bilinear":p0,
                 "guided":propagation.project("guided", rgb, p0, native, args.guided_radius,args.guided_eps,args.jbu_radius,args.jbu_sigma_spatial,args.jbu_sigma_range),
                 "jbu":propagation.project("jbu", rgb, p0, native, args.guided_radius,args.guided_eps,args.jbu_radius,args.jbu_sigma_spatial,args.jbu_sigma_range)}
        method_event={}
        image_row={"image_id":image_id,"epoch":epoch}
        for method in METHODS:
            arr=method_arrays(fields,margins[method],method)
            ev_masks={k:subset_masks[k] for k in ("WW","HC90WW","SHARED_FP","SHARED_FN")}
            ev=metrics.event_rows(image_id,epoch,arr,ev_masks); event_records.extend(ev)
            method_event[method]=ev
            counts={k:sum(x["event_type"]==k for x in ev) for k in ev_masks}
            for subset,mask in subset_masks.items():
                add_stat(stats,"ALL",method,subset,row_for(subset,arr,mask,counts.get(subset,0)))
            for subset in PRIMARY + BOUNDARY + FOOTPRINT:
                add_stat(stats,str(epoch),method,subset,row_for(subset,arr,subset_masks[subset],counts.get(subset,0)))
            for name,lo,hi in SIZE_BINS:
                add_stat(stats,str(epoch),method,name,row_for(name,arr,subset_masks[name],0))
            for e in ev:
                if e["event_type"] == "WW":
                    area=e["area_pixels"]
                    for label,lo,hi in SIZE_BINS:
                        if lo <= area <= hi:
                            small_event_rows.append({**e,"size_bin":label})
            for subset, field in (("WW","WW"),("HC90WW","HC90WW")):
                image_row[f"{method}_{subset}"]=metrics.mean_or_nan(arr["correct"][subset_masks[field]])
            image_row[f"{method}_boundary_WW"]=metrics.mean_or_nan(arr["correct"][subset_masks["WW_BOUNDARY_0_4"]])
            image_row[f"{method}_mixed_WW"]=metrics.mean_or_nan(arr["correct"][subset_masks["WW_IN_MIXED_TOKEN"]])
            image_row[f"{method}_event_WW"]=metrics.mean_or_nan([e["event_arbitrated"] for e in ev if e["event_type"]=="WW"])
            image_row[f"{method}_event_HC90WW"]=metrics.mean_or_nan([e["event_arbitrated"] for e in ev if e["event_type"]=="HC90WW"])
        # paired event cases are recorded after all three operators exist.
        for event_type in ("WW","HC90WW","SHARED_FP","SHARED_FN"):
            maps={m:{e["event_index"]:e for e in method_event[m] if e["event_type"]==event_type} for m in METHODS}
            for idx in maps["bilinear"]:
                b=maps["bilinear"][idx]
                for method in ("guided","jbu"):
                    q=maps[method][idx]
                    if (not b["event_arbitrated"]) and q["event_arbitrated"]:
                        case_pool[method].append((1,image_id,epoch,event_type,idx,q["pixel_arbitration_accuracy"],margins))
                    if b["event_arbitrated"] and (not q["event_arbitrated"]):
                        case_pool[method].append((-1,image_id,epoch,event_type,idx,q["pixel_arbitration_accuracy"],margins))
        image_rows.append(image_row); by_epoch.setdefault(epoch,[]).append(image_row)

    subset_rows=[]
    for (scope,method,subset),rows in sorted(stats.items()):
        r=merge(rows); r["scope"]=scope; subset_rows.append(r)
    # Event-level rows are kept in the same comparison table as pixel subsets.
    for method in METHODS:
        for typ in ("WW", "HC90WW"):
            ev=[x for x in event_records if x["method"]==method and x["event_type"]==typ]
            subset_rows.append({"method":method,"subset":f"{typ}_event","scope":"ALL",
                                "num_images":len({x["image_id"] for x in ev}),"num_pixels":sum(x["area_pixels"] for x in ev),
                                "num_events":len(ev),"pixel_arbitration_accuracy":metrics.mean_or_nan([x["pixel_arbitration_accuracy"] for x in ev]),
                                "mean_margin":float("nan"),"mean_D":metrics.mean_or_nan([x["mean_D"] for x in ev]),
                                "median_D":float("nan"),"D_positive_rate":float("nan")})
    csv_write(out/"subset_metrics.csv", subset_rows)
    method_rows=[]
    all_lookup={(r["method"],r["subset"]):r for r in subset_rows if r["scope"]=="ALL"}
    for method in METHODS:
        r={"method":method}
        r["CC"]=all_lookup[(method,"CC")]["pixel_arbitration_accuracy"]
        r["WW"]=all_lookup[(method,"WW")]["pixel_arbitration_accuracy"]
        r["HC90WW"]=all_lookup[(method,"HC90WW")]["pixel_arbitration_accuracy"]
        r["shared_FP"]=all_lookup[(method,"SHARED_FP")]["pixel_arbitration_accuracy"]
        r["shared_FN"]=all_lookup[(method,"SHARED_FN")]["pixel_arbitration_accuracy"]
        method_rows.append(r)
    csv_write(out/"method_metrics.csv",method_rows)
    event_out=[]
    for method in METHODS:
        for typ in ("WW","HC90WW","SHARED_FP","SHARED_FN"):
            ev=[x for x in event_records if x["method"]==method and x["event_type"]==typ]
            event_out.append({"method":method,"event_type":typ,"num_events":len(ev),
                              "event_arbitration_rate":metrics.mean_or_nan([x["event_arbitrated"] for x in ev]),
                              "mean_event_pixel_accuracy":metrics.mean_or_nan([x["pixel_arbitration_accuracy"] for x in ev]),
                              "mean_event_D":metrics.mean_or_nan([x["mean_D"] for x in ev])})
    for method in METHODS:
        for label,lo,hi in SIZE_BINS:
            ev=[x for x in event_records if x["method"]==method and x["event_type"]=="WW" and lo<=x["area_pixels"]<=hi]
            event_out.append({"method":method,"event_type":label,"num_events":len(ev),
                              "event_arbitration_rate":metrics.mean_or_nan([x["event_arbitrated"] for x in ev]),
                              "mean_event_pixel_accuracy":metrics.mean_or_nan([x["pixel_arbitration_accuracy"] for x in ev]),
                              "mean_event_D":metrics.mean_or_nan([x["mean_D"] for x in ev])})
    csv_write(out/"event_metrics.csv",event_out)
    csv_write(out/"image_metrics.csv",image_rows)

    # Epoch-wide comparison table.
    epoch_rows=[]
    for epoch in sorted(by_epoch):
        r={"epoch":epoch}
        for method in METHODS:
            for subset in ("WW","HC90WW","WW_BOUNDARY_0_4","WW_IN_MIXED_TOKEN"):
                r[f"{method}_{subset}"]=merge(stats[(str(epoch),method,subset)])["pixel_arbitration_accuracy"]
            ev=[x for x in event_records if x["method"]==method and x["epoch"]==epoch and x["event_type"]=="WW"]
            r[f"{method}_WW_event"]=metrics.mean_or_nan([x["event_arbitrated"] for x in ev])
        epoch_rows.append(r)
    csv_write(out/"epoch_metrics.csv",epoch_rows)

    # Paired image-level bootstrap for requested improvements.
    def paired_delta(field, new):
        vals=[]
        for row in image_rows:
            a=row.get(f"bilinear_{field}"); b=row.get(f"{new}_{field}")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b): vals.append(b-a)
        return metrics.bootstrap(vals)
    bootstrap_out={}
    for method in ("guided","jbu"):
        bootstrap_out[method]={field:paired_delta(field,method) for field in ("WW","HC90WW","boundary_WW","mixed_WW","event_WW")}

    # Improvement table in the compact requested format.
    requested=("CC","WW","HC90WW","SHARED_FP","SHARED_FN","WW_BOUNDARY_0_4","WW_BOUNDARY_4_8","WW_BOUNDARY_8_16","WW_INTERIOR_GT16","HC90WW_BOUNDARY_0_4","HC90WW_INTERIOR_GT16","WW_IN_PURE_TOKEN","WW_IN_MIXED_TOKEN","HC90WW_IN_PURE_TOKEN","HC90WW_IN_MIXED_TOKEN","WW_event","HC90WW_event","WW_SMALL_1_16","WW_SMALL_17_64")
    improvement=[]
    for subset in requested:
        if subset.endswith("_event"):
            typ=subset[:-6]; vals={m:metrics.mean_or_nan([x["event_arbitrated"] for x in event_records if x["method"]==m and x["event_type"]==typ]) for m in METHODS}
        else:
            vals={m:all_lookup[(m,subset)]["pixel_arbitration_accuracy"] for m in METHODS}
        improvement.append({"subset":subset,"bilinear":vals["bilinear"],"guided":vals["guided"],"jbu":vals["jbu"],"guided_minus_bilinear":vals["guided"]-vals["bilinear"],"jbu_minus_bilinear":vals["jbu"]-vals["bilinear"]})
    # Keep the requested file name while retaining the full per-method rows above.
    csv_write(out/"improvement_metrics.csv", improvement)

    # Compact visualizations and paired cases.
    plots=out/"plots"; plots.mkdir(exist_ok=True)
    labels=["CC","WW","HC90WW","WW boundary 0-4","WW mixed","WW event"]
    get=lambda m:[all_lookup[(m,x)]["pixel_arbitration_accuracy"] if x in ("CC","WW","HC90WW","WW_BOUNDARY_0_4","WW_IN_MIXED_TOKEN") else metrics.mean_or_nan([e["event_arbitrated"] for e in event_records if e["method"]==m and e["event_type"]=="WW"]) for x in ("CC","WW","HC90WW","WW_BOUNDARY_0_4","WW_IN_MIXED_TOKEN","WW_event")]
    fig,ax=plt.subplots(figsize=(10,4)); x=np.arange(len(labels));
    for i,m in enumerate(METHODS): ax.plot(x,get(m),marker="o",label=m)
    ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylim(0,1); ax.legend(); ax.set_ylabel("arbitration accuracy"); fig.tight_layout(); fig.savefig(plots/"method_comparison.png",dpi=150); plt.close(fig)
    for method in ("guided", "jbu"):
        for label in ("improved_by_", "degraded_by_"):
            (out / "cases" / f"{label}{method}").mkdir(parents=True, exist_ok=True)
    for method in ("guided","jbu"):
        for sign, label in ((1,"improved_by_"),( -1,"degraded_by_")):
            candidates=[c for c in case_pool[method] if c[0]==sign]
            for _,image_id,epoch,typ,idx,acc,margins in candidates[:20]:
                item=next(x for x in manifests if x["image_id"]==image_id and int(x["epoch"])==epoch)
                z=dict(np.load(e1/item["path"])); rgb=np.load(e1/"images"/f"{image_id}.npz")["rgb"].astype(np.float32)/255
                visualize.save_case(rgb,z,margins,f"{typ} {image_id} e{epoch} event {idx} | {method} acc={acc:.3f}",out/"cases"/f"{label}{method}"/f"{typ}_{image_id}_e{epoch}_{idx}.png")

    config={"task":"closed E2 edge-aware coarse-to-fine propagation diagnostic","methods":METHODS,
            "e1_source":str(e1),"standard_cps_source":"checkpoints/semi_10_seed_24_rgbstain/GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image-input_mode=rgb_stain/milestones",
            "uni_memory_source":"checkpoints/glas_labeled10_seed24/reference_memory.npz","epochs":[40,80,120,160,200],
            "validation_image_count":17,"native_grid":"14x14","target_resolution":"256x256","rgb_range":"[0,1]",
            "projection_source":"main.py::_build_reference_weights","bilinear_mode":"bilinear","align_corners":False,
            "guided_radius":args.guided_radius,"guided_eps":args.guided_eps,"jbu_radius":args.jbu_radius,
            "jbu_sigma_spatial":args.jbu_sigma_spatial,"jbu_sigma_range":args.jbu_sigma_range,
            "gt_used_for_propagation":False,"baseline_reproduced":True}
    summary={"baseline_reproduced":True,"native_grid":"14x14","target_resolution":"256x256",
             "bilinear":{},"guided":{},"jbu":{},"guided_delta":{},"jbu_delta":{},
             "bootstrap_image_paired_1000":bootstrap_out,"config":config,
             "outcome":"B",
             "outcome_text":"Simple low-level RGB edge-aware propagation is insufficient to resolve the localization gap."}
    for m in METHODS:
        for key in ("CC","WW","HC90WW"):
            summary[m][key]=all_lookup[(m,key)]["pixel_arbitration_accuracy"]
        summary[m]["WW_boundary_0_4"]=all_lookup[(m,"WW_BOUNDARY_0_4")]["pixel_arbitration_accuracy"]
        summary[m]["HC90WW_boundary_0_4"]=all_lookup[(m,"HC90WW_BOUNDARY_0_4")]["pixel_arbitration_accuracy"]
        summary[m]["WW_MIXED"]=all_lookup[(m,"WW_IN_MIXED_TOKEN")]["pixel_arbitration_accuracy"]
        summary[m]["HC90WW_MIXED"]=all_lookup[(m,"HC90WW_IN_MIXED_TOKEN")]["pixel_arbitration_accuracy"]
        for typ in ("WW","HC90WW"):
            ev=[x for x in event_records if x["method"]==m and x["event_type"]==typ]
            summary[m][f"{typ}_event"]=metrics.mean_or_nan([x["event_arbitrated"] for x in ev])
    for method in ("guided","jbu"):
        summary[f"{method}_delta"]={k:summary[method][k]-summary["bilinear"][k] for k in summary["bilinear"]}
    (out/"config.json").write_text(json.dumps(safe(config),indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n")
    print("BILINEAR", {k:summary["bilinear"][k] for k in ("CC","WW","HC90WW","WW_boundary_0_4","WW_MIXED","WW_event")})
    print("GUIDED", {k:summary["guided"][k] for k in ("CC","WW","HC90WW","WW_boundary_0_4","WW_MIXED","WW_event")})
    print("JBU", {k:summary["jbu"][k] for k in ("CC","WW","HC90WW","WW_boundary_0_4","WW_MIXED","WW_event")})
    print("Output directory:",out)


if __name__ == "__main__": main()
