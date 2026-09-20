#!/usr/bin/env python3
"""Offline per-epoch scale diagnostic for existing E4 contradiction D records."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/"outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv"
OUT=ROOT/"outputs/per_epoch_d_scale"
EPOCHS=[40,80,120,160,200]; EPS=1e-12

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["empty"]); w.writeheader(); w.writerows(rows or [{"empty":True}])
def safe(x):
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,(np.floating,float)): return None if not np.isfinite(x) else float(x)
    if isinstance(x,np.integer): return int(x)
    return x

def stats(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return {"N":0,"mean":np.nan,"std":np.nan,"min":np.nan,"max":np.nan,**{f"p{x}":np.nan for x in (5,10,25,50,75,90,95)}}
    q=np.percentile(a,[5,10,25,50,75,90,95])
    return {"N":int(len(a)),"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max()),**{f"p{x}":float(v) for x,v in zip((5,10,25,50,75,90,95),q)}}

def main():
    if not INPUT.is_file(): raise FileNotFoundError(INPUT)
    grouped={e:[] for e in EPOCHS}; layered={(e,s):[] for e in EPOCHS for s in ("PURE","MIXED")}; total_rows=0; kept_rows=0
    with INPUT.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            total_rows+=1; e=int(r["epoch"])
            if e not in grouped: continue
            agreement=str(r["agreement"]).strip().lower()=="true"; contradiction=str(r["contradiction"]).strip().lower()=="true"; d=float(r["D"])
            if not (agreement and contradiction and np.isfinite(d) and d>0): continue
            grouped[e].append(d); kept_rows+=1
            subset=str(r.get("gt_pure_or_mixed","UNKNOWN")).upper()
            subset="PURE" if subset=="PURE" else "MIXED" if subset=="MIXED" else "UNKNOWN"
            if subset in ("PURE","MIXED"): layered[(e,subset)].append(d)
    if any(len(v)==0 for v in grouped.values()): raise RuntimeError({e:len(v) for e,v in grouped.items()})
    rows=[]
    for e in EPOCHS:
        s=stats(grouped[e]); rows.append({"epoch":e,"N":s["N"],"mean":s["mean"],"std":s["std"],"min":s["min"],"max":s["max"],"p05":s["p5"],"p10":s["p10"],"p25":s["p25"],"p50":s["p50"],"p75":s["p75"],"p90":s["p90"],"p95":s["p95"],"fraction_of_all_selected":s["N"]/kept_rows})
    ref=rows[0]; drift=[]
    for r in rows:
        drift.append({"epoch":r["epoch"],"D_IQR":r["p75"]-r["p25"],"D_p90_minus_p10":r["p90"]-r["p10"],"delta_p10":r["p10"]-ref["p10"],"delta_p25":r["p25"]-ref["p25"],"delta_p50":r["p50"]-ref["p50"],"delta_p75":r["p75"]-ref["p75"],"delta_p90":r["p90"]-ref["p90"],"relative_median_drift":(r["p50"]-ref["p50"])/(ref["p50"]+EPS),"relative_p90_drift":(r["p90"]-ref["p90"])/(ref["p90"]+EPS)})
    layer_rows=[]
    for e in EPOCHS:
        for subset in ("PURE","MIXED"):
            s=stats(layered[(e,subset)]); layer_rows.append({"epoch":e,"subset":subset,"num_tokens":s["N"],"D_mean":s["mean"],"D_std":s["std"],"D_min":s["min"],"D_max":s["max"],"D_p25":s["p25"],"D_p50":s["p50"],"D_p75":s["p75"],"D_p90":s["p90"]})
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"plots").mkdir(exist_ok=True)
    write_csv(OUT/"epoch_d_metrics.csv",rows); write_csv(OUT/"epoch_d_drift.csv",drift); write_csv(OUT/"pure_mixed_d_metrics.csv",layer_rows)
    medians=[r["p50"] for r in rows]; p90s=[r["p90"] for r in rows]; med_ratio=max(medians)/(min(medians)+EPS); med_range=max(medians)-min(medians); p90_ratio=max(p90s)/(min(p90s)+EPS)
    # Descriptive preregistered-style bands: small drift is <=25% for both
    # median and p90; larger but not extreme drift is the middle category.
    max_abs_med=max(abs(r["relative_median_drift"]) for r in drift); max_abs_p90=max(abs(r["relative_p90_drift"]) for r in drift)
    if max_abs_med<=.25 and max_abs_p90<=.25: outcome="A"; text="Outcome A — Raw D has a relatively stable numerical scale across training epochs. A fixed D-to-weight mapping is plausible."
    elif max_abs_med<=.75 and max_abs_p90<=.75: outcome="B"; text="Outcome B — Raw D shows moderate epoch-dependent scale drift. A fixed mapping may work, but scale normalization should be considered carefully."
    else: outcome="C"; text="Outcome C — Raw D has strong epoch-dependent scale drift. Direct use of absolute D with a fixed mapping is not well justified; adaptive normalization by training stage is needed."
    summary={"input":str(INPUT),"epochs":EPOCHS,"filter":{"agreement":True,"contradiction":True,"D_gt":0,"total_input_rows":total_rows,"selected_rows":kept_rows},"median_D":{str(r["epoch"]):r["p50"] for r in rows},"p90_D":{str(r["epoch"]):r["p90"] for r in rows},"median_ratio_max_over_min":med_ratio,"median_absolute_range":med_range,"p90_ratio_max_over_min":p90_ratio,"max_abs_relative_median_drift":max_abs_med,"max_abs_relative_p90_drift":max_abs_p90,"distribution_stability":"stable" if outcome=="A" else "moderate" if outcome=="B" else "strong_drift","outcome":outcome,"conclusion":text}
    (OUT/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n")
    try:
        import matplotlib.pyplot as plt
        x=np.array(EPOCHS)
        plt.figure(figsize=(8,5))
        for q,label in (("p10","p10"),("p25","p25"),("p50","p50"),("p75","p75"),("p90","p90")): plt.plot(x,[r[q] for r in rows],marker="o",label=label)
        plt.xlabel("epoch"); plt.ylabel("D = S_opp - S_cur"); plt.title("Contradiction D quantiles by epoch"); plt.xticks(EPOCHS); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"plots/d_quantiles_vs_epoch.png",dpi=160); plt.close()
        plt.figure(figsize=(8,5)); plt.errorbar(x,[r["p50"] for r in rows],yerr=[[r["p50"]-r["p10"] for r in rows],[r["p90"]-r["p50"] for r in rows]],fmt="o-",capsize=4); plt.fill_between(x,[r["p25"] for r in rows],[r["p75"] for r in rows],alpha=.2,label="p25-p75"); plt.xlabel("epoch"); plt.ylabel("D"); plt.title("Per-epoch D quantile summary"); plt.xticks(EPOCHS); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"plots/d_quantile_summary.png",dpi=160); plt.close()
    except Exception as exc:
        summary["plot_error"]=str(exc); (OUT/"summary.json").write_text(json.dumps(safe(summary),indent=2)+"\n")
    readme=f"""# Per-Epoch Contradiction D Scale Diagnostic\n\nInput: `{INPUT}`\n\nFilter: `agreement == true`, `contradiction == true`, `D > 0`; epochs `{EPOCHS}`. No model, UNI, CPS, or reference-memory forward was run.\n\nSelected records: `{kept_rows}` / `{total_rows}`.\n\nOutcome: **{outcome}**\n\n{text}\n"""
    (OUT/"README.md").write_text(readme)
    print("=== Per-Epoch Contradiction D Scale Diagnostic ===")
    for r in rows: print(f"Epoch {r['epoch']}: N={r['N']} mean={r['mean']:.6f} p10={r['p10']:.6f} p25={r['p25']:.6f} p50={r['p50']:.6f} p75={r['p75']:.6f} p90={r['p90']:.6f}")
    print(f"Median D: min={min(medians):.6f} max={max(medians):.6f} max/min={med_ratio:.6f} range={med_range:.6f}")
    print(f"P90 D: min={min(p90s):.6f} max={max(p90s):.6f}")
    print("PURE median trend:", [round(next(x["D_p50"] for x in layer_rows if x["epoch"]==e and x["subset"]=="PURE"),6) for e in EPOCHS])
    print("MIXED median trend:", [round(next(x["D_p50"] for x in layer_rows if x["epoch"]==e and x["subset"]=="MIXED"),6) for e in EPOCHS])
    print("Outcome:",outcome); print("Output:",OUT)

if __name__=="__main__": main()
