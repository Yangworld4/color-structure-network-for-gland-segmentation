"""Small native-grid visualizations only; no map is enlarged for metrics."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plots(records, epochs, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for subset, color in (("CC","#377eb8"),("WW","#e41a1c"),("HC90WW","#984ea3")):
        vals = [r["D"] for r in records if r[f"is_{subset}"] and np.isfinite(r["D"])]
        if vals: ax.hist(vals, bins=35, density=True, histtype="step", linewidth=1.7, label=f"{subset} (n={len(vals)})", color=color)
    ax.axvline(0,color="black",lw=.8); ax.set(xlabel="D = S_opp - S_cur", ylabel="density"); ax.legend(); fig.tight_layout(); fig.savefig(output/"D_distributions.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,4),sharey=True)
    for axis, score, title in zip(axes,("S_FG","S_BG"),("S_FG","S_BG")):
        for subset, color in (("PURE_FG","#4daf4a"),("PURE_BG","#ff7f00")):
            vals=[r[score] for r in records if (r["is_pure"] and r["gt_token"]==("FG" if subset=="PURE_FG" else "BG"))]
            axis.hist(vals,bins=35,density=True,histtype="step",linewidth=1.6,label=subset,color=color)
        axis.set_title(title); axis.legend()
    fig.tight_layout(); fig.savefig(output/"pure_score_distributions.png",dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for subset,color,label in (("WW","#e41a1c","WW arbitration"),("HC90WW","#984ea3","HC90WW arbitration"),("CC","#377eb8","CC preservation")):
        xs=[]; ys=[]
        for epoch in epochs:
            rs=[r for r in records if r["epoch"]==epoch and r[f"is_{subset}"] and r["UNI_pred"]!="TIE"]
            if rs: xs.append(epoch); ys.append(np.mean([r["UNI_correct"] for r in rs]))
        if xs: ax.plot(xs,ys,marker="o",label=label,color=color)
    ax.axhline(.5,color="black",lw=.8,ls="--"); ax.set(xlabel="CPS epoch",ylabel="UNI accuracy",ylim=(0,1)); ax.legend(); fig.tight_layout(); fig.savefig(output/"epoch_arbitration.png",dpi=150); plt.close(fig)


def token_maps(records, output: Path, limit=4):
    output.mkdir(parents=True, exist_ok=True)
    candidates=[]
    for epoch in sorted({r["epoch"] for r in records}, reverse=True):
        for image_id in sorted({r["image_id"] for r in records if r["epoch"]==epoch}):
            rs=[r for r in records if r["epoch"]==epoch and r["image_id"]==image_id]
            if any(r["is_HC90WW"] for r in rs) or any(r["is_WW"] for r in rs): candidates.append(rs)
            if len(candidates)>=limit: break
        if len(candidates)>=limit: break
    for rs in candidates:
        h=max(r["token_y"] for r in rs)+1; w=max(r["token_x"] for r in rs)+1
        def grid(key, transform=lambda x:x):
            a=np.full((h,w),np.nan)
            for r in rs: a[r["token_y"],r["token_x"]]=transform(r[key])
            return a
        state=grid("state",lambda x:{"MIXED":-1,"CC":0,"WW":1,"HC90WW":2,"AC":3,"BC":4}.get(x,-1))
        panels=[("GT occupancy",grid("gt_occupancy"),"viridis"),("CPS state",state,"tab10"),("S_FG",grid("S_FG"),"magma"),("S_BG",grid("S_BG"),"magma"),("margin",grid("margin"),"coolwarm"),("UNI pred",grid("UNI_pred",lambda x:{"BG":0,"FG":1,"TIE":.5}[x]),"viridis")]
        fig,axes=plt.subplots(2,3,figsize=(9,6))
        for ax,(title,array,cmap) in zip(axes.flat,panels): ax.imshow(array,cmap=cmap); ax.set(title=title,xticks=[],yticks=[])
        fig.suptitle(f"{rs[0]['image_id']} epoch {rs[0]['epoch']} | native {h}x{w}"); fig.tight_layout(); fig.savefig(output/f"token_maps_{rs[0]['image_id']}_e{rs[0]['epoch']:03d}.png",dpi=150); plt.close(fig)
