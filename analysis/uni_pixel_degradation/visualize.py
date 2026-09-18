"""Focused E1 projections/case displays. They are descriptive, not metrics."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_case(data, rgb, title, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    cps=data["hard_A"].astype(float); ww=data["WW"].astype(float); hc=data["HC90WW"].astype(float)
    panels=[("RGB",rgb,None),("GT",data["gt"],"gray"),("CPS consensus",cps,"gray"),("WW / HC90WW",ww+2*hc,"magma"),("native margin",data["native_margin"],"coolwarm"),("pixel margin",data["margin"],"coolwarm"),("S_FG pixel",data["S_FG"],"magma"),("S_BG pixel",data["S_BG"],"magma"),("D pixel",data["D"],"coolwarm"),("UNI correct",data["uni_correct"],"viridis")]
    fig,axes=plt.subplots(2,5,figsize=(16,6))
    for ax,(name,array,cmap) in zip(axes.flat,panels):
        ax.imshow(array,cmap=cmap); ax.set(title=name,xticks=[],yticks=[])
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)
