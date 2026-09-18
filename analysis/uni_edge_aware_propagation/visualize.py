"""Compact E2 comparison plots and case displays."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_case(rgb, data, margins, title, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    panels=[("RGB",rgb,None),("GT",data["gt"],"gray"),
            ("CPS consensus",data["hard_A"],"gray"),("WW/HC90WW",data["WW"].astype(float)+2*data["HC90WW"],"magma"),
            ("native margin",data["native_margin"],"coolwarm"),
            ("Bilinear",margins["bilinear"],"coolwarm"),
            ("Guided",margins["guided"],"coolwarm"),
            ("JBU",margins["jbu"],"coolwarm"),
            ("Guided correct",(margins["guided"]>0)==data["gt"],"viridis"),
            ("JBU correct",(margins["jbu"]>0)==data["gt"],"viridis")]
    fig, axes = plt.subplots(2,5, figsize=(16,6))
    for ax,(name,array,cmap) in zip(axes.flat,panels):
        ax.imshow(array,cmap=cmap); ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)
