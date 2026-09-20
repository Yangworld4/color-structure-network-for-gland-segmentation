"""Small native-grid displays for E4; no pixel-level UNI pseudo-labels."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_token_case(rgb, data, title, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("RGB", rgb, None), ("CPS agreement", data["agreement"], "gray"),
        ("UNI contradiction", data["contradiction"], "magma"),
        ("WW ratio / token", data["WW_ratio"], "viridis"),
        ("HC90WW ratio / token", data["HC90WW_ratio"], "viridis"),
        ("GT occupancy", data["occupancy"], "viridis"),
        ("D", data["D"], "coolwarm"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (name, arr, cmap) in zip(axes.flat, panels):
        ax.imshow(arr, cmap=cmap); ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(panels):]: ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
