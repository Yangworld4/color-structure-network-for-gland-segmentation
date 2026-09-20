#!/usr/bin/env python3
"""Offline visualization and separation diagnostic for a UNI reference memory.

This script only reads the stored FG/BG memory vectors.  It does not load a
dataset, train a model, rebuild the memory, or use evaluation labels.  The
quantitative separation statistics are computed in the original feature
space; t-SNE/PCA are visualization aids only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, silhouette_score


def _scalar(z: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in z.files:
        return default
    value = z[key]
    return value.item() if getattr(value, "ndim", 0) == 0 else value


def _pairwise_values(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Return off-diagonal or cross-class cosine similarities."""
    if b is None:
        sim = a @ a.T
        tri = np.triu_indices(sim.shape[0], k=1)
        return sim[tri]
    return (a @ b.T).reshape(-1)


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else None,
        "std": float(np.std(values)) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p05": float(np.percentile(values, 5)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
    }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_metrics(path: Path, rows: list[dict]):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-path",
        default="checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/uni_memory_diagnostic/glas_reference_memory_uni256",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1500)
    args = parser.parse_args()

    memory_path = Path(args.memory_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(memory_path, allow_pickle=True) as z:
        if "memory_fg" not in z.files or "memory_bg" not in z.files:
            raise RuntimeError("reference memory must contain memory_fg and memory_bg")
        fg = np.asarray(z["memory_fg"], dtype=np.float64)
        bg = np.asarray(z["memory_bg"], dtype=np.float64)
        metadata = {k: _scalar(z, k) for k in z.files if k not in {"memory_fg", "memory_bg"}}

    if fg.ndim != 2 or bg.ndim != 2 or fg.shape[1] != bg.shape[1]:
        raise RuntimeError(f"invalid memory shapes: FG={fg.shape}, BG={bg.shape}")
    feature_dim = int(fg.shape[1])
    # Cosine statistics require unit vectors.  Keep the original data intact,
    # but normalize a copy so this remains valid for slightly different NPZs.
    fg_norms = np.linalg.norm(fg, axis=1)
    bg_norms = np.linalg.norm(bg, axis=1)
    fg = fg / np.clip(fg_norms[:, None], 1e-12, None)
    bg = bg / np.clip(bg_norms[:, None], 1e-12, None)
    X = np.concatenate([fg, bg], axis=0)
    labels = np.concatenate([np.zeros(len(fg), dtype=int), np.ones(len(bg), dtype=int)])

    # Original-space quantitative separation.
    fg_within = _pairwise_values(fg)
    bg_within = _pairwise_values(bg)
    cross = _pairwise_values(fg, bg)
    fg_centroid = fg.mean(axis=0)
    bg_centroid = bg.mean(axis=0)
    fg_centroid /= np.linalg.norm(fg_centroid)
    bg_centroid /= np.linalg.norm(bg_centroid)
    centroid_sim_fg = fg @ fg_centroid
    centroid_sim_bg = bg @ bg_centroid
    centroid_margin = np.concatenate([
        centroid_sim_fg - fg @ bg_centroid,
        bg @ bg_centroid - bg @ fg_centroid,
    ])
    centroid_pred = (X @ bg_centroid > X @ fg_centroid).astype(int)
    centroid_acc = float(np.mean(centroid_pred == labels))
    centroid_balanced = float(balanced_accuracy_score(labels, centroid_pred))
    original_silhouette = float(silhouette_score(X, labels, metric="cosine"))

    # PCA is deterministic and provides a less distorted linear companion.
    pca_components = min(50, feature_dim, len(X) - 1)
    pca = PCA(n_components=pca_components, random_state=args.seed)
    X_pca = pca.fit_transform(X)
    pca_2d = X_pca[:, :2]
    pca_silhouette = float(silhouette_score(X_pca, labels, metric="euclidean"))

    # t-SNE is only for a qualitative 2-D view.  PCA initialization improves
    # repeatability and denoises the very high-dimensional memory vectors.
    max_perplexity = max(2.0, (len(X) - 1) / 3.0)
    perplexity = min(float(args.perplexity), max_perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate=200.0,
        n_iter=args.max_iter,
        random_state=args.seed,
        verbose=0,
    )
    tsne_2d = tsne.fit_transform(X_pca)

    def scatter(points: np.ndarray, filename: str, title: str, subtitle: str):
        fig, ax = plt.subplots(figsize=(8.4, 6.8), dpi=160)
        n_fg = len(fg)
        ax.scatter(points[:n_fg, 0], points[:n_fg, 1], s=17, alpha=0.62,
                   c="#d62728", label=f"Foreground (n={n_fg})", edgecolors="none")
        ax.scatter(points[n_fg:, 0], points[n_fg:, 1], s=17, alpha=0.62,
                   c="#1f77b4", label=f"Background (n={len(bg)})", edgecolors="none")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("dimension 1")
        ax.set_ylabel("dimension 2")
        ax.legend(frameon=True)
        ax.text(0.01, -0.12, subtitle, transform=ax.transAxes, fontsize=9,
                color="#444444", va="top")
        fig.tight_layout()
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)

    scatter(
        tsne_2d,
        "memory_tsne.png",
        "UNI-256 reference memory: t-SNE",
        "Visualization only; quantitative separation is measured in original cosine feature space.",
    )
    scatter(
        pca_2d,
        "memory_pca.png",
        "UNI-256 reference memory: PCA",
        f"Explained variance (PC1+PC2): {100 * pca.explained_variance_ratio_[:2].sum():.2f}%.",
    )

    rows = [
        {"metric": "within_fg_cosine", **_stats(fg_within)},
        {"metric": "within_bg_cosine", **_stats(bg_within)},
        {"metric": "cross_fg_bg_cosine", **_stats(cross)},
        {"metric": "fg_centroid_margin", **_stats(centroid_margin[:len(fg)])},
        {"metric": "bg_centroid_margin", **_stats(centroid_margin[len(fg):])},
    ]
    _write_metrics(out / "cosine_metrics.csv", rows)

    summary = {
        "memory_path": str(memory_path),
        "output_dir": str(out),
        "feature_dim": feature_dim,
        "num_foreground": int(len(fg)),
        "num_background": int(len(bg)),
        "num_total": int(len(X)),
        "metadata": metadata,
        "raw_norms": {
            "fg_mean": float(np.mean(fg_norms)),
            "fg_std": float(np.std(fg_norms)),
            "bg_mean": float(np.mean(bg_norms)),
            "bg_std": float(np.std(bg_norms)),
        },
        "cosine": {
            "within_fg": _stats(fg_within),
            "within_bg": _stats(bg_within),
            "cross_fg_bg": _stats(cross),
            "centroid_fg_margin": _stats(centroid_margin[:len(fg)]),
            "centroid_bg_margin": _stats(centroid_margin[len(fg):]),
            "centroid_accuracy": centroid_acc,
            "centroid_balanced_accuracy": centroid_balanced,
            "original_space_silhouette_cosine": original_silhouette,
        },
        "pca": {
            "components": int(pca_components),
            "explained_variance_ratio_first2": float(pca.explained_variance_ratio_[:2].sum()),
            "silhouette_euclidean": pca_silhouette,
        },
        "tsne": {
            "perplexity": perplexity,
            "max_iter": args.max_iter,
            "random_state": args.seed,
            "interpretation": "qualitative visualization; do not treat t-SNE distances as metric evidence",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")

    readme = f"""# UNI-256 Reference Memory Diagnostic

This is a closed, offline diagnostic of `{memory_path}`. It reads only the
stored foreground/background memory vectors; no image, GT, model training, or
memory rebuilding is performed.

## Memory

- Foreground vectors: `{len(fg)}`
- Background vectors: `{len(bg)}`
- Feature dimension: `{feature_dim}`
- NPZ metadata: `{json.dumps(metadata, sort_keys=True, default=_json_default)}`

## Outputs

- `memory_tsne.png`: qualitative t-SNE view (PCA-50 initialization).
- `memory_pca.png`: linear PCA view.
- `cosine_metrics.csv`: original feature-space within/cross class similarities.
- `summary.json`: all metadata and quantitative diagnostics.

The primary separation evidence is cosine similarity and nearest class
centroid performance in the original normalized feature space. t-SNE can make
clusters visually apparent but can also create or hide apparent gaps, so it is
not used as a standalone proof of discriminability.
"""
    (out / "README.md").write_text(readme)

    print("=== UNI Reference Memory Diagnostic ===")
    print(f"Foreground vectors: {len(fg)}")
    print(f"Background vectors: {len(bg)}")
    print(f"Feature dim: {feature_dim}")
    print(f"Centroid accuracy / balanced: {centroid_acc:.4f} / {centroid_balanced:.4f}")
    print(f"Within FG cosine: {np.mean(fg_within):.4f} | within BG: {np.mean(bg_within):.4f}")
    print(f"Cross FG-BG cosine: {np.mean(cross):.4f}")
    print(f"Original-space cosine silhouette: {original_silhouette:.4f}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
