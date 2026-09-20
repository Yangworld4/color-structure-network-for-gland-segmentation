# UNI-256 Reference Memory Diagnostic

This is a closed, offline diagnostic of `checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz`. It reads only the
stored foreground/background memory vectors; no image, GT, model training, or
memory rebuilding is performed.

## Memory

- Foreground vectors: `437`
- Background vectors: `355`
- Feature dimension: `1024`
- NPZ metadata: `{"anchor_bg": 0.1, "anchor_fallback_count": 0, "anchor_fallback_fraction": 0.1, "anchor_fg": 0.9, "feature_dim": 1024, "memory_bg_count": 355, "memory_fg_count": 437, "memory_per_image_class": 64, "reference_resolution": 256, "seed": 2026, "source_dataset": "GlaS", "source_labeled_ratio": 10, "token_grid": 16, "top_k": 5}`

## Outputs

- `memory_tsne.png`: qualitative t-SNE view (PCA-50 initialization).
- `memory_pca.png`: linear PCA view.
- `cosine_metrics.csv`: original feature-space within/cross class similarities.
- `summary.json`: all metadata and quantitative diagnostics.

The primary separation evidence is cosine similarity and nearest class
centroid performance in the original normalized feature space. t-SNE can make
clusters visually apparent but can also create or hide apparent gaps, so it is
not used as a standalone proof of discriminability.
