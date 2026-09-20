# UNI reference-memory diagnostic

Use this closed, offline diagnostic to inspect whether the fixed UNI-256
foreground and background memory banks are separable. It produces a t-SNE
visualization plus a PCA companion and quantitative diagnostics in the
original normalized feature space.

```bash
python analysis/uni_memory_diagnostic/run_diagnostic.py \
  --memory-path checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz \
  --output-dir outputs/uni_memory_diagnostic/glas_reference_memory_uni256
```

No training, image loading, labels, or memory reconstruction is involved.
The t-SNE plot is descriptive; cosine distributions and the centroid
classifier are the primary separation checks.
