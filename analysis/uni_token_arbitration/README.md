# UNI Native Token-Level Arbitration Diagnostic

This is a closed, offline diagnostic. It never changes CPS, trains no model,
does not modify reference memory, computes no `R`, and never interpolates a
UNI score/margin map for a metric.

## Project-faithful reference implementation

- **UNI model:** frozen local UNI `vit_large_patch16_224` (`models/uni.py`).
- **Input resolution:** 224×224, reconstructed from the project-normalized RGB
  validation tensor and resized bilinearly exactly as `main._uni_dense_features`.
- **Patch size / native grid / dimension:** 16×16 / dynamically obtained from
  the token output (the current UNI is 14×14) / 1024.
- **FG memory / BG memory:** immutable L2-normalized token banks loaded from
  the existing `.npz`; their source-labeled manifest is inferred from metadata
  and checked disjoint from evaluation images.
- **Top-K / scores:** exactly `main._topk_scores`: cosine similarity via
  normalized-token dot product, then mean of the largest `K` values.
  `S_FG=mean(topK(query @ M_FG.T))`; `S_BG` is identical using `M_BG`.
- **Margin:** `S_FG-S_BG`, exactly the project definition.

GT is transformed through the same evaluation resize and UNI 224 resize,
then token occupancy is computed by `adaptive_avg_pool2d`. Pure FG/BG use
occupancy thresholds 0.9 / 0.1; mixed tokens are excluded from primary
arbitration accuracy. CPS probabilities are adaptively averaged directly to
the native token grid before thresholding/confidence.

## Example

```bash
source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate cg-seg
python analysis/uni_token_arbitration/run_diagnostic.py \
  --dataset-name GlaS --dataset-root datasets/GLaS \
  --checkpoint-dir checkpoints/.../CPS-...-cps_mode=uni_reference_state_aware \
  --memory-root checkpoints/glas_labeled10_seed24/reference_memory.npz \
  --epochs-list 40 80 120 160 200 --run-name glas_uni_native_token
```

The default is leakage-safe and stops if evaluation IDs overlap the inferred
memory-source manifest. `--allow-memory-leak-debug` exists only for code
debugging and labels output invalid for scientific conclusions.
