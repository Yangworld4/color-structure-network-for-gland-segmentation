# E1 Native-token → Pixel Arbitration Degradation

E1 is fully offline. It uses the E0 native-token records as the immutable
UNI evidence baseline, loads only frozen **standard CPS** checkpoints for
validation inference, and does not change CPS, UNI, memory, top-K, similarity,
margin, `R`, or any model parameter.

## Projection faithfully reused from training

Source: `main.py::_build_reference_weights`.

```python
margin = F.interpolate(
    margin_tokens.reshape(-1, 1, actual_grid, actual_grid),
    size=logits_a.shape[-2:], mode="bilinear", align_corners=False,
)[:, 0]
```

Thus the primary E1 field is margin-first bilinear interpolation from native
14×14 to the existing segmentation output resolution (256×256). E1 also
projects `S_FG` and `S_BG` separately only as a linear-consistency check; it
does not replace the training pipeline. The measured max difference between
the primary projected margin and `interp(S_FG)-interp(S_BG)` is recorded.

Pixel CPS state is always defined from original 256×256 A/B soft probabilities,
not from token states. Native token types are only used to partition pixel
footprints for the mixed-vs-pure diagnostic. Pixel records are compressed NPZ
per image/epoch; `pixel_records.csv` is their manifest.
