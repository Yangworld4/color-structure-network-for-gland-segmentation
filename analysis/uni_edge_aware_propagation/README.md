# E2 Edge-Aware Coarse-to-Fine Propagation Diagnostic

This is a closed, offline, no-training comparison of three projection
operators for the same frozen UNI native margin and the same standard CPS
predictions:

* `bilinear`: the exact `main.py::_build_reference_weights` margin-first
  `F.interpolate(..., mode="bilinear", align_corners=False)` path;
* `guided`: a standard color guided filter with aligned RGB in `[0,1]` as
  guidance and the bilinear field as input;
* `jbu`: non-learning joint bilateral upsampling from the 14x14 margin. Its
  spatial term is in target pixels, its range term is normalized RGB, and its
  token means are computed from the RGB footprint of each native token.

Defaults are fixed before evaluation and are not selected using validation
labels: guided radius 8, epsilon 1e-3; JBU radius 2 native cells, spatial
sigma 32 pixels, range sigma 0.12. RGB is the spatially aligned image saved by
E1 in `[0,1]`; GT and CPS predictions are used only for evaluation masks.

The script first checks the stored E1 bilinear pixel metrics. A mismatch raises
`E1 BASELINE REPRODUCTION FAILED` and no comparison is produced.
