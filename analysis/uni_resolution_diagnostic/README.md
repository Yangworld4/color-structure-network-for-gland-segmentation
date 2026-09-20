# E3 UNI-224 versus UNI-256 diagnostic

This is a closed, offline, no-training comparison using the same standard CPS
checkpoints, validation split, GT, and CPS states as E0/E1. UNI-224 uses the
existing 224x224 preprocessing and 14x14 native tokens. UNI-256 receives the
aligned 256x256 RGB directly (no 256->224 resize) with the same ImageNet
normalization and produces 16x16 tokens through the existing dynamic-image-size
UNI ViT-L/16 backbone. Both memories use the identical `labeled_10.csv`
source, occupancy thresholds, top-K, normalized cosine memory construction,
sampling seed, and per-image memory cap; only the native grid/resolution
changes.

Both native margins are projected with the current project path:
`main.py::_build_reference_weights`, margin-first bilinear interpolation with
`align_corners=False`, to the fixed 256x256 CPS resolution. No edge-aware
propagation, trainable component, R, CPS loss, or test split is used.
