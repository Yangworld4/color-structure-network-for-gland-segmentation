# E3 result bundle

Closed, offline, no-training UNI-224 versus UNI-256 diagnostic on the same
standard CPS checkpoints and 17 validation images. UNI-224 uses 224x224 input
and a 14x14 native grid; UNI-256 uses the aligned 256x256 RGB directly and a
16x16 native grid. Both use the same normalization, top-K cosine scoring,
memory sampling rules, `labeled_10.csv` source, and margin-first bilinear
projection to 256x256.

Outcome: B. UNI-256 changes native statistics, but produces only marginal
pixel/boundary changes, lowers MIXED-footprint arbitration, and has a negative
WW event net gain.
