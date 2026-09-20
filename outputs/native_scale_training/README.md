# Native-scale training outputs

Formal runs were executed with the locked standard CPS configuration (GLaS,
10% labeled, seed 24, RGB/stain, 200 epochs, 17 validation images).  The
vanilla row in `comparison_summary.csv` is the existing standard-CPS E0/E1
reference; Native-S and Native-SC rows are the two new formal runs.  No test
split is instantiated during training.

The four per-mode CSV streams record training native intervention statistics,
pixel shared-error validation statistics, native token diagnostics, and
connected-component event counts.
