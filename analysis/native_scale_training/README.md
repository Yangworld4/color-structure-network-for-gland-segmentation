# Native-scale Reference-Guided CPS minimal training

This directory documents the two additive training modes in `main.py`:

- `native_suppress`: detached 16x16 Student agreement/UNI contradiction gates
  reduce both original hard-pseudo CPS directions with nearest-expanded scalar
  loss weights (`gamma=0.5`).
- `native_suppress_hc`: the same suppression plus `lambda=0.5` BCE on the
  undetached pooled Student foreground probabilities for HC contradiction
  tokens (`conf_A/conf_B >= 0.9`) toward the frozen UNI-256 token decision.

Both modes use the standard CPS split and optimizer/network settings, UNI
256x256 input (16x16 native grid), and the fixed train-only memory
`checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz`.  GT is never
read by the training gate; it is used only by validation diagnostics.  Test is
deferred and is not instantiated or evaluated inside the training loop.

`compare_runs.py` only assembles the completed validation logs and previously
locked vanilla E0/E1 reference values; it does not train or tune anything.
