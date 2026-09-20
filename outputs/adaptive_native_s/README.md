# Adaptive Native-S formal comparison

This directory summarizes the fixed `native_suppress` and formal 200-epoch `native_suppress_adaptive` runs. Adaptive D values came only from the detached positive contradiction tokens of the unlabeled training batches; validation and E4/E4-D outputs were not used to build the CDF.

The adaptive buffer is FIFO (4096), with a 512-value warm-up and `w=1-0.5*empirical_CDF(D)`. The final comparison is validation-only; test was not evaluated during training.

The result is recorded in `summary.json`.
