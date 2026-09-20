# D-Adaptive Native-S

Formal 200-epoch run, seed 24. No forward pass is performed by this postprocessor.

Formula: `w=1` for disagreement or non-contradiction; `w=max(0, 0.5-D)` for `D>0` agreement tokens.

Outcome: **Outcome A** — D-dependent extra suppression improves Fixed Native-S.

Best mean-Dice epochs: Fixed=191, D-Adaptive=142.

The main comparison is `fixed_vs_d_adaptive_selected.csv`; raw training and evaluator exports are retained alongside it.
