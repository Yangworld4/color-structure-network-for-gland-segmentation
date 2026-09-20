# E4 Raw-D Sigmoid Weight Diagnostic

Offline-only diagnostic over the existing E4 `token_records.csv`. Rows were filtered to `agreement=true`, `contradiction=true`, and `D>0`; no model, training, or forward pass was used. The tested mapping is `w=sigmoid(-D)` with no temperature, scale, normalization, buffer, or floor.

Conclusion: Outcome B — Raw D is too small in scale for sigmoid(-D) to provide meaningful dynamic differentiation. The mapping is mathematically clean but practically close to Fixed Native-S.
