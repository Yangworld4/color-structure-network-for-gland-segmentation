# E4-D Adaptive Suppression Feasibility Diagnostic

Closed offline analysis of `outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv`. No model forward, training, memory rebuild, or threshold selection is performed.

D quartiles/deciles are formed using D only among agreement contradiction tokens (D>0). Ranking scores are descriptive and are not inserted into training. Bootstrap resamples image IDs, retaining all epochs for each sampled image, so token-level bootstrap is avoided.

Outcome rule is fixed in code: Outcome A requires positive global D-risk correlations, Q4>Q1 WW risk, D top-20 ranking above random, and at least half of valid images with positive D-vs-WW rho. Outcome B is used when those conditions hold but D×confidence improves top-20 WW precision by >0.02 over D-only; otherwise Outcome C.

Result: **Outcome A — D is a useful continuous contradiction-reliability signal.**
