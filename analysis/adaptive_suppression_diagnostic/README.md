# E4-D Adaptive Suppression Feasibility Diagnostic

Closed, offline analysis of the E4 native-scale `token_records.csv`. It tests
whether `D = S_opp - S_cur` is a continuous shared-error risk signal. No model
is loaded or trained, and no memory is rebuilt.

```bash
python analysis/adaptive_suppression_diagnostic/run_diagnostic.py \
  --token-records outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv \
  --output-dir outputs/adaptive_suppression_diagnostic/glas_standard_cps_uni256
```

Only agreement contradiction tokens (`D > 0`) form D quartiles/deciles.
Risk-ranking diagnostics use all agreement tokens and compare D-only,
confidence-only, D×confidence, and matched random selections. All bootstrap
intervals resample image IDs, not individual tokens.
