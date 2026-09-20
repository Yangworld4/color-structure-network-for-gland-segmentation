# Per-Epoch Contradiction D Scale Diagnostic

Input: `/data/mission/models/Exp/DUAL/outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv`

Filter: `agreement == true`, `contradiction == true`, `D > 0`; epochs `[40, 80, 120, 160, 200]`. No model, UNI, CPS, or reference-memory forward was run.

Selected records: `3931` / `21760`.

Outcome: **A**

Outcome A — Raw D has a relatively stable numerical scale across training epochs. A fixed D-to-weight mapping is plausible.
