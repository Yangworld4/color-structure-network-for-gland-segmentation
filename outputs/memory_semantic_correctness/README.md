# Memory Bank Semantic Correctness

Input records: `/data/mission/models/Exp/DUAL/outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv`

Memory prediction is defined by `Delta = S_FG - S_BG`: positive predicts FG,
negative predicts BG, and exact zero is recorded as a tie. Ties remain in the
class denominators but are excluded from correctness numerators.

Results are computed on existing token-level validation records only. No model
forward, memory rebuild, CPS run, or training is performed by this Figure 3 script.

The main figure reports foreground accuracy, background accuracy, and balanced
accuracy for all valid validation tokens. It supports only native-scale semantic
validity of the memory bank; it does not claim dense pixel-label accuracy or CPS
confirmation-bias resolution.


