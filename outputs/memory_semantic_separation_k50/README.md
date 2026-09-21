# UNI FG/BG Memory Semantic Separation

    Input: `/data/mission/models/Exp/DUAL/outputs/native_scale_reference/glas_standard_cps_uni256/token_records_k50.csv`

Retrieval Top-K: `50`

The figure uses the existing validation token records and computes `Delta = S_FG - S_BG`.
No model forward, memory rebuild, training, or CPS modification is performed.

GT class source: `direct gt class`.

The main population contains all valid pure FG/BG validation tokens according to the
existing direct class or 0.9/0.1 occupancy rule. It is not filtered by margin sign,
CPS agreement, or confidence. The dashed vertical line at `Delta=0` is the semantic
decision boundary.

Complete margin minimum/maximum are stored in `summary.json`. The plot display uses
the 0.5--99.5 percentile range only to prevent a few extreme outliers from compressing
the main distribution; no rows are removed from the statistics.

Outputs:

- `memory_margin_distribution.png` (300 dpi)
- `memory_margin_distribution.pdf`
- `memory_margin_statistics.csv`
- `memory_margin_supplementary.csv`
- `summary.json`
