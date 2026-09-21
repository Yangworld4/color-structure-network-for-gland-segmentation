# Anti-Confirmation Budget Gradient Diagnostic

## Protocol

- K=50; UNI frozen; existing k=50 memory; existing native fine-selective risky mask.
- Checkpoints: epoch 40, 120, 200; four fixed training batches each.
- Training performed: false. Optimizer step performed: false.
- Validation/test were not used for q calibration.
- `F_train` was collected once from epoch-200 frozen model on the same four unlabeled training batches, using positive contradiction-token D only.

## Q1 — q distribution

Risk fraction: `0.07014211`

| statistic | value |
|---|---:|
| mean | 0.58049157 |
| p05 | 0.05517241 |
| p10 | 0.13103448 |
| p25 | 0.35172415 |
| p50 | 0.63448274 |
| p75 | 0.82758617 |
| p90 | 0.94482756 |
| p95 | 0.98620689 |
| q < 0.10 | 0.07802473 |
| q > 0.90 | 0.15614916 |
| q < 0.05 | 0.04767775 |
| q > 0.95 | 0.09642508 |

`WARNING_Q_COLLAPSE`: `False`.  q is concentrated at the extremes under the requested 80% rule.

## Q2 — pure intervention-budget scaling

- Isolated risky loss B/A mean: `2.00000000`.
- Isolated risky gradient norm B/A mean: Student A `2.00000000`, Student B `2.00000000`.
- Isolated risky gradient cosine mean: Student A `1.00000000`, Student B `1.00000000`.

These values test whether beta=1 is only a positive scalar budget change on identical q/risk pixels.

## Q3 — beta=1 numerical behavior

- Full total-gradient B/A mean: Student A `1.85280247`, Student B `1.87622108`, combined `1.82905518`.
- Max-gradient values and batch-level details are in `batch_level_diagnostic.csv` and `gradient_summary.csv`.
- NaN/Inf: `no`.
- Flags: `none`.

## Q4 — risky intervention relative scale

- Risk-gradient / total-gradient, beta=0.5: `0.45757169`.
- Risk-gradient / total-gradient, beta=1.0: `0.48720822`.

## Q5 — formal-training gate

`PASS`: 两个公式实现正确，q 正常，beta=1 无明显数值异常，可以进入正式训练比较。

This diagnostic does **not** determine which beta gives better segmentation. A full training comparison is required for final performance judgment.
