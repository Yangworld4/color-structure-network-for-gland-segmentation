# K=5 vs K=50 UNI Memory Diagnostic

K=50 已由 labeled training image-level LOO calibration 预先选定。本诊断只比较 K=5 与 K=50，不重新选择 K；未使用 test。

## Semantic reference quality

| Metric | K=5 | K=50 | Δ(50−5) |
|---|---:|---:|---:|
| auroc | 0.969335 | 0.983820 | 0.014485 |
| balanced_accuracy | 0.929967 | 0.940839 | 0.010873 |
| fg_accuracy | 0.951220 | 0.966741 | 0.015521 |
| bg_accuracy | 0.908714 | 0.914938 | 0.006224 |
| overall_accuracy | 0.926242 | 0.936300 | 0.010058 |
| median_gap | 0.327118 | 0.349823 | 0.022705 |

## CPS contradiction diagnostic

| Metric | K=5 | K=50 | Δ(50−5) |
|---|---:|---:|---:|
| Contradictions | 15776.000000 | 15420.000000 | -356.000000 |
| Contradiction rate | 0.725000 | 0.708640 | -0.016360 |
| P(WW|contradiction) | 0.144374 | 0.149652 | 0.005279 |
| P(HC90WW|contradiction) | 0.088910 | 0.091995 | 0.003085 |
| P(CC|contradiction) | 0.090688 | 0.084928 | -0.005760 |
| P(WW|D<=0) | 0.060746 | 0.058599 | -0.002147 |
| P(HC90WW|D<=0) | 0.040483 | 0.039299 | -0.001184 |
| P(CC|D<=0) | 0.914683 | 0.917091 | 0.002407 |
| WW enrichment | 2.376681 | 2.553847 | 0.177167 |
| HC90WW enrichment | 2.196208 | 2.340883 | 0.144675 |
| Direction accuracy | 0.896935 | 0.903777 | 0.006842 |
| HC90 direction accuracy | 0.984883 | 0.986686 | 0.001804 |
| WW pixel coverage | 0.104671 | 0.106049 | 0.001379 |
| HC90WW pixel coverage | 0.050840 | 0.051708 | 0.000868 |
| CC exposure | 0.065749 | 0.060183 | -0.005566 |

## D scale

| Metric | K=5 | K=50 | Δ(50−5) |
|---|---:|---:|---:|
| N | 3944.000000 | 3855.000000 | -89.000000 |
| min | 0.000332 | 0.000210 | -0.000122 |
| max | 0.445486 | 0.353209 | -0.092277 |
| mean | 0.104092 | 0.087181 | -0.016910 |
| std | 0.092816 | 0.064241 | -0.028575 |
| p05 | 0.007465 | 0.007364 | -0.000101 |
| p10 | 0.015513 | 0.016081 | 0.000568 |
| p25 | 0.044237 | 0.035133 | -0.009104 |
| p50 | 0.081018 | 0.074583 | -0.006435 |
| p75 | 0.133884 | 0.126798 | -0.007086 |
| p90 | 0.208592 | 0.168644 | -0.039948 |
| p95 | 0.336602 | 0.208455 | -0.128147 |
| p99 | 0.431349 | 0.290734 | -0.140615 |

D median ratio K50/K5 = **0.920576**.
D p90 ratio K50/K5 = **0.808488**.

## Per-epoch D stability

K=5 max/min median = 1.109508; K=50 max/min median = 1.166392.
Epoch-level details are in `per_epoch_d_comparison.csv`.

## Diagnostic conclusion: Outcome A

Semantic AUROC change (K=50−K=5) = 0.014485; WW-enrichment change = 0.177167. K=50 remains the pre-selected configuration and was not re-selected using validation.

Does the validation diagnostic support K=50? **YES**.

## Scope

This diagnostic used the same GLaS validation set and Vanilla CPS milestones at epochs 40/80/120/160/200. No test set, training, optimizer, loss, memory rebuild, or K sweep was used. Event coverage was not added because no standalone event-coverage function was available without introducing a new definition.
