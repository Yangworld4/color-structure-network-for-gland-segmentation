# Train-Only Top-K Memory Calibration

本实验仅使用 labeled training images，并采用 image-level leave-one-image-out。Validation/test 数据未读取，未使用 Dice/IoU/CPS 指标。

- labeled images: 7 (train_42.png, train_43.png, train_55.png, train_74.png, train_76.png, train_79.png, train_9.png)
- query tokens: 仅保留 FG occupancy ≥ 0.9 或 BG occupancy ≤ 0.1 的 native 16×16 tokens。
- candidate K: [1, 3, 5, 10, 20, 50]
- recreated selected memory count: FG=437, BG=355; file count: FG=437, BG=355

## All candidate results

| K | valid | AUROC | BAcc | FG Acc | BG Acc | Median Gap | Image BAcc mean±std | Image AUROC mean±std |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yes | 0.909238 | 0.861064 | 0.914286 | 0.807843 | 0.330102 | 0.887489±0.112398 | 0.916221±0.122270 |
| 3 | yes | 0.911826 | 0.867311 | 0.922857 | 0.811765 | 0.341301 | 0.890967±0.114520 | 0.922953±0.113671 |
| 5 | yes | 0.915050 | 0.869902 | 0.930000 | 0.809804 | 0.345431 | 0.892846±0.114889 | 0.927828±0.105690 |
| 10 | yes | 0.922459 | 0.866064 | 0.924286 | 0.807843 | 0.351963 | 0.890790±0.118123 | 0.936647±0.091601 |
| 20 | yes | 0.935392 | 0.872227 | 0.938571 | 0.805882 | 0.364567 | 0.896512±0.120203 | 0.946694±0.077871 |
| 50 | yes | 0.951922 | 0.876695 | 0.951429 | 0.801961 | 0.370569 | 0.899981±0.119064 | 0.959606±0.063855 |

## Selection rule and recommendation

按照预注册规则，最高 AUROC 为 0.951922，AUROC 容差池为 K=[50]；随后以 BAcc 容差池 K=[50] 继续筛选。最后比较 MedianGap，并在等价时优先 image-level 稳定性和更小 K。
因此推荐 **K* = 50**。推荐指标：AUROC=0.951922，BAcc=0.876695，FG Acc=0.951429，BG Acc=0.801961，MedianGap=0.370569。

Recommended K for all subsequent validation/test/CPS experiments: **K = 50**.
该 K 只由 labeled training data 的 image-level LOO calibration 选择；validation 和 test 没有用于 K selection。

## K behavior

K=1 的 pooled AUROC/BAcc 为 0.909238/0.861064，image BAcc std 为 0.112398；这反映单最近邻是否表现出更大的 image-level 波动。
最大 K=50 的 pooled AUROC/BAcc 为 0.951922/0.876695；与中间 K 的差异用于判断大邻域平均是否稀释语义证据。
满足相邻 AUROC 与 BAcc 差异均 <0.01 的稳定相邻区间：[[1, 3], [3, 5], [5, 10]]。
若某一 K 在数值上优于其它候选，报告该结果本身，不将其解释为 validation/test 性能。

## Outputs

- `topk_results.csv`：所有候选 K 的 pooled 与 image-level 统计。
- `topk_per_image_results.csv`：每个 LOO image、每个 K 的结果。
- `selected_k_token_records.csv`：推荐 K 的全部 LOO train query token records。
- `summary.json`：选择规则、数据来源和泄漏声明。
