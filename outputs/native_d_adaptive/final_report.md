# D-Adaptive Native-S 离线报告

## 实验

- 模式：`native_suppress_d_adaptive`
- 公式：无 contradiction 或 Student disagreement 时 `w=1`；agreement 且 `D>0` 时 `w=max(0, 0.5-D)`。
- 训练：seed 24，正式 200 epoch；Fixed Native-S 路径未修改。
- CDF、running buffer、percentile、额外超参数和 GT weighting：均未使用。
- 结果判定采用 matched epoch 200；best-Dice epoch 仅作为独立报告，避免不同 checkpoint 选择造成混淆。

## Epoch 200 对照

| 指标 | Fixed Native-S | D-Adaptive Native-S | Δ |
|---|---:|---:|---:|
| Dice | 0.881968 | 0.885392 | +0.003424 |
| IoU | 0.788863 | 0.794353 | +0.005490 |
| WW | 0.098158 | 0.092941 | -0.005217 |
| HC90WW | 0.074629 | 0.070882 | -0.003746 |
| CC | 0.888579 | 0.889531 | +0.000951 |

Fixed best mean-Dice epoch is **191** (0.883458); D-Adaptive best mean-Dice epoch is **142** (0.888632).

## Weight diagnostic at epoch 200

`D` mean/p10/p50/p90 = `0.096403 / 0.008701 / 0.062236 / 0.231647`.
Contradiction weight mean/p10/p50/p90 = `0.403597 / 0.268353 / 0.437764 / 0.491299`.
Zero-weight fraction = `0.000000`. D-to-weight quartile monotonicity = `True`.

## 结论

**Outcome A** — D-dependent extra suppression improves Fixed Native-S.

详细逐 epoch 数据见 `fixed_vs_d_adaptive.csv`、`training_log.csv`、`validation_metrics.csv` 和 `weight_statistics.csv`。
