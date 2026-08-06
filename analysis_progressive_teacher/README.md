# Progressive Teacher 离线可行性分析

该目录只做验证集离线推理与统计。代码不会创建 optimizer、调用 backward、更新模型参数、保存模型或修改 checkpoint。模型始终使用 `eval()` 和 `torch.inference_mode()`；所有结果只写入 `--output-dir`。

## 当前项目适配结论

- 两个 Student 均通过 `models.getnetwork.get_network()` 独立构建。
- Appearance 使用 dataset 返回的 `image_appearance`（RGB 标准化视图）。
- Structure 使用严格空间对齐的 `image_structure`（Hematoxylin、Gray、Sobel）。验证阶段传入 `training=False`，只有固定的 256×256 Resize，没有随机翻转或裁剪。
- `XNet_sb` forward 返回 `(logits, feature)`；脚本取第一个元素。脚本也安全支持直接 Tensor、list 和常见 dict logits。
- mask 在原 dataset 中由 0/255 映射为 0/1。
- 原训练验证对每个模型执行 `softmax(logits)[:,1]`，在 `0.00, 0.02, ..., 0.88` 中选验证 Jaccard 最优阈值。脚本默认以 `--threshold-mode auto` 对每个 checkpoint 重现该口径。
- GLaS `paper_seed_2026/validation.csv` 当前包含 17 张图（CSV 含 1 行表头），test 包含 80 张图。
- 当前示例 checkpoint 是直接 `state_dict`。加载时会检查全部 missing/unexpected/shape mismatch，任何不匹配均直接失败，不会使用 `strict=False` 忽略参数。
- 当前 `periodic_best/epoch_XXXX_best_A/S...` 是“截至 epoch XXXX 的历史最佳快照”，并非 epoch XXXX 当轮模型。因此 Leader Dynamics 和 Complementarity Over Time 是 **best-so-far milestone trajectory**。如果需要真实瞬时训练动态，训练阶段必须另行保留当轮权重；本分析不会修改训练代码补做此事。
- 当前 `semi_10_seed/GLaS/...` 示例权重与 `unet` 完全匹配，而非当前 CLI 默认的 `xnet_sb`，分析该组权重时必须传 `--network unet`。

## 推荐运行方式

```bash
python analysis_progressive_teacher.py \
  --dataset-root datasets/GLaS \
  --dataset-name GLaS \
  --val-split val \
  --checkpoint-dir checkpoints/semi_10_seed/GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image \
  --best-a checkpoints/semi_10_seed/GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image/best_result1_Jc_0.7471.pth \
  --best-s checkpoints/semi_10_seed/GLaS/CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image/best_result2_Jc_0.7620.pth \
  --network unet \
  --epoch-step 10 \
  --boundary-width 3 \
  --gpu 0 \
  --output-dir analysis_results/progressive_teacher_validation
```

`--best-a/--best-s` 可省略，脚本会尝试自动寻找 `best_result1/2_Jc_*.pth`。CPU 静态/小规模验证可使用 `--gpu cpu`。使用 `--no-figures` 可跳过 matplotlib 绘图；使用 `--save-qualitative` 才会生成定性图。

## 自定义 checkpoint 清单

自动识别支持 epoch、A/S、appearance/structure、result1/result2 语义明确的文件名。文件名不规则或同一 epoch 有多个候选时，应传 JSON：

```json
[
  {"epoch": 10, "appearance": "checkpoints/A10.pth", "structure": "checkpoints/S10.pth"},
  {"epoch": 20, "appearance": "checkpoints/A20.pth", "structure": "checkpoints/S20.pth"}
]
```

相对路径以 JSON 所在目录解析：

```bash
python analysis_progressive_teacher.py ... --checkpoint-list checkpoint_pairs.json
```

也可用 `--checkpoint-pattern 'periodic_best/*.pth'` 限制自动扫描范围。发现缺失、重复或分支歧义时脚本会报错，不会静默跳过。

## 输出

- `checkpoint_inventory.json`：checkpoint—缓存—阈值的完整对应关系。
- `prediction_cache/*.npz`：按 checkpoint 身份缓存的二值预测；不写回原预测目录。
- `current_best_summary.json`：最终 A/S 指标和固定 Both-Wrong 定义。
- `historical_recovery_summary.json`、`historical_recovery_per_image.csv`：HRR 与 PBR。
- `both_wrong_temporal_analysis.npz`：首次/末次恢复 epoch、正确次数、recoverable/persistent mask。
- `cumulative_historical_recovery.csv`：chronological 和 reverse 两个方向的累计恢复。
- `oracle_analysis.json`：Current/Historical Oracle；全部是使用 GT 的不可部署理论上界。
- `leader_dynamics.csv`、`leader_dynamics_summary.json`：全局 leader 与 crossover。
- `expertise_dynamics.csv`、`expertise_dynamics_summary.json`：foreground/background/boundary/interior 的局部正确率和 disagreement winner。
- `complementarity_over_time.csv`：A-only、S-only、Both-Wrong、disagreement 和 A/S Oracle gain 的时间轨迹。
- `both_wrong_region_analysis.json`：历史可恢复与 persistent error 的区域分布。
- `bootstrap_ci.json`：以 image 为独立单位的 2000 次 bootstrap 95% CI。
- `figures/*.png`：累计 HRR、leader Dice/gap、complementarity over time。
- `qualitative/*.png`：可选的高 HRR 样本历史恢复图。
- `final_summary.json`、`final_summary.txt`：完整结果与克制的自动解释。

## 如何阅读

- `HRR_union` 高：最终两个 best 同时错误的位置中，较早 A/S 至少一次正确的比例较高，说明历史状态存在理论可利用信息；不代表 EMA 已能恢复它。
- `Leader crossover`：A/S 的全局 Dice leader 是否交换。对当前周期文件，它表示 best-so-far 轨迹的交叉，而非瞬时网络波动。
- `Expertise Dynamics`：在 A/S disagreement 内由 GT 事后判断 winner。若全局 leader 稳定但另一分支在某一区域长期有较高 win rate，更支持后续验证“稳定主 Teacher + 动态区域顾问”，而非仅靠全局 leader 切换。
- `Historical Oracle gain`：允许所有历史预测并用 GT 完美选分支的理论上界，只能衡量可利用空间，不能当作真实推理性能。
