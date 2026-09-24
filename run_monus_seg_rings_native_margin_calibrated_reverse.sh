#!/bin/bash

# MoNuSeg 与 RINGS 的最终 Native-Scale Margin-Calibrated-Reverse 实验
set -e

cd /data/mission/models/Exp/DUAL
source /data/software/miniconda3/etc/profile.d/conda.sh
conda activate cg-seg

echo "=== 开始运行 MoNuSeg 与 RINGS 实验 ==="

run_experiment() {
    local dataset="$1"
    local labeled_ratio="$2"
    local seed="$3"
    local model_dir="$4"
    local uni_k="${5:-50}"

    python main.py \
        --dataset_name "$dataset" \
        --path_dataset "datasets/$dataset" \
        --path_trained_models "checkpoints/$model_dir" \
        --path_seg_results "seg_pred/$model_dir" \
        --labeled_ratio "$labeled_ratio" \
        --sd "$seed" \
        --batch_size 2 \
        --num_epochs 200 \
        --network unet \
        --input_mode rgb_stain \
        --step_size 50 \
        --lr 0.5 \
        --gamma 0.5 \
        --unsup_weight 5.0 \
        --warm_up_duration 20 \
        --momentum 0.9 \
        --cps_mode native_fine_margin_calibrated_reverse \
        --uni_k "$uni_k" \
        --uni_build_reference_memory
}

echo "=== MoNuSeg：开始运行 10% 标注实验 ==="
# MoNuSeg 10% 只有 3 张 labeled 图；image-level LOO 后无法支持 K=50。
# K=5 保持 train-only LOO 校准协议，并使用当前任务目录独立保存校准文件。
run_experiment MoNuSeg 10 24 k50_margin_calibrated_reverse 5
run_experiment MoNuSeg 10 42 k50_margin_calibrated_reverse_42 5
run_experiment MoNuSeg 10 1337 k50_margin_calibrated_reverse_1337 5

echo "=== MoNuSeg：开始运行 20% 标注实验 ==="
run_experiment MoNuSeg 20 24 k50_margin_calibrated_reverse_20
run_experiment MoNuSeg 20 42 k50_margin_calibrated_reverse_20_42
run_experiment MoNuSeg 20 1337 k50_margin_calibrated_reverse_20_1337

echo "=== RINGS：开始运行 10% 标注实验 ==="
run_experiment RINGS 10 24 k50_margin_calibrated_reverse
run_experiment RINGS 10 42 k50_margin_calibrated_reverse
run_experiment RINGS 10 1337 k50_margin_calibrated_reverse

echo "=== RINGS：开始运行 20% 标注实验 ==="
run_experiment RINGS 20 24 k50_margin_calibrated_reverse_20
run_experiment RINGS 20 42 k50_margin_calibrated_reverse_20_42
run_experiment RINGS 20 1337 k50_margin_calibrated_reverse_20_1337

echo "=== MoNuSeg 与 RINGS 实验全部完成 ==="
