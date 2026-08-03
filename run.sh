#!/bin/bash

# 开启错误检测：如果其中某一步程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate cg-seg

echo "=== 开始运行实验 1 with complementarity ==="

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w_comp_seed-4 \
    --feature_loss_mode complementarity \
    --w_comp 0.01 \
    --sd 4

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w_comp_seed-14 \
    --feature_loss_mode complementarity \
    --w_comp 0.01 \
    --sd 14

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w_comp_seed-24 \
    --feature_loss_mode complementarity \
    --w_comp 0.01 \
    --sd 24

echo "=== 开始运行实验 2 without complementarity ==="

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w-o_comp_seed-4 \
    --feature_loss_mode complementarity \
    --w_comp 0 \
    --sd 4

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w-o_comp_seed-14 \
    --feature_loss_mode complementarity \
    --w_comp 0 \
    --sd 14

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_fdcl_10_w-o_comp_seed-24 \
    --feature_loss_mode complementarity \
    --w_comp 0 \
    --sd 24

echo "=== 所有实验已全部完成！ ==="
