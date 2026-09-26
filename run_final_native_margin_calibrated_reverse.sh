#!/bin/bash

# 开启错误检测：如果其中某一步程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate cg-seg

echo "=== 开始运行实验 ==="


python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GlaS \
    --path_trained_models checkpoints/semi_20_seed_1337_rgbstain \
    --input_mode rgb_stain \
    --labeled_ratio 20 \
    --sd 1337

python main.py \
    --dataset_name CRAG      \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/semi_20_seed_1337_rgbstain \
    --input_mode rgb_stain \
    --labeled_ratio 20 \
    --sd 1337

