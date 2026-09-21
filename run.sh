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
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_10_seed_24_rgbstain_dis_piexl_entropy \
    --labeled_ratio 10 \
    --sd 24


python main.py \
    --dataset_name RINGS      \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/semi_10_seed_24_rgb\
    --labeled_ratio 10 \
    --input_mode rgb_rgb \
    --sd 24

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_10_seed_24_rgbstain\
    --labeled_ratio 10 \
    --input_mode rgb_stain \
    --sd 24

# python main.py \
#     --dataset_name CRAG      \
#     --path_dataset datasets/CRAG \
#     --path_trained_models checkpoints/semi_fdcl_10_seed-4 \
#     --labeled_ratio 10 \
#     --sd 4

# python main.py \
#     --dataset_name CRAG      \
#     --path_dataset datasets/CRAG \
#     --path_trained_models checkpoints/semi_fdcl_20_seed-4 \
#     --labeled_ratio 20 \
#     --sd 4

echo "=== 所有实验已全部完成！ ==="


python -u main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/semi_10_seed_24_rgbstain \
    --path_seg_results seg_pred/semi_10_seed_24_rgbstain \
    --labeled_ratio 10 \
    --sd 24 \
    --batch_size 2 \
    --num_epochs 200 \
    --network unet \
    --input_mode rgb_stain \
    --step_size 50 \
    --lr 0.5 \
    --gamma 0.5 \
    --unsup_weight 5.0 \
    --momentum 0.9 \
    --cps_mode standard 


python main.py \
    --dataset_name GlaS \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_1337 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_1337 \
    --labeled_ratio 10 \
    --sd 1337 \
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
    --uni_k 50 \
    --uni_build_reference_memory
