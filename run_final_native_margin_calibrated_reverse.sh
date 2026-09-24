#!/bin/bash

# 开启错误检测：如果其中某一步程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate cg-seg

echo "=== 开始运行实验 ==="


python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse \
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
    --warm_up_duration 20 \
    --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse \
    --uni_k 50 \
    --uni_build_reference_memory

python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_42 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_42 \
    --labeled_ratio 10 \
    --sd 42 \
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

python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
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

python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20 \
    --labeled_ratio 20 \
    --sd 24 \
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

python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20_42 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20_42 \
    --labeled_ratio 20 \
    --sd 42 \
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

python main.py \
    --dataset_name CRAG \
    --path_dataset datasets/CRAG \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20_1337 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20_1337 \
    --labeled_ratio 20 \
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

echo "=== MoNuSeg：开始运行 10% / 20% 标注实验 ==="

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse \
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
    --warm_up_duration 20 \
    --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse \
    --uni_k 50 \
    --uni_build_reference_memory

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_42 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_42 \
    --labeled_ratio 10 \
    --sd 42 \
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

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
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

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20 \
    --labeled_ratio 20 \
    --sd 24 \
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

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20_42 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20_42 \
    --labeled_ratio 20 \
    --sd 42 \
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

python main.py \
    --dataset_name MoNuSeg \
    --path_dataset datasets/MoNuSeg \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse_20_1337 \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse_20_1337 \
    --labeled_ratio 20 \
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

echo "=== RINGS：开始运行 10% / 20% 标注实验 ==="

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse \
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
    --warm_up_duration 20 \
    --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse \
    --uni_k 50 \
    --uni_build_reference_memory

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse \
    --labeled_ratio 10 \
    --sd 42 \
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

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_reverse \
    --path_seg_results seg_pred/k50_margin_calibrated_reverse \
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

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_20 \
    --path_seg_results seg_pred/k50_margin_calibrated_20 \
    --labeled_ratio 20 \
    --sd 24 \
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

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_20_42 \
    --path_seg_results seg_pred/k50_margin_calibrated_20_42 \
    --labeled_ratio 20 \
    --sd 42 \
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

python main.py \
    --dataset_name RINGS \
    --path_dataset datasets/RINGS \
    --path_trained_models checkpoints/k50_margin_calibrated_20_1337 \
    --path_seg_results seg_pred/k50_margin_calibrated_20_1337 \
    --labeled_ratio 20 \
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
