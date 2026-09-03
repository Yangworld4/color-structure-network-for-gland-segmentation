#!/bin/bash

# 开启错误检测：如果其中某一步程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate cg-seg程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/e报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/e报错退出，整个脚本会自动停止，防止后续错误运行
set -e
# 1. 载入 conda 环境脚本（注意：根据你的报错日志，修改为你的实际 miniconda 路径）
source /data/software/miniconda3/etc/profile.d/conda.sh

# 2. 激活指定的虚拟环境
conda activate cg-seg

echo "=== 开始运行实验 ==="

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi__10_seed \
    --labeled_ratio 10 \
    --sd 1337

python main.py \
    --dataset_name GlaS      \
    --path_dataset datasets/GLaS \
    --path_trained_models checkpoints/semi_cps_20_seed \
    --labeled_ratio 20 \
    --sd 1337

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


python main.py \
    --path_trained_models checkpoints/semi_10_seed_24_rgbrgb_cc_graph_v1 \
    --path_seg_results seg_pred/semi_10_seed_24_rgbrgb_cc_graph_v1 \
    --path_dataset datasets/GLaS \
    --dataset_name GlaS \
    --input_mode rgb_rgb \
    --labeled_ratio 10 \
    --batch_size 2 \
    --num_epochs 200 \
    --cc_graph \
    --cc_graph_tau 0.5 \
    --cc_graph_min_coverage 0.5 \
    --cc_graph_min_intersection  10 \
    --cc_graph_agreement_radius 3 \
    --cc_graph_roi_mode cut 
 

python main.py \
    --path_trained_models checkpoints/semi_10_seed_24_rgbstain \
    --path_seg_results seg_pred/semi_10_seed_24_rgbstain \
    --path_dataset datasets/CRAG \
    --dataset_name CRAG \
    --input_mode rgb_stain \
    --labeled_ratio 10 \
    --batch_size 2 \
    --num_epochs 200 \
    --ph-unmatched-mode vanilla 


python main.py \
    --path_trained_models checkpoints/semi_10_seed_24_rgb_e2_persistent_relation\
    --path_seg_results seg_pred/semi_10_seed_24_rgb_e2_persistent_relation \
    --path_dataset datasets/CRAG \
    --dataset_name CRAG \
    --input_mode rgb_rgb \
    --input1 image \
    --labeled_ratio 10 \
    --batch_size 2 \
    --num_epochs 200 \
    --milestone_interval 10 \
    --best_save_interval 10 \
    --step_size 50 \
    --lr 0.5 \
    --gamma 0.5 \
    --unsup_weight 5.0 \
    --warm_up_duration 20 \
    --momentum 0.9 \
    --wd -5 \
    --sd 24 \
    --network unet \
    --ph-directional-cps \
    --ph-start-epoch 20 \
    --ph-mask-dilation 1 \
    --ph-match-radius 16 \
    --ph-unmatched-mode vanilla \
    --ph-conflict-detector persistent_relation \
    --ph-validation-diagnostics-interval 10 \
    --ph-debug-timing\



  python main.py \
    --path_trained_models checkpoints/semi_10_seed_24_rgb_e2_ph_tree_connectivity_outcome13 \
    --path_seg_results seg_pred/semi_10_seed_24_rgb_e2_ph_tree_connectivity_outcome13 \
    --path_dataset datasets/GLaS \
    --dataset_name GlaS \
    --input_mode rgb_rgb \
    --input1 image \
    --labeled_ratio 10 \
    --batch_size 2 \
    --num_epochs 200 \
    --milestone_interval 10 \
    --best_save_interval 10 \
    --step_size 50 \
    --lr 0.5 \
    --gamma 0.5 \
    --unsup_weight 5.0 \
    --warm_up_duration 20 \
    --sd 24 \
    --network unet \
    --ph-directional-cps \
    --ph-start-epoch 20 \
    --ph-mask-dilation 1 \
    --ph-match-radius 16 \
    --ph-unmatched-mode vanilla \
    --ph-conflict-detector persistent_relation \
    --ph-correspondence-mode birth_nearest \
    --ph-relation-backend ph_tree \
    --ph-presence-conflict-mode connectivity_outcome \
    --ph-validation-diagnostics-interval 10 \
    --ph-save-conflict-vis