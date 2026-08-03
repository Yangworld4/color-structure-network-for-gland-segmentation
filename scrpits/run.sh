#!/bin/bash

# 开启错误检测：如果其中某一步程序报错退出，整个脚本会自动停止，防止后续错误运行
set -e
conda init
conda activate cg-seg

echo "=== 开始运行实验 1/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w_sc_seed-14 --feature_loss_mode semantic_private --w_sem 0.1 --w_comp 0 --sd 14

echo "=== 开始运行实验 2/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w_sc_seed-24 --feature_loss_mode semantic_private --w_sem 0.1 --w_comp 0 --sd 24

echo "=== 开始运行实验 3/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w_sc_comp_seed-14 --feature_loss_mode semantic_private --w_sem 0.1 --w_comp 0.01 --sd 14

echo "=== 开始运行实验 4/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w_sc_comp_seed-24 --feature_loss_mode semantic_private --w_sem 0.1 --w_comp 0.01 --sd 24

echo "=== 开始运行实验 5/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w-o_sc_comp_seed-14 --feature_loss_mode semantic_private --w_sem 0 --w_comp 0 --sd 14

echo "=== 开始运行实验 6/6 ==="
python main.py --dataset_name GlaS --path_dataset datasets/GLaS --labeled_ratio 10 --path_trained_models checkpoints/semi_fdcl_10_w-o_sc_comp_seed-24 --feature_loss_mode semantic_private --w_sem 0 --w_comp 0 --sd 24

echo "=== 所有实验已全部完成！ ==="
