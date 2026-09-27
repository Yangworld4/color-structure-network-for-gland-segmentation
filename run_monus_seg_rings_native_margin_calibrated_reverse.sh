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
    local cps_mode="$5"

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
        --cps_mode "$cps_mode" \
        --uni_k 50 \
        --uni_build_reference_memory
}

echo "=== GlaS：开始运行 10% 标注实验 ==="

run_experiment GlaS 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute—12 native_fine_margin_calibrated_reverse
echo "=== GlaS：开始运行 20% 标注实验 ==="
# run_experiment GlaS 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute native_fine_margin_calibrated_reverse_disroute

echo "=== CRAG：开始运行 10% 标注实验 ==="
# run_experiment CRAG 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute native_fine_margin_calibrated_reverse_disroute

echo "=== CRAG：开始运行 20% 标注实验 ==="
# run_experiment CRAG 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute native_fine_margin_calibrated_reverse_disroute


# run_experiment GlaS 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression native_fine_margin_calibrated_reverse_disroute_cd_agree

# run_experiment GlaS 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression native_fine_margin_calibrated_reverse_disroute_cd_agree

echo "=== CRAG：开始运行 10% 标注实验 ==="
# run_experiment CRAG 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression native_fine_margin_calibrated_reverse_disroute_cd_agree

echo "=== CRAG：开始运行 20% 标注实验 ==="
# run_experiment CRAG 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression native_fine_margin_calibrated_reverse_disroute_cd_agree


echo "=== GlaS：开始运行 10% 标注实验 ==="

run_experiment GlaS 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_disroute native_fine_margin_calibrated_reverse_disroute_cd_dis

# run_experiment GlaS 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_disroute native_fine_margin_calibrated_reverse_disroute_cd_dis

echo "=== CRAG：开始运行 10% 标注实验 ==="
# run_experiment CRAG 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_disroute native_fine_margin_calibrated_reverse_disroute_cd_dis

echo "=== CRAG：开始运行 20% 标注实验 ==="
# run_experiment CRAG 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_disroute native_fine_margin_calibrated_reverse_disroute_cd_dis


echo "=== GlaS：开始运行 10% 标注实验 ==="

run_experiment GlaS 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression_disroute native_fine_margin_calibrated_reverse_disroute_cd_both

# run_experiment GlaS 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression_disroute native_fine_margin_calibrated_reverse_disroute_cd_both

echo "=== CRAG：开始运行 10% 标注实验 ==="
# run_experiment CRAG 10 1337 semi_10_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression_disroute native_fine_margin_calibrated_reverse_disroute_cd_both

echo "=== CRAG：开始运行 20% 标注实验 ==="
# run_experiment CRAG 20 1337 semi_20_seed_1337_rgbstain_c2f_CA_reverse_disroute_CD_suppression_disroute native_fine_margin_calibrated_reverse_disroute_cd_both

echo "=== GlaS 与 CRAG 实验全部完成 ==="




  python main.py \
    --dataset_name GlaS \
    --path_dataset datasets/GlaS \
    --path_trained_models checkpoints/semi_10_seed_1337_rgbstain_c2f_ca_reverse_disroute \
    --path_seg_results seg_pred/semi_10_seed_1337_rgbstain_c2f_ca_reverse_disroute \
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
    --cps_mode native_fine_margin_calibrated_reverse_disroute \
    --uni_k 50 \
    --uni_build_reference_memory
