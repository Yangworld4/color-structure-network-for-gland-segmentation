# Semi-Supervised Histopathology Image Segmentation with Feature Diversified Collaborative Learning

![image](https://github.com/user-attachments/assets/1a263994-e0a0-406d-9d58-0534ec180fab)


## Overview

We present **Feature Diversified Collaborative Learning (FDCL)**, a straightforward yet powerful method for semi- supervised histopathology image segmentation. FDCL enhances the power of collaborative training by eliminating a common challenge of this framework—confirmation bias. The core of FDCL is to encourage diverse feature learning across sub-networks, thereby guiding the model to produce robust confidence maps and enhancing the overall consis- tency of predictions.

## Datasets

1. **Download the datasets**:

   - GlaS dataset: [Download GlaS dataset](https://datasets.activeloop.ai/docs/ml/datasets/glas-dataset/)
   - CRAG dataset: [Download CRAG dataset](https://opendatalab.com/OpenDataLab/CRAG/tree/main)

2. **Format the dataset as follows**:
```bash
  dataset/
  ├── GlaS
  │    ├── train_sup_85
  │    │    ├── image
  │    │    │    ├── 1.tif
  │    │    │    ├── 2.tif
  │    │    │    └── ...
  │    │    └── mask
  │    │         ├── 1.png
  │    │         ├── 2.png
  │    │         └── ...
  ├── CRAG
  │    ├── train_sup_35
  │    │    ├── image
  │    │    └── mask
  │    ├── train_unsup_138
  │    │    ├── image
  │    │    └── mask
  │    └── val
  │         ├── image
  │         └── mask
```

The repository also supports the prepared manifest layout directly:

```text
datasets/GlaS/
├── images/{train,test}/
├── masks/{train,test}/
└── manifests/paper_seed_2026/
    ├── labeled_10.csv
    ├── unlabeled_10.csv
    ├── labeled_20.csv
    ├── unlabeled_20.csv
    ├── validation.csv
    └── test.csv
```

Paths in each CSV are resolved relative to that CSV file. Supervised manifests
must contain `image` and `mask` columns; unlabeled manifests only need `image`.
Binary masks may use either `0/1` or `0/255` values.

`paper_seed_2026` first separates the official training split into a training
pool and a validation set. The 10%/20% labeled subsets are then sampled from
the training pool (rounding up), and the remainder is used as unlabeled data:

| Dataset | Train pool | 10% labeled | 10% unlabeled | 20% labeled | 20% unlabeled | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CRAG | 153 | 16 | 137 | 31 | 122 | 20 | 40 |
| GLaS | 68 | 7 | 61 | 14 | 54 | 17 | 80 |

Training and checkpoint selection use `validation.csv`; `test.csv` is reserved
for the final evaluation.

## Training FDCL

1. **Install dependencies**:

Install the required Python libraries by running the following command:

```bash
pip install -r requirements.txt
```

2. Train the model:

Once the dataset is organized and the dependencies are installed, you can start training by running:

Validate the prepared GLaS input without starting GPU training:

```bash
python main.py --check_data
```

Train with the 10% labeled split:

```bash
python main.py --labeled_ratio 10 --input_mode rgb_rgb
```

Choose how inputs are routed to the two CPS students:

```bash
# Baseline CPS: both students receive original RGB.
python main.py --input_mode rgb_rgb

# RGB-Stain CPS: Student1 receives RGB, Student2 receives stain RGB.
python main.py --input_mode rgb_stain
```

Both modes keep the same segmentation architecture, CPS loss, optimizer,
validation loop, and checkpoint flow. Only the second student input changes.

Dataset samples expose:

`rgb_rgb`: `image`, `mask`, `ID`

`rgb_stain`: `image`, `image_rgb`, `image_stain`, `mask`, `ID`

Every epoch is validated and written to the experiment checkpoint directory:

- `train.log`: epoch-level learning rate, average training loss, validation
  Dice/Jaccard, and validation diversity/error relationship metrics.
- `metrics.csv`: iteration-level training losses, student diversity,
  prediction entropy, and pseudo-label overlap metrics.

Use CRAG or the 20% labeled split:

```bash
python main.py --dataset_name CRAG --path_dataset datasets/CRAG --labeled_ratio 20 --input_mode rgb_stain
```

The legacy `train_sup_*/train_unsup_*/val` directory format remains supported
when no manifest directory is found.

## Final test

After training, test the checkpoint with the highest validation Jaccard:

```bash
python test.py --dataset_name GlaS --path_dataset datasets/GlaS
```

Evaluate the same checkpoint on the validation split instead:

```bash
python test.py \
  --dataset_name GlaS \
  --path_dataset datasets/GlaS \
  --split validation
```

`--split test` is the default. The selected split controls the default manifest
(`test.csv` or `validation.csv`) and its separate output subdirectory. An
explicit `--manifest` overrides that default.

When checkpoints exist for only one dataset, `python test.py` identifies that
dataset automatically. An explicit checkpoint also determines the dataset from
its path. Conflicting `--dataset_name`, dataset path, manifest, and checkpoint
combinations are rejected instead of being evaluated silently.

For CRAG or a specific checkpoint:

```bash
python test.py \
  --dataset_name CRAG \
  --path_dataset datasets/CRAG \
  --checkpoint checkpoints/path/to/best_result1_Jc_0.8000.pth
```

New dual-view checkpoints are branch-aware during testing: `best_result1`
uses the appearance view and `best_result2` uses the structure view. Override
the automatic choice when testing a legacy FDCL checkpoint:

```bash
python test.py \
  --dataset_name GLaS \
  --checkpoint checkpoints/path/to/legacy_result2.pth \
  --input_view appearance
```

By default, testing uses a foreground threshold of `0.5`, the standard binary
decision rule and the equivalent of `argmax` for two-class softmax output. Use
`--threshold auto` to select the threshold on `validation.csv` before applying
it to the held-out test set. Predictions are restored to each source image's
resolution.
The output directory contains `test.log`, aggregate `metrics.json`,
`per_image_metrics.csv`, and the predicted masks.

## Citation

If you use this work in your research or project, please cite the following paper:

```bash
```



 python main.py \
    --dataset_name GLaS --path_dataset datasets/GlaS \
    --path_trained_models checkpoints/glas10_seed1337_uni \
    --path_seg_results seg_pred/glas10_seed1337_uni \
    --labeled_ratio 10 --sd 1337 --batch_size 2 --num_epochs 200 \
    --network unet --input_mode rgb_stain \
    --step_size 50 --lr 0.5 --gamma 0.5 --unsup_weight 5.0 \
    --warm_up_duration 20 --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse --uni_k 50 \
    --reference-backbone uni --uni-input-size 256 \
    --uni_build_reference_memory

  UNI 使用项目配置的本地 --uni_weights 和 --uni_config。如果权重不在默认路径，可
  另外传这两个参数。

  ### UNI2-h（ViT-H/14，16×16 网格）

  python main.py \
    --dataset_name GLaS --path_dataset datasets/GlaS \
    --path_trained_models checkpoints/glas10_seed1337_uni2h \
    --path_seg_results seg_pred/glas10_seed1337_uni2h \
    --labeled_ratio 10 --sd 1337 --batch_size 2 --num_epochs 200 \
    --network unet --input_mode rgb_stain \
    --step_size 50 --lr 0.5 --gamma 0.5 --unsup_weight 5.0 \
    --warm_up_duration 20 --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse --uni_k 50 \
    --reference-backbone uni2-h \
    --uni-loader hf-hub --uni-hub-id MahmoodLab/UNI2-h \
    --uni-input-size 224 \
    --uni_build_reference_memory

  缓存中已有 UNI2-h 配置和权重时会优先本地加载；输入 224×224 得到 16×16 网格。

  ### Kaiko ViT-B/8（28×28 网格，784 个 token）

  python main.py \
    --dataset_name GLaS --path_dataset datasets/GlaS \
    --path_trained_models checkpoints/glas10_seed1337_kaiko_vitb8 \
    --path_seg_results seg_pred/glas10_seed1337_kaiko_vitb8 \
    --labeled_ratio 10 --sd 1337 --batch_size 2 --num_epochs 200 \
    --network unet --input_mode rgb_stain \
    --step_size 50 --lr 0.5 --gamma 0.5 --unsup_weight 5.0 \
    --warm_up_duration 20 --momentum 0.9 \
    --cps_mode native_fine_margin_calibrated_reverse --uni_k 50 \
    --reference-backbone kaiko-vitb8 --uni-input-size 224 \
    --uni_build_reference_memory

  Kaiko 使用官方 Torch Hub 加载方式，不需要传 --uni-loader hf-hub 或 --uni-hub-
  id。输入 224×224 保留 28×28 网格；它的 memory、Student coarse pooling 和训练路
  由使用同一网格。

  如果只想建立 Kaiko memory bank、不训练，在 Kaiko 命令末尾加 --build-reference-
  memory-only。


