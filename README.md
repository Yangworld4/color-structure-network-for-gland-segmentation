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
datasets/GLaS/
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
python main.py --labeled_ratio 10
```

Choose how inputs are routed to the two CPS branches:

```bash
# Heterogeneous: Result1=RGB appearance, Result2=H+Gray+Sobel structure.
python main.py --branch_input_mode heterogeneous

# Homogeneous ablation: both branches receive the same augmented RGB tensor.
python main.py --branch_input_mode homogeneous
```

The legacy `--dual_view` / `--no-dual_view` flags remain supported. Homogeneous
runs receive an `-input=homogeneous` experiment-directory suffix so they do not
overwrite heterogeneous checkpoints or metrics.

#### Stage-1 complementarity diagnostics

The diagnostic path is disabled by default and does not change the training
loss, regular metrics CSV, best-model selection, or standard checkpoints. Run
the isolated seed-4 experiment with:

```bash
./run_diagnostics.sh
```

When `--complementarity_diagnostics` is enabled, validation logits that are
already in memory are additionally used to evaluate mean-probability fusion,
entropy-weighted fusion, and a ground-truth oracle upper bound. The default
five-epoch interval limits diagnostic overhead. By default, results are written
under the experiment's `complementarity_diagnostics/` subdirectory; the
provided script uses the predictable path
`outputs/complementarity_stage1/glas_10_seed4/`:

- `diagnostics_metrics.csv`: epoch-level branch, fusion, oracle, and
  boundary/interior statistics;
- `best_summary.json`: the best mean-fusion epoch and its key gains;
- `best_per_image.csv`: per-image overlap and exclusive-correctness ratios;
- `best_predictions.npz`: paired probabilities, masks, and boundary maps;
- `best_pair_model1.pth` / `best_pair_model2.pth`: both branches from the same
  best mean-fusion epoch;
- `visualizations/`: GT, branch, fusion, and error-region panels. Orange means
  appearance-only correct, cyan means structure-only correct, and red means
  both branches are wrong.

The oracle uses ground truth only to quantify the theoretical value of a
perfect per-pixel branch selector. It is never used by the optimizer or pseudo
label generation.

Aligned dual views are enabled by the dataset configuration. The existing
`image` field aliases the normalized RGB `image_appearance`, while
`image_structure` contains `[hematoxylin, gray, sobel]`. Disable the extension
to exercise the legacy data path:

During dual-view training, `model1` receives `image_appearance` and `model2`
receives `image_structure`. With dual views disabled, both models fall back to
the legacy `image` field.

```bash
python main.py --check_data --no-dual_view
```

Run the dual-view checks and save labeled/unlabeled debug panels without
starting training:

```bash
python tools/smoke_test_dual_views.py --check_forward
python tools/visualize_dual_views.py --output_dir outputs/dual_view_debug
```

Every epoch is validated and written to the experiment checkpoint directory:

- `train_YYYYmmdd-HHMMSS.log`: concise human-readable epoch summaries.
- `metrics_YYYYmmdd-HHMMSS.csv`: losses, Jaccard, Dice, threshold, learning rate, and epoch time
  for plotting or further analysis.

Use CRAG or the 20% labeled split:

```bash
python main.py --dataset_name CRAG --path_dataset datasets/CRAG --labeled_ratio 20
```

The legacy `train_sup_*/train_unsup_*/val` directory format remains supported
when no manifest directory is found.

## Final test

After training, test the checkpoint with the highest validation Jaccard:

```bash
python test.py --dataset_name GlaS --path_dataset datasets/GLaS
```

Evaluate the same checkpoint on the validation split instead:

```bash
python test.py \
  --dataset_name GlaS \
  --path_dataset datasets/GLaS \
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
