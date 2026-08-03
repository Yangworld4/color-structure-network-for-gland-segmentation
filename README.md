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
    └── validation.csv
```

Paths in each CSV are resolved relative to that CSV file. Supervised manifests
must contain `image` and `mask` columns; unlabeled manifests only need `image`.
Binary masks may use either `0/1` or `0/255` values.

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

### Optional heterogeneous semantic/private feature objective

The original reciprocal-MSE FDCL feature loss is still the default and its
implementation is unchanged:

```bash
python main.py --feature_loss_mode original
```

Enable the optional heterogeneous-view adaptation with:

```bash
python main.py \
  --dual_view \
  --feature_loss_mode semantic_private \
  --w_sem 0.1 \
  --w_comp 0.05
```

This mode adds two training-only projection pairs at different decoder levels:

- semantic consistency uses the `b1_3_3` output before upsampling
  (`[B, 256, H/4, W/4]`);
- private complementarity uses the `b1_2_3` output before upsampling
  (`[B, 128, H/2, W/2]`);
- the final `[B, 64, H, W]` feature remains dedicated to segmentation.

Class prototypes in the shared-semantic projections are aligned with cosine
distance. Private projections use a bounded cosine-margin complementarity term
plus a small variance floor to prevent collapse. Labeled images use
ground-truth masks; unlabeled images use only pixels on which both branches
agree and both confidence values exceed
`--feature_confidence_threshold` (default `0.7`).

Relevant options:

- `--feature_loss_mode original`: unchanged FDCL loss (default).
- `--feature_loss_mode semantic_private`: semantic consistency and private
  complementarity (`--w_dis` is not used in this mode).
- `--feature_loss_mode none`: no feature-level objective, for ablation.
- `--semantic_feature_channels 256`: H/4 semantic feature width.
- `--private_feature_channels 128`: H/2 private feature width.
- `--feature_projection_dim 32`: semantic/private projection width.
- `--feature_margin 0.2`: unpenalized private cosine-similarity margin.
- `--private_variance_weight 0.1`: private projection anti-collapse weight.

The auxiliary projection heads shape training but are not part of inference
checkpoints. New code paths are marked `HETERO-FDCL ADDITION`; the preserved
loss is marked `ORIGINAL FDCL LOSS`. Metrics CSV files additionally report
`train_loss_feature`, `train_loss_semantic`, `train_loss_complement`, and
`train_feature_valid_ratio`; the last field is the unlabeled high-confidence
consensus-pixel coverage.

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
