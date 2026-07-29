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
