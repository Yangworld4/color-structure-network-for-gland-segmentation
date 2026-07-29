"""Evaluate a trained FDCL segmentation model on the held-out test manifest."""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.getnetwork import get_network


KNOWN_DATASETS = {'glas': 'GlaS', 'crag': 'CRAG', 'monuseg': 'MoNuSeg'}


def infer_dataset_from_path(path):
    for part in Path(path).parts:
        dataset_name = KNOWN_DATASETS.get(part.lower())
        if dataset_name is not None:
            return dataset_name
    return None


def resolve_dataset_name(requested_name, checkpoint, checkpoint_root):
    checkpoint_dataset = infer_dataset_from_path(checkpoint) if checkpoint else None
    if requested_name != 'auto':
        if checkpoint_dataset and checkpoint_dataset != requested_name:
            raise ValueError(
                'Dataset/checkpoint mismatch: --dataset_name={} but checkpoint path belongs to {}.'.format(
                    requested_name, checkpoint_dataset
                )
            )
        return requested_name

    if checkpoint_dataset:
        return checkpoint_dataset

    available = {
        inferred
        for path in Path(checkpoint_root).rglob('best_*_Jc_*.pth')
        for inferred in [infer_dataset_from_path(path)]
        if inferred is not None
    }
    if len(available) == 1:
        return available.pop()
    if not available:
        raise FileNotFoundError('No dataset-specific best checkpoint found below {}'.format(checkpoint_root))
    raise ValueError(
        'Checkpoints for multiple datasets were found ({}); specify --dataset_name.'.format(
            ', '.join(sorted(available))
        )
    )


def find_best_checkpoint(checkpoint_root, dataset_name):
    root = Path(checkpoint_root)
    candidates = [
        path for path in root.rglob('best_*_Jc_*.pth')
        if dataset_name.lower() in (part.lower() for part in path.parts)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No best checkpoint for {} found below {}; pass --checkpoint explicitly.".format(
                dataset_name, checkpoint_root
            )
        )

    def score(path):
        match = re.search(r'_Jc_([0-9]+(?:\.[0-9]+)?)\.pth$', path.name)
        return (float(match.group(1)) if match else -1.0, path.stat().st_mtime)

    return max(candidates, key=score)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint content in {}'.format(checkpoint_path))

    state_dict = {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict, strict=True)


def segmentation_counts(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    return tp, tn, fp, fn


def metrics_from_counts(tp, tn, fp, fn):
    eps = 1e-8
    return {
        'jaccard': tp / (tp + fp + fn + eps),
        'dice': 2 * tp / (2 * tp + fp + fn + eps),
        'precision': tp / (tp + fp + eps),
        'recall': tp / (tp + fn + eps),
        'specificity': tn / (tn + fp + eps),
        'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
    }


def extract_logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output) or output.ndim != 4:
        raise TypeError('Expected model logits shaped [N,C,H,W], got {}'.format(type(output)))
    return output


def select_validation_threshold(model, validation_loader, device):
    thresholds = np.arange(0.0, 0.9, 0.02)
    true_positive = np.zeros_like(thresholds, dtype=np.int64)
    false_positive = np.zeros_like(thresholds, dtype=np.int64)
    false_negative = np.zeros_like(thresholds, dtype=np.int64)

    with torch.inference_mode():
        for batch in validation_loader:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].cpu().numpy().astype(bool)
            logits = extract_logits(model(images))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for index, threshold in enumerate(thresholds):
                predictions = probabilities >= threshold
                true_positive[index] += np.logical_and(predictions, targets).sum()
                false_positive[index] += np.logical_and(predictions, ~targets).sum()
                false_negative[index] += np.logical_and(~predictions, targets).sum()

    jaccard = true_positive / (
        true_positive + false_positive + false_negative + 1e-8
    )
    best_index = int(np.argmax(jaccard))
    return float(thresholds[best_index]), float(jaccard[best_index])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_name', default='auto',
                        choices=['auto', 'GlaS', 'CRAG', 'MoNuSeg'],
                        help='infer from an explicit checkpoint or the available checkpoint directory')
    parser.add_argument('--path_dataset', default=None)
    parser.add_argument('--manifest', default=None,
                        help='defaults to PATH_DATASET/manifests/paper_seed_2026/test.csv')
    parser.add_argument('--validation_manifest', default=None,
                        help='used to select the threshold when --threshold=auto')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint file; when omitted, select the highest-Jc best checkpoint')
    parser.add_argument('--checkpoint_root', default='checkpoints')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--network', default='xnet_sb')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--threshold', default='0.5',
                        help="foreground probability threshold in [0,1] (default: 0.5); "
                             "'auto' selects it on validation.csv")
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threshold != 'auto':
        try:
            requested_threshold = float(args.threshold)
        except ValueError as exc:
            raise ValueError("--threshold must be 'auto' or a number between 0 and 1") from exc
        if not 0.0 <= requested_threshold <= 1.0:
            raise ValueError('--threshold must be between 0 and 1')
    else:
        requested_threshold = None

    dataset_name = resolve_dataset_name(
        args.dataset_name, args.checkpoint, args.checkpoint_root
    )
    cfg = dataset_cfg(dataset_name)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else find_best_checkpoint(args.checkpoint_root, dataset_name)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))
    checkpoint_dataset = infer_dataset_from_path(checkpoint_path)
    if checkpoint_dataset and checkpoint_dataset != dataset_name:
        raise ValueError(
            'Selected checkpoint belongs to {}, not {}: {}'.format(
                checkpoint_dataset, dataset_name, checkpoint_path
            )
        )

    dataset_root = Path(args.path_dataset or cfg['PATH_DATASET'])
    dataset_path_name = infer_dataset_from_path(dataset_root)
    if dataset_path_name and dataset_path_name != dataset_name:
        raise ValueError(
            'Dataset path belongs to {}, but the selected checkpoint belongs to {}: {}'.format(
                dataset_path_name, dataset_name, dataset_root
            )
        )
    manifest_path = Path(args.manifest) if args.manifest else (
        dataset_root / 'manifests' / 'paper_seed_2026' / 'test.csv'
    )
    validation_manifest_path = (
        Path(args.validation_manifest) if args.validation_manifest
        else manifest_path.parent / 'validation.csv'
    )
    manifest_dataset = infer_dataset_from_path(manifest_path)
    if manifest_dataset and manifest_dataset != dataset_name:
        raise ValueError(
            'Test manifest belongs to {}, but the selected checkpoint belongs to {}: {}'.format(
                manifest_dataset, dataset_name, manifest_path
            )
        )

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])
    test_dataset = imagefloder_itn(
        manifest_path=str(manifest_path),
        data_transform_1=transforms['test'],
        data_normalize_1=normalize,
        sup=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    if requested_threshold is None:
        validation_dataset = imagefloder_itn(
            manifest_path=str(validation_manifest_path),
            data_transform_1=transforms['val'],
            data_normalize_1=normalize,
            sup=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
        )
        threshold, validation_jaccard = select_validation_threshold(
            model, validation_loader, device
        )
        threshold_source = 'validation'
    else:
        threshold = requested_threshold
        validation_jaccard = None
        threshold_source = 'fixed'

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path('test_results') / dataset_name / checkpoint_path.stem
    )
    prediction_dir = output_dir / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('fdcl.test')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(output_dir / 'test.log', mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info('Checkpoint: %s', checkpoint_path)
    logger.info('Dataset: %s', dataset_name)
    logger.info('Manifest: %s', manifest_path)
    if validation_jaccard is None:
        logger.info('Device: %s | samples=%d | threshold=%.4f (fixed)',
                    device, len(test_dataset), threshold)
    else:
        logger.info(
            'Device: %s | samples=%d | threshold=%.4f '
            '(selected on validation set, validation Jaccard=%.6f)',
            device, len(test_dataset), threshold, validation_jaccard,
        )

    total_tp = total_tn = total_fp = total_fn = 0
    total_loss = 0.0
    per_image_rows = []
    sample_index = 0

    with torch.inference_mode():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            logits = extract_logits(model(images))
            total_loss += F.cross_entropy(logits, masks).item() * images.shape[0]
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for batch_index, probability in enumerate(probabilities):
                mask_path = test_dataset.mask_paths[sample_index]
                with Image.open(mask_path) as mask_image:
                    original_mask = np.asarray(mask_image.convert('L')) > 0
                    original_size = mask_image.size

                probability_image = Image.fromarray(probability.astype(np.float32), mode='F')
                probability_image = probability_image.resize(original_size, Image.Resampling.BILINEAR)
                prediction = np.asarray(probability_image) >= threshold

                tp, tn, fp, fn = segmentation_counts(prediction, original_mask)
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                image_metrics = metrics_from_counts(tp, tn, fp, fn)
                image_name = batch['ID'][batch_index]
                per_image_rows.append({'image': image_name, **image_metrics})

                output_mask = Image.fromarray((prediction * 255).astype(np.uint8), mode='L')
                output_mask.save(prediction_dir / image_name)
                sample_index += 1

    aggregate = metrics_from_counts(total_tp, total_tn, total_fp, total_fn)
    aggregate.update({
        'loss': total_loss / len(test_dataset),
        'threshold': threshold,
        'threshold_source': threshold_source,
        'validation_jaccard_at_threshold': validation_jaccard,
        'samples': len(test_dataset),
        'checkpoint': str(checkpoint_path),
        'manifest': str(manifest_path),
    })

    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as metrics_file:
        json.dump(aggregate, metrics_file, indent=2, ensure_ascii=False)
    with (output_dir / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['image', 'jaccard', 'dice', 'precision', 'recall',
                        'specificity', 'accuracy'],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    logger.info(
        'Test completed | loss=%.6f | Jaccard=%.6f | Dice=%.6f '
        '| Precision=%.6f | Recall=%.6f | Specificity=%.6f | Accuracy=%.6f',
        aggregate['loss'], aggregate['jaccard'], aggregate['dice'],
        aggregate['precision'], aggregate['recall'],
        aggregate['specificity'], aggregate['accuracy'],
    )
    logger.info('Results saved to: %s', output_dir)


if __name__ == '__main__':
    main()
