"""Evaluate an FDCL model on a validation or test manifest at 256x256.

With ``--threshold auto``, the threshold is selected on the validation
manifest and then applied unchanged to the selected evaluation split.
"""

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


def find_best_checkpoint(checkpoint_root, dataset_name, input_view='auto'):
    root = Path(checkpoint_root)
    candidates = [
        path for path in root.rglob('best_*_Jc_*.pth')
        if dataset_name.lower() in (part.lower() for part in path.parts)
    ]
    # canonical 文件名为 best_Jc_xxxx.pth，不符合上面的历史通配模式，
    # 因此在此显式加入，同时保留 best_result1/2_Jc_xxxx.pth。
    candidates.extend(
        path for path in root.rglob('best_Jc_*.pth')
        if dataset_name.lower() in (part.lower() for part in path.parts)
    )
    if input_view == 'student1':
        candidates = [
            path for path in candidates
            if re.match(r'best_result1_Jc_', path.name)
        ]
    elif input_view == 'student2':
        candidates = [
            path for path in candidates
            if re.match(r'best_result2_Jc_', path.name)
        ]
    if not candidates:
        raise FileNotFoundError(
            "No {} checkpoint for {} found below {}; pass --checkpoint explicitly.".format(
                input_view, dataset_name, checkpoint_root
            )
        )

    def score(path):
        match = re.search(r'_Jc_([0-9]+(?:\.[0-9]+)?)\.pth$', path.name)
        # 兼容修复前的历史实验：同分时优先选择验证提升当轮立即保存的分支权重，
        # 避免误选训练结束时由旧逻辑保存的“伪最佳”权重。
        is_epoch_best = int(
            re.match(r'best_result[12]_Jc_', path.name) is not None
        )
        return (
            float(match.group(1)) if match else -1.0,
            is_epoch_best,
            path.stat().st_mtime,
        )

    return max(candidates, key=score)


def resolve_input_view(requested_view, checkpoint_path):
    """Resolve the model input while refusing ambiguous canonical checkpoints."""
    if requested_view != 'auto':
        return requested_view

    checkpoint_name = Path(checkpoint_path).name.lower()
    if re.match(r'best_result1_jc_', checkpoint_name):
        return 'student1'
    if re.match(r'best_result2_jc_', checkpoint_name):
        return 'student2'

    score_match = re.match(r'best_jc_([0-9]+(?:\.[0-9]+)?)\.pth$', checkpoint_name)
    if score_match:
        score = score_match.group(1)
        branch_candidates = list(Path(checkpoint_path).parent.glob(
            'best_result[12]_Jc_{}.pth'.format(score)
        ))
        inferred_views = {
            'student1' if 'result1' in path.name.lower() else 'student2'
            for path in branch_candidates
        }
        if len(inferred_views) == 1:
            return inferred_views.pop()

    raise ValueError(
        'Cannot infer whether checkpoint {} belongs to student1 or student2. '
        'Pass --input_view student1 or --input_view student2.'.format(checkpoint_path)
    )


def select_model_input(batch, input_view):
    """Select the same branch input for threshold search and final inference."""
    if input_view == 'student1':
        return batch.get('image_rgb', batch['image'])
    if input_view == 'student2':
        return batch.get('image_stain', batch['image'])
    raise ValueError('Unsupported input view: {}'.format(input_view))


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


def select_best_threshold(model, data_loader, device, input_view):
    thresholds = np.arange(0.0, 0.9, 0.02)
    true_positive = np.zeros_like(thresholds, dtype=np.int64)
    false_positive = np.zeros_like(thresholds, dtype=np.int64)
    false_negative = np.zeros_like(thresholds, dtype=np.int64)

    with torch.inference_mode():
        for batch in data_loader:
            images = select_model_input(batch, input_view).to(
                device, non_blocking=True
            )
            targets = batch['mask'].cpu().numpy().astype(bool)
            logits = extract_logits(model(images))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for index, threshold in enumerate(thresholds):
                predictions = probabilities > threshold
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
    parser.add_argument(
        '--split',
        default='test',
        choices=['validation', 'test'],
        help='dataset split to evaluate (default: test)',
    )
    parser.add_argument('--manifest', default=None,
                        help=(
                            'evaluation manifest override; defaults to '
                            'PATH_DATASET/manifests/paper_seed_2026/SPLIT.csv'
                        ))
    parser.add_argument(
        '--validation_manifest',
        default=None,
        help=(
            'validation data used only by --threshold auto; defaults to '
            'PATH_DATASET/manifests/paper_seed_2026/validation.csv'
        ),
    )
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint file; when omitted, select the highest-Jc best checkpoint')
    parser.add_argument('--checkpoint_root', default='checkpoints')
    parser.add_argument(
        '--input_view',
        default='auto',
        choices=['auto', 'student1', 'student2'],
        help=(
            'model input: auto maps best_result1 to student1 and best_result2 to student2'
        ),
    )
    parser.add_argument(
        '--input_mode',
        default='rgb_rgb',
        choices=['rgb_rgb', 'rgb_stain'],
        help='rgb_rgb evaluates RGB input; rgb_stain provides stain RGB for student2',
    )
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--network', default='xnet_sb')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--threshold', default='0.5',
                        help="foreground probability threshold in [0,1] (default: 0.5); "
                             "'auto' selects the Jaccard-optimal threshold on the validation manifest")
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
        else find_best_checkpoint(
            args.checkpoint_root, dataset_name, input_view=args.input_view
        )
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
    input_view = resolve_input_view(args.input_view, checkpoint_path)
    if input_view == 'student2' and args.input_mode != 'rgb_stain':
        raise ValueError('--input_view student2 requires --input_mode rgb_stain')

    dataset_root = Path(args.path_dataset or cfg['PATH_DATASET'])
    dataset_path_name = infer_dataset_from_path(dataset_root)
    if dataset_path_name and dataset_path_name != dataset_name:
        raise ValueError(
            'Dataset path belongs to {}, but the selected checkpoint belongs to {}: {}'.format(
                dataset_path_name, dataset_name, dataset_root
            )
        )
    manifest_path = Path(args.manifest) if args.manifest else (
        dataset_root / 'manifests' / 'paper_seed_2026'
        / '{}.csv'.format(args.split)
    )
    manifest_dataset = infer_dataset_from_path(manifest_path)
    if manifest_dataset and manifest_dataset != dataset_name:
        raise ValueError(
            '{} manifest belongs to {}, but the selected checkpoint belongs '
            'to {}: {}'.format(
                args.split.capitalize(),
                manifest_dataset,
                dataset_name,
                manifest_path,
            )
        )
    validation_manifest_path = (
        Path(args.validation_manifest)
        if args.validation_manifest
        else dataset_root / 'manifests' / 'paper_seed_2026' / 'validation.csv'
    )
    validation_manifest_dataset = infer_dataset_from_path(
        validation_manifest_path
    )
    if (
        validation_manifest_dataset
        and validation_manifest_dataset != dataset_name
    ):
        raise ValueError(
            'Validation manifest belongs to {}, but the selected checkpoint '
            'belongs to {}: {}'.format(
                validation_manifest_dataset,
                dataset_name,
                validation_manifest_path,
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
    evaluation_transform = transforms[
        'val' if args.split == 'validation' else 'test'
    ]
    evaluation_dataset = imagefloder_itn(
        manifest_path=str(manifest_path),
        data_transform_1=evaluation_transform,
        data_normalize_1=normalize,
        sup=True,
        input_mode=args.input_mode,
        training=False,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
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
        if args.split == 'validation' and args.validation_manifest is None:
            validation_loader = evaluation_loader
            threshold_manifest_path = manifest_path
        else:
            validation_dataset = imagefloder_itn(
                manifest_path=str(validation_manifest_path),
                data_transform_1=transforms['val'],
                data_normalize_1=normalize,
                sup=True,
                input_mode=args.input_mode,
                training=False,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == 'cuda',
            )
            threshold_manifest_path = validation_manifest_path
        threshold, threshold_search_jaccard = select_best_threshold(
            model, validation_loader, device, input_view
        )
        threshold_source = 'validation'
        threshold_manifest = str(threshold_manifest_path)
    else:
        threshold = requested_threshold
        threshold_search_jaccard = None
        threshold_source = 'fixed'
        threshold_manifest = None

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path('test_results') / dataset_name
        / '{}_{}'.format(checkpoint_path.stem, input_view) / args.split
    )
    prediction_dir = output_dir / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('fdcl.evaluate')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        output_dir / '{}.log'.format(args.split),
        mode='w',
        encoding='utf-8',
    )
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info('Checkpoint: %s', checkpoint_path)
    logger.info('Dataset: %s', dataset_name)
    logger.info('Evaluation split: %s', args.split)
    logger.info('Manifest: %s', manifest_path)
    logger.info('Input view: %s', input_view)
    if threshold_search_jaccard is None:
        logger.info(
            'Device: %s | samples=%d | evaluation_size=256x256 '
            '| threshold=%.4f (fixed)',
            device, len(evaluation_dataset), threshold,
        )
    else:
        logger.info('Threshold validation manifest: %s', threshold_manifest)
        logger.info(
            'Device: %s | samples=%d | evaluation_size=256x256 '
            '| threshold=%.4f (selected on validation, validation Jaccard=%.6f)',
            device, len(evaluation_dataset), threshold, threshold_search_jaccard,
        )

    total_tp = total_tn = total_fp = total_fn = 0
    total_loss = 0.0
    per_image_rows = []

    with torch.inference_mode():
        for batch in evaluation_loader:
            images = select_model_input(batch, input_view).to(
                device, non_blocking=True
            )
            masks = batch['mask'].to(device, non_blocking=True)
            logits = extract_logits(model(images))
            total_loss += F.cross_entropy(logits, masks).item() * images.shape[0]
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for batch_index, probability in enumerate(probabilities):
                target = masks[batch_index].cpu().numpy().astype(bool)
                prediction = probability > threshold

                tp, tn, fp, fn = segmentation_counts(prediction, target)
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                image_metrics = metrics_from_counts(tp, tn, fp, fn)
                image_name = batch['ID'][batch_index]
                per_image_rows.append({'image': image_name, **image_metrics})

                output_mask = Image.fromarray((prediction * 255).astype(np.uint8), mode='L')
                output_mask.save(prediction_dir / image_name)

    aggregate = metrics_from_counts(total_tp, total_tn, total_fp, total_fn)
    aggregate.update({
        'loss': total_loss / len(evaluation_dataset),
        'threshold': threshold,
        'threshold_source': threshold_source,
        'threshold_manifest': threshold_manifest,
        'threshold_search_jaccard': threshold_search_jaccard,
        'evaluation_size': [256, 256],
        'samples': len(evaluation_dataset),
        'checkpoint': str(checkpoint_path),
        'input_view': input_view,
        'split': args.split,
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
        '%s completed | loss=%.6f | Jaccard=%.6f | Dice=%.6f '
        '| Precision=%.6f | Recall=%.6f | Specificity=%.6f | Accuracy=%.6f',
        args.split.capitalize(),
        aggregate['loss'], aggregate['jaccard'], aggregate['dice'],
        aggregate['precision'], aggregate['recall'],
        aggregate['specificity'], aggregate['accuracy'],
    )
    logger.info('Results saved to: %s', output_dir)


if __name__ == '__main__':
    main()
