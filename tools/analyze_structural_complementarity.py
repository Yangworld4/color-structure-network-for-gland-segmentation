"""Offline CC-based structural complementarity analysis for CPS students."""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import find_objects, label
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from models.getnetwork import get_network


KNOWN_DATASETS = {'glas': 'GlaS', 'crag': 'CRAG', 'monuseg': 'MoNuSeg'}
INSTANCE_IOU_THRESHOLD = 0.5
TOPOLOGY_IOU_THRESHOLD = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument(
        '--dataset',
        required=True,
        help='dataset name such as GLaS, or a dataset root path',
    )
    parser.add_argument(
        '--split',
        default='val',
        choices=['val', 'validation', 'test'],
    )
    parser.add_argument('--epochs', nargs='+', required=True, type=int)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--network', default='unet')
    parser.add_argument(
        '--input_mode',
        default='auto',
        choices=['auto', 'rgb_rgb', 'rgb_stain', 'rgb_hed', 'ucmt_views'],
    )
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--num_visualizations', default=6, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def infer_input_mode(checkpoint_dir):
    match = re.search(r'input_mode=([A-Za-z0-9_]+)', str(checkpoint_dir))
    if match:
        return match.group(1)
    return 'rgb_stain'


def resolve_dataset(dataset_arg):
    path = Path(dataset_arg)
    if path.exists():
        dataset_name = KNOWN_DATASETS.get(path.name.lower())
        if dataset_name is None:
            for part in path.parts:
                dataset_name = KNOWN_DATASETS.get(part.lower())
                if dataset_name is not None:
                    break
        if dataset_name is None:
            raise ValueError(
                'Cannot infer dataset name from path {}; use GLaS/CRAG/MoNuSeg.'.format(
                    path
                )
            )
        return dataset_name, path

    dataset_name = KNOWN_DATASETS.get(dataset_arg.lower())
    if dataset_name is None:
        raise ValueError('Unknown dataset: {}'.format(dataset_arg))
    cfg = dataset_cfg(dataset_name)
    return dataset_name, Path(cfg['PATH_DATASET'])


def split_to_manifest_name(split):
    return 'validation' if split == 'val' else split


def build_dataset(dataset_name, dataset_root, split, input_mode, manifest=None):
    cfg = dataset_cfg(dataset_name)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])
    split_name = split_to_manifest_name(split)
    manifest_path = (
        Path(manifest)
        if manifest is not None
        else dataset_root / 'manifests' / 'paper_seed_2026' / '{}.csv'.format(split_name)
    )
    if manifest_path.is_file():
        return imagefloder_itn(
            manifest_path=str(manifest_path),
            data_transform_1=transforms['val'],
            data_normalize_1=normalize,
            sup=True,
            input_mode=input_mode,
            training=False,
        )

    legacy_split = 'val' if split in ('val', 'validation') else 'test'
    data_dir = dataset_root / legacy_split
    return imagefloder_itn(
        data_dir=str(data_dir),
        input1='image',
        data_transform_1=transforms['val'],
        data_normalize_1=normalize,
        sup=True,
        input_mode=input_mode,
        training=False,
    )


def extract_logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output) or output.ndim != 4:
        raise TypeError('Expected logits shaped [N,C,H,W], got {}'.format(type(output)))
    return output


def foreground_probability(logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)[:, 0]
    return torch.sigmoid(logits[:, 1])


def select_student_inputs(batch):
    input_a = batch.get('image_rgb', batch['image'])
    input_b = batch.get('image_stain', batch['image'])
    return input_a, input_b


def unwrap_state_dict(checkpoint, checkpoint_path):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint content in {}'.format(checkpoint_path))
    return checkpoint


def load_state_dict(model, state_dict, checkpoint_path):
    state_dict = unwrap_state_dict(state_dict, checkpoint_path)
    state_dict = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def resolve_epoch_checkpoints(checkpoint_dir, epoch):
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.name == 'milestones':
        checkpoint_dir = checkpoint_dir.parent
    if checkpoint_dir.name == 'periodic_best':
        checkpoint_dir = checkpoint_dir.parent
    milestone = checkpoint_dir / 'milestones' / 'epoch_{:03d}.pth'.format(epoch)
    if milestone.is_file():
        return milestone, None

    periodic_dir = checkpoint_dir / 'periodic_best'
    path_a = sorted(periodic_dir.glob('epoch_{:04d}_best_A_Jc_*.pth'.format(epoch)))
    path_b = sorted(periodic_dir.glob('epoch_{:04d}_best_S_Jc_*.pth'.format(epoch)))
    if path_a and path_b:
        return path_a[-1], path_b[-1]

    raise FileNotFoundError(
        'No milestone or periodic A/S checkpoint found for epoch {}'.format(epoch)
    )


def load_model(checkpoint_dir, epoch, cfg, network, device):
    model_a = get_network(network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model_b = get_network(network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    path_a, path_b = resolve_epoch_checkpoints(checkpoint_dir, epoch)

    checkpoint_a = torch.load(path_a, map_location='cpu', weights_only=False)
    if path_b is None:
        load_state_dict(model_a, checkpoint_a['appearance_state_dict'], path_a)
        load_state_dict(model_b, checkpoint_a['structure_state_dict'], path_a)
    else:
        checkpoint_b = torch.load(path_b, map_location='cpu', weights_only=False)
        load_state_dict(model_a, checkpoint_a, path_a)
        load_state_dict(model_b, checkpoint_b, path_b)

    model_a.to(device).eval()
    model_b.to(device).eval()
    return model_a, model_b


def predict_mask(model, images, threshold):
    logits = extract_logits(model(images))
    probabilities = foreground_probability(logits)
    return probabilities > threshold


def extract_components(mask):
    mask = np.asarray(mask).astype(bool)
    labeled, _count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    slices = find_objects(labeled)
    components = []
    for index, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        component_mask = labeled == index
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0:
            continue
        y_min, x_min = int(ys.min()), int(xs.min())
        y_max, x_max = int(ys.max()) + 1, int(xs.max()) + 1
        components.append({
            'id': index,
            'area': int(component_mask.sum()),
            'centroid': (float(ys.mean()), float(xs.mean())),
            'bbox': (y_min, x_min, y_max, x_max),
            'mask': component_mask,
        })
    return components


def component_iou(component_a, component_b):
    intersection = np.logical_and(component_a['mask'], component_b['mask']).sum()
    union = np.logical_or(component_a['mask'], component_b['mask']).sum()
    return float(intersection / union) if union else 0.0


def iou_matrix(components_a, components_b):
    matrix = np.zeros((len(components_a), len(components_b)), dtype=np.float32)
    for index_a, component_a in enumerate(components_a):
        for index_b, component_b in enumerate(components_b):
            matrix[index_a, index_b] = component_iou(component_a, component_b)
    return matrix


def match_instances(components_a, components_b, threshold=INSTANCE_IOU_THRESHOLD):
    if not components_a or not components_b:
        return [], set(), set(), iou_matrix(components_a, components_b)

    matrix = iou_matrix(components_a, components_b)
    rows, cols = linear_sum_assignment(-matrix)
    matches = [
        (int(row), int(col), float(matrix[row, col]))
        for row, col in zip(rows, cols)
        if matrix[row, col] > threshold
    ]
    matched_a = {row for row, _col, _iou in matches}
    matched_b = {col for _row, col, _iou in matches}
    return matches, matched_a, matched_b, matrix


def count_multi_overlap_rows(components_a, components_b):
    matrix = iou_matrix(components_a, components_b)
    if matrix.size == 0:
        return 0
    return int(((matrix > TOPOLOGY_IOU_THRESHOLD).sum(axis=1) > 1).sum())


def gt_states(matches_a_gt, matches_b_gt, num_gt):
    a_gt = {gt_index for _pred_index, gt_index, _iou in matches_a_gt}
    b_gt = {gt_index for _pred_index, gt_index, _iou in matches_b_gt}
    states = {}
    for gt_index in range(num_gt):
        states[gt_index] = {
            'A_correct': gt_index in a_gt,
            'B_correct': gt_index in b_gt,
        }
    return states


def analyze_structures(image_name, epoch, pred_a, pred_b, gt):
    components_gt = extract_components(gt)
    components_a = extract_components(pred_a)
    components_b = extract_components(pred_b)

    matches_ab, matched_a_ab, matched_b_ab, _ = match_instances(
        components_a, components_b
    )
    matches_a_gt, matched_a_gt, matched_gt_a, _ = match_instances(
        components_a, components_gt
    )
    matches_b_gt, matched_b_gt, matched_gt_b, _ = match_instances(
        components_b, components_gt
    )

    num_gt = len(components_gt)
    num_a = len(components_a)
    num_b = len(components_b)
    states = gt_states(matches_a_gt, matches_b_gt, num_gt)
    a_only = sum(
        1 for state in states.values()
        if state['A_correct'] and not state['B_correct']
    )
    b_only = sum(
        1 for state in states.values()
        if state['B_correct'] and not state['A_correct']
    )

    row = {
        'image': image_name,
        'epoch': epoch,
        'num_gt_instances': num_gt,
        'num_A_instances': num_a,
        'num_B_instances': num_b,
        'matched_AB_instances': len(matches_ab),
        'unmatched_A_instances': num_a - len(matched_a_ab),
        'unmatched_B_instances': num_b - len(matched_b_ab),
        'A_correct_B_wrong_instances': a_only,
        'B_correct_A_wrong_instances': b_only,
        'A_missing_instances': num_gt - len(matched_gt_a),
        'B_missing_instances': num_gt - len(matched_gt_b),
        'A_false_positive_instances': num_a - len(matched_a_gt),
        'B_false_positive_instances': num_b - len(matched_b_gt),
        'A_merge_cases': count_multi_overlap_rows(components_a, components_gt),
        'B_merge_cases': count_multi_overlap_rows(components_b, components_gt),
        'A_split_cases': count_multi_overlap_rows(components_gt, components_a),
        'B_split_cases': count_multi_overlap_rows(components_gt, components_b),
    }
    return row, states


def state_label(state, source_model):
    if source_model == 'A':
        if state['A_correct'] and not state['B_correct']:
            return 'source_only'
        if state['A_correct'] and state['B_correct']:
            return 'both_correct'
        if not state['A_correct'] and not state['B_correct']:
            return 'both_wrong'
    else:
        if state['B_correct'] and not state['A_correct']:
            return 'source_only'
        if state['A_correct'] and state['B_correct']:
            return 'both_correct'
        if not state['A_correct'] and not state['B_correct']:
            return 'both_wrong'
    return 'other'


def track_temporal_transition(epoch_states):
    rows = []
    ordered_epochs = sorted(epoch_states)
    for start_epoch, end_epoch in zip(ordered_epochs[:-1], ordered_epochs[1:]):
        start_states = epoch_states[start_epoch]
        end_states = epoch_states[end_epoch]
        instance_ids = sorted(set(start_states).intersection(end_states))
        for source_model in ('A', 'B'):
            counts = {
                'correct_propagation': 0,
                'knowledge_erosion': 0,
                'persistent_disagreement': 0,
            }
            for instance_id in instance_ids:
                start_label = state_label(start_states[instance_id], source_model)
                if start_label != 'source_only':
                    continue
                end_label = state_label(end_states[instance_id], source_model)
                if end_label == 'both_correct':
                    counts['correct_propagation'] += 1
                elif end_label == 'both_wrong':
                    counts['knowledge_erosion'] += 1
                elif end_label == 'source_only':
                    counts['persistent_disagreement'] += 1
            for transition_type, count in counts.items():
                rows.append({
                    'start_epoch': start_epoch,
                    'end_epoch': end_epoch,
                    'source_model': source_model,
                    'transition_type': transition_type,
                    'count': count,
                })
    return rows


def knowledge_erosion_rate(transition_rows):
    total = sum(row['count'] for row in transition_rows)
    erosion = sum(
        row['count'] for row in transition_rows
        if row['transition_type'] == 'knowledge_erosion'
    )
    return erosion / total if total else float('nan')


def aggregate_epoch(rows, epoch, pixel_disagreement_sum, pixel_count):
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            'num_gt_instances',
            'num_A_instances',
            'num_B_instances',
            'matched_AB_instances',
            'unmatched_A_instances',
            'unmatched_B_instances',
            'A_correct_B_wrong_instances',
            'B_correct_A_wrong_instances',
            'A_missing_instances',
            'B_missing_instances',
            'A_false_positive_instances',
            'B_false_positive_instances',
            'A_merge_cases',
            'B_merge_cases',
            'A_split_cases',
            'B_split_cases',
        )
    }
    structural_disagreement = (
        totals['A_correct_B_wrong_instances']
        + totals['B_correct_A_wrong_instances']
        + totals['A_missing_instances']
        + totals['B_missing_instances']
        + totals['A_false_positive_instances']
        + totals['B_false_positive_instances']
        + totals['A_merge_cases']
        + totals['B_merge_cases']
        + totals['A_split_cases']
        + totals['B_split_cases']
    )
    complementarity = (
        totals['A_correct_B_wrong_instances']
        + totals['B_correct_A_wrong_instances']
    )
    num_gt = max(totals['num_gt_instances'], 1)
    num_a = max(totals['num_A_instances'], 1)
    num_b = max(totals['num_B_instances'], 1)
    return {
        'epoch': epoch,
        'pixel_disagreement': pixel_disagreement_sum / max(pixel_count, 1),
        'instance_disagreement': (
            totals['unmatched_A_instances'] + totals['unmatched_B_instances']
        ),
        'structural_complementarity_rate': (
            complementarity / structural_disagreement
            if structural_disagreement else 0.0
        ),
        'A_error_rate': (
            totals['A_missing_instances'] + totals['A_false_positive_instances']
        ) / (num_gt + num_a),
        'B_error_rate': (
            totals['B_missing_instances'] + totals['B_false_positive_instances']
        ) / (num_gt + num_b),
        'merge_rate_A': totals['A_merge_cases'] / num_a,
        'merge_rate_B': totals['B_merge_cases'] / num_b,
        'split_rate_A': totals['A_split_cases'] / num_gt,
        'split_rate_B': totals['B_split_cases'] / num_gt,
        **totals,
    }


def make_visualization(gt, pred_a, pred_b, image_name, epoch, output_dir):
    height, width = gt.shape
    canvas = np.zeros((height * 4, width, 3), dtype=np.uint8)
    canvas[0:height, :, 1] = gt.astype(np.uint8) * 255
    canvas[height:height * 2, :, 0] = pred_a.astype(np.uint8) * 255
    canvas[height * 2:height * 3, :, 2] = pred_b.astype(np.uint8) * 255
    disagreement = np.logical_xor(pred_a, pred_b)
    canvas[height * 3:height * 4, :, 0] = disagreement.astype(np.uint8) * 255
    canvas[height * 3:height * 4, :, 1] = gt.astype(np.uint8) * 120
    safe_name = Path(image_name).stem
    Image.fromarray(canvas).save(output_dir / 'epoch_{:03d}_{}.png'.format(epoch, safe_name))


def save_results(output_dir, epoch_rows, summary_rows, transition_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_fields = [
        'image',
        'epoch',
        'num_gt_instances',
        'num_A_instances',
        'num_B_instances',
        'matched_AB_instances',
        'unmatched_A_instances',
        'unmatched_B_instances',
        'A_correct_B_wrong_instances',
        'B_correct_A_wrong_instances',
        'A_missing_instances',
        'B_missing_instances',
        'A_false_positive_instances',
        'B_false_positive_instances',
        'A_merge_cases',
        'B_merge_cases',
        'A_split_cases',
        'B_split_cases',
    ]
    for epoch, rows in epoch_rows.items():
        path = output_dir / 'structural_analysis_epoch_{:03d}.csv'.format(epoch)
        with path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=analysis_fields)
            writer.writeheader()
            writer.writerows(rows)

    summary_fields = [
        'epoch',
        'structural_complementarity_rate',
        'A_error_rate',
        'B_error_rate',
        'merge_rate_A',
        'merge_rate_B',
        'split_rate_A',
        'split_rate_B',
        'pixel_disagreement',
        'instance_disagreement',
        'num_gt_instances',
        'num_A_instances',
        'num_B_instances',
        'matched_AB_instances',
        'unmatched_A_instances',
        'unmatched_B_instances',
        'A_correct_B_wrong_instances',
        'B_correct_A_wrong_instances',
        'A_missing_instances',
        'B_missing_instances',
        'A_false_positive_instances',
        'B_false_positive_instances',
        'A_merge_cases',
        'B_merge_cases',
        'A_split_cases',
        'B_split_cases',
    ]
    with (output_dir / 'structural_summary.csv').open(
        'w', newline='', encoding='utf-8'
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    transition_fields = [
        'start_epoch',
        'end_epoch',
        'source_model',
        'transition_type',
        'count',
    ]
    with (output_dir / 'temporal_knowledge_transition.csv').open(
        'w', newline='', encoding='utf-8'
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=transition_fields)
        writer.writeheader()
        writer.writerows(transition_rows)


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError('--threshold must be in [0, 1]')

    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_name, dataset_root = resolve_dataset(args.dataset)
    cfg = dataset_cfg(dataset_name)
    input_mode = (
        infer_input_mode(checkpoint_dir)
        if args.input_mode == 'auto' else args.input_mode
    )
    if args.gpu.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    dataset = build_dataset(
        dataset_name, dataset_root, args.split, input_mode, args.manifest
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / 'visualization'
    visualization_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    visualization_names = set(
        rng.sample(
            [Path(path).name for path in dataset.mask_paths],
            min(args.num_visualizations, len(dataset.mask_paths)),
        )
    )

    epoch_rows = {}
    summary_rows = []
    epoch_states = {}
    previous_epoch = None

    for epoch in args.epochs:
        model_a, model_b = load_model(
            checkpoint_dir, epoch, cfg, args.network, device
        )
        rows = []
        states_for_epoch = {}
        pixel_disagreement_sum = 0
        pixel_count = 0

        with torch.inference_mode():
            for batch in loader:
                input_a, input_b = select_student_inputs(batch)
                input_a = input_a.to(device, non_blocking=True)
                input_b = input_b.to(device, non_blocking=True)
                pred_a_batch = predict_mask(model_a, input_a, args.threshold).cpu().numpy()
                pred_b_batch = predict_mask(model_b, input_b, args.threshold).cpu().numpy()
                gt_batch = batch['mask'].cpu().numpy().astype(bool)

                for batch_index, image_name in enumerate(batch['ID']):
                    pred_a = pred_a_batch[batch_index].astype(bool)
                    pred_b = pred_b_batch[batch_index].astype(bool)
                    gt = gt_batch[batch_index]
                    pixel_disagreement_sum += int(np.logical_xor(pred_a, pred_b).sum())
                    pixel_count += int(pred_a.size)
                    row, states = analyze_structures(
                        image_name, epoch, pred_a, pred_b, gt
                    )
                    rows.append(row)
                    for gt_index, state in states.items():
                        states_for_epoch[(image_name, gt_index)] = state
                    if image_name in visualization_names:
                        make_visualization(
                            gt, pred_a, pred_b, image_name, epoch, visualization_dir
                        )

        epoch_rows[epoch] = rows
        epoch_states[epoch] = states_for_epoch
        summary = aggregate_epoch(
            rows, epoch, pixel_disagreement_sum, pixel_count
        )
        summary_rows.append(summary)
        transition_subset = (
            track_temporal_transition({
                previous_epoch: epoch_states[previous_epoch],
                epoch: states_for_epoch,
            })
            if previous_epoch is not None else []
        )
        erosion_rate = knowledge_erosion_rate(transition_subset)
        print(
            'Epoch {:03d}: pixel disagreement={:.4f} | '
            'instance disagreement={} | structural complementarity rate={:.4f} | '
            'knowledge erosion rate={:.4f} | merge/split A={}/{} B={}/{}'.format(
                epoch,
                summary['pixel_disagreement'],
                summary['instance_disagreement'],
                summary['structural_complementarity_rate'],
                erosion_rate,
                summary['A_merge_cases'],
                summary['A_split_cases'],
                summary['B_merge_cases'],
                summary['B_split_cases'],
            )
        )

        del model_a
        del model_b
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        previous_epoch = epoch

    transition_rows = track_temporal_transition(epoch_states)
    save_results(output_dir, epoch_rows, summary_rows, transition_rows)
    print('Saved structural analysis to: {}'.format(output_dir))


if __name__ == '__main__':
    main()
