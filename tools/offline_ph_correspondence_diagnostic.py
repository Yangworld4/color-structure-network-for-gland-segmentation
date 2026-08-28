#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import label
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.dataset_config.dataset_cfg import dataset_cfg
from loss.loss_function import DiceLoss
from main import _load_struct_models, build_datasets, validate
from utils.ph_correspondence import (
    build_persistent_relation_conflict_events,
    decompose_h0_branches,
    match_branches_by_birth,
    match_branches_by_support,
)


FIELDS = [
    'epoch',
    'mode',
    'raw_ph_branches_per_image_a',
    'raw_ph_branches_per_image_b',
    'ph_branches_per_gt_gland_mean_a',
    'ph_branches_per_gt_gland_mean_b',
    'ph_branches_per_gt_gland_median_a',
    'ph_branches_per_gt_gland_median_b',
    'ph_branches_per_gt_gland_p90_a',
    'ph_branches_per_gt_gland_p90_b',
    'matched_pairs_per_image',
    'same_gt_entity_match_precision',
    'entity_match_recall',
    'mean_support_iou',
    'median_support_iou',
    'mean_birth_distance',
    'aligned_relation_candidates_per_image',
    'structural_conflicts_per_image',
    'conflict_A_only_correct',
    'conflict_B_only_correct',
    'conflict_both_wrong',
    'conflict_both_correct',
    'complementary_conflict_rate',
    'runtime_sec_per_image',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--epochs', nargs='+', type=int, default=[20, 60, 100, 140, 180])
    parser.add_argument('--output', default=None)
    parser.add_argument('--path-dataset', default='datasets/GLaS')
    parser.add_argument('--dataset-name', default='GlaS')
    parser.add_argument('--input-mode', default='rgb_stain')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--network', default='unet')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--labeled-ratio', type=int, default=10)
    parser.add_argument('--sup-mark', default='9')
    parser.add_argument('--unsup-mark', default='76')
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def build_val_dataset(args, cfg):
    dataset_args = SimpleNamespace(
        manifest_dir=None,
        path_dataset=args.path_dataset,
        input1=args.input1,
        input_mode=args.input_mode,
        labeled_ratio=args.labeled_ratio,
        sup_mark=args.sup_mark,
        unsup_mark=args.unsup_mark,
    )
    datasets, _manifest_dir, _use_manifests = build_datasets(dataset_args, cfg)
    return datasets['val']


def load_models(args, cfg, epoch):
    load_args = SimpleNamespace(
        network=args.network,
        struct_analysis_checkpoint=args.checkpoint_dir,
        struct_analysis_checkpoint_a=None,
        struct_analysis_checkpoint_b=None,
        struct_analysis_epoch=epoch,
    )
    models, _paths, _resolved_epoch = _load_struct_models(load_args, cfg)
    for model in models:
        model.eval()
    return models


def branch_gt_counts(decomposition, gt_labels):
    counts = []
    for component_id in range(1, int(gt_labels.max()) + 1):
        branch_ids = np.unique(decomposition.branch_label_map[gt_labels == component_id])
        branch_ids = branch_ids[branch_ids >= 0]
        counts.append(int(branch_ids.size))
    return counts


def match_gt_stats(matches, gt_labels):
    same = 0
    different = 0
    per_gt_a = {}
    per_gt_b = {}
    for match in matches:
        ay, ax = match['entity_a'].birth_coord
        by, bx = match['entity_b'].birth_coord
        gt_a = int(gt_labels[ay, ax])
        gt_b = int(gt_labels[by, bx])
        if gt_a > 0:
            per_gt_a.setdefault(gt_a, 0)
            per_gt_a[gt_a] += 1
        if gt_b > 0:
            per_gt_b.setdefault(gt_b, 0)
            per_gt_b[gt_b] += 1
        if gt_a > 0 and gt_a == gt_b:
            same += 1
        else:
            different += 1
    recall_denominator = sum(
        min(per_gt_a.get(component_id, 0), per_gt_b.get(component_id, 0))
        for component_id in set(per_gt_a).union(per_gt_b)
    )
    return same, different, same / recall_denominator if recall_denominator else 0.0


def conflict_gt_stats(events, gt_labels):
    counts = {
        'conflict_A_only_correct': 0,
        'conflict_B_only_correct': 0,
        'conflict_both_wrong': 0,
        'conflict_both_correct': 0,
    }
    for event in events:
        left = event['left_match']['entity_a']
        right = event['right_match']['entity_a']
        left_gt = int(gt_labels[left.birth_coord])
        right_gt = int(gt_labels[right.birth_coord])
        gt_connected = left_gt > 0 and left_gt == right_gt
        a_correct = bool(event['connected_a']) == gt_connected
        b_correct = bool(event['connected_b']) == gt_connected
        if a_correct and b_correct:
            counts['conflict_both_correct'] += 1
        elif a_correct:
            counts['conflict_A_only_correct'] += 1
        elif b_correct:
            counts['conflict_B_only_correct'] += 1
        else:
            counts['conflict_both_wrong'] += 1
    return counts


def summarize(values, fn='mean'):
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    if fn == 'median':
        return float(np.median(array))
    if fn == 'p90':
        return float(np.percentile(array, 90))
    return float(np.mean(array))


def analyze_epoch(args, cfg, dataset, epoch):
    models = load_models(args, cfg, epoch)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    losses = {'ce': CrossEntropyLoss().cuda(), 'dice': DiceLoss().cuda()}
    validation = validate(models, losses, dataloader)
    prob_a = torch.softmax(validation['scores1'], dim=1)[:, 1].detach().cpu().numpy()
    prob_b = torch.softmax(validation['scores2'], dim=1)[:, 1].detach().cpu().numpy()
    masks = validation['masks'].detach().cpu().numpy() > 0
    rows = []
    for mode in ('birth_nearest', 'support_identity'):
        totals = {
            'branches_a': [],
            'branches_b': [],
            'gt_counts_a': [],
            'gt_counts_b': [],
            'matched': [],
            'same': 0,
            'different': 0,
            'recall_values': [],
            'support_iou': [],
            'birth_distance': [],
            'relations': [],
            'conflicts': [],
            'runtime': [],
            'conflict_A_only_correct': 0,
            'conflict_B_only_correct': 0,
            'conflict_both_wrong': 0,
            'conflict_both_correct': 0,
        }
        for index in range(prob_a.shape[0]):
            gt_labels, _count = label(masks[index], structure=np.ones((3, 3), dtype=np.uint8))
            dec_a = decompose_h0_branches(prob_a[index], 'A')
            dec_b = decompose_h0_branches(prob_b[index], 'B')
            if mode == 'support_identity':
                matches, _ua, _ub, match_stats = match_branches_by_support(
                    dec_a, dec_b, prob_a[index].shape
                )
            else:
                matches, _ua, _ub, match_stats = match_branches_by_birth(
                    dec_a.branches, dec_b.branches, prob_a[index].shape
                )
            _mask, events, stats = build_persistent_relation_conflict_events(
                prob_a[index:index + 1],
                prob_b[index:index + 1],
                correspondence_mode=mode,
            )
            same, different, recall = match_gt_stats(matches, gt_labels)
            conflict_counts = conflict_gt_stats(events, gt_labels)

            totals['branches_a'].append(stats['raw_ph_branches_a'])
            totals['branches_b'].append(stats['raw_ph_branches_b'])
            totals['gt_counts_a'].extend(branch_gt_counts(dec_a, gt_labels))
            totals['gt_counts_b'].extend(branch_gt_counts(dec_b, gt_labels))
            totals['matched'].append(len(matches))
            totals['same'] += same
            totals['different'] += different
            totals['recall_values'].append(recall)
            totals['support_iou'].extend(match_stats['matched_support_iou_values'])
            totals['birth_distance'].extend(match_stats['birth_distance_values'])
            totals['relations'].append(stats['aligned_relation_candidates'])
            totals['conflicts'].append(stats['connectivity_conflicts'])
            totals['runtime'].append(stats['total_ph_time'])
            for key, value in conflict_counts.items():
                totals[key] += value
        all_conflicts = (
            totals['conflict_A_only_correct']
            + totals['conflict_B_only_correct']
            + totals['conflict_both_wrong']
            + totals['conflict_both_correct']
        )
        rows.append({
            'epoch': epoch,
            'mode': mode,
            'raw_ph_branches_per_image_a': summarize(totals['branches_a']),
            'raw_ph_branches_per_image_b': summarize(totals['branches_b']),
            'ph_branches_per_gt_gland_mean_a': summarize(totals['gt_counts_a']),
            'ph_branches_per_gt_gland_mean_b': summarize(totals['gt_counts_b']),
            'ph_branches_per_gt_gland_median_a': summarize(totals['gt_counts_a'], 'median'),
            'ph_branches_per_gt_gland_median_b': summarize(totals['gt_counts_b'], 'median'),
            'ph_branches_per_gt_gland_p90_a': summarize(totals['gt_counts_a'], 'p90'),
            'ph_branches_per_gt_gland_p90_b': summarize(totals['gt_counts_b'], 'p90'),
            'matched_pairs_per_image': summarize(totals['matched']),
            'same_gt_entity_match_precision': (
                totals['same'] / (totals['same'] + totals['different'])
                if (totals['same'] + totals['different']) else 0.0
            ),
            'entity_match_recall': summarize(totals['recall_values']),
            'mean_support_iou': summarize(totals['support_iou']),
            'median_support_iou': summarize(totals['support_iou'], 'median'),
            'mean_birth_distance': summarize(totals['birth_distance']),
            'aligned_relation_candidates_per_image': summarize(totals['relations']),
            'structural_conflicts_per_image': summarize(totals['conflicts']),
            'conflict_A_only_correct': totals['conflict_A_only_correct'],
            'conflict_B_only_correct': totals['conflict_B_only_correct'],
            'conflict_both_wrong': totals['conflict_both_wrong'],
            'conflict_both_correct': totals['conflict_both_correct'],
            'complementary_conflict_rate': (
                (totals['conflict_A_only_correct'] + totals['conflict_B_only_correct']) / all_conflicts
                if all_conflicts else 0.0
            ),
            'runtime_sec_per_image': summarize(totals['runtime']),
        })
    return rows


def main():
    args = parse_args()
    cfg = dataset_cfg(args.dataset_name)
    dataset = build_val_dataset(args, cfg)
    output = args.output or os.path.join(args.checkpoint_dir, 'ph_correspondence_offline_diagnostics.csv')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    all_rows = []
    for epoch in args.epochs:
        rows = analyze_epoch(args, cfg, dataset, epoch)
        all_rows.extend(rows)
        for row in rows:
            print(
                'epoch={epoch} mode={mode} branches=({raw_ph_branches_per_image_a:.1f},{raw_ph_branches_per_image_b:.1f}) '
                'matched={matched_pairs_per_image:.1f} precision={same_gt_entity_match_precision:.4f} '
                'support_iou={mean_support_iou:.4f} relations={aligned_relation_candidates_per_image:.1f} '
                'conflicts={structural_conflicts_per_image:.1f} runtime={runtime_sec_per_image:.4f}s'.format(**row)
            )
    with open(output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print('Saved:', output)


if __name__ == '__main__':
    main()
