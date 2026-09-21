#!/usr/bin/env python3
"""Train-only image-level LOO calibration for UNI FG/BG Top-K retrieval.

This script intentionally never constructs validation/test datasets.  It uses
only the labeled manifest, extracts frozen UNI features when no local cache is
available, and excludes the held-out image from both reference classes in
every LOO fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project  # noqa: E402


CANDIDATE_K = [1, 3, 5, 10, 20, 50]
EPS = 1e-12
REFERENCE_RESOLUTION = 256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--memory',
        default=str(ROOT / 'checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz'),
    )
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/train_topk_calibration'))
    parser.add_argument('--dataset-name', default='GlaS')
    parser.add_argument('--path-dataset', default=str(ROOT / 'datasets/GLaS'))
    parser.add_argument('--manifest-dir', default='')
    parser.add_argument('--labeled-ratio', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=jsonable)


def safe_float(value):
    return None if value is None or not np.isfinite(value) else float(value)


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return safe_float(np.percentile(values, q)) if values.size else None


def auc_from_scores(labels, scores):
    """Tie-aware AUROC without adding a sklearn dependency."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if not n_pos or not n_neg:
        return None

    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based and average over tied scores.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = ranks[positive].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_data_args(args):
    # This mirrors the existing reference-memory builder.  rgb_rgb is used
    # deliberately: _get_rgb_view() then selects the exact normalized RGB view
    # used by the current native memory construction path.
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        path_dataset=args.path_dataset,
        manifest_dir=args.manifest_dir or str(Path(args.path_dataset) / 'manifests' / 'paper_seed_2026'),
        labeled_ratio=args.labeled_ratio,
        input1='image',
        input_mode='rgb_rgb',
        sup_mark='9',
        unsup_mark='76',
    )


def extract_or_load_tokens(args, output_dir, device):
    cache_path = output_dir / 'train_token_cache.npz'
    if cache_path.is_file():
        cache = np.load(cache_path, allow_pickle=True)
        required = {'image_ids', 'features', 'occupancy'}
        if required.issubset(set(cache.files)):
            image_ids = [str(x) for x in cache['image_ids'].tolist()]
            features = cache['features'].astype(np.float32)
            occupancy = cache['occupancy'].astype(np.float32)
            if (
                features.ndim == 3 and features.shape[1:] == (256, 1024)
                and occupancy.shape == features.shape[:2]
                and len(image_ids) == features.shape[0]
            ):
                print('Using cached labeled-training UNI tokens:', cache_path)
                return image_ids, features, occupancy, True
        print('Existing token cache is incompatible; rebuilding:', cache_path)

    data_args = build_data_args(args)
    cfg = project.dataset_cfg(args.dataset_name)
    manifest_dir = Path(data_args.manifest_dir)
    labeled_manifest = manifest_dir / f'labeled_{args.labeled_ratio}.csv'
    if not labeled_manifest.is_file():
        raise FileNotFoundError(f'Labeled manifest not found: {labeled_manifest}')

    dataset = project._reference_memory_dataset(
        data_args, cfg, str(manifest_dir), use_manifests=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    uni, _, _ = project.load_uni(
        project.UNI_WEIGHTS_DEFAULT,
        project.UNI_CONFIG_DEFAULT,
        expected_sha256=project.EXPECTED_UNI_SHA256,
    )
    uni.eval()
    features_list = []
    occupancy_list = []
    image_ids = []
    with torch.inference_mode():
        for batch in loader:
            rgb = project._reconstruct_rgb(
                project._get_rgb_view(batch).to(device, non_blocking=True), cfg
            )
            gt = batch['mask'].to(device, non_blocking=True).float().unsqueeze(1)
            # The selected memory is explicitly the validated UNI256 memory;
            # do not inherit the project's legacy 224/14 default here.
            tokens = project._uni_dense_features(
                uni, rgb, reference_resolution=REFERENCE_RESOLUTION
            )
            occupancy = F.adaptive_avg_pool2d(gt, (16, 16)).flatten(1)
            features_list.append(tokens.detach().cpu().numpy().astype(np.float32))
            occupancy_list.append(occupancy.detach().cpu().numpy().astype(np.float32))
            image_ids.extend(str(x) for x in batch['ID'])

    features = np.concatenate(features_list, axis=0)
    occupancy = np.concatenate(occupancy_list, axis=0)
    if features.shape != (len(image_ids), 256, 1024):
        raise RuntimeError(f'Unexpected extracted feature shape: {features.shape}')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_ids=np.asarray(image_ids, dtype='U256'),
        features=features,
        occupancy=occupancy,
        reference_resolution=np.array(REFERENCE_RESOLUTION),
        token_grid=np.array(16),
        feature_dim=np.array(1024),
        source_manifest=np.array(str(labeled_manifest)),
    )
    print('Saved labeled-training UNI token cache:', cache_path)
    return image_ids, features, occupancy, False


def select_per_image_tokens(features, occupancy, seed):
    """Reproduce current native memory selection, retaining source ownership."""
    rng = np.random.default_rng(seed)
    selected = []
    fallback = {'fg': 0, 'bg': 0}
    per_image = []
    cap = int(project.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS)
    for image_index in range(features.shape[0]):
        class_indices = {}
        for cls, threshold, descending, name in (
            (1, project.REFERENCE_MEMORY_ANCHOR_FG, True, 'fg'),
            (0, project.REFERENCE_MEMORY_ANCHOR_BG, False, 'bg'),
        ):
            row = occupancy[image_index]
            if cls == 1:
                candidates = np.flatnonzero(row >= threshold)
                order = np.argsort(-row, kind='stable')
            else:
                candidates = np.flatnonzero(row <= threshold)
                order = np.argsort(row, kind='stable')
            if len(candidates) == 0:
                fallback[name] += 1
                fallback_size = max(
                    1,
                    min(
                        cap,
                        int(math.ceil(len(row) * project.REFERENCE_MEMORY_FALLBACK_FRACTION)),
                    ),
                )
                candidates = order[:fallback_size]
            chosen = candidates
            if len(candidates) > cap:
                chosen = np.sort(rng.choice(candidates, cap, replace=False))
            class_indices[cls] = np.asarray(chosen, dtype=np.int64)
            selected.extend((image_index, cls, int(x)) for x in chosen)
        per_image.append(class_indices)
    return per_image, fallback


def build_loo_memory(features, per_image, held_out):
    fg_parts = []
    bg_parts = []
    owners_fg = []
    owners_bg = []
    for image_index, selected in enumerate(per_image):
        if image_index == held_out:
            continue
        fg_idx = selected[1]
        bg_idx = selected[0]
        fg_parts.append(features[image_index, fg_idx])
        bg_parts.append(features[image_index, bg_idx])
        owners_fg.extend([image_index] * len(fg_idx))
        owners_bg.extend([image_index] * len(bg_idx))
    return (
        np.concatenate(fg_parts, axis=0).astype(np.float32),
        np.concatenate(bg_parts, axis=0).astype(np.float32),
        np.asarray(owners_fg, dtype=np.int64),
        np.asarray(owners_bg, dtype=np.int64),
    )


def metric_bundle(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    fg = labels == 1
    bg = labels == 0
    pred_fg = scores > 0
    pred_bg = scores < 0
    ties = scores == 0
    fg_acc = float((pred_fg[fg]).sum() / fg.sum()) if fg.sum() else None
    bg_acc = float((pred_bg[bg]).sum() / bg.sum()) if bg.sum() else None
    bacc = (fg_acc + bg_acc) / 2.0 if fg_acc is not None and bg_acc is not None else None
    correct = (pred_fg & fg) | (pred_bg & bg)
    return {
        'num_tokens': int(labels.size),
        'num_fg': int(fg.sum()),
        'num_bg': int(bg.sum()),
        'fg_accuracy': fg_acc,
        'bg_accuracy': bg_acc,
        'balanced_accuracy': bacc,
        'overall_accuracy': float(correct.sum() / labels.size) if labels.size else None,
        'auroc': auc_from_scores(labels, scores),
        'ties': int(ties.sum()),
        'fg_delta_mean': safe_float(scores[fg].mean()) if fg.sum() else None,
        'fg_delta_median': safe_float(np.median(scores[fg])) if fg.sum() else None,
        'fg_delta_std': safe_float(scores[fg].std()) if fg.sum() else None,
        'fg_delta_p10': percentile(scores[fg], 10) if fg.sum() else None,
        'fg_delta_p25': percentile(scores[fg], 25) if fg.sum() else None,
        'fg_delta_p50': percentile(scores[fg], 50) if fg.sum() else None,
        'fg_delta_p75': percentile(scores[fg], 75) if fg.sum() else None,
        'fg_delta_p90': percentile(scores[fg], 90) if fg.sum() else None,
        'bg_delta_mean': safe_float(scores[bg].mean()) if bg.sum() else None,
        'bg_delta_median': safe_float(np.median(scores[bg])) if bg.sum() else None,
        'bg_delta_std': safe_float(scores[bg].std()) if bg.sum() else None,
        'bg_delta_p10': percentile(scores[bg], 10) if bg.sum() else None,
        'bg_delta_p25': percentile(scores[bg], 25) if bg.sum() else None,
        'bg_delta_p50': percentile(scores[bg], 50) if bg.sum() else None,
        'bg_delta_p75': percentile(scores[bg], 75) if bg.sum() else None,
        'bg_delta_p90': percentile(scores[bg], 90) if bg.sum() else None,
    }


def select_k(rows):
    valid = [row for row in rows if row['valid']]
    if not valid:
        raise RuntimeError('No candidate K is valid for all LOO folds')
    best_auc = max(row['auroc'] for row in valid)
    auc_pool = [row for row in valid if row['auroc'] >= best_auc - 0.002]
    best_bacc = max(row['balanced_accuracy'] for row in auc_pool)
    bacc_pool = [row for row in auc_pool if row['balanced_accuracy'] >= best_bacc - 0.002]
    best_gap = max(row['median_gap'] for row in bacc_pool)
    gap_pool = [row for row in bacc_pool if row['median_gap'] >= best_gap - 1e-9]
    chosen = sorted(gap_pool, key=lambda row: (row['image_bacc_std'], row['K']))[0]
    return chosen, {
        'max_auroc': best_auc,
        'auroc_pool': [r['K'] for r in auc_pool],
        'max_bacc_within_auroc_pool': best_bacc,
        'bacc_pool': [r['K'] for r in bacc_pool],
        'max_median_gap_within_ties': best_gap,
        'final_pool': [r['K'] for r in gap_pool],
    }


def robust_ranges(rows):
    valid = [r for r in rows if r['valid']]
    stable_pairs = []
    for left, right in zip(valid, valid[1:]):
        if (
            abs(left['auroc'] - right['auroc']) < 0.01
            and abs(left['balanced_accuracy'] - right['balanced_accuracy']) < 0.01
        ):
            stable_pairs.append([left['K'], right['K']])
    return stable_pairs


def make_plots(rows, output_dir):
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r['valid']]
    ks = [r['K'] for r in valid]
    auroc = [r['auroc'] for r in valid]
    bacc = [r['balanced_accuracy'] for r in valid]
    gap = [r['median_gap'] for r in valid]
    for suffix in ('png', 'pdf'):
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.plot(ks, auroc, marker='o', linewidth=2, label='AUROC')
        ax.plot(ks, bacc, marker='s', linewidth=2, label='Balanced Accuracy')
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Score')
        ax.set_ylim(0.0, 1.0)
        ax.set_title('Train-Only Top-K Memory Calibration')
        ax.set_xticks(ks)
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / f'topk_auroc_bacc.{suffix}', dpi=300 if suffix == 'png' else None)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.plot(ks, gap, marker='o', linewidth=2, color='#7b3294')
        ax.set_xlabel('Top-K')
        ax.set_ylabel('Median margin gap')
        ax.set_title('Median Semantic Margin Gap vs Top-K')
        ax.set_xticks(ks)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / f'topk_margin_gap.{suffix}', dpi=300 if suffix == 'png' else None)
        plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('UNI calibration requires CUDA because the project UNI loader is CUDA-backed')

    memory_path = Path(args.memory).resolve()
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)
    memory = np.load(memory_path)
    native_fg_count = int(memory['memory_fg'].shape[0])
    native_bg_count = int(memory['memory_bg'].shape[0])

    image_ids, features, occupancy, used_cache = extract_or_load_tokens(args, output_dir, device)
    per_image, fallback = select_per_image_tokens(features, occupancy, args.seed)
    selected_fg_count = sum(len(x[1]) for x in per_image)
    selected_bg_count = sum(len(x[0]) for x in per_image)
    if selected_fg_count != native_fg_count or selected_bg_count != native_bg_count:
        print(
            'WARNING: recreated per-image selection counts differ from current memory: '
            f'recreated FG/BG={selected_fg_count}/{selected_bg_count}, '
            f'file FG/BG={native_fg_count}/{native_bg_count}'
        )

    fold_sizes = []
    for held_out in range(len(image_ids)):
        fg_mem, bg_mem, _, _ = build_loo_memory(features, per_image, held_out)
        fold_sizes.append({'image_id': image_ids[held_out], 'fg': len(fg_mem), 'bg': len(bg_mem)})
    valid_by_k = {
        k: all(fold['fg'] >= k and fold['bg'] >= k for fold in fold_sizes)
        for k in CANDIDATE_K
    }
    invalid_reason = {
        k: 'LOO memory smaller than K in at least one class/fold'
        for k, valid in valid_by_k.items() if not valid
    }

    token_store = {k: [] for k in CANDIDATE_K}
    per_image_rows = []
    for held_out, image_id in enumerate(image_ids):
        fg_mem, bg_mem, _, _ = build_loo_memory(features, per_image, held_out)
        fg_gpu = torch.from_numpy(fg_mem).to(device)
        bg_gpu = torch.from_numpy(bg_mem).to(device)
        pure_fg = occupancy[held_out] >= project.REFERENCE_MEMORY_ANCHOR_FG
        pure_bg = occupancy[held_out] <= project.REFERENCE_MEMORY_ANCHOR_BG
        query_indices = np.flatnonzero(pure_fg | pure_bg)
        if not len(query_indices):
            continue
        q_gpu = torch.from_numpy(features[held_out, query_indices]).to(device)
        max_k = max(k for k in CANDIDATE_K if valid_by_k[k])
        with torch.inference_mode():
            sim_fg = q_gpu @ fg_gpu.t()
            sim_bg = q_gpu @ bg_gpu.t()
            top_fg = torch.topk(sim_fg, k=max_k, dim=1, largest=True).values.cpu().numpy()
            top_bg = torch.topk(sim_bg, k=max_k, dim=1, largest=True).values.cpu().numpy()
        labels = pure_fg[query_indices].astype(np.int64)
        for k in CANDIDATE_K:
            if not valid_by_k[k]:
                continue
            s_fg = top_fg[:, :k].mean(axis=1)
            s_bg = top_bg[:, :k].mean(axis=1)
            delta = s_fg - s_bg
            for token_id, gt, fg_score, bg_score, margin in zip(
                query_indices, labels, s_fg, s_bg, delta
            ):
                pred = 'FG' if margin > 0 else ('BG' if margin < 0 else 'TIE')
                correct = bool((pred == 'FG' and gt == 1) or (pred == 'BG' and gt == 0))
                token_store[k].append({
                    'image_id': image_id,
                    'token_id': int(token_id),
                    'gt_class': 'FG' if gt == 1 else 'BG',
                    'gt': int(gt),
                    'S_FG': float(fg_score),
                    'S_BG': float(bg_score),
                    'delta': float(margin),
                    'pred_class': pred,
                    'correct': int(correct),
                })
            image_metric = metric_bundle(labels, delta)
            per_image_rows.append({
                'image_id': image_id,
                'K': k,
                'num_tokens': image_metric['num_tokens'],
                'num_fg': image_metric['num_fg'],
                'num_bg': image_metric['num_bg'],
                'accuracy': image_metric['overall_accuracy'],
                'balanced_accuracy': image_metric['balanced_accuracy'],
                'auroc': image_metric['auroc'],
            })

    result_rows = []
    for k in CANDIDATE_K:
        row = {'K': k, 'valid': bool(valid_by_k[k]), 'invalid_reason': invalid_reason.get(k, '')}
        if not valid_by_k[k]:
            result_rows.append(row)
            continue
        records = token_store[k]
        labels = np.asarray([r['gt'] for r in records], dtype=np.int64)
        delta = np.asarray([r['delta'] for r in records], dtype=np.float64)
        metrics = metric_bundle(labels, delta)
        image_subset = [r for r in per_image_rows if r['K'] == k]
        image_bacc = np.asarray([r['balanced_accuracy'] for r in image_subset if r['balanced_accuracy'] is not None], dtype=np.float64)
        image_auc = np.asarray([r['auroc'] for r in image_subset if r['auroc'] is not None], dtype=np.float64)
        row.update(metrics)
        row['median_gap'] = float(metrics['fg_delta_median'] - metrics['bg_delta_median'])
        denom = math.sqrt((metrics['fg_delta_std'] ** 2 + metrics['bg_delta_std'] ** 2) / 2.0) + EPS
        row['standardized_separation'] = float((metrics['fg_delta_mean'] - metrics['bg_delta_mean']) / denom)
        row['image_bacc_mean'] = safe_float(image_bacc.mean()) if image_bacc.size else None
        row['image_bacc_std'] = safe_float(image_bacc.std()) if image_bacc.size else None
        row['image_auroc_mean'] = safe_float(image_auc.mean()) if image_auc.size else None
        row['image_auroc_std'] = safe_float(image_auc.std()) if image_auc.size else None
        result_rows.append(row)

    selected, selection_trace = select_k(result_rows)
    stable_pairs = robust_ranges(result_rows)
    selected_k = int(selected['K'])
    selected_records = token_store[selected_k]

    topk_fields = [
        'K', 'valid', 'invalid_reason', 'num_tokens', 'num_queries', 'num_fg', 'num_bg',
        'fg_accuracy', 'bg_accuracy', 'balanced_accuracy', 'overall_accuracy', 'auroc',
        'fg_delta_mean', 'fg_delta_median', 'fg_delta_std', 'fg_delta_p10', 'fg_delta_p25',
        'fg_delta_p50', 'fg_delta_p75', 'fg_delta_p90', 'bg_delta_mean', 'bg_delta_median',
        'bg_delta_std', 'bg_delta_p10', 'bg_delta_p25', 'bg_delta_p50', 'bg_delta_p75',
        'bg_delta_p90', 'median_gap', 'standardized_separation', 'image_bacc_mean',
        'image_bacc_std', 'image_auroc_mean', 'image_auroc_std', 'ties',
    ]
    with open(output_dir / 'topk_results.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=topk_fields)
        writer.writeheader()
        for row in result_rows:
            output_row = {key: row.get(key, '') for key in topk_fields}
            if row.get('valid'):
                output_row['num_queries'] = row['num_tokens']
            writer.writerow(output_row)

    image_fields = ['image_id', 'K', 'num_tokens', 'num_fg', 'num_bg', 'accuracy', 'balanced_accuracy', 'auroc']
    with open(output_dir / 'topk_per_image_results.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(per_image_rows)

    selected_fields = ['image_id', 'token_id', 'gt_class', 'S_FG', 'S_BG', 'delta', 'pred_class', 'correct']
    with open(output_dir / 'selected_k_token_records.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        for row in selected_records:
            writer.writerow({key: row[key] for key in selected_fields})

    make_plots(result_rows, output_dir)

    selected_json = {key: selected.get(key) for key in (
        'K', 'auroc', 'balanced_accuracy', 'fg_accuracy', 'bg_accuracy',
        'overall_accuracy', 'median_gap', 'standardized_separation',
        'image_bacc_mean', 'image_bacc_std', 'image_auroc_mean', 'image_auroc_std',
    )}
    summary = {
        'selection_data': 'labeled training data only',
        'selection_protocol': 'image-level leave-one-image-out',
        'candidate_k': CANDIDATE_K,
        'selected_k': selected_k,
        'selected_metrics': selected_json,
        'selection_trace': selection_trace,
        'robust_range': stable_pairs,
        'validity_by_k': valid_by_k,
        'fold_sizes': fold_sizes,
        'num_labeled_images': len(image_ids),
        'labeled_image_ids': image_ids,
        'query_pure_definition': {
            'fg': f'occupancy >= {project.REFERENCE_MEMORY_ANCHOR_FG}',
            'bg': f'occupancy <= {project.REFERENCE_MEMORY_ANCHOR_BG}',
            'mixed': 'excluded from semantic calibration',
        },
        'memory_selection_policy': {
            'source_memory': str(memory_path),
            'fg_count_in_file': native_fg_count,
            'bg_count_in_file': native_bg_count,
            'recreated_fg_count': selected_fg_count,
            'recreated_bg_count': selected_bg_count,
            'per_image_cap': int(project.REFERENCE_MEMORY_PER_IMAGE_PER_CLASS),
            'seed': int(args.seed),
            'reference_memory_seed': int(project.REFERENCE_MEMORY_SEED),
            'fallback_count': fallback,
            'feature_normalization': 'UNI dense_patch_tokens and L2-normalized reference features',
        },
        'cache_used': used_cache,
        'memory_path': str(memory_path),
        'validation_used': False,
        'test_used': False,
        'final_recommendation': f'Recommended K for all subsequent validation/test/CPS experiments: K = {selected_k}',
    }
    write_json(output_dir / 'summary.json', summary)

    valid_result_rows = [r for r in result_rows if r['valid']]
    report_lines = [
        '# Train-Only Top-K Memory Calibration',
        '',
        '本实验仅使用 labeled training images，并采用 image-level leave-one-image-out。Validation/test 数据未读取，未使用 Dice/IoU/CPS 指标。',
        '',
        f'- labeled images: {len(image_ids)} ({", ".join(image_ids)})',
        f'- query tokens: 仅保留 FG occupancy ≥ {project.REFERENCE_MEMORY_ANCHOR_FG} 或 BG occupancy ≤ {project.REFERENCE_MEMORY_ANCHOR_BG} 的 native 16×16 tokens。',
        f'- candidate K: {CANDIDATE_K}',
        f'- recreated selected memory count: FG={selected_fg_count}, BG={selected_bg_count}; file count: FG={native_fg_count}, BG={native_bg_count}',
        '',
        '## All candidate results',
        '',
        '| K | valid | AUROC | BAcc | FG Acc | BG Acc | Median Gap | Image BAcc mean±std | Image AUROC mean±std |',
        '|---:|:---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result_rows:
        if not row['valid']:
            report_lines.append(f"| {row['K']} | no | invalid | invalid | invalid | invalid | invalid | invalid | invalid |")
            continue
        report_lines.append(
            f"| {row['K']} | yes | {row['auroc']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['fg_accuracy']:.6f} | {row['bg_accuracy']:.6f} | {row['median_gap']:.6f} | "
            f"{row['image_bacc_mean']:.6f}±{row['image_bacc_std']:.6f} | "
            f"{row['image_auroc_mean']:.6f}±{row['image_auroc_std']:.6f} |"
        )
    report_lines += [
        '',
        '## Selection rule and recommendation',
        '',
        f"按照预注册规则，最高 AUROC 为 {selection_trace['max_auroc']:.6f}，AUROC 容差池为 K={selection_trace['auroc_pool']}；随后以 BAcc 容差池 K={selection_trace['bacc_pool']} 继续筛选。最后比较 MedianGap，并在等价时优先 image-level 稳定性和更小 K。",
        f"因此推荐 **K* = {selected_k}**。推荐指标：AUROC={selected['auroc']:.6f}，BAcc={selected['balanced_accuracy']:.6f}，FG Acc={selected['fg_accuracy']:.6f}，BG Acc={selected['bg_accuracy']:.6f}，MedianGap={selected['median_gap']:.6f}。",
        '',
        f"Recommended K for all subsequent validation/test/CPS experiments: **K = {selected_k}**.",
        '该 K 只由 labeled training data 的 image-level LOO calibration 选择；validation 和 test 没有用于 K selection。',
        '',
        '## K behavior',
        '',
        f"K=1 的 pooled AUROC/BAcc 为 {valid_result_rows[0]['auroc']:.6f}/{valid_result_rows[0]['balanced_accuracy']:.6f}，image BAcc std 为 {valid_result_rows[0]['image_bacc_std']:.6f}；这反映单最近邻是否表现出更大的 image-level 波动。",
        f"最大 K={valid_result_rows[-1]['K']} 的 pooled AUROC/BAcc 为 {valid_result_rows[-1]['auroc']:.6f}/{valid_result_rows[-1]['balanced_accuracy']:.6f}；与中间 K 的差异用于判断大邻域平均是否稀释语义证据。",
        f"满足相邻 AUROC 与 BAcc 差异均 <0.01 的稳定相邻区间：{stable_pairs if stable_pairs else '无'}。",
        '若某一 K 在数值上优于其它候选，报告该结果本身，不将其解释为 validation/test 性能。',
        '',
        '## Outputs',
        '',
        '- `topk_results.csv`：所有候选 K 的 pooled 与 image-level 统计。',
        '- `topk_per_image_results.csv`：每个 LOO image、每个 K 的结果。',
        '- `selected_k_token_records.csv`：推荐 K 的全部 LOO train query token records。',
        '- `summary.json`：选择规则、数据来源和泄漏声明。',
    ]
    with open(output_dir / 'final_report.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(report_lines) + '\n')

    print('\n=== Train-Only Top-K Memory Calibration ===')
    print('Protocol: labeled train only; image-level leave-one-image-out; validation used: false; test used: false')
    print('Labeled images:', len(image_ids), image_ids)
    for row in result_rows:
        if not row['valid']:
            print(f"K={row['K']}: INVALID ({row['invalid_reason']})")
            continue
        print(
            f"K={row['K']} AUROC={row['auroc']:.6f} BAcc={row['balanced_accuracy']:.6f} "
            f"FG={row['fg_accuracy']:.6f} BG={row['bg_accuracy']:.6f} "
            f"MedianGap={row['median_gap']:.6f} "
            f"ImageBAcc={row['image_bacc_mean']:.6f}±{row['image_bacc_std']:.6f}"
        )
    print(f'Best / recommended K: {selected_k}')
    print('Robust adjacent pairs:', stable_pairs if stable_pairs else 'none')
    print('Outputs:', output_dir)


if __name__ == '__main__':
    main()
