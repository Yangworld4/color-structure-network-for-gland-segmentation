#!/usr/bin/env python3
"""Offline K=5 versus K=50 UNI reference diagnostic on Vanilla CPS milestones.

UNI token records are read from disk, so no UNI forward is performed here.
The only inference is deterministic validation forward for the five existing
Vanilla CPS checkpoints, used to recover the CPS token states and shared-error
masks.  No training, test data, or K re-selection is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as project  # noqa: E402


EPOCHS = [40, 80, 120, 160, 200]
KS = [5, 50]
EPS = 1e-12
DEFAULT_MILESTONES = ROOT / (
    'checkpoints/semi_10_seed_24_rgb/GLaS/'
    'CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5-w=20-7-61-image-input_mode=rgb_rgb-'
    'cps_mode=standard/milestones'
)
DEFAULT_K5_RECORDS = ROOT / 'outputs/native_scale_reference/glas_standard_cps_uni256/token_records.csv'
DEFAULT_K50_RECORDS = ROOT / 'outputs/native_scale_reference/glas_standard_cps_uni256/token_records_k50.csv'
DEFAULT_K5_ALL_RECORDS = ROOT / 'outputs/native_scale_reference/glas_standard_cps_uni256/token_records_all_k5.csv'
DEFAULT_K50_ALL_RECORDS = ROOT / 'outputs/native_scale_reference/glas_standard_cps_uni256/token_records_all_k50.csv'
DEFAULT_OUTPUT = ROOT / 'outputs/k5_vs_k50_diagnostic'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--milestones-dir', type=Path, default=DEFAULT_MILESTONES)
    p.add_argument('--k5-records', type=Path, default=DEFAULT_K5_RECORDS)
    p.add_argument('--k50-records', type=Path, default=DEFAULT_K50_RECORDS)
    p.add_argument('--k5-all-records', type=Path, default=DEFAULT_K5_ALL_RECORDS)
    p.add_argument('--k50-all-records', type=Path, default=DEFAULT_K50_ALL_RECORDS)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--path-dataset', default=str(ROOT / 'datasets/GlaS'))
    p.add_argument('--dataset-name', default='GlaS')
    p.add_argument('--labeled-ratio', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=24)
    return p.parse_args()


def safe(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def pct(values, q):
    values = np.asarray(values, dtype=np.float64)
    return safe(np.percentile(values, q)) if values.size else None


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    np_, nn = int(pos.sum()), int(neg.sum())
    if not np_ or not nn:
        return None
    order = np.argsort(scores, kind='mergesort')
    ordered = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


def load_records(path: Path, expected_k: int):
    if not path.is_file():
        raise FileNotFoundError(path)
    by_image = defaultdict(dict)
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'image_id', 'token_row', 'token_col', 'S_FG', 'S_BG', 'gt_class'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path} missing fields: {sorted(missing)}')
        detected = set()
        for row in reader:
            if row.get('top_k', '').strip():
                detected.add(int(float(row['top_k'])))
            image_id = row['image_id']
            r = int(row['token_row'])
            c = int(row['token_col'])
            gt_text = row['gt_class'].strip().upper()
            gt_value = 1 if gt_text == 'FG' else (0 if gt_text == 'BG' else None)
            by_image[image_id][(r, c)] = {
                's_fg': float(row['S_FG']),
                's_bg': float(row['S_BG']),
                'gt': gt_value,
                'fg_ratio': float(row.get('fg_ratio', 'nan')),
            }
    if detected and detected != {expected_k}:
        raise ValueError(f'{path} contains top_k={sorted(detected)}, expected {expected_k}')
    if not by_image:
        raise ValueError(f'No records in {path}')
    return by_image


def semantic_stats(records):
    labels, deltas = [], []
    for tokens in records.values():
        for item in tokens.values():
            if item['gt'] is None:
                continue
            labels.append(item['gt'])
            deltas.append(item['s_fg'] - item['s_bg'])
    labels = np.asarray(labels, dtype=np.int64)
    deltas = np.asarray(deltas, dtype=np.float64)
    fg, bg = labels == 1, labels == 0
    fg_acc = float((deltas[fg] > 0).mean())
    bg_acc = float((deltas[bg] < 0).mean())
    return {
        'num_tokens': int(labels.size), 'fg_tokens': int(fg.sum()), 'bg_tokens': int(bg.sum()),
        'fg_accuracy': fg_acc, 'bg_accuracy': bg_acc,
        'balanced_accuracy': 0.5 * (fg_acc + bg_acc),
        'overall_accuracy': float((((deltas > 0) & fg) | ((deltas < 0) & bg)).mean()),
        'auroc': auc(labels, deltas),
        'fg_delta_mean': float(deltas[fg].mean()), 'fg_delta_median': float(np.median(deltas[fg])),
        'fg_delta_std': float(deltas[fg].std()), 'fg_delta_p10': pct(deltas[fg], 10),
        'fg_delta_p25': pct(deltas[fg], 25), 'fg_delta_p50': pct(deltas[fg], 50),
        'fg_delta_p75': pct(deltas[fg], 75), 'fg_delta_p90': pct(deltas[fg], 90),
        'bg_delta_mean': float(deltas[bg].mean()), 'bg_delta_median': float(np.median(deltas[bg])),
        'bg_delta_std': float(deltas[bg].std()), 'bg_delta_p10': pct(deltas[bg], 10),
        'bg_delta_p25': pct(deltas[bg], 25), 'bg_delta_p50': pct(deltas[bg], 50),
        'bg_delta_p75': pct(deltas[bg], 75), 'bg_delta_p90': pct(deltas[bg], 90),
        'median_gap': float(np.median(deltas[fg]) - np.median(deltas[bg])),
    }


def project_args(args):
    return SimpleNamespace(
        manifest_dir=None, path_dataset=args.path_dataset, input1='image',
        input_mode='rgb_rgb', labeled_ratio=args.labeled_ratio,
        sup_mark='7', unsup_mark='61', network='unet', cps_mode='standard',
        lr=0.5, momentum=0.9, wd=-5, step_size=50, gamma=0.5,
        warm_up_duration=20,
    )


def checkpoint_for_epoch(directory: Path, epoch: int) -> Path:
    for candidate in (directory / f'epoch_{epoch:03d}.pth', directory / f'epoch_{epoch}.pth'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Missing epoch {epoch} in {directory}')


def validation_loader(args):
    pa = project_args(args)
    cfg = project.dataset_cfg(args.dataset_name)
    datasets, manifest_dir, use_manifests = project.build_datasets(pa, cfg)
    loader = DataLoader(
        datasets['val'], batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return pa, cfg, loader, manifest_dir, use_manifests


def token_scores_for_image(records, image_id):
    if image_id not in records:
        raise KeyError(f'No UNI token records for validation image {image_id}')
    sfg = np.full((16, 16), np.nan, dtype=np.float64)
    sbg = np.full((16, 16), np.nan, dtype=np.float64)
    gt_class = np.full((16, 16), -1, dtype=np.int64)
    for (r, c), item in records[image_id].items():
        sfg[r, c] = item['s_fg']; sbg[r, c] = item['s_bg']
        if item['gt'] is not None:
            gt_class[r, c] = item['gt']
    if np.isnan(sfg).any() or np.isnan(sbg).any():
        raise RuntimeError(f'Incomplete 16x16 records for {image_id}')
    return sfg, sbg, gt_class


def empty_epoch():
    return defaultdict(float)


def add_group(totals, name, mask, ww_ratio, hc_ratio, cc_ratio):
    count = int(mask.sum())
    totals[name + '_tokens'] += count
    totals[name + '_ww_sum'] += float(ww_ratio[mask].sum())
    totals[name + '_hc_sum'] += float(hc_ratio[mask].sum())
    totals[name + '_cc_sum'] += float(cc_ratio[mask].sum())


def finalize_group(totals, name):
    n = totals[name + '_tokens']
    return {
        'tokens': int(n),
        'p_ww': safe(totals[name + '_ww_sum'] / n) if n else None,
        'p_hc90ww': safe(totals[name + '_hc_sum'] / n) if n else None,
        'p_cc': safe(totals[name + '_cc_sum'] / n) if n else None,
    }


def evaluate_epoch(validation, image_ids, record_map):
    logits_a = validation['scores1'].cuda(non_blocking=True)
    logits_b = validation['scores2'].cuda(non_blocking=True)
    masks = validation['masks'].cuda(non_blocking=True)
    prob_a = project._foreground_probability(logits_a)[:, 0]
    prob_b = project._foreground_probability(logits_b)[:, 0]
    totals = empty_epoch()
    d_values = []
    contradiction_values = []
    for idx, image_id in enumerate(image_ids):
        sfg5, sbg5, _ = token_scores_for_image(record_map[5], image_id)
        sfg50, sbg50, _ = token_scores_for_image(record_map[50], image_id)
        gt = masks[idx].float()
        pa = prob_a[idx]
        pb = prob_b[idx]
        token_pa = F.adaptive_avg_pool2d(pa[None, None], (16, 16))[0, 0]
        token_pb = F.adaptive_avg_pool2d(pb[None, None], (16, 16))[0, 0]
        token_occ = F.adaptive_avg_pool2d(gt[None, None], (16, 16))[0, 0]
        hard_a_px, hard_b_px = pa.ge(0.5), pb.ge(0.5)
        agree_px = hard_a_px.eq(hard_b_px)
        correct_a = hard_a_px.eq(gt.eq(1))
        correct_b = hard_b_px.eq(gt.eq(1))
        ww_px = agree_px & ~correct_a
        conf_a_px = torch.maximum(pa, 1.0 - pa)
        conf_b_px = torch.maximum(pb, 1.0 - pb)
        hc_px = ww_px & conf_a_px.ge(0.90) & conf_b_px.ge(0.90)
        cc_px = correct_a & correct_b
        ww_ratio = F.adaptive_avg_pool2d(ww_px.float()[None, None], (16, 16))[0, 0]
        hc_ratio = F.adaptive_avg_pool2d(hc_px.float()[None, None], (16, 16))[0, 0]
        cc_ratio = F.adaptive_avg_pool2d(cc_px.float()[None, None], (16, 16))[0, 0]
        total_pixels = float(gt.numel())
        for k, sfg, sbg in ((5, sfg5, sbg5), (50, sfg50, sbg50)):
            sfg_t = torch.from_numpy(sfg).to(prob_a.device, dtype=prob_a.dtype)
            sbg_t = torch.from_numpy(sbg).to(prob_a.device, dtype=prob_a.dtype)
            agreement = token_pa.ge(0.5).eq(token_pb.ge(0.5))
            consensus_fg = token_pa.ge(0.5)
            s_cur = torch.where(consensus_fg, sfg_t, sbg_t)
            s_opp = torch.where(consensus_fg, sbg_t, sfg_t)
            d = s_opp - s_cur
            contradiction = agreement & d.gt(0.0)
            non_contradiction = agreement & ~contradiction
            token_conf_a = torch.maximum(token_pa, 1.0 - token_pa)
            token_conf_b = torch.maximum(token_pb, 1.0 - token_pb)
            hc_token = contradiction & token_conf_a.ge(0.90) & token_conf_b.ge(0.90)
            direction_valid = (token_occ - 0.5 * (token_pa + token_pb)).abs().gt(0.05)
            uni_direction = torch.where((sfg_t - sbg_t).gt(0.0), 1.0, -1.0)
            gt_direction = token_occ - 0.5 * (token_pa + token_pb)
            direction_correct = uni_direction * gt_direction > 0
            d_np = d[contradiction].detach().cpu().numpy()
            d_values.extend(d_np.tolist())
            contradiction_values.extend(d_np.tolist())
            prefix = f'k{k}_'
            totals[prefix + 'agreement_tokens'] += int(agreement.sum().item())
            totals[prefix + 'contradiction_tokens'] += int(contradiction.sum().item())
            totals[prefix + 'total_tokens'] += int(agreement.numel())
            add_group(totals, prefix + 'contradiction', contradiction, ww_ratio, hc_ratio, cc_ratio)
            add_group(totals, prefix + 'non_contradiction', non_contradiction, ww_ratio, hc_ratio, cc_ratio)
            totals[prefix + 'direction_correct'] += int((contradiction & direction_valid & direction_correct).sum().item())
            totals[prefix + 'direction_valid'] += int((contradiction & direction_valid).sum().item())
            totals[prefix + 'hc_direction_correct'] += int((hc_token & direction_valid & direction_correct).sum().item())
            totals[prefix + 'hc_direction_valid'] += int((hc_token & direction_valid).sum().item())
            totals[prefix + 'ww_pixel_coverage'] += float((ww_px & contradiction.repeat_interleave(16, 0).repeat_interleave(16, 1)).sum().item())
            totals[prefix + 'hc_pixel_coverage'] += float((hc_px & hc_token.repeat_interleave(16, 0).repeat_interleave(16, 1)).sum().item())
            totals[prefix + 'cc_exposure'] += float((cc_px & contradiction.repeat_interleave(16, 0).repeat_interleave(16, 1)).sum().item())
            totals[prefix + 'all_pixels'] += total_pixels
    rows = {}
    for k in KS:
        prefix = f'k{k}_'
        contr = finalize_group(totals, prefix + 'contradiction')
        non = finalize_group(totals, prefix + 'non_contradiction')
        d_arr = np.asarray([x for x in d_values], dtype=np.float64)
        # d_values are appended for both K in the loop; use a second direct
        # pass below from the per-K contradiction token values recorded in the
        # aggregate lists stored by this function.
        rows[k] = {
            'agreement_tokens': int(totals[prefix + 'agreement_tokens']),
            'contradiction_tokens': int(totals[prefix + 'contradiction_tokens']),
            'total_tokens': int(totals[prefix + 'total_tokens']),
            'contradiction_rate': safe(totals[prefix + 'contradiction_tokens'] / totals[prefix + 'total_tokens']),
            'contradiction': contr, 'non_contradiction': non,
            'direction_accuracy': safe(totals[prefix + 'direction_correct'] / totals[prefix + 'direction_valid']) if totals[prefix + 'direction_valid'] else None,
            'hc_direction_accuracy': safe(totals[prefix + 'hc_direction_correct'] / totals[prefix + 'hc_direction_valid']) if totals[prefix + 'hc_direction_valid'] else None,
            'ww_enrichment': safe((contr['p_ww'] or 0.0) / ((non['p_ww'] or 0.0) + EPS)),
            'hc_enrichment': safe((contr['p_hc90ww'] or 0.0) / ((non['p_hc90ww'] or 0.0) + EPS)),
            'ww_pixel_coverage': safe(totals[prefix + 'ww_pixel_coverage'] / totals[prefix + 'all_pixels']),
            'hc_pixel_coverage': safe(totals[prefix + 'hc_pixel_coverage'] / totals[prefix + 'all_pixels']),
            'cc_exposure': safe(totals[prefix + 'cc_exposure'] / totals[prefix + 'all_pixels']),
            '_direction_correct': int(totals[prefix + 'direction_correct']),
            '_direction_valid': int(totals[prefix + 'direction_valid']),
            '_hc_direction_correct': int(totals[prefix + 'hc_direction_correct']),
            '_hc_direction_valid': int(totals[prefix + 'hc_direction_valid']),
            '_ww_pixel_coverage_sum': float(totals[prefix + 'ww_pixel_coverage']),
            '_hc_pixel_coverage_sum': float(totals[prefix + 'hc_pixel_coverage']),
            '_cc_exposure_sum': float(totals[prefix + 'cc_exposure']),
            '_all_pixels': float(totals[prefix + 'all_pixels']),
        }
    return rows


def aggregate_rows(epoch_rows):
    result = {}
    for k in KS:
        totals = defaultdict(float)
        d_all = []
        for row in epoch_rows:
            item = row[k]
            for key in ('agreement_tokens', 'contradiction_tokens', 'total_tokens'):
                totals[key] += item[key]
            for group in ('contradiction', 'non_contradiction'):
                totals[group + '_tokens'] += item[group]['tokens']
                for metric in ('p_ww', 'p_hc90ww', 'p_cc'):
                    if item[group][metric] is not None:
                        totals[group + '_' + metric + '_sum'] += item[group][metric] * item[group]['tokens']
            for key in ('direction_correct', 'direction_valid', 'hc_direction_correct', 'hc_direction_valid',
                        'ww_pixel_coverage', 'hc_pixel_coverage', 'cc_exposure'):
                # These aggregate rates are reconstructed from epoch rows by
                # pixel/token denominators where available below.
                pass
        # Reconstruct weighted group means and pooled rates from epoch rows.
        contr_n = totals['contradiction_tokens']; non_n = totals['non_contradiction_tokens']
        contr = {
            'tokens': int(contr_n),
            'p_ww': safe(totals['contradiction_p_ww_sum'] / contr_n) if contr_n else None,
            'p_hc90ww': safe(totals['contradiction_p_hc90ww_sum'] / contr_n) if contr_n else None,
            'p_cc': safe(totals['contradiction_p_cc_sum'] / contr_n) if contr_n else None,
        }
        non = {
            'tokens': int(non_n),
            'p_ww': safe(totals['non_contradiction_p_ww_sum'] / non_n) if non_n else None,
            'p_hc90ww': safe(totals['non_contradiction_p_hc90ww_sum'] / non_n) if non_n else None,
            'p_cc': safe(totals['non_contradiction_p_cc_sum'] / non_n) if non_n else None,
        }
        direction_correct = sum(r[k]['_direction_correct'] for r in epoch_rows)
        direction_valid = sum(r[k]['_direction_valid'] for r in epoch_rows)
        hc_direction_correct = sum(r[k]['_hc_direction_correct'] for r in epoch_rows)
        hc_direction_valid = sum(r[k]['_hc_direction_valid'] for r in epoch_rows)
        all_pixels = sum(r[k]['_all_pixels'] for r in epoch_rows)
        result[k] = {
            'agreement_tokens': int(totals['agreement_tokens']),
            'contradiction_tokens': int(contr_n),
            'total_tokens': int(totals['total_tokens']),
            'contradiction_rate': safe(contr_n / totals['total_tokens']),
            'contradiction': contr, 'non_contradiction': non,
            'direction_accuracy': safe(direction_correct / direction_valid) if direction_valid else None,
            'hc_direction_accuracy': safe(hc_direction_correct / hc_direction_valid) if hc_direction_valid else None,
            'ww_enrichment': safe((contr['p_ww'] or 0) / ((non['p_ww'] or 0) + EPS)),
            'hc_enrichment': safe((contr['p_hc90ww'] or 0) / ((non['p_hc90ww'] or 0) + EPS)),
            'ww_pixel_coverage': safe(sum(r[k]['_ww_pixel_coverage_sum'] for r in epoch_rows) / all_pixels),
            'hc_pixel_coverage': safe(sum(r[k]['_hc_pixel_coverage_sum'] for r in epoch_rows) / all_pixels),
            'cc_exposure': safe(sum(r[k]['_cc_exposure_sum'] for r in epoch_rows) / all_pixels),
        }
    return result


def d_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'N': int(values.size), 'min': safe(values.min()) if values.size else None,
        'max': safe(values.max()) if values.size else None,
        'mean': safe(values.mean()) if values.size else None,
        'std': safe(values.std()) if values.size else None,
        'p05': pct(values, 5), 'p10': pct(values, 10), 'p25': pct(values, 25),
        'p50': pct(values, 50), 'p75': pct(values, 75), 'p90': pct(values, 90),
        'p95': pct(values, 95), 'p99': pct(values, 99),
    }


def collect_d_values(record_map, image_ids, validation_list):
    out = {5: [], 50: []}
    for validation in validation_list:
        pa = project._foreground_probability(validation['scores1']).cuda()[:, 0]
        pb = project._foreground_probability(validation['scores2']).cuda()[:, 0]
        for idx, image_id in enumerate(image_ids):
            token_pa = F.adaptive_avg_pool2d(pa[idx][None, None], (16, 16))[0, 0]
            token_pb = F.adaptive_avg_pool2d(pb[idx][None, None], (16, 16))[0, 0]
            agreement = token_pa.ge(0.5).eq(token_pb.ge(0.5))
            consensus_fg = token_pa.ge(0.5)
            for k in KS:
                sfg, sbg, _ = token_scores_for_image(record_map[k], image_id)
                sfg = torch.from_numpy(sfg).cuda().float()
                sbg = torch.from_numpy(sbg).cuda().float()
                d = torch.where(consensus_fg, sbg - sfg, sfg - sbg)
                out[k].extend(d[agreement & d.gt(0)].detach().cpu().numpy().tolist())
    return out


def plot_semantic(records, output_dir, k, xlim):
    values = []
    for tokens in records.values():
        for item in tokens.values():
            values.append((item['s_fg'] - item['s_bg'], item['gt']))
    values = np.asarray(values, dtype=np.float64)
    fg, bg = values[values[:, 1] == 1, 0], values[values[:, 1] == 0, 0]
    bins = np.linspace(xlim[0], xlim[1], 45)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='white')
    ax.hist(bg, bins=bins, density=True, alpha=0.35, color='#377eb8', label='GT Background')
    ax.hist(fg, bins=bins, density=True, alpha=0.35, color='#e41a1c', label='GT Gland')
    try:
        from scipy.stats import gaussian_kde
        for arr, color in ((bg, '#377eb8'), (fg, '#e41a1c')):
            x = np.linspace(xlim[0], xlim[1], 500)
            ax.plot(x, gaussian_kde(arr)(x), color=color, linewidth=2)
    except ImportError:
        pass
    ax.axvline(0, color='black', linestyle='--', linewidth=1.3, label='Semantic boundary (Δ=0)')
    ax.set_xlim(*xlim); ax.set_xlabel(r'Memory similarity margin $\Delta=S_{FG}-S_{BG}$')
    ax.set_ylabel('Density'); ax.set_title(f'UNI Semantic Separation (K={k})')
    ax.legend(frameon=False, fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f'figure2_K{k}_semantic_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'figure2_K{k}_semantic_distribution.pdf', bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for CPS milestone validation')
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    semantic_records = {5: load_records(args.k5_records.resolve(), 5), 50: load_records(args.k50_records.resolve(), 50)}
    records = {5: load_records(args.k5_all_records.resolve(), 5), 50: load_records(args.k50_all_records.resolve(), 50)}
    semantic = {k: semantic_stats(semantic_records[k]) for k in KS}
    all_deltas = np.concatenate([
        np.asarray([x['s_fg'] - x['s_bg'] for tokens in semantic_records[k].values() for x in tokens.values()])
        for k in KS
    ])
    xlim = tuple(np.percentile(all_deltas, [0.5, 99.5]))
    pad = max((xlim[1] - xlim[0]) * 0.05, 1e-3); xlim = (xlim[0] - pad, xlim[1] + pad)
    plot_semantic(semantic_records[5], out, 5, xlim); plot_semantic(semantic_records[50], out, 50, xlim)

    pa = project_args(args); cfg = project.dataset_cfg(args.dataset_name)
    datasets, manifest_dir, use_manifests = project.build_datasets(pa, cfg)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    models = (
        project.get_network('unet', cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda(),
        project.get_network('unet', cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda(),
    )
    losses = {'ce': CrossEntropyLoss().cuda(), 'dice': project.DiceLoss().cuda()}
    validation_list = []
    epoch_rows = []
    image_ids = None
    cache_dir = out / 'validation_prediction_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    for epoch in EPOCHS:
        prediction_cache = cache_dir / f'epoch_{epoch:03d}.npz'
        if prediction_cache.is_file():
            cached = np.load(prediction_cache, allow_pickle=False)
            validation = {
                'scores1': torch.from_numpy(cached['scores1']),
                'scores2': torch.from_numpy(cached['scores2']),
                'masks': torch.from_numpy(cached['masks']),
                'names': cached['names'].astype(str),
            }
            print(f'using cached Vanilla CPS predictions for epoch {epoch}')
        else:
            checkpoint = checkpoint_for_epoch(args.milestones_dir.resolve(), epoch)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            models[0].load_state_dict(payload['appearance_state_dict'], strict=True)
            models[1].load_state_dict(payload['structure_state_dict'], strict=True)
            validation = project.validate(models, losses, val_loader)
            np.savez_compressed(
                prediction_cache,
                scores1=validation['scores1'].numpy().astype(np.float32),
                scores2=validation['scores2'].numpy().astype(np.float32),
                masks=validation['masks'].numpy().astype(np.int64),
                names=np.asarray(validation['names'], dtype='U256'),
            )
        ids = [str(x) for x in validation['names']]
        if image_ids is None: image_ids = ids
        elif ids != image_ids: raise RuntimeError('Validation image order changed across checkpoints')
        validation_list.append(validation)
        epoch_rows.append(evaluate_epoch(validation, ids, records))
        print(f'completed Vanilla CPS epoch {epoch}')
    d_values = collect_d_values(records, image_ids, validation_list)
    d_stats = {k: d_distribution(d_values[k]) for k in KS}
    aggregate = aggregate_rows(epoch_rows)

    semantic_rows = []
    for metric in ('auroc', 'balanced_accuracy', 'fg_accuracy', 'bg_accuracy', 'overall_accuracy', 'median_gap'):
        semantic_rows.append({'metric': metric, 'K5': semantic[5][metric], 'K50': semantic[50][metric], 'delta_K50_minus_K5': semantic[50][metric] - semantic[5][metric]})
    write_csv(out / 'semantic_comparison.csv', semantic_rows, ['metric', 'K5', 'K50', 'delta_K50_minus_K5'])
    e4_metric_map = {
        'Contradictions': ('contradiction_tokens', None), 'Contradiction rate': ('contradiction_rate', None),
        'P(WW|contradiction)': ('contradiction', 'p_ww'), 'P(HC90WW|contradiction)': ('contradiction', 'p_hc90ww'),
        'P(CC|contradiction)': ('contradiction', 'p_cc'), 'P(WW|D<=0)': ('non_contradiction', 'p_ww'),
        'P(HC90WW|D<=0)': ('non_contradiction', 'p_hc90ww'), 'P(CC|D<=0)': ('non_contradiction', 'p_cc'),
        'WW enrichment': ('ww_enrichment', None), 'HC90WW enrichment': ('hc_enrichment', None),
        'Direction accuracy': ('direction_accuracy', None), 'HC90 direction accuracy': ('hc_direction_accuracy', None),
        'WW pixel coverage': ('ww_pixel_coverage', None), 'HC90WW pixel coverage': ('hc_pixel_coverage', None),
        'CC exposure': ('cc_exposure', None),
    }
    e4_rows = []
    for metric, (key, subkey) in e4_metric_map.items():
        v5 = aggregate[5][key] if subkey is None else aggregate[5][key][subkey]
        v50 = aggregate[50][key] if subkey is None else aggregate[50][key][subkey]
        e4_rows.append({'metric': metric, 'K5': v5, 'K50': v50, 'delta_K50_minus_K5': (v50 - v5) if v5 is not None and v50 is not None else None})
    write_csv(out / 'e4_comparison.csv', e4_rows, ['metric', 'K5', 'K50', 'delta_K50_minus_K5'])
    d_rows = []
    for metric in ('N', 'min', 'max', 'mean', 'std', 'p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'):
        d_rows.append({'metric': metric, 'K5': d_stats[5][metric], 'K50': d_stats[50][metric], 'delta_K50_minus_K5': d_stats[50][metric] - d_stats[5][metric] if metric != 'N' else d_stats[50][metric] - d_stats[5][metric]})
    write_csv(out / 'd_distribution_comparison.csv', d_rows, ['metric', 'K5', 'K50', 'delta_K50_minus_K5'])

    per_epoch = []
    for epoch, row in zip(EPOCHS, epoch_rows):
        for k in KS:
            item = row[k]
            per_epoch.append({'epoch': epoch, 'K': k, 'N': item['contradiction']['tokens'], 'mean': None, 'p25': None, 'p50': None, 'p75': None, 'p90': None,
                              'contradiction_rate': item['contradiction_rate'], 'P_WW_contradiction': item['contradiction']['p_ww'],
                              'P_HC90WW_contradiction': item['contradiction']['p_hc90ww'], 'P_CC_contradiction': item['contradiction']['p_cc'],
                              'direction_accuracy': item['direction_accuracy']})
    # Recompute per-epoch D quantiles directly from each validation row to keep
    # this table independent of pooled diagnostics.
    for eidx, validation in enumerate(validation_list):
        ids = image_ids; per_epoch_rows = {k: [] for k in KS}
        pa_gpu = project._foreground_probability(validation['scores1']).cuda()[:, 0]
        pb_gpu = project._foreground_probability(validation['scores2']).cuda()[:, 0]
        for idx, image_id in enumerate(ids):
            ta = F.adaptive_avg_pool2d(pa_gpu[idx][None, None], (16, 16))[0, 0]
            tb = F.adaptive_avg_pool2d(pb_gpu[idx][None, None], (16, 16))[0, 0]
            agree = ta.ge(.5).eq(tb.ge(.5)); fg = ta.ge(.5)
            for k in KS:
                sfg, sbg, _ = token_scores_for_image(records[k], image_id)
                sfg = torch.from_numpy(sfg).cuda().float(); sbg = torch.from_numpy(sbg).cuda().float()
                d = torch.where(fg, sbg - sfg, sfg - sbg)
                per_epoch_rows[k].extend(d[agree & d.gt(0)].cpu().numpy().tolist())
        for k in KS:
            vals = np.asarray(per_epoch_rows[k], dtype=np.float64); target = per_epoch[eidx * 2 + (0 if k == 5 else 1)]
            target.update({'mean': safe(vals.mean()) if vals.size else None, 'p25': pct(vals,25), 'p50': pct(vals,50), 'p75': pct(vals,75), 'p90': pct(vals,90)})
    write_csv(out / 'per_epoch_d_comparison.csv', per_epoch, ['epoch','K','N','mean','p25','p50','p75','p90','contradiction_rate','P_WW_contradiction','P_HC90WW_contradiction','P_CC_contradiction','direction_accuracy'])

    median_ratios = {k: safe(max(row[k]['p50'] for row in per_epoch for _ in [0] if row['K'] == k) / (min(row[k]['p50'] for row in per_epoch for _ in [0] if row['K'] == k) + EPS)) for k in []}
    stability = {}
    for k in KS:
        vals = [row['p50'] for row in per_epoch if row['K'] == k and row['p50'] is not None]
        stability[str(k)] = safe(max(vals) / (min(vals) + EPS)) if vals else None
    summary = {
        'selected_k_before_diagnostic': 50, 'k_selection_source': 'labeled training data only',
        'validation_used_for_k_selection': False, 'test_used': False,
        'epochs': EPOCHS, 'validation_images': len(image_ids), 'validation_ids': image_ids,
        'memory_path': str(ROOT / 'checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz'),
        'semantic': {str(k): semantic[k] for k in KS}, 'e4': {str(k): aggregate[k] for k in KS},
        'd_distribution': {str(k): d_stats[k] for k in KS}, 'per_epoch_median_max_over_min': stability,
        'median_ratio_K50_over_K5': d_stats[50]['p50'] / (d_stats[5]['p50'] + EPS),
        'p90_ratio_K50_over_K5': d_stats[50]['p90'] / (d_stats[5]['p90'] + EPS),
        'event_metrics': 'not added; no standalone event-coverage function was required for this diagnostic',
        'outcome': None, 'downstream_supports_K50': None,
    }
    sem_delta = semantic[50]['auroc'] - semantic[5]['auroc']
    e4_delta = aggregate[50]['ww_enrichment'] - aggregate[5]['ww_enrichment']
    if sem_delta > 0 and e4_delta >= 0 and aggregate[50]['direction_accuracy'] >= aggregate[5]['direction_accuracy']:
        outcome, support = 'Outcome A', 'YES'
    elif sem_delta > 0 and abs(e4_delta) < 0.05:
        outcome, support = 'Outcome B', 'MIXED'
    elif sem_delta > 0 and (e4_delta < 0 or aggregate[50]['direction_accuracy'] < aggregate[5]['direction_accuracy']):
        outcome, support = 'Outcome C', 'MIXED'
    else:
        outcome, support = 'Outcome D', 'MIXED'
    summary['outcome'] = outcome; summary['downstream_supports_K50'] = support
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    report = ['# K=5 vs K=50 UNI Memory Diagnostic', '',
              'K=50 已由 labeled training image-level LOO calibration 预先选定。本诊断只比较 K=5 与 K=50，不重新选择 K；未使用 test。', '',
              '## Semantic reference quality', '', '| Metric | K=5 | K=50 | Δ(50−5) |', '|---|---:|---:|---:|']
    for r in semantic_rows: report.append(f"| {r['metric']} | {r['K5']:.6f} | {r['K50']:.6f} | {r['delta_K50_minus_K5']:.6f} |")
    report += ['', '## CPS contradiction diagnostic', '', '| Metric | K=5 | K=50 | Δ(50−5) |', '|---|---:|---:|---:|']
    for r in e4_rows:
        def fmt(x): return 'NA' if x is None else f'{x:.6f}'
        report.append(f"| {r['metric']} | {fmt(r['K5'])} | {fmt(r['K50'])} | {fmt(r['delta_K50_minus_K5'])} |")
    report += ['', '## D scale', '', '| Metric | K=5 | K=50 | Δ(50−5) |', '|---|---:|---:|---:|']
    for r in d_rows: report.append(f"| {r['metric']} | {r['K5']:.6f} | {r['K50']:.6f} | {r['delta_K50_minus_K5']:.6f} |")
    report += ['', f"D median ratio K50/K5 = **{summary['median_ratio_K50_over_K5']:.6f}**.", f"D p90 ratio K50/K5 = **{summary['p90_ratio_K50_over_K5']:.6f}**.", '',
               '## Per-epoch D stability', '', f"K=5 max/min median = {stability['5']:.6f}; K=50 max/min median = {stability['50']:.6f}.",
               'Epoch-level details are in `per_epoch_d_comparison.csv`.', '',
               f'## Diagnostic conclusion: {outcome}', '',
               f'Semantic AUROC change (K=50−K=5) = {sem_delta:.6f}; WW-enrichment change = {e4_delta:.6f}. K=50 remains the pre-selected configuration and was not re-selected using validation.', '',
               f'Does the validation diagnostic support K=50? **{support}**.', '',
               '## Scope', '', 'This diagnostic used the same GLaS validation set and Vanilla CPS milestones at epochs 40/80/120/160/200. No test set, training, optimizer, loss, memory rebuild, or K sweep was used. Event coverage was not added because no standalone event-coverage function was available without introducing a new definition.']
    (out / 'final_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

    print('\n=== K=5 vs K=50 Downstream UNI Diagnostic ===')
    print('K selection source: labeled training data only; selected K before diagnostic: 50; validation used for K selection: false; test used: false')
    print('Semantic:')
    for r in semantic_rows: print(f"  {r['metric']}: K5={r['K5']:.6f} K50={r['K50']:.6f} Delta={r['delta_K50_minus_K5']:.6f}")
    print('D scale:')
    for metric in ('mean','p10','p25','p50','p75','p90'):
        print(f"  {metric}: K5={d_stats[5][metric]:.6f} K50={d_stats[50][metric]:.6f}")
    print(f"  median ratio K50/K5: {summary['median_ratio_K50_over_K5']:.6f}")
    print(f"  p90 ratio K50/K5: {summary['p90_ratio_K50_over_K5']:.6f}")
    print('Outcome:', outcome, '| downstream support K=50:', support)
    print('Outputs:', out)


if __name__ == '__main__':
    main()
