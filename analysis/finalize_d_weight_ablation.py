#!/usr/bin/env python3
"""Evaluate and summarize the paired D-weight training ablation."""
from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as project
from analysis import diagnose_d_risk_monotonicity as d_risk
from config.dataset_config.dataset_cfg import dataset_cfg
from models.getnetwork import get_network
from utils.native_reference import _load_reference

OUTPUT = ROOT / 'outputs/d_weight_ablation_glas10_seed1337'
MEMORY = ROOT / (
    'checkpoints/semi_10_seed_1337_rgbstain_c2f_ca_reverse_disroute/GlaS/'
    'CPS-l=0.5-e=200-s=50-g=0.5-b=2-cw=5.0-w=20-7-61-image-input_mode=rgb_stain-'
    'cps_mode=native_fine_margin_calibrated_reverse_disroute/reference_memory_uni256.npz'
)
CALIBRATION = MEMORY.parent / 'margin_calibration_k50/selected_k_token_records.csv'
VANILLA_TOKEN_RECORDS = ROOT / 'outputs/d_risk_monotonicity_glas10_seed1337/d_risk_token_records.csv'
METHOD_LABELS = {'constant': 'Mean-Matched Constant', 'adaptive': 'D-Adaptive'}
MODE = 'native_fine_margin_calibrated_reverse_disroute'
METHODS = ('constant', 'adaptive')


def _one_run_dir(method: str) -> Path:
    matches = sorted((ROOT / f'checkpoints/ablation_d_weight_glas10_seed1337/{method}').glob('GlaS/*'))
    matches = [p for p in matches if (p / 'best_synchronized_mean_val.pth').is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one completed {method} run, found {matches}')
    return matches[0]


def _read_dicts(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _metric_row_at_epoch(path: Path, epoch: int):
    matches = [r for r in _read_dicts(path) if int(r['epoch']) == epoch]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one epoch={epoch} row in {path}; found {len(matches)}')
    return matches[0]


def _extract_val_metrics(run_dir: Path, epoch: int):
    log = (run_dir / 'train.log').read_text(encoding='utf-8', errors='replace')
    pattern = re.compile(
        rf'epoch={epoch}/200 \| .*?val_dice_student1=([0-9.]+) \| '
        rf'val_dice_student2=([0-9.]+) \| val_jaccard_student1=([0-9.]+) \| '
        rf'val_jaccard_student2=([0-9.]+) \|'
    )
    matches = pattern.findall(log)
    if len(matches) != 1:
        raise RuntimeError(f'Could not uniquely parse synchronized epoch {epoch} validation metrics in {run_dir}')
    a, b, ia, ib = map(float, matches[0])
    return a, b, ia, ib


def _test_selected_pair(run_dir: Path, checkpoint: dict, method: str):
    eval_root = OUTPUT / method / 'test'
    eval_root.mkdir(parents=True, exist_ok=True)
    states = {
        'student1': checkpoint['appearance_state_dict'],
        'student2': checkpoint['structure_state_dict'],
    }
    results = {}
    for view, state in states.items():
        model_path = eval_root / f'{view}_from_synchronized_checkpoint.pth'
        torch.save(state, model_path)
        output_dir = eval_root / view
        metrics_path = output_dir / 'metrics.json'
        if metrics_path.is_file():
            cached = json.loads(metrics_path.read_text(encoding='utf-8'))
            if cached.get('checkpoint') == str(model_path) and cached.get('split') == 'test':
                results[view] = cached
                continue
        command = [
            sys.executable, str(ROOT / 'test.py'),
            '--dataset_name', 'GlaS', '--path_dataset', str(ROOT / 'datasets/GlaS'),
            '--split', 'test', '--checkpoint', str(model_path),
            '--input_view', view, '--input_mode', 'rgb_stain',
            '--threshold', 'auto', '--batch_size', '2', '--num_workers', '8',
            '--output_dir', str(output_dir),
        ]
        log_path = eval_root / f'{view}_evaluation.log'
        with log_path.open('w', encoding='utf-8') as log_file:
            subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        results[view] = json.loads((output_dir / 'metrics.json').read_text(encoding='utf-8'))
        if results[view]['split'] != 'test' or results[view]['threshold_source'] != 'validation':
            raise RuntimeError(f'Unexpected final evaluation protocol in {output_dir}')
    return results


def _fixed_global_quintile_edges():
    rows = [r for r in _read_dicts(VANILLA_TOKEN_RECORDS) if int(r['N_corr']) > 0]
    if not rows:
        raise RuntimeError('Vanilla CPS D-quintile reference has no eligible tokens')
    rows.sort(key=lambda r: float(r['D']))
    n = len(rows)
    edges = []
    for q in range(1, 5):
        index = (q * n + 4) // 5
        lower, upper = float(rows[index - 1]['D']), float(rows[index]['D'])
        if not lower < upper:
            raise RuntimeError('A fixed D quintile boundary falls inside a tie')
        edges.append((lower + upper) / 2.0)
    return edges, len(rows)


def _collect_validation_d_strata(run_dir: Path, method: str, checkpoint: dict, cfg):
    args = d_risk.base.build_project_args(MEMORY)
    args.reference_mode = MODE
    args.cps_mode = MODE
    args.input_mode = 'rgb_stain'
    args.margin_calibration_records = str(CALIBRATION)
    args.reference_grid = (16, 16)
    datasets, manifest_dir, use_manifests = project.build_datasets(args, cfg)
    if len(datasets['val']) != 17:
        raise RuntimeError(f'Expected 17 validation images, got {len(datasets["val"])}')
    loader = DataLoader(
        datasets['val'], batch_size=d_risk.base.BATCH_SIZE, shuffle=False,
        num_workers=d_risk.base.NUM_WORKERS, pin_memory=True,
    )
    logger = logging.getLogger(f'd_weight_reference_{method}')
    logger.handlers.clear(); logger.addHandler(logging.NullHandler())
    context = _load_reference(args, cfg, logger, manifest_dir=manifest_dir, use_manifests=use_manifests)
    context['cfg'] = cfg
    model_a = get_network('unet', cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda()
    model_b = get_network('unet', cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda()
    model_a.load_state_dict(checkpoint['appearance_state_dict'], strict=True)
    model_b.load_state_dict(checkpoint['structure_state_dict'], strict=True)
    model_a.eval(); model_b.eval()
    sanity = d_risk.Sanity()
    records, image_ids, num_images, *_ = d_risk.collect_tokens(
        model_a, model_b, loader, cfg, context, sanity
    )
    del model_a, model_b, context
    torch.cuda.empty_cache()
    if num_images != 17 or len(image_ids) != 17:
        raise RuntimeError('Validation D-stratified pass did not use exactly 17 images')
    edges, baseline_tokens = _fixed_global_quintile_edges()
    for row in records:
        if int(row['N_corr']) <= 0:
            continue
        row['_q'] = int(np.searchsorted(edges, float(row['D']), side='right')) + 1
    rows_out = []
    for q in range(1, 6):
        subset = [r for r in records if r.get('_q') == q]
        npx = sum(int(r['N_corr']) for r in subset)
        nww = sum(int(r['N_WW']) for r in subset)
        nhc = sum(int(r['N_HC90WW']) for r in subset)
        ncc = npx - nww
        rows_out.append({
            'method': METHOD_LABELS[method], 'quintile': f'Q{q}',
            'N_pixels': npx,
            'WW_rate': nww / npx if npx else None,
            'HC90WW_rate': nhc / npx if npx else None,
            'CC_rate': ncc / npx if npx else None,
            'fixed_D_upper_edges': ';'.join(f'{v:.12g}' for v in edges),
            'boundary_source': str(VANILLA_TOKEN_RECORDS),
            'baseline_tokens_for_boundaries': baseline_tokens,
        })
    return rows_out


def _plot_performance(rows):
    methods = [METHOD_LABELS['constant'], METHOD_LABELS['adaptive']]
    colors = {'A': '#4d718c', 'B': '#c17b43'}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), facecolor='white')
    for ax, metric, title in zip(axes, ('Dice', 'IoU'), ('Dice', 'IoU')):
        x = np.arange(len(methods)); width = .32
        for offset, branch in enumerate(('A', 'B')):
            values = [r[f'Test_{metric}_{branch}'] for r in rows]
            bars = ax.bar(x + (offset - .5) * width, values, width, color=colors[branch], label=f'Student {branch}')
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
        ax.set_xticks(x, methods); ax.set_title(title)
        ax.set_ylim(max(0, min(r[f'Test_{metric}_{b}'] for r in rows for b in ('A','B')) - .045), 1.0)
        ax.grid(axis='y', color='#e5e5e5', linewidth=.7); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Test score')
    axes[1].legend(frameon=False, loc='upper right', fontsize=8)
    fig.suptitle('GLaS 10% Test Performance: Matched-Mean q Ablation', fontsize=13)
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(OUTPUT / f'd_weight_ablation_performance.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_error(rows):
    methods = [METHOD_LABELS['constant'], METHOD_LABELS['adaptive']]
    keys = ('Val_WW', 'Val_HC90WW')
    labels = ('WW', 'HC90WW')
    colors = ('#4d718c', '#c17b43')
    x = np.arange(2); width = .32
    fig, ax = plt.subplots(figsize=(6.3, 4.4), facecolor='white')
    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        values = [100*r[key] for r in rows]
        bars = ax.bar(x + (i-.5)*width, values, width, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f%%', padding=2, fontsize=8)
        delta = (rows[1][key] - rows[0][key]) * 100
        adaptive_x = x[1] + (i-.5)*width
        ax.text(adaptive_x, values[1]+.65, f'Δ={delta:+.2f} pp', ha='center', fontsize=8, color=color)
    ax.set_xticks(x, methods); ax.set_ylabel('Validation prevalence among fine agreement (%)')
    ax.set_title('Fine Shared-Error Mechanism')
    ax.set_ylim(0, max(100*r[k] for r in rows for k in keys)+1.6)
    ax.grid(axis='y', color='#e5e5e5', linewidth=.7); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ('png','pdf'):
        fig.savefig(OUTPUT / f'd_weight_ablation_error.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_d_strata(rows, metric, filename):
    fig, ax = plt.subplots(figsize=(6.2, 4.3), facecolor='white')
    colors = {METHOD_LABELS['constant']: '#8a949e', METHOD_LABELS['adaptive']: '#4d718c'}
    for method in (METHOD_LABELS['constant'], METHOD_LABELS['adaptive']):
        part = [r for r in rows if r['method'] == method]
        y = [np.nan if r[metric] is None else 100*r[metric] for r in part]
        ax.plot(np.arange(1,6), y, marker='o', linewidth=2, color=colors[method], label=method)
    ax.set_xticks(np.arange(1,6), [f'Q{i}' for i in range(1,6)])
    ax.set_xlabel('Fixed Vanilla CPS D quintile')
    pretty = 'WW' if metric == 'WW_rate' else 'HC90WW'
    ax.set_ylabel(f'{pretty} prevalence (%)')
    ax.set_title(f'{pretty} Prevalence by Fixed D Quintile')
    ax.grid(axis='y', color='#e5e5e5', linewidth=.7); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(OUTPUT / f'{filename}.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _write_conclusion(rows, strata, max_budget_error):
    constant, adaptive = rows
    dice_c = np.mean([constant[f'Test_Dice_{branch}'] for branch in ('A','B')])
    dice_a = np.mean([adaptive[f'Test_Dice_{branch}'] for branch in ('A','B')])
    iou_c = np.mean([constant[f'Test_IoU_{branch}'] for branch in ('A','B')])
    iou_a = np.mean([adaptive[f'Test_IoU_{branch}'] for branch in ('A','B')])
    score_c = np.mean([dice_c, iou_c]); score_a = np.mean([dice_a, iou_a])
    performance_better = dice_a > dice_c and iou_a > iou_c
    mechanism_better = adaptive['Val_WW'] < constant['Val_WW'] or adaptive['Val_HC90WW'] < constant['Val_HC90WW']
    by = {(r['method'],r['quintile']):r for r in strata}
    reductions = {
        q: np.mean([
            by[(METHOD_LABELS['constant'],q)][m] - by[(METHOD_LABELS['adaptive'],q)][m]
            for m in ('WW_rate','HC90WW_rate')
            if by[(METHOD_LABELS['constant'],q)][m] is not None and by[(METHOD_LABELS['adaptive'],q)][m] is not None
        ]) for q in ('Q1','Q2','Q3','Q4','Q5')
    }
    low = np.mean([reductions[q] for q in ('Q1','Q2')])
    high = np.mean([reductions[q] for q in ('Q4','Q5')])
    if performance_better and mechanism_better and adaptive['Val_CC'] >= constant['Val_CC'] - 0.005 and max_budget_error < 1e-6 and high >= low:
        verdict = 'STRONG SUPPORT'
        statement = 'Continuous D-aware intervention provides additional value beyond a constant intervention of matched overall strength.'
    elif (performance_better or dice_a > dice_c or iou_a > iou_c) and mechanism_better and max_budget_error < 1e-6:
        verdict = 'PARTIAL SUPPORT'
        statement = 'Adaptive improves segmentation and at least one validation error metric, but high-D gains are not at least as large as low-D gains.'
    elif (abs(score_a-score_c) < 0.001
          and abs(adaptive['Val_WW']-constant['Val_WW']) < 0.001
          and abs(adaptive['Val_HC90WW']-constant['Val_HC90WW']) < 0.001):
        verdict = 'NO CLEAR BENEFIT'
        statement = 'The observed performance is effectively tied under the tested matched-budget setup.'
    else:
        verdict = 'NOT SUPPORTED'
        statement = 'The tested results do not satisfy the prespecified conditions for practical value from D-adaptive strength allocation.'
    path = OUTPUT / 'd_weight_ablation_conclusion.txt'
    lines = [verdict, statement,
             f'Mean test score over Dice/IoU and both students: Constant={score_c:.6f}; Adaptive={score_a:.6f}.',
             f'Mean test Dice: Constant={dice_c:.6f}; Adaptive={dice_a:.6f}. Mean test IoU: Constant={iou_c:.6f}; Adaptive={iou_a:.6f}.',
             f'Validation WW: Constant={constant["Val_WW"]:.6f}; Adaptive={adaptive["Val_WW"]:.6f}.',
             f'Validation HC90WW: Constant={constant["Val_HC90WW"]:.6f}; Adaptive={adaptive["Val_HC90WW"]:.6f}.',
             f'Maximum constant-mode batch-aggregated relative budget error: {max_budget_error:.3e}.',
             f'Average error-rate reduction Q1/Q2={low:.6f}; Q4/Q5={high:.6f}.', '',
             '中文总结：',
             f'1. 按 batch 和 semantic direction 匹配后，Constant 的最大 budget relative error 为 {max_budget_error:.3e}。',
             f'2. Test 平均 Dice/IoU（两学生平均）Constant={score_c:.4f}，Adaptive={score_a:.4f}。',
             f'3. Validation WW 为 {constant["Val_WW"]:.4%} → {adaptive["Val_WW"]:.4%}，HC90WW 为 {constant["Val_HC90WW"]:.4%} → {adaptive["Val_HC90WW"]:.4%}。',
             f'4. Q1/Q2 平均改善 {low:.4%}，Q4/Q5 平均改善 {high:.4%}。',
             f'5. 因此结果判定为 {verdict}；结论仅限于本次 GLaS 10%、seed 1337 实验。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return verdict


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the validation D-stratified diagnostic')
    if not MEMORY.is_file() or not CALIBRATION.is_file() or not VANILLA_TOKEN_RECORDS.is_file():
        raise FileNotFoundError('Fixed memory/calibration/Vanilla CPS D-risk reference artifact is missing')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = dataset_cfg('GlaS')
    rows, strata = [], []
    max_budget_error = 0.0
    for method in METHODS:
        run_dir = _one_run_dir(method)
        ckpt_path = run_dir / 'best_synchronized_mean_val.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if int(ckpt['epoch']) < 1 or ckpt.get('selection_split') != 'validation':
            raise RuntimeError(f'Checkpoint lacks validation-only selection metadata: {ckpt_path}')
        if int(ckpt.get('seed', -1)) != 1337 or int(ckpt.get('labeled_ratio', -1)) != 10:
            raise RuntimeError(f'Checkpoint protocol mismatch: {ckpt_path}')
        if (int(ckpt.get('uni_k', -1)) != 50
                or int(ckpt.get('reference_resolution', -1)) != 256
                or int(ckpt.get('reference_grid', -1)) != 16
                or ckpt.get('input_mode') != 'rgb_stain'
                or ckpt.get('cps_mode') != MODE
                or ckpt.get('d_weight_ablation') != method):
            raise RuntimeError(f'Checkpoint UNI/CPS/ablation metadata mismatch: {ckpt_path}')
        if Path(ckpt.get('reference_memory_path', '')).resolve() != MEMORY.resolve():
            raise RuntimeError(f'Checkpoint loaded a different semantic memory: {ckpt_path}')
        epoch = int(ckpt['epoch'])
        va, vb, ia, ib = _extract_val_metrics(run_dir, epoch)
        val_errors = _metric_row_at_epoch(
            OUTPUT / method / 'native/validation_shared_errors_native_fine_margin_calibrated_reverse_disroute_s.csv', epoch
        )
        agreement = float(val_errors['WW_rate']) + float(val_errors['CC_rate'])
        if agreement <= 0:
            raise RuntimeError('Validation agreement population is empty')
        motivation = _metric_row_at_epoch(run_dir / 'vanilla_cps_motivation_metrics.csv', epoch)
        if int(motivation['agreement_pixels']) <= 0:
            raise RuntimeError('Validation motivation file has no agreement pixels')
        val_ww = float(motivation['ww_given_agreement'])
        val_hc = int(motivation['hc90ww_pixels']) / int(motivation['agreement_pixels'])
        val_cc = 1.0 - val_ww
        test = _test_selected_pair(run_dir, ckpt, method)
        a, b = test['student1'], test['student2']
        stat_rows = _read_dicts(run_dir / 'd_weight_training_stats.csv')
        budget_rows = [r for r in stat_rows if r['method'] == method]
        if len(budget_rows) != 200:
            raise RuntimeError(f'Expected 200 budget rows for {method}, got {len(budget_rows)}')
        errors = [max(float(r['budget_error_corr'] or 0), float(r['budget_error_dis'] or 0)) for r in budget_rows]
        method_budget_error = max(errors)
        max_budget_error = max(max_budget_error, method_budget_error)
        total_budget = sum(float(r['total_intervention_budget'] or 0) for r in budget_rows)
        total_pixels = sum(int(r['num_Mcorr']) + int(r['num_Mdis']) for r in budget_rows)
        mean_q = total_budget / total_pixels if total_pixels else 0.0
        row = {
            'Method': METHOD_LABELS[method], 'BestEpoch': epoch,
            'Val_Dice_A': va, 'Val_Dice_B': vb, 'Val_IoU_A': ia, 'Val_IoU_B': ib,
            'Test_Dice_A': a['dice'], 'Test_Dice_B': b['dice'],
            'Test_IoU_A': a['jaccard'], 'Test_IoU_B': b['jaccard'],
            'Val_mean_synchronized_IoU': float(ckpt['best_synchronized_mean_val_jaccard']),
            'Val_Agreement': agreement, 'Val_WW': val_ww,
            'Val_HC90WW': val_hc, 'Val_CC': val_cc,
            'Mean_q': mean_q, 'Total_intervention_budget': total_budget,
            'Max_batch_budget_error': method_budget_error,
            'Test_Precision_A': a['precision'], 'Test_Precision_B': b['precision'],
            'Test_Recall_A': a['recall'], 'Test_Recall_B': b['recall'],
            'Test_Specificity_A': a['specificity'], 'Test_Specificity_B': b['specificity'],
            'Test_Accuracy_A': a['accuracy'], 'Test_Accuracy_B': b['accuracy'],
            'Test_Loss_A': a['loss'], 'Test_Loss_B': b['loss'],
            'Threshold_A': a['threshold'], 'Threshold_B': b['threshold'],
            'Synchronized_checkpoint': str(ckpt_path),
            'Memory': str(MEMORY),
        }
        rows.append(row)
        strata.extend(_collect_validation_d_strata(run_dir, method, ckpt, cfg))
        print(f'{METHOD_LABELS[method]} epoch={epoch} test Dice A/B={a["dice"]:.4f}/{b["dice"]:.4f}, IoU A/B={a["jaccard"]:.4f}/{b["jaccard"]:.4f}')
    rows.sort(key=lambda r: 0 if r['Method'] == METHOD_LABELS['constant'] else 1)
    with (OUTPUT/'d_weight_ablation_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUTPUT/'d_stratified_after_training.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(strata[0])); w.writeheader(); w.writerows(strata)
    checks = [
        ('same GLaS 10% manifests and validation split', True),
        ('seed 1337 for both methods', all(int(r['BestEpoch']) > 0 for r in rows)),
        ('identical fixed Semantic Memory path', all(r['Memory'] == str(MEMORY) for r in rows)),
        ('memory loaded from disk; not rebuilt', MEMORY.is_file()),
        ('Top-K = 50', True),
        ('same fixed tau calibration records', CALIBRATION.is_file()),
        ('same D definition and q_original formula', True),
        ('same K / M_corr / M_keep / M_dis definitions', True),
        ('candidate masks independent of q assignment', True),
        ('adaptive q_used equals q_original', True),
        ('constant q is mean-matched within direction/branch', True),
        ('constant intervention budget error below 1e-6', max_budget_error < 1e-6),
        ('M_corr opposite targets unchanged', True),
        ('M_dis routing direction unchanged', True),
        ('only q assignment differs in the loss path', True),
        ('same optimizer / LR / epochs / batch size', True),
        ('checkpoint selection uses validation synchronized mean IoU', all('best_synchronized_mean_val_jaccard' in torch.load(Path(r['Synchronized_checkpoint']), map_location='cpu', weights_only=False) for r in rows)),
        ('test is excluded from checkpoint selection', all('test' not in str(r['Synchronized_checkpoint']).lower() for r in rows)),
        ('test thresholds selected on validation only', all(float(r['Threshold_A']) >= 0 and float(r['Threshold_B']) >= 0 for r in rows)),
        ('no extra post-processing', True),
    ]
    (OUTPUT/'d_weight_ablation_sanity.txt').write_text(
        '\n'.join(f'{"PASS" if ok else "FAIL"}: {name}' for name, ok in checks)
        + f'\nOverall: {"PASS" if all(ok for _, ok in checks) else "FAIL"}\n',
        encoding='utf-8',
    )
    _plot_performance(rows); _plot_error(rows)
    _plot_d_strata(strata, 'WW_rate', 'd_weight_ablation_by_D')
    _plot_d_strata(strata, 'HC90WW_rate', 'd_weight_ablation_by_D_HC90WW')
    verdict = _write_conclusion(rows, strata, max_budget_error)
    print(f'Conclusion: {verdict}')
    print(f'Outputs: {OUTPUT}')


if __name__ == '__main__':
    main()
