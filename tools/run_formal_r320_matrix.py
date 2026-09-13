"""Sequential formal Stage G1 R320 training launcher.

The launcher intentionally uses one GPU job at a time and stops immediately
on a failed setting.  It never changes the frozen State-Aware equations or
selects a setting from validation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = (
    ('GlaS', 10, 'GLaS10'),
    ('GlaS', 20, 'GLaS20'),
    ('CRAG', 10, 'CRAG10'),
    ('CRAG', 20, 'CRAG20'),
)


def memory_path(analysis_dir: Path, dataset: str, ratio: int) -> Path:
    return analysis_dir / 'memories' / dataset / f'labeled_{ratio}_seed_24' / 'reference_memory_R320.npz'


def run_one(dataset, ratio, label, analysis_dir, args):
    root_name = f'semi_{ratio}_seed_24_rgbstain_state_aware_refR320'
    checkpoint_root = ROOT / 'checkpoints' / root_name
    segmentation_root = ROOT / 'seg_pred' / root_name
    dataset_path = ROOT / ('datasets/GLaS' if dataset.lower() == 'glas' else 'datasets/CRAG')
    log_path = analysis_dir / 'logs' / f'{label}.console.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dataset = 'GLaS' if dataset.lower() == 'glas' else 'CRAG'
    experiment_dir = checkpoint_root / checkpoint_dataset
    completion = next(experiment_dir.glob('*/state_aware_training_complete.json'), None) if experiment_dir.exists() else None
    if completion is not None and args.resume:
        return {'setting': label, 'status': 'SKIPPED_COMPLETED', 'checkpoint_root': str(completion.parent), 'log': str(log_path)}
    if completion is not None and not args.resume:
        raise RuntimeError(f'{label} already completed; pass --resume to preserve and continue: {completion}')
    command = [
        sys.executable, '-u', str(ROOT / 'main.py'),
        '--path_dataset', str(dataset_path), '--dataset_name', dataset,
        '--labeled_ratio', str(ratio), '--input_mode', 'rgb_stain',
        '--cps_mode', 'state_aware', '--sd', '24', '--reference-resolution', '320',
        '--uni_reference_memory', str(memory_path(analysis_dir, dataset, ratio)),
        '--num_epochs', '200', '--batch_size', '2', '--step_size', '50',
        '--lr', '0.5', '--gamma', '0.5', '--unsup_weight', '5',
        '--warm_up_duration', '20', '--milestone_interval', '10',
        '--best_save_interval', '10', '--state_aware_log_interval', '50',
        '--network', 'unet',
    ]
    partial_resume = bool(args.resume and experiment_dir.exists() and list(experiment_dir.glob('*/resume_checkpoint.pth')))
    if partial_resume:
        command.append('--resume')
    start = time.time()
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write('\n=== START {} {} ===\n'.format(datetime.now(timezone.utc).isoformat(), ' '.join(command)))
        handle.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': args.cuda_visible_devices})
    elapsed = time.time() - start
    if completed.returncode != 0:
        raise RuntimeError(f'{label} failed with code {completed.returncode}; inspect {log_path}')
    milestones = sorted((checkpoint_root / checkpoint_dataset).glob('*/milestones/epoch_*.pth'))
    if len(milestones) != 20:
        raise RuntimeError(f'{label} completed without 20 synchronized milestones: {len(milestones)}')
    completion_files = list((checkpoint_root / checkpoint_dataset).glob('*/state_aware_training_complete.json'))
    if len(completion_files) != 1:
        raise RuntimeError(f'{label} completion metadata missing or ambiguous: {completion_files}')
    return {'setting': label, 'status': 'COMPLETED', 'elapsed_seconds': elapsed, 'checkpoint_root': str(completion_files[0].parent), 'log': str(log_path), 'milestone_count': len(milestones)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default=str(ROOT))
    parser.add_argument('--analysis-dir', default=str(ROOT / 'diagnostics/formal_r320_multidataset'))
    parser.add_argument('--run-order', default='GLaS10,GLaS20,CRAG10,CRAG20')
    parser.add_argument('--only', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda-visible-devices', default='0')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument('--final-test-after-all', action='store_true')
    parser.add_argument('--compare-r224', action='store_true')
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_preflight:
        subprocess.run([sys.executable, str(ROOT / 'tools/r320_protocol_regression.py')], cwd=ROOT, check=True)
    requested = [item.strip() for item in args.run_order.split(',') if item.strip()]
    settings = [item for item in SETTINGS if item[2] in requested and (args.only is None or item[2] == args.only)]
    if not settings:
        raise RuntimeError('No requested formal settings')
    for dataset, ratio, label in settings:
        memory = memory_path(analysis_dir, dataset, ratio)
        if not memory.is_file():
            raise FileNotFoundError(f'Missing frozen R320 memory for {label}: {memory}')
    results = []
    for setting in settings:
        result = run_one(*setting, analysis_dir, args)
        results.append(result)
        (analysis_dir / 'training_summaries').mkdir(exist_ok=True)
        (analysis_dir / 'training_summaries' / f'{setting[2]}.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    inventory_path = analysis_dir / '03_training_run_inventory.csv'
    keys = sorted({key for row in results for key in row})
    with inventory_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, indent=2))
    if args.final_test_after_all:
        subprocess.run([sys.executable, str(ROOT / 'tools/final_test_evaluation.py'), '--analysis-dir', str(analysis_dir)], cwd=ROOT, check=True)
    if args.compare_r224:
        subprocess.run([sys.executable, str(ROOT / 'tools/compare_r224_r320.py'), '--analysis-dir', str(analysis_dir)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
