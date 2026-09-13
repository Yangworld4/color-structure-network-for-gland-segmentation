"""Build and audit the four frozen diagnostic/formal R320 UNI memories.

This utility never constructs a student, optimizer, validation loader, or test
dataset.  It only reads the selected labeled manifest for each formal setting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dataset_config.dataset_cfg import dataset_cfg
from main import (
    EXPECTED_UNI_SHA256,
    REFERENCE_MEMORY_ANCHOR_BG,
    REFERENCE_MEMORY_ANCHOR_FG,
    REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
    REFERENCE_MEMORY_SEED,
    _build_reference_memory,
    _load_uni,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def row_id(row):
    return Path(row.get('image', '')).stem


def write_csv(path: Path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_split_audit(root: Path):
    rows = []
    summary = []
    for dataset in ('GlaS', 'CRAG'):
        manifest_dir = root / 'datasets' / ('GLaS' if dataset == 'GlaS' else dataset) / 'manifests' / 'paper_seed_2026'
        for ratio in (10, 20):
            manifests = {
                'train_sup': manifest_dir / f'labeled_{ratio}.csv',
                'train_unsup': manifest_dir / f'unlabeled_{ratio}.csv',
                'validation': manifest_dir / 'validation.csv',
                'test': manifest_dir / 'test.csv',
            }
            records = {split: read_manifest(path) for split, path in manifests.items()}
            ids = {split: [row_id(row) for row in values] for split, values in records.items()}
            all_ids = []
            for split, values in ids.items():
                for image_id in values:
                    all_ids.append((split, image_id))
                    rows.append({
                        'dataset': dataset, 'labeled_ratio': ratio, 'split': split,
                        'image_id': image_id,
                        'manifest_image': next(row['image'] for row in records[split] if row_id(row) == image_id),
                        'split_count': len(values),
                    })
            seen = {}
            for split, image_id in all_ids:
                if image_id in seen:
                    raise RuntimeError(f'Split overlap: {dataset}{ratio} {image_id} in {seen[image_id]} and {split}')
                seen[image_id] = split
            summary.append({
                'dataset': dataset, 'labeled_ratio': ratio,
                'sup_count': len(ids['train_sup']), 'unsup_count': len(ids['train_unsup']),
                'val_count': len(ids['validation']), 'test_count': len(ids['test']),
                'split_overlap': False,
            })
    return rows, summary


def build_memory(dataset, ratio, output_root, uni_weights, uni_config, commit):
    dataset_dir = 'GLaS' if dataset == 'GlaS' else dataset
    manifest_dir = ROOT / 'datasets' / dataset_dir / 'manifests' / 'paper_seed_2026'
    cfg = dataset_cfg(dataset)
    memory_dir = output_root / 'memories' / dataset / f'labeled_{ratio}_seed_24'
    memory_path = memory_dir / 'reference_memory_R320.npz'
    if memory_path.exists():
        raise RuntimeError(f'Refusing to overwrite existing memory: {memory_path}')
    args = SimpleNamespace(
        dataset_name=dataset, labeled_ratio=ratio, batch_size=2,
        input1='image', input_mode='rgb_stain', sup_mark='9', unsup_mark='76',
        cps_mode='state_aware', uni_build_reference_memory=True,
        uni_k=5, reference_resolution=320,
        uni_weights=str(uni_weights), uni_config=str(uni_config),
    )
    logger = logging.getLogger(f'formal_r320_memory.{dataset}.{ratio}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    uni, weights_path, _ = _load_uni(args)
    _build_reference_memory(
        args, cfg, str(manifest_dir), True, uni, memory_path, logger,
        reference_resolution=320,
    )
    with np.load(memory_path, allow_pickle=False) as memory:
        fg = int(memory['memory_fg_count'])
        bg = int(memory['memory_bg_count'])
        assert int(memory['reference_resolution']) == 320
        assert int(memory['token_grid']) == 20
        assert int(memory['feature_dim']) == 1024
        assert int(memory['top_k']) == 5
        assert float(memory['anchor_fg']) == REFERENCE_MEMORY_ANCHOR_FG
        assert float(memory['anchor_bg']) == REFERENCE_MEMORY_ANCHOR_BG
        assert int(memory['memory_per_image_class']) == REFERENCE_MEMORY_PER_IMAGE_PER_CLASS
        assert int(memory['seed']) == REFERENCE_MEMORY_SEED
        assert np.allclose(np.linalg.norm(memory['memory_fg'], axis=1), 1.0, atol=1e-4)
        assert np.allclose(np.linalg.norm(memory['memory_bg'], axis=1), 1.0, atol=1e-4)
    if dataset == 'GlaS' and ratio == 10:
        assert (fg, bg) == (448, 394), (fg, bg)
    metadata = {
        'dataset': dataset, 'labeled_ratio': ratio, 'seed': 24,
        'labeled_image_ids': [row_id(row) for row in read_manifest(manifest_dir / f'labeled_{ratio}.csv')],
        'labeled_count': len(read_manifest(manifest_dir / f'labeled_{ratio}.csv')),
        'resolution': 320, 'grid': 20, 'feature_dim': 1024, 'K': 5,
        'anchor_fg': REFERENCE_MEMORY_ANCHOR_FG, 'anchor_bg': REFERENCE_MEMORY_ANCHOR_BG,
        'cap': REFERENCE_MEMORY_PER_IMAGE_PER_CLASS, 'memory_seed': REFERENCE_MEMORY_SEED,
        'FG_count': fg, 'BG_count': bg, 'UNI_SHA256': sha256_file(weights_path),
        'memory_SHA256': sha256_file(memory_path), 'git_commit': commit,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'labeled training images + GT only',
    }
    (memory_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
    return {
        'dataset': dataset, 'labeled_ratio': ratio, 'resolution': 320, 'grid': 20,
        'feature_dim': 1024, 'K': 5, 'anchor_fg': REFERENCE_MEMORY_ANCHOR_FG,
        'anchor_bg': REFERENCE_MEMORY_ANCHOR_BG, 'cap': REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
        'memory_seed': REFERENCE_MEMORY_SEED, 'FG_count': fg, 'BG_count': bg,
        'memory_path': str(memory_path), 'memory_SHA256': metadata['memory_SHA256'],
        'UNI_SHA256': metadata['UNI_SHA256'], 'source_labeled_count': metadata['labeled_count'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', default=str(ROOT / 'diagnostics/formal_r320_multidataset'))
    parser.add_argument('--uni-weights', default='/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin')
    parser.add_argument('--uni-config', default='/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/config.json')
    args = parser.parse_args()
    output_root = Path(args.analysis_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert sha256_file(Path(args.uni_weights)) == EXPECTED_UNI_SHA256
    split_rows, split_summary = build_split_audit(ROOT)
    write_csv(output_root / '01_dataset_split_audit.csv', split_rows)
    inventory = []
    for dataset in ('GlaS', 'CRAG'):
        for ratio in (10, 20):
            inventory.append(build_memory(dataset, ratio, output_root, Path(args.uni_weights), Path(args.uni_config), commit))
    write_csv(output_root / '02_r320_memory_inventory.csv', inventory)
    (output_root / 'split_summary.json').write_text(json.dumps(split_summary, indent=2, sort_keys=True), encoding='utf-8')
    (output_root / 'memory_build_metadata.json').write_text(json.dumps({'git_commit': commit, 'resolution': 320, 'UNI_SHA256': EXPECTED_UNI_SHA256, 'memory_count': 4}, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'analysis_dir': str(output_root), 'memories': inventory, 'split_summary': split_summary}, indent=2))


if __name__ == '__main__':
    main()
