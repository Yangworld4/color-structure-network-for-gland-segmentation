"""Final, post-training-only evaluation for the formal Stage G1 runs.

This file is intentionally separate from ``main.py``.  It refuses to create a
test dataset until all four R320 runs have completed their 200 synchronized
milestones.  Thresholds are selected on validation only and then frozen for
the test pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d  # noqa: E402
from config.dataset_config.dataset_cfg import dataset_cfg  # noqa: E402
from dataload.dataset_2d import imagefloder_itn  # noqa: E402
from main import get_training_views  # noqa: E402
from models.getnetwork import get_network  # noqa: E402


SETTINGS = (('GlaS', 10, 'GLaS10'), ('GlaS', 20, 'GLaS20'), ('CRAG', 10, 'CRAG10'), ('CRAG', 20, 'CRAG20'))


def run_dir(dataset: str, ratio: int) -> Path:
    root = ROOT / 'checkpoints' / f'semi_{ratio}_seed_24_rgbstain_state_aware_refR320' / ('GLaS' if dataset.lower() == 'glas' else 'CRAG')
    candidates = list(root.glob('*/state_aware_training_complete.json'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one completed R320 run for {dataset}{ratio}: {candidates}')
    return candidates[0].parent


def read_manifest_count(path: Path) -> int:
    with path.open(newline='', encoding='utf-8') as h:
        return sum(1 for _ in csv.DictReader(h))


def loaders(dataset: str, ratio: int):
    data_name = 'GLaS' if dataset.lower() == 'glas' else 'CRAG'
    data_root = ROOT / 'datasets' / data_name
    manifest = data_root / 'manifests' / 'paper_seed_2026'
    cfg = dataset_cfg(dataset)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])
    common = dict(data_normalize_1=normalize, sup=True, input_mode='rgb_stain', training=False)
    val = imagefloder_itn(manifest_path=str(manifest / 'validation.csv'), data_transform_1=transforms['val'], **common)
    test = imagefloder_itn(manifest_path=str(manifest / 'test.csv'), data_transform_1=transforms['test'], **common)
    return DataLoader(val, batch_size=2, shuffle=False, num_workers=0), DataLoader(test, batch_size=2, shuffle=False, num_workers=0)


@torch.inference_mode()
def predictions(model, loader, device, branch):
    model.eval()
    rows = []
    for batch in loader:
        x1, x2 = get_training_views(batch)
        logits = model((x1 if branch == 'A' else x2).to(device, non_blocking=True))[0]
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        masks = batch['mask'].numpy().astype(np.uint8)
        ids = [str(x) for x in batch['ID']]
        for image_id, p, y in zip(ids, prob, masks):
            rows.append((image_id, p.astype(np.float32), y))
    return rows


def threshold_and_score(rows):
    scores = np.concatenate([p.ravel() for _, p, _ in rows])
    targets = np.concatenate([y.ravel() for _, _, y in rows]).astype(bool)
    thresholds = np.arange(0, 0.9, 0.02)
    best = (-1.0, 0.0, 0.0)
    for t in thresholds:
        pred = scores > t
        tp = float((pred & targets).sum())
        union = float((pred | targets).sum())
        jac = tp / union if union else 0.0
        if jac > best[0]:
            dice = 2 * tp / (float(pred.sum()) + float(targets.sum())) if (pred.sum() + targets.sum()) else 0.0
            best = (jac, float(t), dice)
    return best[1], best[0], best[2]


def image_metrics(rows, threshold):
    output = []
    for image_id, scores, target in rows:
        pred = scores > threshold
        target = target.astype(bool)
        tp = float((pred & target).sum())
        union = float((pred | target).sum())
        denom = float(pred.sum() + target.sum())
        output.append({'image_id': image_id, 'jaccard': tp / union if union else 0.0, 'dice': 2 * tp / denom if denom else 0.0})
    return output


def state_sources(directory: Path):
    complete = json.loads((directory / 'state_aware_training_complete.json').read_text(encoding='utf-8'))
    return [
        ('best_validation', 'A', directory / Path(complete['best_checkpoint_A']).name, int(complete['best_epochs'][0])),
        ('best_validation', 'B', directory / Path(complete['best_checkpoint_B']).name, int(complete['best_epochs'][1])),
        ('best_synchronized_mean_val', 'A', directory / 'best_synchronized_mean_val.pth', int(complete['best_synchronized_epoch'])),
        ('best_synchronized_mean_val', 'B', directory / 'best_synchronized_mean_val.pth', int(complete['best_synchronized_epoch'])),
        ('epoch200', 'A', directory / 'milestones' / 'epoch_200.pth', 200),
        ('epoch200', 'B', directory / 'milestones' / 'epoch_200.pth', 200),
    ]


def state_dict(payload, branch):
    if 'appearance_state_dict' in payload:
        return payload['appearance_state_dict' if branch == 'A' else 'structure_state_dict']
    return payload


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-dir', default=str(ROOT / 'diagnostics/formal_r320_multidataset'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    analysis = Path(args.analysis_dir).resolve()
    # Test hygiene gate: all formal runs must be complete before any test loader exists.
    directories = {label: run_dir(dataset, ratio) for dataset, ratio, label in SETTINGS}
    for label, directory in directories.items():
        complete = json.loads((directory / 'state_aware_training_complete.json').read_text(encoding='utf-8'))
        assert int(complete['completed_epochs']) == 200 and int(complete['milestone_count']) == 20, (label, complete)
        assert len(list((directory / 'milestones').glob('epoch_*.pth'))) == 20
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    summary = []
    for dataset, ratio, label in SETTINGS:
        directory = directories[label]
        val_loader, test_loader = loaders(dataset, ratio)
        cfg = dataset_cfg(dataset)
        run_summary = []
        for checkpoint_type, branch, path, source_epoch in state_sources(directory):
            payload = torch.load(path, map_location='cpu')
            model = get_network('unet', 3, 2)
            model.load_state_dict(state_dict(payload, branch), strict=True)
            model.to(device)
            val_rows = predictions(model, val_loader, device, branch)
            threshold, _, _ = threshold_and_score(val_rows)
            test_rows = predictions(model, test_loader, device, branch)
            per_image = image_metrics(test_rows, threshold)
            fixed = image_metrics(test_rows, 0.5)
            val_image = image_metrics(val_rows, threshold)
            def mean(rows, key): return float(np.mean([r[key] for r in rows])) if rows else float('nan')
            row = {'dataset': dataset, 'labeled_ratio': ratio, 'setting': label, 'checkpoint_type': checkpoint_type, 'branch': branch, 'validation_selected_epoch': source_epoch, 'threshold_mode': 'validation_frozen', 'threshold': threshold, 'Jaccard': mean(per_image, 'jaccard'), 'Dice': mean(per_image, 'dice'), 'fixed_05_Jaccard': mean(fixed, 'jaccard'), 'fixed_05_Dice': mean(fixed, 'dice'), 'validation_frozen_validation_Jaccard': mean(val_image, 'jaccard'), 'validation_frozen_validation_Dice': mean(val_image, 'dice')}
            summary.append(row)
            run_summary.append(row)
            write_csv(directory / ('final_test_metrics.csv' if checkpoint_type == 'epoch200' and branch == 'A' else f'final_test_metrics_{checkpoint_type}_{branch}.csv'), [row])
            write_csv(directory / f'final_test_per_image_{checkpoint_type}_{branch}.csv', per_image)
            del model
            if device.type == 'cuda': torch.cuda.empty_cache()
        write_csv(directory / 'final_test_metrics.csv', run_summary)
    write_csv(analysis / '07_r320_final_test_summary.csv', summary)
    (analysis / 'test_evaluation_audit.json').write_text(json.dumps({'test_loaded_during_training': False, 'test_sample_accessed_during_training': False, 'test_metric_computed_during_training': False, 'final_test_stage_completed_after_all_formal_runs': True, 'settings': list(directories)}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows': len(summary), 'analysis_dir': str(analysis)}, indent=2))


if __name__ == '__main__':
    main()
