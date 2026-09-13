"""Pre-training R224 numerical compatibility gate for Stage G1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from dataload.dataset_2d import imagefloder_itn
from main import (
    EXPECTED_UNI_SHA256,
    UNI_MEAN,
    UNI_STD,
    _build_state_aware_reference_targets,
    _build_state_aware_reference_targets_from_margin,
    _reference_margin,
    _reconstruct_rgb,
    _sha256_file,
    _uni_dense_features,
)
from models.uni import load_uni


def allclose(a, b, name):
    if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
        print(name, 'max_abs_diff=', float((a - b).abs().max().item()), 'shape=', tuple(a.shape), tuple(b.shape))
        raise AssertionError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory', default=str(ROOT / 'checkpoints/glas_labeled10_seed24/reference_memory.npz'))
    parser.add_argument('--output', default=str(ROOT / 'diagnostics/formal_r320_multidataset/protocol/r224_regression.json'))
    parser.add_argument('--uni-weights', default='/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin')
    parser.add_argument('--uni-config', default='/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/config.json')
    args = parser.parse_args()
    device = torch.device('cuda')
    uni, weights_path, _ = load_uni(args.uni_weights, args.uni_config, expected_sha256=EXPECTED_UNI_SHA256)
    assert uni.training is False and all(not p.requires_grad for p in uni.parameters())
    with np.load(args.memory, allow_pickle=False) as z:
        memory_fg = torch.from_numpy(np.asarray(z['memory_fg'], dtype=np.float32)).to(device)
        memory_bg = torch.from_numpy(np.asarray(z['memory_bg'], dtype=np.float32)).to(device)
        assert int(z['token_grid']) == 14 and int(z['feature_dim']) == 1024 and int(z['top_k']) == 5
    context = {
        'uni': uni, 'memory_fg': memory_fg, 'memory_bg': memory_bg,
        'token_grid': 14, 'feature_dim': 1024, 'reference_resolution': 224,
    }
    cfg = dataset_cfg('GlaS')
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg['MEAN'], cfg['STD'])
    dataset = imagefloder_itn(
        manifest_path=str(ROOT / 'datasets/GLaS/manifests/paper_seed_2026/validation.csv'),
        data_transform_1=transforms['val'], data_normalize_1=normalize,
        sup=True, input_mode='rgb_stain', training=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    normalized_rgb = batch['image_rgb'].to(device)
    rgb = _reconstruct_rgb(normalized_rgb, cfg)
    with torch.inference_mode():
        formal_tokens = _uni_dense_features(uni, rgb)
        parameterized_tokens = _uni_dense_features(uni, rgb, 224)
        allclose(formal_tokens, parameterized_tokens, 'R224 UNI dense tokens')
        formal_margin_grid = _reference_margin(formal_tokens, context, 5)
        parameterized_margin_grid = _reference_margin(parameterized_tokens, context, 5)
        allclose(formal_margin_grid, parameterized_margin_grid, 'R224 margin grid')
        formal_margin_grid = formal_margin_grid.reshape(1, 14, 14)
        parameterized_margin_grid = parameterized_margin_grid.reshape(1, 14, 14)
        formal_margin = torch.nn.functional.interpolate(
            formal_margin_grid[:, None], size=(256, 256), mode='bilinear', align_corners=False
        )[:, 0]
        parameterized_margin = torch.nn.functional.interpolate(
            parameterized_margin_grid[:, None], size=(256, 256), mode='bilinear', align_corners=False
        )[:, 0]
        allclose(formal_margin, parameterized_margin, 'R224 dense margin')
        logits_a = torch.zeros((1, 2, 256, 256), device=device)
        logits_b = torch.zeros((1, 2, 256, 256), device=device)
        logits_a[:, 1] = 0.4
        logits_b[:, 1] = -0.2
        protocol_args = SimpleNamespace(reference_resolution=224, uni_k=5)
        old_targets = _build_state_aware_reference_targets(
            logits_a, logits_b, normalized_rgb, context, cfg, protocol_args
        )
        new_targets = _build_state_aware_reference_targets_from_margin(
            logits_a, logits_b, formal_margin
        )
        allclose(old_targets['margin'], formal_margin, 'R224 target margin')
        for key in ('conflict', 'r_a', 'r_b', 'w_a', 'w_b', 'signed_support_a', 'signed_support_b'):
            allclose(old_targets[key], new_targets[key], 'R224 {}'.format(key))
        ce_none = nn.CrossEntropyLoss(reduction='none')
        from main import compute_state_aware_cps_losses
        old_loss = compute_state_aware_cps_losses(ce_none, logits_a, logits_b, old_targets)
        new_loss = compute_state_aware_cps_losses(ce_none, logits_a, logits_b, new_targets)
        allclose(old_loss['total'], new_loss['total'], 'R224 State-Aware loss')
    stagef = ROOT / 'diagnostics/state_aware_failure_analysis/01_reference_overall.csv'
    with stagef.open() as handle:
        aggregate = next(row for row in csv.DictReader(handle) if row['scope'] == 'aggregate')
    assert abs(float(aggregate['auroc']) - 0.943413) < 1e-5
    assert abs(float(aggregate['balanced_accuracy']) - 0.877008) < 1e-5
    result = {
        'status': 'PASS', 'reference_resolution': 224, 'reference_grid': 14,
        'token_count': 196, 'feature_dim': 1024, 'uni_sha256': _sha256_file(weights_path),
        'formal_memory': str(Path(args.memory).resolve()),
        'allclose_atol': 1e-6, 'allclose_rtol': 1e-5,
        'StageF_reference_AUROC': float(aggregate['auroc']),
        'StageF_reference_BAcc': float(aggregate['balanced_accuracy']),
        'training_performed': False, 'optimizer_step': False, 'backward_called': False,
        'test_loaded': False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
