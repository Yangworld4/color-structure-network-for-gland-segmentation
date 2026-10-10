"""Frozen UNI reference construction and retrieval helpers.

This module owns reference-memory and margin-calibration details so the trainer
can stay focused on optimization and validation.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.uni_config.uni_config import (
    DEFAULT_REFERENCE_RESOLUTION,
    EXPECTED_UNI_SHA256,
    REFERENCE_MEMORY_ANCHOR_BG,
    REFERENCE_MEMORY_ANCHOR_FG,
    REFERENCE_MEMORY_FALLBACK_FRACTION,
    REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
    REFERENCE_MEMORY_SEED,
    SUPPORTED_REFERENCE_RESOLUTIONS,
    UNI_PATCH_SIZE,
)
from dataload.dataset_2d import imagefloder_itn
from models.uni import load_uni


MARGIN_CALIBRATED_MODES = (
    'native_fine_margin_suppression',
    'native_fine_margin_reverse',
    'native_fine_margin_calibrated_reverse',
    'native_fine_margin_calibrated_reverse_disroute',
)
NATIVE_REFERENCE_MODES = MARGIN_CALIBRATED_MODES
NATIVE_MODE_TAGS = {
    'native_fine_margin_suppression': 'native_fine_margin_suppression_s',
    'native_fine_margin_reverse': 'native_fine_margin_reverse_s',
    'native_fine_margin_calibrated_reverse': 'native_fine_margin_calibrated_reverse_s',
    'native_fine_margin_calibrated_reverse_disroute': 'native_fine_margin_calibrated_reverse_disroute_s',
}


def _reference_grid(reference_resolution, patch_size=UNI_PATCH_SIZE):
    resolution = int(reference_resolution)
    if patch_size == UNI_PATCH_SIZE and resolution not in SUPPORTED_REFERENCE_RESOLUTIONS:
        raise ValueError(
            'reference_resolution must be one of {}'.format(
                SUPPORTED_REFERENCE_RESOLUTIONS
            )
        )
    if resolution % int(patch_size):
        raise ValueError(
            'reference_resolution={} must be divisible by patch size={}'.format(
                resolution, patch_size
            )
        )
    return resolution // int(patch_size)


def _reference_size(reference_resolution):
    resolution = int(reference_resolution)
    if resolution < 1:
        raise ValueError('reference_resolution must be positive')
    return (resolution, resolution)


def _uses_uni_reference(args):
    return getattr(args, 'cps_mode', 'standard') in NATIVE_REFERENCE_MODES


def _is_native_reference_mode(args):
    return getattr(args, 'cps_mode', 'standard') in NATIVE_REFERENCE_MODES


def _is_margin_calibrated_mode(args):
    return getattr(args, 'cps_mode', 'standard') in MARGIN_CALIBRATED_MODES


def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


def _get_rgb_view(batch):
    return batch.get('image_rgb', batch['image'])


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_uni(args):
    loader = getattr(args, 'uni_loader', 'local')
    backbone = getattr(args, 'reference_backbone', 'uni')
    model, weights_path, config_path = load_uni(
        args.uni_weights,
        args.uni_config,
        expected_sha256=EXPECTED_UNI_SHA256,
        loader=loader,
        hub_model_id=getattr(args, 'uni_hub_id', 'MahmoodLab/UNI'),
        backbone=backbone,
    )
    return model, weights_path, config_path


def _reference_memory_dataset(args, cfg, manifest_dir, use_manifests):
    """Build the labeled loader used to create the immutable memory."""
    mean_key, std_key = _normalization_keys(args.input1)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg[mean_key], cfg[std_key])
    if use_manifests:
        source = {
            'manifest_path': os.path.join(
                manifest_dir, 'labeled_{}.csv'.format(args.labeled_ratio)
            )
        }
    else:
        source = {
            'data_dir': os.path.join(
                args.path_dataset, 'train_sup_' + args.sup_mark
            ),
            'input1': args.input1,
        }
    return imagefloder_itn(
        **source,
        data_transform_1=transforms['val'],
        data_normalize_1=normalize,
        sup=True,
        input_mode=args.input_mode,
        training=False,
    )


def _build_reference_memory(
    args,
    cfg,
    manifest_dir,
    use_manifests,
    uni,
    memory_path,
    logger,
    reference_resolution=None,
):
    """Build an immutable memory once, before the CPS optimizer loop."""
    if reference_resolution is None:
        reference_resolution = getattr(args, 'reference_resolution', DEFAULT_REFERENCE_RESOLUTION)
    expected_grid = _reference_grid(reference_resolution, uni.patch_size)
    dataset = _reference_memory_dataset(args, cfg, manifest_dir, use_manifests)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    memory_by_class = {0: [], 1: []}
    rng = np.random.default_rng(REFERENCE_MEMORY_SEED)
    token_grid = None
    feature_dim = None
    fallback_count = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = _reconstruct_rgb(
                _get_rgb_view(batch).cuda(non_blocking=True), cfg
            )
            gt = batch['mask'].cuda(non_blocking=True).float().unsqueeze(1)
            tokens = _uni_dense_features(uni, rgb, reference_resolution)
            grid = int(math.sqrt(int(tokens.shape[1])))
            token_grid = grid if token_grid is None else token_grid
            feature_dim = int(tokens.shape[-1]) if feature_dim is None else feature_dim
            occupancy = F.adaptive_avg_pool2d(gt, (grid, grid)).flatten(1)
            token_cpu = tokens.detach().cpu().numpy()
            occupancy_cpu = occupancy.detach().cpu().numpy()
            for index in range(len(batch['ID'])):
                for cls, threshold in (
                    (1, REFERENCE_MEMORY_ANCHOR_FG),
                    (0, REFERENCE_MEMORY_ANCHOR_BG),
                ):
                    occupancy_row = occupancy_cpu[index]
                    if cls == 1:
                        candidates = np.flatnonzero(occupancy_row >= threshold)
                        order = np.argsort(-occupancy_row, kind='stable')
                    else:
                        candidates = np.flatnonzero(occupancy_row <= threshold)
                        order = np.argsort(occupancy_row, kind='stable')
                    if not len(candidates):
                        fallback_count += 1
                        fallback_size = max(
                            1,
                            min(
                                REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
                                int(math.ceil(len(occupancy_row) * REFERENCE_MEMORY_FALLBACK_FRACTION)),
                            ),
                        )
                        candidates = order[:fallback_size]
                    chosen = candidates
                    if len(candidates) > REFERENCE_MEMORY_PER_IMAGE_PER_CLASS:
                        chosen = np.sort(
                            rng.choice(
                                candidates,
                                REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
                                replace=False,
                            )
                        )
                    for token_index in chosen:
                        memory_by_class[cls].append(token_cpu[index, token_index])

    if token_grid != expected_grid or feature_dim != uni.feature_dim:
        raise RuntimeError(
            'Unexpected generated memory feature shape: grid={} dim={} (expected grid={} dim={})'.format(
                token_grid, feature_dim, expected_grid, uni.feature_dim
            )
        )
    if not memory_by_class[0] or not memory_by_class[1]:
        raise RuntimeError('Generated reference memory is empty for one class')

    memory_fg = F.normalize(
        torch.from_numpy(np.stack(memory_by_class[1])).float(), p=2, dim=1
    ).numpy().astype(np.float32)
    memory_bg = F.normalize(
        torch.from_numpy(np.stack(memory_by_class[0])).float(), p=2, dim=1
    ).numpy().astype(np.float32)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(memory_path),
        memory_fg=memory_fg,
        memory_bg=memory_bg,
        memory_fg_count=np.array(memory_fg.shape[0]),
        memory_bg_count=np.array(memory_bg.shape[0]),
        feature_dim=np.array(feature_dim),
        token_grid=np.array(token_grid),
        # Memory construction is independent of retrieval K.  Keep the legacy
        # key for compatibility, but record the actual retrieval setting too.
        top_k=np.array(int(args.uni_k)),
        memory_construction=np.array('class-balanced native pathology token memory'),
        retrieval_top_k=np.array(int(args.uni_k)),
        anchor_fg=np.array(REFERENCE_MEMORY_ANCHOR_FG),
        anchor_bg=np.array(REFERENCE_MEMORY_ANCHOR_BG),
        memory_per_image_class=np.array(REFERENCE_MEMORY_PER_IMAGE_PER_CLASS),
        anchor_fallback_fraction=np.array(REFERENCE_MEMORY_FALLBACK_FRACTION),
        anchor_fallback_count=np.array(fallback_count),
        source_labeled_ratio=np.array(args.labeled_ratio),
        seed=np.array(REFERENCE_MEMORY_SEED),
        source_dataset=np.array(str(args.dataset_name)),
        reference_resolution=np.array(int(reference_resolution)),
        encoder_id=np.array(uni.model_id),
        encoder_sha256=np.array(uni.weights_sha256),
    )
    logger.info(
        'Reference memory built before training | source=labeled_%s | fg=%d bg=%d '
        '| cap=%d resolution=%d grid=%dx%d dim=%d fallback=%d | path=%s',
        args.labeled_ratio,
        memory_fg.shape[0],
        memory_bg.shape[0],
        REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
        int(reference_resolution),
        token_grid,
        token_grid,
        feature_dim,
        fallback_count,
        memory_path,
    )


def _build_margin_calibration_train_loo(
    args,
    cfg,
    manifest_dir,
    use_manifests,
    uni,
    memory_path,
    records_path,
    logger,
):
    """Build a fixed train-only LOO margin calibration for ``args.uni_k``.

    This runs once before student training.  It uses only labeled training
    images, reproduces the immutable memory anchor selection, and excludes
    each query image from its own FG/BG retrieval memory.
    """
    records_path = Path(records_path).expanduser().resolve()
    summary_path = records_path.with_name('summary.json')
    if records_path.exists() or summary_path.exists():
        raise RuntimeError(
            'Incomplete or conflicting margin calibration exists: {} / {}'.format(
                records_path, summary_path
            )
        )
    k = int(args.uni_k)
    if k < 1:
        raise ValueError('--uni_k must be positive for margin calibration')
    expected_grid = _reference_grid(args.reference_resolution, uni.patch_size)
    dataset = _reference_memory_dataset(args, cfg, manifest_dir, use_manifests)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    image_ids = []
    features = []
    occupancy = []
    with torch.inference_mode():
        for batch in loader:
            rgb = _reconstruct_rgb(
                _get_rgb_view(batch).cuda(non_blocking=True), cfg
            )
            gt = batch['mask'].cuda(non_blocking=True).float().unsqueeze(1)
            tokens = _uni_dense_features(uni, rgb, args.reference_resolution)
            occ = F.adaptive_avg_pool2d(gt, (expected_grid, expected_grid)).flatten(1)
            features.append(tokens.detach().cpu().numpy().astype(np.float32))
            occupancy.append(occ.detach().cpu().numpy().astype(np.float32))
            image_ids.extend(str(value) for value in batch['ID'])
    features = np.concatenate(features, axis=0)
    occupancy = np.concatenate(occupancy, axis=0)
    if features.shape != (len(image_ids), expected_grid * expected_grid, uni.feature_dim):
        raise RuntimeError('Unexpected train calibration feature shape: {}'.format(features.shape))

    rng = np.random.default_rng(REFERENCE_MEMORY_SEED)
    per_image = []
    cap = int(REFERENCE_MEMORY_PER_IMAGE_PER_CLASS)
    fallback_count = 0
    for image_index in range(len(image_ids)):
        selected = {}
        row = occupancy[image_index]
        for cls, threshold, descending in (
            (1, REFERENCE_MEMORY_ANCHOR_FG, True),
            (0, REFERENCE_MEMORY_ANCHOR_BG, False),
        ):
            if cls == 1:
                candidates = np.flatnonzero(row >= threshold)
                order = np.argsort(-row, kind='stable')
            else:
                candidates = np.flatnonzero(row <= threshold)
                order = np.argsort(row, kind='stable')
            if not len(candidates):
                fallback_count += 1
                fallback_size = max(
                    1,
                    min(cap, int(math.ceil(len(row) * REFERENCE_MEMORY_FALLBACK_FRACTION))),
                )
                candidates = order[:fallback_size]
            if len(candidates) > cap:
                candidates = np.sort(rng.choice(candidates, cap, replace=False))
            selected[cls] = np.asarray(candidates, dtype=np.int64)
        per_image.append(selected)

    records = []
    fg_count = 0
    bg_count = 0
    for held_out, image_id in enumerate(image_ids):
        fg_parts = [features[index, per_image[index][1]] for index in range(len(image_ids)) if index != held_out]
        bg_parts = [features[index, per_image[index][0]] for index in range(len(image_ids)) if index != held_out]
        fg_memory = np.concatenate(fg_parts, axis=0)
        bg_memory = np.concatenate(bg_parts, axis=0)
        if len(fg_memory) < k or len(bg_memory) < k:
            raise RuntimeError(
                'Train-only LOO memory is smaller than --uni_k={} for image {}'.format(k, image_id)
            )
        pure_fg = occupancy[held_out] >= REFERENCE_MEMORY_ANCHOR_FG
        pure_bg = occupancy[held_out] <= REFERENCE_MEMORY_ANCHOR_BG
        query_indices = np.flatnonzero(pure_fg | pure_bg)
        if not len(query_indices):
            continue
        query = torch.from_numpy(features[held_out, query_indices]).cuda(non_blocking=True)
        fg_tensor = torch.from_numpy(fg_memory).cuda(non_blocking=True)
        bg_tensor = torch.from_numpy(bg_memory).cuda(non_blocking=True)
        with torch.inference_mode():
            score_fg = torch.topk(query @ fg_tensor.t(), k=k, dim=1).values.mean(dim=1).cpu().numpy()
            score_bg = torch.topk(query @ bg_tensor.t(), k=k, dim=1).values.mean(dim=1).cpu().numpy()
        labels = pure_fg[query_indices].astype(np.int64)
        for token_id, label, s_fg, s_bg in zip(query_indices, labels, score_fg, score_bg):
            delta = float(s_fg - s_bg)
            records.append({
                'image_id': image_id,
                'token_id': int(token_id),
                'gt_class': 'FG' if int(label) == 1 else 'BG',
                'S_FG': float(s_fg),
                'S_BG': float(s_bg),
                'delta': delta,
            })
            if int(label) == 1:
                fg_count += 1
            else:
                bg_count += 1
    if not records:
        raise RuntimeError('Train-only margin calibration produced no pure tokens')
    calibration_class_fallback = 'none'
    if not fg_count and bg_count:
        calibration_class_fallback = 'FG_from_BG'
    elif not bg_count and fg_count:
        calibration_class_fallback = 'BG_from_FG'
    elif not fg_count and not bg_count:
        raise RuntimeError('Train-only margin calibration has no FG or BG records')
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        'selected_k': k,
        'selection_protocol': 'image-level leave-one-image-out',
        'validation_used': False,
        'test_used': False,
        'source_dataset': str(args.dataset_name),
        'source_labeled_ratio': int(args.labeled_ratio),
        'source_memory': str(Path(memory_path).resolve()),
        'reference_resolution': int(args.reference_resolution),
        'token_grid': expected_grid,
        'feature_dim': uni.feature_dim,
        'encoder_id': uni.model_id,
        'encoder_sha256': uni.weights_sha256,
        'fg_token_count': fg_count,
        'bg_token_count': bg_count,
        'fallback_count': fallback_count,
        'calibration_source': 'labeled train-only image-level LOO with frozen UNI and current reference memory selection',
        'calibration_class_fallback': calibration_class_fallback,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(
        'Train-only margin calibration built | K=%d | FG=%d BG=%d | path=%s',
        k, fg_count, bg_count, records_path,
    )


def _load_margin_calibration(args, logger, uni=None):
    """Load fixed labeled-train-only image-level LOO calibration at ``--uni_k``."""
    records_path = Path(args.margin_calibration_records).expanduser().resolve()
    summary_path = records_path.with_name('summary.json')
    if not records_path.is_file():
        raise FileNotFoundError('Margin calibration records not found: {}'.format(records_path))
    if not summary_path.is_file():
        raise FileNotFoundError('Margin calibration summary not found: {}'.format(summary_path))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if uni is not None:
        if int(summary.get('reference_resolution', -1)) != int(args.reference_resolution):
            raise RuntimeError('Margin calibration resolution does not match current encoder input')
        if int(summary.get('feature_dim', -1)) != int(uni.feature_dim):
            raise RuntimeError('Margin calibration feature dimension does not match current encoder')
        if uni.backbone != 'uni':
            if summary.get('encoder_id', '').lower() != uni.model_id.lower():
                raise RuntimeError('{} requires matching train-only margin calibration'.format(uni.backbone))
            if summary.get('encoder_sha256') != uni.weights_sha256:
                raise RuntimeError('Margin calibration was built from different encoder weights')
    expected_k = int(args.uni_k)
    if int(summary.get('selected_k', -1)) != expected_k:
        raise RuntimeError(
            'Margin calibration selected_k={} does not match --uni_k={}'.format(
                summary.get('selected_k'), expected_k
            )
        )
    if summary.get('selection_protocol') != 'image-level leave-one-image-out':
        raise RuntimeError('Margin calibration is not image-level LOO')
    if bool(summary.get('validation_used', True)) or bool(summary.get('test_used', True)):
        raise RuntimeError('Margin calibration summary reports validation/test usage')
    values = {'FG': [], 'BG': []}
    with records_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            gt_class = str(row.get('gt_class', '')).strip().upper()
            if gt_class not in values:
                raise RuntimeError('Unexpected calibration gt_class: {}'.format(gt_class))
            magnitude = abs(float(row['delta']))
            if not math.isfinite(magnitude):
                raise RuntimeError('Non-finite calibration |Delta|')
            values[gt_class].append(magnitude)
    raw_counts = {key: len(value) for key, value in values.items()}
    fallback_mode = str(summary.get('calibration_class_fallback', 'none'))
    if not values['FG'] or not values['BG']:
        if fallback_mode == 'FG_from_BG' and values['BG']:
            values['FG'] = list(values['BG'])
        elif fallback_mode == 'BG_from_FG' and values['FG']:
            values['BG'] = list(values['FG'])
        else:
            raise RuntimeError('Calibration requires non-empty FG and BG records')
    median_fg = float(np.median(np.asarray(values['FG'], dtype=np.float64)))
    median_bg = float(np.median(np.asarray(values['BG'], dtype=np.float64)))
    tau = 0.5 * (median_fg + median_bg)
    if not math.isfinite(tau) or tau <= 0.0:
        raise RuntimeError('Margin calibration tau must be finite and >0: {}'.format(tau))
    stats = {
        'source': 'labeled train-only image-level LOO token records at requested --uni_k',
        'records_path': str(records_path), 'summary_path': str(summary_path),
        'validation_used': False, 'test_used': False, 'K': expected_k,
        'fg_token_count': raw_counts['FG'], 'bg_token_count': raw_counts['BG'],
        'effective_fg_token_count': len(values['FG']),
        'effective_bg_token_count': len(values['BG']),
        'calibration_class_fallback': fallback_mode,
        'fg_mean_abs_delta': float(np.mean(values['FG'])),
        'fg_median_abs_delta': median_fg,
        'fg_p10_abs_delta': float(np.percentile(values['FG'], 10)),
        'fg_p25_abs_delta': float(np.percentile(values['FG'], 25)),
        'fg_p75_abs_delta': float(np.percentile(values['FG'], 75)),
        'fg_p90_abs_delta': float(np.percentile(values['FG'], 90)),
        'bg_mean_abs_delta': float(np.mean(values['BG'])),
        'bg_median_abs_delta': median_bg,
        'bg_p10_abs_delta': float(np.percentile(values['BG'], 10)),
        'bg_p25_abs_delta': float(np.percentile(values['BG'], 25)),
        'bg_p75_abs_delta': float(np.percentile(values['BG'], 75)),
        'bg_p90_abs_delta': float(np.percentile(values['BG'], 90)),
        'tau': tau,
        'tau_definition': (
            'class-balanced typical semantic separation magnitude of the train-only '
            'UNI Memory reference; missing class mirrors the available train-only '
            'class for tiny labeled splits'
            if fallback_mode != 'none' else
            'class-balanced typical semantic separation magnitude of the train-only UNI Memory reference'
        ),
    }
    tau_path = records_path.with_name('tau_calibration.json')
    stats['tau_calibration_path'] = str(tau_path)
    tau_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(
        'Margin calibration loaded | records=%s | FG=%d BG=%d | effective_FG=%d '
        'effective_BG=%d | fallback=%s | m_FG=%.8f | m_BG=%.8f | tau=%.8f',
        records_path,
        stats['fg_token_count'],
        stats['bg_token_count'],
        stats['effective_fg_token_count'],
        stats['effective_bg_token_count'],
        fallback_mode,
        median_fg,
        median_bg,
        tau,
    )
    print('Margin calibration | FG tokens={} | BG tokens={} | fallback={} | m_FG={:.8f} | m_BG={:.8f} | tau={:.8f}'.format(
        stats['fg_token_count'], stats['bg_token_count'], fallback_mode, median_fg, median_bg, tau
    ))
    return stats


def _load_reference(args, cfg, logger, manifest_dir=None, use_manifests=False):
    """Load the immutable UNI model and class-balanced token memory."""
    reference_resolution = int(getattr(args, 'reference_resolution', DEFAULT_REFERENCE_RESOLUTION))
    memory_path = Path(args.uni_reference_memory).expanduser().resolve()
    uni, weights_path, config_path = _load_uni(args)
    expected_grid = _reference_grid(reference_resolution, uni.patch_size)
    requested_grid = getattr(args, 'reference_grid', expected_grid)
    if _is_native_reference_mode(args) and expected_grid != requested_grid:
        raise RuntimeError(
            'Native CPS reference grid mismatch: expected {}x{}, got {}x{} '
            'from resolution={} and patch={}'.format(
                requested_grid, requested_grid, expected_grid, expected_grid,
                reference_resolution, uni.patch_size
            )
        )
    memory_built = False
    if not memory_path.is_file():
        if (
            _is_native_reference_mode(args) and not args.uni_build_reference_memory
        ):
            raise RuntimeError(
                'Native reference memory is absent; use --uni_build_reference_memory '
                'to build it from the current labeled split: {}'.format(
                    memory_path
                )
            )
        if manifest_dir is None:
            raise RuntimeError(
                'manifest_dir is required when building reference memory'
            )
        logger.info(
            'Reference memory not found; building from current labeled split | path=%s',
            memory_path,
        )
        _build_reference_memory(
            args,
            cfg,
            manifest_dir,
            use_manifests,
            uni,
            memory_path,
            logger,
            reference_resolution=reference_resolution,
        )
        memory_built = True
    elif _is_native_reference_mode(args):
        logger.info(
            'Reference memory exists; reusing without rebuilding | path=%s',
            memory_path,
        )
    if not memory_path.is_file():
        raise FileNotFoundError('Reference memory not found: {}'.format(memory_path))

    with np.load(str(memory_path), allow_pickle=False) as memory:
        required = {
            'memory_fg', 'memory_bg', 'memory_fg_count', 'memory_bg_count',
            'feature_dim', 'token_grid', 'top_k',
        }
        missing = sorted(required.difference(memory.files))
        if missing:
            raise RuntimeError('Reference memory missing keys: {}'.format(missing))
        memory_fg_np = np.asarray(memory['memory_fg'], dtype=np.float32)
        memory_bg_np = np.asarray(memory['memory_bg'], dtype=np.float32)
        memory_fg_count = int(memory['memory_fg_count'])
        memory_bg_count = int(memory['memory_bg_count'])
        feature_dim = int(memory['feature_dim'])
        token_grid = int(memory['token_grid'])
        memory_top_k = int(memory['top_k'])

    assert memory_fg_count == memory_fg_np.shape[0], memory_fg_count
    assert memory_bg_count == memory_bg_np.shape[0], memory_bg_count
    if (
        uni.model_id.lower() == 'mahmoodlab/uni'
        and reference_resolution == 224
        and str(args.dataset_name).strip().lower() == 'glas'
        and args.labeled_ratio == 10
    ):
        assert memory_fg_count == 391, memory_fg_count
        assert memory_bg_count == 294, memory_bg_count
    assert memory_fg_np.shape[1] == uni.feature_dim, memory_fg_np.shape
    assert memory_bg_np.shape[1] == uni.feature_dim, memory_bg_np.shape
    assert feature_dim == uni.feature_dim, feature_dim
    assert token_grid == expected_grid, (token_grid, expected_grid)
    with np.load(str(memory_path), allow_pickle=False) as memory:
        memory_encoder_id = (
            str(memory['encoder_id'].item()) if 'encoder_id' in memory.files else None
        )
        memory_encoder_sha = (
            str(memory['encoder_sha256'].item()) if 'encoder_sha256' in memory.files else None
        )
    if memory_encoder_id is not None and memory_encoder_id.lower() != uni.model_id.lower():
        raise RuntimeError(
            'Reference memory encoder mismatch: memory={} current={}'.format(
                memory_encoder_id, uni.model_id
            )
        )
    if memory_encoder_sha is not None and memory_encoder_sha != uni.weights_sha256:
        raise RuntimeError('Reference memory was built from different encoder weights')
    if uni.backbone != 'uni' and memory_encoder_id is None:
        raise RuntimeError(
            '{} requires a memory built by this encoder backend; do not reuse UNI memory'.format(
                uni.backbone
            )
        )
    if uni.backbone != 'uni' and memory_encoder_sha is None:
        raise RuntimeError('Reference memory is missing its encoder weight fingerprint')
    if memory_top_k < 1:
        raise RuntimeError('Reference memory metadata top_k must be positive: {}'.format(memory_top_k))
    with np.load(str(memory_path), allow_pickle=False) as memory:
        source_ratio = int(memory['source_labeled_ratio']) if 'source_labeled_ratio' in memory.files else 10
        source_dataset = (
            str(memory['source_dataset'].item())
            if 'source_dataset' in memory.files else None
        )
        memory_resolution = int(memory['reference_resolution']) if 'reference_resolution' in memory.files else 224
    assert memory_resolution == reference_resolution, (memory_resolution, reference_resolution)
    assert source_ratio == args.labeled_ratio, (source_ratio, args.labeled_ratio)
    if source_dataset is not None:
        assert source_dataset.strip().lower() == str(args.dataset_name).strip().lower(), (
            source_dataset, args.dataset_name
        )
    assert np.allclose(np.linalg.norm(memory_fg_np, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(memory_bg_np, axis=1), 1.0, atol=1e-4)
    requested_k = int(args.uni_k)
    max_supported_k = min(memory_fg_count, memory_bg_count)
    if requested_k < 1:
        raise ValueError(
            'Requested --uni_k={} must be a positive integer'.format(requested_k)
        )
    if requested_k > max_supported_k:
        logger.warning(
            'Requested --uni_k=%d exceeds the smaller reference class (%d); '
            'Top-K retrieval will use all available references for that class',
            requested_k, max_supported_k,
        )
    memory_fg = torch.from_numpy(memory_fg_np).cuda(non_blocking=True)
    memory_bg = torch.from_numpy(memory_bg_np).cuda(non_blocking=True)
    memory_fg_norm_error = float(
        np.max(np.abs(np.linalg.norm(memory_fg_np, axis=1) - 1.0))
    )
    memory_bg_norm_error = float(
        np.max(np.abs(np.linalg.norm(memory_bg_np, axis=1) - 1.0))
    )
    mode_label = (
        'Native-Scale Reference-Guided CPS ({})'.format(args.reference_mode)
        if _is_native_reference_mode(args)
        else 'State-Aware Reference-Guided CPS'
    )
    logger.info(
        '%s UNI/reference loaded | weights=%s | memory=%s | fg=%d bg=%d '
        '| dim=%d resolution=%d grid=%dx%d k=%d source=labeled_%d',
        mode_label, weights_path, memory_path, memory_fg.shape[0],
        memory_bg.shape[0], feature_dim, reference_resolution, token_grid, token_grid,
        args.uni_k, args.labeled_ratio,
    )
    logger.info(
        '%s protocol | hard_reference_threshold_used=false '
        '| hard_routing_used=false | student_confidence_used=false '
        '| pseudo_label_flipping_used=false | online_memory_update=false',
        mode_label,
    )
    logger.info('%s UNI checkpoint sha256=%s', mode_label, _sha256_file(weights_path))
    logger.info('%s reference memory sha256=%s', mode_label, _sha256_file(memory_path))
    logger.info(
        '%s reference memory build control | flag=%s | built_this_run=%s '
        '| source=labeled_%d GT only',
        mode_label,
        bool(getattr(args, 'uni_build_reference_memory', False)),
        memory_built,
        args.labeled_ratio,
    )
    logger.info('%s UNI implementation=models/uni.py', mode_label)
    margin_calibration = None
    if (
        args.reference_mode in MARGIN_CALIBRATED_MODES
        and not getattr(args, 'build_reference_memory_only', False)
    ):
        calibration_records = Path(args.margin_calibration_records).expanduser().resolve()
        if not calibration_records.is_file():
            _build_margin_calibration_train_loo(
                args,
                cfg,
                manifest_dir,
                use_manifests,
                uni,
                memory_path,
                calibration_records,
                logger,
            )
        margin_calibration = _load_margin_calibration(args, logger, uni=uni)
    return {
        'uni': uni,
        'memory_fg': memory_fg,
        'memory_bg': memory_bg,
        'feature_dim': feature_dim,
        'token_grid': token_grid,
        'reference_resolution': reference_resolution,
        'weights_path': str(weights_path),
        'weights_sha256': _sha256_file(weights_path),
        'memory_path': str(memory_path),
        'memory_sha256': _sha256_file(memory_path),
        'uni_config_path': str(config_path),
        'uni_loader': getattr(args, 'uni_loader', 'local'),
        'uni_hub_id': getattr(args, 'uni_hub_id', 'MahmoodLab/UNI'),
        'reference_backbone': uni.backbone,
        'memory_fg_count': memory_fg_count,
        'memory_bg_count': memory_bg_count,
        'memory_fg_norm_error': memory_fg_norm_error,
        'memory_bg_norm_error': memory_bg_norm_error,
        'memory_top_k': memory_top_k,
        'memory_built': memory_built,
        'memory_build_requested': bool(getattr(args, 'uni_build_reference_memory', False)),
        'memory_build_source': 'labeled_{} GT only'.format(args.labeled_ratio),
        'hard_reference_threshold_used': False,
        'margin_calibration': margin_calibration,
    }






def _uni_dense_features(uni, rgb_01, reference_resolution=DEFAULT_REFERENCE_RESOLUTION):
    mean = torch.tensor(uni.input_mean, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(uni.input_std, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    reference_size = _reference_size(reference_resolution)
    expected_grid = _reference_grid(reference_resolution, uni.patch_size)
    x = F.interpolate(rgb_01, size=reference_size, mode='bilinear', align_corners=False)
    x = (x - mean) / std
    tokens = uni.dense_patch_tokens(x)
    token_count = int(tokens.shape[1])
    token_grid = int(math.sqrt(token_count))
    if token_grid * token_grid != token_count:
        raise RuntimeError('UNI dense token count is not square: {}'.format(token_count))
    if tokens.shape[-1] != uni.feature_dim or token_grid != expected_grid:
        raise RuntimeError(
            'Unexpected pathology encoder dense feature shape: {}'.format(tuple(tokens.shape))
        )
    return tokens


def _topk_scores(query, memory, k):
    similarity = query @ memory.t()
    return similarity.topk(min(int(k), int(memory.shape[0])), dim=1, largest=True).values.mean(dim=1)


def _reference_scores(tokens, context, k):
    """Return direct UNI FG/BG top-k similarity evidence at token scale."""
    flat = tokens.reshape(-1, tokens.shape[-1])
    score_fg = _topk_scores(flat, context['memory_fg'], k)
    score_bg = _topk_scores(flat, context['memory_bg'], k)
    return (
        score_fg.reshape(tokens.shape[0], tokens.shape[1]),
        score_bg.reshape(tokens.shape[0], tokens.shape[1]),
    )


def _reconstruct_rgb(image_rgb, cfg):
    mean = torch.tensor(cfg['MEAN'], device=image_rgb.device, dtype=image_rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(cfg['STD'], device=image_rgb.device, dtype=image_rgb.dtype).view(1, 3, 1, 1)
    return (image_rgb * std + mean).clamp(0.0, 1.0)
