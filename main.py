import argparse
import os
import random

import numpy as np
import torch

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from config.uni_config.uni_config import (
    DEFAULT_REFERENCE_RESOLUTION,
    SUPPORTED_REFERENCE_RESOLUTIONS,
    UNI_CONFIG_DEFAULT,
    UNI_WEIGHTS_DEFAULT,
)
from dataload.dataset_2d import imagefloder_itn
from utils.native_reference import (
    NATIVE_REFERENCE_MODES,
    _is_native_reference_mode,
    _reference_grid,
)
from training.train import (
    _restore_rng_state,
    _snapshot_rng_state,
    build_dataloaders,
    train_model,
)


# Reproducibility and input preparation.


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


# Dataset, model, and optimizer construction.
def build_datasets(args, cfg):
    """Build training and validation datasets; test is deferred out of training."""
    manifest_dir = args.manifest_dir or os.path.join(
        args.path_dataset, 'manifests', 'paper_seed_2026'
    )
    use_manifests = os.path.isdir(manifest_dir)

    mean_key, std_key = _normalization_keys(args.input1)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg[mean_key], cfg[std_key])
    train_transform = transforms['train']

    common = {
        'data_normalize_1': normalize,
        'input_mode': args.input_mode,
    }
    if use_manifests:
        labeled_source = {
            'manifest_path': os.path.join(
                manifest_dir, 'labeled_{}.csv'.format(args.labeled_ratio)
            )
        }
        unlabeled_source = {
            'manifest_path': os.path.join(
                manifest_dir, 'unlabeled_{}.csv'.format(args.labeled_ratio)
            )
        }
        validation_source = {
            'manifest_path': os.path.join(manifest_dir, 'validation.csv')
        }
        test_source = {
            'manifest_path': os.path.join(manifest_dir, 'test.csv')
        }
    else:
        labeled_source = {
            'data_dir': os.path.join(
                args.path_dataset, 'train_sup_' + args.sup_mark
            ),
            'input1': args.input1,
        }
        unlabeled_source = {
            'data_dir': os.path.join(
                args.path_dataset, 'train_unsup_' + args.unsup_mark
            ),
            'input1': args.input1,
        }
        validation_source = {
            'data_dir': os.path.join(args.path_dataset, 'val'),
            'input1': args.input1,
        }
        test_source = {
            'data_dir': os.path.join(args.path_dataset, 'test'),
            'input1': args.input1,
        }

    train_unsup = imagefloder_itn(
        **unlabeled_source,
        **common,
        data_transform_1=train_transform,
        sup=False,
        training=True,
    )
    train_sup_unique = imagefloder_itn(
        **labeled_source,
        **common,
        data_transform_1=train_transform,
        sup=True,
        training=True,
    )
    train_sup = imagefloder_itn(
        **labeled_source,
        **common,
        data_transform_1=train_transform,
        sup=True,
        num_images=len(train_unsup),
        training=True,
    )
    validation = imagefloder_itn(
        **validation_source,
        **common,
        data_transform_1=transforms['val'],
        sup=True,
        training=False,
    )
    # Test data is deliberately not instantiated or read during training for
    # either CPS mode.  Any final frozen test evaluation must be a separate
    # post-training entry point; validation remains the only selection split.
    test = None
    test_count = None

    datasets = {
        'train_sup': train_sup,
        'train_sup_unique': train_sup_unique,
        'train_unsup': train_unsup,
        'val': validation,
        'test': test,
        'test_count': test_count,
    }
    return datasets, manifest_dir, use_manifests


def print_data_summary(args, datasets, manifest_dir, use_manifests):
    sup_sample = datasets['train_sup'][0]
    unsup_sample = datasets['train_unsup'][0]
    val_sample = datasets['val'][0]
    test_sample = datasets['test'][0] if datasets['test'] is not None else None

    print('Data source:', manifest_dir if use_manifests else args.path_dataset)
    print('Input mode:', args.input_mode)
    print(
        'Unique samples: supervised={}, unsupervised={}, validation={}, test=deferred'.format(
            len(datasets['train_sup_unique']),
            len(datasets['train_unsup']),
            len(datasets['val']),
        )
    )
    print(
        'Effective epoch samples after supervised oversampling: supervised={}'.format(
            len(datasets['train_sup'])
        )
    )
    print(
        'Supervised image:',
        tuple(sup_sample['image'].shape),
        sup_sample['image'].dtype,
    )
    if 'image_stain' in sup_sample:
        print(
            'Supervised RGB/Stain views:',
            tuple(sup_sample['image_rgb'].shape),
            tuple(sup_sample['image_stain'].shape),
            tuple(sup_sample['mask'].shape),
        )
    print(
        'Supervised mask:',
        tuple(sup_sample['mask'].shape),
        sup_sample['mask'].dtype,
        'classes={}'.format(torch.unique(sup_sample['mask']).tolist()),
    )
    print(
        'Unsupervised image:',
        tuple(unsup_sample['image'].shape),
        unsup_sample['image'].dtype,
    )
    if 'image_stain' in unsup_sample:
        print(
            'Unsupervised RGB/Stain views:',
            tuple(unsup_sample['image_rgb'].shape),
            tuple(unsup_sample['image_stain'].shape),
        )
    print(
        'Validation image/mask:',
        tuple(val_sample['image'].shape),
        tuple(val_sample['mask'].shape),
    )
    if test_sample is not None:
        print(
            'Test image/mask:',
            tuple(test_sample['image'].shape),
            tuple(test_sample['mask'].shape),
        )


# Native reference-aware losses and diagnostics.


# Command-line entry point.
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Feature Diversified Collaborative Learning (FDCL).'
    )
    parser.add_argument(
        '--path_trained_models',
        default=None,
        help='默认按标注比例保存到 checkpoints/semi_fdcl_paper_10 或 20',
    )
    parser.add_argument(
        '--path_seg_results',
        default=None,
        help='默认按标注比例保存到 seg_pred/semi_fdcl_paper_10 或 20',
    )
    parser.add_argument(
        '-pd', '--path_dataset', default=None,
        help='dataset root; defaults to the selected dataset config path',
    )
    parser.add_argument(
        '--dataset_name',
        default='GlaS',
        choices=['GlaS', 'CRAG', 'MoNuSeg', 'RINGS'],
    )
    parser.add_argument('--input1', default='image')
    parser.add_argument(
        '--input_mode',
        dest='input_mode',
        default='rgb_rgb',
        choices=['rgb_rgb', 'rgb_stain'],
        help='rgb_rgb uses original RGB for both students; ' \
        'rgb_stain uses stain RGB for student2',
    )
    parser.add_argument(
        '--manifest_dir',
        default=None,
        help='manifest directory; defaults to PATH_DATASET/manifests/paper_seed_2026',
    )
    parser.add_argument(
        '--labeled_ratio',
        default=10,
        type=int,
        choices=[10, 20],
        help='percentage of labeled training data in the prepared manifests',
    )
    parser.add_argument(
        '--check_data',
        action='store_true',
        help='load one sample, print its shapes/ranges, and exit before GPU setup',
    )
    parser.add_argument('--sup_mark', default='9')
    parser.add_argument('--unsup_mark', default='76')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument(
        '--milestone_interval',
        dest='milestone_interval',
        default=10,
        type=int,
        help=(
            '每 N 个 epoch 保存同一时刻的两个 student 当前权重；'
            '设为 0 可关闭（默认：10）'
        ),
    )
    parser.add_argument(
        '--best_save_interval',
        dest='best_save_interval',
        default=10,
        type=int,
        help=(
            '每 N 个 epoch 额外保存截至当前的两个 student best 权重；'
            '设为 0 可关闭（默认：10）'
        ),
    )
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=5, type=float)
    parser.add_argument(
        '--cps_mode', '--cps-mode',
        dest='cps_mode',
        default='standard',
        choices=['standard'] + list(NATIVE_REFERENCE_MODES),
        help=(
            'standard uses the original bidirectional CPS; '
            'native_fine_margin_suppression uses (1-q) CPS only on risky pixels; '
            'native_fine_margin_reverse uses CPS plus q-weighted opposite loss; '
            'native_fine_margin_calibrated_reverse '
            'uses q=D/(D+tau) with train-only LOO tau; '
            'native_fine_margin_calibrated_reverse_disroute additionally routes fine disagreement CPS by q.'
        ),
    )
    parser.add_argument(
        '--uni_reference_memory', '--reference-memory', '--reference_memory',
        dest='uni_reference_memory',
        default=None,
        help=(
            'optional compatibility override for reference_memory.npz; when omitted, '
            'the current experiment checkpoint directory is used'
        ),
    )
    parser.add_argument(
        '--uni_build_reference_memory',
        action='store_true',
        help=(
            'build the missing reference memory from the current labeled GT split; '
            'native CPS modes require this flag when the memory is absent'
        ),
    )
    parser.add_argument(
        '--uni_weights',
        default=UNI_WEIGHTS_DEFAULT,
        help='local frozen MahmoodLab/UNI pytorch_model.bin',
    )
    parser.add_argument(
        '--uni_config',
        default=UNI_CONFIG_DEFAULT,
        help='UNI config.json next to the local weight file',
    )
    parser.add_argument('--uni_k', default=50, type=int)
    parser.add_argument(
        '--margin-calibration-records', dest='margin_calibration_records',
        default=None,
        help='Optional train-only image-level LOO token records; by default created per task and --uni_k.',
    )
    parser.add_argument(
        '--native-metrics-dir', dest='native_metrics_dir', default=None,
        help='Optional run-local directory for native training/validation mechanism CSVs.',
    )
    parser.add_argument(
        '--reference-resolution', '--reference_resolution',
        dest='reference_resolution', type=int, default=None,
        choices=list(SUPPORTED_REFERENCE_RESOLUTIONS),
        help='Frozen UNI reference input resolution; native CPS modes default to 256.',
    )
    parser.add_argument(
        '--uni-input-size', dest='uni_input_size', type=int, default=None,
        help='Alias for the frozen UNI reference resolution; native modes require 256.',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume only from the run-local verified resume_checkpoint.pth.',
    )
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay exponent')
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='unet')

    args = parser.parse_args()
    # The CPS mode is now the single public method selector.  The internal
    # reference_mode alias keeps the native implementation and metadata
    # paths normalized without exposing a second CLI selector.
    args.reference_mode = (
        'none' if args.cps_mode == 'standard' else args.cps_mode
    )
    if args.uni_input_size is not None:
        if (
            args.reference_resolution is not None
            and args.reference_resolution != args.uni_input_size
        ):
            parser.error('--uni-input-size conflicts with --reference-resolution')
        args.reference_resolution = int(args.uni_input_size)
    elif args.reference_resolution is None:
        args.reference_resolution = (
            256 if args.cps_mode in NATIVE_REFERENCE_MODES
            else DEFAULT_REFERENCE_RESOLUTION
        )
    if args.reference_mode in NATIVE_REFERENCE_MODES:
        if args.reference_resolution != 256:
            parser.error('native reference modes require --uni-input-size 256')
    if args.milestone_interval < 0:
        parser.error('--milestone_interval must be non-negative')
    if args.best_save_interval < 0:
        parser.error('--best_save_interval must be non-negative')
    if _is_native_reference_mode(args):
        args.reference_grid = _reference_grid(args.reference_resolution)
    if args.path_trained_models is None:
        default_checkpoints = {
            'native_fine_margin_suppression': 'checkpoints/k50_margin_suppression',
            'native_fine_margin_reverse': 'checkpoints/k50_margin_reverse',
            'native_fine_margin_calibrated_reverse': 'checkpoints/k50_margin_calibrated_reverse',
            'native_fine_margin_calibrated_reverse_disroute': 'checkpoints/k50_margin_calibrated_reverse_disroute',
        }
        args.path_trained_models = default_checkpoints.get(
            args.reference_mode,
            'checkpoints/semi_fdcl_paper_{}'.format(args.labeled_ratio),
        )
    if args.path_seg_results is None:
        default_seg_results = {
            'native_fine_margin_suppression': 'seg_pred/k50_margin_suppression',
            'native_fine_margin_reverse': 'seg_pred/k50_margin_reverse',
            'native_fine_margin_calibrated_reverse': 'seg_pred/k50_margin_calibrated_reverse',
            'native_fine_margin_calibrated_reverse_disroute': 'seg_pred/k50_margin_calibrated_reverse_disroute',
        }
        args.path_seg_results = default_seg_results.get(
            args.reference_mode,
            'seg_pred/semi_fdcl_paper_{}'.format(args.labeled_ratio),
        )
    return args


def main():
    args = parse_args()
    init_seeds(args.sd)
    cfg = dataset_cfg(args.dataset_name)
    if args.path_dataset is None:
        args.path_dataset = cfg['PATH_DATASET']
    datasets, manifest_dir, use_manifests = build_datasets(args, cfg)

    if args.check_data:
        print_data_summary(args, datasets, manifest_dir, use_manifests)
        return
    summary_rng = _snapshot_rng_state()
    print_data_summary(args, datasets, manifest_dir, use_manifests)
    _restore_rng_state(summary_rng)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'use --check_data to validate input only.'
        )

    dataloaders = build_dataloaders(datasets, args.batch_size)
    train_model(
        args,
        cfg,
        datasets,
        manifest_dir,
        use_manifests,
        dataloaders,
    )


if __name__ == '__main__':
    main()
