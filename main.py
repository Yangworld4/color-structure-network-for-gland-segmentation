import argparse
import csv
from contextlib import ExitStack, redirect_stdout
from collections import defaultdict
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
from warnings import simplefilter

from utils.ramps import sigmoid_rampup
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import (
    print_and_save_branch_bests,
    print_train_eval,
    print_train_loss,
    print_val_eval,
    print_val_loss,
    save_periodic_branch_bests,
    save_training_milestone,
    save_val_best_2d,
)
from config.uni_config.uni_config import (
    DEFAULT_REFERENCE_RESOLUTION,
    EXPECTED_UNI_SHA256,
    REFERENCE_MEMORY_ANCHOR_BG,
    REFERENCE_MEMORY_ANCHOR_FG,
    REFERENCE_MEMORY_FALLBACK_FRACTION,
    REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
    REFERENCE_MEMORY_SEED,
    SUPPORTED_REFERENCE_RESOLUTIONS,
    UNI_MEAN,
    UNI_PATCH_SIZE,
    UNI_STD,
    UNI_CONFIG_DEFAULT,
    UNI_WEIGHTS_DEFAULT,
)
from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.dataset_2d import imagefloder_itn
from loss.loss_function import DiceLoss
from models.getnetwork import get_network
from models.uni import load_uni
from utils.training_diagnostics import (
    compute_validation_diversity,
    finalize_diagnostics as _finalize_diagnostics,
    finalize_motivation_metrics as _finalize_motivation_metrics,
    foreground_probability as _foreground_probability,
    init_diagnostic_totals as _init_diagnostic_totals,
    init_motivation_totals as _init_motivation_totals,
    linear_cka as _linear_cka,
    log_diagnostics,
    select_feature_module as _select_feature_module,
    update_diagnostic_totals as _update_diagnostic_totals,
    update_motivation_totals as _update_motivation_totals,
)


simplefilter(action='ignore', category=FutureWarning)

METRIC_FIELDS = [
    'iteration', 'epoch',
    'train_loss_sup1', 'train_loss_sup2', 'train_loss_cps',
    'train_loss_total', 'student12_disagreement',
    'student12_dice_similarity', 'student1_entropy', 'student2_entropy',
    'pseudo_dice_student1', 'pseudo_dice_student2',
    'pseudo_jaccard_student1', 'pseudo_jaccard_student2',
]


def _reference_grid(reference_resolution):
    resolution = int(reference_resolution)
    if resolution not in SUPPORTED_REFERENCE_RESOLUTIONS:
        raise ValueError(
            'reference_resolution must be one of {}'.format(
                SUPPORTED_REFERENCE_RESOLUTIONS
            )
        )
    if resolution % UNI_PATCH_SIZE:
        raise ValueError('reference_resolution must be divisible by patch size')
    return resolution // UNI_PATCH_SIZE


def _reference_size(reference_resolution):
    resolution = int(reference_resolution)
    _reference_grid(resolution)
    return (resolution, resolution)

MOTIVATION_METRIC_FIELDS = [
    'epoch', 'agreement', 'soft_disagreement', 'ww_given_agreement',
    'hc90ww_given_ww', 'total_pixels', 'agreement_pixels', 'ww_pixels',
    'hc90ww_pixels',
]

NATIVE_TRAINING_FIELDS = [
    'experiment', 'epoch', 'num_native_tokens', 'num_agreement_tokens',
    'num_contradiction_tokens', 'contradiction_rate',
    'num_hc_contradiction_tokens', 'hc_contradiction_rate',
    'mean_CPS_weight', 'loss_sup', 'loss_cps_raw', 'loss_cps_weighted',
    'loss_native_ref', 'loss_total', 'mean_pA_token_on_HC',
    'mean_pB_token_on_HC', 'UNI_FG_HC_count', 'UNI_BG_HC_count',
    'mean_S_cur_contradiction', 'mean_S_opp_contradiction',
    'mean_absolute_D', 'mean_relative_gap', 'relative_gap_p10',
    'relative_gap_p25', 'relative_gap_p50', 'relative_gap_p75',
    'relative_gap_p90', 'mean_adaptive_weight', 'weight_p10',
    'weight_p25', 'weight_p50', 'weight_p75', 'weight_p90',
    'fraction_weight_0_0.2', 'fraction_weight_0.2_0.4',
    'fraction_weight_0.4_0.6', 'fraction_weight_0.6_0.8',
    'fraction_weight_0.8_1.0', 'gap_Q1_mean_gap', 'gap_Q1_mean_weight',
    'gap_Q2_mean_gap', 'gap_Q2_mean_weight', 'gap_Q3_mean_gap',
    'gap_Q3_mean_weight', 'gap_Q4_mean_gap', 'gap_Q4_mean_weight',
    'mean_D_contradiction', 'D_p10', 'D_p25', 'D_p50', 'D_p75', 'D_p90',
    'mean_contradiction_weight', 'contradiction_weight_p10',
    'contradiction_weight_p25', 'contradiction_weight_p50',
    'contradiction_weight_p75', 'contradiction_weight_p90',
    'fraction_weight_0_0.1', 'fraction_weight_0.1_0.2',
    'fraction_weight_0.2_0.3', 'fraction_weight_0.3_0.4',
    'fraction_weight_0.4_0.5', 'fraction_zero_weight',
    'D_Q1_mean_D', 'D_Q1_mean_weight', 'D_Q2_mean_D', 'D_Q2_mean_weight',
    'D_Q3_mean_D', 'D_Q3_mean_weight', 'D_Q4_mean_D', 'D_Q4_mean_weight',
    'regionwise_candidate_pixels', 'fine_selected_pixels',
    'fine_selected_over_regionwise', 'R1_same_consensus',
    'R2_opposite_consensus', 'R3_student_disagreement',
    'fine_suppression_coverage', 'global_suppression_coverage',
    'mean_applied_pixel_weight', 'num_risk_pixels', 'risk_pixel_fraction',
    'mean_D_risk', 'D_risk_p10', 'D_risk_p25', 'D_risk_p50',
    'D_risk_p75', 'D_risk_p90', 'mean_w_risk', 'mean_r_risk',
    'w_risk_p10', 'w_risk_p50', 'w_risk_p90', 'r_risk_p10',
    'r_risk_p50', 'r_risk_p90', 'mean_w_plus_r', 'reverse_fraction',
    'loss_cps_after_suppression', 'loss_reverse', 'loss_unsup_total',
    'mean_q_risk', 'q_risk_p10', 'q_risk_p25', 'q_risk_p50',
    'q_risk_p75', 'q_risk_p90', 'mean_CPS_weight_risk',
    'mean_opposite_weight_risk', 'risk_budget', 'risk_weight_sum_max_error',
    'loss_risk_severity',
]

# Native UNI methods are selected directly through --cps_mode.
MARGIN_CALIBRATED_MODES = (
    'native_fine_margin_suppression',
    'native_fine_margin_reverse',
    'native_fine_margin_calibrated_reverse',
)
NATIVE_REFERENCE_MODES = MARGIN_CALIBRATED_MODES
NATIVE_MODE_TAGS = {
    'native_fine_margin_suppression': 'native_fine_margin_suppression_s',
    'native_fine_margin_reverse': 'native_fine_margin_reverse_s',
    'native_fine_margin_calibrated_reverse': 'native_fine_margin_calibrated_reverse_s',
}


def _uses_uni_reference(args):
    return getattr(args, 'cps_mode', 'standard') in NATIVE_REFERENCE_MODES


def _is_native_reference_mode(args):
    return getattr(args, 'cps_mode', 'standard') in NATIVE_REFERENCE_MODES


def _is_margin_calibrated_mode(args):
    return getattr(args, 'cps_mode', 'standard') in MARGIN_CALIBRATED_MODES

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _snapshot_rng_state():
    """Capture every RNG stream that reference construction could touch."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def get_training_views(
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the RGB/RGB or RGB/stain inputs for the two students."""
    student1 = batch.get('image_rgb', batch['image'])
    student2 = batch.get('image_stain', batch['image'])
    spatial_shapes = {tuple(view.shape[-2:]) for view in (student1, student2)}
    if len(spatial_shapes) != 1:
        raise ValueError(
            'CPS views are not spatially aligned: student1={}, student2={}'.format(
                tuple(student1.shape), tuple(student2.shape)
            )
        )
    return student1, student2


def _get_rgb_view(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the normalized RGB view exposed by any supported input mode."""
    return batch.get('image_rgb', batch['image'])


def compute_supervised_loss(
    ce_loss,
    dice_loss,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # return 0.5 * (
    #     ce_loss(prediction, target.long())
    #     + dice_loss(prediction, target)
    # )
    return dice_loss(prediction, target)
        
    


def _pixel_entropy(scores: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(scores.detach(), dim=1).clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=1)


def compute_standard_cps_losses(
    ce_loss,
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
    pseudo1: Optional[torch.Tensor] = None,
    pseudo2: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the original bidirectional hard-pseudo-label CPS loss."""
    if pseudo1 is None:
        pseudo1 = prediction1.detach().argmax(dim=1)
    if pseudo2 is None:
        pseudo2 = prediction2.detach().argmax(dim=1)
    loss_1_to_2 = 0.5 * ce_loss(prediction2, pseudo1)
    loss_2_to_1 = 0.5 * ce_loss(prediction1, pseudo2)
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': loss_1_to_2 + loss_2_to_1}


def compute_entropy_routed_cps_losses(
    ce_loss,
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute CPS with entropy routing on pixel-level disagreements.

    Agreement pixels keep the vanilla bidirectional CPS loss. On a
    disagreement pixel, only the lower-entropy branch supervises the other
    branch; the selected teacher never receives a self-supervision loss.
    """
    detached1 = prediction1.detach()
    detached2 = prediction2.detach()
    pseudo1 = detached1.argmax(dim=1)
    pseudo2 = detached2.argmax(dim=1)

    entropy1 = _pixel_entropy(detached1)
    entropy2 = _pixel_entropy(detached2)

    disagreement = pseudo1.ne(pseudo2)
    # Ties are assigned to branch 1 for deterministic, mutually exclusive
    # routing on disagreement pixels.
    branch1_is_stable = entropy1.le(entropy2)
    route_1_to_2 = (~disagreement) | branch1_is_stable
    route_2_to_1 = (~disagreement) | (~branch1_is_stable)

    def masked_ce(prediction, target, route):
        per_pixel = F.cross_entropy(
            prediction,
            target,
            weight=ce_loss.weight,
            ignore_index=ce_loss.ignore_index,
            reduction='none',
        )
        if route.sum().item() == 0:
            return prediction.sum() * 0.0
        return per_pixel[route].mean()

    loss_1_to_2 = 0.5 * masked_ce(prediction2, pseudo1, route_1_to_2)
    loss_2_to_1 = 0.5 * masked_ce(prediction1, pseudo2, route_2_to_1)
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': loss_1_to_2 + loss_2_to_1}


def compute_cps_losses(
    ce_loss,
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
    cps_mode: str = 'standard',
) -> Dict[str, torch.Tensor]:
    """Compute the unchanged standard CPS loss.

    Reference-aware modes supply their targets directly; keeping this dispatcher
    standard-only prevents them from silently introducing entropy routing.
    """
    if cps_mode != 'standard':
        raise ValueError('Unsupported standard CPS mode: {}'.format(cps_mode))
    return compute_standard_cps_losses(ce_loss, prediction1, prediction2)


def _hard_predictions(scores: torch.Tensor) -> torch.Tensor:
    return scores.detach().argmax(dim=1)


def _entropy(scores: torch.Tensor) -> float:
    return float(_pixel_entropy(scores).mean().item())


def _prediction_overlap(scores: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    overlap = _hard_overlap(_hard_predictions(scores), target)
    return {'dice': overlap['dice'], 'jaccard': overlap['jc']}


def build_iteration_metrics(
    global_step,
    epoch,
    supervised_loss1,
    supervised_loss2,
    cps_loss,
    total_loss,
    unsup_scores1,
    unsup_scores2,
    sup_scores1,
    sup_scores2,
    sup_mask,
):
    diversity = _hard_overlap(
        _hard_predictions(unsup_scores1),
        _hard_predictions(unsup_scores2),
    )
    pseudo1 = _prediction_overlap(sup_scores1, sup_mask)
    pseudo2 = _prediction_overlap(sup_scores2, sup_mask)
    return {
        'iteration': global_step,
        'epoch': epoch,
        'train_loss_sup1': supervised_loss1.item(),
        'train_loss_sup2': supervised_loss2.item(),
        'train_loss_cps': cps_loss.item(),
        'train_loss_total': total_loss.item(),
        'student12_disagreement': 1.0 - diversity['dice'],
        'student12_dice_similarity': diversity['dice'],
        'student1_entropy': _entropy(unsup_scores1),
        'student2_entropy': _entropy(unsup_scores2),
        'pseudo_dice_student1': pseudo1['dice'],
        'pseudo_dice_student2': pseudo2['dice'],
        'pseudo_jaccard_student1': pseudo1['jaccard'],
        'pseudo_jaccard_student2': pseudo2['jaccard'],
    }


def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


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


def build_dataloaders(datasets, batch_size):
    loader_options = {
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': 8,
    }
    loaders = {
        'train_sup': DataLoader(
            datasets['train_sup'], shuffle=True, **loader_options
        ),
        'train_unsup': DataLoader(
            datasets['train_unsup'], shuffle=True, **loader_options
        ),
        'val': DataLoader(datasets['val'], shuffle=False, **loader_options),
    }
    return loaders


def _experiment_name(args, split_sup_mark, split_unsup_mark):
    experiment_name = (
        'CPS-l={}-e={}-s={}-g={}-b={}-cw={}-w={}-{}-{}-{}'.format(
            args.lr,
            args.num_epochs,
            args.step_size,
            args.gamma,
            args.batch_size,
            args.unsup_weight,
            args.warm_up_duration,
            split_sup_mark,
            split_unsup_mark,
            args.input1,
        )
    )
    experiment_name += '-input_mode={}'.format(args.input_mode)
    experiment_name += '-cps_mode={}'.format(
        getattr(args, 'cps_mode', 'standard')
    )
    return experiment_name


def build_output_paths(args, datasets, use_manifests):
    split_sup_mark = (
        str(len(datasets['train_sup_unique']))
        if use_manifests
        else str(args.sup_mark)
    )
    split_unsup_mark = (
        str(len(datasets['train_unsup']))
        if use_manifests
        else str(args.unsup_mark)
    )
    experiment_name = _experiment_name(
        args, split_sup_mark, split_unsup_mark
    )
    dataset_dir_name = os.path.basename(os.path.normpath(args.path_dataset))

    trained_models = os.path.join(
        args.path_trained_models, dataset_dir_name, experiment_name
    )
    segmentation_results = os.path.join(
        args.path_seg_results, dataset_dir_name, experiment_name
    )
    os.makedirs(trained_models, exist_ok=True)
    os.makedirs(segmentation_results, exist_ok=True)
    return trained_models, segmentation_results


def create_run_outputs(args, trained_models_path):
    log_path = os.path.join(trained_models_path, 'train.log')
    metrics_path = os.path.join(trained_models_path, 'metrics.csv')
    diagnostic_log_path = os.path.join(
        trained_models_path,
        'diagnostics_{}.log'.format(time.strftime('%Y%m%d-%H%M%S')),
    )

    logger = logging.getLogger('fdcl.train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    log_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    logger.addHandler(log_handler)
    logger.info(
        'Training started | dataset=%s | labeled_ratio=%s | network=%s '
        '| input_mode=%s',
        args.dataset_name,
        args.labeled_ratio,
        args.network,
        args.input_mode,
    )
    logger.info('Arguments: %s', vars(args))

    print('Training log:', log_path)
    print('Metrics CSV:', metrics_path)

    # CPS Diagnostic Analysis
    diagnostic_logger = logging.getLogger('fdcl.diagnostics')
    diagnostic_logger.setLevel(logging.INFO)
    diagnostic_logger.handlers.clear()
    diagnostic_logger.propagate = False
    diagnostic_handler = logging.FileHandler(
        diagnostic_log_path, mode='w', encoding='utf-8'
    )
    diagnostic_handler.setFormatter(logging.Formatter('%(message)s'))
    diagnostic_logger.addHandler(diagnostic_handler)
    print('Diagnostic log:', diagnostic_log_path)
    return logger, metrics_path, diagnostic_logger


def _build_optimizer_and_scheduler(model, args):
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=5 * 10 ** args.wd,
    )
    step_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warm_up_duration,
        after_scheduler=step_scheduler,
    )
    return optimizer, scheduler


def build_training_components(args, cfg):
    """Create both sub-networks, their optimizers/schedulers, and the loss."""
    model1 = get_network(
        args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']
    ).cuda()
    model2 = get_network(
        args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']
    ).cuda()
    losses = {
        'ce': CrossEntropyLoss().cuda(),
        'dice': DiceLoss().cuda(),
    }
    if _is_native_reference_mode(args):
        losses['ce_none'] = CrossEntropyLoss(reduction='none').cuda()

    optimizer1, scheduler1 = _build_optimizer_and_scheduler(model1, args)
    optimizer2, scheduler2 = _build_optimizer_and_scheduler(model2, args)
    return (
        (model1, model2),
        (optimizer1, optimizer2),
        (scheduler1, scheduler2),
        losses,
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_uni(args):
    return load_uni(
        args.uni_weights,
        args.uni_config,
        expected_sha256=EXPECTED_UNI_SHA256,
    )


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
    expected_grid = _reference_grid(reference_resolution)
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

    if token_grid != expected_grid or feature_dim != 1024:
        raise RuntimeError(
            'Unexpected generated memory feature shape: grid={} dim={}'.format(
                token_grid, feature_dim
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
        memory_construction=np.array('class-balanced native UNI token memory'),
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
            tokens = _uni_dense_features(uni, rgb, 256)
            occ = F.adaptive_avg_pool2d(gt, (16, 16)).flatten(1)
            features.append(tokens.detach().cpu().numpy().astype(np.float32))
            occupancy.append(occ.detach().cpu().numpy().astype(np.float32))
            image_ids.extend(str(value) for value in batch['ID'])
    features = np.concatenate(features, axis=0)
    occupancy = np.concatenate(occupancy, axis=0)
    if features.shape != (len(image_ids), 256, 1024):
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
        'reference_resolution': 256,
        'token_grid': 16,
        'feature_dim': 1024,
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


def _load_margin_calibration(args, logger):
    """Load fixed labeled-train-only image-level LOO calibration at ``--uni_k``."""
    records_path = Path(args.margin_calibration_records).expanduser().resolve()
    summary_path = records_path.with_name('summary.json')
    if not records_path.is_file():
        raise FileNotFoundError('Margin calibration records not found: {}'.format(records_path))
    if not summary_path.is_file():
        raise FileNotFoundError('Margin calibration summary not found: {}'.format(summary_path))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
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
    expected_grid = _reference_grid(reference_resolution)
    memory_path = Path(args.uni_reference_memory).expanduser().resolve()
    uni, weights_path, config_path = _load_uni(args)
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
    if reference_resolution == 224 and str(args.dataset_name).strip().lower() == 'glas' and args.labeled_ratio == 10:
        assert memory_fg_count == 391, memory_fg_count
        assert memory_bg_count == 294, memory_bg_count
    assert memory_fg_np.shape[1] == 1024, memory_fg_np.shape
    assert memory_bg_np.shape[1] == 1024, memory_bg_np.shape
    assert feature_dim == 1024, feature_dim
    assert token_grid == expected_grid, (token_grid, expected_grid)
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
    if args.reference_mode in MARGIN_CALIBRATED_MODES:
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
        margin_calibration = _load_margin_calibration(args, logger)
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
    mean = torch.tensor(UNI_MEAN, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(UNI_STD, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    reference_size = _reference_size(reference_resolution)
    expected_grid = _reference_grid(reference_resolution)
    x = F.interpolate(rgb_01, size=reference_size, mode='bilinear', align_corners=False)
    x = (x - mean) / std
    tokens = uni.dense_patch_tokens(x)
    token_count = int(tokens.shape[1])
    token_grid = int(math.sqrt(token_count))
    if token_grid * token_grid != token_count:
        raise RuntimeError('UNI dense token count is not square: {}'.format(token_count))
    if tokens.shape[-1] != 1024 or token_grid != expected_grid:
        raise RuntimeError(
            'Unexpected UNI dense feature shape: {}'.format(tuple(tokens.shape))
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


def _safe_mean(total, count):
    return float(total) / float(count) if count else float('nan')


def _build_native_reference_targets(logits_a, logits_b, rgb_input, context, cfg, args):
    """Build detached UNI-256 native token gating for S/SC.

    The UNI margin remains at its native 16x16 scale.  Only the boolean
    contradiction/HC masks are later expanded as *loss weights* with nearest
    neighbour replication; no semantic UNI value is interpolated here.
    """
    if context['token_grid'] != 16 or context['reference_resolution'] != 256:
        raise AssertionError(
            'Native reference modes require UNI-256/16x16, got resolution={} grid={}'.format(
                context['reference_resolution'], context['token_grid']
            )
        )
    prob_a = _foreground_probability(logits_a)
    prob_b = _foreground_probability(logits_b)
    p_a_token = F.adaptive_avg_pool2d(prob_a, (16, 16))
    p_b_token = F.adaptive_avg_pool2d(prob_b, (16, 16))
    gate_a = p_a_token.detach()
    gate_b = p_b_token.detach()
    hard_a = gate_a.ge(0.5)
    hard_b = gate_b.ge(0.5)
    conf_a = torch.maximum(gate_a, 1.0 - gate_a)
    conf_b = torch.maximum(gate_b, 1.0 - gate_b)
    agreement = hard_a.eq(hard_b)
    with torch.inference_mode():
        tokens = _uni_dense_features(
            context['uni'], _reconstruct_rgb(rgb_input, cfg), 256
        )
        if tokens.shape[1:] != (256, 1024):
            raise AssertionError('UNI native token shape must be [B,256,1024], got {}'.format(tuple(tokens.shape)))
        score_fg, score_bg = _reference_scores(tokens, context, args.uni_k)
        margin = (score_fg - score_bg).reshape(-1, 1, 16, 16)
        score_fg = score_fg.reshape(-1, 1, 16, 16)
        score_bg = score_bg.reshape(-1, 1, 16, 16)
        uni_fg = margin.gt(0.0)
        ties = margin.eq(0.0)
    # Direct semantic-gap evidence at the native 16x16 token scale.  The
    # reference margin is S_FG-S_BG; choose current/opposite by the detached
    # Student consensus, then suppress only positive contradiction gaps.
    s_cur = torch.where(hard_a[:, 0], score_fg[:, 0], score_bg[:, 0]).detach()
    s_opp = torch.where(hard_a[:, 0], score_bg[:, 0], score_fg[:, 0]).detach()
    d_positive = torch.clamp(s_opp - s_cur, min=0.0).detach()
    contradiction = (agreement[:, 0] & d_positive.gt(0.0)).detach()
    # High-confidence correction is no longer a training method.  Keep this
    # fixed diagnostic count for mechanism reporting only.
    hc = (contradiction & conf_a[:, 0].ge(0.9) & conf_b[:, 0].ge(0.9)).detach()
    relative_gap = torch.clamp(
        d_positive / (s_opp.abs() + s_cur.abs() + 1e-6), 0.0, 1.0
    ).detach()
    # Margin-calibrated severity uses a neutral token weight; q is applied
    # only in the fine-selective directional mixture below.
    weight_token = torch.ones_like(margin).detach()
    if not torch.isfinite(weight_token).all().item():
        raise FloatingPointError('Native token weight contains NaN/Inf')
    if not torch.all(weight_token[:, 0][~contradiction].eq(1.0)).item():
        raise AssertionError('Non-contradiction native token weight changed from one')
    weight_pixel = None
    fine_stats = {}
    if args.reference_mode in MARGIN_CALIBRATED_MODES:
        # Fine-selective mode keeps the exact token contradiction and token
        # weight, but applies it only where the full-resolution students still
        # agree with the token consensus class.
        hard_a_pixel = prob_a.detach().ge(0.5)
        hard_b_pixel = prob_b.detach().ge(0.5)
        contradiction_pixel = F.interpolate(
            contradiction.float().unsqueeze(1),
            size=prob_a.shape[-2:], mode='nearest',
        ).bool()
        consensus_pixel = F.interpolate(
            hard_a.detach().float(),
            size=prob_a.shape[-2:], mode='nearest',
        ).ge(0.5)
        token_weight_pixel = F.interpolate(
            weight_token.detach(), size=prob_a.shape[-2:], mode='nearest'
        )
        pixel_agreement = hard_a_pixel.eq(hard_b_pixel)
        same_as_token_consensus = hard_a_pixel.eq(consensus_pixel)
        risk_pixel = (
            contradiction_pixel
            & pixel_agreement
            & same_as_token_consensus
        ).detach()
        weight_pixel = torch.ones_like(prob_a)
        weight_pixel[risk_pixel] = token_weight_pixel[risk_pixel]
        weight_pixel = weight_pixel.detach()
        regionwise_candidate = contradiction_pixel.detach()
        r1 = regionwise_candidate & pixel_agreement & same_as_token_consensus
        r2 = regionwise_candidate & pixel_agreement & ~same_as_token_consensus
        r3 = regionwise_candidate & ~pixel_agreement
        if not torch.equal(risk_pixel, r1):
            raise AssertionError('Fine risk mask is not R1')
        if not torch.equal(r1 | r2 | r3, regionwise_candidate):
            raise AssertionError('Fine R1/R2/R3 do not partition contradiction pixels')
        if weight_pixel.requires_grad or risk_pixel.requires_grad:
            raise AssertionError('Fine pixel weight/mask must be detached')
        fine_stats = {
            'regionwise_candidate_pixels': regionwise_candidate.detach(),
            'fine_selected_pixels': risk_pixel.detach(),
            'R1_same_consensus_pixels': r1.detach(),
            'R2_opposite_consensus_pixels': r2.detach(),
            'R3_student_disagreement_pixels': r3.detach(),
            'all_pixel_count': int(prob_a.numel()),
            'sum_applied_pixel_weight': weight_pixel.detach().sum(),
        }
        if args.reference_mode in MARGIN_CALIBRATED_MODES:
            calibration = context.get('margin_calibration')
            if not calibration:
                raise RuntimeError('Margin-calibrated mode has no fixed tau calibration')
            tau = torch.as_tensor(
                float(calibration['tau']), device=d_positive.device,
                dtype=d_positive.dtype,
            )
            # q=D/(D+tau), with eps only for numerical stability.
            q_token = d_positive / (d_positive + tau + 1e-6)
            q_token = q_token.clamp(0.0, 1.0).detach()
            q_pixel = F.interpolate(
                q_token.unsqueeze(1), size=prob_a.shape[-2:], mode='nearest'
            ).detach()
            opposite_target_pixel = (~consensus_pixel).float().detach()
            fine_stats.update({
                'risk_pixel': risk_pixel.detach(),
                'd_pixel': F.interpolate(
                    d_positive.detach().unsqueeze(1),
                    size=prob_a.shape[-2:], mode='nearest',
                ).detach(),
                'q_pixel': q_pixel,
                'opposite_target_pixel': opposite_target_pixel,
            })
    return {
        'p_a_token': p_a_token, 'p_b_token': p_b_token,
        'hard_a': hard_a, 'hard_b': hard_b,
        'conf_a': conf_a, 'conf_b': conf_b,
        'agreement': agreement, 'uni_fg': uni_fg, 'ties': ties,
        'margin': margin[:, 0], 's_cur': s_cur, 's_opp': s_opp,
        'd_positive': d_positive, 'relative_gap': relative_gap,
        'contradiction': contradiction,
        'hc': hc, 'weight_token': weight_token,
        'weight_pixel': weight_pixel,
        'fine_stats': fine_stats,
    }

def compute_native_margin_calibrated_anti_confirmation_losses(
    ce_none,
    prediction1,
    prediction2,
    pseudo1,
    pseudo2,
    risk_pixel,
    q_pixel,
    opposite_target_pixel,
    mode,
):
    """Margin-calibrated ablations on fine-selective risk pixels.

    Non-risk pixels retain standard CPS.  On the exact fine-selective risk
    mask, the three modes are:

    * suppression-only: ``(1-q) * L_CPS``;
    * reverse-only: ``L_CPS + q * L_opp``;
    * full: ``(1-q) * L_CPS + q * L_opp``.
    """
    if getattr(ce_none, 'reduction', None) != 'none':
        raise AssertionError('Margin-calibrated mode requires reduction=none CE')
    if risk_pixel.shape != pseudo1.shape or q_pixel.shape != pseudo1.shape:
        raise AssertionError('Severity pixel tensors must match pseudo-label shape')
    if opposite_target_pixel.shape != pseudo1.shape:
        raise AssertionError('Severity opposite target shape mismatch')
    valid1 = pseudo1.ne(ce_none.ignore_index)
    valid2 = pseudo2.ne(ce_none.ignore_index)
    if not valid1.any().item() or not valid2.any().item():
        raise AssertionError('Margin-calibrated mode received no valid pseudo-label pixels')
    if not torch.all((q_pixel >= 0.0) & (q_pixel <= 1.0)).item():
        raise AssertionError('Margin-calibrated q escaped [0,1]')
    current_a = ce_none(prediction1, pseudo2)
    current_b = ce_none(prediction2, pseudo1)
    opposite_target = opposite_target_pixel.long().detach()
    opposite_a = ce_none(prediction1, opposite_target)
    opposite_b = ce_none(prediction2, opposite_target)
    if mode == 'native_fine_margin_suppression':
        cps_weight = 1.0 - q_pixel
        opposite_weight = torch.zeros_like(q_pixel)
    elif mode == 'native_fine_margin_reverse':
        cps_weight = torch.ones_like(q_pixel)
        opposite_weight = q_pixel
    elif mode == 'native_fine_margin_calibrated_reverse':
        cps_weight = 1.0 - q_pixel
        opposite_weight = q_pixel
    else:
        raise ValueError('Unknown margin-calibrated ablation mode: {}'.format(mode))
    risk_a = cps_weight * current_a + opposite_weight * opposite_a
    risk_b = cps_weight * current_b + opposite_weight * opposite_b
    effective_a = torch.where(risk_pixel, risk_a, current_a)
    effective_b = torch.where(risk_pixel, risk_b, current_b)
    loss_a_to_b = 0.5 * effective_a[valid1].mean()
    loss_b_to_a = 0.5 * effective_b[valid2].mean()
    raw_a = 0.5 * current_a[valid1].mean()
    raw_b = 0.5 * current_b[valid2].mean()
    risk_loss = (
        0.5 * (risk_a[risk_pixel].mean() + risk_b[risk_pixel].mean())
        if risk_pixel.any().item() else prediction1.sum() * 0.0
    )
    total = loss_a_to_b + loss_b_to_a
    if not torch.isfinite(total).item():
        raise FloatingPointError('Margin-calibrated Anti-Confirmation CPS loss contains NaN/Inf')
    return {
        '1_to_2': loss_a_to_b,
        '2_to_1': loss_b_to_a,
        'total': total,
        'raw_total': raw_a + raw_b,
        'raw_1_to_2': raw_a,
        'raw_2_to_1': raw_b,
        'risk_loss': risk_loss,
        'risk_mix': risk_loss,
        'q_risk': q_pixel[risk_pixel].detach(),
        'risk_pixel_count': risk_pixel.sum().detach(),
        'cps_weight_risk': cps_weight[risk_pixel].detach(),
        'opposite_weight_risk': opposite_weight[risk_pixel].detach(),
    }


def _run_native_startup_checks(args, context, models, optimizers):
    """Protocol checks for the native-scale mode (no training smoke run)."""
    if not _is_native_reference_mode(args):
        raise AssertionError('Native startup check called for non-native mode')
    if context['reference_resolution'] != 256 or context['token_grid'] != 16:
        raise AssertionError('Native UNI geometry must be 256/16x16')
    if context['feature_dim'] != 1024:
        raise AssertionError('Unexpected native UNI feature configuration')
    if args.uni_k < 1:
        raise AssertionError('Requested native UNI Top-K must be positive')
    if any(parameter.requires_grad for parameter in context['uni'].parameters()):
        raise AssertionError('UNI parameters are trainable')
    optimizer_ids = {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    if any(id(parameter) in optimizer_ids for parameter in context['uni'].parameters()):
        raise AssertionError('UNI parameters entered CPS optimizer')
    print('UNI input shape       = [B,3,256,256]')
    print('UNI patch-token shape = [B,256,1024]')
    print('native grid           = 16x16')
    print('feature dim           = 1024')
    if _is_margin_calibrated_mode(args):
        calibration = context.get('margin_calibration')
        if not calibration:
            raise AssertionError('Margin-calibrated mode has no calibration metadata')
        print('Reference mode:       {}'.format(args.reference_mode))
        if args.reference_mode == 'native_fine_margin_suppression':
            print('Adaptive strategy:    labeled-train margin-calibrated suppression-only')
            print('Weight formula:       (1-q) CPS on risky pixels; no opposite loss')
        elif args.reference_mode == 'native_fine_margin_reverse':
            print('Adaptive strategy:    labeled-train margin-calibrated reverse-only')
            print('Weight formula:       CPS + q opposite on risky pixels')
        else:
            print('Adaptive strategy:    labeled-train margin-calibrated Anti-Confirmation')
            print('Weight formula:       (1-q) CPS + q opposite on risky pixels')
        print('Severity formula:     q=D/(D+tau+eps)')
        print('Opposite target:      1 - detached token consensus')
        print('Pixel selection:      token contradiction AND pixel agreement AND pixel consensus=token consensus')
        print('Calibration source:   labeled train-only image-level LOO K={}'.format(args.uni_k))
        print('tau:                  {:.8f}'.format(float(calibration['tau'])))
        print('Old CDF/bootstrap:    disabled')
        print('Extra hyperparameters: none')
def _run_direct_gap_numerical_checks():
    """Batch-level algebra checks; no model forward or training is run."""
    s_cur = torch.tensor([0.2, 0.2, 0.6, 0.0], device='cuda')
    s_opp = torch.tensor([0.1, 0.8, 0.6, 0.0], device='cuda')
    gap = torch.clamp(torch.maximum(s_opp - s_cur, torch.zeros_like(s_cur)) / (s_opp.abs() + s_cur.abs() + 1e-6), 0.0, 1.0)
    weight = (1.0 - gap).detach()
    assert torch.allclose(weight, 1.0 - gap, atol=0.0, rtol=0.0)
    assert weight[0].item() == 1.0
    assert gap[2].item() == 0.0 and weight[2].item() == 1.0
    assert gap[1].item() > gap[0].item() and weight[1].item() < weight[0].item()
    assert torch.isfinite(gap).all().item() and torch.isfinite(weight).all().item()
    assert gap.min().item() >= 0.0 and gap.max().item() <= 1.0
    assert weight.min().item() >= 0.0 and weight.max().item() <= 1.0
    assert not weight.requires_grad
    print('Direct gap numerical checks: PASS')

def train_one_epoch(
    args,
    epoch,
    global_step,
    models,
    optimizers,
    losses,
    dataloaders,
    metrics_writer,
    metrics_file,
    reference_context=None,
):
    model1, model2 = models
    optimizer1, optimizer2 = optimizers
    model1.train()
    model2.train()
    loss_totals = {
        'sup1': 0.0,
        'sup2': 0.0,
        'unsup': 0.0,
        'total': 0.0,
    }
    supervised_batches = iter(dataloaders['train_sup'])
    unsupervised_batches = iter(dataloaders['train_unsup'])
    num_train_batches = len(dataloaders['train_sup'])
    pbar = range(num_train_batches)
    score_list1 = None
    score_list2 = None
    native_totals = None
    if _is_native_reference_mode(args):
        native_totals = {
            'num_tokens': 0.0, 'agreement_tokens': 0.0,
            'contradiction_tokens': 0.0, 'hc_tokens': 0.0,
            'sum_weight': 0.0, 'raw_cps_loss': 0.0,
            'weighted_cps_loss': 0.0, 'native_ref_loss': 0.0,
            'total_loss': 0.0, 'batches': 0.0,
            'sum_p_a_hc': 0.0, 'sum_p_b_hc': 0.0,
            'uni_fg_hc': 0.0, 'uni_bg_hc': 0.0,
            's_cur_values': [], 's_opp_values': [], 'd_values': [],
            'relative_gaps': [], 'adaptive_weights': [],
            'regionwise_candidate_pixels': 0.0,
            'fine_selected_pixels': 0.0,
            'R1_same_consensus_pixels': 0.0,
            'R2_opposite_consensus_pixels': 0.0,
            'R3_student_disagreement_pixels': 0.0,
            'all_pixel_count': 0.0,
            'sum_applied_pixel_weight': 0.0,
            'risk_pixels': 0.0,
            'risk_all_pixel_count': 0.0,
            'sum_unsup_total': 0.0,
            'q_values': [], 'q_d_values': [], 'q_cps_weights': [], 'q_opp_weights': [],
            'q_weight_sum_errors': [], 'q_risk_loss': 0.0,
        }
    mask_list = None

    unsup_weight = args.unsup_weight * (epoch+1) / args.num_epochs

    for batch_index in range(num_train_batches):

        ###########################
            # supervised path #
        ###########################
        sup_batch = next(supervised_batches)
        sup_input1, sup_input2 = get_training_views(sup_batch)
        sup_input1 = sup_input1.cuda(non_blocking=True)
        sup_input2 = sup_input2.cuda(non_blocking=True)
        sup_mask = sup_batch['mask'].cuda(non_blocking=True)

        sup_prediction1 = model1(sup_input1)[0]
        sup_prediction2 = model2(sup_input2)[0]

        if epoch == 0 and batch_index == 0:
            shape_message = (
                'Input shapes | student1={} | student2={} | mask={}'
            ).format(
                tuple(sup_input1.shape), tuple(sup_input2.shape),
                tuple(sup_mask.shape),
            )
            print(shape_message)

        if batch_index == 0:
            score_list1 = sup_prediction1.detach()
            score_list2 = sup_prediction2.detach()
            mask_list = sup_mask.detach()
        elif batch_index <= num_train_batches / 32:
            score_list1 = torch.cat(
                (score_list1, sup_prediction1.detach()), dim=0
            )
            score_list2 = torch.cat(
                (score_list2, sup_prediction2.detach()), dim=0
            )
            mask_list = torch.cat((mask_list, sup_mask.detach()), dim=0)

        supervised_loss1 = compute_supervised_loss(
            losses['ce'],
            losses['dice'],
            sup_prediction1,
            sup_mask,
        )
        supervised_loss2 = compute_supervised_loss(
            losses['ce'],
            losses['dice'],
            sup_prediction2,
            sup_mask,
        )

        ###########################
            # unsupervised path #
        ###########################
        unsup_batch = next(unsupervised_batches)
        unsup_input1, unsup_input2 = get_training_views(unsup_batch)
        unsup_input1 = unsup_input1.cuda(non_blocking=True)
        unsup_input2 = unsup_input2.cuda(non_blocking=True)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        unsup_prediction1 = model1(unsup_input1)[0]
        unsup_prediction2 = model2(unsup_input2)[0]
        
        native_ref_loss = unsup_prediction1.sum() * 0.0
        if _is_native_reference_mode(args):
            reference_targets = _build_native_reference_targets(
                unsup_prediction1, unsup_prediction2, unsup_input1,
                reference_context, reference_context['cfg'], args,
            )
            pseudo_a = unsup_prediction1.detach().argmax(dim=1)
            pseudo_b = unsup_prediction2.detach().argmax(dim=1)
            original_cps_losses = compute_standard_cps_losses(
                losses['ce'], unsup_prediction1, unsup_prediction2,
                pseudo_a, pseudo_b,
            )
            fine_stats = reference_targets['fine_stats']
            native_cps = compute_native_margin_calibrated_anti_confirmation_losses(
                losses['ce_none'], unsup_prediction1, unsup_prediction2,
                pseudo_a, pseudo_b,
                fine_stats['risk_pixel'][:, 0],
                fine_stats['q_pixel'][:, 0],
                fine_stats['opposite_target_pixel'][:, 0],
                args.reference_mode,
            )
            cps_losses = {
                '1_to_2': native_cps['1_to_2'],
                '2_to_1': native_cps['2_to_1'],
                'total': native_cps['total'],
            }
        else:
            cps_losses = compute_cps_losses(
                losses['ce'],
                unsup_prediction1,
                unsup_prediction2,
                cps_mode='standard',
            )
            original_cps_losses = cps_losses
        unsup_loss = cps_losses['total']

        model1_loss = (
            supervised_loss1
            + unsup_weight * cps_losses['2_to_1']
        )
        model2_loss = (
            supervised_loss2
            + unsup_weight * cps_losses['1_to_2']
        )
        weighted_unsup_loss = unsup_weight * unsup_loss
        total_loss = model1_loss + model2_loss
        total_loss.backward()

        optimizer1.step()
        optimizer2.step()
        global_step += 1

        if _is_native_reference_mode(args):
            contradiction = reference_targets['contradiction']
            hc = reference_targets['hc']
            agreement = reference_targets['agreement']
            native_totals['num_tokens'] += float(agreement.numel())
            native_totals['agreement_tokens'] += float(agreement.sum().item())
            native_totals['contradiction_tokens'] += float(contradiction.sum().item())
            native_totals['hc_tokens'] += float(hc.sum().item())
            native_totals['sum_weight'] += float(reference_targets['weight_token'].sum().item())
            native_totals['raw_cps_loss'] += float(native_cps['raw_total'].detach().item())
            native_totals['weighted_cps_loss'] += float(cps_losses['total'].detach().item())
            native_totals['sum_unsup_total'] += float(cps_losses['total'].detach().item())
            native_totals['native_ref_loss'] += float(native_ref_loss.detach().item())
            native_totals['total_loss'] += float(total_loss.detach().item())
            native_totals['batches'] += 1.0
            if _is_margin_calibrated_mode(args):
                fine_stats = reference_targets['fine_stats']
                native_totals['regionwise_candidate_pixels'] += float(
                    fine_stats['regionwise_candidate_pixels'].sum().item()
                )
                native_totals['fine_selected_pixels'] += float(
                    fine_stats['fine_selected_pixels'].sum().item()
                )
                native_totals['R1_same_consensus_pixels'] += float(
                    fine_stats['R1_same_consensus_pixels'].sum().item()
                )
                native_totals['R2_opposite_consensus_pixels'] += float(
                    fine_stats['R2_opposite_consensus_pixels'].sum().item()
                )
                native_totals['R3_student_disagreement_pixels'] += float(
                    fine_stats['R3_student_disagreement_pixels'].sum().item()
                )
                native_totals['all_pixel_count'] += float(fine_stats['all_pixel_count'])
                native_totals['sum_applied_pixel_weight'] += float(
                    fine_stats['sum_applied_pixel_weight'].item()
                )
                if _is_margin_calibrated_mode(args):
                    risk = fine_stats['risk_pixel'][:, 0]
                    risk_count = float(risk.sum().item())
                    native_totals['risk_pixels'] += risk_count
                    native_totals['risk_all_pixel_count'] += float(risk.numel())
                    if risk.any().item():
                        q_px = fine_stats['q_pixel'][:, 0][risk]
                        d_px = fine_stats['d_pixel'][:, 0][risk]
                        w_px = native_cps['cps_weight_risk']
                        r_px = native_cps['opposite_weight_risk']
                        native_totals['q_values'].extend(q_px.detach().cpu().tolist())
                        native_totals['q_d_values'].extend(d_px.detach().cpu().tolist())
                        native_totals['q_cps_weights'].extend(w_px.detach().cpu().tolist())
                        native_totals['q_opp_weights'].extend(r_px.detach().cpu().tolist())
                        if args.reference_mode == 'native_fine_margin_suppression':
                            expected_budget = 1.0 - q_px
                        elif args.reference_mode == 'native_fine_margin_reverse':
                            expected_budget = 1.0 + q_px
                        else:
                            expected_budget = torch.ones_like(q_px)
                        native_totals['q_weight_sum_errors'].extend(
                            (w_px + r_px - expected_budget).detach().cpu().abs().tolist()
                        )
                        native_totals['q_risk_loss'] += float(native_cps['risk_loss'].detach().item())
            if hc.any().item():
                native_totals['sum_p_a_hc'] += float(reference_targets['p_a_token'][hc.unsqueeze(1)].sum().item())
                native_totals['sum_p_b_hc'] += float(reference_targets['p_b_token'][hc.unsqueeze(1)].sum().item())
                native_totals['uni_fg_hc'] += float((reference_targets['uni_fg'] & hc.unsqueeze(1)).sum().item())
            native_totals['uni_bg_hc'] += float(((~reference_targets['uni_fg']) & hc.unsqueeze(1)).sum().item())
        metrics_writer.writerow(
            build_iteration_metrics(
                global_step,
                epoch + 1,
                supervised_loss1,
                supervised_loss2,
                weighted_unsup_loss,
                total_loss,
                unsup_prediction1,
                unsup_prediction2,
                sup_prediction1,
                sup_prediction2,
                sup_mask,
            )
        )
        metrics_file.flush()

        loss_totals['unsup'] += weighted_unsup_loss.item()
        loss_totals['sup1'] += supervised_loss1.item()
        loss_totals['sup2'] += supervised_loss2.item()
        loss_totals['total'] += total_loss.item()
    reference_totals = native_totals
    return (
        loss_totals,
        (score_list1, score_list2, mask_list),
        global_step,
        reference_totals,
    )


@torch.no_grad()
def validate(models, losses, dataloader):
    model1, model2 = models
    model1.eval()
    model2.eval()

    loss1 = 0.0
    loss2 = 0.0
    scores1 = []
    scores2 = []
    masks = []
    names = []
    # CPS Diagnostic Analysis
    diagnostic_totals = _init_diagnostic_totals()
    motivation_totals = _init_motivation_totals()
    diagnostic_features = {'a': None, 'b': None}
    feature_module1 = _select_feature_module(model1)
    feature_module2 = _select_feature_module(model2)
    hook_handles = []
    if feature_module1 is not None and feature_module2 is not None:
        def _capture_feature(name):
            def _hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                diagnostic_features[name] = output.detach()
            return _hook

        hook_handles = [
            feature_module1.register_forward_hook(_capture_feature('a')),
            feature_module2.register_forward_hook(_capture_feature('b')),
        ]

    try:
        for batch in dataloader:
            input1, input2 = get_training_views(batch)
            input1 = input1.cuda(non_blocking=True)
            input2 = input2.cuda(non_blocking=True)
            mask = batch['mask'].cuda(non_blocking=True)

            diagnostic_features['a'] = None
            diagnostic_features['b'] = None
            output1 = model1(input1)[0]
            output2 = model2(input2)[0]
            names.extend(batch['ID'])
            loss1 += compute_supervised_loss(
                losses['ce'],
                losses['dice'],
                output1,
                mask,
            ).item()
            loss2 += compute_supervised_loss(
                losses['ce'],
                losses['dice'],
                output2,
                mask,
            ).item()
            _update_diagnostic_totals(
                diagnostic_totals,
                output1,
                output2,
                mask,
                diagnostic_features['a'],
                diagnostic_features['b'],
            )
            # Validation metrics need the complete epoch logits, but they do
            # not need to remain on GPU after the current batch.  Keeping the
            # cache on CPU avoids an unnecessary validation-set-sized GPU
            # allocation while preserving the exact logits and downstream
            # evaluator behavior.
            scores1.append(output1.detach().cpu())
            scores2.append(output2.detach().cpu())
            masks.append(mask.detach().cpu())
            _update_motivation_totals(
                motivation_totals,
                output1,
                output2,
                mask,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    scores1_tensor = torch.cat(scores1, dim=0)
    scores2_tensor = torch.cat(scores2, dim=0)

    return {
        'loss1': loss1,
        'loss2': loss2,
        'scores1': scores1_tensor,
        'scores2': scores2_tensor,
        'masks': torch.cat(masks, dim=0),
        'names': np.asarray(names),
        'diagnostic_totals': diagnostic_totals,
        'motivation_metrics': _finalize_motivation_metrics(motivation_totals),
    }


def _hard_overlap(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = torch.logical_and(prediction, target).sum().item()
    union = torch.logical_or(prediction, target).sum().item()
    prediction_area = prediction.sum().item()
    target_area = target.sum().item()
    return {
        'jc': float(intersection / union) if union else 1.0,
        'dice': (
            float(2.0 * intersection / (prediction_area + target_area))
            if prediction_area + target_area else 1.0
        ),
        'accuracy': float(prediction.eq(target).float().mean().item()),
    }

def _fixed05_segmentation_metrics(logits, masks):
    probability = torch.softmax(logits.detach(), dim=1)[:, 1]
    prediction = probability.ge(0.5)
    target = masks.eq(1)
    intersection = int((prediction & target).sum().item())
    union = int((prediction | target).sum().item())
    pred_area = int(prediction.sum().item())
    target_area = int(target.sum().item())
    return {
        'jaccard': _safe_mean(intersection, union),
        'dice': _safe_mean(2 * intersection, pred_area + target_area),
    }


def _collect_epoch_metrics(
    epoch,
    epoch_seconds,
    optimizers,
    train_losses,
    val_evaluations,
    validation_diversity,
    best_val_eval_lists,
):
    val_eval1, val_eval2 = val_evaluations
    optimizer1, _ = optimizers
    return {
        'epoch': epoch + 1,
        'learning_rate': optimizer1.param_groups[0]['lr'],
        'average_train_loss': train_losses[3],
        'val_jaccard_student1': val_eval1[1],
        'val_dice_student1': val_eval1[2],
        'val_jaccard_student2': val_eval2[1],
        'val_dice_student2': val_eval2[2],
        'best_val_jaccard1': best_val_eval_lists[0][1],
        'best_val_jaccard2': best_val_eval_lists[1][1],
        'best_val_jaccard': max(
            best_val_eval_lists[0][1], best_val_eval_lists[1][1]
        ),
        'epoch_seconds': epoch_seconds,
        **validation_diversity,
    }


def _native_validation_diagnostic(validation, dataloader, context, cfg, args):
    """Compute validation-only diagnostics for the native UNI intervention.

    Ground-truth masks are used here only to measure shared errors and the
    quality of the detached UNI token direction.  This function is outside
    the training loss path, so validation/test labels never affect training.
    """
    totals = defaultdict(float)
    events = defaultdict(float)
    offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            n = int(batch['mask'].shape[0])
            logits_a = validation['scores1'][offset:offset + n].cuda(non_blocking=True)
            logits_b = validation['scores2'][offset:offset + n].cuda(non_blocking=True)
            offset += n
            gt = batch['mask'].cuda(non_blocking=True)
            valid = gt.ne(-1)
            pa = _foreground_probability(logits_a)
            pb = _foreground_probability(logits_b)
            hard_a = pa.ge(0.5)[:, 0]
            hard_b = pb.ge(0.5)[:, 0]
            conf_a = torch.maximum(pa, 1.0 - pa)[:, 0]
            conf_b = torch.maximum(pb, 1.0 - pb)[:, 0]
            agree_px = hard_a.eq(hard_b)
            correct_a = hard_a.eq(gt)
            correct_b = hard_b.eq(gt)
            cc = correct_a & correct_b & valid
            ww = ~correct_a & ~correct_b & agree_px & valid
            hc = ww & conf_a.ge(0.9) & conf_b.ge(0.9)
            shared_fp = ww & hard_a & gt.eq(0)
            shared_fn = ww & ~hard_a & gt.eq(1)

            targets = _build_native_reference_targets(
                logits_a,
                logits_b,
                _get_rgb_view(batch).cuda(non_blocking=True),
                context,
                cfg,
                args,
            )
            contradiction = targets['contradiction']
            hc_token = targets['hc']
            occ = F.adaptive_avg_pool2d(
                gt.float().unsqueeze(1), (16, 16)
            )[:, 0]
            pca = 0.5 * (
                targets['p_a_token'][:, 0] + targets['p_b_token'][:, 0]
            )
            direction_valid = (occ - pca).abs().gt(0.05)
            uni_direction = torch.where(targets['uni_fg'][:, 0], 1.0, -1.0)
            gt_direction = occ - pca
            direction_correct = (
                uni_direction * gt_direction > 0
            ) & direction_valid
            token_ww = F.adaptive_avg_pool2d(
                ww.float().unsqueeze(1), (16, 16)
            )[:, 0]
            token_hc = F.adaptive_avg_pool2d(
                hc.float().unsqueeze(1), (16, 16)
            )[:, 0]
            token_cc = F.adaptive_avg_pool2d(
                cc.float().unsqueeze(1), (16, 16)
            )[:, 0]

            for key, value in (
                ('valid', valid.sum()),
                ('cc', cc.sum()),
                ('ww', ww.sum()),
                ('hc90ww', hc.sum()),
                ('shared_fp', shared_fp.sum()),
                ('shared_fn', shared_fn.sum()),
                ('contradiction_tokens', contradiction.sum()),
                ('hc_contradiction_tokens', hc_token.sum()),
                ('contradiction_ww_sum', token_ww[contradiction].sum()),
                ('contradiction_hc_sum', token_hc[contradiction].sum()),
                ('contradiction_cc_sum', token_cc[contradiction].sum()),
                ('hc_ww_sum', token_ww[hc_token].sum()),
                ('hc_hc_sum', token_hc[hc_token].sum()),
                ('hc_cc_sum', token_cc[hc_token].sum()),
                (
                    'direction_correct',
                    direction_correct[contradiction & direction_valid].sum(),
                ),
                (
                    'direction_valid',
                    (contradiction & direction_valid).sum(),
                ),
                ('hc_mae_sum', (pca - occ).abs()[hc_token].sum()),
            ):
                totals[key] += float(value.item())

            # Connected-component counts are validation-only diagnostics.
            try:
                from scipy import ndimage

                for idx in range(n):
                    for name, mask in (
                        ('WW', ww[idx]),
                        ('HC90WW', hc[idx]),
                        ('Shared_FP', shared_fp[idx]),
                        ('Shared_FN', shared_fn[idx]),
                    ):
                        count = int(
                            ndimage.label(
                                mask.detach().cpu().numpy(),
                                structure=np.ones((3, 3), dtype=np.uint8),
                            )[1]
                        )
                        events[name] += count
                        events[name + '_images'] += float(count > 0)
            except ImportError:
                # Pixel-level diagnostics remain valid without scipy.
                pass

    if offset != int(validation['masks'].shape[0]):
        raise AssertionError('Native validation loader/logit length mismatch')

    valid = totals['valid']
    contradiction = totals['contradiction_tokens']
    hc_tokens = totals['hc_contradiction_tokens']
    shared = {
        'epoch': 0,
        'valid_pixels': int(valid),
        'CC_rate': _safe_mean(totals['cc'], valid),
        'WW_rate': _safe_mean(totals['ww'], valid),
        'HC90WW_rate': _safe_mean(totals['hc90ww'], valid),
        'Shared_FP_rate': _safe_mean(totals['shared_fp'], valid),
        'Shared_FN_rate': _safe_mean(totals['shared_fn'], valid),
        'WW_pixels': int(totals['ww']),
        'HC90WW_pixels': int(totals['hc90ww']),
        'Shared_FP_pixels': int(totals['shared_fp']),
        'Shared_FN_pixels': int(totals['shared_fn']),
    }
    native = {
        'epoch': 0,
        'contradiction_tokens': int(contradiction),
        'hc_contradiction_tokens': int(hc_tokens),
        'contradiction_rate': _safe_mean(
            contradiction, max(1.0, totals['valid'] / 256.0)
        ),
        'hc_contradiction_rate': _safe_mean(
            hc_tokens, max(1.0, totals['valid'] / 256.0)
        ),
        'mean_WW_ratio_contradiction': _safe_mean(
            totals['contradiction_ww_sum'], contradiction
        ),
        'mean_HC90WW_ratio_contradiction': _safe_mean(
            totals['contradiction_hc_sum'], contradiction
        ),
        'mean_CC_ratio_contradiction': _safe_mean(
            totals['contradiction_cc_sum'], contradiction
        ),
        'mean_WW_ratio_hc_contradiction': _safe_mean(
            totals['hc_ww_sum'], hc_tokens
        ),
        'mean_HC90WW_ratio_hc_contradiction': _safe_mean(
            totals['hc_hc_sum'], hc_tokens
        ),
        'mean_CC_ratio_hc_contradiction': _safe_mean(
            totals['hc_cc_sum'], hc_tokens
        ),
        'direction_accuracy': _safe_mean(
            totals['direction_correct'], totals['direction_valid']
        ),
        'HC_token_MAE': _safe_mean(totals['hc_mae_sum'], hc_tokens),
    }
    event_row = {'epoch': 0}
    for name in ('WW', 'HC90WW', 'Shared_FP', 'Shared_FN'):
        event_row[name + '_events'] = int(events[name])
        event_row[name + '_images'] = int(events[name + '_images'])
        event_row[name + '_events_per_image'] = _safe_mean(
            events[name], len(dataloader.dataset)
        )
    return shared, native, event_row


def _native_checkpoint_metadata(args, context, best_val_eval_lists, best_epochs):
    """Return the immutable-reference metadata stored in native checkpoints."""
    return {
        'dataset_name': args.dataset_name,
        'labeled_ratio': args.labeled_ratio,
        'seed': args.sd,
        'input_mode': args.input_mode,
        'network': args.network,
        'cps_mode': args.cps_mode,
        'reference_mode': args.reference_mode,
        'selection_split': 'validation',
        'selection_metric': 'jaccard',
        'best_val_metrics': [list(values) for values in best_val_eval_lists],
        'best_epochs': list(best_epochs),
        'reference_memory_path': context['memory_path'],
        'reference_memory_sha256': context['memory_sha256'],
        'reference_resolution': 256,
        'reference_grid': 16,
        'reference_memory_source': 'labeled_{} GT only'.format(
            args.labeled_ratio
        ),
        'uni_sha256': context['weights_sha256'],
        'uni_weights': context['weights_path'],
        'uni_k': args.uni_k,
        'margin_calibration_records': (
            str(Path(args.margin_calibration_records).resolve())
            if args.margin_calibration_records else None
        ),
        'native_suppress_gamma': getattr(args, 'native_suppress_gamma', None),
        'native_hc_thresh': getattr(args, 'native_hc_thresh', None),
        'lambda_native_ref': getattr(args, 'lambda_native_ref', None),
        'native_gate': (
            'detach(adaptive_avg_pool2d(pA/pB,(16,16))) agreement and '
            'UNI margin sign; no GT'
        ),
        'native_weight_expansion': 'nearest loss-weight expansion only',
        'native_hc_target': (
            'UNI sign at token scale; BCE on undetached pooled probabilities'
        ),
        'test_evaluation_policy': 'final_frozen_only',
    }


def _log_epoch(logger, args, metrics):
    logger.info(
        'epoch=%d/%d | learning_rate=%.8g | average_train_loss=%.6f '
        '| val_dice_student1=%.6f | val_dice_student2=%.6f '
        '| val_jaccard_student1=%.6f | val_jaccard_student2=%.6f '
        '| student12_disagreement=%.6f '
        '| student12_dice_similarity=%.6f '
        '| student1_correct_student2_wrong=%.6f '
        '| student2_correct_student1_wrong=%.6f '
        '| student12_both_correct=%.6f '
        '| student12_both_wrong=%.6f '
        '| complementary_error=%.6f '
        '| best_val_jaccard1=%.6f | best_val_jaccard2=%.6f '
        '| best_val_jaccard=%.6f | time=%.2fs',
        metrics['epoch'],
        args.num_epochs,
        metrics['learning_rate'],
        metrics['average_train_loss'],
        metrics['val_dice_student1'],
        metrics['val_dice_student2'],
        metrics['val_jaccard_student1'],
        metrics['val_jaccard_student2'],
        metrics['student12_disagreement'],
        metrics['student12_dice_similarity'],
        metrics['student1_correct_student2_wrong'],
        metrics['student2_correct_student1_wrong'],
        metrics['student12_both_correct'],
        metrics['student12_both_wrong'],
        metrics['complementary_error'],
        metrics['best_val_jaccard1'],
        metrics['best_val_jaccard2'],
        metrics['best_val_jaccard'],
        metrics['epoch_seconds'],
    )


def _model_state_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _binary_metrics_at_threshold(logits, masks, threshold, inclusive=False):
    probability = torch.softmax(logits.detach(), dim=1)[:, 1]
    prediction = (
        probability.ge(threshold) if inclusive else probability.gt(threshold)
    )
    target = masks.eq(1)
    intersection = int((prediction & target).sum().item())
    union = int((prediction | target).sum().item())
    pred_area = int(prediction.sum().item())
    target_area = int(target.sum().item())
    return {
        'jaccard': _safe_mean(intersection, union),
        'dice': _safe_mean(2 * intersection, pred_area + target_area),
    }


def train_model(
    args,
    cfg,
    datasets,
    manifest_dir,
    use_manifests,
):
    """Run FDCL training, validation, logging, and checkpoint selection."""
    dataloaders = build_dataloaders(datasets, args.batch_size)
    num_batches = {
        name: len(dataloader) for name, dataloader in dataloaders.items()
    }
    trained_models_path, segmentation_results_path = build_output_paths(
        args, datasets, use_manifests
    )
    if _is_margin_calibrated_mode(args) and args.margin_calibration_records is None:
        args.margin_calibration_records = str(
            Path(trained_models_path)
            / 'margin_calibration_k{}'.format(args.uni_k)
            / 'selected_k_token_records.csv'
        )
    if _uses_uni_reference(args):
        existing_training_artifacts = [
            path for path in Path(trained_models_path).rglob('*')
            if path.is_file()
        ]
        completed_run = (
            Path(trained_models_path, 'native_training_complete.json').is_file()
            or any(Path(trained_models_path, 'milestones').glob('epoch_*.pth'))
        )
        if existing_training_artifacts and completed_run and not args.resume:
            raise RuntimeError(
                'Reference formal output is not empty; refusing resume/overwrite: {}'.format(
                    trained_models_path
                )
            )
        if args.resume and not os.path.isfile(os.path.join(trained_models_path, 'resume_checkpoint.pth')):
            raise RuntimeError('Requested resume but resume_checkpoint.pth is missing: {}'.format(trained_models_path))
    if _is_native_reference_mode(args):
        if args.uni_reference_memory is None:
            args.uni_reference_memory = str(
                Path(trained_models_path) / 'reference_memory_uni256.npz'
            )
    logger, metrics_path, diagnostic_logger = create_run_outputs(
        args, trained_models_path
    )
    if _is_native_reference_mode(args):
        if args.reference_mode == 'native_fine_margin_suppression':
            adaptive_strategy = 'labeled_train_margin_calibrated_suppression_only'
            loss_formula = 'w_cps=1-q; w_opp=0 on fine risky pixels'
        elif args.reference_mode == 'native_fine_margin_reverse':
            adaptive_strategy = 'labeled_train_margin_calibrated_reverse_only'
            loss_formula = 'w_cps=1; w_opp=q on fine risky pixels'
            reverse_target_description = 'opposite detached token consensus; additive only on fine risky pixels'
        else:
            adaptive_strategy = 'labeled_train_margin_calibrated_directional_mixture'
            loss_formula = 'w_cps=1-q; w_opp=q on fine risky pixels'
            reverse_target_description = 'opposite detached token consensus; mixed only on fine risky pixels'
        if args.reference_mode == 'native_fine_margin_suppression':
            reverse_target_description = 'disabled'
        native_config = {
            'experiment': args.reference_mode,
            'dataset_name': args.dataset_name,
            'labeled_ratio': args.labeled_ratio,
            'seed': args.sd,
            'cps_mode': args.cps_mode,
            'input_mode': args.input_mode,
            'uni_input_resolution': 256,
            'native_grid': '16x16',
            'feature_dim': 1024,
            'reference_memory': str(Path(args.uni_reference_memory).resolve()) if args.uni_reference_memory else 'fixed_uni256',
            'margin_calibration_records': str(Path(args.margin_calibration_records).resolve()) if args.margin_calibration_records else None,
            'adaptive_strategy': adaptive_strategy,
            'cdf': 'not used',
            'running_buffer': 'disabled',
            'd_adaptive_formula': 'q=D/(D+tau+eps); {}; tau is fixed labeled-train LOO statistic'.format(loss_formula),
            'pixel_selection': 'token contradiction AND pixel student agreement AND pixel consensus equals token consensus',
            'reverse_target': reverse_target_description,
            'contradiction_gate': 'detach pooled pA/pB; agreement and UNI margin sign only; no GT',
            'loss_weight_expansion': 'nearest 16x16 -> student pixel resolution',
            'hc_correction': 'disabled',
            'test_evaluation_policy': 'not evaluated during training',
            'command': ' '.join(sys.argv),
        }
        for root in (Path(trained_models_path), Path('outputs/native_scale_training')):
            root.mkdir(parents=True, exist_ok=True)
            (root / 'config_{}.json'.format(NATIVE_MODE_TAGS[args.reference_mode])).write_text(
                json.dumps(native_config, indent=2, ensure_ascii=False), encoding='utf-8'
            )
    reference_context = None
    native_training_writer = None
    native_training_file = None
    native_validation_writer = None
    native_validation_file = None
    native_state_writer = None
    native_state_file = None
    native_events_writer = None
    native_events_file = None
    motivation_writer = None
    motivation_file = None
    motivation_logged_epochs = set()
    if _uses_uni_reference(args):
        reference_context = _load_reference(
            args,
            cfg,
            logger,
            manifest_dir=manifest_dir,
            use_manifests=use_manifests,
        )
        reference_context['reference_build_rng_isolated'] = False
        reference_context['cfg'] = cfg

    (
        models,
        optimizers,
        schedulers,
        losses,
    ) = build_training_components(args, cfg)

    model1, model2 = models
    resume_payload = None
    if args.resume:
        resume_path = os.path.join(trained_models_path, 'resume_checkpoint.pth')
        resume_payload = torch.load(resume_path, map_location='cpu')
        expected = {
            'dataset_name': args.dataset_name,
            'labeled_ratio': args.labeled_ratio,
            'seed': args.sd,
            'reference_resolution': args.reference_resolution,
            'reference_memory_sha256': reference_context['memory_sha256'],
            'uni_sha256': reference_context['weights_sha256'],
        }
        for key, value in expected.items():
            if resume_payload.get(key) != value:
                raise RuntimeError('Resume protocol mismatch for {}: {} != {}'.format(key, resume_payload.get(key), value))
        model1.load_state_dict(resume_payload['appearance_state_dict'], strict=True)
        model2.load_state_dict(resume_payload['structure_state_dict'], strict=True)
        logger.info('Resume student weights loaded strictly | epoch=%d', int(resume_payload['epoch']))
    if reference_context is not None and _is_native_reference_mode(args):
        _run_native_startup_checks(args, reference_context, models, optimizers)
        logger.info(
            'Native-scale reference startup checks passed | mode=%s | UNI input=256 | token grid=16x16',
            args.reference_mode,
        )
    best_val_eval_lists = (
        [0 for _ in range(4)],
        [0 for _ in range(4)],
    )
    best_model_states = (None, None)
    best_epochs = (0, 0)
    best_sync_mean = -float('inf')
    best_sync_state = None
    best_sync_epoch = 0
    start_epoch = 0
    if resume_payload is not None:
        best_val_eval_lists = resume_payload['best_val_eval_lists']
        best_model_states = resume_payload['best_model_states']
        best_epochs = tuple(resume_payload['best_epochs'])
        best_sync_mean = float(resume_payload['best_sync_mean'])
        best_sync_state = resume_payload['best_sync_state']
        best_sync_epoch = int(resume_payload['best_sync_epoch'])
        start_epoch = int(resume_payload['epoch'])
        optimizers[0].load_state_dict(resume_payload['optimizer1'])
        optimizers[1].load_state_dict(resume_payload['optimizer2'])
        schedulers[0].load_state_dict(resume_payload['scheduler1'])
        schedulers[1].load_state_dict(resume_payload['scheduler2'])
        global_step = int(resume_payload['global_step'])
        _restore_rng_state(resume_payload['rng_state'])
    else:
        global_step = 0
    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)
    training_start = time.time()
    # global_step is restored above for a resumed run.

    with ExitStack() as stack:
        metrics_file = stack.enter_context(
            open(metrics_path, 'a' if args.resume else 'w', newline='', encoding='utf-8')
        )
        if _is_native_reference_mode(args):
            native_root = Path(
                args.native_metrics_dir
                if getattr(args, 'native_metrics_dir', None)
                else 'outputs/native_scale_training'
            )
            native_root.mkdir(parents=True, exist_ok=True)
            mode_tag = NATIVE_MODE_TAGS[args.reference_mode]
            native_training_file = stack.enter_context(open(
                native_root / 'training_{}.csv'.format(mode_tag),
                'a' if args.resume else 'w', newline='', encoding='utf-8'
            ))
            native_validation_file = stack.enter_context(open(
                native_root / 'validation_shared_errors_{}.csv'.format(mode_tag),
                'a' if args.resume else 'w', newline='', encoding='utf-8'
            ))
            native_state_file = stack.enter_context(open(
                native_root / 'validation_native_states_{}.csv'.format(mode_tag),
                'a' if args.resume else 'w', newline='', encoding='utf-8'
            ))
            native_events_file = stack.enter_context(open(
                native_root / 'validation_events_{}.csv'.format(mode_tag),
                'a' if args.resume else 'w', newline='', encoding='utf-8'
            ))
            native_training_writer = csv.DictWriter(native_training_file, fieldnames=NATIVE_TRAINING_FIELDS)
            shared_fields = ['epoch', 'valid_pixels', 'CC_rate', 'WW_rate', 'HC90WW_rate', 'Shared_FP_rate', 'Shared_FN_rate', 'WW_pixels', 'HC90WW_pixels', 'Shared_FP_pixels', 'Shared_FN_pixels']
            native_validation_writer = csv.DictWriter(native_validation_file, fieldnames=shared_fields)
            state_fields = ['epoch', 'contradiction_tokens', 'hc_contradiction_tokens', 'contradiction_rate', 'hc_contradiction_rate', 'mean_WW_ratio_contradiction', 'mean_HC90WW_ratio_contradiction', 'mean_CC_ratio_contradiction', 'mean_WW_ratio_hc_contradiction', 'mean_HC90WW_ratio_hc_contradiction', 'mean_CC_ratio_hc_contradiction', 'direction_accuracy', 'HC_token_MAE']
            native_state_writer = csv.DictWriter(native_state_file, fieldnames=state_fields)
            event_fields = ['epoch'] + [x for name in ('WW', 'HC90WW', 'Shared_FP', 'Shared_FN') for x in (name + '_events', name + '_images', name + '_events_per_image')]
            native_events_writer = csv.DictWriter(native_events_file, fieldnames=event_fields)
            if not args.resume:
                native_training_writer.writeheader(); native_validation_writer.writeheader(); native_state_writer.writeheader(); native_events_writer.writeheader()
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=METRIC_FIELDS
        )
        if not args.resume:
            metrics_writer.writeheader()

        # Motivation logging is independent from training metrics and is never
        # used for loss, checkpoint selection, or early stopping.  Every CPS
        # mode writes the same file inside its own checkpoint experiment path.
        motivation_path = Path(trained_models_path) / 'vanilla_cps_motivation_metrics.csv'
        if args.resume and motivation_path.is_file():
            with motivation_path.open('r', newline='', encoding='utf-8') as existing_file:
                existing_reader = csv.DictReader(existing_file)
                for existing_row in existing_reader:
                    try:
                        motivation_logged_epochs.add(int(existing_row['epoch']))
                    except (KeyError, TypeError, ValueError):
                        continue
        motivation_file = stack.enter_context(
            motivation_path.open(
                'a' if args.resume else 'w',
                newline='',
                encoding='utf-8',
            )
        )
        motivation_writer = csv.DictWriter(
            motivation_file,
            fieldnames=MOTIVATION_METRIC_FIELDS,
        )
        if not args.resume or motivation_file.tell() == 0:
            motivation_writer.writeheader()
            motivation_file.flush()

        for epoch in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            torch.cuda.reset_peak_memory_stats()
            loss_totals, train_predictions, global_step, reference_totals = train_one_epoch(
                args,
                epoch,
                global_step,
                models,
                optimizers,
                losses,
                dataloaders,
                metrics_writer,
                metrics_file,
                reference_context=reference_context,
            )
            schedulers[0].step()
            schedulers[1].step()

            current_epoch = epoch + 1
            print_epoch = (
                current_epoch % 5 == 0 or current_epoch == args.num_epochs
            )
            if print_epoch:
                print('=' * print_num)
                print(
                    '| Epoch {}/{}'.format(
                        current_epoch, args.num_epochs
                    ).ljust(print_num_minus, ' '),
                    '|',
                )
                train_losses = print_train_loss(
                    loss_totals['sup1'],
                    loss_totals['sup2'],
                    loss_totals['unsup'],
                    loss_totals['total'],
                    num_batches,
                    print_num,
                    print_num_half,
                )
                print_train_eval(
                    cfg['NUM_CLASSES'],
                    *train_predictions,
                    print_num_half,
                )
            else:
                train_losses = tuple(
                    loss_totals[key] / num_batches['train_sup']
                    for key in ('sup1', 'sup2', 'unsup', 'total')
                )

            validation = validate(models, losses, dataloaders['val'])
            if print_epoch:
                print_val_loss(
                    validation['loss1'],
                    validation['loss2'],
                    num_batches,
                    print_num,
                    print_num_half,
                )
                val_eval1, val_eval2, _, _ = print_val_eval(
                    cfg['NUM_CLASSES'],
                    validation['scores1'],
                    validation['scores2'],
                    validation['masks'],
                    print_num_half,
                )
            else:
                with redirect_stdout(io.StringIO()):
                    val_eval1, val_eval2, _, _ = print_val_eval(
                        cfg['NUM_CLASSES'],
                        validation['scores1'],
                        validation['scores2'],
                        validation['masks'],
                        print_num_half,
                    )
            if motivation_writer is not None:
                motivation = validation['motivation_metrics']
                motivation_row = {
                    'epoch': current_epoch,
                    **motivation,
                }
                if current_epoch not in motivation_logged_epochs:
                    motivation_writer.writerow(motivation_row)
                    motivation_file.flush()
                    motivation_logged_epochs.add(current_epoch)
                print(
                    '[Motivation][Epoch {:03d}] Agreement={:.6f} '
                    'SoftDisagreement={:.6f} WW|Agree={:.6f} '
                    'HC90WW|WW={:.6f}'.format(
                        current_epoch,
                        motivation['agreement'],
                        motivation['soft_disagreement'],
                        motivation['ww_given_agreement'],
                        motivation['hc90ww_given_ww'],
                    )
                )
            if _is_native_reference_mode(args):
                shared_row, native_row, event_row = _native_validation_diagnostic(
                    validation, dataloaders['val'], reference_context, cfg, args
                )
                shared_row['epoch'] = current_epoch
                native_row['epoch'] = current_epoch
                event_row['epoch'] = current_epoch
                native_validation_writer.writerow(shared_row); native_validation_file.flush()
                native_state_writer.writerow(native_row); native_state_file.flush()
                native_events_writer.writerow(event_row); native_events_file.flush()

            # Test is intentionally absent from the training loop for both
            # standard and state-aware CPS.  Epoch selection and diagnostics
            # use validation only; final test evaluation is a separate stage.
            diagnostics = _finalize_diagnostics(
                validation['diagnostic_totals'],
                val_eval1[2],
                val_eval2[2],
            )
            log_diagnostics(
                diagnostic_logger,
                epoch + 1,
                global_step,
                diagnostics,
            )
            validation_diversity = compute_validation_diversity(validation)
            if print_epoch:
                print(
                    '| Val diversity: disagreement={:.4f} | dice_similarity={:.4f} | '
                    'complementary_error={:.4f}'.format(
                        validation_diversity['student12_disagreement'],
                        validation_diversity['student12_dice_similarity'],
                        validation_diversity['complementary_error'],
                    )
                )
            (
                best_val_eval_lists,
                best_model_states,
                best_epochs,
            ) = save_val_best_2d(
                best_val_eval_lists,
                best_model_states,
                best_epochs,
                model1,
                model2,
                val_eval1,
                val_eval2,
                epoch,
                split_name='Val',
            )
            sync_mean = 0.5 * (float(val_eval1[1]) + float(val_eval2[1]))
            if sync_mean > best_sync_mean:
                best_sync_mean = sync_mean
                best_sync_epoch = current_epoch
                best_sync_state = (
                    _model_state_snapshot(model1), _model_state_snapshot(model2)
                )
                sync_metadata = {}
                if _is_native_reference_mode(args):
                    sync_metadata = _native_checkpoint_metadata(
                        args, reference_context, best_val_eval_lists, best_epochs
                    )
                torch.save({
                    'epoch': current_epoch,
                    'appearance_state_dict': best_sync_state[0],
                    'structure_state_dict': best_sync_state[1],
                    'best_synchronized_mean_val_jaccard': sync_mean,
                    **sync_metadata,
                }, os.path.join(trained_models_path, 'best_synchronized_mean_val.pth'))
            if (
                args.milestone_interval > 0
                and current_epoch % args.milestone_interval == 0
            ):
                milestone_metadata = {
                    'global_step': global_step,
                    'dataset_name': args.dataset_name,
                    'labeled_ratio': args.labeled_ratio,
                    'seed': args.sd,
                    'input_mode': args.input_mode,
                    'network': args.network,
                    'cps_mode': args.cps_mode,
                    'reference_mode': getattr(args, 'reference_mode', 'none'),
                    'selection_split': 'validation',
                    'selection_metric': 'jaccard',
                    'current_test_metrics': None,
                    'test_evaluation_policy': 'not_evaluated_during_training',
                    'current_val_metrics': [
                        list(val_eval1),
                        list(val_eval2),
                    ],
                    'best_val_metrics': [
                        list(values)
                        for values in best_val_eval_lists
                    ],
                    'best_epochs': list(best_epochs),
                    'args': vars(args).copy(),
                }
                if _is_native_reference_mode(args):
                    milestone_metadata.update(_native_checkpoint_metadata(
                        args, reference_context, best_val_eval_lists, best_epochs
                    ))
                milestone_path = save_training_milestone(
                    model1,
                    model2,
                    trained_models_path,
                    current_epoch,
                    metadata=milestone_metadata,
                )
                logger.info(
                    'Current-weight milestone saved | epoch=%d | path=%s',
                    current_epoch,
                    milestone_path,
                )
                print(
                    '| Current student weights saved at epoch {}'.format(
                        current_epoch
                    ).ljust(print_num_minus, ' '),
                    '|',
                )
                if _is_native_reference_mode(args):
                    torch.save({
                        'epoch': current_epoch,
                        'global_step': global_step,
                        'appearance_state_dict': _model_state_snapshot(model1),
                        'structure_state_dict': _model_state_snapshot(model2),
                        'optimizer1': optimizers[0].state_dict(),
                        'optimizer2': optimizers[1].state_dict(),
                        'scheduler1': schedulers[0].state_dict(),
                        'scheduler2': schedulers[1].state_dict(),
                        'best_val_eval_lists': best_val_eval_lists,
                        'best_model_states': best_model_states,
                        'best_epochs': best_epochs,
                        'best_sync_mean': best_sync_mean,
                        'best_sync_state': best_sync_state,
                        'best_sync_epoch': best_sync_epoch,
                        'rng_state': _snapshot_rng_state(),
                        **_native_checkpoint_metadata(
                            args, reference_context, best_val_eval_lists, best_epochs
                        ),
                    }, os.path.join(trained_models_path, 'resume_checkpoint.pth'))
            if (
                args.best_save_interval > 0
                and current_epoch % args.best_save_interval == 0
            ):
                periodic_checkpoints = save_periodic_branch_bests(
                    best_val_eval_lists,
                    best_model_states,
                    best_epochs,
                    trained_models_path,
                    current_epoch,
                    split_name='Val',
                )
                for checkpoint in periodic_checkpoints:
                    logger.info(
                        'Periodic best saved | current_epoch=%d | branch=%s '
                        '| best_epoch=%d | best_val_jaccard=%.6f | path=%s',
                        current_epoch,
                        checkpoint['branch'],
                        checkpoint['best_epoch'],
                        checkpoint['best_jaccard'],
                        checkpoint['path'],
                    )
                print(
                    '| Periodic best student weights saved at epoch {}'.format(
                        current_epoch
                    ).ljust(print_num_minus, ' '),
                    '|',
                )

            epoch_seconds = time.time() - epoch_start
            if print_epoch:
                print('-' * print_num)
                print(
                    '| Epoch Time: {:.4f}s'.format(epoch_seconds).ljust(
                        print_num_minus, ' '
                    ),
                    '|',
                )
            metrics = _collect_epoch_metrics(
                epoch,
                epoch_seconds,
                optimizers,
                train_losses,
                (val_eval1, val_eval2),
                validation_diversity,
                best_val_eval_lists,
            )
            _log_epoch(logger, args, metrics)
            if _is_native_reference_mode(args):
                native = reference_totals
                token_count = native['num_tokens']
                agreement_count = native['agreement_tokens']
                contradiction_count = native['contradiction_tokens']
                hc_count = native['hc_tokens']
                s_cur_values = np.asarray(native.get('s_cur_values', []), dtype=np.float64)
                s_opp_values = np.asarray(native.get('s_opp_values', []), dtype=np.float64)
                d_values = np.asarray(native.get('d_values', []), dtype=np.float64)
                relative_gaps = np.asarray(native.get('relative_gaps', []), dtype=np.float64)
                adaptive_weights = np.asarray(native.get('adaptive_weights', []), dtype=np.float64)
                adaptive_stats = {
                    'mean_S_cur_contradiction': float(s_cur_values.mean()) if s_cur_values.size else None,
                    'mean_S_opp_contradiction': float(s_opp_values.mean()) if s_opp_values.size else None,
                    'mean_absolute_D': float(d_values.mean()) if d_values.size else None,
                    'mean_relative_gap': float(relative_gaps.mean()) if relative_gaps.size else None,
                    'relative_gap_p10': float(np.percentile(relative_gaps, 10)) if relative_gaps.size else None,
                    'relative_gap_p25': float(np.percentile(relative_gaps, 25)) if relative_gaps.size else None,
                    'relative_gap_p50': float(np.percentile(relative_gaps, 50)) if relative_gaps.size else None,
                    'relative_gap_p75': float(np.percentile(relative_gaps, 75)) if relative_gaps.size else None,
                    'relative_gap_p90': float(np.percentile(relative_gaps, 90)) if relative_gaps.size else None,
                    'mean_adaptive_weight': float(adaptive_weights.mean()) if adaptive_weights.size else None,
                    'weight_p10': float(np.percentile(adaptive_weights, 10)) if adaptive_weights.size else None,
                    'weight_p25': float(np.percentile(adaptive_weights, 25)) if adaptive_weights.size else None,
                    'weight_p50': float(np.percentile(adaptive_weights, 50)) if adaptive_weights.size else None,
                    'weight_p75': float(np.percentile(adaptive_weights, 75)) if adaptive_weights.size else None,
                    'weight_p90': float(np.percentile(adaptive_weights, 90)) if adaptive_weights.size else None,
                    'fraction_weight_0_0.2': float(((adaptive_weights >= 0.0) & (adaptive_weights < .2)).mean()) if adaptive_weights.size else None,
                    'fraction_weight_0.2_0.4': float(((adaptive_weights >= .2) & (adaptive_weights < .4)).mean()) if adaptive_weights.size else None,
                    'fraction_weight_0.4_0.6': float(((adaptive_weights >= .4) & (adaptive_weights < .6)).mean()) if adaptive_weights.size else None,
                    'fraction_weight_0.6_0.8': float(((adaptive_weights >= .6) & (adaptive_weights < .8)).mean()) if adaptive_weights.size else None,
                    'fraction_weight_0.8_1.0': float(((adaptive_weights >= .8) & (adaptive_weights <= 1.0)).mean()) if adaptive_weights.size else None,
                }
                if d_values.size:
                    adaptive_stats.update({
                        'mean_D_contradiction': float(d_values.mean()),
                        'D_p10': float(np.percentile(d_values, 10)),
                        'D_p25': float(np.percentile(d_values, 25)),
                        'D_p50': float(np.percentile(d_values, 50)),
                        'D_p75': float(np.percentile(d_values, 75)),
                        'D_p90': float(np.percentile(d_values, 90)),
                    })
                if adaptive_weights.size:
                    adaptive_stats.update({
                        'mean_contradiction_weight': float(adaptive_weights.mean()),
                        'contradiction_weight_p10': float(np.percentile(adaptive_weights, 10)),
                        'contradiction_weight_p25': float(np.percentile(adaptive_weights, 25)),
                        'contradiction_weight_p50': float(np.percentile(adaptive_weights, 50)),
                        'contradiction_weight_p75': float(np.percentile(adaptive_weights, 75)),
                        'contradiction_weight_p90': float(np.percentile(adaptive_weights, 90)),
                        'fraction_weight_0_0.1': float(((adaptive_weights >= 0.0) & (adaptive_weights < .1)).mean()),
                        'fraction_weight_0.1_0.2': float(((adaptive_weights >= .1) & (adaptive_weights < .2)).mean()),
                        'fraction_weight_0.2_0.3': float(((adaptive_weights >= .2) & (adaptive_weights < .3)).mean()),
                        'fraction_weight_0.3_0.4': float(((adaptive_weights >= .3) & (adaptive_weights < .4)).mean()),
                        'fraction_weight_0.4_0.5': float(((adaptive_weights >= .4) & (adaptive_weights <= .5)).mean()),
                        'fraction_zero_weight': float((adaptive_weights == 0.0).mean()),
                    })
                if adaptive_weights.size:
                    if not np.isfinite(relative_gaps).all() or not np.isfinite(adaptive_weights).all() or not np.all((relative_gaps >= 0.0) & (relative_gaps <= 1.0)) or not np.all((adaptive_weights >= 0.0) & (adaptive_weights <= 1.0)):
                        raise AssertionError('Direct relative gap/weight escaped [0,1] or became non-finite')
                if _is_margin_calibrated_mode(args):
                    candidate = native['regionwise_candidate_pixels']
                    selected = native['fine_selected_pixels']
                    all_pixels = native['all_pixel_count']
                    adaptive_stats.update({
                        'regionwise_candidate_pixels': int(candidate),
                        'fine_selected_pixels': int(selected),
                        'fine_selected_over_regionwise': _safe_mean(selected, candidate),
                        'R1_same_consensus': _safe_mean(native['R1_same_consensus_pixels'], candidate),
                        'R2_opposite_consensus': _safe_mean(native['R2_opposite_consensus_pixels'], candidate),
                        'R3_student_disagreement': _safe_mean(native['R3_student_disagreement_pixels'], candidate),
                        'fine_suppression_coverage': _safe_mean(selected, candidate),
                        'global_suppression_coverage': _safe_mean(selected, all_pixels),
                        'mean_applied_pixel_weight': _safe_mean(native['sum_applied_pixel_weight'], all_pixels),
                    })
                    rsum = sum(adaptive_stats[name] for name in (
                        'R1_same_consensus', 'R2_opposite_consensus', 'R3_student_disagreement'
                    ))
                    if candidate and abs(rsum - 1.0) > 1e-6:
                        raise AssertionError('Fine R1+R2+R3 != 1')
                if _is_margin_calibrated_mode(args):
                    q_risk = np.asarray(native['q_values'], dtype=np.float64)
                    d_risk = np.asarray(native['q_d_values'], dtype=np.float64)
                    q_cps = np.asarray(native['q_cps_weights'], dtype=np.float64)
                    q_opp = np.asarray(native['q_opp_weights'], dtype=np.float64)
                    q_errors = np.asarray(native['q_weight_sum_errors'], dtype=np.float64)
                    if q_risk.size:
                        if not np.isfinite(q_risk).all() or not np.all((q_risk >= 0.0) & (q_risk <= 1.0)):
                            raise AssertionError('Margin-calibrated risk values escaped [0,1]')
                        if q_errors.size and float(q_errors.max()) >= 1e-6:
                            raise AssertionError('Margin-calibrated risky budget sum is not exact')
                        adaptive_stats.update({
                            'num_risk_pixels': int(native['risk_pixels']),
                            'risk_pixel_fraction': _safe_mean(native['risk_pixels'], native['risk_all_pixel_count']),
                            'mean_D_risk': float(d_risk.mean()),
                            'D_risk_p10': float(np.percentile(d_risk, 10)),
                            'D_risk_p25': float(np.percentile(d_risk, 25)),
                            'D_risk_p50': float(np.percentile(d_risk, 50)),
                            'D_risk_p75': float(np.percentile(d_risk, 75)),
                            'D_risk_p90': float(np.percentile(d_risk, 90)),
                            'mean_q_risk': float(q_risk.mean()),
                            'q_risk_p10': float(np.percentile(q_risk, 10)),
                            'q_risk_p25': float(np.percentile(q_risk, 25)),
                            'q_risk_p50': float(np.percentile(q_risk, 50)),
                            'q_risk_p75': float(np.percentile(q_risk, 75)),
                            'q_risk_p90': float(np.percentile(q_risk, 90)),
                            'mean_CPS_weight_risk': float(q_cps.mean()),
                            'mean_opposite_weight_risk': float(q_opp.mean()),
                            'risk_budget': float((q_cps + q_opp).mean()),
                            'risk_weight_sum_max_error': float(q_errors.max()) if q_errors.size else 0.0,
                            'loss_risk_severity': _safe_mean(native['q_risk_loss'], native['batches']),
                        })
                native_training_writer.writerow({
                    'experiment': args.reference_mode,
                    'epoch': epoch + 1,
                    'num_native_tokens': int(token_count),
                    'num_agreement_tokens': int(agreement_count),
                    'num_contradiction_tokens': int(contradiction_count),
                    'contradiction_rate': _safe_mean(contradiction_count, agreement_count),
                    'num_hc_contradiction_tokens': int(hc_count),
                    'hc_contradiction_rate': _safe_mean(hc_count, contradiction_count),
                    'mean_CPS_weight': _safe_mean(native['sum_weight'], token_count),
                    'loss_sup': 0.5 * (train_losses[0] + train_losses[1]) / num_batches['train_sup'],
                    'loss_cps_raw': _safe_mean(native['raw_cps_loss'], native['batches']),
                    'loss_cps_weighted': _safe_mean(native['weighted_cps_loss'], native['batches']),
                    'loss_native_ref': _safe_mean(native['native_ref_loss'], native['batches']),
                    'loss_total': _safe_mean(native['total_loss'], native['batches']),
                    'mean_pA_token_on_HC': _safe_mean(native['sum_p_a_hc'], hc_count * 1.0),
                    'mean_pB_token_on_HC': _safe_mean(native['sum_p_b_hc'], hc_count * 1.0),
                    'UNI_FG_HC_count': int(native['uni_fg_hc']),
                    'UNI_BG_HC_count': int(native['uni_bg_hc']),
                    **adaptive_stats,
                })
                native_training_file.flush()

    if _is_native_reference_mode(args):
        native_last_path = os.path.join(trained_models_path, 'last_checkpoint.pth')
        torch.save({
            'epoch': args.num_epochs,
            'appearance_state_dict': _model_state_snapshot(model1),
            'structure_state_dict': _model_state_snapshot(model2),
            **_native_checkpoint_metadata(
                args, reference_context, best_val_eval_lists, best_epochs
            ),
        }, native_last_path)
    elapsed = time.time() - training_start
    minutes, seconds = divmod(elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print('=' * print_num)
    print(
        '| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(
            hours, minutes, seconds
        ).ljust(print_num_minus, ' '),
        '|',
    )
    print('-' * print_num)
    print_and_save_branch_bests(
        cfg['NUM_CLASSES'],
        best_val_eval_lists,
        best_model_states,
        best_epochs,
        trained_models_path,
        print_num_minus,
        split_name='Val',
    )
    print('=' * print_num)
    logger.info(
        'Training completed | result1_best_epoch=%d '
        '| result1_best_val_jaccard=%.6f | result2_best_epoch=%d '
        '| result2_best_val_jaccard=%.6f',
        best_epochs[0],
        best_val_eval_lists[0][1],
        best_epochs[1],
        best_val_eval_lists[1][1],
    )
    if _uses_uni_reference(args):
        best_paths = [
            os.path.join(
                trained_models_path,
                'best_result{}_Jc_{:.4f}.pth'.format(
                    index, best_val_eval_lists[index - 1][1]
                ),
            )
            for index in (1, 2)
        ]
        completion = {
            'completed_epochs': args.num_epochs,
            'milestone_count': len(list(
                (Path(trained_models_path) / 'milestones').glob('epoch_*.pth')
            )),
            'best_epochs': list(best_epochs),
            'best_synchronized_epoch': best_sync_epoch,
            'best_synchronized_mean_val_jaccard': best_sync_mean,
            'best_validation_metrics': [
                list(values) for values in best_val_eval_lists
            ],
            'best_checkpoint_A': best_paths[0],
            'best_checkpoint_B': best_paths[1],
            'epoch200_checkpoint': str(
                Path(trained_models_path) / 'milestones' / 'epoch_200.pth'
            ),
            'last_checkpoint': native_last_path,
            'final_frozen_evaluation': None,
            'test_policy': 'final_frozen_only_after_epoch200',
            'training_elapsed_seconds_including_final_evaluation': elapsed,
        }
        completion_name = 'native_training_complete.json'
        with open(
            os.path.join(trained_models_path, completion_name),
            'w', encoding='utf-8'
        ) as handle:
            json.dump(completion, handle, indent=2, ensure_ascii=False)


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
            'uses q=D/(D+tau) with train-only LOO tau.'
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

    train_model(args, cfg, datasets, manifest_dir, use_manifests)


if __name__ == '__main__':
    main()
