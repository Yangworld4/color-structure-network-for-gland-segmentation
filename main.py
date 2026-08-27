import argparse
import csv
import logging
import os
import random
import time
from typing import Dict, Mapping, Tuple
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
from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.dataset_2d import imagefloder_itn
from loss.cssml import cdfl_loss, gmps_loss, masked_ce
from loss.loss_function import DiceLoss
from models.getnetwork import get_network


simplefilter(action='ignore', category=FutureWarning)

METRIC_FIELDS = [
    'iteration', 'epoch',
    'train_loss_sup1', 'train_loss_sup2', 'train_loss_cps',
    'train_loss_total', 'student12_disagreement',
    'student12_dice_similarity', 'student1_entropy', 'student2_entropy',
    'pseudo_dice_student1', 'pseudo_dice_student2',
    'pseudo_jaccard_student1', 'pseudo_jaccard_student2',
]

CSSML_DIAGNOSTIC_FIELDS = [
    'epoch',
    'cdfl_loss_labeled',
    'cdfl_loss_unlabeled',
    'cdfl_loss_total',
    'feature_cosine_labeled',
    'feature_cosine_unlabeled',
    'feature_cka',
    'agreement_ratio',
    'conflict_ratio',
    'prediction_similarity',
    'same_pixels',
    'conflict_pixels',
    'same_ratio',
    'same_a2b_pixels',
    'same_b2a_pixels',
    'conflict_a2b_pixels',
    'conflict_b2a_pixels',
    'a2b_total_ratio',
    'b2a_total_ratio',
    'loss_gmps_same',
    'loss_gmps_conflict',
    'loss_gmps_total',
    'loss_sup',
    'loss_total',
    'conflict_pixels_gt',
    'gmps_conflict_teacher_correct_pixels',
    'gmps_conflict_teacher_wrong_pixels',
    'gmps_conflict_selector_accuracy',
    'gmps_conflict_harmful_ratio',
    'same_pixels_gt',
    'same_both_correct_ratio',
    'same_both_wrong_ratio',
]

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_training_views(
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Student1 and Student2 inputs."""
    if 'image_aug1' in batch and 'image_aug2' in batch and 'mask' not in batch:
        student1 = batch['image_aug1']
        student2 = batch['image_aug2']
    elif 'image_aug1' in batch and 'image_aug2' in batch:
        student1 = batch['image']
        student2 = batch['image']
    else:
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


def compute_supervised_loss(
    ce_loss,
    dice_loss,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        ce_loss(prediction, target.long())
        + dice_loss(prediction, target)
    )


def compute_cps_losses(
    ce_loss,
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute hard-pseudo-label CPS while detaching every teacher path."""
    pseudo1 = prediction1.detach().argmax(dim=1)
    pseudo2 = prediction2.detach().argmax(dim=1)
    loss_1_to_2 = 0.5 * ce_loss(prediction2, pseudo1)
    loss_2_to_1 = 0.5 * ce_loss(prediction1, pseudo2)
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': loss_1_to_2 + loss_2_to_1}


def _pseudo_entropy_from_logits(logits: torch.Tensor):
    logits = logits.detach()
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits[:, 0, ...])
        background = 1.0 - foreground
        pseudo = (foreground >= 0.5).long()
        entropy = -(
            foreground * torch.log(foreground + 1e-6)
            + background * torch.log(background + 1e-6)
        )
    else:
        probabilities = torch.softmax(logits, dim=1)
        pseudo = probabilities.argmax(dim=1)
        entropy = -(
            probabilities * torch.log(probabilities + 1e-6)
        ).sum(dim=1)
    return pseudo, entropy


def compute_relative_reliable_cps(prediction1, prediction2):
    pseudo1, entropy1 = _pseudo_entropy_from_logits(prediction1)
    pseudo2, entropy2 = _pseudo_entropy_from_logits(prediction2)
    route_1_to_2 = (entropy1 < entropy2).detach()
    route_2_to_1 = (entropy2 < entropy1).detach()
    loss_1_to_2 = 0.5 * masked_ce(prediction2, pseudo1, route_1_to_2)
    loss_2_to_1 = 0.5 * masked_ce(prediction1, pseudo2, route_2_to_1)
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': loss_1_to_2 + loss_2_to_1,
    }


def compute_cps_losses_by_mode(args, ce_loss, prediction1, prediction2):
    if args.cps_mode == 'relative_reliable':
        return compute_relative_reliable_cps(prediction1, prediction2)
    return compute_cps_losses(ce_loss, prediction1, prediction2)


def _hard_predictions(scores: torch.Tensor) -> torch.Tensor:
    return scores.detach().argmax(dim=1)


def _entropy(scores: torch.Tensor) -> float:
    probabilities = torch.softmax(scores.detach(), dim=1).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    return float(entropy.mean().item())


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


def _needs_cssml_features(args):
    return args.use_cdfl or (args.cps_mode == 'cssml' and args.use_gmps)


def _forward_student(model, inputs, return_features):
    if return_features:
        logits, feature = model(inputs, return_features=True)
        return logits, feature
    output = model(inputs)
    return output[0], None


def _init_cssml_diagnostic_totals():
    return {
        key: 0.0
        for key in [
            'batches',
            'pixels',
            'cdfl_labeled_sum',
            'cdfl_unlabeled_sum',
            'cdfl_total_sum',
            'cos_labeled_sum',
            'cos_unlabeled_sum',
            'cka_sum',
            'cka_count',
            'same_pixels',
            'conflict_pixels',
            'same_a2b_pixels',
            'same_b2a_pixels',
            'conflict_a2b_pixels',
            'conflict_b2a_pixels',
            'loss_gmps_same_sum',
            'loss_gmps_same_pixels',
            'loss_gmps_conflict_sum',
            'loss_gmps_conflict_pixels',
            'loss_sup_sum',
            'loss_total_sum',
            'conflict_pixels_gt',
            'gmps_conflict_teacher_correct_pixels',
            'gmps_conflict_teacher_wrong_pixels',
            'same_pixels_gt',
            'same_both_correct_pixels',
            'same_both_wrong_pixels',
        ]
    }


@torch.no_grad()
def _pseudo_from_logits(logits):
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0, ...]) >= 0.5).long()
    return torch.softmax(logits, dim=1).argmax(dim=1)


@torch.no_grad()
def _update_cssml_prediction_diagnostics(totals, logits_a, logits_b):
    pseudo_a = _pseudo_from_logits(logits_a)
    pseudo_b = _pseudo_from_logits(logits_b)
    same = pseudo_a.eq(pseudo_b)
    conflict = ~same
    totals['pixels'] += float(same.numel())
    totals['same_pixels'] += float(same.sum().item())
    totals['conflict_pixels'] += float(conflict.sum().item())


def _update_cssml_cdfl_diagnostics(totals, cdfl_result, feat_a, feat_b, pixels):
    totals['batches'] += 1.0
    totals['cdfl_labeled_sum'] += cdfl_result['labeled'].item() * pixels
    totals['cdfl_unlabeled_sum'] += cdfl_result['unlabeled'].item() * pixels
    totals['cdfl_total_sum'] += cdfl_result['total'].item() * pixels
    totals['cos_labeled_sum'] += cdfl_result['cos_labeled'].item() * pixels
    totals['cos_unlabeled_sum'] += cdfl_result['cos_unlabeled'].item() * pixels
    cka = _linear_cka(feat_a.detach(), feat_b.detach())
    if not np.isnan(cka):
        totals['cka_sum'] += cka
        totals['cka_count'] += 1.0


@torch.no_grad()
def _update_cssml_gmps_diagnostics(totals, gmps_result, logits_a, logits_b):
    del logits_a, logits_b
    for key, mask_key in (
        ('same_a2b_pixels', 'same_a2b'),
        ('same_b2a_pixels', 'same_b2a'),
        ('conflict_a2b_pixels', 'conflict_a2b'),
        ('conflict_b2a_pixels', 'conflict_b2a'),
    ):
        totals[key] += float(gmps_result[mask_key].sum().item())
    same_pixels = float(
        (
            gmps_result['same_a2b'] | gmps_result['same_b2a']
        ).sum().item()
    )
    conflict_pixels = float(
        (
            gmps_result['conflict_a2b'] | gmps_result['conflict_b2a']
        ).sum().item()
    )
    totals['loss_gmps_same_sum'] += gmps_result['loss_same'].item() * same_pixels
    totals['loss_gmps_same_pixels'] += same_pixels
    totals['loss_gmps_conflict_sum'] += (
        gmps_result['loss_conflict'].item() * conflict_pixels
    )
    totals['loss_gmps_conflict_pixels'] += conflict_pixels


@torch.no_grad()
def _update_cssml_gt_diagnostics(totals, gmps_result, logits_a, logits_b, mask):
    pseudo_a = _pseudo_from_logits(logits_a)
    pseudo_b = _pseudo_from_logits(logits_b)
    target = mask.long()
    same = pseudo_a.eq(pseudo_b)
    conflict = pseudo_a.ne(pseudo_b)
    correct_a = pseudo_a.eq(target)
    correct_b = pseudo_b.eq(target)

    same_both_correct = same & correct_a & correct_b
    same_both_wrong = same & (~correct_a) & (~correct_b)
    totals['same_pixels_gt'] += float(same.sum().item())
    totals['same_both_correct_pixels'] += float(same_both_correct.sum().item())
    totals['same_both_wrong_pixels'] += float(same_both_wrong.sum().item())

    if gmps_result is None:
        return
    conflict_a2b = gmps_result['conflict_a2b'] & conflict
    conflict_b2a = gmps_result['conflict_b2a'] & conflict
    teacher_correct = (conflict_a2b & correct_a) | (conflict_b2a & correct_b)
    teacher_wrong = (conflict_a2b & (~correct_a)) | (conflict_b2a & (~correct_b))
    totals['conflict_pixels_gt'] += float((conflict_a2b | conflict_b2a).sum().item())
    totals['gmps_conflict_teacher_correct_pixels'] += float(
        teacher_correct.sum().item()
    )
    totals['gmps_conflict_teacher_wrong_pixels'] += float(teacher_wrong.sum().item())


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else 0.0


def _finalize_cssml_diagnostics(epoch, totals):
    pixels = totals['pixels']
    same = totals['same_pixels']
    conflict = totals['conflict_pixels']
    routed_a2b = totals['same_a2b_pixels'] + totals['conflict_a2b_pixels']
    routed_b2a = totals['same_b2a_pixels'] + totals['conflict_b2a_pixels']
    routed = routed_a2b + routed_b2a
    gmps_same = _ratio(
        totals['loss_gmps_same_sum'], totals['loss_gmps_same_pixels']
    )
    gmps_conflict = _ratio(
        totals['loss_gmps_conflict_sum'], totals['loss_gmps_conflict_pixels']
    )
    return {
        'epoch': epoch,
        'cdfl_loss_labeled': _ratio(totals['cdfl_labeled_sum'], pixels),
        'cdfl_loss_unlabeled': _ratio(totals['cdfl_unlabeled_sum'], pixels),
        'cdfl_loss_total': _ratio(totals['cdfl_total_sum'], pixels),
        'feature_cosine_labeled': _ratio(totals['cos_labeled_sum'], pixels),
        'feature_cosine_unlabeled': _ratio(totals['cos_unlabeled_sum'], pixels),
        'feature_cka': _ratio(totals['cka_sum'], totals['cka_count']),
        'agreement_ratio': _ratio(same, pixels),
        'conflict_ratio': _ratio(conflict, pixels),
        'prediction_similarity': _ratio(same, pixels),
        'same_pixels': int(same),
        'conflict_pixels': int(conflict),
        'same_ratio': _ratio(same, pixels),
        'same_a2b_pixels': int(totals['same_a2b_pixels']),
        'same_b2a_pixels': int(totals['same_b2a_pixels']),
        'conflict_a2b_pixels': int(totals['conflict_a2b_pixels']),
        'conflict_b2a_pixels': int(totals['conflict_b2a_pixels']),
        'a2b_total_ratio': _ratio(routed_a2b, routed),
        'b2a_total_ratio': _ratio(routed_b2a, routed),
        'loss_gmps_same': gmps_same,
        'loss_gmps_conflict': gmps_conflict,
        'loss_gmps_total': gmps_same + gmps_conflict,
        'loss_sup': _ratio(totals['loss_sup_sum'], totals['batches']),
        'loss_total': _ratio(totals['loss_total_sum'], totals['batches']),
        'conflict_pixels_gt': int(totals['conflict_pixels_gt']),
        'gmps_conflict_teacher_correct_pixels': int(
            totals['gmps_conflict_teacher_correct_pixels']
        ),
        'gmps_conflict_teacher_wrong_pixels': int(
            totals['gmps_conflict_teacher_wrong_pixels']
        ),
        'gmps_conflict_selector_accuracy': _ratio(
            totals['gmps_conflict_teacher_correct_pixels'],
            totals['conflict_pixels_gt'],
        ),
        'gmps_conflict_harmful_ratio': _ratio(
            totals['gmps_conflict_teacher_wrong_pixels'],
            totals['conflict_pixels_gt'],
        ),
        'same_pixels_gt': int(totals['same_pixels_gt']),
        'same_both_correct_ratio': _ratio(
            totals['same_both_correct_pixels'], totals['same_pixels_gt']
        ),
        'same_both_wrong_ratio': _ratio(
            totals['same_both_wrong_pixels'], totals['same_pixels_gt']
        ),
    }


def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


def build_datasets(args, cfg):
    """Build the supervised, unlabeled, and validation datasets."""
    manifest_dir = args.manifest_dir or os.path.join(
        args.path_dataset, 'manifests', 'paper_seed_2026'
    )
    use_manifests = os.path.isdir(manifest_dir)

    mean_key, std_key = _normalization_keys(args.input1)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg[mean_key], cfg[std_key])
    train_transform = (
        transforms['val'] if args.input_mode == 'ucmt_views' else transforms['train']
    )

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

    datasets = {
        'train_sup': train_sup,
        'train_sup_unique': train_sup_unique,
        'train_unsup': train_unsup,
        'val': validation,
    }
    return datasets, manifest_dir, use_manifests


def print_data_summary(args, datasets, manifest_dir, use_manifests):
    sup_sample = datasets['train_sup'][0]
    unsup_sample = datasets['train_unsup'][0]
    val_sample = datasets['val'][0]

    print('Data source:', manifest_dir if use_manifests else args.path_dataset)
    print('Input mode:', args.input_mode)
    print(
        'Unique samples: supervised={}, unsupervised={}, validation={}'.format(
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
    if args.input_mode == 'ucmt_views':
        print('Teacher view: clean RGB')
        print('Student1 view: appearance aug x1')
        print('Student2 view: sequential appearance aug x2')
        for name, tensor in (
            ('Teacher clean RGB', unsup_sample['image']),
            ('Student1 aug x1', unsup_sample['image_aug1']),
            ('Student2 aug x2', unsup_sample['image_aug2']),
        ):
            stats_tensor = tensor.float()
            print(
                '{}: shape={} dtype={} min={:.6f} max={:.6f} mean={:.6f} std={:.6f}'.format(
                    name,
                    tuple(tensor.shape),
                    tensor.dtype,
                    stats_tensor.min().item(),
                    stats_tensor.max().item(),
                    stats_tensor.mean().item(),
                    stats_tensor.std().item(),
                )
            )
        print(
            'UCMT view spatial shapes: teacher={} student1={} student2={}'.format(
                tuple(unsup_sample['image'].shape[-2:]),
                tuple(unsup_sample['image_aug1'].shape[-2:]),
                tuple(unsup_sample['image_aug2'].shape[-2:]),
            )
        )
        print(
            'UCMT appearance mean abs diff: teacher-student1={:.6f} student1-student2={:.6f}'.format(
                (unsup_sample['image'] - unsup_sample['image_aug1']).abs().mean().item(),
                (unsup_sample['image_aug1'] - unsup_sample['image_aug2']).abs().mean().item(),
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


def build_dataloaders(datasets, batch_size):
    loader_options = {
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': 8,
    }
    return {
        'train_sup': DataLoader(
            datasets['train_sup'], shuffle=True, **loader_options
        ),
        'train_unsup': DataLoader(
            datasets['train_unsup'], shuffle=True, **loader_options
        ),
        'val': DataLoader(datasets['val'], shuffle=False, **loader_options),
    }


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
    cssml_diagnostics_path = os.path.join(
        trained_models_path, 'cssml_diagnostics.csv'
    )
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
    if args.use_cdfl or args.use_gmps or args.cps_mode == 'cssml':
        print('CSSML diagnostics CSV:', cssml_diagnostics_path)

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
    return logger, metrics_path, diagnostic_logger, cssml_diagnostics_path


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

    optimizer1, scheduler1 = _build_optimizer_and_scheduler(model1, args)
    optimizer2, scheduler2 = _build_optimizer_and_scheduler(model2, args)
    return (
        (model1, model2),
        (optimizer1, optimizer2),
        (scheduler1, scheduler2),
        losses,
    )


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
    mask_list = None
    return_features = _needs_cssml_features(args)
    cssml_totals = (
        _init_cssml_diagnostic_totals()
        if args.use_cdfl or args.use_gmps or args.cps_mode == 'cssml'
        else None
    )

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

        sup_prediction1, sup_feature1 = _forward_student(
            model1, sup_input1, return_features
        )
        sup_prediction2, sup_feature2 = _forward_student(
            model2, sup_input2, return_features
        )

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

        unsup_prediction1, unsup_feature1 = _forward_student(
            model1, unsup_input1, return_features
        )
        unsup_prediction2, unsup_feature2 = _forward_student(
            model2, unsup_input2, return_features
        )

        if cssml_totals is not None:
            _update_cssml_prediction_diagnostics(
                cssml_totals, unsup_prediction1, unsup_prediction2
            )

        loss_cdfl = unsup_prediction1.sum() * 0.0
        cdfl_result = None
        if args.use_cdfl:
            cdfl_result = cdfl_loss(
                sup_feature1,
                sup_feature2,
                unsup_feature1,
                unsup_feature2,
            )
            loss_cdfl = cdfl_result['total']

        gmps_result = None
        if args.cps_mode == 'cssml':
            gmps_result = gmps_loss(
                unsup_prediction1,
                unsup_prediction2,
                unsup_feature1,
                unsup_feature2,
                args.num_classes,
                epoch,
                args.num_epochs,
            )
            cps_losses = {
                '1_to_2': gmps_result['loss'] * 0.5,
                '2_to_1': gmps_result['loss'] * 0.5,
                'total': gmps_result['loss'],
            }
        else:
            cps_losses = compute_cps_losses_by_mode(
                args,
                losses['ce'],
                unsup_prediction1,
                unsup_prediction2,
            )
        unsup_loss = cps_losses['total']

        if cssml_totals is not None:
            pixels = float(unsup_prediction1.shape[0] * unsup_prediction1.shape[2] * unsup_prediction1.shape[3])
            if cdfl_result is not None:
                _update_cssml_cdfl_diagnostics(
                    cssml_totals,
                    cdfl_result,
                    unsup_feature1,
                    unsup_feature2,
                    pixels,
                )
            else:
                cssml_totals['batches'] += 1.0
            if gmps_result is not None:
                _update_cssml_gmps_diagnostics(
                    cssml_totals,
                    gmps_result,
                    unsup_prediction1,
                    unsup_prediction2,
                )
            _update_cssml_gt_diagnostics(
                cssml_totals,
                gmps_result,
                sup_prediction1,
                sup_prediction2,
                sup_mask,
            )

        model1_loss = (
            supervised_loss1
            + unsup_weight * cps_losses['2_to_1']
        )
        model2_loss = (
            supervised_loss2
            + unsup_weight * cps_losses['1_to_2']
        )
        weighted_unsup_loss = unsup_weight * unsup_loss
        weighted_cdfl_loss = args.lambda_cdfl * loss_cdfl
        total_loss = model1_loss + model2_loss + weighted_cdfl_loss

        if cssml_totals is not None:
            cssml_totals['loss_sup_sum'] += (
                supervised_loss1.item() + supervised_loss2.item()
            )
            cssml_totals['loss_total_sum'] += total_loss.item()
        total_loss.backward()

        optimizer1.step()
        optimizer2.step()
        global_step += 1

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
    return loss_totals, (score_list1, score_list2, mask_list), global_step, cssml_totals


# CPS Diagnostic Analysis
def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)[:, 1:2, ...]


# CPS Diagnostic Analysis
def _select_feature_module(model):
    candidate_names = (
        'Conv5',
        'conv4_0',
        'aspp_bridge',
        'bridge',
        'b1_5_4',
        'layers.3',
        'encoder.layers.3',
    )
    modules = dict(model.named_modules())
    for name in candidate_names:
        if name in modules:
            return modules[name]
    leaf_module = None
    for _name, module in model.named_modules():
        if (
            len(list(module.children())) == 0
            and hasattr(module, 'register_forward_hook')
        ):
            leaf_module = module
    return leaf_module


# CPS Diagnostic Analysis
def _linear_cka(feature_a: torch.Tensor, feature_b: torch.Tensor) -> float:
    if feature_a.dim() == 4:
        pooled_hw = (
            min(16, feature_a.shape[-2], feature_b.shape[-2]),
            min(16, feature_a.shape[-1], feature_b.shape[-1]),
        )
        feature_a = torch.nn.functional.adaptive_avg_pool2d(
            feature_a.float(), pooled_hw
        )
        feature_b = torch.nn.functional.adaptive_avg_pool2d(
            feature_b.float(), pooled_hw
        )
        feature_a = feature_a.permute(0, 2, 3, 1).reshape(-1, feature_a.shape[1])
        feature_b = feature_b.permute(0, 2, 3, 1).reshape(-1, feature_b.shape[1])
    else:
        feature_a = feature_a.float().reshape(feature_a.shape[0], -1)
        feature_b = feature_b.float().reshape(feature_b.shape[0], -1)

    sample_count = min(feature_a.shape[0], feature_b.shape[0])
    feature_dim = min(feature_a.shape[1], feature_b.shape[1])
    if sample_count < 2 or feature_dim < 1:
        return float('nan')
    feature_a = feature_a[:sample_count, :feature_dim]
    feature_b = feature_b[:sample_count, :feature_dim]
    feature_a = feature_a - feature_a.mean(dim=0, keepdim=True)
    feature_b = feature_b - feature_b.mean(dim=0, keepdim=True)
    cross = torch.norm(feature_a.T @ feature_b, p='fro').pow(2)
    self_a = torch.norm(feature_a.T @ feature_a, p='fro')
    self_b = torch.norm(feature_b.T @ feature_b, p='fro')
    denom = self_a * self_b
    if denom.item() <= 0:
        return float('nan')
    return float((cross / denom).clamp(0.0, 1.0).item())


# CPS Diagnostic Analysis
def _init_diagnostic_totals():
    return {
        'valid_pixels': 0.0,
        'pixel_disagreement': 0.0,
        'prob_disagreement': 0.0,
        'both_correct': 0.0,
        'a_only_correct': 0.0,
        'b_only_correct': 0.0,
        'both_wrong': 0.0,
        'wrong_agreement': 0.0,
        'disagree_pixels': 0.0,
        'a_correct_disagree': 0.0,
        'b_correct_disagree': 0.0,
        'feature_cka_sum': 0.0,
        'feature_cka_count': 0.0,
    }


# CPS Diagnostic Analysis
def _update_diagnostic_totals(
    totals,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
    feature_a=None,
    feature_b=None,
    ignore_index=-1,
):
    prob_a = _foreground_probability(logits_a).detach()
    prob_b = _foreground_probability(logits_b).detach()
    pred_a = prob_a[:, 0, ...] > 0.5
    pred_b = prob_b[:, 0, ...] > 0.5
    valid = mask.ne(ignore_index)
    gt = mask.float().gt(0.5)
    valid_pixels = float(valid.sum().item())
    if valid_pixels <= 0:
        return

    correct_a = pred_a.eq(gt) & valid
    correct_b = pred_b.eq(gt) & valid
    disagree = pred_a.ne(pred_b) & valid
    both_correct = correct_a & correct_b
    a_only_correct = correct_a & (~correct_b) & valid
    b_only_correct = (~correct_a) & correct_b & valid
    both_wrong = (~correct_a) & (~correct_b) & valid
    wrong_agreement = pred_a.eq(pred_b) & pred_a.ne(gt) & valid

    totals['valid_pixels'] += valid_pixels
    totals['pixel_disagreement'] += float(disagree.sum().item())
    totals['prob_disagreement'] += float(
        torch.abs(prob_a[:, 0, ...] - prob_b[:, 0, ...])[valid].sum().item()
    )
    totals['both_correct'] += float(both_correct.sum().item())
    totals['a_only_correct'] += float(a_only_correct.sum().item())
    totals['b_only_correct'] += float(b_only_correct.sum().item())
    totals['both_wrong'] += float(both_wrong.sum().item())
    totals['wrong_agreement'] += float(wrong_agreement.sum().item())
    totals['disagree_pixels'] += float(disagree.sum().item())
    totals['a_correct_disagree'] += float((correct_a & disagree).sum().item())
    totals['b_correct_disagree'] += float((correct_b & disagree).sum().item())

    if feature_a is not None and feature_b is not None:
        cka = _linear_cka(feature_a.detach(), feature_b.detach())
        if not np.isnan(cka):
            totals['feature_cka_sum'] += cka
            totals['feature_cka_count'] += 1.0


# CPS Diagnostic Analysis
def _finalize_diagnostics(totals, dice_a, dice_b):
    valid_pixels = totals['valid_pixels']
    disagree_pixels = totals['disagree_pixels']
    cka_count = totals['feature_cka_count']
    return {
        'dice_a': dice_a,
        'dice_b': dice_b,
        'pixel_disagreement': totals['pixel_disagreement'] / valid_pixels,
        'prob_disagreement': totals['prob_disagreement'] / valid_pixels,
        'feature_cka': (
            totals['feature_cka_sum'] / cka_count
            if cka_count > 0 else float('nan')
        ),
        'both_correct_ratio': totals['both_correct'] / valid_pixels,
        'a_only_correct_ratio': totals['a_only_correct'] / valid_pixels,
        'b_only_correct_ratio': totals['b_only_correct'] / valid_pixels,
        'both_wrong_ratio': totals['both_wrong'] / valid_pixels,
        'wrong_agreement_ratio': totals['wrong_agreement'] / valid_pixels,
        'a_correct_given_disagreement': (
            totals['a_correct_disagree'] / disagree_pixels
            if disagree_pixels > 0 else float('nan')
        ),
        'b_correct_given_disagreement': (
            totals['b_correct_disagree'] / disagree_pixels
            if disagree_pixels > 0 else float('nan')
        ),
    }


# CPS Diagnostic Analysis
def log_diagnostics(diagnostic_logger, epoch, global_step, diagnostics):
    diagnostic_logger.info(
        '[Diagnostic] epoch=%d global_step=%d '
        'dice_a=%.4f dice_b=%.4f '
        'pixel_disagreement=%.4f prob_disagreement=%.4f '
        'feature_cka=%.4f '
        'both_correct_ratio=%.4f '
        'a_only_correct_ratio=%.4f '
        'b_only_correct_ratio=%.4f '
        'both_wrong_ratio=%.4f '
        'wrong_agreement_ratio=%.4f '
        'a_correct_given_disagreement=%.4f '
        'b_correct_given_disagreement=%.4f',
        epoch,
        global_step,
        diagnostics['dice_a'],
        diagnostics['dice_b'],
        diagnostics['pixel_disagreement'],
        diagnostics['prob_disagreement'],
        diagnostics['feature_cka'],
        diagnostics['both_correct_ratio'],
        diagnostics['a_only_correct_ratio'],
        diagnostics['b_only_correct_ratio'],
        diagnostics['both_wrong_ratio'],
        diagnostics['wrong_agreement_ratio'],
        diagnostics['a_correct_given_disagreement'],
        diagnostics['b_correct_given_disagreement'],
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
            scores1.append(output1)
            scores2.append(output2)
            masks.append(mask)
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


def compute_validation_diversity(validation):
    """Compare validation predictions with ground truth and each other."""
    masks = validation['masks'].long()
    hard = {
        'student1': _hard_predictions(validation['scores1']),
        'student2': _hard_predictions(validation['scores2']),
    }
    diversity = _hard_overlap(hard['student1'], hard['student2'])
    correct1 = hard['student1'].eq(masks)
    correct2 = hard['student2'].eq(masks)
    result = {}
    result.update({
        'student12_disagreement': 1.0 - diversity['dice'],
        'student12_dice_similarity': diversity['dice'],
        'student1_correct_student2_wrong': float((correct1 & ~correct2).float().mean().item()),
        'student2_correct_student1_wrong': float((correct2 & ~correct1).float().mean().item()),
        'student12_both_correct': float((correct1 & correct2).float().mean().item()),
        'student12_both_wrong': float((~correct1 & ~correct2).float().mean().item()),
    })
    result['complementary_error'] = (
        result['student1_correct_student2_wrong']
        + result['student2_correct_student1_wrong']
    )
    return result


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
        '| best_jaccard1=%.6f | best_jaccard2=%.6f '
        '| best_jaccard=%.6f | time=%.2fs',
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


def train_model(
    args,
    cfg,
    datasets,
    use_manifests,
):
    """Run FDCL training, validation, logging, and checkpoint selection."""
    dataloaders = build_dataloaders(datasets, args.batch_size)
    num_batches = {
        name: len(dataloader) for name, dataloader in dataloaders.items()
    }
    trained_models_path, _ = build_output_paths(
        args, datasets, use_manifests
    )
    logger, metrics_path, diagnostic_logger, cssml_diagnostics_path = create_run_outputs(
        args, trained_models_path
    )
    (
        models,
        optimizers,
        schedulers,
        losses,
    ) = build_training_components(args, cfg)

    model1, model2 = models
    best_val_eval_lists = (
        [0 for _ in range(4)],
        [0 for _ in range(4)],
    )
    best_model_states = (None, None)
    best_epochs = (0, 0)
    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)
    training_start = time.time()
    global_step = 0

    cssml_file = None
    cssml_writer = None
    if args.use_cdfl or args.use_gmps or args.cps_mode == 'cssml':
        cssml_file = open(
            cssml_diagnostics_path, 'w', newline='', encoding='utf-8'
        )
        cssml_writer = csv.DictWriter(
            cssml_file, fieldnames=CSSML_DIAGNOSTIC_FIELDS
        )
        cssml_writer.writeheader()

    try:
        with open(metrics_path, 'w', newline='', encoding='utf-8') as metrics_file:
            metrics_writer = csv.DictWriter(
                metrics_file, fieldnames=METRIC_FIELDS
            )
            metrics_writer.writeheader()

            for epoch in range(args.num_epochs):
                epoch_start = time.time()
                loss_totals, train_predictions, global_step, cssml_totals = train_one_epoch(
                    args,
                    epoch,
                    global_step,
                    models,
                    optimizers,
                    losses,
                    dataloaders,
                    metrics_writer,
                    metrics_file,
                )
                if cssml_writer is not None:
                    cssml_writer.writerow(
                        _finalize_cssml_diagnostics(epoch + 1, cssml_totals)
                    )
                    cssml_file.flush()
                schedulers[0].step()
                schedulers[1].step()

                print('=' * print_num)
                print(
                    '| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(
                        print_num_minus, ' '
                    ),
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

                validation = validate(models, losses, dataloaders['val'])
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
                )
                current_epoch = epoch + 1
                if (
                    args.milestone_interval > 0
                    and current_epoch % args.milestone_interval == 0
                ):
                    milestone_path = save_training_milestone(
                        model1,
                        model2,
                        trained_models_path,
                        current_epoch,
                        metadata={
                            'global_step': (
                                global_step
                            ),
                            'dataset_name': args.dataset_name,
                            'labeled_ratio': args.labeled_ratio,
                            'seed': args.sd,
                            'input_mode': args.input_mode,
                            'network': args.network,
                            'best_val_metrics': [
                                list(values)
                                for values in best_val_eval_lists
                            ],
                            'best_epochs': list(best_epochs),
                            'args': vars(args).copy(),
                        },
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
    finally:
        if cssml_file is not None:
            cssml_file.close()

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
    parser.add_argument('-pd', '--path_dataset', default='datasets/GLaS')
    parser.add_argument(
        '--dataset_name',
        default='GlaS',
        choices=['GlaS', 'CRAG', 'MoNuSeg'],
    )
    parser.add_argument('--input1', default='image')
    parser.add_argument(
        '--input_mode',
        dest='input_mode',
        default='rgb_rgb',
        choices=['rgb_rgb', 'rgb_stain', 'rgb_hed', 'ucmt_views'],
        help='rgb_rgb uses original RGB for both students; ' \
        'rgb_stain uses stain RGB for student2; ' \
        'rgb_hed uses HED for student2;' \
        'ucmt_views uses clean/Aug1/sequential Aug2 RGB',
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
        '--cps-mode',
        dest='cps_mode',
        default='vanilla',
        choices=['vanilla', 'relative_reliable', 'cssml'],
        help='Mutual pseudo-supervision mode.',
    )
    parser.add_argument(
        '--use-cdfl',
        dest='use_cdfl',
        action='store_true',
        help='Enable CSSML Conflict-aware Distinct Feature Learning.',
    )
    parser.add_argument(
        '--use-gmps',
        dest='use_gmps',
        action='store_true',
        help='Enable CSSML Geometry-aware Mutual Pseudo Supervision.',
    )
    parser.add_argument(
        '--lambda-cdfl',
        dest='lambda_cdfl',
        default=0.1,
        type=float,
        help='Weight for CSSML CDFL loss.',
    )
    parser.add_argument('--loss', default='dice')
    parser.add_argument(
        '--consistency',
        default=0.1,
        type=float,
        help='CPS consistency weight, following SSL4MIS CPS.',
    )
    parser.add_argument(
        '--consistency_rampup',
        default=200.0,
        type=float,
        help='CPS consistency ramp-up length, following SSL4MIS CPS.',
    )
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay exponent')
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='unet')

    args = parser.parse_args()
    if args.milestone_interval < 0:
        parser.error('--milestone_interval must be non-negative')
    if args.best_save_interval < 0:
        parser.error('--best_save_interval must be non-negative')
    if args.lambda_cdfl < 0:
        parser.error('--lambda-cdfl must be non-negative')
    if args.cps_mode == 'cssml' and not args.use_gmps:
        parser.error('--cps-mode cssml requires --use-gmps')
    if args.use_gmps and args.cps_mode != 'cssml':
        parser.error('--use-gmps requires --cps-mode cssml')
    if args.path_trained_models is None:
        args.path_trained_models = 'checkpoints/semi_fdcl_paper_{}'.format(
            args.labeled_ratio
        )
    if args.path_seg_results is None:
        args.path_seg_results = 'seg_pred/semi_fdcl_paper_{}'.format(
            args.labeled_ratio
        )
    return args


def main():
    args = parse_args()
    init_seeds(args.sd)
    cfg = dataset_cfg(args.dataset_name)
    args.num_classes = cfg['NUM_CLASSES']
    datasets, manifest_dir, use_manifests = build_datasets(args, cfg)

    if args.check_data:
        print_data_summary(args, datasets, manifest_dir, use_manifests)
        return
    print_data_summary(args, datasets, manifest_dir, use_manifests)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'FDCL training requires a CUDA GPU; '
            'use --check_data to validate input only.'
        )

    train_model(args, cfg, datasets, use_manifests)


if __name__ == '__main__':
    main()
