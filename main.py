import argparse
import csv
import logging
import os
import random
import re
import time
from typing import Dict, Mapping, Tuple
from warnings import simplefilter

from utils.ramps import sigmoid_rampup
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
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
from loss.loss_function import DiceLoss
from models.getnetwork import get_network
from utils.ph_mask_only import (
    accumulate_ph_mask_state,
    build_ph_conflict_mask,
    finalize_ph_mask_epoch_state,
    new_ph_mask_epoch_state,
    update_ph_mask_loss_state,
)
from utils.ph_diagnostics import (
    PH_DIAGNOSTIC_FIELDS,
    build_ph_diagnostics_row,
)
from utils.ph_directional_cps import (
    compute_ph_directional_cps_losses,
    finalize_ph_directional_epoch_state,
    new_ph_directional_epoch_state,
    save_ph_match_examples,
    update_ph_directional_loss_state,
    update_ph_directional_vanilla_state,
)
from utils.structural_diagnostics import StructuralDiagnostics
from utils.structural_projection import (
    STRUCTURAL_PROJECTION_FIELDS,
    compute_structural_projection,
    compute_structural_projection_cps_losses,
    finalize_structural_projection_epoch_state,
    has_compiled_mincut_backend,
    new_structural_projection_epoch_state,
    update_structural_projection_epoch_state,
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

PH_MASK_ONLY_FIELDS = [
    'epoch',
    'ph_enabled',
    'ph_conflict_events',
    'merge_conflict_events',
    'split_conflict_events',
    'ph_conflict_pixels',
    'total_unlabeled_pixels',
    'ph_conflict_pixel_ratio',
    'ph_mask_efficiency',
    'safe_cps_pixels',
    'safe_cps_pixel_ratio',
    'batches_with_ph_conflict',
    'batches_without_ph_conflict',
    'mean_conflict_roi_size',
    'max_conflict_roi_size',
    'loss_cps_original_estimate',
    'loss_cps_safe',
    'masked_cps_loss_fraction',
    'val_dice_a',
    'val_dice_b',
    'val_jaccard_a',
    'val_jaccard_b',
    'pixel_disagreement',
    'both_struct_correct_ratio',
    'a_only_struct_correct_ratio',
    'b_only_struct_correct_ratio',
    'both_struct_wrong_ratio',
    'structural_complementary_ratio',
    'merge_error_a',
    'merge_error_b',
    'merge_error_total',
    'split_error_a',
    'split_error_b',
    'split_error_total',
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


def compute_masked_cps_losses(
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
    safe_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pseudo1 = prediction1.detach().argmax(dim=1)
    pseudo2 = prediction2.detach().argmax(dim=1)
    safe = safe_mask.detach().float()
    denom = safe.sum().clamp_min(1.0)
    loss_2_to_1_map = F.cross_entropy(
        prediction1, pseudo2, reduction='none'
    )
    loss_1_to_2_map = F.cross_entropy(
        prediction2, pseudo1, reduction='none'
    )
    loss_2_to_1 = 0.5 * (loss_2_to_1_map * safe).sum() / denom
    loss_1_to_2 = 0.5 * (loss_1_to_2_map * safe).sum() / denom
    original_estimate = 0.5 * (
        loss_2_to_1_map.mean() + loss_1_to_2_map.mean()
    )
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': loss_1_to_2 + loss_2_to_1,
        'original_estimate': original_estimate,
    }


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
    if getattr(args, 'ph_mask_only', False):
        experiment_name += '-ph_mask_only-start={}-d={}'.format(
            args.ph_start_epoch,
            args.ph_mask_dilation,
        )
    if getattr(args, 'ph_directional_cps', False):
        experiment_name += '-ph_directional_cps-start={}-d={}-mr={}-um={}-det={}'.format(
            args.ph_start_epoch,
            args.ph_mask_dilation,
            args.ph_match_radius,
            args.ph_unmatched_mode,
            args.ph_conflict_detector,
        )
        if getattr(args, 'ph_relation_backend', 'cc') != 'cc':
            experiment_name += '-rel={}'.format(args.ph_relation_backend)
        if getattr(args, 'ph_presence_conflict_mode', 'strict') != 'strict':
            experiment_name += '-presence={}'.format(args.ph_presence_conflict_mode)
    if getattr(args, 'use_structural_projection', False):
        experiment_name += (
            '-struct_projection-start={}-tau={}-margin={}-max={}-kappa={}'
        ).format(
            args.struct_start_epoch,
            args.struct_tau,
            args.struct_graph_margin,
            args.struct_graph_max_pixels,
            args.struct_arbitration_threshold,
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
    ph_mask_state=None,
    structural_projection_state=None,
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
        
        ph_mask_active = (
            args.ph_mask_only
            and ph_mask_state is not None
            and (epoch + 1) >= args.ph_start_epoch
        )
        ph_directional_active = (
            args.ph_directional_cps
            and ph_mask_state is not None
            and (epoch + 1) >= args.ph_start_epoch
        )
        structural_projection_active = (
            args.use_structural_projection
            and structural_projection_state is not None
            and (epoch + 1) >= args.struct_start_epoch
        )
        if structural_projection_active:
            with torch.no_grad():
                prob_a = torch.softmax(unsup_prediction1.detach(), dim=1)[:, 1]
                prob_b = torch.softmax(unsup_prediction2.detach(), dim=1)[:, 1]
                projection = compute_structural_projection(
                    prob_a.cpu().numpy(),
                    prob_b.cpu().numpy(),
                    tau=args.struct_tau,
                    graph_margin=args.struct_graph_margin,
                    graph_max_pixels=args.struct_graph_max_pixels,
                    arbitration_threshold=args.struct_arbitration_threshold,
                    use_ph_corridor=args.struct_use_ph_corridor,
                    use_mincut_fallback=args.struct_use_mincut_fallback,
                    connectivity=args.struct_connectivity,
                    matching=args.struct_matching,
                )
            update_structural_projection_epoch_state(
                structural_projection_state, projection['stats']
            )
            has_structural_pixels = bool(
                projection['struct_mask_a'].any()
                or projection['struct_mask_b'].any()
                or projection['abstain_mask_a'].any()
                or projection['abstain_mask_b'].any()
            )
            if not has_structural_pixels:
                # No accepted relation and no abstention: preserve vanilla CPS exactly.
                cps_losses = compute_cps_losses(
                    losses['ce'], unsup_prediction1, unsup_prediction2
                )
            else:
                cps_losses = compute_structural_projection_cps_losses(
                    unsup_prediction1, unsup_prediction2, projection
                )
        elif ph_directional_active:
            cps_losses = compute_ph_directional_cps_losses(
                unsup_prediction1,
                unsup_prediction2,
                dilation=args.ph_mask_dilation,
                match_radius=args.ph_match_radius,
                unmatched_mode=args.ph_unmatched_mode,
                detector_mode=args.ph_conflict_detector,
                debug_timing=args.ph_debug_timing,
                batch_index=batch_index,
                correspondence_mode=args.ph_correspondence_mode,
                relation_backend=args.ph_relation_backend,
                presence_conflict_mode=args.ph_presence_conflict_mode,
            )
            update_ph_directional_loss_state(ph_mask_state, cps_losses)
        elif ph_mask_active:
            with torch.no_grad():
                prob_a = torch.softmax(unsup_prediction1.detach(), dim=1)[:, 1]
                prob_b = torch.softmax(unsup_prediction2.detach(), dim=1)[:, 1]
                ph_mask_np, ph_stats = build_ph_conflict_mask(
                    prob_a.cpu().numpy(),
                    prob_b.cpu().numpy(),
                    dilation=args.ph_mask_dilation,
                )
                accumulate_ph_mask_state(ph_mask_state, ph_mask_np, ph_stats)
                safe_mask = torch.from_numpy(~ph_mask_np).to(
                    unsup_prediction1.device,
                    non_blocking=True,
                )
            cps_losses = compute_masked_cps_losses(
                unsup_prediction1,
                unsup_prediction2,
                safe_mask,
            )
            update_ph_mask_loss_state(
                ph_mask_state,
                cps_losses['original_estimate'].detach().item(),
                cps_losses['total'].detach().item(),
            )
        else:
            cps_losses = compute_cps_losses(
                losses['ce'],
                unsup_prediction1,
                unsup_prediction2,
            )
            if ph_mask_state is not None and args.ph_mask_only:
                empty_mask = np.zeros(
                    (
                        unsup_prediction1.shape[0],
                        unsup_prediction1.shape[2],
                        unsup_prediction1.shape[3],
                    ),
                    dtype=bool,
                )
                accumulate_ph_mask_state(
                    ph_mask_state,
                    empty_mask,
                    {'events': 0, 'merge_events': 0, 'split_events': 0, 'roi_sizes': []},
                )
                update_ph_mask_loss_state(
                    ph_mask_state,
                    cps_losses['total'].detach().item(),
                    cps_losses['total'].detach().item(),
                )
            elif ph_mask_state is not None and args.ph_directional_cps:
                update_ph_directional_vanilla_state(
                    ph_mask_state,
                    cps_losses,
                    unsup_prediction1,
                )
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
    return loss_totals, (score_list1, score_list2, mask_list), global_step


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
def validate(
    models,
    losses,
    dataloader,
    structural_diagnostics=None,
    epoch_state=None,
    collect_inputs=False,
):
    model1, model2 = models
    model1.eval()
    model2.eval()

    loss1 = 0.0
    loss2 = 0.0
    scores1 = []
    scores2 = []
    masks = []
    inputs1 = []
    inputs2 = []
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
            if structural_diagnostics is not None and epoch_state is not None:
                structural_diagnostics.update_batch(
                    epoch_state,
                    output1,
                    output2,
                    mask,
                    batch['ID'],
                    input1,
                    input2,
                )
            scores1.append(output1)
            scores2.append(output2)
            masks.append(mask)
            if collect_inputs:
                inputs1.append(input1.detach().cpu())
                inputs2.append(input2.detach().cpu())
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

    result = {
        'loss1': loss1,
        'loss2': loss2,
        'scores1': scores1_tensor,
        'scores2': scores2_tensor,
        'masks': torch.cat(masks, dim=0),
        'names': np.asarray(names),
        'diagnostic_totals': diagnostic_totals,
    }
    if collect_inputs:
        result['inputs1'] = torch.cat(inputs1, dim=0)
        result['inputs2'] = torch.cat(inputs2, dim=0)
    return result


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


def _write_ph_mask_only_diagnostics(writer, ph_state, val_eval1, val_eval2, diagnostics, structural_summary):
    row = finalize_ph_mask_epoch_state(ph_state)
    row.update({
        'val_dice_a': val_eval1[2],
        'val_dice_b': val_eval2[2],
        'val_jaccard_a': val_eval1[1],
        'val_jaccard_b': val_eval2[1],
        'pixel_disagreement': diagnostics['pixel_disagreement'],
    })
    if structural_summary is None:
        row.update({
            'both_struct_correct_ratio': 0.0,
            'a_only_struct_correct_ratio': 0.0,
            'b_only_struct_correct_ratio': 0.0,
            'both_struct_wrong_ratio': 0.0,
            'structural_complementary_ratio': 0.0,
            'merge_error_a': 0,
            'merge_error_b': 0,
            'merge_error_total': 0,
            'split_error_a': 0,
            'split_error_b': 0,
            'split_error_total': 0,
        })
    else:
        merge_a = int(structural_summary['merge_error_a'])
        merge_b = int(structural_summary['merge_error_b'])
        split_a = int(structural_summary['split_error_a'])
        split_b = int(structural_summary['split_error_b'])
        row.update({
            'both_struct_correct_ratio': structural_summary['both_struct_correct_ratio'],
            'a_only_struct_correct_ratio': structural_summary['a_only_struct_correct_ratio'],
            'b_only_struct_correct_ratio': structural_summary['b_only_struct_correct_ratio'],
            'both_struct_wrong_ratio': structural_summary['both_struct_wrong_ratio'],
            'structural_complementary_ratio': structural_summary['structural_complementary_ratio'],
            'merge_error_a': merge_a,
            'merge_error_b': merge_b,
            'merge_error_total': merge_a + merge_b,
            'split_error_a': split_a,
            'split_error_b': split_b,
            'split_error_total': split_a + split_b,
        })
    writer.writerow(row)


def _write_ph_directional_diagnostics(writer, ph_state, val_eval1, val_eval2, structural_summary):
    row = finalize_ph_directional_epoch_state(ph_state)
    row.update({
        'val_dice_a': val_eval1[2],
        'val_dice_b': val_eval2[2],
        'val_jaccard_a': val_eval1[1],
        'val_jaccard_b': val_eval2[1],
    })
    if structural_summary is None:
        row.update({
            'both_struct_correct_ratio': 0.0,
            'a_only_struct_correct_ratio': 0.0,
            'b_only_struct_correct_ratio': 0.0,
            'both_struct_wrong_ratio': 0.0,
            'structural_complementary_ratio': 0.0,
            'merge_error_a': 0,
            'merge_error_b': 0,
            'split_error_a': 0,
            'split_error_b': 0,
        })
    else:
        row.update({
            'both_struct_correct_ratio': structural_summary['both_struct_correct_ratio'],
            'a_only_struct_correct_ratio': structural_summary['a_only_struct_correct_ratio'],
            'b_only_struct_correct_ratio': structural_summary['b_only_struct_correct_ratio'],
            'both_struct_wrong_ratio': structural_summary['both_struct_wrong_ratio'],
            'structural_complementary_ratio': structural_summary['structural_complementary_ratio'],
            'merge_error_a': int(structural_summary['merge_error_a']),
            'merge_error_b': int(structural_summary['merge_error_b']),
            'split_error_a': int(structural_summary['split_error_a']),
            'split_error_b': int(structural_summary['split_error_b']),
        })
    writer.writerow(row)


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
    logger, metrics_path, diagnostic_logger = create_run_outputs(
        args, trained_models_path
    )
    (
        models,
        optimizers,
        schedulers,
        losses,
    ) = build_training_components(args, cfg)
    structural_diagnostics = None
    ph_diagnostics_path = os.path.join(trained_models_path, 'ph_diagnostics.csv')
    structural_projection_path = os.path.join(
        trained_models_path, 'structural_projection_metrics.csv'
    )
    use_ph_diagnostics_csv = (
        args.ph_directional_cps
        and args.ph_conflict_detector == 'persistent_relation'
        and args.ph_correspondence_mode != 'event_projection'
    )
    if use_ph_diagnostics_csv:
        print('PH diagnostics CSV:', ph_diagnostics_path)
    use_structural_projection_csv = bool(args.use_structural_projection)
    if use_structural_projection_csv:
        print('Structural projection metrics CSV:', structural_projection_path)
    ph_vis_state = {'sample_ids': None}

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

    with open(metrics_path, 'w', newline='', encoding='utf-8') as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=METRIC_FIELDS
        )
        metrics_writer.writeheader()
        ph_diagnostics_file = None
        ph_diagnostics_writer = None
        if use_ph_diagnostics_csv:
            ph_diagnostics_file = open(
                ph_diagnostics_path, 'w', newline='', encoding='utf-8'
            )
            ph_diagnostics_writer = csv.DictWriter(
                ph_diagnostics_file, fieldnames=PH_DIAGNOSTIC_FIELDS
            )
            ph_diagnostics_writer.writeheader()
        structural_projection_file = None
        structural_projection_writer = None
        if use_structural_projection_csv:
            structural_projection_file = open(
                structural_projection_path, 'w', newline='', encoding='utf-8'
            )
            structural_projection_writer = csv.DictWriter(
                structural_projection_file,
                fieldnames=STRUCTURAL_PROJECTION_FIELDS,
            )
            structural_projection_writer.writeheader()

        for epoch in range(args.num_epochs):
            epoch_start = time.time()
            if args.ph_mask_only:
                ph_mask_state = new_ph_mask_epoch_state(
                    epoch + 1,
                    (epoch + 1) >= args.ph_start_epoch,
                )
            elif args.ph_directional_cps:
                ph_mask_state = new_ph_directional_epoch_state(
                    epoch + 1,
                    (epoch + 1) >= args.ph_start_epoch,
                )
            else:
                ph_mask_state = None
            structural_projection_state = (
                new_structural_projection_epoch_state(
                    epoch + 1,
                    (epoch + 1) >= args.struct_start_epoch,
                )
                if args.use_structural_projection else None
            )
            loss_totals, train_predictions, global_step = train_one_epoch(
                args,
                epoch,
                global_step,
                models,
                optimizers,
                losses,
                dataloaders,
                metrics_writer,
                metrics_file,
                ph_mask_state,
                structural_projection_state,
            )
            if structural_projection_writer is not None:
                structural_projection_writer.writerow(
                    finalize_structural_projection_epoch_state(
                        structural_projection_state
                    )
                )
                structural_projection_file.flush()
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

            struct_epoch_state = (
                structural_diagnostics.new_epoch(epoch + 1)
                if structural_diagnostics is not None else None
            )
            ph_validation_interval = args.ph_validation_diagnostics_interval
            if ph_validation_interval is None:
                ph_validation_interval = args.best_save_interval
            run_ph_validation_diagnostics = (
                ph_validation_interval > 0
                and (epoch + 1) % ph_validation_interval == 0
            )
            validation = validate(
                models,
                losses,
                dataloaders['val'],
                structural_diagnostics,
                struct_epoch_state,
                collect_inputs=(
                    args.ph_directional_cps
                    and (
                        args.save_ph_match_examples
                        or args.ph_save_conflict_vis
                    )
                    and run_ph_validation_diagnostics
                ),
            )
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
            structural_summary = None
            if structural_diagnostics is not None:
                struct_epoch_state['pixel_complementary_ratio'] = (
                    diagnostics['a_only_correct_ratio']
                    + diagnostics['b_only_correct_ratio']
                )
                structural_summary = structural_diagnostics.finalize_epoch(
                    struct_epoch_state
                )
            if (
                ph_diagnostics_writer is not None
                and ph_mask_state is not None
                and ph_mask_state['ph_enabled']
                and run_ph_validation_diagnostics
            ):
                ph_diagnostics_writer.writerow(
                    build_ph_diagnostics_row(
                        epoch + 1,
                        validation['names'],
                        validation['scores1'],
                        validation['scores2'],
                        validation['masks'],
                        dilation=args.ph_mask_dilation,
                        unmatched_mode=args.ph_unmatched_mode,
                        correspondence_mode=args.ph_correspondence_mode,
                        relation_backend=args.ph_relation_backend,
                        presence_conflict_mode=args.ph_presence_conflict_mode,
                        output_dir=trained_models_path,
                        save_conflict_vis=args.ph_save_conflict_vis,
                        vis_interval=args.ph_vis_interval,
                        vis_num_samples=args.ph_vis_num_samples,
                        vis_state=ph_vis_state,
                        images=validation.get('inputs1'),
                    )
                )
                ph_diagnostics_file.flush()
                if args.save_ph_match_examples:
                    save_ph_match_examples(
                        trained_models_path,
                        epoch + 1,
                        validation['scores1'],
                        validation['scores2'],
                        validation['masks'],
                        validation.get('inputs1'),
                        validation.get('inputs2'),
                        dilation=args.ph_mask_dilation,
                        match_radius=args.ph_match_radius,
                        unmatched_mode=args.ph_unmatched_mode,
                        min_area=args.struct_min_area,
                        match_iou=args.struct_match_iou,
                        detector_mode=args.ph_conflict_detector,
                        correspondence_mode=args.ph_correspondence_mode,
                        relation_backend=args.ph_relation_backend,
                        presence_conflict_mode=args.ph_presence_conflict_mode,
                    )
            if args.ph_directional_cps:
                ph_timing = finalize_ph_directional_epoch_state(ph_mask_state)
                logger.info(
                    'PH timing | epoch=%d | ph_extract=%.4fs | support=%.4fs | support_overlap=%.4fs | entity_match=%.4fs | relation=%.4fs | roi=%.4fs | total=%.4fs',
                    epoch + 1,
                    ph_timing['ph_extract_time'],
                    ph_timing['support_build_time'],
                    ph_timing['support_overlap_time'],
                    ph_timing['entity_match_time'],
                    ph_timing['relation_time'],
                    ph_timing['roi_time'],
                    ph_timing['total_ph_time'],
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
        if ph_diagnostics_file is not None:
            ph_diagnostics_file.close()
        if structural_projection_file is not None:
            structural_projection_file.close()

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


def _unwrap_state_dict(checkpoint, checkpoint_path):
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported checkpoint content in {}'.format(checkpoint_path))
    return checkpoint


def _load_state_dict(model, state_dict, checkpoint_path):
    state_dict = _unwrap_state_dict(state_dict, checkpoint_path)
    state_dict = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def _resolve_struct_checkpoint_pair(checkpoint_arg, checkpoint_a, checkpoint_b, epoch):
    if checkpoint_a and checkpoint_b:
        return checkpoint_a, checkpoint_b, None
    checkpoint_path = os.path.abspath(checkpoint_arg)
    if os.path.isdir(checkpoint_path):
        base_dir = checkpoint_path
        if os.path.basename(base_dir) in ('milestones', 'periodic_best'):
            base_dir = os.path.dirname(base_dir)
        if epoch > 0:
            milestone = os.path.join(base_dir, 'milestones', 'epoch_{:03d}.pth'.format(epoch))
            if os.path.isfile(milestone):
                return milestone, None, epoch
            periodic_dir = os.path.join(base_dir, 'periodic_best')
            paths_a = sorted(
                [
                    os.path.join(periodic_dir, name)
                    for name in os.listdir(periodic_dir)
                    if re.match(r'epoch_{:04d}_best_A_Jc_.*\.pth$'.format(epoch), name)
                ]
            ) if os.path.isdir(periodic_dir) else []
            paths_b = sorted(
                [
                    os.path.join(periodic_dir, name)
                    for name in os.listdir(periodic_dir)
                    if re.match(r'epoch_{:04d}_best_S_Jc_.*\.pth$'.format(epoch), name)
                ]
            ) if os.path.isdir(periodic_dir) else []
            if paths_a and paths_b:
                return paths_a[-1], paths_b[-1], epoch
            recursive_milestones = []
            recursive_pairs = []
            for root, _dirs, files in os.walk(base_dir):
                milestone_name = 'epoch_{:03d}.pth'.format(epoch)
                if os.path.basename(root) == 'milestones' and milestone_name in files:
                    recursive_milestones.append(os.path.join(root, milestone_name))
                if os.path.basename(root) == 'periodic_best':
                    root_a = sorted(
                        [
                            os.path.join(root, name)
                            for name in files
                            if re.match(r'epoch_{:04d}_best_A_Jc_.*\.pth$'.format(epoch), name)
                        ]
                    )
                    root_b = sorted(
                        [
                            os.path.join(root, name)
                            for name in files
                            if re.match(r'epoch_{:04d}_best_S_Jc_.*\.pth$'.format(epoch), name)
                        ]
                    )
                    if root_a and root_b:
                        recursive_pairs.append((root_a[-1], root_b[-1]))
            if len(recursive_milestones) == 1:
                return recursive_milestones[0], None, epoch
            if len(recursive_pairs) == 1:
                return recursive_pairs[0][0], recursive_pairs[0][1], epoch
            raise FileNotFoundError(
                'No milestone or periodic A/S checkpoint found for epoch {}'.format(epoch)
            )
        best_a = sorted(
            [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if re.match(r'best_result1_Jc_.*\.pth$', name)
            ]
        )
        best_b = sorted(
            [
                os.path.join(base_dir, name)
                for name in os.listdir(base_dir)
                if re.match(r'best_result2_Jc_.*\.pth$', name)
            ]
        )
        if best_a and best_b:
            return best_a[-1], best_b[-1], 0
        recursive_best_pairs = []
        for root, _dirs, files in os.walk(base_dir):
            root_best_a = sorted(
                [
                    os.path.join(root, name)
                    for name in files
                    if re.match(r'best_result1_Jc_.*\.pth$', name)
                ]
            )
            root_best_b = sorted(
                [
                    os.path.join(root, name)
                    for name in files
                    if re.match(r'best_result2_Jc_.*\.pth$', name)
                ]
            )
            if root_best_a and root_best_b:
                recursive_best_pairs.append((root_best_a[-1], root_best_b[-1]))
        if len(recursive_best_pairs) == 1:
            return recursive_best_pairs[0][0], recursive_best_pairs[0][1], 0
        raise FileNotFoundError(
            'Directory must contain best_result1/best_result2 checkpoints, '
            'or pass --struct-analysis-epoch for milestones/periodic_best: {}'.format(
                base_dir
            )
        )
    return checkpoint_path, None, epoch


def _load_struct_models(args, cfg):
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda()
    model2 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).cuda()
    path_a, path_b, resolved_epoch = _resolve_struct_checkpoint_pair(
        args.struct_analysis_checkpoint,
        args.struct_analysis_checkpoint_a,
        args.struct_analysis_checkpoint_b,
        args.struct_analysis_epoch,
    )
    checkpoint = torch.load(path_a, map_location='cpu', weights_only=False)
    if path_b is None:
        if (
            isinstance(checkpoint, dict)
            and 'appearance_state_dict' in checkpoint
            and 'structure_state_dict' in checkpoint
        ):
            _load_state_dict(model1, checkpoint['appearance_state_dict'], path_a)
            _load_state_dict(model2, checkpoint['structure_state_dict'], path_a)
        else:
            raise ValueError(
                'Single-checkpoint structural analysis expects a milestone '
                'with appearance_state_dict and structure_state_dict: {}'.format(path_a)
            )
    else:
        checkpoint_b = torch.load(path_b, map_location='cpu', weights_only=False)
        _load_state_dict(model1, checkpoint, path_a)
        _load_state_dict(model2, checkpoint_b, path_b)
    return (model1, model2), (path_a, path_b), (resolved_epoch or 0)


def run_structural_analysis(args, cfg, datasets):
    if not args.enable_struct_diagnostics:
        raise ValueError(
            '--struct-analysis-checkpoint requires --enable-struct-diagnostics'
        )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for structural checkpoint analysis.')
    models, checkpoint_paths, resolved_epoch = _load_struct_models(args, cfg)
    dataloader = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )
    losses = {
        'ce': CrossEntropyLoss().cuda(),
        'dice': DiceLoss().cuda(),
    }
    if args.struct_output_dir is not None:
        output_dir = args.struct_output_dir
    elif os.path.isdir(args.struct_analysis_checkpoint):
        output_dir = os.path.abspath(args.struct_analysis_checkpoint)
        if os.path.basename(output_dir) in ('milestones', 'periodic_best'):
            output_dir = os.path.dirname(output_dir)
    else:
        output_dir = os.path.dirname(checkpoint_paths[0])
    diagnostics = StructuralDiagnostics(
        output_dir,
        min_area=args.struct_min_area,
        match_iou=args.struct_match_iou,
        boundary_width=args.struct_boundary_width,
        save_examples=args.save_struct_examples,
    )
    epoch_for_log = resolved_epoch if resolved_epoch > 0 else 0
    epoch_state = diagnostics.new_epoch(epoch_for_log)
    validation = validate(models, losses, dataloader, diagnostics, epoch_state)
    val_eval1, val_eval2, _, _ = print_val_eval(
        cfg['NUM_CLASSES'],
        validation['scores1'],
        validation['scores2'],
        validation['masks'],
        38,
    )
    pixel_diagnostics = _finalize_diagnostics(
        validation['diagnostic_totals'],
        val_eval1[2],
        val_eval2[2],
    )
    epoch_state['pixel_complementary_ratio'] = (
        pixel_diagnostics['a_only_correct_ratio']
        + pixel_diagnostics['b_only_correct_ratio']
    )
    summary = diagnostics.finalize_epoch(epoch_state)
    print('Structural diagnostics saved to: {}'.format(output_dir))
    print(
        'Structural complementary ratio: {:.6f}'.format(
            summary['structural_complementary_ratio']
        )
    )


def _infer_input_mode_from_path(path):
    match = re.search(r'input_mode=([A-Za-z0-9_]+)', str(path))
    if match:
        return match.group(1)
    return None


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
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay exponent')
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='unet')
    parser.add_argument(
        '--enable-struct-diagnostics',
        action='store_true',
        help='enable CC-based validation structural diagnostics',
    )
    parser.add_argument(
        '--struct-min-area',
        default=20,
        type=int,
        help='minimum CC area in pixels for structural diagnostics only',
    )
    parser.add_argument(
        '--struct-match-iou',
        default=0.1,
        type=float,
        help='minimum component IoU for structural matching/overlap diagnostics',
    )
    parser.add_argument(
        '--struct-boundary-width',
        default=3,
        type=int,
        help='morphological GT boundary width for structural diagnostics',
    )
    parser.add_argument(
        '--save-struct-examples',
        action='store_true',
        help='save up to five structural example mosaics per category',
    )
    parser.add_argument(
        '--ph-mask-only',
        action='store_true',
        help='E1 ablation: abstain vanilla bidirectional CPS inside PH-H0 topology conflict masks',
    )
    parser.add_argument(
        '--ph-directional-cps',
        action='store_true',
        help='E2 ablation: PH-guided directional CPS inside topology conflict ROIs',
    )
    parser.add_argument(
        '--use-structural-projection',
        action='store_true',
        help='enable persistent relation-guided receiver-side structural projection CPS',
    )
    parser.add_argument(
        '--struct-start-epoch',
        default=20,
        type=int,
        help='first epoch that enables structural projection; earlier epochs use vanilla CPS',
    )
    parser.add_argument(
        '--struct-tau',
        default=0.5,
        type=float,
        help='task operating threshold for binary relation validation (not PH extraction)',
    )
    parser.add_argument(
        '--struct-matching',
        choices=('birth_nearest',),
        default='birth_nearest',
        help='persistent branch identity matching method',
    )
    parser.add_argument(
        '--struct-graph-margin',
        default=2,
        type=int,
        help='small dilation margin for relation-local structural graph domains',
    )
    parser.add_argument(
        '--struct-graph-max-pixels',
        default=4096,
        type=int,
        help='maximum relation-local domain size before abstention',
    )
    parser.add_argument(
        '--struct-arbitration-threshold',
        default=0.2,
        type=float,
        help='minimum normalized structural compatibility energy gap',
    )
    parser.add_argument(
        '--struct-use-ph-corridor',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='try the PH merge corridor before local cut fallback',
    )
    parser.add_argument(
        '--struct-use-mincut-fallback',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='use optional compiled local mincut when the PH corridor fails (auto by backend)',
    )
    parser.add_argument(
        '--struct-connectivity',
        choices=(4, 8),
        default=8,
        type=int,
        help='pixel connectivity used by structural relation counterfactual checks',
    )
    parser.add_argument(
        '--ph-start-epoch',
        default=20,
        type=int,
        help='first epoch that enables PH-guided CPS; earlier epochs use vanilla CPS',
    )
    parser.add_argument(
        '--ph-mask-dilation',
        default=1,
        type=int,
        help='dilation radius for PH topology conflict mask in unlabeled CPS only',
    )
    parser.add_argument(
        '--ph-match-radius',
        default=16.0,
        type=float,
        help='maximum birth/death critical point distance in pixels for cross-branch PH matching',
    )
    parser.add_argument(
        '--ph-unmatched-mode',
        choices=('vanilla', 'abstain'),
        default='vanilla',
        help='handling for unmatched PH structure conflict candidates in directional CPS',
    )
    parser.add_argument(
        '--ph-conflict-detector',
        choices=('legacy', 'persistent_relation'),
        default='persistent_relation',
        help='PH conflict detector used by --ph-directional-cps',
    )
    parser.add_argument(
        '--save-ph-match-examples',
        action='store_true',
        help='save PH matching example visualizations when supported',
    )
    parser.add_argument(
        '--ph-validation-diagnostics-interval',
        default=None,
        type=int,
        help=(
            'run expensive PH validation oracle/event/example diagnostics every N epochs; '
            'default follows --best_save_interval, 0 disables them'
        ),
    )
    parser.add_argument(
        '--ph-debug-timing',
        action='store_true',
        help='print per-batch persistent-relation PH timing and slow-stage warnings',
    )
    parser.add_argument(
        '--ph-correspondence-mode',
        choices=('birth_nearest', 'support_identity', 'event_projection'),
        default='birth_nearest',
        help='persistent_relation correspondence backend for E2 ablations',
    )
    parser.add_argument(
        '--ph-relation-backend',
        choices=('cc', 'ph_tree'),
        default='cc',
        help='persistent_relation structural relation backend for E2 ablations',
    )
    parser.add_argument(
        '--ph-presence-conflict-mode',
        '--ph_presence_conflict_mode',
        dest='ph_presence_conflict_mode',
        choices=('strict', 'connectivity_outcome'),
        default='strict',
        help='PH-tree ABSENT handling for structural conflict training',
    )
    parser.add_argument(
        '--ph-save-conflict-vis',
        '--ph_save_conflict_vis',
        '--save-conflicts-vis',
        '--save_conflicts_vis',
        dest='ph_save_conflict_vis',
        action='store_true',
        help='save fixed-sample validation PH pure-relation conflict visualizations',
    )
    parser.add_argument(
        '--ph-vis-interval',
        '--ph_vis_interval',
        dest='ph_vis_interval',
        default=10,
        type=int,
        help='epoch interval for --ph-save-conflict-vis outputs',
    )
    parser.add_argument(
        '--ph-vis-num-samples',
        '--ph_vis_num_samples',
        dest='ph_vis_num_samples',
        default=4,
        type=int,
        help='number of fixed validation samples to visualize for PH conflicts',
    )
    parser.add_argument(
        '--struct-analysis-checkpoint',
        default=None,
        help=(
            'run validation structural diagnostics from an existing checkpoint '
            'or checkpoint directory and exit'
        ),
    )
    parser.add_argument(
        '--struct-analysis-checkpoint-a',
        default=None,
        help='student A checkpoint for standalone structural analysis',
    )
    parser.add_argument(
        '--struct-analysis-checkpoint-b',
        default=None,
        help='student B checkpoint for standalone structural analysis',
    )
    parser.add_argument(
        '--struct-analysis-epoch',
        default=0,
        type=int,
        help='epoch to resolve inside a checkpoint directory; 0 uses final best_result files',
    )
    parser.add_argument(
        '--struct-output-dir',
        default=None,
        help='output directory for standalone structural diagnostics',
    )

    args = parser.parse_args()
    if args.milestone_interval < 0:
        parser.error('--milestone_interval must be non-negative')
    if args.best_save_interval < 0:
        parser.error('--best_save_interval must be non-negative')
    if args.struct_min_area < 0:
        parser.error('--struct-min-area must be non-negative')
    if args.struct_match_iou < 0.0 or args.struct_match_iou > 1.0:
        parser.error('--struct-match-iou must be in [0, 1]')
    if args.struct_boundary_width < 0:
        parser.error('--struct-boundary-width must be non-negative')
    if args.ph_start_epoch < 1:
        parser.error('--ph-start-epoch must be >= 1')
    if args.struct_start_epoch < 1:
        parser.error('--struct-start-epoch must be >= 1')
    if not 0.0 < args.struct_tau < 1.0:
        parser.error('--struct-tau must be in (0, 1)')
    if args.struct_graph_margin < 0:
        parser.error('--struct-graph-margin must be non-negative')
    if args.struct_graph_max_pixels < 1:
        parser.error('--struct-graph-max-pixels must be positive')
    if args.struct_arbitration_threshold < 0.0 or args.struct_arbitration_threshold > 1.0:
        parser.error('--struct-arbitration-threshold must be in [0, 1]')
    if args.ph_mask_dilation < 0:
        parser.error('--ph-mask-dilation must be non-negative')
    if args.ph_match_radius < 0:
        parser.error('--ph-match-radius must be non-negative')
    if (
        args.ph_validation_diagnostics_interval is not None
        and args.ph_validation_diagnostics_interval < 0
    ):
        parser.error('--ph-validation-diagnostics-interval must be non-negative')
    if args.ph_mask_only and args.ph_directional_cps:
        parser.error('--ph-mask-only and --ph-directional-cps are mutually exclusive')
    if args.use_structural_projection and (args.ph_mask_only or args.ph_directional_cps):
        parser.error(
            '--use-structural-projection cannot be combined with legacy PH CPS modes'
        )
    if (
        args.struct_analysis_checkpoint is None
        and bool(args.struct_analysis_checkpoint_a) != bool(args.struct_analysis_checkpoint_b)
    ):
        parser.error(
            '--struct-analysis-checkpoint-a and --struct-analysis-checkpoint-b '
            'must be provided together'
        )
    if args.struct_analysis_checkpoint is None and args.struct_analysis_checkpoint_a:
        args.struct_analysis_checkpoint = args.struct_analysis_checkpoint_a
    if args.path_trained_models is None:
        args.path_trained_models = 'checkpoints/semi_fdcl_paper_{}'.format(
            args.labeled_ratio
        )
    if args.path_seg_results is None:
        args.path_seg_results = 'seg_pred/semi_fdcl_paper_{}'.format(
            args.labeled_ratio
        )
    if args.struct_use_mincut_fallback is None:
        args.struct_use_mincut_fallback = has_compiled_mincut_backend()
    return args


def main():
    args = parse_args()
    init_seeds(args.sd)
    cfg = dataset_cfg(args.dataset_name)
    if args.struct_analysis_checkpoint is not None:
        inferred_input_mode = _infer_input_mode_from_path(
            args.struct_analysis_checkpoint
        )
        if inferred_input_mode is not None and args.input_mode == 'rgb_rgb':
            args.input_mode = inferred_input_mode
    datasets, manifest_dir, use_manifests = build_datasets(args, cfg)

    if args.check_data:
        print_data_summary(args, datasets, manifest_dir, use_manifests)
        return
    print_data_summary(args, datasets, manifest_dir, use_manifests)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'use --check_data to validate input only.'
        )
    if args.struct_analysis_checkpoint is not None:
        run_structural_analysis(args, cfg, datasets)
        return

    train_model(args, cfg, datasets, use_manifests)


if __name__ == '__main__':
    main()
