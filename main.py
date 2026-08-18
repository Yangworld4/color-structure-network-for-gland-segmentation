import argparse
import csv
import logging
import os
import random
import time
from typing import Dict, Mapping, Tuple
from warnings import simplefilter

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from config.eval_config.eval import evaluate
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
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network


simplefilter(action='ignore', category=FutureWarning)

METRIC_FIELDS = [
    'epoch', 'lr_model1', 'lr_model2', 'unsup_weight',
    'train_loss_sup1', 'train_loss_sup2',
    'train_loss_aw_to_s', 'train_loss_as_to_s', 'train_loss_s_to_as',
    'train_loss_unsup', 'train_loss_total',
    'train_threshold1', 'train_jaccard1', 'train_dice1',
    'train_threshold2', 'train_jaccard2', 'train_dice2',
    'val_loss_aw', 'val_loss_as', 'val_loss_s',
    'val_threshold1', 'val_jaccard1', 'val_dice1',
    'val_threshold2', 'val_jaccard2', 'val_dice2',
    'val_threshold_aw', 'val_jaccard_aw', 'val_dice_aw',
    'val_threshold_as', 'val_jaccard_as', 'val_dice_as',
    'val_threshold_s', 'val_jaccard_s', 'val_dice_s',
    'pseudo_jc_aw', 'pseudo_dice_aw', 'pseudo_accuracy_aw',
    'pseudo_jc_as', 'pseudo_dice_as', 'pseudo_accuracy_as',
    'pseudo_jc_s', 'pseudo_dice_s', 'pseudo_accuracy_s',
    'aw_as_disagreement', 'aw_correct_as_wrong',
    'as_correct_aw_wrong', 'aw_as_both_correct', 'aw_as_both_wrong',
    'best_val_jaccard1', 'best_val_jaccard2',
    'best_val_jaccard', 'epoch_seconds',
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


def get_model_inputs(
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select clean Appearance and Structure inputs for validation/inference."""
    appearance = batch.get(
        'image_appearance_weak', batch.get('image_appearance', batch['image'])
    )
    structure = batch.get('image_structure', batch['image'])
    return appearance, structure


def get_training_views(
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned Appearance weak/strong and Structure tensors."""
    weak = batch.get(
        'image_appearance_weak', batch.get('image_appearance', batch['image'])
    )
    strong = batch.get(
        'image_appearance_strong', batch.get('image_appearance', batch['image'])
    )
    structure = batch.get('image_structure', batch['image'])
    spatial_shapes = {tuple(view.shape[-2:]) for view in (weak, strong, structure)}
    if len(spatial_shapes) != 1:
        raise ValueError(
            'CPS views are not spatially aligned: weak={}, strong={}, structure={}'.format(
                tuple(weak.shape), tuple(strong.shape), tuple(structure.shape)
            )
        )
    return weak, strong, structure


def compute_cps_losses(
    routing: str,
    criterion,
    prediction_aw: torch.Tensor,
    prediction_as: torch.Tensor,
    prediction_s: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute hard-pseudo-label CPS while detaching every teacher path."""
    pseudo_s = prediction_s.detach().argmax(dim=1)
    zero = prediction_s.new_zeros(())
    if routing == 'symmetric':
        pseudo_as = prediction_as.detach().argmax(dim=1)
        loss_aw_to_s = zero
        loss_as_to_s = criterion(prediction_s, pseudo_as)
    elif routing == 'weak_structure_strong':
        pseudo_aw = prediction_aw.detach().argmax(dim=1)
        loss_aw_to_s = criterion(prediction_s, pseudo_aw)
        loss_as_to_s = zero
    else:
        raise ValueError('Unsupported CPS routing: {}'.format(routing))

    loss_s_to_as = criterion(prediction_as, pseudo_s)
    return {
        'aw_to_s': loss_aw_to_s,
        'as_to_s': loss_as_to_s,
        's_to_as': loss_s_to_as,
        'total': loss_aw_to_s + loss_as_to_s + loss_s_to_as,
    }


def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


def build_datasets(args, cfg):
    """Build the supervised, unlabeled, and validation datasets."""
    dual_view_cfg = cfg['dual_view']
    if args.branch_input_mode is not None:
        dual_view_cfg['enabled'] = (
            args.branch_input_mode == 'heterogeneous'
        )
    elif args.dual_view is not None:
        dual_view_cfg['enabled'] = args.dual_view
    args.effective_branch_input_mode = (
        'heterogeneous' if dual_view_cfg['enabled'] else 'homogeneous'
    )
    if (
        args.cps_routing == 'weak_structure_strong'
        and args.effective_branch_input_mode != 'heterogeneous'
    ):
        raise ValueError(
            '--cps-routing weak_structure_strong requires heterogeneous '
            'weak/strong/structure views'
        )

    manifest_dir = args.manifest_dir or os.path.join(
        args.path_dataset, 'manifests', 'paper_seed_2026'
    )
    use_manifests = os.path.isdir(manifest_dir)

    mean_key, std_key = _normalization_keys(args.input1)
    transforms = data_transform_2d()
    normalize = data_normalize_2d(cfg[mean_key], cfg[std_key])

    common = {
        'data_normalize_1': normalize,
        'dual_view_config': dual_view_cfg,
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
        data_transform_1=transforms['train'],
        sup=False,
        training=True,
    )
    train_sup_unique = imagefloder_itn(
        **labeled_source,
        **common,
        data_transform_1=transforms['train'],
        sup=True,
        training=True,
    )
    train_sup = imagefloder_itn(
        **labeled_source,
        **common,
        data_transform_1=transforms['train'],
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
    print('Branch input mode:', args.effective_branch_input_mode)
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
    print(
        'Supervised image:',
        tuple(sup_sample['image'].shape),
        sup_sample['image'].dtype,
    )
    if 'image_structure' in sup_sample:
        print(
            'Supervised aligned views (rgb_weak, rgb_strong, structure, mask):',
            tuple(sup_sample['image_appearance_weak'].shape),
            tuple(sup_sample['image_appearance_strong'].shape),
            tuple(sup_sample['image_structure'].shape),
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
    if 'image_structure' in unsup_sample:
        print(
            'Unsupervised aligned views (rgb_weak, rgb_strong, structure):',
            tuple(unsup_sample['image_appearance_weak'].shape),
            tuple(unsup_sample['image_appearance_strong'].shape),
            tuple(unsup_sample['image_structure'].shape),
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
    if args.effective_branch_input_mode == 'homogeneous':
        experiment_name += '-input=homogeneous'
    experiment_name += '-routing={}'.format(args.cps_routing)
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
    run_id = time.strftime('%Y%m%d-%H%M%S')
    log_path = os.path.join(
        trained_models_path, 'train_{}.log'.format(run_id)
    )
    metrics_path = os.path.join(
        trained_models_path, 'metrics_{}.csv'.format(run_id)
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
        '| branch_input_mode=%s',
        args.dataset_name,
        args.labeled_ratio,
        args.network,
        args.effective_branch_input_mode,
    )
    logger.info('CPS routing: %s', args.cps_routing)
    logger.info('Arguments: %s', vars(args))

    print('Training log:', log_path)
    print('Metrics CSV:', metrics_path)
    return logger, metrics_path


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
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer1, scheduler1 = _build_optimizer_and_scheduler(model1, args)
    optimizer2, scheduler2 = _build_optimizer_and_scheduler(model2, args)
    return (
        (model1, model2),
        (optimizer1, optimizer2),
        (scheduler1, scheduler2),
        criterion,
    )


def train_one_epoch(
    args,
    epoch,
    models,
    optimizers,
    criterion,
    dataloaders,
):
    model1, model2 = models
    optimizer1, optimizer2 = optimizers
    model1.train()
    model2.train()
    loss_totals = {
        'sup1': 0.0,
        'sup2': 0.0,
        'aw_to_s': 0.0,
        'as_to_s': 0.0,
        's_to_as': 0.0,
        'unsup': 0.0,
        'total': 0.0,
    }
    unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
    supervised_batches = iter(dataloaders['train_sup'])
    unsupervised_batches = iter(dataloaders['train_unsup'])
    num_train_batches = len(dataloaders['train_sup'])

    score_list1 = None
    score_list2 = None
    mask_list = None

    for batch_index in range(num_train_batches):
        unsup_batch = next(unsupervised_batches)
        unsup_aw, unsup_as, unsup_s = get_training_views(unsup_batch)
        unsup_aw = unsup_aw.cuda(non_blocking=True)
        unsup_as = unsup_as.cuda(non_blocking=True)
        unsup_s = unsup_s.cuda(non_blocking=True)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # A_weak and A_strong use this single model1. The baseline does not
        # forward unused unlabeled A_weak, avoiding an unintended BN update.
        if args.cps_routing == 'weak_structure_strong':
            unsup_prediction_aw = model1(unsup_aw)[0]
        else:
            unsup_prediction_aw = None
        unsup_prediction_as = model1(unsup_as)[0]
        unsup_prediction_s = model2(unsup_s)[0]
        cps_losses = compute_cps_losses(
            args.cps_routing,
            criterion,
            unsup_prediction_aw,
            unsup_prediction_as,
            unsup_prediction_s,
        )
        unsup_loss = cps_losses['total'] * unsup_weight
        unsup_loss.backward()

        sup_batch = next(supervised_batches)
        sup_aw, sup_as, sup_s = get_training_views(sup_batch)
        sup_aw = sup_aw.cuda(non_blocking=True)
        sup_as = sup_as.cuda(non_blocking=True)
        sup_s = sup_s.cuda(non_blocking=True)
        sup_mask = sup_batch['mask'].cuda(non_blocking=True)

        sup_prediction_aw = model1(sup_aw)[0]
        sup_prediction_as = model1(sup_as)[0]
        sup_prediction_s = model2(sup_s)[0]

        if epoch == 0 and batch_index == 0:
            shape_message = (
                'Aligned input shapes | rgb_weak={} | rgb_strong={} '
                '| structure_view={} | mask={}'
            ).format(
                tuple(sup_aw.shape), tuple(sup_as.shape),
                tuple(sup_s.shape), tuple(sup_mask.shape),
            )
            print(shape_message)

        if batch_index == 0:
            score_list1 = sup_prediction_aw.detach()
            score_list2 = sup_prediction_s.detach()
            mask_list = sup_mask.detach()
        elif batch_index <= num_train_batches / 32:
            score_list1 = torch.cat(
                (score_list1, sup_prediction_aw.detach()), dim=0
            )
            score_list2 = torch.cat(
                (score_list2, sup_prediction_s.detach()), dim=0
            )
            mask_list = torch.cat((mask_list, sup_mask.detach()), dim=0)

        supervised_loss1 = 0.5 * (
            criterion(sup_prediction_aw, sup_mask)
            + criterion(sup_prediction_as, sup_mask)
        )
        supervised_loss2 = criterion(sup_prediction_s, sup_mask)
        supervised_loss = supervised_loss1 + supervised_loss2
        supervised_loss.backward()

        optimizer1.step()
        optimizer2.step()

        total_loss = unsup_loss + supervised_loss
        loss_totals['aw_to_s'] += (
            cps_losses['aw_to_s'] * unsup_weight
        ).item()
        loss_totals['as_to_s'] += (
            cps_losses['as_to_s'] * unsup_weight
        ).item()
        loss_totals['s_to_as'] += (
            cps_losses['s_to_as'] * unsup_weight
        ).item()
        loss_totals['unsup'] += unsup_loss.item()
        loss_totals['sup1'] += supervised_loss1.item()
        loss_totals['sup2'] += supervised_loss2.item()
        loss_totals['total'] += total_loss.item()
    return loss_totals, unsup_weight, (score_list1, score_list2, mask_list)


@torch.no_grad()
def validate(models, criterion, dataloader):
    model1, model2 = models
    model1.eval()
    model2.eval()

    loss_aw = 0.0
    loss_as = 0.0
    loss_s = 0.0
    scores_aw = []
    scores_as = []
    scores_s = []
    masks = []
    names = []

    for batch in dataloader:
        input_aw, input_as, input_s = get_training_views(batch)
        input_aw = input_aw.cuda(non_blocking=True)
        input_as = input_as.cuda(non_blocking=True)
        input_s = input_s.cuda(non_blocking=True)
        mask = batch['mask'].cuda(non_blocking=True)

        output_aw = model1(input_aw)[0]
        output_as = model1(input_as)[0]
        output_s = model2(input_s)[0]
        scores_aw.append(output_aw)
        scores_as.append(output_as)
        scores_s.append(output_s)
        masks.append(mask)
        names.extend(batch['ID'])
        loss_aw += criterion(output_aw, mask).item()
        loss_as += criterion(output_as, mask).item()
        loss_s += criterion(output_s, mask).item()

    scores_aw_tensor = torch.cat(scores_aw, dim=0)
    scores_as_tensor = torch.cat(scores_as, dim=0)
    scores_s_tensor = torch.cat(scores_s, dim=0)

    return {
        'loss_aw': loss_aw,
        'loss_as': loss_as,
        'loss_s': loss_s,
        'scores_aw': scores_aw_tensor,
        'scores_as': scores_as_tensor,
        'scores_s': scores_s_tensor,
        # Compatibility aliases: formal validation is clean Appearance + Structure.
        'loss1': loss_aw,
        'loss2': loss_s,
        'scores1': scores_aw_tensor,
        'scores2': scores_s_tensor,
        'masks': torch.cat(masks, dim=0),
        'names': np.asarray(names),
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


def compute_pseudo_label_diagnostics(validation):
    """Compare hard validation pseudo labels and weak/strong disagreement."""
    masks = validation['masks'].long()
    hard = {
        'aw': validation['scores_aw'].detach().argmax(dim=1),
        'as': validation['scores_as'].detach().argmax(dim=1),
        's': validation['scores_s'].detach().argmax(dim=1),
    }
    result = {}
    for name, prediction in hard.items():
        overlap = _hard_overlap(prediction, masks)
        for metric, value in overlap.items():
            result['pseudo_{}_{}'.format(metric, name)] = value

    correct_aw = hard['aw'].eq(masks)
    correct_as = hard['as'].eq(masks)
    result.update({
        'aw_as_disagreement': float(hard['aw'].ne(hard['as']).float().mean().item()),
        'aw_correct_as_wrong': float((correct_aw & ~correct_as).float().mean().item()),
        'as_correct_aw_wrong': float((correct_as & ~correct_aw).float().mean().item()),
        'aw_as_both_correct': float((correct_aw & correct_as).float().mean().item()),
        'aw_as_both_wrong': float((~correct_aw & ~correct_as).float().mean().item()),
    })
    return result


def _collect_epoch_metrics(
    epoch,
    epoch_seconds,
    unsup_weight,
    optimizers,
    train_losses,
    train_evaluations,
    val_losses,
    val_evaluations,
    val_eval_as,
    pseudo_diagnostics,
    best_val_eval_lists,
):
    train_eval1, train_eval2 = train_evaluations
    val_eval1, val_eval2 = val_evaluations
    optimizer1, optimizer2 = optimizers
    return {
        'epoch': epoch + 1,
        'lr_model1': optimizer1.param_groups[0]['lr'],
        'lr_model2': optimizer2.param_groups[0]['lr'],
        'unsup_weight': unsup_weight,
        'train_loss_sup1': train_losses[0],
        'train_loss_sup2': train_losses[1],
        'train_loss_aw_to_s': train_losses[4],
        'train_loss_as_to_s': train_losses[5],
        'train_loss_s_to_as': train_losses[6],
        'train_loss_unsup': train_losses[2],
        'train_loss_total': train_losses[3],
        'train_threshold1': train_eval1[0],
        'train_jaccard1': train_eval1[1],
        'train_dice1': train_eval1[2],
        'train_threshold2': train_eval2[0],
        'train_jaccard2': train_eval2[1],
        'train_dice2': train_eval2[2],
        'val_loss_aw': val_losses[0],
        'val_loss_as': val_losses[2],
        'val_loss_s': val_losses[1],
        'val_threshold1': val_eval1[0],
        'val_jaccard1': val_eval1[1],
        'val_dice1': val_eval1[2],
        'val_threshold2': val_eval2[0],
        'val_jaccard2': val_eval2[1],
        'val_dice2': val_eval2[2],
        'val_threshold_aw': val_eval1[0],
        'val_jaccard_aw': val_eval1[1],
        'val_dice_aw': val_eval1[2],
        'val_threshold_as': val_eval_as[0],
        'val_jaccard_as': val_eval_as[1],
        'val_dice_as': val_eval_as[2],
        'val_threshold_s': val_eval2[0],
        'val_jaccard_s': val_eval2[1],
        'val_dice_s': val_eval2[2],
        'best_val_jaccard1': best_val_eval_lists[0][1],
        'best_val_jaccard2': best_val_eval_lists[1][1],
        'best_val_jaccard': max(
            best_val_eval_lists[0][1], best_val_eval_lists[1][1]
        ),
        'epoch_seconds': epoch_seconds,
        **pseudo_diagnostics,
    }


def _log_epoch(logger, args, metrics):
    logger.info(
        'Epoch %d/%d | routing=%s '
        '| train_loss=%.6f (sup_a=%.6f, sup_s=%.6f, unsup=%.6f, '
        'Aw->S=%.6f, As->S=%.6f, S->As=%.6f) '
        '| train_jaccard1=%.6f | train_dice1=%.6f '
        '| train_jaccard2=%.6f | train_dice2=%.6f '
        '| val_aw(jc=%.6f,dice=%.6f) '
        '| val_as(jc=%.6f,dice=%.6f) '
        '| val_s(jc=%.6f,dice=%.6f) '
        '| pseudo_jc(aw=%.6f,as=%.6f,s=%.6f) '
        '| Aw-As_disagreement=%.6f '
        '| best_jaccard1=%.6f | best_jaccard2=%.6f '
        '| best_jaccard=%.6f | lr=%.8g | time=%.2fs',
        metrics['epoch'],
        args.num_epochs,
        args.cps_routing,
        metrics['train_loss_total'],
        metrics['train_loss_sup1'],
        metrics['train_loss_sup2'],
        metrics['train_loss_unsup'],
        metrics['train_loss_aw_to_s'],
        metrics['train_loss_as_to_s'],
        metrics['train_loss_s_to_as'],
        metrics['train_jaccard1'],
        metrics['train_dice1'],
        metrics['train_jaccard2'],
        metrics['train_dice2'],
        metrics['val_jaccard1'],
        metrics['val_dice1'],
        metrics['val_jaccard_as'],
        metrics['val_dice_as'],
        metrics['val_jaccard2'],
        metrics['val_dice2'],
        metrics['pseudo_jc_aw'],
        metrics['pseudo_jc_as'],
        metrics['pseudo_jc_s'],
        metrics['aw_as_disagreement'],
        metrics['best_val_jaccard1'],
        metrics['best_val_jaccard2'],
        metrics['best_val_jaccard'],
        metrics['lr_model1'],
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
    trained_models_path, segmentation_results_path = build_output_paths(
        args, datasets, use_manifests
    )
    logger, metrics_path = create_run_outputs(args, trained_models_path)
    diagnostics = None
    if getattr(args, 'complementarity_diagnostics', False):
        if cfg['NUM_CLASSES'] != 2:
            raise ValueError(
                'complementarity diagnostics currently support binary '
                'segmentation only'
            )
        from tools.complementarity_diagnostics import (
            ComplementarityDiagnostics,
        )

        diagnostics_dir = args.diagnostics_output_dir or os.path.join(
            trained_models_path,
            'complementarity_diagnostics',
        )
        diagnostics = ComplementarityDiagnostics(
            output_dir=diagnostics_dir,
            interval=args.diagnostics_interval,
            boundary_width=args.diagnostics_boundary_width,
            max_visualizations=args.diagnostics_max_visualizations,
            save_paired_checkpoints=(
                args.diagnostics_save_paired_checkpoints
            ),
        )
        logger.info(
            'Complementarity diagnostics enabled | output=%s | interval=%d',
            diagnostics_dir,
            args.diagnostics_interval,
        )
    (
        models,
        optimizers,
        schedulers,
        criterion,
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

    with open(metrics_path, 'w', newline='', encoding='utf-8') as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=METRIC_FIELDS
        )
        metrics_writer.writeheader()

        for epoch in range(args.num_epochs):
            epoch_start = time.time()
            loss_totals, unsup_weight, train_predictions = train_one_epoch(
                args,
                epoch,
                models,
                optimizers,
                criterion,
                dataloaders,
            )
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
            train_losses = train_losses + (
                loss_totals['aw_to_s'] / num_batches['train_sup'],
                loss_totals['as_to_s'] / num_batches['train_sup'],
                loss_totals['s_to_as'] / num_batches['train_sup'],
            )
            print(
                '| CPS directions: Aw->S {:.4f} | As->S {:.4f} | S->As {:.4f}'.format(
                    train_losses[4], train_losses[5], train_losses[6]
                )
            )
            train_eval1, train_eval2, _, _ = print_train_eval(
                cfg['NUM_CLASSES'],
                *train_predictions,
                print_num_half,
            )

            validation = validate(models, criterion, dataloaders['val'])
            val_losses = print_val_loss(
                validation['loss1'],
                validation['loss2'],
                num_batches,
                print_num,
                print_num_half,
            )
            val_losses = val_losses + (
                validation['loss_as'] / num_batches['val'],
            )
            val_eval1, val_eval2, _, _ = print_val_eval(
                cfg['NUM_CLASSES'],
                validation['scores1'],
                validation['scores2'],
                validation['masks'],
                print_num_half,
            )
            val_eval_as = evaluate(
                validation['scores_as'], validation['masks']
            )
            pseudo_diagnostics = compute_pseudo_label_diagnostics(validation)
            print(
                '| Val As Jc: {:.4f} | Dice: {:.4f} | '
                'pseudo Jc Aw/As/S: {:.4f}/{:.4f}/{:.4f} | '
                'Aw-As disagree: {:.4f}'.format(
                    val_eval_as[1], val_eval_as[2],
                    pseudo_diagnostics['pseudo_jc_aw'],
                    pseudo_diagnostics['pseudo_jc_as'],
                    pseudo_diagnostics['pseudo_jc_s'],
                    pseudo_diagnostics['aw_as_disagreement'],
                )
            )
            if diagnostics is not None:
                diagnostic_summary = diagnostics.update(
                    epoch,
                    models,
                    validation,
                )
                if diagnostic_summary is not None:
                    logger.info(
                        'Complementarity diagnostics | epoch=%d '
                        '| mean_jaccard=%.6f | entropy_jaccard=%.6f '
                        '| oracle_jaccard=%.6f | oracle_gain=%.6f',
                        diagnostic_summary['epoch'],
                        diagnostic_summary['mean_jaccard'],
                        diagnostic_summary['entropy_jaccard'],
                        diagnostic_summary['oracle_jaccard'],
                        diagnostic_summary[
                            'oracle_gain_over_best_branch'
                        ],
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
                            current_epoch * num_batches['train_sup']
                        ),
                        'dataset_name': args.dataset_name,
                        'labeled_ratio': args.labeled_ratio,
                        'seed': args.sd,
                        'routing': args.cps_routing,
                        'branch_input_mode': (
                            args.effective_branch_input_mode
                        ),
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
                    '| Current A/S weights saved at epoch {}'.format(
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
                    '| Periodic best_A/best_S saved at epoch {}'.format(
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
                unsup_weight,
                optimizers,
                train_losses,
                (train_eval1, train_eval2),
                val_losses,
                (val_eval1, val_eval2),
                val_eval_as,
                pseudo_diagnostics,
                best_val_eval_lists,
            )
            metrics_writer.writerow(metrics)
            metrics_file.flush()
            _log_epoch(logger, args, metrics)

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
    parser.add_argument(
        '--dual_view',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='enable aligned appearance/structure dataset views (default: dataset config)',
    )
    parser.add_argument(
        '--branch_input_mode',
        '--branch-input-mode',
        dest='branch_input_mode',
        choices=['heterogeneous', 'homogeneous'],
        default=None,
        help=(
            'branch inputs: heterogeneous routes RGB to Result1 and '
            'H+Gray+Sobel to Result2; homogeneous routes the same augmented '
            'RGB image to both branches (default: dataset config)'
        ),
    )
    parser.add_argument(
        '--cps-routing',
        '--cps_routing',
        dest='cps_routing',
        choices=['symmetric', 'weak_structure_strong'],
        default='symmetric',
        help=(
            'symmetric: A_strong <-> Structure baseline; '
            'weak_structure_strong: A_weak -> Structure -> A_strong'
        ),
    )
    parser.add_argument('--sup_mark', default='9')
    parser.add_argument('--unsup_mark', default='76')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument(
        '--milestone_interval',
        '--milestone-interval',
        dest='milestone_interval',
        default=10,
        type=int,
        help=(
            '每 N 个 epoch 保存同一时刻的 Appearance/Structure 当前权重；'
            '设为 0 可关闭（默认：10）'
        ),
    )
    parser.add_argument(
        '--best_save_interval',
        '--best-save-interval',
        dest='best_save_interval',
        default=10,
        type=int,
        help=(
            '每 N 个 epoch 额外保存截至当前的 best_A 和 best_S；'
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
    parser.add_argument(
        '--complementarity_diagnostics',
        action='store_true',
        help=(
            'enable separate dual-branch fusion/oracle diagnostics; disabled '
            'by default and does not change training or checkpoint selection'
        ),
    )
    parser.add_argument(
        '--diagnostics_output_dir',
        default=None,
        help=(
            'diagnostics directory; defaults to a separate subdirectory of '
            'the current experiment checkpoint directory'
        ),
    )
    parser.add_argument(
        '--diagnostics_interval',
        default=5,
        type=int,
        help='run optional complementarity diagnostics every N epochs',
    )
    parser.add_argument(
        '--diagnostics_boundary_width',
        default=3,
        type=int,
        help='ground-truth boundary half-width in pixels for diagnostics',
    )
    parser.add_argument(
        '--diagnostics_max_visualizations',
        default=8,
        type=int,
        help='maximum number of best-epoch diagnostic panels to save',
    )
    parser.add_argument(
        '--diagnostics_save_paired_checkpoints',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='save both branch weights at the best mean-fusion epoch',
    )
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='xnet_sb')

    args = parser.parse_args()
    if args.branch_input_mode is not None and args.dual_view is not None:
        requested_dual_view = args.branch_input_mode == 'heterogeneous'
        if requested_dual_view != args.dual_view:
            parser.error(
                '--branch_input_mode conflicts with --dual_view/--no-dual_view'
            )
    if args.milestone_interval < 0:
        parser.error('--milestone_interval must be non-negative')
    if args.best_save_interval < 0:
        parser.error('--best_save_interval must be non-negative')
    if args.diagnostics_interval < 1:
        parser.error('--diagnostics_interval must be at least one')
    if args.diagnostics_boundary_width < 1:
        parser.error('--diagnostics_boundary_width must be at least one')
    if args.diagnostics_max_visualizations < 0:
        parser.error('--diagnostics_max_visualizations must be non-negative')
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
    datasets, manifest_dir, use_manifests = build_datasets(args, cfg)

    if args.check_data:
        print_data_summary(args, datasets, manifest_dir, use_manifests)
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            'FDCL training requires a CUDA GPU; '
            'use --check_data to validate input only.'
        )

    train_model(args, cfg, datasets, use_manifests)


if __name__ == '__main__':
    main()
