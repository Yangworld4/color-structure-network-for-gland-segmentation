import argparse
import csv
import logging
import os
import random
import time
from typing import Mapping, Tuple
from warnings import simplefilter

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from config.augmentation.online_aug import data_normalize_2d, data_transform_2d
from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import (
    print_best,
    print_train_eval,
    print_train_loss,
    print_val_eval,
    print_val_loss,
    save_val_best_2d,
)
from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.dataset_2d import imagefloder_itn
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network


simplefilter(action='ignore', category=FutureWarning)

METRIC_FIELDS = [
    'epoch', 'lr_model1', 'lr_model2', 'unsup_weight',
    'train_loss_sup1', 'train_loss_sup2', 'train_loss_unsup', 'train_loss_total',
    'train_threshold1', 'train_jaccard1', 'train_dice1',
    'train_threshold2', 'train_jaccard2', 'train_dice2',
    'val_loss1', 'val_loss2',
    'val_threshold1', 'val_jaccard1', 'val_dice1',
    'val_threshold2', 'val_jaccard2', 'val_dice2',
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
    """Route RGB appearance to model1 and H/Gray/Sobel structure to model2."""
    appearance = batch.get('image_appearance', batch['image'])
    structure = batch.get('image_structure', batch['image'])
    return appearance, structure


def _normalization_keys(input_name):
    if input_name == 'image':
        return 'MEAN', 'STD'
    return 'MEAN_' + input_name, 'STD_' + input_name


def build_datasets(args, cfg):
    """Build the supervised, unlabeled, and validation datasets."""
    dual_view_cfg = cfg['dual_view']
    if args.dual_view is not None:
        dual_view_cfg['enabled'] = args.dual_view

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
            'Supervised dual views:',
            tuple(sup_sample['image_appearance'].shape),
            tuple(sup_sample['image_structure'].shape),
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
            'Unsupervised dual views:',
            tuple(unsup_sample['image_appearance'].shape),
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
    return (
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
        'Training started | dataset=%s | labeled_ratio=%s | network=%s',
        args.dataset_name,
        args.labeled_ratio,
        args.network,
    )
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


def _feature_diversification_loss(feature1, feature2):
    distance1 = 1 / (
        torch.square(feature1.detach() - feature2).mean() + 1e-6
    )
    distance2 = 1 / (
        torch.square(feature2.detach() - feature1).mean() + 1e-6
    )
    return (distance1 + distance2) / 2


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
        unsup_input1, unsup_input2 = get_model_inputs(unsup_batch)
        unsup_input1 = unsup_input1.cuda(non_blocking=True)
        unsup_input2 = unsup_input2.cuda(non_blocking=True)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        unsup_prediction1, unsup_feature1 = model1(unsup_input1)
        unsup_prediction2, unsup_feature2 = model2(unsup_input2)
        pseudo_label1 = unsup_prediction1.argmax(dim=1)
        pseudo_label2 = unsup_prediction2.argmax(dim=1)

        unsup_diversification = _feature_diversification_loss(
            unsup_feature1, unsup_feature2
        )
        unsup_loss = (
            criterion(unsup_prediction1, pseudo_label2)
            + criterion(unsup_prediction2, pseudo_label1)
            + args.w_dis * unsup_diversification
        ) * unsup_weight
        unsup_loss.backward()

        sup_batch = next(supervised_batches)
        sup_input1, sup_input2 = get_model_inputs(sup_batch)
        sup_input1 = sup_input1.cuda(non_blocking=True)
        sup_input2 = sup_input2.cuda(non_blocking=True)
        sup_mask = sup_batch['mask'].cuda(non_blocking=True)

        sup_prediction1, sup_feature1 = model1(sup_input1)
        sup_prediction2, sup_feature2 = model2(sup_input2)

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

        supervised_loss1 = criterion(sup_prediction1, sup_mask)
        supervised_loss2 = criterion(sup_prediction2, sup_mask)
        sup_diversification = _feature_diversification_loss(
            sup_feature1, sup_feature2
        )
        supervised_loss = (
            supervised_loss1
            + supervised_loss2
            + args.w_dis * sup_diversification
        )
        supervised_loss.backward()

        optimizer1.step()
        optimizer2.step()

        total_loss = unsup_loss + supervised_loss
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

    loss1 = 0.0
    loss2 = 0.0
    scores1 = []
    scores2 = []
    masks = []
    names = []

    for batch in dataloader:
        input1, input2 = get_model_inputs(batch)
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        mask = batch['mask'].cuda(non_blocking=True)

        output1 = model1(input1)[0]
        output2 = model2(input2)[0]
        scores1.append(output1)
        scores2.append(output2)
        masks.append(mask)
        names.extend(batch['ID'])
        loss1 += criterion(output1, mask).item()
        loss2 += criterion(output2, mask).item()

    return {
        'loss1': loss1,
        'loss2': loss2,
        'scores1': torch.cat(scores1, dim=0),
        'scores2': torch.cat(scores2, dim=0),
        'masks': torch.cat(masks, dim=0),
        'names': np.asarray(names),
    }


def _collect_epoch_metrics(
    epoch,
    epoch_seconds,
    unsup_weight,
    optimizers,
    train_losses,
    train_evaluations,
    val_losses,
    val_evaluations,
    best_val_eval_list,
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
        'train_loss_unsup': train_losses[2],
        'train_loss_total': train_losses[3],
        'train_threshold1': train_eval1[0],
        'train_jaccard1': train_eval1[1],
        'train_dice1': train_eval1[2],
        'train_threshold2': train_eval2[0],
        'train_jaccard2': train_eval2[1],
        'train_dice2': train_eval2[2],
        'val_loss1': val_losses[0],
        'val_loss2': val_losses[1],
        'val_threshold1': val_eval1[0],
        'val_jaccard1': val_eval1[1],
        'val_dice1': val_eval1[2],
        'val_threshold2': val_eval2[0],
        'val_jaccard2': val_eval2[1],
        'val_dice2': val_eval2[2],
        'best_val_jaccard': best_val_eval_list[1],
        'epoch_seconds': epoch_seconds,
    }


def _log_epoch(logger, args, metrics):
    logger.info(
        'Epoch %d/%d | train_loss=%.6f (sup1=%.6f, sup2=%.6f, unsup=%.6f) '
        '| train_jaccard1=%.6f | train_dice1=%.6f '
        '| train_jaccard2=%.6f | train_dice2=%.6f '
        '| val_loss1=%.6f | val_loss2=%.6f '
        '| val_jaccard1=%.6f | val_dice1=%.6f '
        '| val_jaccard2=%.6f | val_dice2=%.6f '
        '| best_jaccard=%.6f | lr=%.8g | time=%.2fs',
        metrics['epoch'],
        args.num_epochs,
        metrics['train_loss_total'],
        metrics['train_loss_sup1'],
        metrics['train_loss_sup2'],
        metrics['train_loss_unsup'],
        metrics['train_jaccard1'],
        metrics['train_dice1'],
        metrics['train_jaccard2'],
        metrics['train_dice2'],
        metrics['val_loss1'],
        metrics['val_loss2'],
        metrics['val_jaccard1'],
        metrics['val_dice1'],
        metrics['val_jaccard2'],
        metrics['val_dice2'],
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
    models, optimizers, schedulers, criterion = build_training_components(
        args, cfg
    )

    model1, model2 = models
    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for _ in range(4)]
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
            val_eval1, val_eval2, _, _ = print_val_eval(
                cfg['NUM_CLASSES'],
                validation['scores1'],
                validation['scores2'],
                validation['masks'],
                print_num_half,
            )
            best_val_eval_list, best_model, best_result = save_val_best_2d(
                cfg['NUM_CLASSES'],
                best_model,
                best_val_eval_list,
                best_result,
                model1,
                model2,
                validation['scores1'],
                validation['scores2'],
                validation['names'],
                val_eval1,
                val_eval2,
                trained_models_path,
                segmentation_results_path,
                cfg['PALETTE'],
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
                best_val_eval_list,
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
    print_best(
        cfg['NUM_CLASSES'],
        best_val_eval_list,
        best_model,
        best_result,
        trained_models_path,
        print_num_minus,
    )
    print('=' * print_num)
    logger.info(
        'Training completed | best_result=%s | best_val_jaccard=%.6f',
        best_result,
        best_val_eval_list[1],
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
    parser.add_argument('--sup_mark', default='9')
    parser.add_argument('--unsup_mark', default='76')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=5, type=float)
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay exponent')
    parser.add_argument(
        '--w_dis',
        default=1.0,
        type=float,
        help='weight of the feature diversification loss',
    )
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='xnet_sb')

    args = parser.parse_args()
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
