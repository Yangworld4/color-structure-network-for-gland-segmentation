import argparse
import csv
from contextlib import ExitStack, redirect_stdout
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
from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.dataset_2d import imagefloder_itn
from loss.loss_function import DiceLoss
from models.getnetwork import get_network
from models.uni import load_uni


simplefilter(action='ignore', category=FutureWarning)

METRIC_FIELDS = [
    'iteration', 'epoch',
    'train_loss_sup1', 'train_loss_sup2', 'train_loss_cps',
    'train_loss_total', 'student12_disagreement',
    'student12_dice_similarity', 'student1_entropy', 'student2_entropy',
    'pseudo_dice_student1', 'pseudo_dice_student2',
    'pseudo_jaccard_student1', 'pseudo_jaccard_student2',
]

UNI_WEIGHTS_DEFAULT = (
    '/home/yangorange/.cache/huggingface/hub/models--MahmoodLab--UNI/'
    'snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin'
)
UNI_CONFIG_DEFAULT = str(Path(UNI_WEIGHTS_DEFAULT).with_name('config.json'))
UNI_SIZE = (224, 224)
UNI_MEAN = (0.485, 0.456, 0.406)
UNI_STD = (0.229, 0.224, 0.225)
REFERENCE_MEMORY_ANCHOR_FG = 0.90
REFERENCE_MEMORY_ANCHOR_BG = 0.10
REFERENCE_MEMORY_PER_IMAGE_PER_CLASS = 64
REFERENCE_MEMORY_SEED = 2026
EXPECTED_UNI_SHA256 = (
    '56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40'
)


def _reference_memory_dataset_slug(dataset_name):
    """Return a filesystem-safe, stable dataset identifier."""
    slug = ''.join(
        character if character.isalnum() else '_'
        for character in str(dataset_name).strip().lower()
    ).strip('_')
    return slug or 'dataset'


def _default_reference_memory_path(dataset_name, labeled_ratio, seed):
    """Resolve the default dataset-isolated memory path under checkpoints."""
    root = Path(__file__).resolve().parent / 'checkpoints'
    slug = _reference_memory_dataset_slug(dataset_name)
    return root / (
        '{}_labeled{}_seed{}'.format(slug, int(labeled_ratio), int(seed))
    ) / 'reference_memory.npz'

SC_METRIC_FIELDS = [
    'epoch', 'total_unlabeled_pixels', 'agreement_pixels', 'agreement_rate',
    'disagreement_pixels', 'disagreement_rate', 'fg_consensus_pixels',
    'bg_consensus_pixels', 'reference_conflict_pixels',
    'reference_conflict_rate', 'fg_conflict_pixels', 'fg_conflict_rate',
    'bg_conflict_pixels', 'bg_conflict_rate',
    'mean_signed_support_agreement', 'mean_r_agreement', 'mean_r_conflict',
    'mean_r_fg', 'mean_r_bg', 'sum_reference_target_mass_shift',
    'mean_reference_target_mass_shift', 'potential_reverse_A_rate',
    'potential_reverse_B_rate', 'potential_reverse_any_rate',
    'potential_reverse_both_rate', 'original_cps_loss',
    'soft_correction_cps_loss', 'cps_loss_delta', 'supervised_loss_A',
    'supervised_loss_B', 'total_loss', 'learning_rate',
]

SC_ITER_METRIC_FIELDS = [
    'iteration', 'global_step', 'epoch', 'agreement_rate', 'conflict_rate',
    'mean_r', 'potential_reverse_A_rate', 'potential_reverse_B_rate',
    'original_cps_loss', 'soft_correction_cps_loss', 'loss_delta',
]

SC_VALIDATION_FIELDS = [
    'epoch', 'valid_pixels', 'CC', 'WW', 'AC', 'BC', 'D',
    'agreement_rate', 'disagreement_rate', 'absolute_WW_rate',
    'shared_FP', 'shared_FN', 'HC90_WW_all', 'HC95_WW_all',
    'mean_r_CC', 'mean_r_WW', 'conflict_rate_CC', 'conflict_rate_WW',
    'WW_recovery_rate_A', 'WW_recovery_rate_B', 'WW_any_reverse',
    'WW_both_reverse', 'CC_harmful_reverse_A', 'CC_harmful_reverse_B',
    'CC_any_harm', 'CC_both_harm', 'WW_helpful_reverse_mass',
    'CC_harmful_reverse_mass', 'FG_WW_recovery_rate',
    'FG_CC_harmful_reverse_rate', 'FG_WW_helpful_mass',
    'FG_CC_harmful_mass', 'BG_WW_recovery_rate',
    'BG_CC_harmful_reverse_rate', 'BG_WW_helpful_mass',
    'BG_CC_harmful_mass', 'val_dice_student1', 'val_dice_student2',
    'val_jaccard_student1', 'val_jaccard_student2',
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
    """Build training, validation, and test datasets."""
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
    test = imagefloder_itn(
        **test_source,
        **common,
        data_transform_1=transforms['test'],
        sup=True,
        training=False,
    )

    datasets = {
        'train_sup': train_sup,
        'train_sup_unique': train_sup_unique,
        'train_unsup': train_unsup,
        'val': validation,
        'test': test,
    }
    return datasets, manifest_dir, use_manifests


def print_data_summary(args, datasets, manifest_dir, use_manifests):
    sup_sample = datasets['train_sup'][0]
    unsup_sample = datasets['train_unsup'][0]
    val_sample = datasets['val'][0]
    test_sample = datasets['test'][0]

    print('Data source:', manifest_dir if use_manifests else args.path_dataset)
    print('Input mode:', args.input_mode)
    print(
        'Unique samples: supervised={}, unsupervised={}, validation={}, test={}'.format(
            len(datasets['train_sup_unique']),
            len(datasets['train_unsup']),
            len(datasets['val']),
            len(datasets['test']),
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
    return {
        'train_sup': DataLoader(
            datasets['train_sup'], shuffle=True, **loader_options
        ),
        'train_unsup': DataLoader(
            datasets['train_unsup'], shuffle=True, **loader_options
        ),
        'val': DataLoader(datasets['val'], shuffle=False, **loader_options),
        'test': DataLoader(datasets['test'], shuffle=False, **loader_options),
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
):
    """Build an immutable memory once, before the CPS optimizer loop."""
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
    with torch.inference_mode():
        for batch in loader:
            rgb = _reconstruct_rgb(
                batch['image_rgb'].cuda(non_blocking=True), cfg
            )
            gt = batch['mask'].cuda(non_blocking=True).float().unsqueeze(1)
            tokens = _uni_dense_features(uni, rgb)
            grid = int(math.sqrt(int(tokens.shape[1])))
            token_grid = grid if token_grid is None else token_grid
            feature_dim = int(tokens.shape[-1]) if feature_dim is None else feature_dim
            occupancy = F.adaptive_avg_pool2d(gt, (grid, grid)).flatten(1)
            token_cpu = tokens.detach().cpu().numpy()
            occupancy_cpu = occupancy.detach().cpu().numpy()
            for index in range(len(batch['ID'])):
                for cls, threshold, relation in (
                    (1, REFERENCE_MEMORY_ANCHOR_FG, lambda value: value >= threshold),
                    (0, REFERENCE_MEMORY_ANCHOR_BG, lambda value: value <= threshold),
                ):
                    candidates = np.flatnonzero(relation(occupancy_cpu[index]))
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

    if token_grid != 14 or feature_dim != 1024:
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
        top_k=np.array(5),
        anchor_fg=np.array(REFERENCE_MEMORY_ANCHOR_FG),
        anchor_bg=np.array(REFERENCE_MEMORY_ANCHOR_BG),
        memory_per_image_class=np.array(REFERENCE_MEMORY_PER_IMAGE_PER_CLASS),
        source_labeled_ratio=np.array(args.labeled_ratio),
        seed=np.array(REFERENCE_MEMORY_SEED),
        source_dataset=np.array(str(args.dataset_name)),
    )
    logger.info(
        'Reference memory built before training | source=labeled_%s | fg=%d bg=%d '
        '| cap=%d grid=%dx%d dim=%d | path=%s',
        args.labeled_ratio,
        memory_fg.shape[0],
        memory_bg.shape[0],
        REFERENCE_MEMORY_PER_IMAGE_PER_CLASS,
        token_grid,
        token_grid,
        feature_dim,
        memory_path,
    )


def _load_reference(args, cfg, logger, manifest_dir=None, use_manifests=False):
    """Load the immutable UNI model and class-balanced token memory."""
    memory_path = Path(args.uni_reference_memory).expanduser().resolve()
    uni, weights_path, config_path = _load_uni(args)
    memory_built = False
    if not memory_path.is_file():
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
        )
        memory_built = True
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
    if str(args.dataset_name).strip().lower() == 'glas' and args.labeled_ratio == 10:
        assert memory_fg_count == 391, memory_fg_count
        assert memory_bg_count == 294, memory_bg_count
    assert memory_fg_np.shape[1] == 1024, memory_fg_np.shape
    assert memory_bg_np.shape[1] == 1024, memory_bg_np.shape
    assert feature_dim == 1024, feature_dim
    assert token_grid == 14, token_grid
    assert memory_top_k == 5, memory_top_k
    with np.load(str(memory_path), allow_pickle=False) as memory:
        source_ratio = int(memory['source_labeled_ratio']) if 'source_labeled_ratio' in memory.files else 10
        source_dataset = (
            str(memory['source_dataset'].item())
            if 'source_dataset' in memory.files else None
        )
    assert source_ratio == args.labeled_ratio, (source_ratio, args.labeled_ratio)
    if source_dataset is not None:
        assert source_dataset.strip().lower() == str(args.dataset_name).strip().lower(), (
            source_dataset, args.dataset_name
        )
    assert np.allclose(np.linalg.norm(memory_fg_np, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(memory_bg_np, axis=1), 1.0, atol=1e-4)
    assert args.uni_k == 5, args.uni_k
    memory_fg = torch.from_numpy(memory_fg_np).cuda(non_blocking=True)
    memory_bg = torch.from_numpy(memory_bg_np).cuda(non_blocking=True)
    memory_fg_norm_error = float(
        np.max(np.abs(np.linalg.norm(memory_fg_np, axis=1) - 1.0))
    )
    memory_bg_norm_error = float(
        np.max(np.abs(np.linalg.norm(memory_bg_np, axis=1) - 1.0))
    )
    mode_label = 'Soft Correction'
    logger.info(
        '%s UNI/reference loaded | weights=%s | memory=%s | fg=%d bg=%d '
        '| dim=%d grid=%dx%d k=%d source=labeled_%d',
        mode_label, weights_path, memory_path, memory_fg.shape[0],
        memory_bg.shape[0], feature_dim, token_grid, token_grid,
        args.uni_k, args.labeled_ratio,
    )
    logger.info(
        'Soft Correction protocol | hard_reference_threshold_used=false '
        '| student_confidence_used=false | pseudo_label_flipping_used=false'
    )
    logger.info('%s UNI checkpoint sha256=%s', mode_label, _sha256_file(weights_path))
    logger.info('%s reference memory sha256=%s', mode_label, _sha256_file(memory_path))
    logger.info('%s UNI implementation=models/uni.py', mode_label)
    return {
        'uni': uni,
        'memory_fg': memory_fg,
        'memory_bg': memory_bg,
        'feature_dim': feature_dim,
        'token_grid': token_grid,
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
        'hard_reference_threshold_used': False,
    }


def _uni_dense_features(uni, rgb_01):
    mean = torch.tensor(UNI_MEAN, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(UNI_STD, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
    x = F.interpolate(rgb_01, size=UNI_SIZE, mode='bilinear', align_corners=False)
    x = (x - mean) / std
    tokens = uni.dense_patch_tokens(x)
    token_count = int(tokens.shape[1])
    token_grid = int(math.sqrt(token_count))
    if token_grid * token_grid != token_count:
        raise RuntimeError('UNI dense token count is not square: {}'.format(token_count))
    if tokens.shape[-1] != 1024 or token_grid != 14:
        raise RuntimeError(
            'Unexpected UNI dense feature shape: {}'.format(tuple(tokens.shape))
        )
    return tokens


def _topk_scores(query, memory, k):
    similarity = query @ memory.t()
    return similarity.topk(min(int(k), int(memory.shape[0])), dim=1, largest=True).values.mean(dim=1)


def _reference_margin(tokens, context, k):
    flat = tokens.reshape(-1, tokens.shape[-1])
    score_fg = _topk_scores(flat, context['memory_fg'], k)
    score_bg = _topk_scores(flat, context['memory_bg'], k)
    margin = score_fg - score_bg
    return margin.reshape(tokens.shape[0], tokens.shape[1])


def _reconstruct_rgb(image_rgb, cfg):
    mean = torch.tensor(cfg['MEAN'], device=image_rgb.device, dtype=image_rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(cfg['STD'], device=image_rgb.device, dtype=image_rgb.dtype).view(1, 3, 1, 1)
    return (image_rgb * std + mean).clamp(0.0, 1.0)


def _safe_mean(total, count):
    return float(total) / float(count) if count else float('nan')


def _reference_weights_from_margin(pseudo_a, pseudo_b, margin):
    """Compute reference conflict strength without changing CPS targets."""
    if pseudo_a.shape != pseudo_b.shape or pseudo_a.shape != margin.shape:
        raise ValueError(
            'Reference tensor shape mismatch: A={} B={} margin={}'.format(
                tuple(pseudo_a.shape), tuple(pseudo_b.shape), tuple(margin.shape)
            )
        )
    agreement = pseudo_a.eq(pseudo_b)
    disagreement = ~agreement
    consensus_fg = agreement & pseudo_a.eq(1)
    consensus_bg = agreement & pseudo_a.eq(0)
    signed_support = torch.where(pseudo_a.eq(1), margin, -margin)
    conflict = torch.zeros_like(margin)
    conflict[agreement] = torch.clamp(
        -signed_support[agreement], min=0.0, max=1.0
    )
    weight = torch.ones_like(margin)
    weight[agreement] = 1.0 - conflict[agreement]
    conflict_pixels = agreement & conflict.gt(0.0)
    if not torch.isfinite(weight).all().item():
        raise FloatingPointError('Reference weight contains NaN/Inf')
    if not torch.all(weight[disagreement].eq(1.0)).item():
        raise AssertionError('Reference changed a disagreement pixel weight')
    if not torch.all(weight[agreement & signed_support.ge(0.0)].eq(1.0)).item():
        raise AssertionError('Reference changed an agreement pixel with s>=0')
    contradicted = agreement & signed_support.lt(0.0)
    if contradicted.any().item():
        contradicted_weight = weight[contradicted]
        if not torch.all(
            (contradicted_weight >= 0.0) & (contradicted_weight <= 1.0)
        ).item():
            raise AssertionError('Reference contradicted weight escaped [0,1]')
        # For a negative support whose magnitude is below float32 resolution,
        # the mathematically strict ``1 - r < 1`` can round back to 1.  Keep
        # the exact prescribed formula and enforce strict attenuation only
        # where the computed float32 conflict is representable at unit scale.
        representable = conflict[contradicted].gt(torch.finfo(weight.dtype).eps)
        if representable.any().item() and not torch.all(
            contradicted_weight[representable].lt(1.0)
        ).item():
            raise AssertionError('Reference contradiction was not attenuated')
    return {
        'pseudo_a': pseudo_a,
        'pseudo_b': pseudo_b,
        'agreement': agreement,
        'disagreement': disagreement,
        'fg_consensus': consensus_fg,
        'bg_consensus': consensus_bg,
        'signed_support': signed_support,
        'conflict': conflict,
        'conflict_pixels_mask': conflict_pixels,
        'fg_conflict_mask': conflict_pixels & consensus_fg,
        'bg_conflict_mask': conflict_pixels & consensus_bg,
        'weight': weight,
    }


def _build_reference_weights(logits_a, logits_b, rgb_input, context, cfg, args):
    """Compute immutable UNI evidence and reference conflict weights."""
    if logits_a.shape[1] != 2 or logits_b.shape[1] != 2:
        raise RuntimeError('UNI reference weighting requires two-class CPS logits')
    pseudo_a = logits_a.detach().argmax(dim=1)
    pseudo_b = logits_b.detach().argmax(dim=1)
    original_pseudo_a = pseudo_a.clone()
    original_pseudo_b = pseudo_b.clone()
    with torch.inference_mode():
        tokens = _uni_dense_features(
            context['uni'], _reconstruct_rgb(rgb_input, cfg)
        )
        actual_grid = int(math.sqrt(int(tokens.shape[1])))
        if actual_grid != context['token_grid']:
            raise RuntimeError('UNI token grid changed during training')
        margin_tokens = _reference_margin(tokens, context, args.uni_k)
        margin = F.interpolate(
            margin_tokens.reshape(-1, 1, actual_grid, actual_grid),
            size=logits_a.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )[:, 0]
        result = _reference_weights_from_margin(pseudo_a, pseudo_b, margin)
    if not torch.equal(result['pseudo_a'], original_pseudo_a):
        raise AssertionError('Reference modified pseudo_A')
    if not torch.equal(result['pseudo_b'], original_pseudo_b):
        raise AssertionError('Reference modified pseudo_B')
    result['margin'] = margin
    return result


def compute_soft_correction_cps_losses(ce_loss, prediction1, prediction2, targets):
    """Continuous target-mass transfer on agreement, original CPS on D.

    Each direction first forms one pixel loss,
    ``(1-r)*CE(current) + r*CE(opposite)`` on
    agreement and the original peer-target CE on disagreement, then applies
    the same 0.5 bidirectional mean reduction as the original CPS.
    """
    # _build_reference_weights runs UNI/reference work under inference_mode(). Clone
    # these target-side tensors before they participate in an autograd graph;
    # inference tensors cannot be saved for backward by the composed CE loss.
    pseudo1 = targets['pseudo_a'].detach().clone()
    pseudo2 = targets['pseudo_b'].detach().clone()
    agreement = targets['agreement'].detach().clone()
    conflict = targets['conflict'].detach().clone()

    def corrected_direction(prediction, target):
        current = F.cross_entropy(
            prediction, target, weight=ce_loss.weight,
            ignore_index=ce_loss.ignore_index, reduction='none',
        )
        opposite_target = 1 - target
        opposite = F.cross_entropy(
            prediction, opposite_target, weight=ce_loss.weight,
            ignore_index=ce_loss.ignore_index, reduction='none',
        )
        pixel_loss = torch.where(
            agreement,
            (1.0 - conflict) * current + conflict * opposite,
            current,
        )
        valid = target.ne(ce_loss.ignore_index)
        return 0.5 * pixel_loss[valid].mean(), pixel_loss, current, opposite

    loss_1_to_2, pixel_1_to_2, current_1_to_2, opposite_1_to_2 = corrected_direction(
        prediction2, pseudo1
    )
    loss_2_to_1, pixel_2_to_1, current_2_to_1, opposite_2_to_1 = corrected_direction(
        prediction1, pseudo2
    )
    total = loss_1_to_2 + loss_2_to_1
    if not torch.isfinite(total).item():
        raise FloatingPointError('Soft Correction CPS loss contains NaN/Inf')
    return {
        '1_to_2': loss_1_to_2,
        '2_to_1': loss_2_to_1,
        'total': total,
        'pixel_loss_1_to_2': pixel_1_to_2.detach(),
        'pixel_loss_2_to_1': pixel_2_to_1.detach(),
        'current_loss_1_to_2': current_1_to_2.detach(),
        'current_loss_2_to_1': current_2_to_1.detach(),
        'opposite_loss_1_to_2': opposite_1_to_2.detach(),
        'opposite_loss_2_to_1': opposite_2_to_1.detach(),
    }


def _sc_batch_stats(targets, logits_a, logits_b):
    """Training-only SC aggregates; no GT-dependent terminology is used."""
    agreement = targets['agreement']
    fg = targets['fg_consensus']
    bg = targets['bg_consensus']
    conflict_mask = targets['conflict_pixels_mask']
    conflict = targets['conflict']
    signed = targets['signed_support']
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    pseudo = targets['pseudo_a']
    popp_a = torch.where(pseudo.eq(1), 1.0 - prob_a, prob_a)
    popp_b = torch.where(pseudo.eq(1), 1.0 - prob_b, prob_b)
    potential_a = agreement & (conflict > popp_a)
    potential_b = agreement & (conflict > popp_b)
    return {
        'unlabeled_pixels': int(agreement.numel()),
        'agreement_pixels': int(agreement.sum().item()),
        'disagreement_pixels': int(targets['disagreement'].sum().item()),
        'fg_consensus_pixels': int(fg.sum().item()),
        'bg_consensus_pixels': int(bg.sum().item()),
        'conflict_pixels': int(conflict_mask.sum().item()),
        'fg_conflict_pixels': int(targets['fg_conflict_mask'].sum().item()),
        'bg_conflict_pixels': int(targets['bg_conflict_mask'].sum().item()),
        'signed_support_sum_agreement': float(signed[agreement].sum().item()),
        'r_sum_agreement': float(conflict[agreement].sum().item()),
        'r_sum_conflict': float(conflict[conflict_mask].sum().item()),
        'r_sum_fg': float(conflict[fg].sum().item()),
        'r_sum_bg': float(conflict[bg].sum().item()),
        'potential_reverse_A': int(potential_a.sum().item()),
        'potential_reverse_B': int(potential_b.sum().item()),
        'potential_reverse_any': int((potential_a | potential_b).sum().item()),
        'potential_reverse_both': int((potential_a & potential_b).sum().item()),
    }


def _sc_new_epoch_totals(learning_rate):
    return {
        'unlabeled_pixels': 0, 'agreement_pixels': 0,
        'disagreement_pixels': 0, 'fg_consensus_pixels': 0,
        'bg_consensus_pixels': 0, 'conflict_pixels': 0,
        'fg_conflict_pixels': 0, 'bg_conflict_pixels': 0,
        'signed_support_sum_agreement': 0.0, 'r_sum_agreement': 0.0,
        'r_sum_conflict': 0.0, 'r_sum_fg': 0.0, 'r_sum_bg': 0.0,
        'potential_reverse_A': 0, 'potential_reverse_B': 0,
        'potential_reverse_any': 0, 'potential_reverse_both': 0,
        'original_loss_pixel_sum': 0.0, 'sc_loss_pixel_sum': 0.0,
        'supervised_pixels': 0, 'supervised_loss_A_pixel_sum': 0.0,
        'supervised_loss_B_pixel_sum': 0.0, 'total_loss_pixel_sum': 0.0,
        'learning_rate': float(learning_rate),
    }


def _sc_update_epoch_totals(
    totals, stats, original_loss, sc_losses,
    supervised_loss1, supervised_loss2, total_loss, supervised_pixels,
):
    for key in (
        'unlabeled_pixels', 'agreement_pixels', 'disagreement_pixels',
        'fg_consensus_pixels', 'bg_consensus_pixels', 'conflict_pixels',
        'fg_conflict_pixels', 'bg_conflict_pixels',
        'potential_reverse_A', 'potential_reverse_B',
        'potential_reverse_any', 'potential_reverse_both',
    ):
        totals[key] += stats[key]
    for key in (
        'signed_support_sum_agreement', 'r_sum_agreement', 'r_sum_conflict',
        'r_sum_fg', 'r_sum_bg',
    ):
        totals[key] += stats[key]
    totals['original_loss_pixel_sum'] += float(original_loss.item()) * stats['unlabeled_pixels']
    totals['sc_loss_pixel_sum'] += float(sc_losses['total'].item()) * stats['unlabeled_pixels']
    totals['supervised_pixels'] += int(supervised_pixels)
    totals['supervised_loss_A_pixel_sum'] += float(supervised_loss1.item()) * supervised_pixels
    totals['supervised_loss_B_pixel_sum'] += float(supervised_loss2.item()) * supervised_pixels
    totals['total_loss_pixel_sum'] += float(total_loss.item()) * supervised_pixels


def _sc_epoch_row(epoch, totals):
    pixels = totals['unlabeled_pixels']
    agreement = totals['agreement_pixels']
    conflict = totals['conflict_pixels']
    return {
        'epoch': epoch, 'total_unlabeled_pixels': pixels,
        'agreement_pixels': agreement, 'agreement_rate': _safe_mean(agreement, pixels),
        'disagreement_pixels': totals['disagreement_pixels'],
        'disagreement_rate': _safe_mean(totals['disagreement_pixels'], pixels),
        'fg_consensus_pixels': totals['fg_consensus_pixels'],
        'bg_consensus_pixels': totals['bg_consensus_pixels'],
        'reference_conflict_pixels': conflict,
        'reference_conflict_rate': _safe_mean(conflict, pixels),
        'fg_conflict_pixels': totals['fg_conflict_pixels'],
        'fg_conflict_rate': _safe_mean(totals['fg_conflict_pixels'], totals['fg_consensus_pixels']),
        'bg_conflict_pixels': totals['bg_conflict_pixels'],
        'bg_conflict_rate': _safe_mean(totals['bg_conflict_pixels'], totals['bg_consensus_pixels']),
        'mean_signed_support_agreement': _safe_mean(totals['signed_support_sum_agreement'], agreement),
        'mean_r_agreement': _safe_mean(totals['r_sum_agreement'], agreement),
        'mean_r_conflict': _safe_mean(totals['r_sum_conflict'], conflict),
        'mean_r_fg': _safe_mean(totals['r_sum_fg'], totals['fg_consensus_pixels']),
        'mean_r_bg': _safe_mean(totals['r_sum_bg'], totals['bg_consensus_pixels']),
        'sum_reference_target_mass_shift': totals['r_sum_agreement'],
        'mean_reference_target_mass_shift': _safe_mean(totals['r_sum_agreement'], agreement),
        'potential_reverse_A_rate': _safe_mean(totals['potential_reverse_A'], agreement),
        'potential_reverse_B_rate': _safe_mean(totals['potential_reverse_B'], agreement),
        'potential_reverse_any_rate': _safe_mean(totals['potential_reverse_any'], agreement),
        'potential_reverse_both_rate': _safe_mean(totals['potential_reverse_both'], agreement),
        'original_cps_loss': _safe_mean(totals['original_loss_pixel_sum'], pixels),
        'soft_correction_cps_loss': _safe_mean(totals['sc_loss_pixel_sum'], pixels),
        'cps_loss_delta': _safe_mean(totals['sc_loss_pixel_sum'], pixels) - _safe_mean(totals['original_loss_pixel_sum'], pixels),
        'supervised_loss_A': _safe_mean(totals['supervised_loss_A_pixel_sum'], totals['supervised_pixels']),
        'supervised_loss_B': _safe_mean(totals['supervised_loss_B_pixel_sum'], totals['supervised_pixels']),
        'total_loss': _safe_mean(totals['total_loss_pixel_sum'], totals['supervised_pixels']),
        'learning_rate': totals['learning_rate'],
    }


def _sc_iter_row(epoch, iteration, global_step, stats, original_loss, sc_loss):
    agreement = stats['agreement_pixels']
    return {
        'iteration': iteration, 'global_step': global_step, 'epoch': epoch,
        'agreement_rate': _safe_mean(agreement, stats['unlabeled_pixels']),
        'conflict_rate': _safe_mean(stats['conflict_pixels'], stats['unlabeled_pixels']),
        'mean_r': _safe_mean(stats['r_sum_agreement'], agreement),
        'potential_reverse_A_rate': _safe_mean(stats['potential_reverse_A'], agreement),
        'potential_reverse_B_rate': _safe_mean(stats['potential_reverse_B'], agreement),
        'original_cps_loss': float(original_loss.item()),
        'soft_correction_cps_loss': float(sc_loss.item()),
        'loss_delta': float(sc_loss.item() - original_loss.item()),
    }


def _run_sc_startup_checks(args, cfg, context, models, optimizers, losses):
    """Static, dataset-agnostic SC protocol and algebra checks."""
    exact = {
        'seed': (args.sd, 24),
        'batch_size': (args.batch_size, 2), 'num_epochs': (args.num_epochs, 200),
        'network': (args.network, 'unet'), 'input_mode': (args.input_mode, 'rgb_stain'),
        'step_size': (args.step_size, 50), 'lr': (args.lr, 0.5),
        'gamma': (args.gamma, 0.5), 'unsup_weight': (args.unsup_weight, 5.0),
        'warm_up_duration': (args.warm_up_duration, 20),
        'milestone_interval': (args.milestone_interval, 10),
    }
    mismatch = ['{}={} expected {}'.format(k, a, b) for k, (a, b) in exact.items() if a != b]
    if mismatch:
        raise RuntimeError('SC matched-protocol assertion failed: ' + '; '.join(mismatch))
    if context['weights_sha256'] != EXPECTED_UNI_SHA256:
        raise AssertionError('SC UNI SHA256 mismatch')
    if context['memory_fg_count'] <= 0 or context['memory_bg_count'] <= 0:
        raise AssertionError('SC reference memory is empty')
    if (context['feature_dim'], context['token_grid'], context['memory_top_k'], args.uni_k) != (1024, 14, 5, 5):
        raise AssertionError('SC reference geometry/K mismatch')
    if max(context['memory_fg_norm_error'], context['memory_bg_norm_error']) > 1e-4:
        raise AssertionError('SC reference memory is not L2 normalized')
    if context['hard_reference_threshold_used']:
        raise AssertionError('SC must not use hard reference thresholds')
    if any(p.requires_grad for p in context['uni'].parameters()):
        raise AssertionError('SC UNI parameters are not frozen')
    optimizer_ids = {id(p) for opt in optimizers for group in opt.param_groups for p in group['params']}
    if any(id(p) in optimizer_ids for p in context['uni'].parameters()):
        raise AssertionError('SC UNI entered optimizer')

    # 4-D CE logits require a 3-D spatial target: [N,H,W].
    pa = torch.tensor([[[0, 0, 1, 1, 0]]], device='cuda', dtype=torch.long)
    pb = torch.tensor([[[0, 1, 1, 1, 1]]], device='cuda', dtype=torch.long)
    margin = torch.tensor([[[0.2, -0.3, 0.4, -0.6, 0.1]]], device='cuda')
    original_a, original_b = pa.clone(), pb.clone()
    targets = _reference_weights_from_margin(pa, pb, margin)
    assert torch.equal(pa, original_a) and torch.equal(pb, original_b)
    r = targets['conflict']
    assert torch.all((r >= 0.0) & (r <= 1.0))
    assert torch.allclose(r[targets['agreement'] & targets['signed_support'].ge(0.0)], torch.zeros_like(r[targets['agreement'] & targets['signed_support'].ge(0.0)]))
    assert torch.allclose((1.0 - r) + r, torch.ones_like(r))

    logits_a = torch.tensor([[[[1.0, -0.2, 0.4, 0.1, 0.3]], [[-0.1, 0.4, 0.2, 0.6, -0.3]]]], device='cuda', requires_grad=True)
    logits_b = torch.tensor([[[[0.1, 0.8, -0.2, 0.7, 0.2]], [[0.6, -0.3, 0.5, -0.4, 0.9]]]], device='cuda', requires_grad=True)
    standard = compute_standard_cps_losses(losses['ce'], logits_a, logits_b, pa, pb)
    zero_targets = _reference_weights_from_margin(pa, pb, torch.zeros_like(margin))
    sc_zero = compute_soft_correction_cps_losses(losses['ce'], logits_a, logits_b, zero_targets)
    assert torch.allclose(standard['total'], sc_zero['total'], atol=1e-7, rtol=1e-7)
    sc = compute_soft_correction_cps_losses(losses['ce'], logits_a, logits_b, targets)
    assert torch.isfinite(sc['total']).item()
    assert torch.equal(targets['pseudo_a'], pa) and torch.equal(targets['pseudo_b'], pb)
    disagreement = targets['disagreement']
    assert torch.allclose(sc['pixel_loss_1_to_2'][disagreement], sc['current_loss_1_to_2'][disagreement])
    assert torch.allclose(sc['pixel_loss_2_to_1'][disagreement], sc['current_loss_2_to_1'][disagreement])
    if not all(model.training for model in models):
        raise AssertionError('Student mode changed before SC training')


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
    sc_iter_writer=None,
    sc_iter_file=None,
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
    sc_totals = (
        _sc_new_epoch_totals(optimizer1.param_groups[0]['lr'])
        if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction'
        else None
    )
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
        
        if args.cps_mode == 'uni_reference_soft_correction':
            sc_targets = _build_reference_weights(
                unsup_prediction1,
                unsup_prediction2,
                unsup_input1,
                reference_context,
                reference_context['cfg'],
                args,
            )
            with torch.no_grad():
                original_cps_losses = compute_standard_cps_losses(
                    losses['ce'],
                    unsup_prediction1,
                    unsup_prediction2,
                    sc_targets['pseudo_a'],
                    sc_targets['pseudo_b'],
                )
            cps_losses = compute_soft_correction_cps_losses(
                losses['ce'],
                unsup_prediction1,
                unsup_prediction2,
                sc_targets,
            )
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

        if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction':
            sc_stats = _sc_batch_stats(sc_targets, unsup_prediction1, unsup_prediction2)
            _sc_update_epoch_totals(
                sc_totals,
                sc_stats,
                original_cps_losses['total'],
                cps_losses,
                supervised_loss1,
                supervised_loss2,
                total_loss,
                int(sup_mask.numel()),
            )
            if (
                sc_iter_writer is not None
                and args.sc_log_interval > 0
                and global_step % args.sc_log_interval == 0
            ):
                sc_iter_writer.writerow(
                    _sc_iter_row(
                        epoch + 1,
                        batch_index + 1,
                        global_step,
                        sc_stats,
                        original_cps_losses['total'],
                        cps_losses['total'],
                    )
                )
                sc_iter_file.flush()

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
    reference_totals = (
        sc_totals
    )
    return (
        loss_totals,
        (score_list1, score_list2, mask_list),
        global_step,
        reference_totals,
    )


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
    test_evaluations,
    validation_diversity,
    best_val_eval_lists,
):
    val_eval1, val_eval2 = val_evaluations
    test_eval1, test_eval2 = test_evaluations
    optimizer1, _ = optimizers
    return {
        'epoch': epoch + 1,
        'learning_rate': optimizer1.param_groups[0]['lr'],
        'average_train_loss': train_losses[3],
        'val_jaccard_student1': val_eval1[1],
        'val_dice_student1': val_eval1[2],
        'val_jaccard_student2': val_eval2[1],
        'val_dice_student2': val_eval2[2],
        'test_jaccard_student1': test_eval1[1],
        'test_dice_student1': test_eval1[2],
        'test_jaccard_student2': test_eval2[1],
        'test_dice_student2': test_eval2[2],
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
        '| test_dice_student1=%.6f | test_dice_student2=%.6f '
        '| test_jaccard_student1=%.6f | test_jaccard_student2=%.6f '
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
        metrics['test_dice_student1'],
        metrics['test_dice_student2'],
        metrics['test_jaccard_student1'],
        metrics['test_jaccard_student2'],
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


def _evaluate_frozen_pair(
    models, losses, dataloaders, cfg, checkpoint_label, source_epochs,
):
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    validation = validate(models, losses, dataloaders['val'])
    with redirect_stdout(io.StringIO()):
        val_eval1, val_eval2, _, _ = print_val_eval(
            cfg['NUM_CLASSES'], validation['scores1'], validation['scores2'],
            validation['masks'], 30,
        )
    test = validate(models, losses, dataloaders['test'])
    rows = []
    for index, (branch, val_eval) in enumerate(
        (('A', val_eval1), ('B', val_eval2))
    ):
        val_logits = validation['scores1'] if index == 0 else validation['scores2']
        test_logits = test['scores1'] if index == 0 else test['scores2']
        val_fixed = _binary_metrics_at_threshold(
            val_logits, validation['masks'], 0.5, inclusive=True
        )
        test_from_val = _binary_metrics_at_threshold(
            test_logits, test['masks'], float(val_eval[0]), inclusive=False
        )
        test_fixed = _binary_metrics_at_threshold(
            test_logits, test['masks'], 0.5, inclusive=True
        )
        rows.extend([
            {
                'checkpoint': checkpoint_label, 'branch': branch,
                'source_epoch': int(source_epochs[index]), 'split': 'validation',
                'threshold': float(val_eval[0]),
                'threshold_source': 'validation_project_search',
                'jaccard': float(val_eval[1]), 'dice': float(val_eval[2]),
                'fixed05_jaccard': val_fixed['jaccard'],
                'fixed05_dice': val_fixed['dice'],
            },
            {
                'checkpoint': checkpoint_label, 'branch': branch,
                'source_epoch': int(source_epochs[index]), 'split': 'test',
                'threshold': float(val_eval[0]),
                'threshold_source': 'frozen_validation_threshold',
                'jaccard': test_from_val['jaccard'],
                'dice': test_from_val['dice'],
                'fixed05_jaccard': test_fixed['jaccard'],
                'fixed05_dice': test_fixed['dice'],
            },
        ])
    return rows


def _sc_validation_diagnostic(
    validation, dataloader, context, cfg, args, val_eval1, val_eval2,
):
    """Retrospective SC direction diagnostic; GT never feeds training."""
    keys = (
        'pixels', 'CC', 'WW', 'AC', 'BC', 'D', 'shared_FP', 'shared_FN',
        'hc90_WW', 'hc95_WW', 'sum_r_CC', 'sum_r_WW', 'conflict_CC',
        'conflict_WW', 'ww_rev_A', 'ww_rev_B', 'ww_any', 'ww_both',
        'cc_harm_A', 'cc_harm_B', 'cc_any', 'cc_both', 'ww_help_mass',
        'cc_harm_mass', 'fg_ww_count', 'fg_cc_count', 'fg_ww', 'fg_cc',
        'fg_help', 'fg_harm', 'bg_ww_count', 'bg_cc_count', 'bg_ww',
        'bg_cc', 'bg_help', 'bg_harm',
    )
    totals = {key: 0.0 for key in keys}
    offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch_size = int(batch['mask'].shape[0])
            logits_a = validation['scores1'][offset:offset + batch_size]
            logits_b = validation['scores2'][offset:offset + batch_size]
            offset += batch_size
            rgb_input = batch['image_rgb'].cuda(non_blocking=True)
            gt = batch['mask'].cuda(non_blocking=True).eq(1)
            targets = _build_reference_weights(logits_a, logits_b, rgb_input, context, cfg, args)
            pa = targets['pseudo_a'].eq(1)
            pb = targets['pseudo_b'].eq(1)
            agreement = targets['agreement']
            disagreement = targets['disagreement']
            cc = agreement & pa.eq(gt)
            ww = agreement & pa.ne(gt)
            ac = disagreement & pa.eq(gt) & pb.ne(gt)
            bc = disagreement & pa.ne(gt) & pb.eq(gt)
            if int((ac | bc).sum().item()) != int(disagreement.sum().item()):
                raise AssertionError('SC validation D != AC union BC')
            s = targets['signed_support']
            r = targets['conflict']
            prob_a = torch.softmax(logits_a, dim=1)[:, 1]
            prob_b = torch.softmax(logits_b, dim=1)[:, 1]
            popp_a = torch.where(pa, 1.0 - prob_a, prob_a)
            popp_b = torch.where(pa, 1.0 - prob_b, prob_b)
            rev_a = agreement & (r > popp_a)
            rev_b = agreement & (r > popp_b)
            shared_fp = ww & pa
            shared_fn = ww & ~pa
            fg = agreement & pa
            bg = agreement & ~pa
            for key, value in (
                ('pixels', gt.numel()), ('CC', cc.sum()), ('WW', ww.sum()),
                ('AC', ac.sum()), ('BC', bc.sum()), ('D', disagreement.sum()),
                ('shared_FP', shared_fp.sum()), ('shared_FN', shared_fn.sum()),
                ('hc90_WW', (ww & torch.minimum(torch.maximum(prob_a, 1.0 - prob_a), torch.maximum(prob_b, 1.0 - prob_b)).ge(0.90)).sum()),
                ('hc95_WW', (ww & torch.minimum(torch.maximum(prob_a, 1.0 - prob_a), torch.maximum(prob_b, 1.0 - prob_b)).ge(0.95)).sum()),
                ('sum_r_CC', r[cc].sum()), ('sum_r_WW', r[ww].sum()),
                ('conflict_CC', (cc & s.lt(0.0)).sum()), ('conflict_WW', (ww & s.lt(0.0)).sum()),
                ('ww_rev_A', (ww & rev_a).sum()), ('ww_rev_B', (ww & rev_b).sum()),
                ('ww_any', (ww & (rev_a | rev_b)).sum()), ('ww_both', (ww & rev_a & rev_b).sum()),
                ('cc_harm_A', (cc & rev_a).sum()), ('cc_harm_B', (cc & rev_b).sum()),
                ('cc_any', (cc & (rev_a | rev_b)).sum()), ('cc_both', (cc & rev_a & rev_b).sum()),
                ('ww_help_mass', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[ww].sum()),
                ('cc_harm_mass', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[cc].sum()),
                ('fg_ww_count', (ww & fg).sum()), ('fg_cc_count', (cc & fg).sum()),
                ('fg_ww', (ww & fg & (rev_a | rev_b)).sum()), ('fg_cc', (cc & fg & (rev_a | rev_b)).sum()),
                ('fg_help', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[ww & fg].sum()),
                ('fg_harm', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[cc & fg].sum()),
                ('bg_ww_count', (ww & bg).sum()), ('bg_cc_count', (cc & bg).sum()),
                ('bg_ww', (ww & bg & (rev_a | rev_b)).sum()), ('bg_cc', (cc & bg & (rev_a | rev_b)).sum()),
                ('bg_help', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[ww & bg].sum()),
                ('bg_harm', (torch.clamp(r - popp_a, min=0.0) + torch.clamp(r - popp_b, min=0.0))[cc & bg].sum()),
            ):
                totals[key] += float(value.item() if torch.is_tensor(value) else value)
    if offset != int(validation['masks'].shape[0]):
        raise AssertionError('SC validation loader/logit length mismatch')
    agreement = totals['CC'] + totals['WW']
    return {
        'valid_pixels': int(totals['pixels']), 'CC': int(totals['CC']),
        'WW': int(totals['WW']), 'AC': int(totals['AC']), 'BC': int(totals['BC']),
        'D': int(totals['D']), 'agreement_rate': _safe_mean(agreement, totals['pixels']),
        'disagreement_rate': _safe_mean(totals['D'], totals['pixels']),
        'absolute_WW_rate': _safe_mean(totals['WW'], totals['pixels']),
        'shared_FP': int(totals['shared_FP']), 'shared_FN': int(totals['shared_FN']),
        'HC90_WW_all': int(totals['hc90_WW']), 'HC95_WW_all': int(totals['hc95_WW']),
        'mean_r_CC': _safe_mean(totals['sum_r_CC'], totals['CC']),
        'mean_r_WW': _safe_mean(totals['sum_r_WW'], totals['WW']),
        'conflict_rate_CC': _safe_mean(totals['conflict_CC'], totals['CC']),
        'conflict_rate_WW': _safe_mean(totals['conflict_WW'], totals['WW']),
        'WW_recovery_rate_A': _safe_mean(totals['ww_rev_A'], totals['WW']),
        'WW_recovery_rate_B': _safe_mean(totals['ww_rev_B'], totals['WW']),
        'WW_any_reverse': _safe_mean(totals['ww_any'], totals['WW']),
        'WW_both_reverse': _safe_mean(totals['ww_both'], totals['WW']),
        'CC_harmful_reverse_A': _safe_mean(totals['cc_harm_A'], totals['CC']),
        'CC_harmful_reverse_B': _safe_mean(totals['cc_harm_B'], totals['CC']),
        'CC_any_harm': _safe_mean(totals['cc_any'], totals['CC']),
        'CC_both_harm': _safe_mean(totals['cc_both'], totals['CC']),
        'WW_helpful_reverse_mass': totals['ww_help_mass'],
        'CC_harmful_reverse_mass': totals['cc_harm_mass'],
        'FG_WW_recovery_rate': _safe_mean(totals['fg_ww'], totals['fg_ww_count']),
        'FG_CC_harmful_reverse_rate': _safe_mean(totals['fg_cc'], totals['fg_cc_count']),
        'FG_WW_helpful_mass': totals['fg_help'], 'FG_CC_harmful_mass': totals['fg_harm'],
        'BG_WW_recovery_rate': _safe_mean(totals['bg_ww'], totals['bg_ww_count']),
        'BG_CC_harmful_reverse_rate': _safe_mean(totals['bg_cc'], totals['bg_cc_count']),
        'BG_WW_helpful_mass': totals['bg_help'], 'BG_CC_harmful_mass': totals['bg_harm'],
        'val_dice_student1': float(val_eval1[2]), 'val_dice_student2': float(val_eval2[2]),
        'val_jaccard_student1': float(val_eval1[1]), 'val_jaccard_student2': float(val_eval2[1]),
    }


def _sc_checkpoint_metadata(args, context, best_val_eval_lists, best_epochs):
    return {
        'dataset_name': args.dataset_name, 'labeled_ratio': args.labeled_ratio,
        'seed': args.sd, 'input_mode': args.input_mode, 'network': args.network,
        'cps_mode': args.cps_mode, 'selection_split': 'validation',
        'selection_metric': 'jaccard', 'best_val_metrics': [list(v) for v in best_val_eval_lists],
        'best_epochs': list(best_epochs), 'reference_memory_path': context['memory_path'],
        'reference_memory_sha256': context['memory_sha256'], 'uni_sha256': context['weights_sha256'],
        'uni_weights': context['weights_path'], 'uni_k': args.uni_k,
        'margin_definition': 'S_fg - S_bg',
        'signed_support_definition': 's=(2*y_c-1)*m',
        'reference_conflict_definition': 'r=clamp(-s,0,1)',
        'soft_correction_definition': '(1-r)*CE(current)+r*CE(opposite) on agreement; original CE on disagreement',
        'softmax_ce_gradient_relation': 'g_SC = p_opp - r; r>p_opp yields opposite-direction corrective gradient',
        'loss_reduction': 'mean((1-r)*CE(current)+r*CE(opposite)) per direction; no sum(wL)/sum(w)',
        'hard_reference_threshold_used': False, 'student_confidence_used': False,
        'pseudo_label_flipping_used': False, 'test_evaluation_policy': 'final_frozen_only',
    }


def _write_sc_protocol_audit(path, args, cfg, datasets, context, optimizers, startup_passed):
    optimizer_parameter_ids = {id(p) for opt in optimizers for group in opt.param_groups for p in group['params']}
    audit = {
        'experiment': 'Reference-Conflict Soft Correction CPS', 'formal_training': True,
        'fresh_initialization': True, 'resume_checkpoint': None,
        'startup_assertions_passed': startup_passed, 'command': ' '.join(sys.argv),
        'dataset_name': args.dataset_name, 'path_dataset': str(Path(args.path_dataset).resolve()),
        'dataset_counts': {'labeled_unique': len(datasets['train_sup_unique']), 'unlabeled': len(datasets['train_unsup']), 'validation': len(datasets['val']), 'test': len(datasets['test'])},
        'config': {'labeled_ratio': args.labeled_ratio, 'seed': args.sd, 'batch_size': args.batch_size, 'num_epochs': args.num_epochs, 'network': args.network, 'input_mode': args.input_mode, 'cps_mode': args.cps_mode, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': 5 * 10 ** args.wd, 'step_size': args.step_size, 'gamma': args.gamma, 'unsup_weight': args.unsup_weight, 'warm_up_duration': args.warm_up_duration, 'milestone_interval': args.milestone_interval, 'best_save_interval': args.best_save_interval},
        'actual_cps_loss_criterion': 'torch.nn.CrossEntropyLoss, bidirectional hard-target CE; 0.5 each direction',
        'soft_correction': {'margin': 'S_fg-S_bg', 'signed_support': 's=(2*y_c-1)*m', 'conflict': 'r=clamp(-s,0,1)', 'formula': '(1-r)*CE(current)+r*CE(opposite) on agreement; original CE on disagreement', 'loss_reduction': 'pixel mean after composition', 'softmax_ce_gradient_relation': 'g_SC=p_opp-r', 'hard_reference_threshold_used': False, 'student_confidence_used': False, 'pseudo_label_flipping_used': False},
        'UNI': {'architecture': 'ViT-L/16', 'feature_dim': 1024, 'dense_token_grid': [14, 14], 'sha256': context['weights_sha256'], 'path': context['weights_path'], 'frozen': True, 'eval': not context['uni'].training, 'requires_grad': any(p.requires_grad for p in context['uni'].parameters()), 'in_optimizer': any(id(p) in optimizer_parameter_ids for p in context['uni'].parameters())},
        'reference_memory': {'path': context['memory_path'], 'sha256': context['memory_sha256'], 'foreground_count': context['memory_fg_count'], 'background_count': context['memory_bg_count'], 'feature_dim': context['feature_dim'], 'token_grid': context['token_grid'], 'K': context['memory_top_k'], 'built_before_optimizer': bool(context.get('memory_built', False)), 'online_updates': False, 'source': 'current labeled split GT only'},
        'test_policy': 'not evaluated during training; frozen evaluation only after epoch200',
        'code': {'main_path': str(Path(__file__).resolve()), 'main_sha256': _sha256_file(Path(__file__).resolve()), 'torch': torch.__version__, 'numpy': np.__version__, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)


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
    if args.cps_mode == 'uni_reference_soft_correction':
        existing_training_artifacts = [
            path for path in Path(trained_models_path).rglob('*')
            if path.is_file()
        ]
        if existing_training_artifacts:
            raise RuntimeError(
                'Reference formal output is not empty; refusing resume/overwrite: {}'.format(
                    trained_models_path
                )
            )
    logger, metrics_path, diagnostic_logger = create_run_outputs(
        args, trained_models_path
    )
    reference_context = None
    sc_epoch_writer = None
    sc_metrics_file = None
    sc_validation_writer = None
    sc_validation_file = None
    sc_iter_writer = None
    sc_iter_file = None
    if args.cps_mode == 'uni_reference_soft_correction':
        reference_context = _load_reference(
            args,
            cfg,
            logger,
            manifest_dir=manifest_dir,
            use_manifests=use_manifests,
        )
        reference_context['cfg'] = cfg

    (
        models,
        optimizers,
        schedulers,
        losses,
    ) = build_training_components(args, cfg)

    model1, model2 = models
    if reference_context is not None:
        _run_sc_startup_checks(
            args, cfg, reference_context, models, optimizers, losses
        )
        logger.info(
            'Soft Correction startup consistency checks passed | '
            'hard_reference_threshold_used=false | student_confidence_used=false '
            '| loss=(1-r)*CE(current)+r*CE(opposite) on agreement; original CE on disagreement'
        )
        _write_sc_protocol_audit(
            os.path.join(trained_models_path, 'soft_correction_protocol_audit.json'),
            args, cfg, datasets, reference_context, optimizers, True,
        )
        logger.info(
            'Soft Correction fresh initialization confirmed | resume_checkpoint=None '
            '| checkpoint_root=%s | segmentation_root=%s',
            trained_models_path, segmentation_results_path,
        )
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

    with ExitStack() as stack:
        metrics_file = stack.enter_context(
            open(metrics_path, 'w', newline='', encoding='utf-8')
        )
        if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction':
            sc_metrics_path = os.path.join(
                trained_models_path, 'soft_correction_reference_metrics.csv'
            )
            sc_validation_path = os.path.join(
                trained_models_path, 'soft_correction_validation_diagnostic.csv'
            )
            sc_metrics_file = stack.enter_context(
                open(sc_metrics_path, 'w', newline='', encoding='utf-8')
            )
            sc_validation_file = stack.enter_context(
                open(sc_validation_path, 'w', newline='', encoding='utf-8')
            )
            sc_epoch_writer = csv.DictWriter(
                sc_metrics_file, fieldnames=SC_METRIC_FIELDS
            )
            sc_epoch_writer.writeheader()
            sc_validation_writer = csv.DictWriter(
                sc_validation_file, fieldnames=SC_VALIDATION_FIELDS
            )
            sc_validation_writer.writeheader()
            if args.sc_log_interval > 0:
                sc_iter_path = os.path.join(
                    trained_models_path, 'soft_correction_reference_metrics_iter.csv'
                )
                sc_iter_file = stack.enter_context(
                    open(sc_iter_path, 'w', newline='', encoding='utf-8')
                )
                sc_iter_writer = csv.DictWriter(
                    sc_iter_file, fieldnames=SC_ITER_METRIC_FIELDS
                )
                sc_iter_writer.writeheader()
        metrics_writer = csv.DictWriter(
            metrics_file, fieldnames=METRIC_FIELDS
        )
        metrics_writer.writeheader()

        for epoch in range(args.num_epochs):
            epoch_start = time.time()
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
                sc_iter_writer=sc_iter_writer,
                sc_iter_file=sc_iter_file,
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
            if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction':
                sc_validation_row = _sc_validation_diagnostic(
                    validation, dataloaders['val'], reference_context, cfg, args,
                    val_eval1, val_eval2,
                )
                sc_validation_row['epoch'] = epoch + 1
                sc_validation_writer.writerow(sc_validation_row)
                sc_validation_file.flush()

            if args.cps_mode == 'uni_reference_soft_correction':
                # Reference-aware training inspects test only after all epochs.
                test_evaluation = None
                test_eval1 = [float('nan')] * 4
                test_eval2 = [float('nan')] * 4
            else:
                test_evaluation = validate(models, losses, dataloaders['test'])
                if print_epoch:
                    print_val_loss(
                        test_evaluation['loss1'],
                        test_evaluation['loss2'],
                        num_batches,
                        print_num,
                        print_num_half,
                        split_name='Test',
                        loader_name='test',
                    )
                    test_eval1, test_eval2, _, _ = print_val_eval(
                        cfg['NUM_CLASSES'],
                        test_evaluation['scores1'],
                        test_evaluation['scores2'],
                        test_evaluation['masks'],
                        print_num_half,
                        split_name='Test',
                    )
                else:
                    with redirect_stdout(io.StringIO()):
                        test_eval1, test_eval2, _, _ = print_val_eval(
                            cfg['NUM_CLASSES'],
                            test_evaluation['scores1'],
                            test_evaluation['scores2'],
                            test_evaluation['masks'],
                            print_num_half,
                            split_name='Test',
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
                    'selection_split': 'validation',
                    'selection_metric': 'jaccard',
                    'current_test_metrics': (
                        None if args.cps_mode == 'uni_reference_soft_correction'
                        else [list(test_eval1), list(test_eval2)]
                    ),
                    'test_evaluation_policy': (
                        'final_frozen_only' if args.cps_mode == 'uni_reference_soft_correction'
                        else 'monitoring_only_not_used_for_selection'
                    ),
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
                if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction':
                    milestone_metadata.update({
                        'reference_memory_path': reference_context['memory_path'],
                        'reference_memory_sha256': reference_context['memory_sha256'],
                        'uni_sha256': reference_context['weights_sha256'],
                        'uni_weights': reference_context['weights_path'],
                        'uni_k': args.uni_k,
                        'margin_definition': 'S_fg - S_bg',
                        'signed_support_definition': 's=(2*y_c-1)*m',
                        'reference_conflict_definition': 'r=clamp(-s,0,1)',
                        'soft_correction_definition': '(1-r)*CE(current)+r*CE(opposite) on agreement; original CE on disagreement',
                        'softmax_ce_gradient_relation': 'g_SC=p_opp-r',
                        'loss_reduction': 'mean composed pixel loss per direction; no sum(wL)/sum(w)',
                        'hard_reference_threshold_used': False,
                        'student_confidence_used': False,
                        'pseudo_label_flipping_used': False,
                    })
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
                (test_eval1, test_eval2),
                validation_diversity,
                best_val_eval_lists,
            )
            _log_epoch(logger, args, metrics)
            if reference_context is not None and args.cps_mode == 'uni_reference_soft_correction':
                sc_epoch_writer.writerow(_sc_epoch_row(epoch + 1, reference_totals))
                sc_metrics_file.flush()

    sc_final_rows = None
    sc_last_path = None
    if args.cps_mode == 'uni_reference_soft_correction':
        epoch200_states = tuple(_model_state_snapshot(model) for model in models)
        sc_metadata = _sc_checkpoint_metadata(
            args, reference_context, best_val_eval_lists, best_epochs
        )
        sc_last_path = os.path.join(trained_models_path, 'last_checkpoint.pth')
        torch.save({
            'epoch': args.num_epochs,
            'appearance_state_dict': epoch200_states[0],
            'structure_state_dict': epoch200_states[1],
            **sc_metadata,
        }, sc_last_path)
        sc_final_rows = _evaluate_frozen_pair(
            models, losses, dataloaders, cfg, 'epoch200',
            (args.num_epochs, args.num_epochs),
        )
        model1.load_state_dict(best_model_states[0], strict=True)
        model2.load_state_dict(best_model_states[1], strict=True)
        sc_final_rows.extend(
            _evaluate_frozen_pair(
                models, losses, dataloaders, cfg, 'best_validation', best_epochs,
            )
        )
        model1.load_state_dict(epoch200_states[0], strict=True)
        model2.load_state_dict(epoch200_states[1], strict=True)
        final_path = os.path.join(
            trained_models_path, 'soft_correction_final_frozen_evaluation.csv'
        )
        with open(final_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    'checkpoint', 'branch', 'source_epoch', 'split',
                    'threshold', 'threshold_source', 'jaccard', 'dice',
                    'fixed05_jaccard', 'fixed05_dice',
                ],
            )
            writer.writeheader()
            writer.writerows(sc_final_rows)
        logger.info(
            'Soft Correction final frozen evaluation complete | test inspected '
            'only now after epoch 200 | path=%s', final_path,
        )

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
    if args.cps_mode == 'uni_reference_soft_correction':
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
            'best_validation_metrics': [
                list(values) for values in best_val_eval_lists
            ],
            'best_checkpoint_A': best_paths[0],
            'best_checkpoint_B': best_paths[1],
            'epoch200_checkpoint': str(
                Path(trained_models_path) / 'milestones' / 'epoch_200.pth'
            ),
            'last_checkpoint': sc_last_path,
            'final_frozen_evaluation': sc_final_rows,
            'test_policy': 'final_frozen_only_after_epoch200',
            'training_elapsed_seconds_including_final_evaluation': elapsed,
        }
        completion_name = 'soft_correction_training_complete.json'
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
        '--cps_mode', '--cps-mode',
        dest='cps_mode',
        default='standard',
        choices=['standard', 'uni_reference_soft_correction'],
        help=(
            'standard uses the original bidirectional CPS; '
            'uni_reference_soft_correction transfers r target mass to the opposite '
            'class on agreement without hard flipping.'
        ),
    )
    parser.add_argument(
        '--uni_reference_memory',
        default=None,
        help=(
            'reference_memory.npz used by UNI reference modes; when omitted, '
            'checkpoints/<dataset>_labeled<ratio>_seed<seed>/reference_memory.npz '
            'is selected automatically'
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
    parser.add_argument('--uni_k', default=5, type=int)
    parser.add_argument(
        '--sc_log_interval',
        default=50,
        type=int,
        help='write one Soft Correction iteration row every N optimizer steps; 0 disables it',
    )
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay exponent')
    parser.add_argument('--sd', default=4, type=int, help='random seed')
    parser.add_argument('-n', '--network', default='unet')

    args = parser.parse_args()
    if args.uni_reference_memory is None:
        args.uni_reference_memory = str(
            _default_reference_memory_path(
                args.dataset_name, args.labeled_ratio, args.sd
            )
        )
    if args.milestone_interval < 0:
        parser.error('--milestone_interval must be non-negative')
    if args.best_save_interval < 0:
        parser.error('--best_save_interval must be non-negative')
    if args.sc_log_interval < 0:
        parser.error('--sc_log_interval must be non-negative')
    if args.cps_mode == 'uni_reference_soft_correction':
        if args.uni_k != 5:
            parser.error('--uni_k must remain 5 for Soft Correction')
    if args.path_trained_models is None:
        if args.cps_mode == 'uni_reference_soft_correction':
            args.path_trained_models = (
                'checkpoints/semi_{}_seed_{}_rgbstain_uni_reference_soft_correction'.format(
                    args.labeled_ratio, args.sd
                )
            )
        else:
            args.path_trained_models = 'checkpoints/semi_fdcl_paper_{}'.format(
                args.labeled_ratio
            )
    if args.path_seg_results is None:
        if args.cps_mode == 'uni_reference_soft_correction':
            args.path_seg_results = (
                'seg_pred/semi_{}_seed_{}_rgbstain_uni_reference_soft_correction'.format(
                    args.labeled_ratio, args.sd
                )
            )
        else:
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
    print_data_summary(args, datasets, manifest_dir, use_manifests)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'use --check_data to validate input only.'
        )

    train_model(args, cfg, datasets, manifest_dir, use_manifests)


if __name__ == '__main__':
    main()
