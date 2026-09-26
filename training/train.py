"""Training, validation, metrics, and checkpoint selection."""

import csv
import io
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

from config.train_test_config.train_test_config import (
    print_and_save_branch_bests, print_train_eval, print_train_loss,
    print_val_eval, print_val_loss, save_periodic_branch_bests,
    save_training_milestone, save_val_best_2d,
)
from config.warmup_config.warmup import GradualWarmupScheduler
from loss.loss_function import DiceLoss
from models.getnetwork import get_network
from utils.training_diagnostics import (
    compute_validation_diversity,
    finalize_diagnostics as _finalize_diagnostics,
    finalize_motivation_metrics as _finalize_motivation_metrics,
    foreground_probability as _foreground_probability,
    init_diagnostic_totals as _init_diagnostic_totals,
    init_motivation_totals as _init_motivation_totals,
    log_diagnostics,
    select_feature_module as _select_feature_module,
    update_diagnostic_totals as _update_diagnostic_totals,
    update_motivation_totals as _update_motivation_totals,
)
from utils.experiment_logging import (
    METRIC_FIELDS, MOTIVATION_METRIC_FIELDS, NATIVE_TRAINING_FIELDS,
    build_output_paths, create_run_outputs, log_epoch as _log_epoch,
)
from utils.native_reference import (
    MARGIN_CALIBRATED_MODES, NATIVE_MODE_TAGS,
    _is_margin_calibrated_mode, _is_native_reference_mode,
    _load_reference, _reconstruct_rgb, _reference_scores,
    _uni_dense_features, _uses_uni_reference,
)
from training.cps import (
    _pixel_entropy, compute_cps_losses,
    compute_supervised_loss, get_training_views,
)


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

def _get_rgb_view(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the normalized RGB view exposed by any supported input mode."""
    return batch.get('image_rgb', batch['image'])

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
        print('Extra hyperparameters: none')

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
            losses['dice'],
            sup_prediction1,
            sup_mask,
        )
        supervised_loss2 = compute_supervised_loss(
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
        reference_targets = None
        native_cps = None
        if _is_native_reference_mode(args):
            reference_targets = _build_native_reference_targets(
                unsup_prediction1, unsup_prediction2, unsup_input1,
                reference_context, reference_context['cfg'], args,
            )
        cps_losses = compute_cps_losses(
            losses['ce'],
            unsup_prediction1,
            unsup_prediction2,
            cps_mode=args.cps_mode,
            ce_none=losses.get('ce_none'),
            reference_targets=reference_targets,
        )
        if _is_native_reference_mode(args):
            native_cps = cps_losses
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
                losses['dice'],
                output1,
                mask,
            ).item()
            loss2 += compute_supervised_loss(
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

def _model_state_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

def train_model(
    args,
    cfg,
    datasets,
    manifest_dir,
    use_manifests,
    dataloaders,
):
    """Run training, validation, and checkpoint selection."""
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

    models, optimizers, schedulers, losses = build_training_components(args, cfg)
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
