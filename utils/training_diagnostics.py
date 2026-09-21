"""Validation-time diagnostics that are not part of formal training.

The functions in this module only collect or summarize metrics.  They do not
update model parameters, influence the loss, or select checkpoints.
"""

import math
from typing import Dict

import numpy as np
import torch


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Return the foreground probability with a stable binary-model fallback."""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)[:, 1:2, ...]


def select_feature_module(model):
    """Select a useful feature layer for the optional CKA diagnostic."""
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
        if len(list(module.children())) == 0 and hasattr(module, 'register_forward_hook'):
            leaf_module = module
    return leaf_module


def linear_cka(feature_a: torch.Tensor, feature_b: torch.Tensor) -> float:
    """Compute linear CKA for two captured feature tensors."""
    if feature_a.dim() == 4:
        pooled_hw = (
            min(16, feature_a.shape[-2], feature_b.shape[-2]),
            min(16, feature_a.shape[-1], feature_b.shape[-1]),
        )
        feature_a = torch.nn.functional.adaptive_avg_pool2d(feature_a.float(), pooled_hw)
        feature_b = torch.nn.functional.adaptive_avg_pool2d(feature_b.float(), pooled_hw)
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


def init_diagnostic_totals():
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


def update_diagnostic_totals(
    totals,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
    feature_a=None,
    feature_b=None,
    ignore_index=-1,
):
    prob_a = foreground_probability(logits_a).detach()
    prob_b = foreground_probability(logits_b).detach()
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
        cka = linear_cka(feature_a.detach(), feature_b.detach())
        if not np.isnan(cka):
            totals['feature_cka_sum'] += cka
            totals['feature_cka_count'] += 1.0


def init_motivation_totals():
    """Initialize accumulators for the validation-only Vanilla CPS report."""
    return {
        'total_pixels': 0,
        'agreement_pixels': 0,
        'soft_disagreement_sum': 0.0,
        'ww_pixels': 0,
        'hc90ww_pixels': 0,
    }


def update_motivation_totals(
    totals,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
    ignore_index=-1,
):
    """Accumulate validation-only agreement and wrong-agreement metrics."""
    prob_a = foreground_probability(logits_a).detach()
    prob_b = foreground_probability(logits_b).detach()
    if prob_a.shape != prob_b.shape or prob_a.dim() != 4 or prob_a.shape[1] != 1:
        raise AssertionError(
            'Motivation probability shape mismatch: A={} B={}'.format(
                tuple(prob_a.shape), tuple(prob_b.shape)
            )
        )

    if mask.dim() == 4 and mask.shape[1] == 1:
        gt = mask[:, 0]
    elif mask.dim() == 3:
        gt = mask
    else:
        raise AssertionError(
            'Motivation GT must have shape [B,H,W] or [B,1,H,W], got {}'.format(
                tuple(mask.shape)
            )
        )

    prob_a = prob_a[:, 0]
    prob_b = prob_b[:, 0]
    if prob_a.shape != gt.shape:
        raise AssertionError(
            'Motivation prediction/GT shape mismatch: prediction={} GT={}'.format(
                tuple(prob_a.shape), tuple(gt.shape)
            )
        )

    valid = mask.ne(ignore_index)
    gt_binary = gt.gt(0.5)
    pred_a = prob_a.ge(0.5)
    pred_b = prob_b.ge(0.5)
    agreement = pred_a.eq(pred_b) & valid
    ww = agreement & pred_a.ne(gt_binary)
    confidence_a = torch.maximum(prob_a, 1.0 - prob_a)
    confidence_b = torch.maximum(prob_b, 1.0 - prob_b)
    hc90ww = ww & confidence_a.ge(0.90) & confidence_b.ge(0.90)

    totals['total_pixels'] += int(valid.sum().item())
    totals['agreement_pixels'] += int(agreement.sum().item())
    totals['soft_disagreement_sum'] += float(
        torch.abs(prob_a - prob_b)[valid].sum().item()
    )
    totals['ww_pixels'] += int(ww.sum().item())
    totals['hc90ww_pixels'] += int(hc90ww.sum().item())


def finalize_motivation_metrics(totals):
    """Finalize validation-only motivation metrics."""
    total_pixels = int(totals['total_pixels'])
    agreement_pixels = int(totals['agreement_pixels'])
    ww_pixels = int(totals['ww_pixels'])
    if total_pixels < 0 or agreement_pixels < 0 or ww_pixels < 0:
        raise AssertionError('Motivation counters cannot be negative')
    if agreement_pixels > total_pixels or ww_pixels > agreement_pixels:
        raise AssertionError('Motivation counter relationship is invalid')

    metrics = {
        'agreement': agreement_pixels / total_pixels if total_pixels else 0.0,
        'soft_disagreement': totals['soft_disagreement_sum'] / total_pixels if total_pixels else 0.0,
        'ww_given_agreement': ww_pixels / agreement_pixels if agreement_pixels else 0.0,
        'hc90ww_given_ww': totals['hc90ww_pixels'] / ww_pixels if ww_pixels else 0.0,
        'total_pixels': total_pixels,
        'agreement_pixels': agreement_pixels,
        'ww_pixels': ww_pixels,
        'hc90ww_pixels': int(totals['hc90ww_pixels']),
    }
    for key in ('agreement', 'soft_disagreement', 'ww_given_agreement', 'hc90ww_given_ww'):
        value = float(metrics[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise AssertionError(
                'Motivation metric {} is outside [0,1]: {}'.format(key, value)
            )
    return metrics


def finalize_diagnostics(totals, dice_a, dice_b):
    valid_pixels = totals['valid_pixels']
    disagree_pixels = totals['disagree_pixels']
    cka_count = totals['feature_cka_count']
    return {
        'dice_a': dice_a,
        'dice_b': dice_b,
        'pixel_disagreement': totals['pixel_disagreement'] / valid_pixels,
        'prob_disagreement': totals['prob_disagreement'] / valid_pixels,
        'feature_cka': totals['feature_cka_sum'] / cka_count if cka_count > 0 else float('nan'),
        'both_correct_ratio': totals['both_correct'] / valid_pixels,
        'a_only_correct_ratio': totals['a_only_correct'] / valid_pixels,
        'b_only_correct_ratio': totals['b_only_correct'] / valid_pixels,
        'both_wrong_ratio': totals['both_wrong'] / valid_pixels,
        'wrong_agreement_ratio': totals['wrong_agreement'] / valid_pixels,
        'a_correct_given_disagreement': totals['a_correct_disagree'] / disagree_pixels if disagree_pixels > 0 else float('nan'),
        'b_correct_given_disagreement': totals['b_correct_disagree'] / disagree_pixels if disagree_pixels > 0 else float('nan'),
    }


def log_diagnostics(diagnostic_logger, epoch, global_step, diagnostics):
    diagnostic_logger.info(
        '[Diagnostic] epoch=%d global_step=%d '
        'dice_a=%.4f dice_b=%.4f '
        'pixel_disagreement=%.4f prob_disagreement=%.4f '
        'feature_cka=%.4f both_correct_ratio=%.4f '
        'a_only_correct_ratio=%.4f b_only_correct_ratio=%.4f '
        'both_wrong_ratio=%.4f wrong_agreement_ratio=%.4f '
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


def _hard_predictions(scores: torch.Tensor) -> torch.Tensor:
    return scores.detach().argmax(dim=1)


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
    result = {
        'student12_disagreement': 1.0 - diversity['dice'],
        'student12_dice_similarity': diversity['dice'],
        'student1_correct_student2_wrong': float((correct1 & ~correct2).float().mean().item()),
        'student2_correct_student1_wrong': float((correct2 & ~correct1).float().mean().item()),
        'student12_both_correct': float((correct1 & correct2).float().mean().item()),
        'student12_both_wrong': float((~correct1 & ~correct2).float().mean().item()),
    }
    result['complementary_error'] = (
        result['student1_correct_student2_wrong']
        + result['student2_correct_student1_wrong']
    )
    return result
