"""Student input selection and CPS loss functions."""

from typing import Dict, Mapping, Optional, Tuple

import torch


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

def compute_supervised_loss(
    dice_loss,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
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

def compute_cps_losses(
    ce_loss,
    prediction1: torch.Tensor,
    prediction2: torch.Tensor,
    cps_mode: str = 'standard',
    *,
    ce_none=None,
    reference_targets=None,
) -> Dict[str, torch.Tensor]:
    """Dispatch the selected standard or Native Reference CPS loss."""
    if cps_mode == 'standard':
        return compute_standard_cps_losses(ce_loss, prediction1, prediction2)

    native_modes = {
        'native_fine_margin_suppression',
        'native_fine_margin_reverse',
        'native_fine_margin_calibrated_reverse',
    }
    if cps_mode not in native_modes:
        raise ValueError('Unsupported CPS mode: {}'.format(cps_mode))
    if ce_none is None or reference_targets is None:
        raise ValueError('Native CPS modes require CE-none loss and reference targets')

    pseudo1 = prediction1.detach().argmax(dim=1)
    pseudo2 = prediction2.detach().argmax(dim=1)
    fine = reference_targets['fine_stats']
    return compute_native_margin_calibrated_anti_confirmation_losses(
        ce_none,
        prediction1,
        prediction2,
        pseudo1,
        pseudo2,
        fine['risk_pixel'][:, 0],
        fine['q_pixel'][:, 0],
        fine['opposite_target_pixel'][:, 0],
        cps_mode,
    )

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
