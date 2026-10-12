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
    collect_q_dynamics=False,
) -> Dict[str, torch.Tensor]:
    """Dispatch the selected standard or Native Reference CPS loss."""
    if cps_mode == 'standard':
        return compute_standard_cps_losses(ce_loss, prediction1, prediction2)

    native_modes = {
        'native_fine_margin_suppression',
        'native_fine_margin_reverse',
        'native_fine_margin_calibrated_reverse',
        'native_fine_margin_calibrated_reverse_disroute',
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
        fine,
        cps_mode,
        collect_q_dynamics=collect_q_dynamics,
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
    fine_stats,
    mode,
    collect_q_dynamics=False,
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
    q_original = q_pixel.detach()
    q_used = q_original.clone()
    d_weight_stats = {}
    if mode == 'native_fine_margin_calibrated_reverse_disroute':
        m_corr = fine_stats['M_corr_pixel'][:, 0]
        m_keep = fine_stats['M_keep_pixel'][:, 0]
        m_dis = fine_stats['M_dis_pixel'][:, 0]
        k_mask = fine_stats['K_pixel'][:, 0]
        ct_fg = fine_stats['Ct_pixel'][:, 0].bool()
        if not torch.equal(m_corr | m_keep | m_dis, k_mask):
            raise AssertionError('Adaptive/constant candidate masks do not partition K')
        if not torch.equal(m_corr, risk_pixel):
            raise AssertionError('Adaptive/constant M_corr candidate changed')
        if not torch.equal(m_corr | m_keep | m_dis, k_mask):
            raise AssertionError('Adaptive/constant K population changed')
        if getattr(fine_stats, 'get', None) is None:
            raise AssertionError('Fine state is not a mapping')
        if fine_stats.get('d_weight_ablation', 'adaptive') == 'constant':
            # Match q budget independently within each branch and semantic
            # direction. This only changes q assignment; all masks and targets
            # remain the original native disroute population.
            for population in (m_corr, m_dis):
                for direction_mask in (ct_fg, ~ct_fg):
                    mask = population & direction_mask
                    if mask.any().item():
                        q_used[mask] = q_original[mask].mean()
        if fine_stats.get('d_weight_ablation', 'adaptive') == 'adaptive':
            if not torch.equal(q_used, q_original):
                raise AssertionError('Adaptive q must equal original q')
        elif fine_stats.get('d_weight_ablation') != 'constant':
            raise ValueError('Unknown d-weight ablation: {}'.format(
                fine_stats.get('d_weight_ablation')
            ))
        if fine_stats.get('d_weight_ablation') == 'constant':
            # Every active subgroup has a constant q and exactly matches its
            # original sum (within floating point reduction error).
            for population in (m_corr, m_dis):
                for direction_mask in (ct_fg, ~ct_fg):
                    mask = population & direction_mask
                    if mask.any().item():
                        if q_used[mask].std(unbiased=False).item() > 1e-7:
                            raise AssertionError('Constant q varies within a subgroup')
                        before = q_original[mask].sum()
                        after = q_used[mask].sum()
                        rel = ((after - before).abs() / before.abs().clamp_min(1e-12)).item()
                        if rel >= 1e-6:
                            raise AssertionError('Constant q does not match subgroup budget')
        q_pixel = q_used
        d_pixel = fine_stats['d_pixel'][:, 0].detach()
        d_weight_stats = {
            'num_K': k_mask.sum().detach(),
            'num_Mcorr': m_corr.sum().detach(),
            'num_Mkeep': m_keep.sum().detach(),
            'num_Mdis': m_dis.sum().detach(),
            'sum_D_corr': d_pixel[m_corr].sum().detach(),
            'count_D_corr': m_corr.sum().detach(),
            'sum_D_dis': d_pixel[m_dis].sum().detach(),
            'count_D_dis': m_dis.sum().detach(),
        }
        for name, mask in (('corr', m_corr), ('dis', m_dis)):
            for suffix, values in (('original', q_original), ('used', q_used)):
                selected = values[mask]
                d_weight_stats['sum_q_{}_{}'.format(suffix, name)] = selected.sum().detach()
                d_weight_stats['sum_q2_{}_{}'.format(suffix, name)] = selected.square().sum().detach()
                d_weight_stats['count_q_{}'.format(name)] = mask.sum().detach()
            original_sum = q_original[mask].sum()
            used_sum = q_used[mask].sum()
            d_weight_stats['budget_error_{}'.format(name)] = (
                (used_sum - original_sum).abs() / original_sum.abs().clamp_min(1e-12)
            ).detach()
        for name, direction_mask in (('FG2BG', ct_fg), ('BG2FG', ~ct_fg)):
            mask = (m_corr | m_dis) & direction_mask
            d_weight_stats['count_{}'.format(name)] = mask.sum().detach()
            d_weight_stats['sum_q_original_{}'.format(name)] = q_original[mask].sum().detach()
            d_weight_stats['sum_q_used_{}'.format(name)] = q_used[mask].sum().detach()
    current_a = ce_none(prediction1, pseudo2)
    current_b = ce_none(prediction2, pseudo1)
    opposite_target = opposite_target_pixel.long().detach()
    if mode == 'native_fine_margin_calibrated_reverse_disroute':
        # Outside M_corr, retain student-generated targets even in this
        # unused intermediate tensor. In particular, never construct a
        # UNI-equivalent pixel target on M_dis.
        opposite_target_a = torch.where(risk_pixel, opposite_target, pseudo2)
        opposite_target_b = torch.where(risk_pixel, opposite_target, pseudo1)
    else:
        opposite_target_a = opposite_target
        opposite_target_b = opposite_target
    opposite_a = ce_none(prediction1, opposite_target_a)
    opposite_b = ce_none(prediction2, opposite_target_b)
    if mode == 'native_fine_margin_suppression':
        cps_weight = 1.0 - q_pixel
        opposite_weight = torch.zeros_like(q_pixel)
    elif mode == 'native_fine_margin_reverse':
        cps_weight = torch.ones_like(q_pixel)
        opposite_weight = q_pixel
    elif mode in (
        'native_fine_margin_calibrated_reverse',
        'native_fine_margin_calibrated_reverse_disroute',
    ):
        cps_weight = 1.0 - q_pixel
        opposite_weight = q_pixel
    else:
        raise ValueError('Unknown margin-calibrated ablation mode: {}'.format(mode))
    risk_a = cps_weight * current_a + opposite_weight * opposite_a
    risk_b = cps_weight * current_b + opposite_weight * opposite_b
    effective_a = torch.where(risk_pixel, risk_a, current_a)
    effective_b = torch.where(risk_pixel, risk_b, current_b)
    routing_metrics = {}
    q_dynamics = None
    if mode == 'native_fine_margin_calibrated_reverse_disroute':
        m_corr = fine_stats['M_corr_pixel'][:, 0]
        m_keep = fine_stats['M_keep_pixel'][:, 0]
        m_dis = fine_stats['M_dis_pixel'][:, 0]
        aligned_a = fine_stats['aligned_A_pixel'][:, 0]
        aligned_b = fine_stats['aligned_B_pixel'][:, 0]
        k_mask = fine_stats['K_pixel'][:, 0]
        if not torch.equal(m_corr, risk_pixel):
            raise AssertionError('M_corr differs from the existing risk mask')
        if torch.any((m_corr & m_keep) | (m_corr & m_dis) | (m_keep & m_dis)):
            raise AssertionError('Fine contradiction partition overlaps')
        if not torch.equal(m_corr | m_keep | m_dis, k_mask):
            raise AssertionError('Fine contradiction partition does not cover K')
        if m_dis.any().item() and not torch.all(aligned_a[m_dis] ^ aligned_b[m_dis]).item():
            raise AssertionError('Exactly one Student must align with UNI in M_dis')
        weight_a = torch.ones_like(q_pixel)
        weight_b = torch.ones_like(q_pixel)
        weight_a[aligned_a] = 1.0 - q_pixel[aligned_a]
        weight_b[aligned_b] = 1.0 - q_pixel[aligned_b]
        if not (torch.all((weight_a >= 0) & (weight_a <= 1)).item()
                and torch.all((weight_b >= 0) & (weight_b <= 1)).item()):
            raise AssertionError('Directional CPS weight escaped [0,1]')
        routed_a = weight_a * current_a
        routed_b = weight_b * current_b
        effective_a = torch.where(m_dis, routed_a, effective_a)
        effective_b = torch.where(m_dis, routed_b, effective_b)
        if not torch.equal(effective_a[m_corr], risk_a[m_corr]) or not torch.equal(
            effective_b[m_corr], risk_b[m_corr]
        ):
            raise AssertionError('M_corr loss changed under fine routing')
        untouched = m_keep | ~k_mask
        if not torch.equal(effective_a[untouched], current_a[untouched]) or not torch.equal(
            effective_b[untouched], current_b[untouched]
        ):
            raise AssertionError('M_keep/non-K loss changed under fine routing')
        routing_metrics = {
            'num_K_pixels': k_mask.sum().detach(),
            'num_M_corr_pixels': m_corr.sum().detach(),
            'num_M_keep_pixels': m_keep.sum().detach(),
            'num_M_dis_pixels': m_dis.sum().detach(),
            'aligned_A_pixels': aligned_a.sum().detach(),
            'aligned_B_pixels': aligned_b.sum().detach(),
            'q_M_dis': q_pixel[m_dis].detach(),
            'q_original_M_dis': q_original[m_dis].detach(),
            'harmful_weight_M_dis': (1.0 - q_pixel[m_dis]).detach(),
            'effective_suppressed_mass': (m_dis.to(q_pixel.dtype) * q_pixel).detach(),
            'loss_A_from_B_before': current_a[valid1].detach(),
            'loss_A_from_B_after': effective_a[valid1].detach(),
            'loss_B_from_A_before': current_b[valid2].detach(),
            'loss_B_from_A_after': effective_b[valid2].detach(),
        }
    loss_a_to_b = 0.5 * effective_a[valid1].mean()
    loss_b_to_a = 0.5 * effective_b[valid2].mean()
    raw_a = 0.5 * current_a[valid1].mean()
    raw_b = 0.5 * current_b[valid2].mean()
    risk_loss = (
        0.5 * (risk_a[risk_pixel].mean() + risk_b[risk_pixel].mean())
        if risk_pixel.any().item() else prediction1.sum() * 0.0
    )
    if collect_q_dynamics:
        # Detached diagnostic maps only. The scalar loss above is unchanged;
        # weighted contribution maps include the existing 0.5 and valid-pixel
        # reductions so their active-pixel sums match this branch's contribution
        # to the current bidirectional CPS scalar.
        current_a_log = current_a.detach()
        current_b_log = current_b.detach()
        opposite_a_log = opposite_a.detach()
        opposite_b_log = opposite_b.detach()
        n_valid1 = valid1.sum().to(current_a.dtype).clamp_min(1.0)
        n_valid2 = valid2.sum().to(current_b.dtype).clamp_min(1.0)
        q_dynamics = {
            'q_pixel': q_pixel.detach(),
            'raw_cps': (0.5 * (current_a_log + current_b_log)).detach(),
            'raw_opp': (0.5 * (opposite_a_log + opposite_b_log)).detach(),
            'weighted_cps': (
                0.5 * (cps_weight.detach() * current_a_log / n_valid1
                       + cps_weight.detach() * current_b_log / n_valid2)
            ).detach(),
            'weighted_opp': (
                0.5 * (opposite_weight.detach() * opposite_a_log / n_valid1
                       + opposite_weight.detach() * opposite_b_log / n_valid2)
            ).detach(),
        }
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
        'd_weight_stats': d_weight_stats,
        'cps_weight_risk': cps_weight[risk_pixel].detach(),
        'opposite_weight_risk': opposite_weight[risk_pixel].detach(),
        'q_dynamics': q_dynamics,
        **routing_metrics,
    }


def deterministic_disroute_sanity():
    """Check hand-authored directional CPS weights and endpoint behavior."""
    q = torch.tensor([0.8, 0.8])
    uni = torch.tensor([0, 0])  # BG
    a = torch.tensor([0, 1])
    b = torch.tensor([1, 0])
    aligned_a, aligned_b = a.eq(uni), b.eq(uni)
    weight_a, weight_b = torch.ones_like(q), torch.ones_like(q)
    weight_a[aligned_a] = 1.0 - q[aligned_a]
    weight_b[aligned_b] = 1.0 - q[aligned_b]
    if not torch.allclose(weight_a, torch.tensor([0.2, 1.0])):
        raise AssertionError('Case A/B A<-B directional weight is reversed')
    if not torch.allclose(weight_b, torch.tensor([1.0, 0.2])):
        raise AssertionError('Case A/B B<-A directional weight is reversed')
    if not torch.equal(1.0 - torch.zeros_like(q), torch.ones_like(q)):
        raise AssertionError('q=0 must reduce to vanilla CPS')
    if not torch.equal(1.0 - torch.ones_like(q), torch.zeros_like(q)):
        raise AssertionError('q=1 must close the harmful direction')
    # At q=1, only the aligned->opposed incoming direction remains.
    q_one = torch.ones_like(q)
    w_a_one = torch.ones_like(q_one)
    w_b_one = torch.ones_like(q_one)
    w_a_one[aligned_a] = 1.0 - q_one[aligned_a]
    w_b_one[aligned_b] = 1.0 - q_one[aligned_b]
    if not torch.equal(w_a_one, torch.tensor([0.0, 1.0])):
        raise AssertionError('q=1 must retain only A->B when A is aligned')
    if not torch.equal(w_b_one, torch.tensor([1.0, 0.0])):
        raise AssertionError('q=1 must retain only B->A when B is aligned')
    # At q=0 on M_dis, compare the new mode against the current mode while
    # keeping q nonzero on M_corr. This covers M_corr, M_keep and non-K.
    ce_none = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    logits_a = torch.tensor([[[[3., 0., 3., 0., 3.]], [[0., 3., 0., 3., 0.]]]])
    logits_b = torch.tensor([[[[0., 3., 0., 3., 0.]], [[3., 0., 3., 0., 3.]]]])
    k = torch.tensor([[[[True, True, True, True, False]]]])
    corr = torch.tensor([[[[True, False, False, False, False]]]])
    keep = torch.tensor([[[[False, True, False, False, False]]]])
    dis = torch.tensor([[[[False, False, True, True, False]]]])
    q_pixels = torch.tensor([[[[0.6, 0.2, 0., 0., 0.]]]])
    fine = {
        'risk_pixel': corr,
        'q_pixel': q_pixels,
        'opposite_target_pixel': torch.tensor([[[[1, 0, 0, 0, 0]]]]),
        'K_pixel': k,
        'Ct_pixel': torch.zeros_like(k),
        'd_pixel': torch.zeros_like(q_pixels),
        'M_corr_pixel': corr,
        'M_keep_pixel': keep,
        'M_dis_pixel': dis,
        'aligned_A_pixel': torch.tensor([[[[False, False, True, False, False]]]]),
        'aligned_B_pixel': torch.tensor([[[[False, False, False, True, False]]]]),
    }
    reference = {'fine_stats': fine}
    old_loss = compute_cps_losses(
        ce_none, logits_a, logits_b,
        cps_mode='native_fine_margin_calibrated_reverse',
        ce_none=ce_none, reference_targets=reference,
    )
    new_loss = compute_cps_losses(
        ce_none, logits_a, logits_b,
        cps_mode='native_fine_margin_calibrated_reverse_disroute',
        ce_none=ce_none, reference_targets=reference,
    )
    if not torch.equal(old_loss['total'], new_loss['total']):
        raise AssertionError('M_corr/M_keep/non-K changed when M_dis q=0')
    return True
