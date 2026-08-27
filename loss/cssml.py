from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _target_2d(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0, ...]
    return target.long()


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits)
        return torch.cat((1.0 - foreground, foreground), dim=1)
    return torch.softmax(logits, dim=1)


def _pixel_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _target_2d(target)
    if logits.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits[:, 0, ...], target.float(), reduction='none'
        )
    return F.cross_entropy(logits, target, reduction='none')


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.detach().bool()
    per_pixel = _pixel_ce(logits, target)
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    return per_pixel[mask].mean()


def masked_ce_sum(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.detach().bool()
    per_pixel = _pixel_ce(logits.detach(), target.detach())
    return float(per_pixel[mask].sum().item()), float(mask.sum().item())


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    mask = mask.detach().bool()
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    probabilities = _probabilities(logits)
    target_one_hot = _one_hot_labels(target.detach(), probabilities.shape[1])
    valid = mask.unsqueeze(1).float()
    intersection = (probabilities * target_one_hot * valid).sum(dim=(0, 2, 3))
    denominator = ((probabilities + target_one_hot) * valid).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def cdfl_loss(
    feat_a_l: torch.Tensor,
    feat_b_l: torch.Tensor,
    feat_a_u: torch.Tensor,
    feat_b_u: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_l_a2b = 1.0 + cos(feat_a_l.detach(), feat_b_l).mean()
    loss_l_b2a = 1.0 + cos(feat_b_l.detach(), feat_a_l).mean()
    loss_l = 0.5 * (loss_l_a2b + loss_l_b2a)

    loss_u_a2b = 1.0 + cos(feat_a_u.detach(), feat_b_u).mean()
    loss_u_b2a = 1.0 + cos(feat_b_u.detach(), feat_a_u).mean()
    loss_u = 0.5 * (loss_u_a2b + loss_u_b2a)
    return {
        'labeled': loss_l,
        'unlabeled': loss_u,
        'total': loss_l + loss_u,
        'cos_labeled': cos(feat_a_l.detach(), feat_b_l.detach()).mean(),
        'cos_unlabeled': cos(feat_a_u.detach(), feat_b_u.detach()).mean(),
    }


def _one_hot_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.clamp_min(0), num_classes=num_classes).permute(
        0, 3, 1, 2
    ).float()


def _gather_class_map(class_maps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return class_maps.gather(1, labels.unsqueeze(1)).squeeze(1)


def _resize_feature(
    feature: torch.Tensor,
    output_size,
) -> torch.Tensor:
    if feature.shape[-2:] != output_size:
        feature = F.interpolate(
            feature, size=output_size, mode='bilinear', align_corners=False
        )
    return feature


def _class_prototypes(
    feature: torch.Tensor,
    pseudo: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    one_hot = _one_hot_labels(pseudo, num_classes)
    numerator = torch.einsum('bchw,bkhw->bkc', feature, one_hot)
    denominator = one_hot.sum(dim=(2, 3)).clamp_min(1.0).unsqueeze(-1)
    prototypes = numerator / denominator
    return F.normalize(prototypes, dim=2, eps=1e-6)


def _prototype_similarity(
    feature: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    feature = F.normalize(feature, dim=1, eps=1e-6)
    return torch.einsum('bchw,bkc->bkhw', feature, prototypes)


def compute_geometry_reliability(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    pseudo_a: torch.Tensor,
    pseudo_b: torch.Tensor,
    conflict_mask: torch.Tensor,
    epoch: int,
    num_epochs: int,
    num_classes: int,
):
    del logits_a, logits_b, conflict_mask
    num_classes = max(int(num_classes), 2)
    output_size = pseudo_a.shape[-2:]
    del epoch, num_epochs
    feat_a = _resize_feature(feat_a, output_size)
    feat_b = _resize_feature(feat_b, output_size)

    prototypes_a = _class_prototypes(feat_a, pseudo_a, num_classes)
    prototypes_b = _class_prototypes(feat_b, pseudo_b, num_classes)
    similarity_a = _prototype_similarity(feat_a, prototypes_a)
    similarity_b = _prototype_similarity(feat_b, prototypes_b)

    geometry_score_a = similarity_a.max(dim=1).values
    geometry_score_b = similarity_b.max(dim=1).values
    return geometry_score_a.detach(), geometry_score_b.detach()


def gmps_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    num_classes: int,
    epoch: int,
    num_epochs: int,
) -> Dict[str, torch.Tensor]:
    prob_a = _probabilities(logits_a)
    prob_b = _probabilities(logits_b)
    conf_a, pseudo_a = prob_a.detach().max(dim=1)
    conf_b, pseudo_b = prob_b.detach().max(dim=1)

    same_mask = pseudo_a.eq(pseudo_b).detach()
    conflict_mask = pseudo_a.ne(pseudo_b).detach()

    class_id = pseudo_a.detach()
    p_same_a = prob_a.detach().gather(1, class_id.unsqueeze(1)).squeeze(1)
    p_same_b = prob_b.detach().gather(1, class_id.unsqueeze(1)).squeeze(1)
    same_a2b = (same_mask & (p_same_a > p_same_b)).detach()
    same_b2a = (same_mask & (~same_a2b)).detach()

    loss_same_a2b = masked_dice_loss(logits_b, pseudo_a, same_a2b)
    loss_same_b2a = masked_dice_loss(logits_a, pseudo_b, same_b2a)
    loss_same = loss_same_a2b + loss_same_b2a

    geometry_score_a = torch.zeros_like(conf_a)
    geometry_score_b = torch.zeros_like(conf_b)
    conflict_a2b = torch.zeros_like(conflict_mask)
    conflict_b2a = torch.zeros_like(conflict_mask)
    loss_conflict = logits_a.sum() * 0.0

    if conflict_mask.any().item():
        geometry_score_a, geometry_score_b = compute_geometry_reliability(
            logits_a.detach(),
            logits_b.detach(),
            feat_a.detach(),
            feat_b.detach(),
            pseudo_a,
            pseudo_b,
            conflict_mask,
            epoch,
            num_epochs,
            num_classes,
        )
        conflict_b2a = (conflict_mask & (geometry_score_a <= geometry_score_b)).detach()
        conflict_a2b = (conflict_mask & (~conflict_b2a)).detach()
        loss_conflict = (
            masked_dice_loss(logits_b, pseudo_a, conflict_a2b)
            + masked_dice_loss(logits_a, pseudo_b, conflict_b2a)
        )

    route_a2b = (same_a2b | conflict_a2b).detach()
    route_b2a = (same_b2a | conflict_b2a).detach()
    return {
        'loss': loss_same + loss_conflict,
        'loss_same': loss_same,
        'loss_conflict': loss_conflict,
        'same_mask': same_mask,
        'conflict_mask': conflict_mask,
        'route_a2b': route_a2b,
        'route_b2a': route_b2a,
        'same_a2b': same_a2b,
        'same_b2a': same_b2a,
        'conflict_a2b': conflict_a2b,
        'conflict_b2a': conflict_b2a,
        'pseudo_a': pseudo_a,
        'pseudo_b': pseudo_b,
        'geometry_score_a': geometry_score_a,
        'geometry_score_b': geometry_score_b,
    }
