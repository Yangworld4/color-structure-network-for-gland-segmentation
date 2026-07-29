import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
from torch.nn.modules.loss import _Loss
from typing import Union, Tuple, Dict

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[0], target)
        for i in range(1, len(output)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(output[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(output, target)

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1).float()

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=False, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, output, target, **kwargs):
        # *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(output[0], target_one_hot, valid_mask)
        for i in range(1, len(output)):
            aux_loss = self._base_forward(output[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, output, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(output, target)
        else:
            valid_mask = (target != self.ignore_index).long()
            target_one_hot = F.one_hot(torch.clamp_min(target, 0))
            return self._base_forward(output, target_one_hot, valid_mask)


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

class BCELossBoud(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        target_one_hot = F.one_hot(torch.clamp_min(target, 0), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        predict = torch.softmax(predict, 1)

        bs, category, depth, width, heigt = target_one_hot.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:,i]
            targ_i = target_one_hot[:,i]
            tt = np.log(depth * width * heigt / (target_one_hot[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
            torch.log(torch.mul(std, std))) - 1


def segmentation_loss(loss='CE', aux=False, **kwargs):

    if loss == 'dice' or loss == 'DICE':
        seg_loss = DiceLoss(aux=aux)
    elif loss == 'crossentropy' or loss == 'CE':
        seg_loss = MixSoftmaxCrossEntropyLoss(aux=aux)
    elif loss == 'bce':
        seg_loss = nn.BCELoss(size_average=True)
    elif loss == 'bcebound':
        seg_loss = BCELossBoud(num_classes=kwargs['num_classes'])
    else:
        print('sorry, the loss you input is not supported yet')
        sys.exit()

    return seg_loss


class GlobalBoundedDiversityLoss(nn.Module):
    """
    Global Bounded Diversity Loss.

    输入:
        feature_1: [B, C, H_f, W_f]
        feature_2: [B, C, H_f, W_f]

    损失:
        L_gdiv = 0.5 * [
            mean(ReLU(S_1to2 - margin))
            +
            mean(ReLU(S_2to1 - margin))
        ]

    其中:
        S_1to2 = Z1 · stop_gradient(Z2)
        S_2to1 = Z2 · stop_gradient(Z1)
    """

    def __init__(
        self,
        margin: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not -1.0 <= margin <= 1.0:
            raise ValueError(
                f"margin 必须位于 [-1, 1]，当前为 {margin}"
            )

        if eps <= 0:
            raise ValueError(
                f"eps 必须大于 0，当前为 {eps}"
            )

        self.margin = margin
        self.eps = eps

    def _channel_l2_normalize(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        对每个空间位置的 C 维通道向量进行归一化。

        feature:
            [B, C, H_f, W_f]

        norm:
            [B, 1, H_f, W_f]
        """
        # 混合精度训练时使用 float32 计算范数，
        # 避免 FP16 下数值不稳定。
        if feature.dtype in (torch.float16, torch.bfloat16):
            feature_for_norm = feature.float()
        else:
            feature_for_norm = feature

        channel_norm = torch.linalg.vector_norm(
            feature_for_norm,
            ord=2,
            dim=1,
            keepdim=True,
        )

        normalized_feature = (
            feature_for_norm / (channel_norm + self.eps)
        )

        return normalized_feature

    def forward(
        self,
        feature_1: torch.Tensor,
        feature_2: torch.Tensor,
        return_stats: bool = True,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        if feature_1.ndim != 4 or feature_2.ndim != 4:
            raise ValueError(
                "输入特征必须是四维张量 [B, C, H_f, W_f]，"
                f"当前分别为 {feature_1.shape} 和 {feature_2.shape}"
            )

        if feature_1.shape != feature_2.shape:
            raise ValueError(
                "两个网络的特征形状必须一致，"
                f"当前分别为 {feature_1.shape} 和 {feature_2.shape}"
            )

        # Z1, Z2: [B, C, H_f, W_f]
        z1 = self._channel_l2_normalize(feature_1)
        z2 = self._channel_l2_normalize(feature_2)

        # 网络一作为优化对象，网络二停止梯度。
        # [B, H_f, W_f]
        similarity_1_to_2 = torch.sum(
            z1 * z2.detach(),
            dim=1,
        )

        # 网络二作为优化对象，网络一停止梯度。
        # [B, H_f, W_f]
        similarity_2_to_1 = torch.sum(
            z2 * z1.detach(),
            dim=1,
        )

        # phi(S) = max(0, S - margin)
        penalty_1_to_2 = F.relu(
            similarity_1_to_2 - self.margin
        )

        penalty_2_to_1 = F.relu(
            similarity_2_to_1 - self.margin
        )

        loss_1_to_2 = penalty_1_to_2.mean()
        loss_2_to_1 = penalty_2_to_1.mean()

        loss_gdiv = 0.5 * (
            loss_1_to_2 + loss_2_to_1
        )

        if not return_stats:
            return loss_gdiv

        with torch.no_grad():
            stats = {
                "gdiv_loss": loss_gdiv.detach(),
                "similarity_mean": (
                    0.5
                    * (
                        similarity_1_to_2.mean()
                        + similarity_2_to_1.mean()
                    )
                ).detach(),
                "similarity_min": (
                    torch.minimum(
                        similarity_1_to_2.min(),
                        similarity_2_to_1.min(),
                    )
                ).detach(),
                "similarity_max": (
                    torch.maximum(
                        similarity_1_to_2.max(),
                        similarity_2_to_1.max(),
                    )
                ).detach(),
                # 相似度仍高于 margin 的空间位置比例。
                "active_ratio": (
                    0.5
                    * (
                        (similarity_1_to_2 > self.margin)
                        .float()
                        .mean()
                        +
                        (similarity_2_to_1 > self.margin)
                        .float()
                        .mean()
                    )
                ).detach(),
            }

        return loss_gdiv, stats