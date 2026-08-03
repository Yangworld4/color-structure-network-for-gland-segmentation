"""Complementarity objective for aligned heterogeneous FDCL views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ComplementarityFeatureLossOutput:
    """Named outputs used by the trainer and experiment logger."""

    total: torch.Tensor
    complement: torch.Tensor
    valid_ratio: torch.Tensor


class _PrivateProjection(nn.Module):
    """Project one view into its private feature space."""

    def __init__(self, in_channels: int, projection_dim: int) -> None:
        super().__init__()
        num_groups = min(8, projection_dim)
        while projection_dim % num_groups != 0:
            num_groups -= 1

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                projection_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, projection_dim),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.head(feature)


class ComplementarityFeatureLoss(nn.Module):
    """Bounded private-feature complementarity with collapse prevention.

    Projected appearance and structure features are encouraged to differ only
    until their cosine similarity reaches ``margin``. A small variance floor
    prevents zero-valued projections from trivially satisfying the objective.

    Labeled samples use every valid pixel. Unlabeled samples use only pixels
    where both branches agree and both confidence values exceed
    ``confidence_threshold``.
    """

    def __init__(
        self,
        private_in_channels: int,
        num_classes: int,
        projection_dim: int = 32,
        margin: float = 0.2,
        confidence_threshold: float = 0.7,
        private_variance_weight: float = 0.1,
        private_target_std: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if private_in_channels <= 0:
            raise ValueError("private_in_channels must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if not -1.0 <= margin <= 1.0:
            raise ValueError("margin must be in [-1, 1]")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if private_variance_weight < 0.0:
            raise ValueError("private_variance_weight must be non-negative")
        if private_target_std < 0.0:
            raise ValueError("private_target_std must be non-negative")

        self.private_in_channels = private_in_channels
        self.num_classes = num_classes
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.private_variance_weight = private_variance_weight
        self.private_target_std = private_target_std
        self.eps = eps

        self.appearance_projection = _PrivateProjection(
            private_in_channels,
            projection_dim,
        )
        self.structure_projection = _PrivateProjection(
            private_in_channels,
            projection_dim,
        )

    def appearance_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters optimized together with the appearance network."""

        return self.appearance_projection.parameters()

    def structure_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters optimized together with the structure network."""

        return self.structure_projection.parameters()

    def forward(
        self,
        appearance_feature: torch.Tensor,
        structure_feature: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        appearance_logits: torch.Tensor | None = None,
        structure_logits: torch.Tensor | None = None,
    ) -> ComplementarityFeatureLossOutput:
        self._validate_feature_pair(appearance_feature, structure_feature)
        private_a = self.appearance_projection(appearance_feature)
        private_s = self.structure_projection(structure_feature)

        valid = self._build_valid_mask(
            labels=labels,
            appearance_logits=appearance_logits,
            structure_logits=structure_logits,
        )
        if valid.shape[0] != appearance_feature.shape[0]:
            raise ValueError(
                "target/logit batch size does not match feature batch size"
            )
        private_valid = self._resize_valid(
            valid,
            appearance_feature.shape[-2:],
        )
        complement_loss = self._private_complement_loss(
            private_a,
            private_s,
            private_valid,
        )
        return ComplementarityFeatureLossOutput(
            total=complement_loss,
            complement=complement_loss,
            valid_ratio=valid.float().mean(),
        )

    def _validate_feature_pair(
        self,
        appearance_feature: torch.Tensor,
        structure_feature: torch.Tensor,
    ) -> None:
        if appearance_feature.ndim != 4 or structure_feature.ndim != 4:
            raise ValueError("private features must use [B, C, H, W] layout")
        if appearance_feature.shape != structure_feature.shape:
            raise ValueError(
                "appearance and structure private features must have identical "
                "shapes; got {} and {}".format(
                    tuple(appearance_feature.shape),
                    tuple(structure_feature.shape),
                )
            )
        if appearance_feature.shape[1] != self.private_in_channels:
            raise ValueError(
                "configured private channels={} but model returned {}".format(
                    self.private_in_channels,
                    appearance_feature.shape[1],
                )
            )

    def _build_valid_mask(
        self,
        *,
        labels: torch.Tensor | None,
        appearance_logits: torch.Tensor | None,
        structure_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels is not None:
            if labels.ndim != 3:
                raise ValueError("labels must use [B, H, W] layout")
            return (labels >= 0) & (labels < self.num_classes)

        if appearance_logits is None or structure_logits is None:
            raise ValueError(
                "provide labels for supervised data or both logits for "
                "unlabeled data"
            )
        if appearance_logits.ndim != 4 or structure_logits.ndim != 4:
            raise ValueError("logits must use [B, C, H, W] layout")
        if appearance_logits.shape != structure_logits.shape:
            raise ValueError(
                "appearance and structure logits must have identical shapes"
            )
        if appearance_logits.shape[1] != self.num_classes:
            raise ValueError(
                "logits have {} classes, expected {}".format(
                    appearance_logits.shape[1],
                    self.num_classes,
                )
            )

        probability_a = F.softmax(appearance_logits.detach(), dim=1)
        probability_s = F.softmax(structure_logits.detach(), dim=1)
        confidence_a, prediction_a = probability_a.max(dim=1)
        confidence_s, prediction_s = probability_s.max(dim=1)
        return (
            prediction_a.eq(prediction_s)
            & confidence_a.ge(self.confidence_threshold)
            & confidence_s.ge(self.confidence_threshold)
        )

    @staticmethod
    def _resize_valid(
        valid: torch.Tensor,
        feature_size: torch.Size,
    ) -> torch.Tensor:
        if valid.shape[-2:] == feature_size:
            return valid
        return F.interpolate(
            valid.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1).bool()

    def _private_complement_loss(
        self,
        private_a: torch.Tensor,
        private_s: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if not valid.any():
            return self._differentiable_zero(private_a, private_s)

        normalized_a = F.normalize(private_a, p=2, dim=1, eps=self.eps)
        normalized_s = F.normalize(private_s, p=2, dim=1, eps=self.eps)
        cosine = (normalized_a * normalized_s).sum(dim=1)
        diversity = F.relu(cosine[valid] - self.margin).mean()

        selected_a = private_a.permute(0, 2, 3, 1)[valid]
        selected_s = private_s.permute(0, 2, 3, 1)[valid]
        variance = self._variance_floor(selected_a) + self._variance_floor(
            selected_s
        )
        return diversity + self.private_variance_weight * 0.5 * variance

    def _variance_floor(self, selected: torch.Tensor) -> torch.Tensor:
        if selected.shape[0] < 2:
            return selected.sum() * 0.0
        channel_std = torch.sqrt(
            selected.var(dim=0, unbiased=False) + self.eps
        )
        return F.relu(self.private_target_std - channel_std).mean()

    @staticmethod
    def _differentiable_zero(*features: torch.Tensor) -> torch.Tensor:
        return sum(feature.sum() for feature in features) * 0.0
