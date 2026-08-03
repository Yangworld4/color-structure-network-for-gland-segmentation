"""Heterogeneous-view feature objectives for the optional FDCL extension.

This module is intentionally separate from ``loss_function.py`` so the
original FDCL losses remain unchanged and directly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeterogeneousFeatureLossOutput:
    """Named outputs used by the trainer and experiment logger."""

    total: torch.Tensor
    semantic: torch.Tensor
    complement: torch.Tensor
    valid_ratio: torch.Tensor


class _ProjectionPair(nn.Module):
    """Project one view into shared-semantic and private feature spaces."""

    def __init__(
        self,
        semantic_in_channels: int,
        private_in_channels: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        num_groups = min(8, projection_dim)
        while projection_dim % num_groups != 0:
            num_groups -= 1

        def make_head(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    projection_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(num_groups, projection_dim),
                nn.GELU(),
            )

        self.semantic = make_head(semantic_in_channels)
        self.private = make_head(private_in_channels)

    def forward(
        self,
        semantic_feature: torch.Tensor,
        private_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.semantic(semantic_feature),
            self.private(private_feature),
        )


class HeterogeneousFeatureLoss(nn.Module):
    """Semantic agreement plus bounded private-feature complementarity.

    Semantic features are aligned through class prototypes. Private features
    are encouraged to differ only until their cosine similarity reaches the
    configured margin. A small variance term prevents a zero-valued private
    projection from trivially satisfying the complementarity objective.

    For labeled samples, ``labels`` supplies the class masks. For unlabeled
    samples, pixels are used only when both branches agree and both confidence
    values exceed ``confidence_threshold``.
    """

    def __init__(
        self,
        semantic_in_channels: int,
        num_classes: int,
        private_in_channels: Optional[int] = None,
        projection_dim: int = 32,
        margin: float = 0.2,
        confidence_threshold: float = 0.7,
        private_variance_weight: float = 0.1,
        private_target_std: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if semantic_in_channels <= 0:
            raise ValueError("semantic_in_channels must be positive")
        if private_in_channels is None:
            private_in_channels = semantic_in_channels
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

        self.semantic_in_channels = semantic_in_channels
        self.private_in_channels = private_in_channels
        self.num_classes = num_classes
        self.margin = margin
        self.confidence_threshold = confidence_threshold
        self.private_variance_weight = private_variance_weight
        self.private_target_std = private_target_std
        self.eps = eps

        # HETERO-FDCL ADDITION: view-specific heads prevent RGB and structure
        # statistics from being forced through one shared projection.
        self.appearance_projection = _ProjectionPair(
            semantic_in_channels,
            private_in_channels,
            projection_dim,
        )
        self.structure_projection = _ProjectionPair(
            semantic_in_channels,
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
        appearance_semantic_feature: torch.Tensor,
        structure_semantic_feature: torch.Tensor,
        *,
        appearance_private_feature: Optional[torch.Tensor] = None,
        structure_private_feature: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        appearance_logits: Optional[torch.Tensor] = None,
        structure_logits: Optional[torch.Tensor] = None,
    ) -> HeterogeneousFeatureLossOutput:
        if appearance_private_feature is None:
            appearance_private_feature = appearance_semantic_feature
        if structure_private_feature is None:
            structure_private_feature = structure_semantic_feature

        self._validate_feature_pair(
            appearance_semantic_feature,
            structure_semantic_feature,
            self.semantic_in_channels,
            "semantic",
        )
        self._validate_feature_pair(
            appearance_private_feature,
            structure_private_feature,
            self.private_in_channels,
            "private",
        )
        if (
            appearance_semantic_feature.shape[0]
            != appearance_private_feature.shape[0]
        ):
            raise ValueError(
                "semantic and private feature batch sizes must match"
            )

        semantic_a, private_a = self.appearance_projection(
            appearance_semantic_feature,
            appearance_private_feature,
        )
        semantic_s, private_s = self.structure_projection(
            structure_semantic_feature,
            structure_private_feature,
        )

        target, valid = self._build_target_and_valid_mask(
            labels=labels,
            appearance_logits=appearance_logits,
            structure_logits=structure_logits,
        )
        if target.shape[0] != appearance_semantic_feature.shape[0]:
            raise ValueError(
                "target/logit batch size does not match feature batch size"
            )
        semantic_target, semantic_valid = self._resize_target_and_valid(
            target,
            valid,
            appearance_semantic_feature.shape[-2:],
        )
        _, private_valid = self._resize_target_and_valid(
            target,
            valid,
            appearance_private_feature.shape[-2:],
        )
        semantic_loss = self._semantic_prototype_loss(
            semantic_a,
            semantic_s,
            semantic_target,
            semantic_valid,
        )
        complement_loss = self._private_complement_loss(
            private_a,
            private_s,
            private_valid,
        )
        valid_ratio = valid.float().mean()
        return HeterogeneousFeatureLossOutput(
            total=semantic_loss + complement_loss,
            semantic=semantic_loss,
            complement=complement_loss,
            valid_ratio=valid_ratio,
        )

    @staticmethod
    def _validate_feature_pair(
        appearance_feature: torch.Tensor,
        structure_feature: torch.Tensor,
        expected_channels: int,
        feature_name: str,
    ) -> None:
        if appearance_feature.ndim != 4 or structure_feature.ndim != 4:
            raise ValueError(
                "{} features must use [B, C, H, W] layout".format(
                    feature_name
                )
            )
        if appearance_feature.shape != structure_feature.shape:
            raise ValueError(
                "appearance and structure {} features must have identical "
                "shapes; got {} and {}".format(
                    feature_name,
                    tuple(appearance_feature.shape),
                    tuple(structure_feature.shape),
                )
            )
        if appearance_feature.shape[1] != expected_channels:
            raise ValueError(
                "configured {} channels={} but model returned {}".format(
                    feature_name,
                    expected_channels,
                    appearance_feature.shape[1],
                )
            )

    def _build_target_and_valid_mask(
        self,
        *,
        labels: Optional[torch.Tensor],
        appearance_logits: Optional[torch.Tensor],
        structure_logits: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels is not None:
            target = labels.long()
            if target.ndim != 3:
                raise ValueError("labels must use [B, H, W] layout")
            valid = (target >= 0) & (target < self.num_classes)
            return target, valid

        if appearance_logits is None or structure_logits is None:
            raise ValueError(
                "provide labels for supervised data or both logits for "
                "unlabeled data"
            )
        if appearance_logits.ndim != 4 or structure_logits.ndim != 4:
            raise ValueError("logits must use [B, C, H, W] layout")
        if appearance_logits.shape[0] != structure_logits.shape[0]:
            raise ValueError(
                "appearance and structure logits batch sizes differ"
            )
        if appearance_logits.shape[1] != self.num_classes:
            raise ValueError(
                "appearance logits have {} classes, expected {}".format(
                    appearance_logits.shape[1], self.num_classes
                )
            )
        if structure_logits.shape[1] != self.num_classes:
            raise ValueError(
                "structure logits have {} classes, expected {}".format(
                    structure_logits.shape[1], self.num_classes
                )
            )

        if appearance_logits.shape[-2:] != structure_logits.shape[-2:]:
            raise ValueError(
                "appearance and structure logits spatial sizes differ"
            )

        probability_a = F.softmax(appearance_logits.detach(), dim=1)
        probability_s = F.softmax(structure_logits.detach(), dim=1)
        confidence_a, prediction_a = probability_a.max(dim=1)
        confidence_s, prediction_s = probability_s.max(dim=1)
        valid = (
            prediction_a.eq(prediction_s)
            & confidence_a.ge(self.confidence_threshold)
            & confidence_s.ge(self.confidence_threshold)
        )
        return prediction_a, valid

    @staticmethod
    def _resize_target_and_valid(
        target: torch.Tensor,
        valid: torch.Tensor,
        feature_size: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target.shape[-2:] == feature_size:
            return target, valid
        resized_target = F.interpolate(
            target.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1).long()
        resized_valid = F.interpolate(
            valid.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1).bool()
        return resized_target, resized_valid

    def _semantic_prototype_loss(
        self,
        semantic_a: torch.Tensor,
        semantic_s: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for class_index in range(self.num_classes):
            class_mask = valid & target.eq(class_index)
            if not class_mask.any():
                continue
            weights = class_mask.unsqueeze(1).to(dtype=semantic_a.dtype)
            denominator = weights.sum().clamp_min(1.0)
            prototype_a = (semantic_a * weights).sum((0, 2, 3)) / denominator
            prototype_s = (semantic_s * weights).sum((0, 2, 3)) / denominator
            losses.append(
                1.0
                - F.cosine_similarity(
                    prototype_a.unsqueeze(0),
                    prototype_s.unsqueeze(0),
                    dim=1,
                    eps=self.eps,
                ).squeeze(0)
            )

        if not losses:
            return self._differentiable_zero(semantic_a, semantic_s)
        return torch.stack(losses).mean()

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
