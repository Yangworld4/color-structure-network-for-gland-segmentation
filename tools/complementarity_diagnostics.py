"""Optional dual-branch diagnostics for complementarity experiments.

This module is deliberately independent from the training objective. It only
consumes validation logits and masks that the regular training loop already
computed, and writes all artifacts into a separate diagnostics directory.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


_THRESHOLDS = np.arange(0.0, 0.9, 0.02, dtype=np.float32)


def _binary_overlap(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    union = np.logical_or(prediction, target).sum(dtype=np.float64)
    prediction_area = prediction.sum(dtype=np.float64)
    target_area = target.sum(dtype=np.float64)
    jaccard = intersection / union if union else 1.0
    denominator = prediction_area + target_area
    dice = 2.0 * intersection / denominator if denominator else 1.0
    return float(jaccard), float(dice)


def _best_binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    best = {'threshold': 0.0, 'jaccard': -1.0, 'dice': 0.0}
    for threshold in _THRESHOLDS:
        prediction = probability > threshold
        jaccard, dice = _binary_overlap(prediction, target)
        if jaccard > best['jaccard']:
            best = {
                'threshold': float(threshold),
                'jaccard': jaccard,
                'dice': dice,
            }
    return best


def _boundary_mask(target: torch.Tensor, width: int) -> torch.Tensor:
    if width < 1:
        raise ValueError('boundary width must be at least one pixel')
    foreground = target.float().unsqueeze(1)
    kernel_size = 2 * width + 1
    dilated = F.max_pool2d(
        foreground,
        kernel_size=kernel_size,
        stride=1,
        padding=width,
    )
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground,
        kernel_size=kernel_size,
        stride=1,
        padding=width,
    )
    return dilated.squeeze(1).ne(eroded.squeeze(1))


def _region_statistics(
    correct_a: np.ndarray,
    correct_s: np.ndarray,
    region: np.ndarray,
) -> dict[str, float]:
    region_pixels = int(region.sum())
    if region_pixels == 0:
        return {
            'pixel_count': 0,
            'both_correct_ratio': 0.0,
            'appearance_only_correct_ratio': 0.0,
            'structure_only_correct_ratio': 0.0,
            'both_wrong_ratio': 0.0,
            'disagreement_ratio': 0.0,
        }

    both_correct = region & correct_a & correct_s
    appearance_only = region & correct_a & ~correct_s
    structure_only = region & ~correct_a & correct_s
    both_wrong = region & ~correct_a & ~correct_s
    denominator = float(region_pixels)
    return {
        'pixel_count': region_pixels,
        'both_correct_ratio': float(both_correct.sum() / denominator),
        'appearance_only_correct_ratio': float(
            appearance_only.sum() / denominator
        ),
        'structure_only_correct_ratio': float(
            structure_only.sum() / denominator
        ),
        'both_wrong_ratio': float(both_wrong.sum() / denominator),
        'disagreement_ratio': float(
            np.logical_xor(correct_a, correct_s)[region].mean()
        ),
    }


def _prefix_statistics(
    destination: dict[str, float],
    prefix: str,
    statistics: Mapping[str, float],
) -> None:
    for name, value in statistics.items():
        destination['{}_{}'.format(prefix, name)] = value


def compute_complementarity_diagnostics(
    scores_a: torch.Tensor,
    scores_s: torch.Tensor,
    masks: torch.Tensor,
    *,
    boundary_width: int = 3,
) -> dict[str, object]:
    """Compute fusion, oracle, and branch-exclusive correctness statistics."""

    if scores_a.shape != scores_s.shape:
        raise ValueError('appearance and structure score shapes must match')
    if scores_a.ndim != 4 or scores_a.shape[1] != 2:
        raise ValueError('diagnostics currently require binary [N,2,H,W] logits')
    if masks.shape != scores_a.shape[:1] + scores_a.shape[2:]:
        raise ValueError('mask shape must match the score batch and spatial size')

    probability_a_all = torch.softmax(scores_a.detach().float(), dim=1).cpu()
    probability_s_all = torch.softmax(scores_s.detach().float(), dim=1).cpu()
    probability_a = probability_a_all[:, 1].numpy()
    probability_s = probability_s_all[:, 1].numpy()
    target_tensor = masks.detach().long().cpu()
    target = target_tensor.numpy().astype(bool)

    probability_mean = 0.5 * (probability_a + probability_s)
    entropy_a = -(
        probability_a_all
        * probability_a_all.clamp_min(1e-8).log()
    ).sum(dim=1).numpy()
    entropy_s = -(
        probability_s_all
        * probability_s_all.clamp_min(1e-8).log()
    ).sum(dim=1).numpy()
    reliability_a = np.exp(-entropy_a)
    reliability_s = np.exp(-entropy_s)
    probability_entropy = (
        reliability_a * probability_a + reliability_s * probability_s
    ) / np.maximum(reliability_a + reliability_s, 1e-8)

    metrics_a = _best_binary_metrics(probability_a, target)
    metrics_s = _best_binary_metrics(probability_s, target)
    metrics_mean = _best_binary_metrics(probability_mean, target)
    metrics_entropy = _best_binary_metrics(probability_entropy, target)

    prediction_a = probability_a > metrics_a['threshold']
    prediction_s = probability_s > metrics_s['threshold']
    prediction_mean = probability_mean > metrics_mean['threshold']
    prediction_entropy = probability_entropy > metrics_entropy['threshold']
    correct_a = prediction_a == target
    correct_s = prediction_s == target

    # The oracle is a theoretical upper bound: ground truth selects a correct
    # branch wherever at least one branch is correct. It is never used to train.
    oracle_prediction = np.where(correct_a, prediction_a, prediction_s)
    oracle_jaccard, oracle_dice = _binary_overlap(oracle_prediction, target)

    boundary = _boundary_mask(target_tensor, boundary_width).numpy()
    all_region = np.ones_like(target, dtype=bool)
    interior = ~boundary
    summary: dict[str, float] = {}
    for prefix, metrics in (
        ('appearance', metrics_a),
        ('structure', metrics_s),
        ('mean', metrics_mean),
        ('entropy', metrics_entropy),
    ):
        for name, value in metrics.items():
            summary['{}_{}'.format(prefix, name)] = value
    summary['oracle_jaccard'] = oracle_jaccard
    summary['oracle_dice'] = oracle_dice
    summary['oracle_gain_over_best_branch'] = oracle_jaccard - max(
        metrics_a['jaccard'], metrics_s['jaccard']
    )
    summary['mean_gain_over_best_branch'] = metrics_mean['jaccard'] - max(
        metrics_a['jaccard'], metrics_s['jaccard']
    )
    summary['entropy_gain_over_best_branch'] = (
        metrics_entropy['jaccard']
        - max(metrics_a['jaccard'], metrics_s['jaccard'])
    )
    summary['boundary_pixel_ratio'] = float(boundary.mean())
    _prefix_statistics(
        summary,
        'overall',
        _region_statistics(correct_a, correct_s, all_region),
    )
    _prefix_statistics(
        summary,
        'boundary',
        _region_statistics(correct_a, correct_s, boundary),
    )
    _prefix_statistics(
        summary,
        'interior',
        _region_statistics(correct_a, correct_s, interior),
    )

    per_image = []
    for index in range(target.shape[0]):
        row: dict[str, float | int] = {'index': index}
        for prefix, prediction in (
            ('appearance', prediction_a[index]),
            ('structure', prediction_s[index]),
            ('mean', prediction_mean[index]),
            ('entropy', prediction_entropy[index]),
            ('oracle', oracle_prediction[index]),
        ):
            jaccard, dice = _binary_overlap(prediction, target[index])
            row['{}_jaccard'.format(prefix)] = jaccard
            row['{}_dice'.format(prefix)] = dice
        image_stats = _region_statistics(
            correct_a[index],
            correct_s[index],
            np.ones_like(target[index], dtype=bool),
        )
        for name, value in image_stats.items():
            row[name] = value
        per_image.append(row)

    return {
        'summary': summary,
        'per_image': per_image,
        'arrays': {
            'probability_a': probability_a,
            'probability_s': probability_s,
            'probability_mean': probability_mean,
            'probability_entropy': probability_entropy,
            'prediction_a': prediction_a,
            'prediction_s': prediction_s,
            'prediction_mean': prediction_mean,
            'prediction_entropy': prediction_entropy,
            'oracle_prediction': oracle_prediction,
            'target': target,
            'boundary': boundary,
            'correct_a': correct_a,
            'correct_s': correct_s,
        },
    }


def _mask_panel(mask: np.ndarray) -> Image.Image:
    image = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(image, mode='L').convert('RGB')


def _error_panel(
    correct_a: np.ndarray,
    correct_s: np.ndarray,
    boundary: np.ndarray,
) -> Image.Image:
    panel = np.zeros(correct_a.shape + (3,), dtype=np.uint8)
    panel[correct_a & correct_s] = (45, 45, 45)
    panel[correct_a & ~correct_s] = (255, 166, 0)
    panel[~correct_a & correct_s] = (0, 190, 255)
    panel[~correct_a & ~correct_s] = (230, 40, 40)
    panel[boundary] = np.maximum(panel[boundary], np.asarray((70, 70, 70)))
    return Image.fromarray(panel, mode='RGB')


def _save_visualizations(
    output_dir: str,
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
    max_visualizations: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    count = min(len(names), max_visualizations)
    labels = ('GT', 'Appearance', 'Structure', 'Mean', 'Entropy', 'Errors')
    for index in range(count):
        panels = [
            _mask_panel(arrays['target'][index]),
            _mask_panel(arrays['prediction_a'][index]),
            _mask_panel(arrays['prediction_s'][index]),
            _mask_panel(arrays['prediction_mean'][index]),
            _mask_panel(arrays['prediction_entropy'][index]),
            _error_panel(
                arrays['correct_a'][index],
                arrays['correct_s'][index],
                arrays['boundary'][index],
            ),
        ]
        width, height = panels[0].size
        title_height = 20
        canvas = Image.new(
            'RGB',
            (width * len(panels), height + title_height),
            color='white',
        )
        draw = ImageDraw.Draw(canvas)
        for panel_index, (label, panel) in enumerate(zip(labels, panels)):
            x = panel_index * width
            canvas.paste(panel, (x, title_height))
            draw.text((x + 4, 3), label, fill='black')
        filename = os.path.splitext(os.path.basename(str(names[index])))[0]
        canvas.save(os.path.join(output_dir, filename + '.png'))


@dataclass
class ComplementarityDiagnostics:
    """Manage separate diagnostic metrics and best paired artifacts."""

    output_dir: str
    interval: int = 5
    boundary_width: int = 3
    max_visualizations: int = 8
    save_paired_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError('diagnostic interval must be at least one epoch')
        os.makedirs(self.output_dir, exist_ok=True)
        self.best_mean_jaccard = -1.0

    def should_run(self, epoch: int) -> bool:
        epoch_number = epoch + 1
        return epoch_number == 1 or epoch_number % self.interval == 0

    def update(
        self,
        epoch: int,
        models: Sequence[torch.nn.Module],
        validation: Mapping[str, object],
    ) -> dict[str, float] | None:
        if not self.should_run(epoch):
            return None

        result = compute_complementarity_diagnostics(
            validation['scores1'],
            validation['scores2'],
            validation['masks'],
            boundary_width=self.boundary_width,
        )
        summary = {'epoch': epoch + 1, **result['summary']}
        self._append_summary(summary)
        if summary['mean_jaccard'] > self.best_mean_jaccard:
            self.best_mean_jaccard = summary['mean_jaccard']
            self._save_best(epoch, models, validation, result, summary)
        return summary

    def _append_summary(self, summary: Mapping[str, float]) -> None:
        path = os.path.join(self.output_dir, 'diagnostics_metrics.csv')
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

    def _save_best(
        self,
        epoch: int,
        models: Sequence[torch.nn.Module],
        validation: Mapping[str, object],
        result: Mapping[str, object],
        summary: Mapping[str, float],
    ) -> None:
        metadata = {
            **summary,
            'selection_metric': 'mean_jaccard',
            'boundary_width': self.boundary_width,
            'entropy_fusion': 'exp(-categorical_entropy) normalized weights',
            'oracle_note': 'ground-truth upper bound; never used for training',
        }
        with open(
            os.path.join(self.output_dir, 'best_summary.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

        arrays = result['arrays']
        np.savez_compressed(
            os.path.join(self.output_dir, 'best_predictions.npz'),
            epoch=np.asarray(epoch + 1),
            names=np.asarray(validation['names']).astype(str),
            probability_a=arrays['probability_a'].astype(np.float16),
            probability_s=arrays['probability_s'].astype(np.float16),
            probability_mean=arrays['probability_mean'].astype(np.float16),
            probability_entropy=arrays['probability_entropy'].astype(np.float16),
            target=arrays['target'].astype(np.uint8),
            boundary=arrays['boundary'].astype(np.uint8),
        )
        self._save_per_image(result['per_image'], validation['names'])
        _save_visualizations(
            os.path.join(self.output_dir, 'visualizations'),
            arrays,
            validation['names'],
            self.max_visualizations,
        )
        if self.save_paired_checkpoints:
            torch.save(
                models[0].state_dict(),
                os.path.join(self.output_dir, 'best_pair_model1.pth'),
            )
            torch.save(
                models[1].state_dict(),
                os.path.join(self.output_dir, 'best_pair_model2.pth'),
            )

    def _save_per_image(
        self,
        rows: Sequence[Mapping[str, float]],
        names: Sequence[str],
    ) -> None:
        path = os.path.join(self.output_dir, 'best_per_image.csv')
        output_rows = []
        for name, row in zip(names, rows):
            output_rows.append({'name': str(name), **row})
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(output_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(output_rows)
