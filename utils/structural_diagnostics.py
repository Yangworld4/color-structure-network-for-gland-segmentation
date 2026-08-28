import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, label
from scipy.optimize import linear_sum_assignment


SUMMARY_FIELDS = [
    'epoch',
    'num_gt_components',
    'num_a_components',
    'num_b_components',
    'component_count_error_a',
    'component_count_error_b',
    'merge_error_a',
    'merge_error_b',
    'merge_affected_glands_a',
    'merge_affected_glands_b',
    'split_error_a',
    'split_error_b',
    'split_affected_glands_a',
    'split_affected_glands_b',
    'missed_gland_a',
    'missed_gland_b',
    'extra_gland_a',
    'extra_gland_b',
    'both_struct_correct_ratio',
    'a_only_struct_correct_ratio',
    'b_only_struct_correct_ratio',
    'both_struct_wrong_ratio',
    'pixel_complementary_ratio',
    'structural_complementary_ratio',
    'pixel_disagreement_ratio',
    'structure_critical_disagreement_ratio',
    'structure_critical_fraction_of_disagreement',
    'structure_critical_pixels',
    'total_disagreement_pixels',
    'disagreement_on_boundary_ratio',
    'disagreement_inside_gland_ratio',
    'disagreement_background_ratio',
    'a_correct_b_wrong_boundary_ratio',
    'a_wrong_b_correct_boundary_ratio',
    'a_complement_merge',
    'a_complement_split',
    'a_complement_miss',
    'a_complement_extra',
    'b_complement_merge',
    'b_complement_split',
    'b_complement_miss',
    'b_complement_extra',
]


PER_IMAGE_FIELDS = [
    'epoch',
    'image_id',
    'gt_components',
    'a_components',
    'b_components',
    'a_merge',
    'b_merge',
    'a_split',
    'b_split',
    'a_miss',
    'b_miss',
    'a_extra',
    'b_extra',
    'a_only_struct_correct',
    'b_only_struct_correct',
    'both_struct_correct',
    'both_struct_wrong',
    'pixel_disagreement_ratio',
    'structure_critical_disagreement_ratio',
    'boundary_disagreement_ratio',
]


EXAMPLE_CATEGORIES = (
    'a_correct_b_merge',
    'b_correct_a_merge',
    'a_correct_b_split',
    'b_correct_a_split',
    'a_correct_b_miss',
    'b_correct_a_miss',
    'both_wrong',
)


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _foreground_mask(logits):
    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits)[:, 0]
    else:
        prob = torch.softmax(logits, dim=1)[:, 1]
    return prob > 0.5


def _to_numpy_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask).astype(bool)


def _extract_components(mask, min_area):
    mask = np.asarray(mask).astype(bool)
    labeled, _count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    slices = find_objects(labeled)
    filtered = np.zeros_like(labeled, dtype=np.int32)
    components = []
    next_id = 1
    for original_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        component_mask = labeled == original_id
        area = int(component_mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0:
            continue
        filtered[component_mask] = next_id
        components.append({
            'id': next_id,
            'area': area,
            'mask': component_mask,
            'bbox': (
                int(ys.min()),
                int(xs.min()),
                int(ys.max()) + 1,
                int(xs.max()) + 1,
            ),
        })
        next_id += 1
    return components, filtered


def _iou_matrix(pred_components, gt_components):
    matrix = np.zeros((len(pred_components), len(gt_components)), dtype=np.float32)
    for pred_index, pred_component in enumerate(pred_components):
        pred_mask = pred_component['mask']
        for gt_index, gt_component in enumerate(gt_components):
            gt_mask = gt_component['mask']
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            matrix[pred_index, gt_index] = float(intersection / union) if union else 0.0
    return matrix


def _match_components(pred_components, gt_components, match_iou):
    matrix = _iou_matrix(pred_components, gt_components)
    if matrix.size == 0:
        return [], set(), set(), matrix
    rows, cols = linear_sum_assignment(-matrix)
    matches = []
    for row, col in zip(rows, cols):
        iou = float(matrix[row, col])
        if iou >= match_iou:
            matches.append((int(row), int(col), iou))
    matched_pred = {pred for pred, _gt, _iou in matches}
    matched_gt = {gt for _pred, gt, _iou in matches}
    return matches, matched_pred, matched_gt, matrix


def _boundary(mask, width):
    mask = np.asarray(mask).astype(bool)
    if width <= 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    dilated = mask
    eroded = mask
    for _ in range(width):
        dilated = binary_dilation(dilated, structure=structure)
        eroded = binary_erosion(eroded, structure=structure, border_value=0)
    return np.logical_xor(dilated, eroded)


def _component_error_sets(pred_components, gt_components, match_iou):
    matches, matched_pred, matched_gt, matrix = _match_components(
        pred_components, gt_components, match_iou
    )
    missed_gt = set(range(len(gt_components))) - matched_gt
    extra_pred = set(range(len(pred_components))) - matched_pred

    if matrix.size == 0:
        pred_to_gt = {}
        gt_to_pred = {}
    else:
        pred_to_gt = {
            pred_index: set(np.where(matrix[pred_index] >= match_iou)[0].tolist())
            for pred_index in range(matrix.shape[0])
        }
        gt_to_pred = {
            gt_index: set(np.where(matrix[:, gt_index] >= match_iou)[0].tolist())
            for gt_index in range(matrix.shape[1])
        }

    merge_pred = {
        pred_index for pred_index, gt_indices in pred_to_gt.items()
        if len(gt_indices) >= 2
    }
    split_gt = {
        gt_index for gt_index, pred_indices in gt_to_pred.items()
        if len(pred_indices) >= 2
    }
    merge_gt = set()
    for pred_index in merge_pred:
        merge_gt.update(pred_to_gt[pred_index])
    split_pred = set()
    for gt_index in split_gt:
        split_pred.update(gt_to_pred[gt_index])

    gt_errors = defaultdict(set)
    for gt_index in missed_gt:
        gt_errors[gt_index].add('miss')
    for gt_index in merge_gt:
        gt_errors[gt_index].add('merge')
    for gt_index in split_gt:
        gt_errors[gt_index].add('split')
    for pred_index in extra_pred:
        overlaps = set()
        if matrix.size:
            overlaps = set(np.where(matrix[pred_index] > 0.0)[0].tolist())
        for gt_index in overlaps:
            gt_errors[gt_index].add('extra')

    matched_gt_by_hungarian = {gt for _pred, gt, _iou in matches}
    struct_correct_gt = {
        gt_index for gt_index in range(len(gt_components))
        if (
            gt_index in matched_gt_by_hungarian
            and gt_index not in missed_gt
            and gt_index not in merge_gt
            and gt_index not in split_gt
        )
    }
    return {
        'matches': matches,
        'matrix': matrix,
        'matched_pred': matched_pred,
        'matched_gt': matched_gt,
        'missed_gt': missed_gt,
        'extra_pred': extra_pred,
        'merge_pred': merge_pred,
        'merge_gt': merge_gt,
        'split_gt': split_gt,
        'split_pred': split_pred,
        'gt_errors': gt_errors,
        'struct_correct_gt': struct_correct_gt,
    }


class StructuralDiagnostics:
    def __init__(
        self,
        output_dir,
        min_area=20,
        match_iou=0.1,
        boundary_width=3,
        save_examples=False,
        max_examples_per_category=5,
    ):
        self.output_dir = Path(output_dir)
        self.min_area = int(min_area)
        self.match_iou = float(match_iou)
        self.boundary_width = int(boundary_width)
        self.save_examples = bool(save_examples)
        self.max_examples_per_category = int(max_examples_per_category)
        self.summary_path = self.output_dir / 'structural_diagnostics.csv'
        self.per_image_path = self.output_dir / 'structural_diagnostics_per_image.csv'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.example_counts = {category: 0 for category in EXAMPLE_CATEGORIES}
        self._init_csvs()

    def _init_csvs(self):
        with self.summary_path.open('w', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS).writeheader()
        with self.per_image_path.open('w', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS).writeheader()

    def new_epoch(self, epoch, pixel_complementary_ratio=0.0):
        return {
            'epoch': int(epoch),
            'pixel_complementary_ratio': float(pixel_complementary_ratio),
            'num_gt_components': 0,
            'num_a_components': 0,
            'num_b_components': 0,
            'component_count_error_a': 0,
            'component_count_error_b': 0,
            'merge_error_a': 0,
            'merge_error_b': 0,
            'split_error_a': 0,
            'split_error_b': 0,
            'missed_gland_a': 0,
            'missed_gland_b': 0,
            'extra_gland_a': 0,
            'extra_gland_b': 0,
            'merge_affected_glands_a': 0,
            'merge_affected_glands_b': 0,
            'split_affected_glands_a': 0,
            'split_affected_glands_b': 0,
            'both_struct_correct': 0,
            'a_only_struct_correct': 0,
            'b_only_struct_correct': 0,
            'both_struct_wrong': 0,
            'gt_glands': 0,
            'pixels': 0,
            'pixel_disagreement_pixels': 0,
            'structure_critical_pixels': 0,
            'boundary_pixels': 0,
            'inside_pixels': 0,
            'background_pixels': 0,
            'boundary_disagreement_pixels': 0,
            'inside_disagreement_pixels': 0,
            'background_disagreement_pixels': 0,
            'a_correct_b_wrong_boundary': 0,
            'a_correct_b_wrong_pixels': 0,
            'a_wrong_b_correct_boundary': 0,
            'a_wrong_b_correct_pixels': 0,
            'a_complement_merge': 0,
            'a_complement_split': 0,
            'a_complement_miss': 0,
            'a_complement_extra': 0,
            'b_complement_merge': 0,
            'b_complement_split': 0,
            'b_complement_miss': 0,
            'b_complement_extra': 0,
        }

    def update_batch(
        self,
        epoch_state,
        logits_a,
        logits_b,
        gt,
        image_ids,
        image_a=None,
        image_b=None,
    ):
        with torch.no_grad():
            mask_a_batch = _foreground_mask(logits_a).detach().cpu().numpy()
            mask_b_batch = _foreground_mask(logits_b).detach().cpu().numpy()
            gt_batch = gt.detach().cpu().numpy() > 0

        for index, image_id in enumerate(image_ids):
            rgb = image_a[index].detach().cpu() if image_a is not None else None
            stain = image_b[index].detach().cpu() if image_b is not None else None
            row = self.update_image(
                epoch_state,
                int(epoch_state['epoch']),
                image_id,
                mask_a_batch[index],
                mask_b_batch[index],
                gt_batch[index],
                rgb,
                stain,
            )
            self._append_per_image(row)

    def update_image(
        self,
        epoch_state,
        epoch,
        image_id,
        mask_a,
        mask_b,
        mask_gt,
        image_a=None,
        image_b=None,
    ):
        mask_a = _to_numpy_mask(mask_a)
        mask_b = _to_numpy_mask(mask_b)
        mask_gt = _to_numpy_mask(mask_gt)

        components_gt, _cc_gt = _extract_components(mask_gt, self.min_area)
        components_a, _cc_a = _extract_components(mask_a, self.min_area)
        components_b, _cc_b = _extract_components(mask_b, self.min_area)
        errors_a = _component_error_sets(components_a, components_gt, self.match_iou)
        errors_b = _component_error_sets(components_b, components_gt, self.match_iou)

        num_gt = len(components_gt)
        num_a = len(components_a)
        num_b = len(components_b)
        epoch_state['num_gt_components'] += num_gt
        epoch_state['num_a_components'] += num_a
        epoch_state['num_b_components'] += num_b
        epoch_state['component_count_error_a'] += abs(num_a - num_gt)
        epoch_state['component_count_error_b'] += abs(num_b - num_gt)

        a_merge = len(errors_a['merge_pred'])
        b_merge = len(errors_b['merge_pred'])
        a_split = len(errors_a['split_gt'])
        b_split = len(errors_b['split_gt'])
        a_miss = len(errors_a['missed_gt'])
        b_miss = len(errors_b['missed_gt'])
        a_extra = len(errors_a['extra_pred'])
        b_extra = len(errors_b['extra_pred'])

        epoch_state['merge_error_a'] += a_merge
        epoch_state['merge_error_b'] += b_merge
        epoch_state['split_error_a'] += a_split
        epoch_state['split_error_b'] += b_split
        epoch_state['missed_gland_a'] += a_miss
        epoch_state['missed_gland_b'] += b_miss
        epoch_state['extra_gland_a'] += a_extra
        epoch_state['extra_gland_b'] += b_extra
        epoch_state['merge_affected_glands_a'] += len(errors_a['merge_gt'])
        epoch_state['merge_affected_glands_b'] += len(errors_b['merge_gt'])
        epoch_state['split_affected_glands_a'] += len(errors_a['split_gt'])
        epoch_state['split_affected_glands_b'] += len(errors_b['split_gt'])
        epoch_state['gt_glands'] += num_gt

        a_correct_set = errors_a['struct_correct_gt']
        b_correct_set = errors_b['struct_correct_gt']
        both_correct = 0
        a_only = 0
        b_only = 0
        both_wrong = 0
        affected_gt = set()

        for gt_index in range(num_gt):
            a_correct = gt_index in a_correct_set
            b_correct = gt_index in b_correct_set
            if a_correct and b_correct:
                both_correct += 1
            elif a_correct and not b_correct:
                a_only += 1
                self._count_complement(epoch_state, 'a', errors_b['gt_errors'][gt_index])
                affected_gt.add(gt_index)
            elif b_correct and not a_correct:
                b_only += 1
                self._count_complement(epoch_state, 'b', errors_a['gt_errors'][gt_index])
                affected_gt.add(gt_index)
            else:
                both_wrong += 1
                affected_gt.add(gt_index)
            if errors_a['gt_errors'].get(gt_index) != errors_b['gt_errors'].get(gt_index):
                affected_gt.add(gt_index)

        epoch_state['both_struct_correct'] += both_correct
        epoch_state['a_only_struct_correct'] += a_only
        epoch_state['b_only_struct_correct'] += b_only
        epoch_state['both_struct_wrong'] += both_wrong

        disagreement = np.logical_xor(mask_a, mask_b)
        boundary_gt = _boundary(mask_gt, self.boundary_width)
        inside_gt = mask_gt & ~boundary_gt
        background = ~mask_gt
        structure_critical = self._structure_critical_mask(
            disagreement,
            components_gt,
            components_a,
            components_b,
            errors_a,
            errors_b,
            affected_gt,
            num_a != num_b,
        )

        pixels = int(mask_gt.size)
        disagreement_pixels = int(disagreement.sum())
        structure_critical_pixels = int(structure_critical.sum())
        boundary_disagreement = int((disagreement & boundary_gt).sum())

        epoch_state['pixels'] += pixels
        epoch_state['pixel_disagreement_pixels'] += disagreement_pixels
        epoch_state['structure_critical_pixels'] += structure_critical_pixels
        epoch_state['boundary_pixels'] += int(boundary_gt.sum())
        epoch_state['inside_pixels'] += int(inside_gt.sum())
        epoch_state['background_pixels'] += int(background.sum())
        epoch_state['boundary_disagreement_pixels'] += boundary_disagreement
        epoch_state['inside_disagreement_pixels'] += int((disagreement & inside_gt).sum())
        epoch_state['background_disagreement_pixels'] += int((disagreement & background).sum())

        a_correct_b_wrong_pixels = self._gt_union_mask(mask_gt.shape, components_gt, [
            gt_index for gt_index in range(num_gt)
            if gt_index in a_correct_set and gt_index not in b_correct_set
        ])
        b_correct_a_wrong_pixels = self._gt_union_mask(mask_gt.shape, components_gt, [
            gt_index for gt_index in range(num_gt)
            if gt_index in b_correct_set and gt_index not in a_correct_set
        ])
        a_c_b_w_disagree = disagreement & a_correct_b_wrong_pixels
        b_c_a_w_disagree = disagreement & b_correct_a_wrong_pixels
        epoch_state['a_correct_b_wrong_boundary'] += int((a_c_b_w_disagree & boundary_gt).sum())
        epoch_state['a_correct_b_wrong_pixels'] += int(a_c_b_w_disagree.sum())
        epoch_state['a_wrong_b_correct_boundary'] += int((b_c_a_w_disagree & boundary_gt).sum())
        epoch_state['a_wrong_b_correct_pixels'] += int(b_c_a_w_disagree.sum())

        example_category = self._example_category(a_only, b_only, both_wrong, errors_a, errors_b)
        if example_category is not None:
            self._save_example(
                example_category,
                epoch,
                image_id,
                image_a,
                image_b,
                mask_gt,
                mask_a,
                mask_b,
                disagreement,
                num_gt,
                num_a,
                num_b,
            )

        return {
            'epoch': epoch,
            'image_id': image_id,
            'gt_components': num_gt,
            'a_components': num_a,
            'b_components': num_b,
            'a_merge': a_merge,
            'b_merge': b_merge,
            'a_split': a_split,
            'b_split': b_split,
            'a_miss': a_miss,
            'b_miss': b_miss,
            'a_extra': a_extra,
            'b_extra': b_extra,
            'a_only_struct_correct': a_only,
            'b_only_struct_correct': b_only,
            'both_struct_correct': both_correct,
            'both_struct_wrong': both_wrong,
            'pixel_disagreement_ratio': _safe_ratio(disagreement_pixels, pixels),
            'structure_critical_disagreement_ratio': _safe_ratio(
                structure_critical_pixels, pixels
            ),
            'boundary_disagreement_ratio': _safe_ratio(
                boundary_disagreement, max(int(boundary_gt.sum()), 1)
            ),
        }

    def _count_complement(self, epoch_state, owner, errors):
        if not errors:
            epoch_state['{}_complement_extra'.format(owner)] += 0
            return
        for error_name in ('merge', 'split', 'miss', 'extra'):
            if error_name in errors:
                epoch_state['{}_complement_{}'.format(owner, error_name)] += 1

    def _gt_union_mask(self, shape, components_gt, gt_indices):
        if not components_gt:
            return np.zeros(shape, dtype=bool)
        output = np.zeros_like(components_gt[0]['mask'], dtype=bool)
        for gt_index in gt_indices:
            output |= components_gt[gt_index]['mask']
        return output

    def _structure_critical_mask(
        self,
        disagreement,
        components_gt,
        components_a,
        components_b,
        errors_a,
        errors_b,
        affected_gt,
        component_number_differs,
    ):
        if not disagreement.any():
            return np.zeros_like(disagreement, dtype=bool)
        if component_number_differs:
            critical = disagreement.copy()
        else:
            critical = np.zeros_like(disagreement, dtype=bool)
        for gt_index in affected_gt:
            critical |= disagreement & components_gt[gt_index]['mask']
        for errors, components in ((errors_a, components_a), (errors_b, components_b)):
            affected_pred = (
                errors['extra_pred']
                | errors['merge_pred']
                | errors['split_pred']
            )
            for pred_index in affected_pred:
                critical |= disagreement & components[pred_index]['mask']
        return critical

    def _example_category(self, a_only, b_only, both_wrong, errors_a, errors_b):
        if a_only > 0:
            if errors_b['merge_pred']:
                return 'a_correct_b_merge'
            if errors_b['split_gt']:
                return 'a_correct_b_split'
            if errors_b['missed_gt']:
                return 'a_correct_b_miss'
        if b_only > 0:
            if errors_a['merge_pred']:
                return 'b_correct_a_merge'
            if errors_a['split_gt']:
                return 'b_correct_a_split'
            if errors_a['missed_gt']:
                return 'b_correct_a_miss'
        if both_wrong > 0:
            return 'both_wrong'
        return None

    def _save_example(
        self,
        category,
        epoch,
        image_id,
        image_a,
        image_b,
        mask_gt,
        mask_a,
        mask_b,
        disagreement,
        num_gt,
        num_a,
        num_b,
    ):
        if not self.save_examples:
            return
        if self.example_counts[category] >= self.max_examples_per_category:
            return
        category_dir = self.output_dir / 'struct_examples' / category
        category_dir.mkdir(parents=True, exist_ok=True)
        panels = [
            self._image_panel(image_a),
            self._image_panel(image_b),
            self._mask_panel(mask_gt, (0, 220, 0)),
            self._mask_panel(mask_a, (230, 40, 40)),
            self._mask_panel(mask_b, (40, 90, 230)),
            self._mask_panel(disagreement, (240, 220, 0)),
        ]
        height, width = panels[0].shape[:2]
        title_height = 28
        canvas = np.full(
            (height + title_height, width * len(panels), 3),
            255,
            dtype=np.uint8,
        )
        for panel_index, panel in enumerate(panels):
            canvas[title_height:, panel_index * width:(panel_index + 1) * width] = panel
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text(
            (4, 6),
            'A components={}  B components={}  GT components={}  error type={}'.format(
                num_a, num_b, num_gt, category
            ),
            fill=(0, 0, 0),
        )
        safe_id = Path(str(image_id)).stem.replace(os.sep, '_')
        path = category_dir / 'epoch_{:03d}_{}.png'.format(epoch, safe_id)
        image.save(path)
        self.example_counts[category] += 1

    def _image_panel(self, tensor):
        if tensor is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        array = tensor.float().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        array = array.astype(np.float32)
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return array[:, :, :3]

    def _mask_panel(self, mask, color):
        mask = np.asarray(mask).astype(bool)
        panel = np.zeros(mask.shape + (3,), dtype=np.uint8)
        panel[mask] = color
        return panel

    def _append_per_image(self, row):
        with self.per_image_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS)
            writer.writerow(row)

    def finalize_epoch(self, epoch_state):
        gt_glands = epoch_state['gt_glands']
        pixels = epoch_state['pixels']
        disagreement_pixels = epoch_state['pixel_disagreement_pixels']
        summary = {
            'epoch': epoch_state['epoch'],
            'num_gt_components': epoch_state['num_gt_components'],
            'num_a_components': epoch_state['num_a_components'],
            'num_b_components': epoch_state['num_b_components'],
            'component_count_error_a': epoch_state['component_count_error_a'],
            'component_count_error_b': epoch_state['component_count_error_b'],
            'merge_error_a': epoch_state['merge_error_a'],
            'merge_error_b': epoch_state['merge_error_b'],
            'merge_affected_glands_a': epoch_state['merge_affected_glands_a'],
            'merge_affected_glands_b': epoch_state['merge_affected_glands_b'],
            'split_error_a': epoch_state['split_error_a'],
            'split_error_b': epoch_state['split_error_b'],
            'split_affected_glands_a': epoch_state['split_affected_glands_a'],
            'split_affected_glands_b': epoch_state['split_affected_glands_b'],
            'missed_gland_a': epoch_state['missed_gland_a'],
            'missed_gland_b': epoch_state['missed_gland_b'],
            'extra_gland_a': epoch_state['extra_gland_a'],
            'extra_gland_b': epoch_state['extra_gland_b'],
            'both_struct_correct_ratio': _safe_ratio(
                epoch_state['both_struct_correct'], gt_glands
            ),
            'a_only_struct_correct_ratio': _safe_ratio(
                epoch_state['a_only_struct_correct'], gt_glands
            ),
            'b_only_struct_correct_ratio': _safe_ratio(
                epoch_state['b_only_struct_correct'], gt_glands
            ),
            'both_struct_wrong_ratio': _safe_ratio(
                epoch_state['both_struct_wrong'], gt_glands
            ),
            'pixel_complementary_ratio': epoch_state['pixel_complementary_ratio'],
            'structural_complementary_ratio': _safe_ratio(
                epoch_state['a_only_struct_correct']
                + epoch_state['b_only_struct_correct'],
                gt_glands,
            ),
            'pixel_disagreement_ratio': _safe_ratio(disagreement_pixels, pixels),
            'structure_critical_disagreement_ratio': _safe_ratio(
                epoch_state['structure_critical_pixels'], pixels
            ),
            'structure_critical_fraction_of_disagreement': _safe_ratio(
                epoch_state['structure_critical_pixels'], disagreement_pixels
            ),
            'structure_critical_pixels': epoch_state['structure_critical_pixels'],
            'total_disagreement_pixels': disagreement_pixels,
            'disagreement_on_boundary_ratio': _safe_ratio(
                epoch_state['boundary_disagreement_pixels'],
                epoch_state['boundary_pixels'],
            ),
            'disagreement_inside_gland_ratio': _safe_ratio(
                epoch_state['inside_disagreement_pixels'],
                epoch_state['inside_pixels'],
            ),
            'disagreement_background_ratio': _safe_ratio(
                epoch_state['background_disagreement_pixels'],
                epoch_state['background_pixels'],
            ),
            'a_correct_b_wrong_boundary_ratio': _safe_ratio(
                epoch_state['a_correct_b_wrong_boundary'],
                epoch_state['a_correct_b_wrong_pixels'],
            ),
            'a_wrong_b_correct_boundary_ratio': _safe_ratio(
                epoch_state['a_wrong_b_correct_boundary'],
                epoch_state['a_wrong_b_correct_pixels'],
            ),
            'a_complement_merge': epoch_state['a_complement_merge'],
            'a_complement_split': epoch_state['a_complement_split'],
            'a_complement_miss': epoch_state['a_complement_miss'],
            'a_complement_extra': epoch_state['a_complement_extra'],
            'b_complement_merge': epoch_state['b_complement_merge'],
            'b_complement_split': epoch_state['b_complement_split'],
            'b_complement_miss': epoch_state['b_complement_miss'],
            'b_complement_extra': epoch_state['b_complement_extra'],
        }
        with self.summary_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writerow(summary)
        return summary
