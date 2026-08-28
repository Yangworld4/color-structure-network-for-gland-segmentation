import argparse
import csv
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader

from config.dataset_config.dataset_cfg import dataset_cfg
from main import (
    _load_state_dict,
    _resolve_struct_checkpoint_pair,
    build_datasets,
    get_training_views,
    init_seeds,
)
from models.getnetwork import get_network
from utils.ph_h0 import compute_h0_pairs_superlevel
from utils.structural_diagnostics import _component_error_sets, _extract_components


EVENT_FIELDS = [
    'epoch',
    'checkpoint',
    'image_id',
    'event_id',
    'event_type',
    'oracle_correct_branch',
    'wrong_branch',
    'gt_component_ids',
    'a_component_ids',
    'b_component_ids',
    'num_related_ph_pairs_a',
    'num_related_ph_pairs_b',
    'birth_a',
    'death_a',
    'persistence_a',
    'birth_y_a',
    'birth_x_a',
    'death_y_a',
    'death_x_a',
    'birth_b',
    'death_b',
    'persistence_b',
    'birth_y_b',
    'birth_x_b',
    'death_y_b',
    'death_x_b',
    'correct_structure_persistence',
    'wrong_structure_persistence',
    'correct_structure_higher_persistence',
    'correct_death_on_event_roi',
    'wrong_death_on_event_roi',
    'correct_death_on_disagreement',
    'wrong_death_on_disagreement',
    'correct_death_on_gt_error',
    'wrong_death_on_gt_error',
    'correct_death_on_gt_boundary',
    'wrong_death_on_gt_boundary',
    'counterfactual_fix_success',
    'event_roi_pixels',
    'event_disagreement_pixels',
    'ph_event_matched',
]


SUMMARY_FIELDS = [
    'epoch',
    'checkpoint',
    'total_merge_events',
    'total_split_events',
    'total_structural_events',
    'ph_matched_event_count',
    'ph_match_rate',
    'death_on_event_roi_rate',
    'death_on_disagreement_rate',
    'death_on_gt_error_rate',
    'death_on_gt_boundary_rate',
    'merge_death_on_gt_error_rate',
    'split_death_on_gt_error_rate',
    'counterfactual_fix_rate',
    'merge_counterfactual_fix_rate',
    'split_counterfactual_fix_rate',
    'mean_persistence_correct_structure',
    'mean_persistence_wrong_structure',
    'correct_structure_higher_persistence_rate',
    'mean_num_h0_pairs_a',
    'mean_num_h0_pairs_b',
    'mean_persistence_a',
    'mean_persistence_b',
    'unmatched_ph_event_count',
]


def _ids(indices, components):
    return [int(components[index]['id']) for index in sorted(indices)]


def _mask_union(shape, components, indices):
    output = np.zeros(shape, dtype=bool)
    for index in sorted(indices):
        output |= components[index]['mask']
    return output


def _indices_over_gt(errors, gt_index, min_iou=True):
    if errors['matrix'].size == 0:
        return set()
    threshold = 0.0 if not min_iou else None
    if threshold is None:
        return set(np.where(errors['matrix'][:, gt_index] >= 1e-12)[0].tolist())
    return set(np.where(errors['matrix'][:, gt_index] > threshold)[0].tolist())


def _pred_indices_for_gt(errors, gt_indices, match_iou):
    if errors['matrix'].size == 0:
        return set()
    pred = set()
    for gt_index in gt_indices:
        pred.update(np.where(errors['matrix'][:, gt_index] >= match_iou)[0].tolist())
    return pred


def _gt_indices_for_pred(errors, pred_index, match_iou):
    if errors['matrix'].size == 0:
        return set()
    return set(np.where(errors['matrix'][pred_index] >= match_iou)[0].tolist())


def _boundary(mask, width):
    if width <= 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    dilated = mask.astype(bool)
    eroded = mask.astype(bool)
    for _ in range(width):
        dilated = binary_dilation(dilated, structure=structure)
        eroded = binary_erosion(eroded, structure=structure, border_value=0)
    return np.logical_xor(dilated, eroded)


def _dilate(mask, radius):
    output = mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, int(radius))):
        output = binary_dilation(output, structure=structure)
    return output


def _in_mask(coord, mask):
    if coord is None:
        return False
    y, x = coord
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _as_list(value):
    return ';'.join(str(v) for v in value)


class PHH0StructuralAnalyzer:
    def __init__(
        self,
        output_dir,
        min_area=20,
        match_iou=0.1,
        roi_dilation=5,
        boundary_width=3,
        critical_radius=2,
        save_examples=False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_area = int(min_area)
        self.match_iou = float(match_iou)
        self.roi_dilation = int(roi_dilation)
        self.boundary_width = int(boundary_width)
        self.critical_radius = int(critical_radius)
        self.save_examples = bool(save_examples)
        self.event_csv = self.output_dir / 'ph_h0_structural_events.csv'
        self.summary_csv = self.output_dir / 'ph_h0_structural_summary.csv'
        self.pairs_jsonl = self.output_dir / 'ph_h0_pairs.jsonl'
        self.example_counts = {'merge': 0, 'split': 0}
        self._init_csvs()

    def _init_csvs(self):
        with self.event_csv.open('w', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=EVENT_FIELDS).writeheader()
        with self.summary_csv.open('w', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS).writeheader()
        with self.pairs_jsonl.open('w', encoding='utf-8') as handle:
            handle.write('')

    def analyze_checkpoint(self, models, dataloader, device, epoch, checkpoint_label):
        model_a, model_b = models
        model_a.eval()
        model_b.eval()
        state = self._new_state(epoch, checkpoint_label)
        event_id = 0
        with torch.no_grad():
            for batch in dataloader:
                input_a, input_b = get_training_views(batch)
                input_a = input_a.to(device, non_blocking=True)
                input_b = input_b.to(device, non_blocking=True)
                gt = batch['mask'].to(device, non_blocking=True)
                logits_a = self._extract_logits(model_a(input_a))
                logits_b = self._extract_logits(model_b(input_b))
                prob_a = torch.softmax(logits_a, dim=1)[:, 1].detach().cpu().numpy()
                prob_b = torch.softmax(logits_b, dim=1)[:, 1].detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy() > 0
                image_a = input_a.detach().cpu()
                image_b = input_b.detach().cpu()

                for index, image_id in enumerate(batch['ID']):
                    event_id = self._analyze_image(
                        state,
                        event_id,
                        int(epoch),
                        checkpoint_label,
                        str(image_id),
                        prob_a[index],
                        prob_b[index],
                        gt_np[index],
                        image_a[index],
                        image_b[index],
                    )
        summary = self._finalize(state)
        with self.summary_csv.open('a', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS).writerow(summary)
        return summary

    def _extract_logits(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _new_state(self, epoch, checkpoint):
        return {
            'epoch': int(epoch),
            'checkpoint': checkpoint,
            'events': 0,
            'merge_events': 0,
            'split_events': 0,
            'matched': 0,
            'wrong_death_event_roi': 0,
            'wrong_death_disagreement': 0,
            'wrong_death_gt_error': 0,
            'wrong_death_boundary': 0,
            'merge_wrong_death_gt_error': 0,
            'split_wrong_death_gt_error': 0,
            'counterfactual_fix': 0,
            'merge_counterfactual_fix': 0,
            'split_counterfactual_fix': 0,
            'higher_persistence': 0,
            'persistence_correct': [],
            'persistence_wrong': [],
            'num_h0_pairs_a': [],
            'num_h0_pairs_b': [],
            'persistence_a_all': [],
            'persistence_b_all': [],
        }

    def _analyze_image(
        self,
        state,
        event_id,
        epoch,
        checkpoint,
        image_id,
        prob_a,
        prob_b,
        gt,
        image_a,
        image_b,
    ):
        pred_a = prob_a > 0.5
        pred_b = prob_b > 0.5
        components_gt, _ = _extract_components(gt, self.min_area)
        components_a, _ = _extract_components(pred_a, self.min_area)
        components_b, _ = _extract_components(pred_b, self.min_area)
        errors_a = _component_error_sets(components_a, components_gt, self.match_iou)
        errors_b = _component_error_sets(components_b, components_gt, self.match_iou)
        ph_a, essential_a = compute_h0_pairs_superlevel(prob_a)
        ph_b, essential_b = compute_h0_pairs_superlevel(prob_b)
        self._append_pairs(epoch, checkpoint, image_id, ph_a, ph_b, essential_a, essential_b)
        state['num_h0_pairs_a'].append(len(ph_a))
        state['num_h0_pairs_b'].append(len(ph_b))
        state['persistence_a_all'].extend(pair['persistence'] for pair in ph_a)
        state['persistence_b_all'].extend(pair['persistence'] for pair in ph_b)

        events = self._structural_events(
            gt.shape,
            components_gt,
            components_a,
            components_b,
            errors_a,
            errors_b,
        )
        disagreement = np.logical_xor(pred_a, pred_b)
        boundary_gt = _boundary(gt, self.boundary_width)
        for event in events:
            event_id += 1
            row = self._analyze_event(
                epoch,
                checkpoint,
                image_id,
                event_id,
                event,
                prob_a,
                prob_b,
                pred_a,
                pred_b,
                gt,
                disagreement,
                boundary_gt,
                ph_a,
                ph_b,
                image_a,
                image_b,
            )
            self._append_event(row)
            self._accumulate(state, row)
        return event_id

    def _structural_events(
        self,
        shape,
        components_gt,
        components_a,
        components_b,
        errors_a,
        errors_b,
    ):
        events = []
        for pred_index in sorted(errors_b['merge_pred']):
            gt_indices = _gt_indices_for_pred(errors_b, pred_index, self.match_iou)
            if gt_indices and gt_indices.issubset(errors_a['struct_correct_gt']):
                events.append(self._make_event(
                    'merge', 'A', 'B', shape, gt_indices,
                    _pred_indices_for_gt(errors_a, gt_indices, self.match_iou),
                    {pred_index}, components_gt, components_a, components_b,
                ))
        for pred_index in sorted(errors_a['merge_pred']):
            gt_indices = _gt_indices_for_pred(errors_a, pred_index, self.match_iou)
            if gt_indices and gt_indices.issubset(errors_b['struct_correct_gt']):
                events.append(self._make_event(
                    'merge', 'B', 'A', shape, gt_indices,
                    {pred_index},
                    _pred_indices_for_gt(errors_b, gt_indices, self.match_iou),
                    components_gt, components_a, components_b,
                ))
        for gt_index in sorted(errors_b['split_gt']):
            if gt_index in errors_a['struct_correct_gt']:
                pred_b = set(errors_b['split_pred']) & _pred_indices_for_gt(errors_b, {gt_index}, self.match_iou)
                events.append(self._make_event(
                    'split', 'A', 'B', shape, {gt_index},
                    _pred_indices_for_gt(errors_a, {gt_index}, self.match_iou),
                    pred_b, components_gt, components_a, components_b,
                ))
        for gt_index in sorted(errors_a['split_gt']):
            if gt_index in errors_b['struct_correct_gt']:
                pred_a = set(errors_a['split_pred']) & _pred_indices_for_gt(errors_a, {gt_index}, self.match_iou)
                events.append(self._make_event(
                    'split', 'B', 'A', shape, {gt_index},
                    pred_a,
                    _pred_indices_for_gt(errors_b, {gt_index}, self.match_iou),
                    components_gt, components_a, components_b,
                ))
        return events

    def _make_event(
        self,
        event_type,
        correct_branch,
        wrong_branch,
        shape,
        gt_indices,
        a_indices,
        b_indices,
        components_gt,
        components_a,
        components_b,
    ):
        base_roi = (
            _mask_union(shape, components_gt, gt_indices)
            | _mask_union(shape, components_a, a_indices)
            | _mask_union(shape, components_b, b_indices)
        )
        return {
            'event_type': event_type,
            'correct_branch': correct_branch,
            'wrong_branch': wrong_branch,
            'gt_indices': set(gt_indices),
            'a_indices': set(a_indices),
            'b_indices': set(b_indices),
            'gt_component_ids': _ids(gt_indices, components_gt),
            'a_component_ids': _ids(a_indices, components_a),
            'b_component_ids': _ids(b_indices, components_b),
            'roi': _dilate(base_roi, self.roi_dilation),
        }

    def _analyze_event(
        self,
        epoch,
        checkpoint,
        image_id,
        event_id,
        event,
        prob_a,
        prob_b,
        pred_a,
        pred_b,
        gt,
        disagreement,
        boundary_gt,
        ph_a,
        ph_b,
        image_a,
        image_b,
    ):
        roi = event['roi']
        disagreement_roi = roi & disagreement
        selected_a, related_a = self._select_pair(ph_a, roi, disagreement_roi)
        selected_b, related_b = self._select_pair(ph_b, roi, disagreement_roi)
        correct_pair = selected_a if event['correct_branch'] == 'A' else selected_b
        wrong_pair = selected_a if event['wrong_branch'] == 'A' else selected_b
        wrong_pred = pred_a if event['wrong_branch'] == 'A' else pred_b
        correct_prob = prob_a if event['correct_branch'] == 'A' else prob_b
        wrong_prob = prob_a if event['wrong_branch'] == 'A' else prob_b
        wrong_error = np.logical_xor(wrong_pred, gt)
        correct_error = np.logical_xor(
            pred_a if event['correct_branch'] == 'A' else pred_b,
            gt,
        )
        cf_success = self._counterfactual(
            event['event_type'], wrong_pair, wrong_prob, correct_prob, wrong_pred, gt
        )
        row = {
            'epoch': epoch,
            'checkpoint': checkpoint,
            'image_id': image_id,
            'event_id': event_id,
            'event_type': event['event_type'],
            'oracle_correct_branch': event['correct_branch'],
            'wrong_branch': event['wrong_branch'],
            'gt_component_ids': _as_list(event['gt_component_ids']),
            'a_component_ids': _as_list(event['a_component_ids']),
            'b_component_ids': _as_list(event['b_component_ids']),
            'num_related_ph_pairs_a': len(related_a),
            'num_related_ph_pairs_b': len(related_b),
        }
        row.update(self._pair_fields('a', selected_a))
        row.update(self._pair_fields('b', selected_b))
        cp = self._persistence(correct_pair)
        wp = self._persistence(wrong_pair)
        row.update({
            'correct_structure_persistence': cp,
            'wrong_structure_persistence': wp,
            'correct_structure_higher_persistence': int(cp != '' and wp != '' and float(cp) > float(wp)),
            'correct_death_on_event_roi': int(_in_mask(self._death_coord(correct_pair), roi)),
            'wrong_death_on_event_roi': int(_in_mask(self._death_coord(wrong_pair), roi)),
            'correct_death_on_disagreement': int(_in_mask(self._death_coord(correct_pair), disagreement)),
            'wrong_death_on_disagreement': int(_in_mask(self._death_coord(wrong_pair), disagreement)),
            'correct_death_on_gt_error': int(_in_mask(self._death_coord(correct_pair), correct_error)),
            'wrong_death_on_gt_error': int(_in_mask(self._death_coord(wrong_pair), wrong_error)),
            'correct_death_on_gt_boundary': int(_in_mask(self._death_coord(correct_pair), boundary_gt)),
            'wrong_death_on_gt_boundary': int(_in_mask(self._death_coord(wrong_pair), boundary_gt)),
            'counterfactual_fix_success': int(cf_success),
            'event_roi_pixels': int(roi.sum()),
            'event_disagreement_pixels': int(disagreement_roi.sum()),
            'ph_event_matched': int(len(related_a) > 0 or len(related_b) > 0),
        })
        if self.save_examples:
            self._save_example(
                row, event, image_a, image_b, gt, pred_a, pred_b, disagreement,
                roi, wrong_error, boundary_gt, selected_a, selected_b,
            )
        return row

    def _select_pair(self, pairs, roi, disagreement_roi):
        related = [
            pair for pair in pairs
            if _in_mask(pair['birth_coord'], roi) or _in_mask(pair['death_coord'], roi)
        ]
        if not related:
            return None, []
        in_disagreement = [
            pair for pair in related if _in_mask(pair['death_coord'], disagreement_roi)
        ]
        if in_disagreement:
            return max(in_disagreement, key=lambda pair: pair['persistence']), related
        if disagreement_roi.any():
            distance = distance_transform_edt(~disagreement_roi)
            return min(
                related,
                key=lambda pair: (
                    float(distance[pair['death_coord'][0], pair['death_coord'][1]]),
                    -float(pair['persistence']),
                ),
            ), related
        return max(related, key=lambda pair: pair['persistence']), related

    def _pair_fields(self, suffix, pair):
        if pair is None:
            return {
                'birth_{}'.format(suffix): '',
                'death_{}'.format(suffix): '',
                'persistence_{}'.format(suffix): '',
                'birth_y_{}'.format(suffix): '',
                'birth_x_{}'.format(suffix): '',
                'death_y_{}'.format(suffix): '',
                'death_x_{}'.format(suffix): '',
            }
        by, bx = pair['birth_coord']
        dy, dx = pair['death_coord']
        return {
            'birth_{}'.format(suffix): pair['birth'],
            'death_{}'.format(suffix): pair['death'],
            'persistence_{}'.format(suffix): pair['persistence'],
            'birth_y_{}'.format(suffix): by,
            'birth_x_{}'.format(suffix): bx,
            'death_y_{}'.format(suffix): dy,
            'death_x_{}'.format(suffix): dx,
        }

    def _persistence(self, pair):
        return '' if pair is None else float(pair['persistence'])

    def _death_coord(self, pair):
        return None if pair is None else pair['death_coord']

    def _counterfactual(self, event_type, wrong_pair, wrong_prob, correct_prob, wrong_pred, gt):
        if wrong_pair is None:
            return False
        y, x = wrong_pair['death_coord']
        h, w = wrong_prob.shape
        y0 = max(0, y - self.critical_radius)
        y1 = min(h, y + self.critical_radius + 1)
        x0 = max(0, x - self.critical_radius)
        x1 = min(w, x + self.critical_radius + 1)
        edited = wrong_prob.copy()
        if event_type == 'merge':
            edited[y0:y1, x0:x1] = np.minimum(
                edited[y0:y1, x0:x1], correct_prob[y0:y1, x0:x1]
            )
        else:
            edited[y0:y1, x0:x1] = np.maximum(
                edited[y0:y1, x0:x1], correct_prob[y0:y1, x0:x1]
            )
        before = wrong_pred.astype(bool)
        after = edited > 0.5
        before_components, _ = _extract_components(before, self.min_area)
        after_components, _ = _extract_components(after, self.min_area)
        gt_components, _ = _extract_components(gt, self.min_area)
        before_errors = _component_error_sets(before_components, gt_components, self.match_iou)
        after_errors = _component_error_sets(after_components, gt_components, self.match_iou)
        if event_type == 'merge' and len(after_errors['merge_pred']) < len(before_errors['merge_pred']):
            return True
        if event_type == 'split' and len(after_errors['split_gt']) < len(before_errors['split_gt']):
            return True
        gt_count = len(gt_components)
        return abs(len(after_components) - gt_count) < abs(len(before_components) - gt_count)

    def _append_event(self, row):
        with self.event_csv.open('a', newline='', encoding='utf-8') as handle:
            csv.DictWriter(handle, fieldnames=EVENT_FIELDS).writerow(row)

    def _append_pairs(self, epoch, checkpoint, image_id, ph_a, ph_b, essential_a, essential_b):
        record = {
            'epoch': epoch,
            'checkpoint': checkpoint,
            'image_id': image_id,
            'ph_a': ph_a,
            'ph_b': ph_b,
            'essential_count_a': int(essential_a),
            'essential_count_b': int(essential_b),
        }
        with self.pairs_jsonl.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def _accumulate(self, state, row):
        state['events'] += 1
        if row['event_type'] == 'merge':
            state['merge_events'] += 1
        else:
            state['split_events'] += 1
        state['matched'] += int(row['ph_event_matched'])
        state['wrong_death_event_roi'] += int(row['wrong_death_on_event_roi'])
        state['wrong_death_disagreement'] += int(row['wrong_death_on_disagreement'])
        state['wrong_death_gt_error'] += int(row['wrong_death_on_gt_error'])
        state['wrong_death_boundary'] += int(row['wrong_death_on_gt_boundary'])
        if row['event_type'] == 'merge':
            state['merge_wrong_death_gt_error'] += int(row['wrong_death_on_gt_error'])
            state['merge_counterfactual_fix'] += int(row['counterfactual_fix_success'])
        else:
            state['split_wrong_death_gt_error'] += int(row['wrong_death_on_gt_error'])
            state['split_counterfactual_fix'] += int(row['counterfactual_fix_success'])
        state['counterfactual_fix'] += int(row['counterfactual_fix_success'])
        if row['correct_structure_persistence'] != '' and row['wrong_structure_persistence'] != '':
            state['higher_persistence'] += int(row['correct_structure_higher_persistence'])
            state['persistence_correct'].append(float(row['correct_structure_persistence']))
            state['persistence_wrong'].append(float(row['wrong_structure_persistence']))

    def _finalize(self, state):
        events = state['events']
        merge = state['merge_events']
        split = state['split_events']
        persistence_n = len(state['persistence_correct'])
        return {
            'epoch': state['epoch'],
            'checkpoint': state['checkpoint'],
            'total_merge_events': merge,
            'total_split_events': split,
            'total_structural_events': events,
            'ph_matched_event_count': state['matched'],
            'ph_match_rate': self._ratio(state['matched'], events),
            'death_on_event_roi_rate': self._ratio(state['wrong_death_event_roi'], events),
            'death_on_disagreement_rate': self._ratio(state['wrong_death_disagreement'], events),
            'death_on_gt_error_rate': self._ratio(state['wrong_death_gt_error'], events),
            'death_on_gt_boundary_rate': self._ratio(state['wrong_death_boundary'], events),
            'merge_death_on_gt_error_rate': self._ratio(state['merge_wrong_death_gt_error'], merge),
            'split_death_on_gt_error_rate': self._ratio(state['split_wrong_death_gt_error'], split),
            'counterfactual_fix_rate': self._ratio(state['counterfactual_fix'], events),
            'merge_counterfactual_fix_rate': self._ratio(state['merge_counterfactual_fix'], merge),
            'split_counterfactual_fix_rate': self._ratio(state['split_counterfactual_fix'], split),
            'mean_persistence_correct_structure': self._mean(state['persistence_correct']),
            'mean_persistence_wrong_structure': self._mean(state['persistence_wrong']),
            'correct_structure_higher_persistence_rate': self._ratio(state['higher_persistence'], persistence_n),
            'mean_num_h0_pairs_a': self._mean(state['num_h0_pairs_a']),
            'mean_num_h0_pairs_b': self._mean(state['num_h0_pairs_b']),
            'mean_persistence_a': self._mean(state['persistence_a_all']),
            'mean_persistence_b': self._mean(state['persistence_b_all']),
            'unmatched_ph_event_count': events - state['matched'],
        }

    def _ratio(self, numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    def _mean(self, values):
        return float(np.mean(values)) if values else 0.0

    def _panel_image(self, tensor):
        array = tensor.float().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        array = array.astype(np.float32)
        min_value = float(array.min())
        max_value = float(array.max())
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        return np.clip(array, 0, 1)

    def _save_example(
        self,
        row,
        event,
        image_a,
        image_b,
        gt,
        pred_a,
        pred_b,
        disagreement,
        roi,
        wrong_error,
        boundary_gt,
        pair_a,
        pair_b,
    ):
        event_type = event['event_type']
        if self.example_counts[event_type] >= 10:
            return
        out_dir = self.output_dir / 'ph_struct_examples' / event_type
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        panels = [
            (self._panel_image(image_a), 'RGB image'),
            (self._panel_image(image_b), 'Stain image'),
            (gt, 'GT'),
            (pred_a, 'Pred A'),
            (pred_b, 'Pred B'),
            (disagreement, 'A/B disagreement'),
            (roi, 'event ROI'),
            (wrong_error, 'GT error mask'),
            (boundary_gt, 'GT boundary'),
        ]
        for ax, (panel, title) in zip(axes.flat, panels):
            if isinstance(panel, np.ndarray) and panel.ndim == 2:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        self._draw_pair(axes.flat[3], pair_a)
        self._draw_pair(axes.flat[4], pair_b)
        fig.suptitle(
            '{} | correct={} | persistence_A={} | persistence_B={} | cf={}'.format(
                row['event_type'],
                row['oracle_correct_branch'],
                row['persistence_a'],
                row['persistence_b'],
                row['counterfactual_fix_success'],
            ),
            fontsize=10,
        )
        safe_id = Path(str(row['image_id'])).stem.replace(os.sep, '_')
        path = out_dir / 'epoch_{:03d}_event_{:06d}_{}.png'.format(
            int(row['epoch']), int(row['event_id']), safe_id
        )
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        self.example_counts[event_type] += 1

    def _draw_pair(self, ax, pair):
        if pair is None:
            return
        by, bx = pair['birth_coord']
        dy, dx = pair['death_coord']
        ax.scatter([bx], [by], c='lime', marker='o', s=45, linewidths=1, edgecolors='black')
        ax.scatter([dx], [dy], c='red', marker='x', s=80, linewidths=2.5)
        ax.scatter([dx], [dy], facecolors='none', edgecolors='yellow', marker='o', s=130, linewidths=2)


def _load_models(args, cfg, device, checkpoint_path, epoch):
    model_a = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).to(device)
    model_b = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES']).to(device)
    path_a, path_b, resolved_epoch = _resolve_struct_checkpoint_pair(
        checkpoint_path,
        None,
        None,
        epoch,
    )
    checkpoint = torch.load(path_a, map_location='cpu', weights_only=False)
    if path_b is None:
        if not (
            isinstance(checkpoint, dict)
            and 'appearance_state_dict' in checkpoint
            and 'structure_state_dict' in checkpoint
        ):
            raise ValueError(
                'Single checkpoint must contain appearance_state_dict and structure_state_dict: {}'.format(path_a)
            )
        _load_state_dict(model_a, checkpoint['appearance_state_dict'], path_a)
        _load_state_dict(model_b, checkpoint['structure_state_dict'], path_a)
        label = path_a
    else:
        checkpoint_b = torch.load(path_b, map_location='cpu', weights_only=False)
        _load_state_dict(model_a, checkpoint, path_a)
        _load_state_dict(model_b, checkpoint_b, path_b)
        label = '{};{}'.format(path_a, path_b)
    return (model_a, model_b), label, int(resolved_epoch or epoch or 0)


def _checkpoint_jobs(args):
    if args.checkpoint:
        return [(args.checkpoint, int(args.epoch or 0))]
    if not args.checkpoint_dir:
        raise ValueError('Pass --checkpoint or --checkpoint-dir.')
    if args.epochs:
        return [(args.checkpoint_dir, int(epoch)) for epoch in args.epochs]
    return [(args.checkpoint_dir, 0)]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline PH-H0 structural conflict diagnostics for RGB/Stain CPS checkpoints.'
    )
    parser.add_argument('--checkpoint', default=None, help='single milestone checkpoint or checkpoint directory')
    parser.add_argument('--checkpoint-dir', default=None, help='directory containing milestones/periodic_best checkpoints')
    parser.add_argument('--epochs', nargs='*', type=int, default=None, help='epochs to resolve from --checkpoint-dir')
    parser.add_argument('--epoch', type=int, default=0, help='epoch to resolve for --checkpoint when it is a directory')
    parser.add_argument('--dataset_name', default='GlaS', choices=['GlaS', 'CRAG', 'MoNuSeg'])
    parser.add_argument('--path_dataset', default='datasets/GLaS')
    parser.add_argument('--input_mode', default='rgb_stain', choices=['rgb_rgb', 'rgb_stain', 'rgb_hed', 'ucmt_views'])
    parser.add_argument('--input1', default='image')
    parser.add_argument('--manifest_dir', default=None)
    parser.add_argument('--labeled_ratio', type=int, default=10, choices=[10, 20])
    parser.add_argument('--sup_mark', default='9')
    parser.add_argument('--unsup_mark', default='76')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--network', default='unet')
    parser.add_argument('--struct-min-area', type=int, default=20)
    parser.add_argument('--struct-match-iou', type=float, default=0.1)
    parser.add_argument('--ph-roi-dilation', type=int, default=5)
    parser.add_argument('--ph-boundary-width', type=int, default=3)
    parser.add_argument('--ph-critical-radius', type=int, default=2)
    parser.add_argument('--save-ph-struct-examples', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    init_seeds(args.seed)
    cfg = dataset_cfg(args.dataset_name)
    dataset_args = SimpleNamespace(
        manifest_dir=args.manifest_dir,
        path_dataset=args.path_dataset,
        input1=args.input1,
        input_mode=args.input_mode,
        labeled_ratio=args.labeled_ratio,
        sup_mark=args.sup_mark,
        unsup_mark=args.unsup_mark,
    )
    datasets, _manifest_dir, _use_manifests = build_datasets(dataset_args, cfg)
    dataloader = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
    )
    root = args.output_dir
    if root is None:
        source = args.checkpoint or args.checkpoint_dir
        root = os.path.join(os.path.dirname(os.path.abspath(source)), 'ph_h0_structural')
    analyzer = PHH0StructuralAnalyzer(
        root,
        min_area=args.struct_min_area,
        match_iou=args.struct_match_iou,
        roi_dilation=args.ph_roi_dilation,
        boundary_width=args.ph_boundary_width,
        critical_radius=args.ph_critical_radius,
        save_examples=args.save_ph_struct_examples,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for checkpoint_path, epoch in _checkpoint_jobs(args):
        models, label, resolved_epoch = _load_models(args, cfg, device, checkpoint_path, epoch)
        summary = analyzer.analyze_checkpoint(
            models, dataloader, device, resolved_epoch, label
        )
        print(
            'epoch={} events={} ph_match_rate={:.6f} death_on_gt_error_rate={:.6f} counterfactual_fix_rate={:.6f}'.format(
                summary['epoch'],
                summary['total_structural_events'],
                summary['ph_match_rate'],
                summary['death_on_gt_error_rate'],
                summary['counterfactual_fix_rate'],
            )
        )


if __name__ == '__main__':
    main()
