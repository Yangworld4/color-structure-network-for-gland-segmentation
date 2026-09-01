from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import label

from utils.ph_correspondence import build_persistent_relation_conflict_events
from utils.ph_directional_cps import _event_routing_weights, _predicted_direction
from utils.ph_relation_tree import ABSENT, CONNECTED, SEPARATED


PH_DIAGNOSTIC_FIELDS = [
    'epoch',
    'branches_a',
    'branches_b',
    'matched_branches',
    'match_rate',
    'mean_birth_match_distance',
    'relation_questions',
    'pair_present_both',
    'pair_presence_involved',
    'pure_relation_conflicts',
    'presence_involved_conflicts',
    'total_structural_conflicts',
    'pure_relation_conflict_rate',
    'presence_involved_conflict_rate',
    'a_connected_b_separated',
    'a_separated_b_connected',
    'a_connected_b_absent',
    'b_connected_a_absent',
    'gt_relation_valid_questions',
    'relation_error_rate_a',
    'relation_error_rate_b',
    'relation_both_wrong_rate',
    'relation_one_correct_rate',
    'false_merge_rate_a',
    'false_split_rate_a',
    'false_merge_rate_b',
    'false_split_rate_b',
    'pure_relation_roi_pixel_ratio',
    'pure_relation_roi_gt_error_rate',
    'global_disagreement_gt_error_rate',
    'pure_relation_error_enrichment',
    'presence_roi_gt_error_rate',
    'ph_cc_relation_agreement',
    'routing_oracle_accuracy',
]


def build_ph_diagnostics_row(
    epoch,
    image_ids,
    logits_a,
    logits_b,
    gt,
    dilation=1,
    unmatched_mode='vanilla',
    correspondence_mode='birth_nearest',
    relation_backend='ph_tree',
    presence_conflict_mode='strict',
    output_dir=None,
    save_conflict_vis=False,
    vis_interval=10,
    vis_num_samples=4,
    vis_state=None,
    images=None,
):
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    pred_a = prob_a.cpu().numpy() >= 0.5
    pred_b = prob_b.cpu().numpy() >= 0.5
    gt_np = gt.detach().cpu().numpy() > 0

    _mask, events, stats = build_persistent_relation_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        unmatched_mode=unmatched_mode,
        correspondence_mode=correspondence_mode,
        relation_backend=relation_backend,
        collect_cc_agreement=(relation_backend == 'ph_tree'),
        presence_conflict_mode=presence_conflict_mode,
    )
    row, sample_summaries = _aggregate_row(
        int(epoch), stats, events, image_ids, pred_a, pred_b, gt_np, prob_a, prob_b
    )
    if (
        save_conflict_vis
        and output_dir is not None
        and images is not None
        and vis_interval > 0
        and int(epoch) % int(vis_interval) == 0
    ):
        save_conflict_visualizations(
            output_dir,
            int(epoch),
            image_ids,
            images,
            pred_a,
            pred_b,
            gt_np,
            events,
            sample_summaries,
            vis_state if vis_state is not None else {},
            vis_num_samples,
        )
    return row


def _aggregate_row(epoch, stats, events, image_ids, pred_a, pred_b, gt_np, prob_a, prob_b):
    questions = list(stats.get('relation_question_records', []))
    total_questions = int(stats.get('num_relation_questions', len(questions)))
    pair_present_both = 0
    pair_presence_involved = 0
    pure_conflicts = 0
    presence_conflicts = 0
    a_connected_b_separated = 0
    a_separated_b_connected = 0
    a_connected_b_absent = 0
    b_connected_a_absent = 0
    gt_valid = 0
    error_a = error_b = both_wrong = one_correct = 0
    false_merge_a = false_split_a = false_merge_b = false_split_b = 0
    ph_cc_agree = int(stats.get('ph_cc_agree_A', 0)) + int(stats.get('ph_cc_agree_B', 0))
    ph_cc_total = 2 * int(stats.get('valid_compare_count', 0))
    sample_summaries = {}
    labels_by_sample = {}

    for sample in range(gt_np.shape[0]):
        labels_by_sample[sample], _ = label(
            gt_np[sample], structure=np.ones((3, 3), dtype=np.uint8)
        )
        sample_summaries[sample] = {
            'pure_relation_conflicts': 0,
            'presence_involved_conflicts': 0,
            'gt_valid': 0,
            'error_a': 0,
            'error_b': 0,
        }

    for question in questions:
        sample = int(question.get('sample_index', 0))
        state_a = _query_state(question, 'a')
        state_b = _query_state(question, 'b')
        present_both = bool(question.get('pair_present_a')) and bool(question.get('pair_present_b'))
        if present_both:
            pair_present_both += 1
        else:
            pair_presence_involved += 1
        pure_conflict = (
            present_both
            and state_a in (CONNECTED, SEPARATED)
            and state_b in (CONNECTED, SEPARATED)
            and state_a != state_b
        )
        presence_conflict = (
            state_a != state_b
            and (state_a == ABSENT or state_b == ABSENT)
            and ((state_a == CONNECTED) != (state_b == CONNECTED))
        )
        if pure_conflict:
            pure_conflicts += 1
            a_connected_b_separated += int(state_a == CONNECTED and state_b == SEPARATED)
            a_separated_b_connected += int(state_a == SEPARATED and state_b == CONNECTED)
            sample_summaries[sample]['pure_relation_conflicts'] += 1
        if presence_conflict:
            presence_conflicts += 1
            a_connected_b_absent += int(state_a == CONNECTED and state_b == ABSENT)
            b_connected_a_absent += int(state_b == CONNECTED and state_a == ABSENT)
            sample_summaries[sample]['presence_involved_conflicts'] += 1
        if not present_both or state_a not in (CONNECTED, SEPARATED) or state_b not in (CONNECTED, SEPARATED):
            continue
        gt_relation = _gt_relation_for_question(question, labels_by_sample[sample])
        if gt_relation is None:
            continue
        gt_valid += 1
        pred_rel_a = state_a == CONNECTED
        pred_rel_b = state_b == CONNECTED
        err_a = pred_rel_a != gt_relation
        err_b = pred_rel_b != gt_relation
        error_a += int(err_a)
        error_b += int(err_b)
        both_wrong += int(err_a and err_b)
        one_correct += int(err_a != err_b)
        false_merge_a += int((not gt_relation) and pred_rel_a)
        false_split_a += int(gt_relation and (not pred_rel_a))
        false_merge_b += int((not gt_relation) and pred_rel_b)
        false_split_b += int(gt_relation and (not pred_rel_b))
        sample_summaries[sample]['gt_valid'] += 1
        sample_summaries[sample]['error_a'] += int(err_a)
        sample_summaries[sample]['error_b'] += int(err_b)

    pure_roi = _event_union_mask(events, pred_a.shape, 'pure_relation')
    presence_roi = _event_union_mask(events, pred_a.shape, 'presence_involved')
    any_gt_error = np.logical_or(pred_a != gt_np, pred_b != gt_np)
    disagreement = pred_a != pred_b
    pure_roi_pixels = int(pure_roi.sum())
    presence_roi_pixels = int(presence_roi.sum())
    global_disagree_pixels = int(disagreement.sum())
    pure_roi_error_rate = _ratio(int(np.logical_and(pure_roi, any_gt_error).sum()), pure_roi_pixels)
    global_disagreement_error_rate = _ratio(
        int(np.logical_and(disagreement, any_gt_error).sum()), global_disagree_pixels
    )
    presence_roi_error_rate = _ratio(
        int(np.logical_and(presence_roi, any_gt_error).sum()), presence_roi_pixels
    )
    routing_accuracy = _routing_oracle_accuracy(events, labels_by_sample, prob_a, prob_b)
    branches_a = int(stats.get('raw_ph_branches_a', 0))
    branches_b = int(stats.get('raw_ph_branches_b', 0))
    matched = int(stats.get('matched_branches', 0))
    return {
        'epoch': epoch,
        'branches_a': branches_a,
        'branches_b': branches_b,
        'matched_branches': matched,
        'match_rate': _ratio(matched, max(min(branches_a, branches_b), 1)),
        'mean_birth_match_distance': _ratio(
            float(stats.get('birth_distance_sum', 0.0)), matched
        ),
        'relation_questions': total_questions,
        'pair_present_both': pair_present_both,
        'pair_presence_involved': pair_presence_involved,
        'pure_relation_conflicts': pure_conflicts,
        'presence_involved_conflicts': presence_conflicts,
        'total_structural_conflicts': pure_conflicts + presence_conflicts,
        'pure_relation_conflict_rate': _ratio(pure_conflicts, max(pair_present_both, 1)),
        'presence_involved_conflict_rate': _ratio(presence_conflicts, max(total_questions, 1)),
        'a_connected_b_separated': a_connected_b_separated,
        'a_separated_b_connected': a_separated_b_connected,
        'a_connected_b_absent': a_connected_b_absent,
        'b_connected_a_absent': b_connected_a_absent,
        'gt_relation_valid_questions': gt_valid,
        'relation_error_rate_a': _ratio(error_a, gt_valid),
        'relation_error_rate_b': _ratio(error_b, gt_valid),
        'relation_both_wrong_rate': _ratio(both_wrong, gt_valid),
        'relation_one_correct_rate': _ratio(one_correct, gt_valid),
        'false_merge_rate_a': _ratio(false_merge_a, gt_valid),
        'false_split_rate_a': _ratio(false_split_a, gt_valid),
        'false_merge_rate_b': _ratio(false_merge_b, gt_valid),
        'false_split_rate_b': _ratio(false_split_b, gt_valid),
        'pure_relation_roi_pixel_ratio': _ratio(pure_roi_pixels, pred_a.size),
        'pure_relation_roi_gt_error_rate': pure_roi_error_rate,
        'global_disagreement_gt_error_rate': global_disagreement_error_rate,
        'pure_relation_error_enrichment': _ratio(
            pure_roi_error_rate, max(global_disagreement_error_rate, 1e-8)
        ),
        'presence_roi_gt_error_rate': presence_roi_error_rate,
        'ph_cc_relation_agreement': _ratio(ph_cc_agree, ph_cc_total),
        'routing_oracle_accuracy': routing_accuracy,
    }, sample_summaries


def _query_state(question, branch):
    query = question.get('query_' + branch)
    return query.state if query is not None else ABSENT


def _gt_relation_for_question(question, gt_labels):
    left = question['left_match']
    right = question['right_match']
    coords = (
        left['entity_a'].birth_coord,
        right['entity_a'].birth_coord,
        left['entity_b'].birth_coord,
        right['entity_b'].birth_coord,
    )
    labels = []
    for y, x in coords:
        if y < 0 or x < 0 or y >= gt_labels.shape[0] or x >= gt_labels.shape[1]:
            return None
        value = int(gt_labels[y, x])
        if value <= 0:
            return None
        labels.append(value)
    relation_a = labels[0] == labels[1]
    relation_b = labels[2] == labels[3]
    if relation_a != relation_b:
        return None
    return relation_a


def _event_union_mask(events, shape, structural_type):
    mask = np.zeros(shape, dtype=bool)
    for event in events:
        if event.get('structural_conflict_type') != structural_type:
            continue
        sample = int(event['sample_index'])
        y_slice, x_slice = event['roi_slice']
        mask[sample, y_slice, x_slice] |= np.asarray(event['roi'], dtype=bool)
    return mask


def _routing_oracle_accuracy(events, labels_by_sample, prob_a, prob_b):
    pure_events = [
        event for event in events
        if event.get('structural_conflict_type') == 'pure_relation'
    ]
    if not pure_events:
        return 0.0
    weights = _event_routing_weights(prob_a, prob_b, pure_events)
    correct = 0
    total = 0
    for event in pure_events:
        sample = int(event['sample_index'])
        gt_relation = _gt_relation_for_question(event, labels_by_sample[sample])
        if gt_relation is None:
            continue
        a_error = bool(event['connected_a']) != gt_relation
        b_error = bool(event['connected_b']) != gt_relation
        if a_error == b_error:
            continue
        oracle = 'b2a' if a_error else 'a2b'
        predicted, _weight = _predicted_direction(weights[id(event)])
        total += 1
        correct += int(predicted == oracle)
    return _ratio(correct, total)


def save_conflict_visualizations(
    output_dir,
    epoch,
    image_ids,
    images,
    pred_a,
    pred_b,
    gt_np,
    events,
    sample_summaries,
    vis_state,
    vis_num_samples,
):
    pure_counts = {}
    for event in events:
        if event.get('structural_conflict_type') == 'pure_relation':
            sample = int(event['sample_index'])
            pure_counts[sample] = pure_counts.get(sample, 0) + 1
    if vis_state.get('sample_ids') is None:
        ranked = sorted(
            range(len(image_ids)),
            key=lambda index: (-pure_counts.get(index, 0), str(image_ids[index])),
        )
        vis_state['sample_ids'] = [int(index) for index in ranked[:max(0, int(vis_num_samples))]]
    output_path = Path(output_dir) / 'conflict_vis' / 'epoch_{:03d}'.format(epoch)
    output_path.mkdir(parents=True, exist_ok=True)
    image_np = images.detach().cpu()
    for sample in vis_state.get('sample_ids', []):
        sample_events = [
            event for event in events
            if int(event['sample_index']) == int(sample)
            and event.get('structural_conflict_type') == 'pure_relation'
        ]
        sample_events.sort(key=lambda event: int(event.get('roi_size', 0)), reverse=True)
        presence_events = [
            event for event in events
            if int(event['sample_index']) == int(sample)
            and event.get('structural_conflict_type') == 'presence_involved'
        ]
        canvas = _make_conflict_canvas(
            image_np[sample],
            gt_np[sample],
            pred_a[sample],
            pred_b[sample],
            sample_events[:3],
            presence_events,
            sample_summaries.get(sample, {}),
            epoch,
            image_ids[sample],
        )
        canvas.save(output_path / '{}.png'.format(_safe_filename(image_ids[sample])))


def _make_conflict_canvas(image, gt_mask, pred_a, pred_b, pure_events, presence_events, summary, epoch, sample_id):
    panels = [
        _tensor_to_image(image),
        _mask_to_image(gt_mask),
        _mask_to_image(pred_a),
        _mask_to_image(pred_b),
        _overlay_panel(image, gt_mask, pure_events, presence_events),
    ]
    width, height = panels[0].size
    title_height = 58
    label_height = 18
    canvas = Image.new('RGB', (width * 5, height + title_height + label_height), 'white')
    draw = ImageDraw.Draw(canvas)
    title = (
        'epoch={} sample={} pure_relation_conflicts={} presence_involved_conflicts={} '
        'relation_error_A={}/{} relation_error_B={}/{}'
    ).format(
        epoch,
        sample_id,
        int(summary.get('pure_relation_conflicts', 0)),
        int(summary.get('presence_involved_conflicts', 0)),
        int(summary.get('error_a', 0)),
        int(summary.get('gt_valid', 0)),
        int(summary.get('error_b', 0)),
        int(summary.get('gt_valid', 0)),
    )
    draw.text((6, 6), title[:220], fill=(0, 0, 0))
    labels = ('Original RGB', 'Ground Truth', 'Student A', 'Student B', 'Conflict overlay')
    for index, panel in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, title_height))
        draw.text((x + 4, title_height + height + 2), labels[index], fill=(0, 0, 0))
    return canvas


def _overlay_panel(image, gt_mask, pure_events, presence_events):
    panel = _tensor_to_image(image).convert('RGBA')
    draw = ImageDraw.Draw(panel)
    gt_labels, _ = label(gt_mask, structure=np.ones((3, 3), dtype=np.uint8))
    for event in presence_events[:8]:
        _draw_roi_outline(draw, event, (255, 180, 0, 255), width=1)
    for event in pure_events:
        _draw_roi_outline(draw, event, (255, 0, 0, 255), width=2)
        _draw_endpoints(draw, event)
        saddle = event.get('saddle_coord_a') if event.get('connected_a') else event.get('saddle_coord_b')
        if saddle is not None:
            _draw_cross(draw, saddle, (255, 0, 255, 255))
        text = _event_label(event, gt_labels)
        y_slice, x_slice = event['roi_slice']
        draw.text((int(x_slice.start), max(0, int(y_slice.start) - 24)), text, fill=(255, 0, 0, 255))
    return panel.convert('RGB')


def _event_label(event, gt_labels):
    state_a = 'C' if event.get('connected_a') else 'S'
    state_b = 'C' if event.get('connected_b') else 'S'
    query_a = event.get('relation_query_a')
    query_b = event.get('relation_query_b')
    if query_a is not None and query_a.state == ABSENT:
        state_a = 'A'
    if query_b is not None and query_b.state == ABSENT:
        state_b = 'A'
    gt_relation = _gt_relation_for_question(event, gt_labels) if gt_labels is not None else None
    gt_text = 'I' if gt_relation is None else ('C' if gt_relation else 'S')
    return 'A:{}\nB:{}\nGT:{}'.format(state_a, state_b, gt_text)


def _draw_roi_outline(draw, event, color, width=2):
    y_slice, x_slice = event['roi_slice']
    ys, xs = np.nonzero(event['roi'])
    if ys.size == 0:
        return
    box = (
        int(x_slice.start + xs.min()),
        int(y_slice.start + ys.min()),
        int(x_slice.start + xs.max()),
        int(y_slice.start + ys.max()),
    )
    draw.rectangle(box, outline=color, width=width)


def _draw_endpoints(draw, event):
    left = event['left_match']
    right = event['right_match']
    for coord, color in (
        (left['entity_a'].birth_coord, (0, 120, 255, 255)),
        (right['entity_a'].birth_coord, (0, 120, 255, 255)),
        (left['entity_b'].birth_coord, (0, 200, 120, 255)),
        (right['entity_b'].birth_coord, (0, 200, 120, 255)),
    ):
        _draw_point(draw, coord, color)


def _draw_point(draw, coord, color):
    y, x = coord
    radius = 3
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def _draw_cross(draw, coord, color):
    y, x = coord
    radius = 5
    draw.line((x - radius, y, x + radius, y), fill=color, width=2)
    draw.line((x, y - radius, x, y + radius), fill=color, width=2)


def _tensor_to_image(tensor):
    array = tensor.detach().float().cpu().numpy()
    if array.ndim == 3:
        array = np.transpose(array[:3], (1, 2, 0))
    array = array - array.min()
    max_value = array.max()
    if max_value > 0:
        array = array / max_value
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array).convert('RGB')


def _mask_to_image(mask):
    array = np.asarray(mask).astype(np.uint8) * 255
    return Image.fromarray(array).convert('RGB')


def _safe_filename(value):
    text = str(value)
    return ''.join(char if char.isalnum() or char in ('-', '_', '.') else '_' for char in text)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0
