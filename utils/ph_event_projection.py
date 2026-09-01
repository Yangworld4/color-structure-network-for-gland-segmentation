from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label

from utils.ph_correspondence import decompose_h0_branches


EVENT_PROJECTION_DIAGNOSTIC_FIELDS = [
    'epoch',
    'raw_ph_branches_A',
    'raw_ph_branches_B',
    'merge_events_A',
    'merge_events_B',
    'valid_source_events_A',
    'valid_source_events_B',
    'unmatched_target_events_A2B',
    'unmatched_target_events_B2A',
    'queried_events_A2B',
    'queried_events_B2A',
    'same_relation_events',
    'structural_conflict_events',
    'conflict_rate',
    'A_separate_B_connected',
    'A_connected_B_separate',
    'conflict_roi_count',
    'conflict_pixel_count',
    'conflict_pixel_ratio',
    'mean_roi_size',
    'median_roi_size',
    'a2b_routed_events',
    'b2a_routed_events',
    'mean_weight_A2B',
    'mean_weight_B2A',
    'ph_extract_time',
    'cc_time',
    'event_query_time',
    'roi_time',
    'routing_time',
    'total_ph_directional_time',
]


EVENT_PROJECTION_ORACLE_FIELDS = [
    'epoch',
    'oracle_valid_conflicts',
    'A_correct_B_wrong',
    'A_wrong_B_correct',
    'event_conflict_accuracy_A',
    'event_conflict_accuracy_B',
    'routing_correct',
    'routing_wrong',
    'routing_accuracy',
    'false_merge_conflicts',
    'false_split_conflicts',
]


@dataclass
class PersistentMergeEvent:
    event_id: int
    source_branch_id: int
    parent_branch_id: int
    merge_level: float
    saddle_coord: tuple
    child_support_id: int
    parent_support_id: int


def build_event_projection_conflict_events(
    prob_a,
    prob_b,
    pred_a,
    pred_b,
    dilation=1,
    debug_timing=False,
    batch_index=None,
):
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    batch, height, width = prob_a.shape
    masks = np.zeros((batch, height, width), dtype=bool)
    events = []
    stats = _new_event_projection_stats()

    for index in range(batch):
        sample_mask, sample_events, sample_stats = detect_projected_conflicts(
            prob_a[index],
            prob_b[index],
            pred_a[index],
            pred_b[index],
            dilation=dilation,
        )
        masks[index] = sample_mask
        _merge_stats(stats, sample_stats)
        for event in sample_events:
            event['sample_index'] = index
            events.append(event)

    stats['events'] = len(events)
    stats['ph_conflict_pixels'] = int(masks.sum())
    stats['conflict_pixel_ratio'] = _ratio(stats['ph_conflict_pixels'], masks.size)
    if debug_timing:
        _print_timing(batch_index, stats)
    return masks, events, stats


def extract_merge_events(probability, branch_prefix='A'):
    decomposition = decompose_h0_branches(probability, branch_prefix)
    events = []
    for event_id, edge in enumerate(decomposition.edges):
        events.append(PersistentMergeEvent(
            event_id=event_id,
            source_branch_id=int(edge.child_branch_id),
            parent_branch_id=int(edge.parent_branch_id),
            merge_level=float(edge.merge_level),
            saddle_coord=tuple(edge.merge_coord),
            child_support_id=int(edge.child_branch_id),
            parent_support_id=int(edge.parent_branch_id),
        ))
    return decomposition, events


def detect_projected_conflicts(prob_a, prob_b, pred_a, pred_b, dilation=1):
    timings = {key: 0.0 for key in _timing_keys()}
    total_start = time.perf_counter()

    start = time.perf_counter()
    decomposition_a, merge_events_a = extract_merge_events(prob_a, 'A')
    decomposition_b, merge_events_b = extract_merge_events(prob_b, 'B')
    timings['ph_extract_time'] = time.perf_counter() - start

    start = time.perf_counter()
    fg_a = np.asarray(pred_a) == 1
    fg_b = np.asarray(pred_b) == 1
    cc_a, _ = label(fg_a, structure=np.ones((3, 3), dtype=np.uint8))
    cc_b, _ = label(fg_b, structure=np.ones((3, 3), dtype=np.uint8))
    disagreement = np.asarray(pred_a) != np.asarray(pred_b)
    disagreement_labels, _ = label(
        disagreement,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    disagreement_slices = find_objects(disagreement_labels)
    disagreement_nonzero = np.nonzero(disagreement_labels)
    timings['cc_time'] = time.perf_counter() - start

    stats = _new_event_projection_stats()
    stats['raw_ph_branches_A'] = len(decomposition_a.branches)
    stats['raw_ph_branches_B'] = len(decomposition_b.branches)
    stats['raw_ph_branches_a'] = len(decomposition_a.branches)
    stats['raw_ph_branches_b'] = len(decomposition_b.branches)
    stats['merge_events_A'] = len(merge_events_a)
    stats['merge_events_B'] = len(merge_events_b)
    stats['direct_merge_edges_a'] = len(merge_events_a)
    stats['direct_merge_edges_b'] = len(merge_events_b)

    start = time.perf_counter()
    conflicts = []
    for event in merge_events_a:
        query = query_event_in_target(
            event,
            decomposition_a.branch_label_map,
            prob_a,
            fg_a,
            cc_a,
            prob_b,
            fg_b,
            cc_b,
        )
        _record_query_stats(stats, query, 'A')
        if query.get('conflict'):
            conflicts.append(_conflict_from_query('A', event, query))
    for event in merge_events_b:
        query = query_event_in_target(
            event,
            decomposition_b.branch_label_map,
            prob_b,
            fg_b,
            cc_b,
            prob_a,
            fg_a,
            cc_a,
        )
        _record_query_stats(stats, query, 'B')
        if query.get('conflict'):
            conflicts.append(_conflict_from_query('B', event, query))
    timings['event_query_time'] = time.perf_counter() - start

    start = time.perf_counter()
    masks, events, roi_sizes = build_event_conflict_roi(
        conflicts,
        disagreement_labels,
        disagreement_slices,
        disagreement_nonzero,
        dilation=dilation,
    )
    timings['roi_time'] = time.perf_counter() - start
    timings['total_ph_directional_time'] = time.perf_counter() - total_start
    stats.update(timings)
    stats['structural_conflict_events'] = len(conflicts)
    stats['connectivity_conflicts'] = len(conflicts)
    stats['persistent_connectivity_conflicts'] = len(conflicts)
    stats['directional_events_from_persistent_conflicts'] = len(events)
    stats['events'] = len(events)
    stats['ph_conflict_pixels'] = int(masks.sum())
    stats['conflict_pixel_count'] = int(masks.sum())
    stats['conflict_pixel_ratio'] = _ratio(stats['ph_conflict_pixels'], masks.size)
    stats['conflict_roi_count'] = len(roi_sizes)
    stats['roi_sizes'] = roi_sizes
    stats['mean_roi_size'] = _mean(roi_sizes)
    stats['median_roi_size'] = _median(roi_sizes)
    stats['mean_conflict_roi_pixels_sum'] = float(sum(roi_sizes))
    stats['mean_conflict_roi_pixels_count'] = len(roi_sizes)
    queried = stats['queried_events_A2B'] + stats['queried_events_B2A']
    stats['conflict_rate'] = _ratio(stats['structural_conflict_events'], queried)
    stats['support_build_time'] = stats['ph_extract_time']
    stats['support_overlap_time'] = 0.0
    stats['entity_match_time'] = 0.0
    stats['relation_time'] = stats['event_query_time']
    stats['total_ph_time'] = stats['total_ph_directional_time']
    return masks, events, stats


def query_event_in_target(
    event,
    source_label_map,
    source_prob,
    source_fg,
    source_cc,
    target_prob,
    target_fg,
    target_cc,
):
    child_support = source_label_map == event.child_support_id
    parent_support = source_label_map == event.parent_support_id

    q_child_source = _best_foreground_anchor(child_support, source_fg, source_prob)
    q_parent_source = _best_foreground_anchor(parent_support, source_fg, source_prob)
    if q_child_source is None or q_parent_source is None:
        return {'status': 'invalid_source_event'}

    source_connected = _connected(source_cc, q_child_source, q_parent_source)
    q_child_target = _best_foreground_anchor(child_support, target_fg, target_prob)
    q_parent_target = _best_foreground_anchor(parent_support, target_fg, target_prob)
    if q_child_target is None or q_parent_target is None:
        return {
            'status': 'unmatched_target_event',
            'q_child_source': q_child_source,
            'q_parent_source': q_parent_source,
            'source_connected': source_connected,
        }

    target_connected = _connected(target_cc, q_child_target, q_parent_target)
    return {
        'status': 'queried',
        'q_child_source': q_child_source,
        'q_parent_source': q_parent_source,
        'q_child_target': q_child_target,
        'q_parent_target': q_parent_target,
        'source_connected': source_connected,
        'target_connected': target_connected,
        'conflict': source_connected != target_connected,
    }


def build_event_conflict_roi(
    conflicts,
    disagreement_labels,
    disagreement_slices,
    disagreement_nonzero,
    dilation=1,
):
    masks = np.zeros_like(disagreement_labels, dtype=bool)
    events = []
    roi_sizes = []
    if not conflicts or disagreement_nonzero[0].size == 0:
        return masks, events, roi_sizes
    for conflict in conflicts:
        roi_slice, roi = _conflict_roi_from_labels(
            disagreement_labels,
            disagreement_slices,
            disagreement_nonzero,
            conflict['saddle_coord'],
            dilation,
        )
        if roi is None:
            continue
        masks[roi_slice] |= roi
        roi_size = int(roi.sum())
        roi_sizes.append(roi_size)
        event = dict(conflict)
        event.update({
            'event_type': 'connectivity_conflict',
            'conflict_type': conflict['conflict_type'],
            'roi_slice': roi_slice,
            'roi': roi,
            'roi_size': roi_size,
            'matched_pair': False,
            'event_projection': True,
        })
        events.append(event)
    return masks, events, roi_sizes


def event_projection_diagnostics_row(state):
    event_count = int(state.get('event_weight_count', 0))
    total_pixels = int(state.get('total_unlabeled_pixels', 0))
    queried = int(state.get('queried_events_A2B', 0)) + int(
        state.get('queried_events_B2A', 0)
    )
    row = {
        'epoch': state['epoch'],
        'raw_ph_branches_A': state.get('raw_ph_branches_A', 0),
        'raw_ph_branches_B': state.get('raw_ph_branches_B', 0),
        'merge_events_A': state.get('merge_events_A', 0),
        'merge_events_B': state.get('merge_events_B', 0),
        'valid_source_events_A': state.get('valid_source_events_A', 0),
        'valid_source_events_B': state.get('valid_source_events_B', 0),
        'unmatched_target_events_A2B': state.get('unmatched_target_events_A2B', 0),
        'unmatched_target_events_B2A': state.get('unmatched_target_events_B2A', 0),
        'queried_events_A2B': state.get('queried_events_A2B', 0),
        'queried_events_B2A': state.get('queried_events_B2A', 0),
        'same_relation_events': state.get('same_relation_events', 0),
        'structural_conflict_events': state.get('structural_conflict_events', 0),
        'conflict_rate': _ratio(state.get('structural_conflict_events', 0), queried),
        'A_separate_B_connected': state.get('A_separate_B_connected', 0),
        'A_connected_B_separate': state.get('A_connected_B_separate', 0),
        'conflict_roi_count': state.get('conflict_roi_count', 0),
        'conflict_pixel_count': state.get('ph_conflict_pixels', 0),
        'conflict_pixel_ratio': _ratio(state.get('ph_conflict_pixels', 0), total_pixels),
        'mean_roi_size': _ratio(
            state.get('event_projection_roi_size_sum', 0.0),
            state.get('conflict_roi_count', 0),
        ),
        'median_roi_size': (
            float(np.median(state.get('event_projection_roi_sizes', [])))
            if state.get('event_projection_roi_sizes', []) else 0.0
        ),
        'a2b_routed_events': state.get('a2b_events', 0),
        'b2a_routed_events': state.get('b2a_events', 0),
        'mean_weight_A2B': _ratio(state.get('weight_a2b_sum', 0.0), event_count),
        'mean_weight_B2A': _ratio(state.get('weight_b2a_sum', 0.0), event_count),
        'ph_extract_time': _ratio(
            state.get('event_projection_ph_extract_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
        'cc_time': _ratio(
            state.get('event_projection_cc_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
        'event_query_time': _ratio(
            state.get('event_projection_event_query_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
        'roi_time': _ratio(
            state.get('event_projection_roi_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
        'routing_time': _ratio(
            state.get('event_projection_routing_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
        'total_ph_directional_time': _ratio(
            state.get('event_projection_total_ph_directional_time_sum', 0.0),
            state.get('ph_timing_batches', 0),
        ),
    }
    return row


def event_projection_oracle_row(state):
    valid = int(state.get('event_projection_oracle_valid_conflicts', 0))
    routing_total = (
        int(state.get('event_projection_routing_correct', 0))
        + int(state.get('event_projection_routing_wrong', 0))
    )
    return {
        'epoch': state['epoch'],
        'oracle_valid_conflicts': valid,
        'A_correct_B_wrong': state.get('event_projection_A_correct_B_wrong', 0),
        'A_wrong_B_correct': state.get('event_projection_A_wrong_B_correct', 0),
        'event_conflict_accuracy_A': _ratio(
            state.get('event_projection_A_correct_B_wrong', 0), valid
        ),
        'event_conflict_accuracy_B': _ratio(
            state.get('event_projection_A_wrong_B_correct', 0), valid
        ),
        'routing_correct': state.get('event_projection_routing_correct', 0),
        'routing_wrong': state.get('event_projection_routing_wrong', 0),
        'routing_accuracy': _ratio(
            state.get('event_projection_routing_correct', 0), routing_total
        ),
        'false_merge_conflicts': state.get('event_projection_false_merge_conflicts', 0),
        'false_split_conflicts': state.get('event_projection_false_split_conflicts', 0),
    }


def update_event_projection_oracle_state(state, prob_a, prob_b, gt, events, weights):
    gt_np = gt.detach().cpu().numpy() > 0
    labels_by_sample = {}
    for event in events:
        sample = int(event['sample_index'])
        if sample not in labels_by_sample:
            labels_by_sample[sample], _ = label(
                gt_np[sample],
                structure=np.ones((3, 3), dtype=np.uint8),
            )
        gt_labels = labels_by_sample[sample]
        source = event['source']
        if source == 'A':
            q_child = event['q_child_A']
            q_parent = event['q_parent_A']
            connected_a = bool(event['source_connected'])
            connected_b = bool(event['target_connected'])
        else:
            q_child = event['q_child_B']
            q_parent = event['q_parent_B']
            connected_b = bool(event['source_connected'])
            connected_a = bool(event['target_connected'])
        gt_connected = _gt_connected(gt_labels, q_child, q_parent)
        if gt_connected is None:
            continue
        state['event_projection_oracle_valid_conflicts'] += 1
        a_correct = connected_a == gt_connected
        b_correct = connected_b == gt_connected
        if a_correct and not b_correct:
            state['event_projection_A_correct_B_wrong'] += 1
            oracle_direction = 'a2b'
        elif b_correct and not a_correct:
            state['event_projection_A_wrong_B_correct'] += 1
            oracle_direction = 'b2a'
        else:
            oracle_direction = None
        if connected_a and not gt_connected:
            state['event_projection_false_merge_conflicts'] += 1
        if (not connected_a) and gt_connected:
            state['event_projection_false_split_conflicts'] += 1
        if connected_b and not gt_connected:
            state['event_projection_false_merge_conflicts'] += 1
        if (not connected_b) and gt_connected:
            state['event_projection_false_split_conflicts'] += 1
        if oracle_direction is None:
            continue
        routing, _routing_weight = _predicted_direction(weights[id(event)])
        if routing == oracle_direction:
            state['event_projection_routing_correct'] += 1
        else:
            state['event_projection_routing_wrong'] += 1


def _record_query_stats(stats, query, source):
    if query['status'] == 'invalid_source_event':
        stats['invalid_source_events_' + source] += 1
        return
    stats['valid_source_events_' + source] += 1
    if query['status'] == 'unmatched_target_event':
        key = 'unmatched_target_events_A2B' if source == 'A' else 'unmatched_target_events_B2A'
        stats[key] += 1
        return
    key = 'queried_events_A2B' if source == 'A' else 'queried_events_B2A'
    stats[key] += 1
    if query['source_connected'] == query['target_connected']:
        stats['same_relation_events'] += 1
        return
    if source == 'A':
        if (not query['source_connected']) and query['target_connected']:
            stats['A_separate_B_connected'] += 1
            stats['a_separate_b_connected'] += 1
        else:
            stats['A_connected_B_separate'] += 1
            stats['a_connected_b_separate'] += 1
    else:
        if query['target_connected'] and (not query['source_connected']):
            stats['A_connected_B_separate'] += 1
            stats['a_connected_b_separate'] += 1
        else:
            stats['A_separate_B_connected'] += 1
            stats['a_separate_b_connected'] += 1


def _conflict_from_query(source, event, query):
    if source == 'A':
        connected_a = bool(query['source_connected'])
        connected_b = bool(query['target_connected'])
        conflict_type = (
            'a_connected_b_separate'
            if connected_a else 'a_separate_b_connected'
        )
        return {
            'source': 'A',
            'event_id': event.event_id,
            'source_branch_id': event.source_branch_id,
            'parent_branch_id': event.parent_branch_id,
            'source_connected': connected_a,
            'target_connected': connected_b,
            'connected_a': connected_a,
            'connected_b': connected_b,
            'saddle_coord': event.saddle_coord,
            'saddle_coord_A': event.saddle_coord,
            'saddle_coord_a': event.saddle_coord,
            'q_child_A': query['q_child_source'],
            'q_parent_A': query['q_parent_source'],
            'q_child_B': query['q_child_target'],
            'q_parent_B': query['q_parent_target'],
            'conflict_type': conflict_type,
        }
    connected_b = bool(query['source_connected'])
    connected_a = bool(query['target_connected'])
    conflict_type = (
        'a_connected_b_separate'
        if connected_a else 'a_separate_b_connected'
    )
    return {
        'source': 'B',
        'event_id': event.event_id,
        'source_branch_id': event.source_branch_id,
        'parent_branch_id': event.parent_branch_id,
        'source_connected': connected_b,
        'target_connected': connected_a,
        'connected_a': connected_a,
        'connected_b': connected_b,
        'saddle_coord': event.saddle_coord,
        'saddle_coord_B': event.saddle_coord,
        'saddle_coord_b': event.saddle_coord,
        'q_child_B': query['q_child_source'],
        'q_parent_B': query['q_parent_source'],
        'q_child_A': query['q_child_target'],
        'q_parent_A': query['q_parent_target'],
        'conflict_type': conflict_type,
    }


def _best_foreground_anchor(support, foreground, probability):
    candidates = support & foreground
    if not candidates.any():
        return None
    values = np.where(candidates, probability, -np.inf)
    flat = int(np.argmax(values))
    return tuple(int(v) for v in np.unravel_index(flat, probability.shape))


def _connected(labels, coord_a, coord_b):
    ay, ax = coord_a
    by, bx = coord_b
    label_a = int(labels[ay, ax])
    return label_a > 0 and label_a == int(labels[by, bx])


def _gt_connected(gt_labels, coord_a, coord_b):
    ay, ax = coord_a
    by, bx = coord_b
    label_a = int(gt_labels[ay, ax])
    label_b = int(gt_labels[by, bx])
    if label_a <= 0 or label_b <= 0:
        return None
    return label_a == label_b


def _conflict_roi_from_labels(labels, component_slices, nonzero, saddle_coord, dilation):
    y, x = saddle_coord
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return None, None
    component_id = int(labels[y, x])
    if component_id <= 0:
        ys, xs = nonzero
        if ys.size == 0:
            return None, None
        nearest = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
        component_id = int(labels[ys[nearest], xs[nearest]])
    if component_id <= 0:
        return None, None
    component_slice = component_slices[component_id - 1]
    if component_slice is None:
        return None, None
    roi_slice = _expand_slice(component_slice, labels.shape, dilation)
    component = labels[roi_slice] == component_id
    return roi_slice, _dilate(component, dilation)


def _dilate(mask, radius):
    output = mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, int(radius))):
        output = binary_dilation(output, structure=structure)
    return output


def _expand_slice(component_slice, shape, padding):
    padding = max(0, int(padding))
    y_slice, x_slice = component_slice
    return (
        slice(max(0, y_slice.start - padding), min(shape[0], y_slice.stop + padding)),
        slice(max(0, x_slice.start - padding), min(shape[1], x_slice.stop + padding)),
    )


def _predicted_direction(weights):
    a2b = float(weights['w_a2b'])
    b2a = float(weights['w_b2a'])
    if a2b > b2a and a2b > 0.0:
        return 'a2b', a2b
    if b2a > a2b and b2a > 0.0:
        return 'b2a', b2a
    return 'abstain', 0.0


def _new_event_projection_stats():
    stats = {
        'events': 0,
        'ph_conflict_pixels': 0,
        'raw_ph_branches_A': 0,
        'raw_ph_branches_B': 0,
        'raw_ph_branches_a': 0,
        'raw_ph_branches_b': 0,
        'merge_events_A': 0,
        'merge_events_B': 0,
        'direct_merge_edges_a': 0,
        'direct_merge_edges_b': 0,
        'valid_source_events_A': 0,
        'valid_source_events_B': 0,
        'invalid_source_events_A': 0,
        'invalid_source_events_B': 0,
        'unmatched_target_events_A2B': 0,
        'unmatched_target_events_B2A': 0,
        'queried_events_A2B': 0,
        'queried_events_B2A': 0,
        'same_relation_events': 0,
        'structural_conflict_events': 0,
        'connectivity_conflicts': 0,
        'persistent_connectivity_conflicts': 0,
        'conflict_rate': 0.0,
        'A_separate_B_connected': 0,
        'A_connected_B_separate': 0,
        'a_separate_b_connected': 0,
        'a_connected_b_separate': 0,
        'conflict_roi_count': 0,
        'conflict_pixel_count': 0,
        'conflict_pixel_ratio': 0.0,
        'roi_sizes': [],
        'mean_roi_size': 0.0,
        'median_roi_size': 0.0,
        'mean_conflict_roi_pixels_sum': 0.0,
        'mean_conflict_roi_pixels_count': 0,
        'directional_events_from_persistent_conflicts': 0,
    }
    for key in _timing_keys():
        stats[key] = 0.0
    return stats


def _timing_keys():
    return (
        'ph_extract_time',
        'cc_time',
        'event_query_time',
        'roi_time',
        'routing_time',
        'total_ph_directional_time',
    )


def _merge_stats(total, update):
    for key, value in update.items():
        if key == 'roi_sizes':
            total[key].extend(value)
        elif key in ('conflict_pixel_ratio', 'conflict_rate'):
            continue
        elif key in total:
            total[key] += value


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _median(values):
    return float(np.median(values)) if len(values) else 0.0


def _print_timing(batch_index, stats):
    batch_text = '-' if batch_index is None else str(batch_index)
    print(
        '[PH-event-projection Timing] batch={} branches_A={} branches_B={} '
        'events_A={} events_B={} queried_A2B={} queried_B2A={} conflicts={} '
        'ph={:.4f}s cc={:.4f}s query={:.4f}s roi={:.4f}s total={:.4f}s'.format(
            batch_text,
            stats['raw_ph_branches_A'],
            stats['raw_ph_branches_B'],
            stats['merge_events_A'],
            stats['merge_events_B'],
            stats['queried_events_A2B'],
            stats['queried_events_B2A'],
            stats['structural_conflict_events'],
            stats['ph_extract_time'],
            stats['cc_time'],
            stats['event_query_time'],
            stats['roi_time'],
            stats['total_ph_directional_time'],
        )
    )
