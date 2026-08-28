import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label
from scipy.optimize import linear_sum_assignment

from utils.ph_h0 import (
    collect_h0_pairs_for_components_superlevel,
    mark_h0_death_components_superlevel,
)


def new_ph_mask_epoch_state(epoch, enabled):
    return {
        'epoch': int(epoch),
        'ph_enabled': int(bool(enabled)),
        'ph_conflict_events': 0,
        'merge_conflict_events': 0,
        'split_conflict_events': 0,
        'ph_conflict_pixels': 0,
        'total_unlabeled_pixels': 0,
        'safe_cps_pixels': 0,
        'batches_with_ph_conflict': 0,
        'batches_without_ph_conflict': 0,
        'roi_sizes': [],
        'loss_cps_original_estimate_sum': 0.0,
        'loss_cps_safe_sum': 0.0,
        'loss_batches': 0,
    }


def update_ph_mask_loss_state(state, original_loss, safe_loss):
    state['loss_cps_original_estimate_sum'] += float(original_loss)
    state['loss_cps_safe_sum'] += float(safe_loss)
    state['loss_batches'] += 1


def finalize_ph_mask_epoch_state(state):
    total_pixels = state['total_unlabeled_pixels']
    loss_batches = state['loss_batches']
    original = (
        state['loss_cps_original_estimate_sum'] / loss_batches
        if loss_batches else 0.0
    )
    safe = state['loss_cps_safe_sum'] / loss_batches if loss_batches else 0.0
    roi_sizes = state['roi_sizes']
    return {
        'epoch': state['epoch'],
        'ph_enabled': state['ph_enabled'],
        'ph_conflict_events': state['ph_conflict_events'],
        'merge_conflict_events': state['merge_conflict_events'],
        'split_conflict_events': state['split_conflict_events'],
        'ph_conflict_pixels': state['ph_conflict_pixels'],
        'total_unlabeled_pixels': total_pixels,
        'ph_conflict_pixel_ratio': _ratio(state['ph_conflict_pixels'], total_pixels),
        'ph_mask_efficiency': _ratio(state['ph_conflict_pixels'], total_pixels),
        'safe_cps_pixels': state['safe_cps_pixels'],
        'safe_cps_pixel_ratio': _ratio(state['safe_cps_pixels'], total_pixels),
        'batches_with_ph_conflict': state['batches_with_ph_conflict'],
        'batches_without_ph_conflict': state['batches_without_ph_conflict'],
        'mean_conflict_roi_size': float(np.mean(roi_sizes)) if roi_sizes else 0.0,
        'max_conflict_roi_size': int(max(roi_sizes)) if roi_sizes else 0,
        'loss_cps_original_estimate': original,
        'loss_cps_safe': safe,
        'masked_cps_loss_fraction': (original - safe) / (original + 1e-8),
    }


def build_ph_conflict_mask(prob_a, prob_b, dilation=1):
    """Build unlabeled PH Mask Only conflict mask from A/B predictions.

    PH uses the shared H0 foreground superlevel implementation. No GT is used.
    A disagreement connected component becomes masked only when it contains an
    H0 death critical point from either branch and the local A/B component count
    differs, which approximates an unlabeled merge/split connectivity conflict.
    """
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    batch, height, width = prob_a.shape
    masks = np.zeros((batch, height, width), dtype=bool)
    stats = {
        'events': 0,
        'merge_events': 0,
        'split_events': 0,
        'roi_sizes': [],
    }
    for index in range(batch):
        sample_mask, sample_stats, _sample_events = _build_sample_mask(
            prob_a[index], prob_b[index], dilation
        )
        masks[index] = sample_mask
        for key in ('events', 'merge_events', 'split_events'):
            stats[key] += sample_stats[key]
        stats['roi_sizes'].extend(sample_stats['roi_sizes'])
    return masks, stats


def build_ph_conflict_events(
    prob_a,
    prob_b,
    dilation=1,
    use_matching=False,
    match_radius=16.0,
    unmatched_mode='vanilla',
):
    """Build PH conflict mask and return per-event ROI metadata."""
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    batch, height, width = prob_a.shape
    masks = np.zeros((batch, height, width), dtype=bool)
    events = []
    stats = _new_stats()
    for index in range(batch):
        sample_mask, sample_stats, sample_events = _build_sample_mask(
            prob_a[index],
            prob_b[index],
            dilation,
            use_matching=use_matching,
            match_radius=match_radius,
            unmatched_mode=unmatched_mode,
        )
        masks[index] = sample_mask
        _merge_stats(stats, sample_stats)
        for event in sample_events:
            event['sample_index'] = index
            events.append(event)
    return masks, events, stats


def accumulate_ph_mask_state(state, mask, stats):
    conflict_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    state['ph_conflict_events'] += int(stats['events'])
    state['merge_conflict_events'] += int(stats['merge_events'])
    state['split_conflict_events'] += int(stats['split_events'])
    state['ph_conflict_pixels'] += conflict_pixels
    state['total_unlabeled_pixels'] += total_pixels
    state['safe_cps_pixels'] += total_pixels - conflict_pixels
    if conflict_pixels > 0:
        state['batches_with_ph_conflict'] += 1
    else:
        state['batches_without_ph_conflict'] += 1
    state['roi_sizes'].extend(stats['roi_sizes'])


def _new_stats():
    return {
        'events': 0,
        'merge_events': 0,
        'split_events': 0,
        'roi_sizes': [],
        'ph_pairs_a': 0,
        'ph_pairs_b': 0,
        'matched_ph_pairs': 0,
        'unmatched_ph_a': 0,
        'unmatched_ph_b': 0,
        'matched_connectivity_conflicts': 0,
        'matched_merge_conflicts': 0,
        'matched_split_conflicts': 0,
        'unmatched_conflict_candidates': 0,
        'match_score_sum': 0.0,
        'match_overlap_sum': 0.0,
        'birth_distance_sum': 0.0,
        'death_distance_sum': 0.0,
        'directional_events_from_matched_pairs': 0,
    }


def _merge_stats(total, update):
    for key, value in update.items():
        if key == 'roi_sizes':
            total[key].extend(value)
        elif key in total:
            total[key] += value


def _build_sample_mask(
    prob_a,
    prob_b,
    dilation,
    use_matching=False,
    match_radius=16.0,
    unmatched_mode='vanilla',
):
    pred_a = prob_a > 0.5
    pred_b = prob_b > 0.5
    disagreement = np.logical_xor(pred_a, pred_b)
    output = np.zeros_like(disagreement, dtype=bool)
    stats = _new_stats()
    if not disagreement.any():
        return output, stats, []

    labeled, count = label(disagreement, structure=np.ones((3, 3), dtype=np.uint8))
    slices = find_objects(labeled)
    candidates = []
    component_counts = {}
    for component_id, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        a_count = _local_component_count(pred_a, labeled, component_id, component_slice)
        b_count = _local_component_count(pred_b, labeled, component_id, component_slice)
        if a_count == b_count:
            continue
        candidates.append(component_id)
        component_counts[component_id] = (a_count, b_count)
    if not candidates:
        return output, stats, []
    if use_matching:
        return _build_matched_sample_mask(
            prob_a,
            prob_b,
            labeled,
            slices,
            candidates,
            component_counts,
            output,
            stats,
            dilation,
            match_radius,
            unmatched_mode,
        )

    has_death = set()
    has_death.update(
        mark_h0_death_components_superlevel(prob_a, labeled, candidates)
    )
    has_death.update(
        mark_h0_death_components_superlevel(prob_b, labeled, candidates)
    )
    if not has_death:
        return output, stats, []

    events = []

    for component_id in candidates:
        if component_id not in has_death:
            continue
        component_slice = slices[component_id - 1]
        a_count, b_count = component_counts[component_id]
        roi_slice, roi = _component_roi(labeled, component_id, component_slice, dilation)
        output[roi_slice] |= roi
        stats['events'] += 1
        if a_count < b_count:
            event_type = 'merge'
            stats['merge_events'] += 1
        else:
            event_type = 'split'
            stats['split_events'] += 1
        stats['roi_sizes'].append(int(roi.sum()))
        events.append({
            'component_id': int(component_id),
            'event_type': event_type,
            'roi_slice': roi_slice,
            'roi': roi,
            'roi_size': int(roi.sum()),
            'a_count': int(a_count),
            'b_count': int(b_count),
        })
    return output, stats, events


def _build_matched_sample_mask(
    prob_a,
    prob_b,
    labeled,
    slices,
    candidates,
    component_counts,
    output,
    stats,
    dilation,
    match_radius,
    unmatched_mode,
):
    pairs_a = _component_representative_pairs(
        collect_h0_pairs_for_components_superlevel(prob_a, labeled, candidates)
    )
    pairs_b = _component_representative_pairs(
        collect_h0_pairs_for_components_superlevel(prob_b, labeled, candidates)
    )
    stats['ph_pairs_a'] = len(pairs_a)
    stats['ph_pairs_b'] = len(pairs_b)
    matches, unmatched_a, unmatched_b = _match_ph_pairs(
        pairs_a,
        pairs_b,
        labeled.shape,
        match_radius,
    )
    stats['matched_ph_pairs'] = len(matches)
    stats['unmatched_ph_a'] = len(unmatched_a)
    stats['unmatched_ph_b'] = len(unmatched_b)
    if matches:
        stats['match_score_sum'] = sum(match['score'] for match in matches)
        stats['match_overlap_sum'] = sum(match['overlap'] for match in matches)
        stats['birth_distance_sum'] = sum(match['birth_distance'] for match in matches)
        stats['death_distance_sum'] = sum(match['death_distance'] for match in matches)

    events = []
    conflict_components = set()
    for match in matches:
        pair_a = match['pair_a']
        pair_b = match['pair_b']
        component_id = int(pair_a['component_id'])
        if component_id != int(pair_b['component_id']):
            continue
        if component_id not in component_counts:
            continue
        a_count, b_count = component_counts[component_id]
        if a_count == b_count:
            continue
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        roi_slice, roi = _component_roi(labeled, component_id, component_slice, dilation)
        output[roi_slice] |= roi
        stats['events'] += 1
        stats['matched_connectivity_conflicts'] += 1
        stats['directional_events_from_matched_pairs'] += 1
        if a_count < b_count:
            event_type = 'merge'
            stats['merge_events'] += 1
            stats['matched_merge_conflicts'] += 1
        else:
            event_type = 'split'
            stats['split_events'] += 1
            stats['matched_split_conflicts'] += 1
        stats['roi_sizes'].append(int(roi.sum()))
        conflict_components.add(component_id)
        events.append({
            'component_id': component_id,
            'event_type': event_type,
            'roi_slice': roi_slice,
            'roi': roi,
            'roi_size': int(roi.sum()),
            'a_count': int(a_count),
            'b_count': int(b_count),
            'match_score': float(match['score']),
            'match_overlap': float(match['overlap']),
            'birth_distance': float(match['birth_distance']),
            'death_distance': float(match['death_distance']),
            'pair_a': pair_a,
            'pair_b': pair_b,
            'matched_pair': True,
        })

    unmatched_components = _unmatched_candidate_components(
        candidates,
        pairs_a,
        pairs_b,
        unmatched_a,
        unmatched_b,
        conflict_components,
    )
    stats['unmatched_conflict_candidates'] = len(unmatched_components)
    if unmatched_mode == 'abstain':
        for component_id in sorted(unmatched_components):
            component_slice = slices[component_id - 1]
            if component_slice is None:
                continue
            roi_slice, roi = _component_roi(labeled, component_id, component_slice, dilation)
            output[roi_slice] |= roi
            events.append({
                'component_id': int(component_id),
                'event_type': 'unmatched',
                'roi_slice': roi_slice,
                'roi': roi,
                'roi_size': int(roi.sum()),
                'a_count': int(component_counts[component_id][0]),
                'b_count': int(component_counts[component_id][1]),
                'force_abstain': True,
                'matched_pair': False,
            })
    return output, stats, events


def _match_ph_pairs(pairs_a, pairs_b, shape, match_radius):
    if not pairs_a or not pairs_b:
        return [], list(range(len(pairs_a))), list(range(len(pairs_b)))
    score = np.zeros((len(pairs_a), len(pairs_b)), dtype=np.float32)
    details = {}
    diagonal = float(np.hypot(shape[0], shape[1]))
    for index_a, pair_a in enumerate(pairs_a):
        for index_b, pair_b in enumerate(pairs_b):
            values = _matching_score(pair_a, pair_b, diagonal)
            score[index_a, index_b] = values['score']
            details[(index_a, index_b)] = values
    rows, cols = linear_sum_assignment(-score)
    matches = []
    matched_a = set()
    matched_b = set()
    radius = float(match_radius)
    for row, col in zip(rows, cols):
        values = details[(int(row), int(col))]
        if max(values['birth_distance_pixels'], values['death_distance_pixels']) > radius:
            continue
        matched_a.add(int(row))
        matched_b.add(int(col))
        matches.append({
            'pair_a': pairs_a[int(row)],
            'pair_b': pairs_b[int(col)],
            **values,
        })
    unmatched_a = [index for index in range(len(pairs_a)) if index not in matched_a]
    unmatched_b = [index for index in range(len(pairs_b)) if index not in matched_b]
    return matches, unmatched_a, unmatched_b


def _component_representative_pairs(pairs):
    representatives = {}
    for pair in pairs:
        component_id = int(pair['component_id'])
        current = representatives.get(component_id)
        if current is None or _pair_rank(pair) > _pair_rank(current):
            representatives[component_id] = pair
    return list(representatives.values())


def _pair_rank(pair):
    by, bx = pair['birth_coord']
    dy, dx = pair['death_coord']
    return (
        float(pair['persistence']),
        -float(np.hypot(float(by) - float(dy), float(bx) - float(dx))),
        -int(by),
        -int(bx),
    )


def _matching_score(pair_a, pair_b, diagonal):
    birth_distance_pixels = _coord_distance(pair_a['birth_coord'], pair_b['birth_coord'])
    death_distance_pixels = _coord_distance(pair_a['death_coord'], pair_b['death_coord'])
    birth_distance = birth_distance_pixels / max(diagonal, 1e-8)
    death_distance = death_distance_pixels / max(diagonal, 1e-8)
    birth_similarity = 1.0 - min(birth_distance, 1.0)
    death_similarity = 1.0 - min(death_distance, 1.0)
    overlap = 1.0 if int(pair_a['component_id']) == int(pair_b['component_id']) else 0.0
    persistence_denominator = max(
        abs(float(pair_a['persistence'])),
        abs(float(pair_b['persistence'])),
        1e-8,
    )
    persistence_similarity = 1.0 - min(
        abs(float(pair_a['persistence']) - float(pair_b['persistence']))
        / persistence_denominator,
        1.0,
    )
    score = (
        2.0 * overlap
        + birth_similarity
        + death_similarity
        + 0.25 * persistence_similarity
    )
    return {
        'score': float(score),
        'overlap': float(overlap),
        'birth_distance': float(birth_distance),
        'death_distance': float(death_distance),
        'birth_distance_pixels': float(birth_distance_pixels),
        'death_distance_pixels': float(death_distance_pixels),
    }


def _coord_distance(coord_a, coord_b):
    ay, ax = coord_a
    by, bx = coord_b
    return float(np.hypot(float(ay) - float(by), float(ax) - float(bx)))


def _unmatched_candidate_components(
    candidates,
    pairs_a,
    pairs_b,
    unmatched_a,
    unmatched_b,
    conflict_components,
):
    unmatched_components = set()
    for index in unmatched_a:
        unmatched_components.add(int(pairs_a[index]['component_id']))
    for index in unmatched_b:
        unmatched_components.add(int(pairs_b[index]['component_id']))
    return (
        set(int(component_id) for component_id in candidates)
        & unmatched_components
        - set(conflict_components)
    )


def _local_component_count(mask, labeled_components, component_id, component_slice):
    local_slice = _expand_slice(component_slice, labeled_components.shape, 1)
    roi = labeled_components[local_slice] == component_id
    local_mask = mask[local_slice] & _dilate(roi, 1)
    labeled, count = label(local_mask, structure=np.ones((3, 3), dtype=np.uint8))
    del labeled
    return int(count)


def _coord_in_mask(coord, mask):
    y, x = coord
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _dilate(mask, radius):
    output = mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, int(radius))):
        output = binary_dilation(output, structure=structure)
    return output


def _component_roi(labeled_components, component_id, component_slice, dilation):
    local_slice = _expand_slice(component_slice, labeled_components.shape, dilation)
    component = labeled_components[local_slice] == component_id
    return local_slice, _dilate(component, dilation)


def _expand_slice(component_slice, shape, padding):
    padding = max(0, int(padding))
    y_slice, x_slice = component_slice
    y_start = max(0, y_slice.start - padding)
    y_stop = min(shape[0], y_slice.stop + padding)
    x_start = max(0, x_slice.start - padding)
    x_stop = min(shape[1], x_slice.stop + padding)
    return (slice(y_start, y_stop), slice(x_start, x_stop))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0
