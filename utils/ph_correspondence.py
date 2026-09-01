from dataclasses import dataclass
import time

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label
from scipy.spatial import cKDTree
from utils.ph_relation_tree import (
    build_merge_hierarchy,
    build_relation_candidates,
    detect_ph_relation_conflicts,
)

try:
    from numba import njit
except ImportError:
    njit = None


@dataclass
class PersistentBranch:
    branch_id: int
    branch_id_str: str
    birth: float
    death: float
    birth_coord: tuple
    death_coord: tuple
    parent_branch_id: int
    is_essential: bool

    @property
    def entity_id(self):
        return self.branch_id

    @property
    def branch_label(self):
        return self.branch_id_str


PersistentEntity = PersistentBranch


@dataclass
class DirectMergeEdge:
    child_branch_id: int
    parent_branch_id: int
    merge_level: float
    merge_coord: tuple

    @property
    def child_id(self):
        return self.child_branch_id

    @property
    def parent_id(self):
        return self.parent_branch_id

    @property
    def saddle_coord(self):
        return self.merge_coord


@dataclass
class BranchDecomposition:
    branches: list
    edges: list
    branch_label_map: np.ndarray
    support_sizes: np.ndarray


def build_persistent_relation_conflict_events(
    prob_a,
    prob_b,
    dilation=1,
    tau=0.5,
    unmatched_mode='vanilla',
    debug_timing=False,
    batch_index=None,
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    collect_cc_agreement=False,
    presence_conflict_mode='strict',
):
    del unmatched_mode
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    batch, height, width = prob_a.shape
    masks = np.zeros((batch, height, width), dtype=bool)
    events = []
    stats = _new_correspondence_stats()

    for index in range(batch):
        sample_mask, sample_events, sample_stats = detect_connectivity_conflicts(
            prob_a[index],
            prob_b[index],
            dilation=dilation,
            tau=tau,
            correspondence_mode=correspondence_mode,
            relation_backend=relation_backend,
            collect_cc_agreement=collect_cc_agreement,
            presence_conflict_mode=presence_conflict_mode,
        )
        masks[index] = sample_mask
        _merge_correspondence_stats(stats, sample_stats)
        for event in sample_events:
            event['sample_index'] = index
            events.append(event)
        for question in stats.get('relation_question_records', []):
            question.setdefault('sample_index', index)

    stats['conflict_pixel_ratio'] = _ratio(stats['ph_conflict_pixels'], masks.size)
    if debug_timing:
        _print_timing(batch_index, stats, correspondence_mode)
    return masks, events, stats


def detect_connectivity_conflicts(
    prob_a,
    prob_b,
    dilation=1,
    tau=0.5,
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    collect_cc_agreement=False,
    presence_conflict_mode='strict',
):
    timings = {key: 0.0 for key in _timing_keys()}
    total_start = time.perf_counter()

    start = time.perf_counter()
    decomposition_a = decompose_h0_branches(prob_a, 'A')
    decomposition_b = decompose_h0_branches(prob_b, 'B')
    timings['ph_extract_time'] = time.perf_counter() - start
    timings['support_build_time'] = timings['ph_extract_time']

    start = time.perf_counter()
    if correspondence_mode == 'support_identity':
        matches, unmatched_a, unmatched_b, match_stats = match_branches_by_support(
            decomposition_a,
            decomposition_b,
            prob_a.shape,
        )
        timings['support_overlap_time'] = match_stats['support_overlap_time']
    elif correspondence_mode == 'birth_nearest':
        matches, unmatched_a, unmatched_b, match_stats = match_branches_by_birth(
            decomposition_a.branches,
            decomposition_b.branches,
            prob_a.shape,
        )
    else:
        raise ValueError('Unsupported PH correspondence mode: {}'.format(correspondence_mode))
    timings['entity_match_time'] = time.perf_counter() - start
    timings['birth_matching_time'] = timings['entity_match_time']

    start = time.perf_counter()
    pred_a = np.asarray(prob_a) >= tau
    pred_b = np.asarray(prob_b) >= tau
    if relation_backend == 'ph_tree':
        candidate_start = time.perf_counter()
        tree_a = build_merge_hierarchy(decomposition_a)
        tree_b = build_merge_hierarchy(decomposition_b)
        relation_candidates = build_relation_candidates(
            decomposition_a.edges,
            decomposition_b.edges,
            matches,
        )
        timings['candidate_build_time'] = time.perf_counter() - candidate_start
        query_start = time.perf_counter()
        conflict_specs, relation_tree_stats = detect_ph_relation_conflicts(
            relation_candidates,
            tree_a,
            tree_b,
            matches,
            tau=tau,
            collect_cc_agreement=collect_cc_agreement,
            pred_a=pred_a,
            pred_b=pred_b,
            presence_conflict_mode=presence_conflict_mode,
        )
        timings['relation_tree_query_time'] = time.perf_counter() - query_start
    elif relation_backend == 'cc':
        candidate_start = time.perf_counter()
        cc_a, _count_a = label(pred_a, structure=np.ones((3, 3), dtype=np.uint8))
        cc_b, _count_b = label(pred_b, structure=np.ones((3, 3), dtype=np.uint8))
        relation_candidates = _aligned_direct_edges(
            decomposition_a.edges,
            decomposition_b.edges,
            matches,
        )
        timings['candidate_build_time'] = time.perf_counter() - candidate_start
        conflict_specs = _connectivity_conflicts(
            relation_candidates,
            decomposition_a.branches,
            decomposition_b.branches,
            matches,
            cc_a,
            cc_b,
        )
        relation_tree_stats = {}
        timings['relation_tree_query_time'] = 0.0
    else:
        raise ValueError('Unsupported PH relation backend: {}'.format(relation_backend))
    timings['relation_time'] = time.perf_counter() - start

    start = time.perf_counter()
    masks, events, roi_sizes = _build_conflict_rois(
        pred_a,
        pred_b,
        conflict_specs,
        dilation,
    )
    timings['roi_time'] = time.perf_counter() - start
    timings['total_ph_time'] = time.perf_counter() - total_start

    stats = _new_correspondence_stats()
    stats.update(timings)
    _fill_correspondence_stats(
        stats,
        decomposition_a,
        decomposition_b,
        matches,
        unmatched_a,
        unmatched_b,
        match_stats,
        relation_candidates,
        conflict_specs,
        events,
        roi_sizes,
        masks,
        relation_tree_stats,
    )
    return masks, events, stats


def decompose_h0_branches(probability, branch_id='A'):
    probability = np.asarray(probability, dtype=np.float32)
    height, width = probability.shape
    order = np.lexsort((
        np.arange(height * width, dtype=np.int64),
        -probability.reshape(-1),
    ))
    if njit is not None:
        arrays = _decompose_h0_branches_numba(probability, order)
        return _decomposition_from_arrays(arrays, branch_id)
    return _decompose_h0_branches_python(probability, order, branch_id)


def match_branches_by_birth(branches_a, branches_b, shape):
    if not branches_a or not branches_b:
        return [], list(range(len(branches_a))), list(range(len(branches_b))), _empty_match_stats()

    diagonal = max(float(np.hypot(shape[0], shape[1])), 1e-8)
    coords_a = np.asarray([branch.birth_coord for branch in branches_a], dtype=np.float32)
    coords_b = np.asarray([branch.birth_coord for branch in branches_b], dtype=np.float32)
    tree_b = cKDTree(coords_b)
    tree_a = cKDTree(coords_a)
    distances_ab, nearest_b_for_a = tree_b.query(coords_a, k=1)
    _distances_ba, nearest_a_for_b = tree_a.query(coords_b, k=1)

    matched_a = set()
    matched_b = set()
    matches = []
    for row, col in enumerate(nearest_b_for_a):
        row = int(row)
        col = int(col)
        if int(nearest_a_for_b[col]) != row:
            continue
        distance = float(distances_ab[row] / diagonal)
        matched_a.add(row)
        matched_b.add(col)
        matches.append(_new_match(branches_a[row], branches_b[col], 0, 0.0, distance))

    stats = _empty_match_stats()
    stats['matched_support_iou_values'] = [match['support_iou'] for match in matches]
    stats['birth_distance_values'] = [match['birth_distance'] for match in matches]
    return (
        matches,
        [index for index in range(len(branches_a)) if index not in matched_a],
        [index for index in range(len(branches_b)) if index not in matched_b],
        stats,
    )


def match_branches_by_support(decomposition_a, decomposition_b, shape):
    stats = _empty_match_stats()
    start = time.perf_counter()
    pairs = _support_overlap_pairs(
        decomposition_a.branch_label_map,
        decomposition_b.branch_label_map,
        decomposition_a.support_sizes,
        decomposition_b.support_sizes,
    )
    stats['support_overlap_time'] = time.perf_counter() - start
    stats['support_candidate_pairs'] = len(pairs)
    if not pairs:
        return (
            [],
            list(range(len(decomposition_a.branches))),
            list(range(len(decomposition_b.branches))),
            stats,
        )

    diagonal = max(float(np.hypot(shape[0], shape[1])), 1e-8)
    ranked = []
    for branch_a_id, branch_b_id, intersection, support_iou in pairs:
        branch_a = decomposition_a.branches[branch_a_id]
        branch_b = decomposition_b.branches[branch_b_id]
        birth_distance = _coord_distance(branch_a.birth_coord, branch_b.birth_coord) / diagonal
        ranked.append((
            -float(support_iou),
            float(birth_distance),
            int(branch_a_id),
            int(branch_b_id),
            int(intersection),
            float(support_iou),
        ))
    ranked.sort()

    matched_a = set()
    matched_b = set()
    matches = []
    for _neg_iou, birth_distance, branch_a_id, branch_b_id, intersection, support_iou in ranked:
        if branch_a_id in matched_a or branch_b_id in matched_b:
            continue
        matched_a.add(branch_a_id)
        matched_b.add(branch_b_id)
        matches.append(_new_match(
            decomposition_a.branches[branch_a_id],
            decomposition_b.branches[branch_b_id],
            intersection,
            support_iou,
            birth_distance,
        ))

    stats['matched_support_iou_values'] = [match['support_iou'] for match in matches]
    stats['birth_distance_values'] = [match['birth_distance'] for match in matches]
    return (
        matches,
        [index for index in range(len(decomposition_a.branches)) if index not in matched_a],
        [index for index in range(len(decomposition_b.branches)) if index not in matched_b],
        stats,
    )


def _support_overlap_pairs(label_a, label_b, sizes_a, sizes_b):
    valid = (label_a >= 0) & (label_b >= 0)
    if not valid.any():
        return []
    count_b = int(sizes_b.shape[0])
    encoded = label_a[valid].astype(np.int64) * count_b + label_b[valid].astype(np.int64)
    keys, counts = np.unique(encoded, return_counts=True)
    pairs = []
    for key, intersection in zip(keys, counts):
        branch_a_id = int(key // count_b)
        branch_b_id = int(key - branch_a_id * count_b)
        union = int(sizes_a[branch_a_id]) + int(sizes_b[branch_b_id]) - int(intersection)
        if union <= 0:
            continue
        pairs.append((
            branch_a_id,
            branch_b_id,
            int(intersection),
            float(intersection / union),
        ))
    return pairs


def _aligned_direct_edges(edges_a, edges_b, matches):
    a_to_canonical = {}
    b_to_canonical = {}
    for canonical_id, match in enumerate(matches):
        a_to_canonical[match['entity_a'].branch_id] = canonical_id
        b_to_canonical[match['entity_b'].branch_id] = canonical_id

    candidates = {}
    for edge in edges_a:
        if edge.child_branch_id not in a_to_canonical or edge.parent_branch_id not in a_to_canonical:
            continue
        key = tuple(sorted((
            a_to_canonical[edge.child_branch_id],
            a_to_canonical[edge.parent_branch_id],
        )))
        if key[0] == key[1]:
            continue
        candidates.setdefault(key, {})['edge_a'] = edge
    for edge in edges_b:
        if edge.child_branch_id not in b_to_canonical or edge.parent_branch_id not in b_to_canonical:
            continue
        key = tuple(sorted((
            b_to_canonical[edge.child_branch_id],
            b_to_canonical[edge.parent_branch_id],
        )))
        if key[0] == key[1]:
            continue
        candidates.setdefault(key, {})['edge_b'] = edge
    return candidates


def _connectivity_conflicts(relation_candidates, branches_a, branches_b, matches, cc_a, cc_b):
    del branches_a, branches_b
    conflicts = []
    for canonical_pair, edge_refs in relation_candidates.items():
        left_id, right_id = canonical_pair
        left_match = matches[left_id]
        right_match = matches[right_id]
        left_a = left_match['entity_a']
        right_a = right_match['entity_a']
        left_b = left_match['entity_b']
        right_b = right_match['entity_b']
        connected_a = _connected(cc_a, left_a.birth_coord, right_a.birth_coord)
        connected_b = _connected(cc_b, left_b.birth_coord, right_b.birth_coord)
        if connected_a == connected_b:
            continue
        if connected_a:
            conflict_type = 'a_connected_b_separate'
            saddle = _edge_coord(edge_refs.get('edge_a'), left_a, right_a)
        else:
            conflict_type = 'a_separate_b_connected'
            saddle = _edge_coord(edge_refs.get('edge_b'), left_b, right_b)
        conflicts.append({
            'canonical_pair': canonical_pair,
            'left_match': left_match,
            'right_match': right_match,
            'edge_a': edge_refs.get('edge_a'),
            'edge_b': edge_refs.get('edge_b'),
            'connected_a': connected_a,
            'connected_b': connected_b,
            'conflict_type': conflict_type,
            'saddle_coord': saddle,
        })
    return conflicts


def _build_conflict_rois(pred_a, pred_b, conflict_specs, dilation):
    masks = np.zeros_like(pred_a, dtype=bool)
    events = []
    roi_sizes = []
    if not conflict_specs:
        return masks, events, roi_sizes

    disagreement = np.logical_xor(pred_a, pred_b)
    if not disagreement.any():
        return masks, events, roi_sizes

    disagreement_labels, _count = label(
        disagreement,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_slices = find_objects(disagreement_labels)
    ys, xs = np.nonzero(disagreement_labels)

    for spec in conflict_specs:
        roi_slice, roi = _conflict_roi_from_labels(
            disagreement_labels,
            component_slices,
            (ys, xs),
            spec['saddle_coord'],
            dilation,
        )
        if roi is None:
            continue
        masks[roi_slice] |= roi
        roi_size = int(roi.sum())
        roi_sizes.append(roi_size)
        events.append({
            'event_type': 'connectivity_conflict',
            'conflict_type': spec['conflict_type'],
            'roi_slice': roi_slice,
            'roi': roi,
            'roi_size': roi_size,
            'matched_pair': True,
            'persistent_relation': True,
            'left_match': spec['left_match'],
            'right_match': spec['right_match'],
            'edge_a': spec['edge_a'],
            'edge_b': spec['edge_b'],
            'mu_a': _spec_merge_level(spec, 'a'),
            'mu_b': _spec_merge_level(spec, 'b'),
            'connected_a': spec['connected_a'],
            'connected_b': spec['connected_b'],
            'saddle_coord_a': _spec_saddle_coord(spec, 'a'),
            'saddle_coord_b': _spec_saddle_coord(spec, 'b'),
            'relation_query_a': spec.get('query_a'),
            'relation_query_b': spec.get('query_b'),
            'relation_source': spec.get('source', ''),
            'structural_conflict_type': spec.get('structural_conflict_type', ''),
            'connected_side': spec.get('connected_side', ''),
            'candidate_id': spec.get('candidate_id', ''),
            'pair_present_a': spec.get('pair_present_a', False),
            'pair_present_b': spec.get('pair_present_b', False),
        })
    return masks, events, roi_sizes


def _fill_correspondence_stats(
    stats,
    decomposition_a,
    decomposition_b,
    matches,
    unmatched_a,
    unmatched_b,
    match_stats,
    relation_candidates,
    conflict_specs,
    events,
    roi_sizes,
    masks,
    relation_tree_stats=None,
):
    support_iou_values = match_stats['matched_support_iou_values']
    birth_distance_values = match_stats['birth_distance_values']
    stats['ph_entities_a'] = len(decomposition_a.branches)
    stats['ph_entities_b'] = len(decomposition_b.branches)
    stats['raw_ph_branches_a'] = len(decomposition_a.branches)
    stats['raw_ph_branches_b'] = len(decomposition_b.branches)
    stats['matched_entities'] = len(matches)
    stats['matched_branches'] = len(matches)
    stats['unmatched_entities_a'] = len(unmatched_a)
    stats['unmatched_entities_b'] = len(unmatched_b)
    stats['unmatched_branches_a'] = len(unmatched_a)
    stats['unmatched_branches_b'] = len(unmatched_b)
    stats['support_candidate_pairs'] = match_stats['support_candidate_pairs']
    stats['mean_birth_match_distance'] = _mean(birth_distance_values)
    stats['birth_distance_sum'] = float(sum(birth_distance_values))
    stats['mean_matched_birth_distance'] = _mean(birth_distance_values)
    stats['mean_matched_support_iou'] = _mean(support_iou_values)
    stats['median_matched_support_iou'] = _median(support_iou_values)
    stats['support_iou_sum'] = float(sum(support_iou_values))
    stats['support_iou_count'] = len(support_iou_values)
    stats['support_iou_values'] = list(float(value) for value in support_iou_values)
    stats['mean_support_area_a'] = _mean(decomposition_a.support_sizes)
    stats['mean_support_area_b'] = _mean(decomposition_b.support_sizes)
    stats['direct_merge_edges_a'] = len(decomposition_a.edges)
    stats['direct_merge_edges_b'] = len(decomposition_b.edges)
    stats['aligned_relation_candidates'] = len(relation_candidates)
    stats['relation_candidates'] = len(relation_candidates)
    stats['connectivity_conflicts'] = len(conflict_specs)
    stats['persistent_connectivity_conflicts'] = len(conflict_specs)
    stats['directional_events_from_persistent_conflicts'] = len(events)
    stats['events'] = len(events)
    stats['ph_conflict_pixels'] = int(masks.sum())
    stats['roi_sizes'] = roi_sizes
    stats['mean_conflict_roi_pixels_sum'] = float(sum(roi_sizes))
    stats['mean_conflict_roi_pixels_count'] = len(roi_sizes)
    stats['a_separate_b_connected'] = sum(
        1 for event in events if event['conflict_type'] == 'a_separate_b_connected'
    )
    stats['a_connected_b_separate'] = sum(
        1 for event in events if event['conflict_type'] == 'a_connected_b_separate'
    )
    stats['conflict_pixel_ratio'] = _ratio(stats['ph_conflict_pixels'], masks.size)
    stats['pure_relation_conflict_pixels'] = int(sum(
        event['roi_size']
        for event in events
        if event.get('structural_conflict_type') == 'pure_relation'
    ))
    stats['presence_involved_conflict_pixels'] = int(sum(
        event['roi_size']
        for event in events
        if event.get('structural_conflict_type') == 'presence_involved'
    ))
    if relation_tree_stats:
        for key, value in relation_tree_stats.items():
            if key in stats:
                stats[key] += value
            else:
                stats[key] = value


def _new_match(branch_a, branch_b, support_intersection, support_iou, birth_distance):
    return {
        'entity_a': branch_a,
        'entity_b': branch_b,
        'branch_a_id': branch_a.branch_id,
        'branch_b_id': branch_b.branch_id,
        'support_intersection': int(support_intersection),
        'support_iou': float(support_iou),
        'birth_distance': float(birth_distance),
        'score': float(support_iou),
        'core_iou': float(support_iou),
        'centroid_distance': float(birth_distance),
    }


def _decomposition_from_arrays(arrays, branch_id):
    (
        branch_count,
        branch_birth,
        branch_death,
        birth_y,
        birth_x,
        death_y,
        death_x,
        parent_branch,
        is_essential,
        edge_count,
        edge_child,
        edge_parent,
        edge_level,
        edge_y,
        edge_x,
        branch_label_map,
        support_sizes,
    ) = arrays
    branches = []
    for index in range(int(branch_count)):
        branches.append(PersistentBranch(
            branch_id=index,
            branch_id_str=branch_id,
            birth=float(branch_birth[index]),
            death=float(branch_death[index]),
            birth_coord=(int(birth_y[index]), int(birth_x[index])),
            death_coord=(int(death_y[index]), int(death_x[index])),
            parent_branch_id=(
                None if int(parent_branch[index]) < 0 else int(parent_branch[index])
            ),
            is_essential=bool(is_essential[index]),
        ))
    edges = []
    for index in range(int(edge_count)):
        edges.append(DirectMergeEdge(
            child_branch_id=int(edge_child[index]),
            parent_branch_id=int(edge_parent[index]),
            merge_level=float(edge_level[index]),
            merge_coord=(int(edge_y[index]), int(edge_x[index])),
        ))
    return BranchDecomposition(
        branches=branches,
        edges=edges,
        branch_label_map=branch_label_map.astype(np.int64, copy=False),
        support_sizes=support_sizes[:int(branch_count)].astype(np.int64, copy=False),
    )


def _decompose_h0_branches_python(probability, order, branch_id='A'):
    height, width = probability.shape
    parent = np.arange(height * width, dtype=np.int64)
    active = np.zeros(height * width, dtype=bool)
    birth = np.zeros(height * width, dtype=np.float32)
    birth_y = np.zeros(height * width, dtype=np.int64)
    birth_x = np.zeros(height * width, dtype=np.int64)
    root_branch = np.full(height * width, -1, dtype=np.int64)
    branch_label_map = np.full(height * width, -1, dtype=np.int64)
    branches = []
    edges = []
    support_sizes = []

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def find(item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def root_is_less(root_a, root_b):
        return (
            float(birth[root_a]),
            int(birth_y[root_a]),
            int(birth_x[root_a]),
        ) > (
            float(birth[root_b]),
            int(birth_y[root_b]),
            int(birth_x[root_b]),
        )

    for flat in order:
        flat = int(flat)
        y, x = divmod(flat, width)
        roots = set()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbor = ny * width + nx
                if active[neighbor]:
                    roots.add(find(neighbor))

        value = float(probability[y, x])
        active[flat] = True
        parent[flat] = flat
        birth[flat] = value
        birth_y[flat] = y
        birth_x[flat] = x

        if not roots:
            branch_index = len(branches)
            root_branch[flat] = branch_index
            branch_label_map[flat] = branch_index
            support_sizes.append(1)
            branches.append(PersistentBranch(
                branch_id=branch_index,
                branch_id_str=branch_id,
                birth=value,
                death=value,
                birth_coord=(y, x),
                death_coord=(y, x),
                parent_branch_id=None,
                is_essential=True,
            ))
            continue

        roots.add(flat)
        survivor = min(roots, key=lambda root: (-float(birth[root]), int(birth_y[root]), int(birth_x[root])))
        branch_label = int(root_branch[survivor])
        branch_label_map[flat] = branch_label
        if branch_label >= 0:
            support_sizes[branch_label] += 1

        for root in sorted(roots):
            root = find(root)
            survivor = find(survivor)
            if root == survivor:
                continue
            dead_root = root
            if root_is_less(root, survivor):
                dead_root = survivor
                survivor = root
            child = int(root_branch[dead_root])
            parent_id = int(root_branch[survivor])
            if child >= 0:
                branches[child].death = value
                branches[child].death_coord = (y, x)
                branches[child].parent_branch_id = parent_id
                branches[child].is_essential = False
                if parent_id >= 0:
                    edges.append(DirectMergeEdge(child, parent_id, value, (y, x)))
            parent[dead_root] = survivor
            if root_branch[survivor] < 0 and child >= 0:
                root_branch[survivor] = child

    return BranchDecomposition(
        branches=branches,
        edges=edges,
        branch_label_map=branch_label_map.reshape(height, width),
        support_sizes=np.asarray(support_sizes, dtype=np.int64),
    )


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


def _connected(labels, coord_a, coord_b):
    ay, ax = coord_a
    by, bx = coord_b
    label_a = int(labels[ay, ax])
    return label_a > 0 and label_a == int(labels[by, bx])


def _edge_coord(edge, branch_a, branch_b):
    if edge is not None:
        return edge.merge_coord
    ay = int(0.5 * (branch_a.birth_coord[0] + branch_b.birth_coord[0]))
    ax = int(0.5 * (branch_a.birth_coord[1] + branch_b.birth_coord[1]))
    return (ay, ax)


def _edge_level(edge):
    if edge is None:
        return 0.0
    return float(edge.merge_level)


def _spec_merge_level(spec, branch):
    query = spec.get('query_' + branch)
    if query is not None:
        return float(query.merge_level)
    return _edge_level(spec.get('edge_' + branch))


def _spec_saddle_coord(spec, branch):
    query = spec.get('query_' + branch)
    if query is not None and query.saddle_coord is not None:
        return query.saddle_coord
    left_key = 'entity_' + branch
    return _edge_coord(
        spec.get('edge_' + branch),
        spec['left_match'][left_key],
        spec['right_match'][left_key],
    )


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


def _empty_match_stats():
    return {
        'support_overlap_time': 0.0,
        'support_candidate_pairs': 0,
        'matched_support_iou_values': [],
        'birth_distance_values': [],
    }


def _new_correspondence_stats():
    stats = {
        'events': 0,
        'merge_events': 0,
        'split_events': 0,
        'roi_sizes': [],
        'ph_conflict_pixels': 0,
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
        'ph_entities_a': 0,
        'ph_entities_b': 0,
        'raw_ph_branches_a': 0,
        'raw_ph_branches_b': 0,
        'matched_entities': 0,
        'matched_branches': 0,
        'unmatched_entities_a': 0,
        'unmatched_entities_b': 0,
        'unmatched_branches_a': 0,
        'unmatched_branches_b': 0,
        'support_candidate_pairs': 0,
        'mean_birth_match_distance': 0.0,
        'mean_matched_birth_distance': 0.0,
        'mean_matched_support_iou': 0.0,
        'median_matched_support_iou': 0.0,
        'support_iou_sum': 0.0,
        'support_iou_count': 0,
        'support_iou_values': [],
        'core_iou_sum': 0.0,
        'centroid_distance_sum': 0.0,
        'mean_support_area_a': 0.0,
        'mean_support_area_b': 0.0,
        'direct_merge_edges_a': 0,
        'direct_merge_edges_b': 0,
        'aligned_relation_candidates': 0,
        'relation_candidates': 0,
        'connectivity_conflicts': 0,
        'persistent_connectivity_conflicts': 0,
        'a_separate_b_connected': 0,
        'a_connected_b_separate': 0,
        'existence_conflict_a_only': 0,
        'existence_conflict_b_only': 0,
        'merge_tree_cc_consistency_sum': 0.0,
        'merge_tree_cc_consistency_count': 0,
        'conflict_pixel_ratio': 0.0,
        'directional_events_from_persistent_conflicts': 0,
        'mean_conflict_roi_pixels_sum': 0.0,
        'mean_conflict_roi_pixels_count': 0,
        'num_valid_relation_queries': 0,
        'num_absent_A': 0,
        'num_absent_B': 0,
        'num_connected_A': 0,
        'num_separated_A': 0,
        'num_connected_B': 0,
        'num_separated_B': 0,
        'merge_level_A_sum': 0.0,
        'merge_level_B_sum': 0.0,
        'abs_g_A_sum': 0.0,
        'abs_g_B_sum': 0.0,
        'num_relation_conflicts': 0,
        'num_A_connected_B_separated': 0,
        'num_A_separated_B_connected': 0,
        'conflict_merge_level_A_sum': 0.0,
        'conflict_merge_level_B_sum': 0.0,
        'valid_compare_count': 0,
        'ph_cc_agree_A': 0,
        'ph_cc_agree_B': 0,
        'num_relation_questions': 0,
        'num_pair_present_both': 0,
        'num_pair_absent_a': 0,
        'num_pair_absent_b': 0,
        'num_pair_absent_both': 0,
        'num_pure_relation_conflicts': 0,
        'num_presence_involved_conflicts': 0,
        'num_total_structural_conflicts': 0,
        'num_presence_disagreement_nonconflict': 0,
        'pure_relation_conflict_pixels': 0,
        'presence_involved_conflict_pixels': 0,
        'old_cc_conflicts': 0,
        'new_conflicts': 0,
        'old_cc_retained_conflicts': 0,
        'old_cc_lost_conflicts': 0,
        'new_extra_conflicts': 0,
        'old_cc_to_pure_relation': 0,
        'old_cc_to_presence_involved': 0,
        'old_cc_to_nonconflict': 0,
        'old_cc_candidate_missing': 0,
        'relation_question_records': [],
    }
    for key in _timing_keys():
        stats[key] = 0.0
    return stats


def _merge_correspondence_stats(total, update):
    for key, value in update.items():
        if key in ('roi_sizes', 'support_iou_values', 'relation_question_records'):
            total[key].extend(value)
        elif key == 'conflict_pixel_ratio':
            continue
        elif key in total:
            total[key] += value


def _print_timing(batch_index, stats, mode):
    batch_text = '-' if batch_index is None else str(batch_index)
    print(
        '[PH-v2 Timing] mode={} batch={} branches_A={} branches_B={} '
        'support_candidates={} matched={} edges_A={} edges_B={} '
        'relation_candidates={} conflicts={} ph={:.4f}s support={:.4f}s '
        'overlap={:.4f}s matching={:.4f}s relation={:.4f}s roi={:.4f}s total={:.4f}s'.format(
            mode,
            batch_text,
            stats['raw_ph_branches_a'],
            stats['raw_ph_branches_b'],
            stats['support_candidate_pairs'],
            stats['matched_branches'],
            stats['direct_merge_edges_a'],
            stats['direct_merge_edges_b'],
            stats['aligned_relation_candidates'],
            stats['connectivity_conflicts'],
            stats['ph_extract_time'],
            stats['support_build_time'],
            stats['support_overlap_time'],
            stats['entity_match_time'],
            stats['relation_time'],
            stats['roi_time'],
            stats['total_ph_time'],
        )
    )
    if stats['total_ph_time'] > 5.0:
        slow_key = max(_timing_keys()[:-1], key=lambda key: stats[key])
        print(
            '[PH-v2 Timing][WARNING] mode={} batch={} total_ph_time={:.4f}s slowest_stage={} {:.4f}s'.format(
                mode,
                batch_text,
                stats['total_ph_time'],
                slow_key,
                stats[slow_key],
            )
        )


def _timing_keys():
    return (
        'ph_extract_time',
        'support_build_time',
        'support_overlap_time',
        'entity_match_time',
        'relation_time',
        'roi_time',
        'total_ph_time',
        'birth_matching_time',
        'candidate_build_time',
        'relation_tree_query_time',
    )


def _coord_distance(coord_a, coord_b):
    return float(np.hypot(float(coord_a[0]) - float(coord_b[0]), float(coord_a[1]) - float(coord_b[1])))


def _mean(values):
    if values is None:
        return 0.0
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _median(values):
    if values is None:
        return 0.0
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


if njit is not None:
    @njit(cache=True)
    def _find_numba(parent, item):
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root


    @njit(cache=True)
    def _root_is_better_numba(root_a, root_b, birth, birth_y, birth_x):
        if birth[root_a] > birth[root_b]:
            return True
        if birth[root_a] < birth[root_b]:
            return False
        if birth_y[root_a] < birth_y[root_b]:
            return True
        if birth_y[root_a] > birth_y[root_b]:
            return False
        return birth_x[root_a] < birth_x[root_b]


    @njit(cache=True)
    def _decompose_h0_branches_numba(probability, order):
        height, width = probability.shape
        size = height * width
        flat_prob = probability.reshape(size)

        parent = np.arange(size, dtype=np.int64)
        active = np.zeros(size, dtype=np.bool_)
        birth = np.zeros(size, dtype=np.float32)
        birth_y = np.zeros(size, dtype=np.int64)
        birth_x = np.zeros(size, dtype=np.int64)
        root_branch = np.full(size, -1, dtype=np.int64)
        branch_label_flat = np.full(size, -1, dtype=np.int64)

        branch_birth = np.zeros(size, dtype=np.float32)
        branch_death = np.zeros(size, dtype=np.float32)
        branch_birth_y = np.zeros(size, dtype=np.int64)
        branch_birth_x = np.zeros(size, dtype=np.int64)
        branch_death_y = np.zeros(size, dtype=np.int64)
        branch_death_x = np.zeros(size, dtype=np.int64)
        branch_parent = np.full(size, -1, dtype=np.int64)
        branch_essential = np.ones(size, dtype=np.bool_)
        support_sizes = np.zeros(size, dtype=np.int64)
        branch_count = 0

        edge_child = np.zeros(size, dtype=np.int64)
        edge_parent = np.zeros(size, dtype=np.int64)
        edge_level = np.zeros(size, dtype=np.float32)
        edge_y = np.zeros(size, dtype=np.int64)
        edge_x = np.zeros(size, dtype=np.int64)
        edge_count = 0

        dy_values = np.array((-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int64)
        dx_values = np.array((-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int64)

        for order_index in range(size):
            flat = int(order[order_index])
            y = flat // width
            x = flat - y * width

            roots = np.empty(9, dtype=np.int64)
            num_roots = 0
            for neighbor_index in range(8):
                ny = y + int(dy_values[neighbor_index])
                nx = x + int(dx_values[neighbor_index])
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = ny * width + nx
                    if active[neighbor]:
                        root = _find_numba(parent, neighbor)
                        exists = False
                        for root_index in range(num_roots):
                            if roots[root_index] == root:
                                exists = True
                                break
                        if not exists:
                            roots[num_roots] = root
                            num_roots += 1

            value = flat_prob[flat]
            active[flat] = True
            parent[flat] = flat
            birth[flat] = value
            birth_y[flat] = y
            birth_x[flat] = x

            if num_roots == 0:
                branch_id = branch_count
                branch_count += 1
                root_branch[flat] = branch_id
                branch_label_flat[flat] = branch_id
                branch_birth[branch_id] = value
                branch_death[branch_id] = value
                branch_birth_y[branch_id] = y
                branch_birth_x[branch_id] = x
                branch_death_y[branch_id] = y
                branch_death_x[branch_id] = x
                support_sizes[branch_id] = 1
                continue

            roots[num_roots] = flat
            num_roots += 1
            survivor = roots[0]
            for root_index in range(1, num_roots):
                root = roots[root_index]
                if _root_is_better_numba(root, survivor, birth, birth_y, birth_x):
                    survivor = root

            owner = root_branch[survivor]
            branch_label_flat[flat] = owner
            if owner >= 0:
                support_sizes[owner] += 1

            for left in range(num_roots - 1):
                min_index = left
                for right in range(left + 1, num_roots):
                    if roots[right] < roots[min_index]:
                        min_index = right
                if min_index != left:
                    temp = roots[left]
                    roots[left] = roots[min_index]
                    roots[min_index] = temp

            for root_index in range(num_roots):
                root = _find_numba(parent, roots[root_index])
                survivor = _find_numba(parent, survivor)
                if root == survivor:
                    continue
                dead_root = root
                if _root_is_better_numba(root, survivor, birth, birth_y, birth_x):
                    dead_root = survivor
                    survivor = root
                child = root_branch[dead_root]
                parent_branch = root_branch[survivor]
                if child >= 0:
                    branch_death[child] = value
                    branch_death_y[child] = y
                    branch_death_x[child] = x
                    branch_parent[child] = parent_branch
                    branch_essential[child] = False
                    if parent_branch >= 0:
                        edge_child[edge_count] = child
                        edge_parent[edge_count] = parent_branch
                        edge_level[edge_count] = value
                        edge_y[edge_count] = y
                        edge_x[edge_count] = x
                        edge_count += 1
                parent[dead_root] = survivor
                if root_branch[survivor] < 0 and child >= 0:
                    root_branch[survivor] = child

        return (
            branch_count,
            branch_birth,
            branch_death,
            branch_birth_y,
            branch_birth_x,
            branch_death_y,
            branch_death_x,
            branch_parent,
            branch_essential,
            edge_count,
            edge_child,
            edge_parent,
            edge_level,
            edge_y,
            edge_x,
            branch_label_flat.reshape((height, width)),
            support_sizes,
        )
