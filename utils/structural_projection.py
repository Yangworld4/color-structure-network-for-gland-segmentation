"""Persistent-relation structural projection for unlabeled CPS.

This module deliberately keeps PH semantics, edit geometry, and CPS targets
separate: PH supplies the relation question, the receiver supplies pixel
costs/geometry, and CPS consumes only validated receiver-side edits.
"""

import heapq
import math
import time

import numpy as np
from scipy.ndimage import binary_dilation, label
import torch.nn.functional as F

from utils.ph_correspondence import (
    decompose_h0_branches,
    match_branches_by_birth,
)
from utils.ph_relation_tree import (
    build_merge_hierarchy,
    build_relation_candidates,
    query_pair_merge_fast,
)

try:
    import maxflow  # PyMaxflow, an optional compiled backend.
except ImportError:  # pragma: no cover - depends on the training environment
    maxflow = None


EPS = 1e-6
INF_CAPACITY = 1e12

STRUCTURAL_PROJECTION_FIELDS = [
    'epoch', 'branches_a', 'branches_b', 'matched_branches',
    'match_rate', 'relation_questions', 'relation_conflicts',
    'conflict_rate', 'connected_to_notconnected_cases',
    'notconnected_to_connected_cases', 'accepted_relations',
    'abstained_relations', 'accept_rate', 'abstain_rate',
    'corridor_attempts', 'corridor_successes', 'corridor_flip_rate',
    'mincut_fallbacks', 'mincut_successes', 'mincut_flip_rate',
    'completion_attempts', 'completion_successes', 'completion_flip_rate',
    'mean_disconnect_energy', 'median_disconnect_energy',
    'mean_connect_energy', 'median_connect_energy', 'mean_energy_gap',
    'mean_kappa', 'mean_cut_pixels', 'median_cut_pixels',
    'mean_completion_pixels', 'median_completion_pixels',
    'mean_graph_domain_pixels', 'median_graph_domain_pixels',
    'max_graph_domain_pixels', 'oversized_domains', 'struct_pixels_a', 'struct_pixels_b',
    'abstain_pixels_a', 'abstain_pixels_b', 'ph_time', 'matching_time',
    'tree_preprocess_time', 'relation_query_time', 'domain_build_time',
    'corridor_time', 'mincut_time', 'completion_time', 'arbitration_time',
    'projection_time', 'total_topology_time',
]


def _new_stats():
    return {
        'branches_a': 0, 'branches_b': 0, 'matched_branches': 0,
        'relation_questions': 0, 'relation_conflicts': 0,
        'connected_to_notconnected_cases': 0,
        'notconnected_to_connected_cases': 0, 'accepted_relations': 0,
        'abstained_relations': 0, 'oversized_domains': 0,
        'corridor_attempts': 0, 'corridor_successes': 0,
        'mincut_fallbacks': 0, 'mincut_successes': 0,
        'completion_attempts': 0, 'completion_successes': 0,
        'disconnect_energies': [], 'connect_energies': [],
        'energy_gaps': [], 'kappas': [], 'cut_pixels': [],
        'completion_pixels': [], 'graph_domain_pixels': [],
        'struct_pixels_a': 0, 'struct_pixels_b': 0,
        'abstain_pixels_a': 0, 'abstain_pixels_b': 0,
        'ph_time': 0.0, 'matching_time': 0.0,
        'tree_preprocess_time': 0.0, 'relation_query_time': 0.0,
        'domain_build_time': 0.0, 'corridor_time': 0.0,
        'mincut_time': 0.0, 'completion_time': 0.0,
        'arbitration_time': 0.0, 'projection_time': 0.0,
        'total_topology_time': 0.0,
    }


def new_structural_projection_epoch_state(epoch, enabled):
    state = {'epoch': int(epoch), 'enabled': int(bool(enabled))}
    for key, value in _new_stats().items():
        state[key] = [] if isinstance(value, list) else value
    return state


def _merge_stats(state, stats):
    for key, value in stats.items():
        if key not in state:
            continue
        if isinstance(value, list):
            state[key].extend(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            state[key] += value


def update_structural_projection_epoch_state(state, stats):
    _merge_stats(state, stats)


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _median(values):
    return float(np.median(values)) if values else 0.0


def finalize_structural_projection_epoch_state(state):
    questions = int(state['relation_questions'])
    conflicts = int(state['relation_conflicts'])
    decisions = int(state['accepted_relations']) + int(state['abstained_relations'])
    corridor_attempts = int(state['corridor_attempts'])
    mincut_fallbacks = int(state['mincut_fallbacks'])
    completion_attempts = int(state['completion_attempts'])
    return {
        'epoch': state['epoch'],
        'branches_a': state['branches_a'],
        'branches_b': state['branches_b'],
        'matched_branches': state['matched_branches'],
        'match_rate': _ratio(2 * state['matched_branches'], state['branches_a'] + state['branches_b']),
        'relation_questions': questions,
        'relation_conflicts': conflicts,
        'conflict_rate': _ratio(conflicts, questions),
        'connected_to_notconnected_cases': state['connected_to_notconnected_cases'],
        'notconnected_to_connected_cases': state['notconnected_to_connected_cases'],
        'accepted_relations': state['accepted_relations'],
        'abstained_relations': state['abstained_relations'],
        'accept_rate': _ratio(state['accepted_relations'], decisions),
        'abstain_rate': _ratio(state['abstained_relations'], decisions),
        'corridor_attempts': corridor_attempts,
        'corridor_successes': state['corridor_successes'],
        'corridor_flip_rate': _ratio(state['corridor_successes'], corridor_attempts),
        'mincut_fallbacks': mincut_fallbacks,
        'mincut_successes': state['mincut_successes'],
        'mincut_flip_rate': _ratio(state['mincut_successes'], mincut_fallbacks),
        'completion_attempts': completion_attempts,
        'completion_successes': state['completion_successes'],
        'completion_flip_rate': _ratio(state['completion_successes'], completion_attempts),
        'oversized_domains': state['oversized_domains'],
        'mean_disconnect_energy': _mean(state['disconnect_energies']),
        'median_disconnect_energy': _median(state['disconnect_energies']),
        'mean_connect_energy': _mean(state['connect_energies']),
        'median_connect_energy': _median(state['connect_energies']),
        'mean_energy_gap': _mean(state['energy_gaps']),
        'mean_kappa': _mean(state['kappas']),
        'mean_cut_pixels': _mean(state['cut_pixels']),
        'median_cut_pixels': _median(state['cut_pixels']),
        'mean_completion_pixels': _mean(state['completion_pixels']),
        'median_completion_pixels': _median(state['completion_pixels']),
        'mean_graph_domain_pixels': _mean(state['graph_domain_pixels']),
        'median_graph_domain_pixels': _median(state['graph_domain_pixels']),
        'max_graph_domain_pixels': max(state['graph_domain_pixels'], default=0),
        'struct_pixels_a': state['struct_pixels_a'],
        'struct_pixels_b': state['struct_pixels_b'],
        'abstain_pixels_a': state['abstain_pixels_a'],
        'abstain_pixels_b': state['abstain_pixels_b'],
        **{
            key: _ratio(state[key], 1) for key in (
                'ph_time', 'matching_time', 'tree_preprocess_time',
                'relation_query_time', 'domain_build_time', 'corridor_time',
                'mincut_time', 'completion_time', 'arbitration_time',
                'projection_time', 'total_topology_time',
            )
        },
    }


def _connectivity_structure(connectivity):
    if int(connectivity) == 4:
        return np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
    return np.ones((3, 3), dtype=np.uint8)


def _connected(mask, source, sink, connectivity):
    mask = np.asarray(mask, dtype=bool)
    source = np.asarray(source, dtype=bool) & mask
    sink = np.asarray(sink, dtype=bool) & mask
    if not source.any() or not sink.any():
        return False
    labels, _ = label(mask, structure=_connectivity_structure(connectivity))
    return bool(np.intersect1d(labels[source], labels[sink]).size > 0)


def _finite_energy(value):
    return bool(np.isfinite(value) and value >= 0.0)


def pixel_remove_cost(probability, eps=EPS):
    p = float(np.clip(probability, eps, 1.0 - eps))
    return float(math.log(p / (1.0 - p)))


def pixel_add_cost(probability, eps=EPS):
    p = float(np.clip(probability, eps, 1.0 - eps))
    return float(math.log((1.0 - p) / p))


def build_branch_core(probability, decomposition, branch_id, tau=0.5, radius=1):
    """Return a small birth-centered protected/latent terminal region."""
    probability = np.asarray(probability, dtype=np.float32)
    height, width = probability.shape
    branch = decomposition.branches[int(branch_id)]
    y, x = (int(branch.birth_coord[0]), int(branch.birth_coord[1]))
    if not (0 <= y < height and 0 <= x < width):
        return np.zeros_like(probability, dtype=bool)
    support = decomposition.branch_label_map == int(branch_id)
    point = np.zeros_like(support, dtype=bool)
    point[y, x] = True
    neighborhood = binary_dilation(point, iterations=max(0, int(radius)))
    core = neighborhood & support
    high_core = core & (probability >= float(tau))
    if high_core.any():
        core = high_core
    elif core.any():
        core = core
    else:
        core = point
    return core


def build_relation_domain(
    decomposition,
    tree,
    branch_u,
    branch_v,
    saddle_coord=None,
    margin=2,
):
    """Build a relation-local support/path domain; never uses student XOR."""
    shape = decomposition.branch_label_map.shape
    path_ids = tree.path_branch_ids(branch_u, branch_v)
    # Essential roots own the low-probability remainder of the image in the
    # decomposition. Including that support would turn a local question into
    # a full-image graph, so their birth point remains in the domain while
    # finite branches contribute their reconstructed supports.
    support_ids = [
        branch_id for branch_id in path_ids
        if not decomposition.branches[int(branch_id)].is_essential
    ]
    domain = np.isin(decomposition.branch_label_map, support_ids)
    points = [
        tuple(decomposition.branches[int(branch_u)].birth_coord),
        tuple(decomposition.branches[int(branch_v)].birth_coord),
    ]
    if saddle_coord is not None:
        points.append(tuple(saddle_coord))
    point_mask = np.zeros(shape, dtype=bool)
    for y, x in points:
        y, x = int(y), int(x)
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            point_mask[y, x] = True
    domain |= point_mask
    if not domain.any():
        return domain
    domain = binary_dilation(domain, iterations=max(0, int(margin)))
    return domain


def _relation_terminals(probability, decomposition, branch_u, branch_v, tau, pred):
    core_u = build_branch_core(probability, decomposition, branch_u, tau=tau)
    core_v = build_branch_core(probability, decomposition, branch_v, tau=tau)
    if not core_u.any() or not core_v.any() or np.logical_and(core_u, core_v).any():
        return None, None
    return core_u, core_v


def extract_ph_merge_corridor(
    probability,
    pred,
    domain,
    source,
    sink,
    saddle_coord,
    connectivity,
    max_radius,
):
    """Try a small saddle-centered foreground removal before invoking cut."""
    if saddle_coord is None or not _connected(pred, source, sink, connectivity):
        return None
    point = np.zeros_like(pred, dtype=bool)
    y, x = (int(saddle_coord[0]), int(saddle_coord[1]))
    if not (0 <= y < pred.shape[0] and 0 <= x < pred.shape[1]):
        return None
    point[y, x] = True
    max_radius = max(1, int(max_radius))
    for radius in range(max_radius + 1):
        candidate = binary_dilation(
            point,
            structure=_connectivity_structure(connectivity),
            iterations=radius,
        ) & pred & domain
        candidate &= ~source
        candidate &= ~sink
        if not candidate.any():
            continue
        edited = pred & ~candidate
        if not _connected(edited, source, sink, connectivity):
            return candidate
    return None


def _compiled_mincut_available():
    return maxflow is not None


def has_compiled_mincut_backend():
    """Whether the optional training-time local cut backend is importable."""
    return _compiled_mincut_available()


def solve_relation_cut(probability, pred, domain, source, sink, connectivity):
    """Solve a local protected-terminal vertex cut using optional PyMaxflow."""
    if maxflow is None:
        return None
    nodes = np.argwhere(np.asarray(pred, dtype=bool) & np.asarray(domain, dtype=bool))
    if nodes.size == 0 or not _connected(pred, source, sink, connectivity):
        return None
    coords = [tuple(int(v) for v in row) for row in nodes]
    node_index = {coord: index for index, coord in enumerate(coords)}
    try:
        graph = maxflow.Graph[float]()
        graph.add_nodes(2 * len(coords))
        source_ids = set(tuple(int(v) for v in row) for row in np.argwhere(source))
        sink_ids = set(tuple(int(v) for v in row) for row in np.argwhere(sink))
        structure = _connectivity_structure(connectivity)
        neighbors = [
            (dy, dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and structure[dy + 1, dx + 1]
        ]
        for index, (y, x) in enumerate(coords):
            cost = pixel_remove_cost(probability[y, x])
            if (y, x) in source_ids or (y, x) in sink_ids:
                cost = INF_CAPACITY
            graph.add_edge(2 * index, 2 * index + 1, cost, 0.0)
        for index, (y, x) in enumerate(coords):
            for dy, dx in neighbors:
                neighbor = (y + dy, x + dx)
                neighbor_index = node_index.get(neighbor)
                if neighbor_index is not None:
                    graph.add_edge(2 * index + 1, 2 * neighbor_index, INF_CAPACITY, 0.0)
        for coord in source_ids:
            index = node_index.get(coord)
            if index is not None:
                graph.add_tedge(2 * index, INF_CAPACITY, 0.0)
        for coord in sink_ids:
            index = node_index.get(coord)
            if index is not None:
                graph.add_tedge(2 * index + 1, 0.0, INF_CAPACITY)
        graph.maxflow()
        cut = np.zeros_like(pred, dtype=bool)
        for index, (y, x) in enumerate(coords):
            if graph.get_segment(2 * index) == 0 and graph.get_segment(2 * index + 1) == 1:
                cut[y, x] = True
        cut &= ~source
        cut &= ~sink
        return cut if cut.any() else None
    except (RuntimeError, ValueError, TypeError, IndexError):
        return None


def solve_relation_completion(probability, pred, domain, source, sink, connectivity):
    """Find a minimum receiver-geometry path whose added pixels connect terminals."""
    allowed = np.asarray(domain, dtype=bool)
    source = np.asarray(source, dtype=bool) & allowed
    sink = np.asarray(sink, dtype=bool) & allowed
    if not source.any() or not sink.any() or np.logical_and(source, sink).any():
        return None
    height, width = probability.shape
    structure = _connectivity_structure(connectivity)
    neighbors = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy or dx) and structure[dy + 1, dx + 1]
    ]
    distances = {}
    previous = {}
    heap = []
    for y, x in np.argwhere(source):
        node = (int(y), int(x))
        distances[node] = 0.0
        heapq.heappush(heap, (0.0, node[0], node[1]))
    target = None
    while heap:
        distance, y, x = heapq.heappop(heap)
        node = (y, x)
        if distance != distances.get(node):
            continue
        if sink[y, x]:
            target = node
            break
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and allowed[ny, nx]):
                continue
            step = 0.0 if pred[ny, nx] else pixel_add_cost(probability[ny, nx])
            if not _finite_energy(step):
                continue
            new_distance = distance + step
            neighbor = (ny, nx)
            if new_distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(heap, (new_distance, ny, nx))
    if target is None:
        return None
    path = set()
    current = target
    while current not in source:
        path.add(current)
        current = previous.get(current)
        if current is None:
            return None
    path.add(current)
    completion = np.zeros_like(pred, dtype=bool)
    for y, x in path:
        if not pred[y, x]:
            completion[y, x] = True
    return completion


def _edit_energy(probability, edit, direction):
    costs = []
    for y, x in np.argwhere(edit):
        value = pixel_remove_cost(probability[y, x]) if direction == 'remove' else pixel_add_cost(probability[y, x])
        if not _finite_energy(value):
            return float('inf')
        costs.append(value)
    return float(sum(costs))


def compute_structural_edit_energy(probability, edit, direction):
    return _edit_energy(np.asarray(probability), np.asarray(edit, dtype=bool), direction)


def arbitrate_structural_relation(disconnect_energy, connect_energy, threshold):
    if not _finite_energy(disconnect_energy) or not _finite_energy(connect_energy):
        return None, float('inf'), 0.0
    gap = abs(float(disconnect_energy) - float(connect_energy))
    kappa = gap / (float(disconnect_energy) + float(connect_energy) + EPS)
    if kappa < float(threshold):
        return None, gap, kappa
    return (
        'NOT_CONNECTED' if disconnect_energy < connect_energy else 'CONNECTED',
        gap,
        kappa,
    )


def _assign_projection(target, abstain, pixels, value):
    """Merge one edit and preserve contradictions as pixel-level abstention."""
    if pixels is None or not pixels.any():
        return 0
    contradiction = pixels & (target >= 0) & (target != int(value))
    abstain |= contradiction
    target[contradiction] = -1
    valid = pixels & ~abstain
    target[valid] = int(value)
    return int(contradiction.sum())


def project_relation_to_receiver(target, abstain, edit_pixels, relation):
    """Apply only the receiver's edit pixels; peer mask/path is never copied."""
    return _assign_projection(target, abstain, edit_pixels, 1 if relation == 'CONNECTED' else 0)


def merge_structural_edits(target, abstain, proposals):
    for edit_pixels, relation in proposals:
        project_relation_to_receiver(target, abstain, edit_pixels, relation)
    target[abstain] = -1


def _run_single_sample(
    probability_a,
    probability_b,
    tau,
    graph_margin,
    graph_max_pixels,
    arbitration_threshold,
    use_ph_corridor,
    use_mincut_fallback,
    connectivity,
):
    total_start = time.perf_counter()
    stats = _new_stats()
    shape = probability_a.shape
    pred_a = probability_a >= float(tau)
    pred_b = probability_b >= float(tau)
    target_a = np.full(shape, -1, dtype=np.int8)
    target_b = np.full(shape, -1, dtype=np.int8)
    abstain_a = np.zeros(shape, dtype=bool)
    abstain_b = np.zeros(shape, dtype=bool)
    edit_records = []

    start = time.perf_counter()
    decomposition_a = decompose_h0_branches(probability_a, 'A')
    decomposition_b = decompose_h0_branches(probability_b, 'B')
    stats['ph_time'] = time.perf_counter() - start
    stats['branches_a'] = len(decomposition_a.branches)
    stats['branches_b'] = len(decomposition_b.branches)

    start = time.perf_counter()
    matches, _unmatched_a, _unmatched_b, _match_stats = match_branches_by_birth(
        decomposition_a.branches, decomposition_b.branches, shape
    )
    stats['matching_time'] = time.perf_counter() - start
    stats['matched_branches'] = len(matches)

    start = time.perf_counter()
    tree_a = build_merge_hierarchy(decomposition_a)
    tree_b = build_merge_hierarchy(decomposition_b)
    tree_a.build_query_index()
    tree_b.build_query_index()
    stats['tree_preprocess_time'] = time.perf_counter() - start

    start = time.perf_counter()
    candidates = build_relation_candidates(
        decomposition_a.edges, decomposition_b.edges, matches
    )
    stats['relation_questions'] = len(candidates)
    relation_records = []
    for canonical_pair, edge_refs in candidates.items():
        match_u = matches[int(canonical_pair[0])]
        match_v = matches[int(canonical_pair[1])]
        mu_a, saddle_a, _edge_a = query_pair_merge_fast(
            tree_a, match_u['entity_a'].branch_id, match_v['entity_a'].branch_id
        )
        mu_b, saddle_b, _edge_b = query_pair_merge_fast(
            tree_b, match_u['entity_b'].branch_id, match_v['entity_b'].branch_id
        )
        if mu_a is None or mu_b is None:
            continue
        relation_a = bool(mu_a >= float(tau))
        relation_b = bool(mu_b >= float(tau))
        if relation_a == relation_b:
            continue
        relation_records.append({
            'canonical_pair': canonical_pair,
            'u_a': match_u['entity_a'].branch_id,
            'v_a': match_v['entity_a'].branch_id,
            'u_b': match_u['entity_b'].branch_id,
            'v_b': match_v['entity_b'].branch_id,
            'relation_a': relation_a,
            'relation_b': relation_b,
            'saddle_a': saddle_a,
            'saddle_b': saddle_b,
            'edge_refs': edge_refs,
        })
    stats['relation_query_time'] = time.perf_counter() - start
    stats['relation_conflicts'] = len(relation_records)

    proposals_a = []
    proposals_b = []
    for record in relation_records:
        if record['relation_a']:
            stats['connected_to_notconnected_cases'] += 1
        else:
            stats['notconnected_to_connected_cases'] += 1
        domain_start = time.perf_counter()
        domain_a = build_relation_domain(
            decomposition_a, tree_a, record['u_a'], record['v_a'],
            record['saddle_a'], graph_margin,
        )
        domain_b = build_relation_domain(
            decomposition_b, tree_b, record['u_b'], record['v_b'],
            record['saddle_b'], graph_margin,
        )
        stats['domain_build_time'] += time.perf_counter() - domain_start
        domain_size = max(int(domain_a.sum()), int(domain_b.sum()))
        stats['graph_domain_pixels'].append(domain_size)
        if not domain_a.any() or not domain_b.any():
            stats['abstained_relations'] += 1
            continue
        if int(domain_a.sum()) > int(graph_max_pixels) or int(domain_b.sum()) > int(graph_max_pixels):
            stats['oversized_domains'] += 1
            stats['abstained_relations'] += 1
            continue

        core_a_u, core_a_v = _relation_terminals(
            probability_a, decomposition_a, record['u_a'], record['v_a'], tau, pred_a
        )
        core_b_u, core_b_v = _relation_terminals(
            probability_b, decomposition_b, record['u_b'], record['v_b'], tau, pred_b
        )
        if core_a_u is None or core_a_v is None or core_b_u is None or core_b_v is None:
            stats['abstained_relations'] += 1
            continue

        disconnect_edit = None
        connect_edit = None
        disconnect_side = 'A' if record['relation_a'] else 'B'
        connect_side = 'B' if record['relation_a'] else 'A'
        disconnect_pred = pred_a if disconnect_side == 'A' else pred_b
        disconnect_prob = probability_a if disconnect_side == 'A' else probability_b
        disconnect_domain = domain_a if disconnect_side == 'A' else domain_b
        disconnect_source = core_a_u if disconnect_side == 'A' else core_b_u
        disconnect_sink = core_a_v if disconnect_side == 'A' else core_b_v
        disconnect_saddle = record['saddle_a'] if disconnect_side == 'A' else record['saddle_b']
        connect_pred = pred_a if connect_side == 'A' else pred_b
        connect_prob = probability_a if connect_side == 'A' else probability_b
        connect_domain = domain_a if connect_side == 'A' else domain_b
        connect_source = core_a_u if connect_side == 'A' else core_b_u
        connect_sink = core_a_v if connect_side == 'A' else core_b_v

        corridor_start = time.perf_counter()
        if use_ph_corridor:
            stats['corridor_attempts'] += 1
            disconnect_edit = extract_ph_merge_corridor(
                disconnect_prob, disconnect_pred, disconnect_domain,
                disconnect_source, disconnect_sink, disconnect_saddle,
                connectivity, max(1, int(graph_margin) * 2),
            )
            if disconnect_edit is not None:
                stats['corridor_successes'] += 1
        stats['corridor_time'] += time.perf_counter() - corridor_start

        disconnect_solver = 'ph_corridor' if disconnect_edit is not None else 'failed'
        if disconnect_edit is None and use_mincut_fallback and _compiled_mincut_available():
            stats['mincut_fallbacks'] += 1
            mincut_start = time.perf_counter()
            disconnect_edit = solve_relation_cut(
                disconnect_prob, disconnect_pred, disconnect_domain,
                disconnect_source, disconnect_sink, connectivity,
            )
            stats['mincut_time'] += time.perf_counter() - mincut_start
            if disconnect_edit is not None:
                if _connected(disconnect_pred & ~disconnect_edit, disconnect_source, disconnect_sink, connectivity):
                    disconnect_edit = None
                else:
                    stats['mincut_successes'] += 1
                    disconnect_solver = 'local_mincut'
        elif disconnect_edit is None:
            disconnect_solver = 'failed'

        completion_start = time.perf_counter()
        stats['completion_attempts'] += 1
        connect_edit = solve_relation_completion(
            connect_prob, connect_pred, connect_domain,
            connect_source, connect_sink, connectivity,
        )
        stats['completion_time'] += time.perf_counter() - completion_start
        if connect_edit is not None:
            if _connected(connect_pred | connect_edit, connect_source, connect_sink, connectivity):
                stats['completion_successes'] += 1
            else:
                connect_edit = None

        disconnect_energy = (
            compute_structural_edit_energy(disconnect_prob, disconnect_edit, 'remove')
            if disconnect_edit is not None else float('inf')
        )
        connect_energy = (
            compute_structural_edit_energy(connect_prob, connect_edit, 'add')
            if connect_edit is not None else float('inf')
        )
        if _finite_energy(disconnect_energy):
            stats['disconnect_energies'].append(disconnect_energy)
            stats['cut_pixels'].append(int(disconnect_edit.sum()))
        if _finite_energy(connect_energy):
            stats['connect_energies'].append(connect_energy)
            stats['completion_pixels'].append(int(connect_edit.sum()))

        arbitration_start = time.perf_counter()
        selected_relation, energy_gap, kappa = arbitrate_structural_relation(
            disconnect_energy, connect_energy, arbitration_threshold
        )
        stats['arbitration_time'] += time.perf_counter() - arbitration_start
        if selected_relation is None:
            stats['abstained_relations'] += 1
            continue
        stats['energy_gaps'].append(energy_gap)
        stats['kappas'].append(kappa)
        stats['accepted_relations'] += 1
        # The relation selected above is the peer relation. The edit is always
        # computed on the receiver, so its own edit must be used for projection.
        if selected_relation == 'NOT_CONNECTED':
            receiver = disconnect_side
            receiver_edit = disconnect_edit if receiver == disconnect_side else None
        else:
            receiver = connect_side
            receiver_edit = connect_edit if receiver == connect_side else None
        if receiver_edit is None:
            # This is defensive: arbitration cannot project a missing receiver edit.
            stats['accepted_relations'] -= 1
            stats['abstained_relations'] += 1
            continue
        if receiver == 'A':
            proposals_a.append((receiver_edit, selected_relation))
        else:
            proposals_b.append((receiver_edit, selected_relation))
        edit_records.append({
            'canonical_pair': tuple(int(value) for value in record['canonical_pair']),
            'selected_relation': selected_relation,
            'receiver': receiver,
            'disconnect_solver_type': disconnect_solver,
            'disconnect_energy': float(disconnect_energy),
            'connect_energy': float(connect_energy),
            'energy_gap': float(energy_gap),
            'kappa': float(kappa),
            'edit_pixels': receiver_edit.copy(),
        })

    projection_start = time.perf_counter()
    merge_structural_edits(target_a, abstain_a, proposals_a)
    merge_structural_edits(target_b, abstain_b, proposals_b)
    target_a[abstain_a] = -1
    target_b[abstain_b] = -1
    stats['projection_time'] = time.perf_counter() - projection_start
    stats['struct_pixels_a'] = int((target_a >= 0).sum())
    stats['struct_pixels_b'] = int((target_b >= 0).sum())
    stats['abstain_pixels_a'] = int(abstain_a.sum())
    stats['abstain_pixels_b'] = int(abstain_b.sum())
    stats['total_topology_time'] = time.perf_counter() - total_start
    return target_a, target_b, abstain_a, abstain_b, stats, edit_records


def compute_structural_projection(
    prob_a,
    prob_b,
    tau=0.5,
    graph_margin=2,
    graph_max_pixels=4096,
    arbitration_threshold=0.2,
    use_ph_corridor=True,
    use_mincut_fallback=True,
    connectivity=8,
    matching='birth_nearest',
):
    """Run the complete CPU funnel and return one target/mask set per batch."""
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    if prob_a.shape != prob_b.shape or prob_a.ndim != 3:
        raise ValueError('prob_a and prob_b must have matching [N,H,W] shapes')
    if matching != 'birth_nearest':
        raise ValueError('Only birth_nearest matching is supported for structural projection')
    batch, height, width = prob_a.shape
    targets_a = np.full((batch, height, width), -1, dtype=np.int8)
    targets_b = np.full((batch, height, width), -1, dtype=np.int8)
    abstains_a = np.zeros((batch, height, width), dtype=bool)
    abstains_b = np.zeros((batch, height, width), dtype=bool)
    stats = _new_stats()
    edit_records = []
    for index in range(batch):
        result = _run_single_sample(
            prob_a[index], prob_b[index], tau, graph_margin, graph_max_pixels,
            arbitration_threshold, use_ph_corridor, use_mincut_fallback,
            connectivity,
        )
        (
            sample_targets_a, sample_targets_b, sample_abstain_a,
            sample_abstain_b, sample_stats, sample_edit_records,
        ) = result
        targets_a[index] = sample_targets_a
        targets_b[index] = sample_targets_b
        abstains_a[index] = sample_abstain_a
        abstains_b[index] = sample_abstain_b
        _merge_stats(stats, sample_stats)
        edit_records.extend(
            {'sample_index': index, **record}
            for record in sample_edit_records
        )
    return {
        'target_a': targets_a,
        'target_b': targets_b,
        'struct_mask_a': targets_a >= 0,
        'struct_mask_b': targets_b >= 0,
        'abstain_mask_a': abstains_a,
        'abstain_mask_b': abstains_b,
        'relation_edits': edit_records,
        'stats': stats,
    }


def compute_structural_projection_cps_losses(prediction_a, prediction_b, projection):
    """Build bidirectional CPS losses with structural pixels replacing vanilla PL."""
    pseudo_a = prediction_a.detach().argmax(dim=1)
    pseudo_b = prediction_b.detach().argmax(dim=1)
    packed = np.stack((
        projection['target_a'],
        projection['target_b'],
        projection['abstain_mask_a'].astype(np.int8),
        projection['abstain_mask_b'].astype(np.int8),
    ), axis=1)
    packed = torch_from_numpy(packed, prediction_a.device)
    raw_target_a = packed[:, 0]
    raw_target_b = packed[:, 1]
    target_a = raw_target_a.clamp_min(0).long()
    target_b = raw_target_b.clamp_min(0).long()
    abstain_a = packed[:, 2].bool()
    abstain_b = packed[:, 3].bool()
    struct_mask_a = raw_target_a >= 0
    struct_mask_b = raw_target_b >= 0
    safe_a = ~(struct_mask_a | abstain_a)
    safe_b = ~(struct_mask_b | abstain_b)
    loss_a_map = F.cross_entropy(prediction_a, pseudo_b, reduction='none')
    loss_b_map = F.cross_entropy(prediction_b, pseudo_a, reduction='none')
    structural_a_map = F.cross_entropy(prediction_a, target_a, reduction='none')
    structural_b_map = F.cross_entropy(prediction_b, target_b, reduction='none')
    loss_a = 0.5 * (
        (loss_a_map * safe_a.float()).sum() / safe_a.float().sum().clamp_min(1.0)
        + (structural_a_map * struct_mask_a.float()).sum() / struct_mask_a.float().sum().clamp_min(1.0)
    )
    loss_b = 0.5 * (
        (loss_b_map * safe_b.float()).sum() / safe_b.float().sum().clamp_min(1.0)
        + (structural_b_map * struct_mask_b.float()).sum() / struct_mask_b.float().sum().clamp_min(1.0)
    )
    return {
        '1_to_2': loss_b,
        '2_to_1': loss_a,
        'total': loss_a + loss_b,
        'struct_mask_a': struct_mask_a,
        'struct_mask_b': struct_mask_b,
        'abstain_mask_a': abstain_a,
        'abstain_mask_b': abstain_b,
    }


def torch_from_numpy(array, device):
    # Local import keeps this module's CPU PH/edit path import-light.
    import torch
    return torch.from_numpy(np.asarray(array)).to(device=device, non_blocking=True)
