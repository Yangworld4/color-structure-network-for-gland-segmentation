"""Connected-component graph structural conflict analysis for V1 CPS.

This module deliberately has no dependency on the persistent-homology code.
All structural decisions are made from thresholded foreground masks and the
two students' foreground probabilities.
"""

import csv
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra, maximum_flow


CC_GRAPH_CAPACITY_SCALE = 10000


CC_GRAPH_FIELDS = [
    'epoch',
    'samples',
    'components_a',
    'components_b',
    'graph_edges',
    'one_to_one_groups',
    'eligible_pair_conflicts',
    'skipped_multiterminal_events',
    'unmatched_a',
    'unmatched_b',
    'merge_conflicts',
    'split_conflicts',
    'complex_conflicts',
    'total_structural_conflicts',
    'valid_cut_events',
    'cut_failure_events',
    'cut_area_skip_events',
    'cut_bbox_skip_events',
    'cut_attempt_events',
    'cut_success_events',
    'cut_flip_success_events',
    'cut_flip_success_rate',
    'valid_join_events',
    'join_failure_events',
    'join_flip_success_events',
    'join_flip_success_rate',
    'dual_agree_events',
    'dual_agree_rate',
    'mean_cut_cost',
    'median_cut_cost',
    'mean_join_cost',
    'median_join_cost',
    'mean_cut_pixels',
    'median_cut_pixels',
    'mean_join_pixels',
    'median_join_pixels',
    'mean_cut_join_iou',
    'mean_cut_join_min_distance',
    'structural_roi_pixels',
    'structural_roi_pixel_ratio',
    'masked_cps_pixels',
    'masked_cps_pixel_ratio',
    'mean_connected_component_area',
    'mean_separated_component_area',
    'runtime_cc',
    'runtime_correspondence',
    'runtime_cut',
    'runtime_cut_build',
    'runtime_cut_solve',
    'runtime_cut_validate',
    'runtime_join',
    'runtime_total',
    'cut_node_count_mean',
    'cut_node_count_max',
    'cut_edge_count_mean',
    'cut_edge_count_max',
    'cut_local_bbox_pixels_mean',
    'cut_local_bbox_pixels_max',
    'gt_valid_conflicts',
    'connected_side_correct_events',
    'separated_side_correct_events',
    'both_wrong_structure_events',
    'connected_side_correct_rate',
    'separated_side_correct_rate',
    'cut_roi_gt_error_rate',
    'join_roi_gt_error_rate',
    'cut_flip_and_connected_wrong_rate',
    'join_flip_and_separated_wrong_rate',
]


@dataclass
class Component:
    id: int
    mask: Optional[np.ndarray]
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[slice, slice]


@dataclass
class StructuralConflict:
    sample_idx: int
    conflict_type: str
    connected_student: str
    connected_component_id: int
    separated_component_ids: Tuple[int, ...]
    connected_component: Component
    separated_components: Tuple[Component, ...]
    edge_pairs: Tuple[Tuple[int, int], ...]


def _empty_float_list() -> List[float]:
    return []


def new_cc_graph_epoch_state(epoch: int, enabled: bool = True) -> Dict:
    """Create an epoch accumulator used by both train and validation."""
    return {
        'epoch': int(epoch),
        'enabled': bool(enabled),
        'samples': 0,
        'components_a': 0,
        'components_b': 0,
        'graph_edges': 0,
        'one_to_one_groups': 0,
        'eligible_pair_conflicts': 0,
        'skipped_multiterminal_events': 0,
        'unmatched_a': 0,
        'unmatched_b': 0,
        'merge_conflicts': 0,
        'split_conflicts': 0,
        'complex_conflicts': 0,
        'total_structural_conflicts': 0,
        'valid_cut_events': 0,
        'cut_failure_events': 0,
        'cut_area_skip_events': 0,
        'cut_bbox_skip_events': 0,
        'cut_attempt_events': 0,
        'cut_success_events': 0,
        'cut_flip_success_events': 0,
        'valid_join_events': 0,
        'join_failure_events': 0,
        'join_flip_success_events': 0,
        'dual_agree_events': 0,
        'cut_costs': _empty_float_list(),
        'join_costs': _empty_float_list(),
        'cut_pixels': [],
        'join_pixels': [],
        'cut_join_ious': [],
        'cut_join_distances': [],
        'connected_areas': [],
        'separated_areas': [],
        'structural_roi_pixels': 0,
        'masked_cps_pixels': 0,
        'total_pixels': 0,
        'runtime_cc': 0.0,
        'runtime_correspondence': 0.0,
        'runtime_cut': 0.0,
        'runtime_cut_build': 0.0,
        'runtime_cut_solve': 0.0,
        'runtime_cut_validate': 0.0,
        'runtime_join': 0.0,
        'runtime_total': 0.0,
        'cut_node_counts': [],
        'cut_edge_counts': [],
        'cut_local_bbox_pixels': [],
        'gt_valid_conflicts': 0,
        'connected_side_correct_events': 0,
        'separated_side_correct_events': 0,
        'both_wrong_structure_events': 0,
        'cut_roi_gt_error_pixels': 0,
        'cut_roi_gt_pixels': 0,
        'join_roi_gt_error_pixels': 0,
        'join_roi_gt_pixels': 0,
        'cut_flip_and_connected_wrong_events': 0,
        'join_flip_and_separated_wrong_events': 0,
        'visualizations': [],
    }


def _as_batch_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError('CC-Graph probabilities must have shape [B,H,W].')
    return value.astype(np.float32, copy=False)


def _component_decomposition(prediction: np.ndarray) -> Tuple[np.ndarray, List[Component]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(prediction.astype(bool), structure=structure)
    count = int(count)
    if count == 0:
        return labels, []
    flat_labels = labels.ravel().astype(np.int64, copy=False)
    ids = np.arange(1, count + 1, dtype=np.int64)
    areas = np.bincount(flat_labels, minlength=count + 1)[1:]
    height, width = labels.shape
    yy, xx = np.indices(labels.shape)
    sum_y = np.bincount(flat_labels, weights=yy.ravel(), minlength=count + 1)[1:]
    sum_x = np.bincount(flat_labels, weights=xx.ravel(), minlength=count + 1)[1:]
    objects = ndimage.find_objects(labels, max_label=count)
    components = []
    for component_id, area, sum_y_value, sum_x_value, component_slice in zip(
        ids, areas, sum_y, sum_x, objects
    ):
        if area <= 0 or component_slice is None:
            continue
        components.append(Component(
            id=int(component_id),
            # Materialized only for conflict endpoints; this avoids creating
            # one full-resolution mask per tiny unmatched component.
            mask=None,
            area=int(area),
            centroid=(float(sum_y_value / area), float(sum_x_value / area)),
            bbox=component_slice,
        ))
    return labels, components


def _materialize_components(components: Sequence[Component], labels: np.ndarray,
                            component_ids: Iterable[int]):
    component_by_id = {component.id: component for component in components}
    for component_id in component_ids:
        component = component_by_id.get(int(component_id))
        if component is not None and component.mask is None:
            component.mask = labels == component.id


def _intersection_matrix(labels_a: np.ndarray, labels_b: np.ndarray,
                         count_a: int, count_b: int) -> np.ndarray:
    base = count_b + 1
    flat = labels_a.astype(np.int64).ravel() * base + labels_b.astype(np.int64).ravel()
    matrix = np.bincount(flat, minlength=(count_a + 1) * base)
    return matrix.reshape(count_a + 1, base)[1:, 1:]


def build_cc_graph_conflicts(
    prob_a,
    prob_b,
    tau: float = 0.5,
    min_coverage: float = 0.5,
    min_intersection: int = 10,
    sample_offset: int = 0,
) -> Dict:
    """Build 8-connected CCs, a vectorized bipartite graph, and conflict events."""
    prob_a = _as_batch_numpy(prob_a)
    prob_b = _as_batch_numpy(prob_b)
    if prob_a.shape != prob_b.shape:
        raise ValueError('CC-Graph probability shapes must match.')

    pred_a = prob_a >= float(tau)
    pred_b = prob_b >= float(tau)
    samples = []
    cc_runtime = 0.0
    correspondence_runtime = 0.0
    for batch_index in range(prob_a.shape[0]):
        cc_start = time.perf_counter()
        labels_a, components_a = _component_decomposition(pred_a[batch_index])
        labels_b, components_b = _component_decomposition(pred_b[batch_index])
        cc_runtime += time.perf_counter() - cc_start
        correspondence_start = time.perf_counter()
        intersections = _intersection_matrix(
            labels_a, labels_b, len(components_a), len(components_b)
        )
        if intersections.size:
            areas_a = np.asarray([component.area for component in components_a])[:, None]
            areas_b = np.asarray([component.area for component in components_b])[None, :]
            coverage_a = intersections / np.maximum(areas_a, 1)
            coverage_b = intersections / np.maximum(areas_b, 1)
            edge_mask = (
                (intersections >= int(min_intersection))
                & (np.maximum(coverage_a, coverage_b) >= float(min_coverage))
            )
        else:
            edge_mask = np.zeros((len(components_a), len(components_b)), dtype=bool)

        graph = nx.Graph()
        graph.add_nodes_from(('a', component.id) for component in components_a)
        graph.add_nodes_from(('b', component.id) for component in components_b)
        edge_pairs = []
        for component_a_index, component_b_index in np.argwhere(edge_mask):
            component_a = components_a[int(component_a_index)]
            component_b = components_b[int(component_b_index)]
            graph.add_edge(('a', component_a.id), ('b', component_b.id))
            edge_pairs.append((component_a.id, component_b.id))

        conflicts = []
        one_to_one_groups = 0
        for graph_nodes in nx.connected_components(graph):
            nodes_a = sorted(node[1] for node in graph_nodes if node[0] == 'a')
            nodes_b = sorted(node[1] for node in graph_nodes if node[0] == 'b')
            if not nodes_a or not nodes_b:
                continue
            group_edges = tuple(
                pair for pair in edge_pairs
                if pair[0] in nodes_a and pair[1] in nodes_b
            )
            if len(nodes_a) == 1 and len(nodes_b) == 1:
                one_to_one_groups += 1
                continue
            # Keep masks lazy for unmatched/single-match components. A
            # localization event is the only path that needs pixel masks.
            _materialize_components(components_a, labels_a, nodes_a)
            _materialize_components(components_b, labels_b, nodes_b)
            if len(nodes_b) == 1 and len(nodes_a) >= 2:
                conflicts.append(StructuralConflict(
                    sample_idx=sample_offset + batch_index,
                    conflict_type='b_connected_a_separated',
                    connected_student='b',
                    connected_component_id=nodes_b[0],
                    separated_component_ids=tuple(nodes_a),
                    connected_component=components_b[nodes_b[0] - 1],
                    separated_components=tuple(components_a[index - 1] for index in nodes_a),
                    edge_pairs=group_edges,
                ))
            elif len(nodes_a) == 1 and len(nodes_b) >= 2:
                conflicts.append(StructuralConflict(
                    sample_idx=sample_offset + batch_index,
                    conflict_type='a_connected_b_separated',
                    connected_student='a',
                    connected_component_id=nodes_a[0],
                    separated_component_ids=tuple(nodes_b),
                    connected_component=components_a[nodes_a[0] - 1],
                    separated_components=tuple(components_b[index - 1] for index in nodes_b),
                    edge_pairs=group_edges,
                ))
            else:
                conflicts.append(StructuralConflict(
                    sample_idx=sample_offset + batch_index,
                    conflict_type='complex_conflict',
                    connected_student='none',
                    connected_component_id=-1,
                    separated_component_ids=tuple(nodes_a + nodes_b),
                    connected_component=components_a[nodes_a[0] - 1],
                    separated_components=tuple(
                        components_a[index - 1] for index in nodes_a
                    ) + tuple(components_b[index - 1] for index in nodes_b),
                    edge_pairs=group_edges,
                ))

        connected_a = {pair[0] for pair in edge_pairs}
        connected_b = {pair[1] for pair in edge_pairs}
        unmatched_a = [component.id for component in components_a if component.id not in connected_a]
        unmatched_b = [component.id for component in components_b if component.id not in connected_b]
        correspondence_runtime += time.perf_counter() - correspondence_start
        samples.append({
            'sample_idx': sample_offset + batch_index,
            'prob_a': prob_a[batch_index],
            'prob_b': prob_b[batch_index],
            'pred_a': pred_a[batch_index],
            'pred_b': pred_b[batch_index],
            'labels_a': labels_a,
            'labels_b': labels_b,
            'components_a': components_a,
            'components_b': components_b,
            'edge_pairs': tuple(edge_pairs),
            'one_to_one_groups': one_to_one_groups,
            'unmatched_a': unmatched_a,
            'unmatched_b': unmatched_b,
            'conflicts': conflicts,
        })
    return {
        'samples': samples,
        'conflicts': [
            conflict for sample in samples for conflict in sample['conflicts']
        ],
        'diagnostics': {
            'samples': len(samples),
            'components_a': sum(len(sample['components_a']) for sample in samples),
            'components_b': sum(len(sample['components_b']) for sample in samples),
            'graph_edges': sum(len(sample['edge_pairs']) for sample in samples),
        },
        'runtime_cc': cc_runtime,
        'runtime_correspondence': correspondence_runtime,
    }


def _expanded_bbox(masks: Iterable[np.ndarray], shape: Tuple[int, int], padding: int = 0):
    coords = np.argwhere(np.logical_or.reduce(list(masks)))
    if coords.size == 0:
        return slice(0, 0), slice(0, 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return (
        slice(max(0, int(y0) - padding), min(shape[0], int(y1) + padding)),
        slice(max(0, int(x0) - padding), min(shape[1], int(x1) + padding)),
    )


def _terminal_cores(connected: Component, separated: Sequence[Component]) -> List[np.ndarray]:
    cores = []
    structure = np.ones((3, 3), dtype=bool)
    for component in separated:
        terminal = connected.mask & component.mask
        core = ndimage.binary_erosion(terminal, structure=structure, iterations=2)
        if not core.any():
            core = terminal
        if core.any():
            cores.append(core)
    return cores


def _minimum_vertex_cut(
    foreground: np.ndarray,
    probability: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    tau: float,
    protected_mask: Optional[np.ndarray] = None,
    padding: int = 16,
) -> Tuple[np.ndarray, float, Dict]:
    """Solve a weighted 8-neighbour vertex cut with a sparse max-flow graph.

    The graph uses node splitting, but all split and 8-neighbour edges are
    assembled as NumPy arrays before construction of the CSR matrix. The
    returned graph statistics are used only for epoch-level diagnostics.
    """
    terminal_union = np.logical_or(source_mask, target_mask)
    terminal_coords = np.argwhere(terminal_union)
    if terminal_coords.size == 0:
        raise ValueError('empty terminal')
    y0, x0 = terminal_coords.min(axis=0)
    y1, x1 = terminal_coords.max(axis=0) + 1
    y0 = max(0, int(y0) - int(padding))
    x0 = max(0, int(x0) - int(padding))
    y1 = min(foreground.shape[0], int(y1) + int(padding))
    x1 = min(foreground.shape[1], int(x1) + int(padding))
    local_bbox_pixels = int((y1 - y0) * (x1 - x0))

    local_fg = foreground[y0:y1, x0:x1].astype(bool, copy=False)
    local_prob = probability[y0:y1, x0:x1]
    local_source = source_mask[y0:y1, x0:x1] & local_fg
    local_target = target_mask[y0:y1, x0:x1] & local_fg
    local_protected = (
        np.zeros_like(local_fg, dtype=bool) if protected_mask is None
        else protected_mask[y0:y1, x0:x1] & local_fg
    )
    if not local_fg.any() or not local_source.any() or not local_target.any():
        raise ValueError('empty foreground or terminal')

    pixel_index = -np.ones(local_fg.shape, dtype=np.int64)
    pixel_index[local_fg] = np.arange(int(local_fg.sum()), dtype=np.int64)
    node_count = int(local_fg.sum())
    pixel_ids = np.arange(node_count, dtype=np.int64)
    node_in = 2 * pixel_ids
    node_out = node_in + 1
    source = 2 * node_count
    sink = source + 1

    costs = np.maximum(local_prob[local_fg] - float(tau), 1e-4)
    cost_int = np.maximum(
        1, np.rint(costs * CC_GRAPH_CAPACITY_SCALE)
    ).astype(np.int64)
    finite_capacity_sum = int(cost_int.sum(dtype=np.int64))
    if finite_capacity_sum >= np.iinfo(np.int64).max - 1:
        raise OverflowError('pixel cut capacity sum exceeds int64')
    inf = np.int64(finite_capacity_sum + 1)
    terminal = local_source | local_target | local_protected

    build_start = time.perf_counter()
    # Node-splitting and terminal edges are built in bulk. The four offsets
    # cover each undirected 8-neighbour pair exactly once, then both directed
    # arcs are emitted for the max-flow graph.
    rows_parts = [node_in]
    cols_parts = [node_out]
    data_parts = [np.where(terminal[local_fg], inf, cost_int)]
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        source_y = slice(max(0, -dy), min(local_fg.shape[0], local_fg.shape[0] - dy))
        source_x = slice(max(0, -dx), min(local_fg.shape[1], local_fg.shape[1] - dx))
        target_y = slice(max(0, dy), min(local_fg.shape[0], local_fg.shape[0] + dy))
        target_x = slice(max(0, dx), min(local_fg.shape[1], local_fg.shape[1] + dx))
        pair_mask = local_fg[source_y, source_x] & local_fg[target_y, target_x]
        source_ids = pixel_index[source_y, source_x][pair_mask]
        target_ids = pixel_index[target_y, target_x][pair_mask]
        rows_parts.extend((2 * source_ids + 1, 2 * target_ids + 1))
        cols_parts.extend((2 * target_ids, 2 * source_ids))
        data_parts.extend((
            np.full(source_ids.shape, inf, dtype=np.int64),
            np.full(source_ids.shape, inf, dtype=np.int64),
        ))

    source_ids = pixel_index[local_source]
    target_ids = pixel_index[local_target]
    rows_parts.extend((np.full(source_ids.shape, source, dtype=np.int64),
                       2 * target_ids + 1))
    cols_parts.extend((2 * source_ids, np.full(target_ids.shape, sink, dtype=np.int64)))
    data_parts.extend((
        np.full(source_ids.shape, inf, dtype=np.int64),
        np.full(target_ids.shape, inf, dtype=np.int64),
    ))

    graph = csr_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=(sink + 1, sink + 1),
        dtype=np.int64,
    )
    graph.sum_duplicates()
    graph.eliminate_zeros()
    build_runtime = time.perf_counter() - build_start
    edge_count = int(graph.nnz)

    solve_start = time.perf_counter()
    flow_result = maximum_flow(graph, source, sink)
    solve_runtime = time.perf_counter() - solve_start

    # SciPy stores signed flow in residual. Subtracting it from the original
    # capacity yields positive residual capacities in both directions.
    residual_capacity = (graph - flow_result.residual).tocsr()
    residual_capacity.data = (residual_capacity.data > 0).astype(np.int8)
    residual_capacity.eliminate_zeros()
    reachable = breadth_first_order(
        residual_capacity,
        source,
        directed=True,
        return_predecessors=False,
    )
    reachable_nodes = np.zeros(sink + 1, dtype=bool)
    reachable_nodes[reachable] = True
    cut_ids = pixel_ids[
        reachable_nodes[node_in] & ~reachable_nodes[node_out]
    ]
    if cut_ids.size and terminal[local_fg][cut_ids].any():
        raise ValueError('minimum cut crossed a protected terminal')
    if cut_ids.size == 0:
        raise ValueError('minimum cut returned an empty cut')

    cut_local = np.zeros(local_fg.shape, dtype=bool)
    local_foreground_ids = np.flatnonzero(local_fg)
    cut_local.flat[local_foreground_ids[cut_ids]] = True
    cut_mask = np.zeros(foreground.shape, dtype=bool)
    cut_mask[y0:y1, x0:x1] = cut_local
    return cut_mask, float(costs[cut_ids].sum()), {
        'build_runtime': build_runtime,
        'solve_runtime': solve_runtime,
        'node_count': node_count,
        'edge_count': edge_count,
        'local_bbox_pixels': local_bbox_pixels,
    }


def _minimum_completion_path(
    probability: np.ndarray,
    foreground: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    tau: float,
    bbox_masks: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, float]:
    """Find a minimum node-cost 8-neighbour path between terminal groups."""
    bbox = _expanded_bbox(
        bbox_masks if bbox_masks is not None else (source_mask, target_mask),
        probability.shape,
    )
    y_slice, x_slice = bbox
    if y_slice.stop <= y_slice.start or x_slice.stop <= x_slice.start:
        raise ValueError('empty completion-path bounding box')
    local_prob = probability[bbox]
    local_fg = foreground[bbox]
    local_source = source_mask[bbox]
    local_target = target_mask[bbox]
    height, width = local_prob.shape
    count = height * width
    node_cost = np.where(local_fg, 0.0, np.maximum(float(tau) - local_prob, 0.0))
    source_nodes = np.flatnonzero(local_source.ravel())
    target_nodes = np.flatnonzero(local_target.ravel())
    if source_nodes.size == 0 or target_nodes.size == 0:
        raise ValueError('empty completion terminal')
    rows = []
    cols = []
    data = []
    for y in range(height):
        for x in range(width):
            node = y * width + x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx_ < width:
                        rows.append(node)
                        cols.append(ny * width + nx_)
                        data.append(float(node_cost[ny, nx_]))
    source_super = count
    # Add every source terminal to the same super-source so the path is
    # minimized over the complete terminal core, not an arbitrary pixel.
    rows.extend([source_super] * int(source_nodes.size))
    cols.extend(int(node) for node in source_nodes)
    data.extend([0.0] * int(source_nodes.size))
    graph = csr_matrix((data, (rows, cols)), shape=(count + 1, count + 1))
    distances, predecessors = dijkstra(
        graph,
        directed=True,
        indices=source_super,
        return_predecessors=True,
    )
    target_node = int(target_nodes[np.argmin(distances[target_nodes])])
    if not np.isfinite(distances[target_node]):
        raise ValueError('completion path is unreachable')
    source_node = source_super
    path_nodes = []
    node = target_node
    while node != source_node and node >= 0:
        path_nodes.append(node)
        node = int(predecessors[node])
    if node != source_node:
        raise ValueError('completion path predecessor chain failed')
    # Do not include the virtual source in the returned pixel mask.
    path = np.zeros((height, width), dtype=bool)
    for path_node in path_nodes:
        path[path_node // width, path_node % width] = True
    path_mask = np.zeros(probability.shape, dtype=bool)
    path_mask[y_slice, x_slice] = path & (~foreground[bbox])
    return path_mask, float(distances[target_node])


def _groups_connected(mask: np.ndarray, groups: Sequence[np.ndarray]) -> bool:
    labels, _count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    group_labels = []
    for group in groups:
        values = np.unique(labels[group])
        values = values[values > 0]
        if values.size != 1:
            return False
        group_labels.append(int(values[0]))
    return bool(group_labels) and len(set(group_labels)) == 1


def _groups_all_separated(mask: np.ndarray, groups: Sequence[np.ndarray]) -> bool:
    labels, _count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    group_labels = []
    for group in groups:
        values = np.unique(labels[group])
        values = values[values > 0]
        if values.size != 1:
            return False
        group_labels.append(int(values[0]))
    return len(group_labels) >= 2 and len(set(group_labels)) == len(group_labels)


def _mask_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if not mask_a.any() or not mask_b.any():
        return float('inf')
    return float(distance_transform_edt(~mask_a)[mask_b].min())


def _empty_localization_result(shape, compute_cut=True, compute_join=True, reason=''):
    skipped = reason in ('multiterminal', 'complex', 'diagnostics_only')
    return {
        'cut_mask': np.zeros(shape, dtype=bool),
        'join_mask': np.zeros(shape, dtype=bool),
        'cut_cost': float('nan'),
        'join_cost': float('nan'),
        'cut_flip_success': False,
        'join_flip_success': False,
        'localization_valid': False,
        'join_localization_valid': False,
        'dual_agreement': False,
        'cut_join_iou': float('nan'),
        'cut_join_min_distance': float('inf'),
        'cut_failure': bool(reason) and not skipped if compute_cut else False,
        'join_failure': bool(reason) and not skipped if compute_join else False,
        'cut_runtime': 0.0,
        'cut_build_runtime': 0.0,
        'cut_solve_runtime': 0.0,
        'cut_validate_runtime': 0.0,
        'cut_node_count': 0,
        'cut_edge_count': 0,
        'cut_local_bbox_pixels': 0,
        'join_runtime': 0.0,
        'compute_cut': bool(compute_cut),
        'cut_attempt': False,
        'cut_success': False,
        'compute_join': bool(compute_join),
        'skipped_multiterminal': reason in ('multiterminal', 'complex'),
        'cut_area_skip': reason == 'area_guard',
        'cut_bbox_skip': reason == 'bbox_guard',
        'skip_reason': reason,
    }


def localize_cc_graph_conflict(
    conflict: StructuralConflict,
    prob_a,
    prob_b,
    tau: float = 0.5,
    agreement_radius: float = 3.0,
    compute_cut: bool = True,
    compute_join: bool = True,
    max_terminal_count: Optional[int] = None,
    max_cut_area: Optional[int] = None,
    max_cut_bbox_pixels: Optional[int] = None,
    cut_padding: int = 16,
) -> Dict:
    """Localize one merge/split event with minimum cut and completion paths."""
    probability_a = np.asarray(prob_a, dtype=np.float32)
    probability_b = np.asarray(prob_b, dtype=np.float32)
    if conflict.conflict_type == 'complex_conflict':
        return _empty_localization_result(
            probability_a.shape,
            compute_cut=compute_cut,
            compute_join=compute_join,
            reason='complex',
        )

    if conflict.connected_student == 'a':
        connected_probability = probability_a
        separated_probability = probability_b
        separated_foreground = probability_b >= tau
    else:
        connected_probability = probability_b
        separated_probability = probability_a
        separated_foreground = probability_a >= tau
    connected_component = conflict.connected_component
    separated_components = conflict.separated_components
    terminal_cores = _terminal_cores(connected_component, separated_components)
    empty = np.zeros(probability_a.shape, dtype=bool)
    if len(terminal_cores) < 2:
        return _empty_localization_result(
            probability_a.shape,
            compute_cut=compute_cut,
            compute_join=compute_join,
            reason='terminals',
        )
    if max_terminal_count is not None and len(terminal_cores) != max_terminal_count:
        return _empty_localization_result(
            probability_a.shape,
            compute_cut=compute_cut,
            compute_join=compute_join,
            reason='multiterminal',
        )

    cut_mask = empty.copy()
    join_mask = empty.copy()
    cut_failure = False
    join_failure = False
    cut_runtime = 0.0
    cut_build_runtime = 0.0
    cut_solve_runtime = 0.0
    cut_validate_runtime = 0.0
    cut_node_count = 0
    cut_edge_count = 0
    cut_local_bbox_pixels = 0
    join_runtime = 0.0
    cut_attempt = False
    cut_area_skip = False
    cut_bbox_skip = False

    if compute_cut:
        cut_bbox = _expanded_bbox(
            (terminal_cores[0], terminal_cores[1]),
            probability_a.shape,
            padding=cut_padding,
        )
        bbox_height = cut_bbox[0].stop - cut_bbox[0].start
        bbox_width = cut_bbox[1].stop - cut_bbox[1].start
        if max_cut_area is not None and connected_component.area > max_cut_area:
            cut_failure = True
            cut_area_skip = True
        elif max_cut_bbox_pixels is not None and bbox_height * bbox_width > max_cut_bbox_pixels:
            cut_failure = True
            cut_bbox_skip = True

    # The training call has max_terminal_count=2, so this branch contains
    # exactly one pair and cannot enumerate O(k^2) terminal pairs.
    if compute_cut and not cut_failure:
        cut_pairs = (
            [(0, 1)] if len(terminal_cores) == 2
            else list(combinations(range(len(terminal_cores)), 2))
        )
        for first, second in cut_pairs:
            cut_attempt = True
            cut_start = time.perf_counter()
            try:
                pair_cut, pair_cost, pair_stats = _minimum_vertex_cut(
                    connected_component.mask,
                    connected_probability,
                    terminal_cores[first],
                    terminal_cores[second],
                    tau,
                    protected_mask=np.logical_or.reduce(
                        [connected_component.mask & component.mask
                         for component in separated_components]
                    ),
                    padding=cut_padding,
                )
                cut_mask |= pair_cut
                cut_build_runtime += pair_stats['build_runtime']
                cut_solve_runtime += pair_stats['solve_runtime']
                cut_node_count = max(cut_node_count, pair_stats['node_count'])
                cut_edge_count = max(cut_edge_count, pair_stats['edge_count'])
                cut_local_bbox_pixels = max(
                    cut_local_bbox_pixels, pair_stats['local_bbox_pixels']
                )
            except Exception:
                cut_failure = True
            cut_runtime += time.perf_counter() - cut_start

    if compute_join:
        join_pairs = (
            [(0, 1)] if len(terminal_cores) == 2
            else list(combinations(range(len(terminal_cores)), 2))
        )
        for first, second in join_pairs:
            join_start = time.perf_counter()
            try:
                pair_join, pair_cost = _minimum_completion_path(
                    separated_probability,
                    separated_foreground,
                    terminal_cores[first],
                    terminal_cores[second],
                    tau,
                    bbox_masks=[component.mask for component in separated_components],
                )
                join_mask |= pair_join
            except Exception:
                join_failure = True
            join_runtime += time.perf_counter() - join_start

    # Report the cost of the deduplicated event ROI, not repeated pair costs.
    cut_cost = float(
        np.maximum(connected_probability - float(tau), 1e-4)[cut_mask].sum()
    ) if cut_mask.any() else 0.0
    join_cost = float(
        np.maximum(float(tau) - separated_probability, 0.0)[join_mask].sum()
    ) if join_mask.any() else 0.0

    cut_validate_start = time.perf_counter()
    cut_flip_success = bool(
        compute_cut and cut_mask.any() and _groups_all_separated(
            connected_component.mask & (~cut_mask), terminal_cores
        )
    )
    if compute_cut:
        cut_validate_runtime = time.perf_counter() - cut_validate_start
        cut_runtime += cut_validate_runtime
    join_flip_success = bool(
        compute_join and join_mask.any() and _groups_connected(
            separated_foreground | join_mask, terminal_cores
        )
    )
    if compute_cut and not cut_flip_success:
        cut_failure = True
    if compute_join and not join_flip_success:
        join_failure = True
    if compute_join:
        cut_iou_intersection = np.logical_and(cut_mask, join_mask).sum()
        cut_iou_union = np.logical_or(cut_mask, join_mask).sum()
        cut_join_iou = (
            float(cut_iou_intersection / cut_iou_union)
            if cut_iou_union else 0.0
        )
        cut_join_min_distance = _mask_distance(cut_mask, join_mask)
        dual_agreement = bool(
            cut_join_iou > 0.0
            or cut_join_min_distance <= float(agreement_radius)
        )
    else:
        cut_join_iou = float('nan')
        cut_join_min_distance = float('nan')
        dual_agreement = False
    return {
        'cut_mask': cut_mask,
        'join_mask': join_mask,
        'cut_cost': float(cut_cost) if cut_mask.any() else float('nan'),
        'join_cost': float(join_cost) if join_mask.any() else float('nan'),
        'cut_flip_success': cut_flip_success,
        'join_flip_success': join_flip_success,
        'localization_valid': bool(cut_mask.any() and not cut_failure),
        'join_localization_valid': bool(join_mask.any() and not join_failure),
        'dual_agreement': dual_agreement,
        'cut_join_iou': cut_join_iou,
        'cut_join_min_distance': cut_join_min_distance,
        'cut_failure': cut_failure,
        'join_failure': join_failure,
        'cut_runtime': cut_runtime,
        'cut_build_runtime': cut_build_runtime,
        'cut_solve_runtime': cut_solve_runtime,
        'cut_validate_runtime': cut_validate_runtime,
        'cut_node_count': cut_node_count,
        'cut_edge_count': cut_edge_count,
        'cut_local_bbox_pixels': cut_local_bbox_pixels,
        'join_runtime': join_runtime,
        'compute_cut': bool(compute_cut),
        'cut_attempt': cut_attempt,
        'cut_success': cut_flip_success,
        'compute_join': bool(compute_join),
        'skipped_multiterminal': False,
        'cut_area_skip': cut_area_skip,
        'cut_bbox_skip': cut_bbox_skip,
        'skip_reason': (
            'area_guard' if cut_area_skip
            else 'bbox_guard' if cut_bbox_skip else ''
        ),
        'connected_probability': connected_probability,
        'separated_probability': separated_probability,
        'terminal_cores': terminal_cores,
    }


def _gt_relation(conflict: StructuralConflict, gt: np.ndarray) -> str:
    if conflict.connected_student == 'a':
        connected = conflict.connected_component.mask
        separated = conflict.separated_components
    elif conflict.connected_student == 'b':
        connected = conflict.connected_component.mask
        separated = conflict.separated_components
    else:
        return 'invalid'
    del connected
    gt_bool = gt.astype(bool)
    labels, _count = ndimage.label(gt_bool, structure=np.ones((3, 3), dtype=np.uint8))
    group_labels = []
    for component in separated:
        terminal = conflict.connected_component.mask & component.mask
        values = np.unique(labels[terminal & gt_bool])
        values = values[values > 0]
        if values.size != 1:
            return 'invalid'
        group_labels.append(int(values[0]))
    if len(group_labels) < 2:
        return 'invalid'
    return 'connected' if len(set(group_labels)) == 1 else 'separated'


def _nanmean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else 0.0


def _nanmedian(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else 0.0


def _record_sample_statistics(state: Dict, sample: Dict, localizations: Sequence[Dict], gt=None):
    state['samples'] += 1
    state['components_a'] += len(sample['components_a'])
    state['components_b'] += len(sample['components_b'])
    state['graph_edges'] += len(sample['edge_pairs'])
    state['one_to_one_groups'] += sample['one_to_one_groups']
    state['unmatched_a'] += len(sample['unmatched_a'])
    state['unmatched_b'] += len(sample['unmatched_b'])
    state['total_structural_conflicts'] += len(sample['conflicts'])
    state['merge_conflicts'] += sum(
        conflict.conflict_type == 'b_connected_a_separated'
        for conflict in sample['conflicts']
    )
    state['split_conflicts'] += sum(
        conflict.conflict_type == 'a_connected_b_separated'
        for conflict in sample['conflicts']
    )
    state['complex_conflicts'] += sum(
        conflict.conflict_type == 'complex_conflict'
        for conflict in sample['conflicts']
    )
    state['total_pixels'] += int(sample['pred_a'].size)
    for conflict, result in zip(sample['conflicts'], localizations):
        if len(conflict.separated_components) == 2:
            state['eligible_pair_conflicts'] += 1
        if result.get('skipped_multiterminal', False):
            state['skipped_multiterminal_events'] += 1
        if result.get('cut_attempt', False):
            state['cut_attempt_events'] += 1
        if result.get('cut_success', False):
            state['cut_success_events'] += 1
        if result.get('cut_area_skip', False):
            state['cut_area_skip_events'] += 1
        if result.get('cut_bbox_skip', False):
            state['cut_bbox_skip_events'] += 1
        if result.get('cut_attempt', False) and result.get('cut_node_count', 0) > 0:
            state['cut_node_counts'].append(int(result['cut_node_count']))
            state['cut_edge_counts'].append(int(result['cut_edge_count']))
            state['cut_local_bbox_pixels'].append(
                int(result['cut_local_bbox_pixels'])
            )
        connected_area = conflict.connected_component.area
        separated_area = sum(component.area for component in conflict.separated_components)
        state['connected_areas'].append(float(connected_area))
        state['separated_areas'].append(float(separated_area))
        if result.get('compute_cut', True) and result['localization_valid']:
            state['valid_cut_events'] += 1
            state['cut_costs'].append(result['cut_cost'])
            state['cut_pixels'].append(int(result['cut_mask'].sum()))
            if result.get('compute_join', True):
                state['cut_join_ious'].append(result['cut_join_iou'])
                state['cut_join_distances'].append(result['cut_join_min_distance'])
            state['structural_roi_pixels'] += int(result['cut_mask'].sum())
            if result['cut_flip_success']:
                state['cut_flip_success_events'] += 1
            if result['dual_agreement']:
                state['dual_agree_events'] += 1
        elif (
            result.get('compute_cut', True)
            and not result.get('skipped_multiterminal', False)
            and not result.get('cut_area_skip', False)
            and not result.get('cut_bbox_skip', False)
            and result.get('skip_reason', '') != 'diagnostics_only'
        ):
            state['cut_failure_events'] += 1
        if result.get('compute_join', True) and result['join_localization_valid']:
            state['valid_join_events'] += 1
            state['join_costs'].append(result['join_cost'])
            state['join_pixels'].append(int(result['join_mask'].sum()))
            if result['join_flip_success']:
                state['join_flip_success_events'] += 1
        elif (
            result.get('compute_join', True)
            and not result.get('skipped_multiterminal', False)
            and result.get('skip_reason', '') != 'diagnostics_only'
        ):
            state['join_failure_events'] += 1
        if gt is not None and conflict.conflict_type != 'complex_conflict':
            relation = _gt_relation(conflict, gt)
            if relation == 'invalid':
                state['both_wrong_structure_events'] += 1
            else:
                state['gt_valid_conflicts'] += 1
                if relation == 'connected':
                    state['connected_side_correct_events'] += 1
                elif relation == 'separated':
                    state['separated_side_correct_events'] += 1
            cut_pixels = result['cut_mask']
            join_pixels = result['join_mask']
            state['cut_roi_gt_pixels'] += int(cut_pixels.sum())
            state['cut_roi_gt_error_pixels'] += int((cut_pixels & (~gt.astype(bool))).sum())
            state['join_roi_gt_pixels'] += int(join_pixels.sum())
            state['join_roi_gt_error_pixels'] += int((join_pixels & gt.astype(bool)).sum())
            if result['cut_flip_success'] and relation == 'separated':
                state['cut_flip_and_connected_wrong_events'] += 1
            if result['join_flip_success'] and relation == 'connected':
                state['join_flip_and_separated_wrong_events'] += 1


def _merge_runtime(state: Dict, runtime: Dict):
    for key in (
        'runtime_cc', 'runtime_correspondence', 'runtime_cut',
        'runtime_cut_build', 'runtime_cut_solve', 'runtime_cut_validate',
        'runtime_join', 'runtime_total',
    ):
        state[key] += float(runtime.get(key, 0.0))


def analyze_cc_graph_batch(
    prob_a,
    prob_b,
    tau: float = 0.5,
    min_coverage: float = 0.5,
    min_intersection: int = 10,
    agreement_radius: float = 3.0,
    gt=None,
    compute_cut: bool = True,
    compute_join: bool = True,
    max_terminal_count: Optional[int] = None,
    max_cut_area: Optional[int] = None,
    max_cut_bbox_pixels: Optional[int] = None,
    cut_padding: int = 16,
) -> Dict:
    """Analyze one detached batch and return localizations plus timing."""
    total_start = time.perf_counter()
    graph_result = build_cc_graph_conflicts(
        prob_a, prob_b, tau, min_coverage, min_intersection
    )
    samples = graph_result['samples']
    cc_time = graph_result['runtime_cc']
    correspondence_time = graph_result['runtime_correspondence']
    cut_time = 0.0
    join_time = 0.0
    localizations = []
    for sample in samples:
        sample_results = []
        for conflict in sample['conflicts']:
            try:
                if not compute_cut and not compute_join:
                    result = _empty_localization_result(
                        sample['prob_a'].shape,
                        compute_cut=False,
                        compute_join=False,
                        reason='diagnostics_only',
                    )
                else:
                    result = localize_cc_graph_conflict(
                        conflict,
                        sample['prob_a'],
                        sample['prob_b'],
                        tau=tau,
                        agreement_radius=agreement_radius,
                        compute_cut=compute_cut,
                        compute_join=compute_join,
                        max_terminal_count=max_terminal_count,
                        max_cut_area=max_cut_area,
                        max_cut_bbox_pixels=max_cut_bbox_pixels,
                        cut_padding=cut_padding,
                    )
            except Exception:
                # A single malformed/oversized event must never abort the
                # batch.  Its all-one weights are the Vanilla CPS fallback.
                result = _empty_localization_result(
                    sample['prob_a'].shape,
                    compute_cut=compute_cut,
                    compute_join=compute_join,
                    reason='exception',
                )
            cut_time += result['cut_runtime']
            join_time += result['join_runtime']
            sample_results.append(result)
        localizations.append(sample_results)
    runtime = {
        'runtime_cc': cc_time,
        'runtime_correspondence': correspondence_time,
        'runtime_cut': cut_time,
        'runtime_cut_build': sum(
            result['cut_build_runtime']
            for sample_results in localizations
            for result in sample_results
        ),
        'runtime_cut_solve': sum(
            result['cut_solve_runtime']
            for sample_results in localizations
            for result in sample_results
        ),
        'runtime_cut_validate': sum(
            result['cut_validate_runtime']
            for sample_results in localizations
            for result in sample_results
        ),
        'runtime_join': join_time,
        'runtime_total': time.perf_counter() - total_start,
    }
    state = new_cc_graph_epoch_state(0, enabled=True)
    for sample, sample_results in zip(samples, localizations):
        _record_sample_statistics(
            state, sample, sample_results,
            None if gt is None else _as_batch_numpy(gt)[sample['sample_idx']]
        )
    _merge_runtime(state, runtime)
    return {
        'samples': samples,
        'localizations': localizations,
        'state': state,
        'runtime': runtime,
    }


def build_cc_graph_cps_weights(
    prob_a,
    prob_b,
    tau: float = 0.5,
    min_coverage: float = 0.5,
    min_intersection: int = 10,
    agreement_radius: float = 3.0,
    require_dual_agreement: bool = False,
    roi_mode: str = 'cut',
    compute_join: bool = False,
    max_cut_area: Optional[int] = 6000,
    max_cut_bbox_pixels: Optional[int] = 12000,
    cut_padding: int = 16,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return bidirectional CPS weights and one-batch CC-Graph diagnostics."""
    if roi_mode != 'cut':
        raise ValueError("CC-Graph V1 only supports roi_mode='cut'.")
    prob_a_np = _as_batch_numpy(prob_a)
    prob_b_np = _as_batch_numpy(prob_b)
    analysis = analyze_cc_graph_batch(
        prob_a_np, prob_b_np, tau, min_coverage, min_intersection,
        agreement_radius,
        compute_cut=True,
        compute_join=compute_join,
        max_terminal_count=2,
        max_cut_area=max_cut_area,
        max_cut_bbox_pixels=max_cut_bbox_pixels,
        cut_padding=cut_padding,
    )
    weight_a_from_b = np.ones(prob_a_np.shape, dtype=np.float32)
    weight_b_from_a = np.ones(prob_b_np.shape, dtype=np.float32)
    for sample, sample_results in zip(analysis['samples'], analysis['localizations']):
        for result in sample_results:
            if not result['localization_valid']:
                continue
            if compute_join and require_dual_agreement and not result['dual_agreement']:
                continue
            roi = result['cut_mask']
            weight_a_from_b[sample['sample_idx']] [roi] = 0.0
            weight_b_from_a[sample['sample_idx']] [roi] = 0.0
            analysis['state']['masked_cps_pixels'] += int(roi.sum())
    analysis['state']['structural_roi_pixel_ratio'] = (
        analysis['state']['structural_roi_pixels'] / max(analysis['state']['total_pixels'], 1)
    )
    analysis['state']['masked_cps_pixel_ratio'] = (
        analysis['state']['masked_cps_pixels'] / max(analysis['state']['total_pixels'], 1)
    )
    return weight_a_from_b, weight_b_from_a, analysis


def accumulate_cc_graph_state(target: Dict, source: Dict):
    """Add a batch state into an epoch state without sharing mutable lists."""
    for key in (
        'samples', 'components_a', 'components_b', 'graph_edges',
        'one_to_one_groups', 'eligible_pair_conflicts',
        'skipped_multiterminal_events', 'unmatched_a', 'unmatched_b',
        'merge_conflicts', 'split_conflicts', 'complex_conflicts',
        'total_structural_conflicts', 'valid_cut_events', 'cut_failure_events',
        'cut_area_skip_events', 'cut_bbox_skip_events',
        'cut_attempt_events', 'cut_success_events',
        'cut_flip_success_events', 'valid_join_events', 'join_failure_events',
        'join_flip_success_events', 'dual_agree_events',
        'structural_roi_pixels', 'masked_cps_pixels', 'total_pixels',
        'gt_valid_conflicts', 'connected_side_correct_events',
        'separated_side_correct_events', 'both_wrong_structure_events',
        'cut_roi_gt_error_pixels', 'cut_roi_gt_pixels',
        'join_roi_gt_error_pixels', 'join_roi_gt_pixels',
        'cut_flip_and_connected_wrong_events',
        'join_flip_and_separated_wrong_events',
    ):
        target[key] += source[key]
    for key in (
        'cut_costs', 'join_costs', 'cut_pixels', 'join_pixels',
        'cut_join_ious', 'cut_join_distances', 'connected_areas',
        'separated_areas', 'cut_node_counts', 'cut_edge_counts',
        'cut_local_bbox_pixels',
    ):
        target[key].extend(source[key])
    _merge_runtime(target, source)


def finalize_cc_graph_epoch_state(state: Dict) -> Dict:
    """Convert an epoch accumulator to the standalone CSV schema."""
    valid_cut = state['valid_cut_events']
    valid_join = state['valid_join_events']
    total_conflicts = state['total_structural_conflicts']
    gt_valid = state['gt_valid_conflicts']
    return {
        'epoch': state['epoch'],
        'samples': state['samples'],
        'components_a': state['components_a'],
        'components_b': state['components_b'],
        'graph_edges': state['graph_edges'],
        'one_to_one_groups': state['one_to_one_groups'],
        'eligible_pair_conflicts': state['eligible_pair_conflicts'],
        'skipped_multiterminal_events': state['skipped_multiterminal_events'],
        'unmatched_a': state['unmatched_a'],
        'unmatched_b': state['unmatched_b'],
        'merge_conflicts': state['merge_conflicts'],
        'split_conflicts': state['split_conflicts'],
        'complex_conflicts': state['complex_conflicts'],
        'total_structural_conflicts': total_conflicts,
        'valid_cut_events': valid_cut,
        'cut_failure_events': state['cut_failure_events'],
        'cut_area_skip_events': state['cut_area_skip_events'],
        'cut_bbox_skip_events': state['cut_bbox_skip_events'],
        'cut_attempt_events': state['cut_attempt_events'],
        'cut_success_events': state['cut_success_events'],
        'cut_flip_success_events': state['cut_flip_success_events'],
        'cut_flip_success_rate': state['cut_flip_success_events'] / max(valid_cut, 1),
        'valid_join_events': valid_join,
        'join_failure_events': state['join_failure_events'],
        'join_flip_success_events': state['join_flip_success_events'],
        'join_flip_success_rate': state['join_flip_success_events'] / max(valid_join, 1),
        'dual_agree_events': state['dual_agree_events'],
        'dual_agree_rate': state['dual_agree_events'] / max(valid_cut, 1),
        'mean_cut_cost': _nanmean(state['cut_costs']),
        'median_cut_cost': _nanmedian(state['cut_costs']),
        'mean_join_cost': _nanmean(state['join_costs']),
        'median_join_cost': _nanmedian(state['join_costs']),
        'mean_cut_pixels': _nanmean(state['cut_pixels']),
        'median_cut_pixels': _nanmedian(state['cut_pixels']),
        'mean_join_pixels': _nanmean(state['join_pixels']),
        'median_join_pixels': _nanmedian(state['join_pixels']),
        'mean_cut_join_iou': _nanmean(state['cut_join_ious']),
        'mean_cut_join_min_distance': _nanmean(state['cut_join_distances']),
        'structural_roi_pixels': state['structural_roi_pixels'],
        'structural_roi_pixel_ratio': state['structural_roi_pixels'] / max(state['total_pixels'], 1),
        'masked_cps_pixels': state['masked_cps_pixels'],
        'masked_cps_pixel_ratio': state['masked_cps_pixels'] / max(state['total_pixels'], 1),
        'mean_connected_component_area': _nanmean(state['connected_areas']),
        'mean_separated_component_area': _nanmean(state['separated_areas']),
        'runtime_cc': state['runtime_cc'],
        'runtime_correspondence': state['runtime_correspondence'],
        'runtime_cut': state['runtime_cut'],
        'runtime_cut_build': state['runtime_cut_build'],
        'runtime_cut_solve': state['runtime_cut_solve'],
        'runtime_cut_validate': state['runtime_cut_validate'],
        'runtime_join': state['runtime_join'],
        'runtime_total': state['runtime_total'],
        'cut_node_count_mean': _nanmean(state['cut_node_counts']),
        'cut_node_count_max': max(state['cut_node_counts'], default=0),
        'cut_edge_count_mean': _nanmean(state['cut_edge_counts']),
        'cut_edge_count_max': max(state['cut_edge_counts'], default=0),
        'cut_local_bbox_pixels_mean': _nanmean(state['cut_local_bbox_pixels']),
        'cut_local_bbox_pixels_max': max(state['cut_local_bbox_pixels'], default=0),
        'gt_valid_conflicts': gt_valid,
        'connected_side_correct_events': state['connected_side_correct_events'],
        'separated_side_correct_events': state['separated_side_correct_events'],
        'both_wrong_structure_events': state['both_wrong_structure_events'],
        'connected_side_correct_rate': state['connected_side_correct_events'] / max(gt_valid, 1),
        'separated_side_correct_rate': state['separated_side_correct_events'] / max(gt_valid, 1),
        'cut_roi_gt_error_rate': state['cut_roi_gt_error_pixels'] / max(state['cut_roi_gt_pixels'], 1),
        'join_roi_gt_error_rate': state['join_roi_gt_error_pixels'] / max(state['join_roi_gt_pixels'], 1),
        'cut_flip_and_connected_wrong_rate': state['cut_flip_and_connected_wrong_events'] / max(state['cut_flip_success_events'], 1),
        'join_flip_and_separated_wrong_rate': state['join_flip_and_separated_wrong_events'] / max(state['join_flip_success_events'], 1),
    }


def update_cc_graph_state(target: Dict, analysis: Dict, gt=None):
    """Record a batch analysis; optional GT is diagnostics-only."""
    if gt is not None:
        gt_np = _as_batch_numpy(gt)
        gt_state = new_cc_graph_epoch_state(target.get('epoch', 0), enabled=True)
        for sample, sample_results in zip(analysis['samples'], analysis['localizations']):
            _record_sample_statistics(
                gt_state, sample, sample_results, gt_np[sample['sample_idx']]
            )
        gt_state['samples'] = 0
        gt_state['components_a'] = 0
        gt_state['components_b'] = 0
        gt_state['graph_edges'] = 0
        gt_state['one_to_one_groups'] = 0
        gt_state['unmatched_a'] = 0
        gt_state['unmatched_b'] = 0
        gt_state['merge_conflicts'] = 0
        gt_state['split_conflicts'] = 0
        gt_state['complex_conflicts'] = 0
        gt_state['total_structural_conflicts'] = 0
        gt_state['valid_cut_events'] = 0
        gt_state['cut_failure_events'] = 0
        gt_state['cut_flip_success_events'] = 0
        gt_state['valid_join_events'] = 0
        gt_state['join_failure_events'] = 0
        gt_state['join_flip_success_events'] = 0
        gt_state['dual_agree_events'] = 0
        gt_state['connected_areas'] = []
        gt_state['separated_areas'] = []
        gt_state['cut_costs'] = []
        gt_state['join_costs'] = []
        gt_state['cut_pixels'] = []
        gt_state['join_pixels'] = []
        gt_state['cut_join_ious'] = []
        gt_state['cut_join_distances'] = []
        gt_state['structural_roi_pixels'] = 0
        gt_state['masked_cps_pixels'] = 0
        gt_state['total_pixels'] = 0
        gt_state['runtime_cc'] = 0.0
        gt_state['runtime_correspondence'] = 0.0
        gt_state['runtime_cut'] = 0.0
        gt_state['runtime_join'] = 0.0
        gt_state['runtime_total'] = 0.0
        for key in (
            'gt_valid_conflicts', 'connected_side_correct_events',
            'separated_side_correct_events', 'both_wrong_structure_events',
            'cut_roi_gt_error_pixels', 'cut_roi_gt_pixels',
            'join_roi_gt_error_pixels', 'join_roi_gt_pixels',
            'cut_flip_and_connected_wrong_events',
            'join_flip_and_separated_wrong_events',
        ):
            target[key] += gt_state[key]
        return
    accumulate_cc_graph_state(target, analysis['state'])


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image.astype(np.float32)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
    return (image * 255).clip(0, 255).astype(np.uint8)


def save_cc_graph_overview(
    output_dir: str,
    epoch: int,
    sample_index: int,
    image,
    sample: Dict,
    localizations: Sequence[Dict],
):
    """Save one compact 8-panel validation overview for a selected sample."""
    import matplotlib.pyplot as plt

    if not localizations:
        return None
    image = _to_uint8_rgb(image)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title('RGB')
    axes[1].imshow(sample['labels_a'], cmap='nipy_spectral')
    axes[1].set_title('pred A CC labels')
    axes[2].imshow(sample['labels_b'], cmap='nipy_spectral')
    axes[2].set_title('pred B CC labels')
    conflict_overlay = np.zeros((*sample['pred_a'].shape, 3), dtype=np.float32)
    for result in localizations:
        conflict_overlay[result['cut_mask']] = (1.0, 0.0, 0.0)
        conflict_overlay[result['join_mask']] = (0.0, 1.0, 0.0)
    axes[3].imshow(image)
    axes[3].imshow(conflict_overlay, alpha=0.65)
    axes[3].set_title('correspondence/conflict ROI')
    connected_probability = localizations[0].get('connected_probability', sample['prob_a'])
    separated_probability = localizations[0].get('separated_probability', sample['prob_b'])
    axes[4].imshow(connected_probability, vmin=0, vmax=1, cmap='viridis')
    axes[4].set_title('connected-side probability')
    cut_mask = np.logical_or.reduce([result['cut_mask'] for result in localizations])
    axes[5].imshow(cut_mask, cmap='Reds')
    axes[5].set_title('cut ROI')
    axes[6].imshow(separated_probability, vmin=0, vmax=1, cmap='viridis')
    axes[6].set_title('separated-side probability')
    join_mask = np.logical_or.reduce([result['join_mask'] for result in localizations])
    axes[7].imshow(join_mask, cmap='Greens')
    title = []
    for result in localizations:
        title.append(
            'cut={:.3g} join={:.3g} cut_flip={} join_flip={} agree={}'.format(
                result['cut_cost'], result['join_cost'],
                int(result['cut_flip_success']), int(result['join_flip_success']),
                int(result['dual_agreement']),
            )
        )
    axes[7].set_title('join path\n' + '\n'.join(title))
    for axis in axes:
        axis.axis('off')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'cc_graph_epoch_{:03d}_sample_{:04d}.png'.format(epoch, sample_index))
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_cc_graph_header(file_obj):
    writer = csv.DictWriter(file_obj, fieldnames=CC_GRAPH_FIELDS)
    writer.writeheader()
    return writer
