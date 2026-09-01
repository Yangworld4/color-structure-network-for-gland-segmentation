from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


ABSENT = 'ABSENT'
SEPARATED = 'SEPARATED'
CONNECTED = 'CONNECTED'


@dataclass
class MergeEdge:
    child_id: int
    parent_id: int
    merge_level: float
    saddle_coord: tuple


@dataclass
class PHRelationQuery:
    state: str
    merge_level: float
    saddle_coord: tuple
    bottleneck_edge_id: int


class PHMergeTree:
    def __init__(self, branches, edges):
        self.branches = {int(branch.branch_id): branch for branch in branches}
        self.edges = []
        self.edge_by_child = {}
        for index, edge in enumerate(edges):
            merge_edge = MergeEdge(
                child_id=int(edge.child_branch_id),
                parent_id=int(edge.parent_branch_id),
                merge_level=float(edge.merge_level),
                saddle_coord=tuple(edge.merge_coord),
            )
            self.edges.append(merge_edge)
            self.edge_by_child[merge_edge.child_id] = (index, merge_edge)


def build_merge_hierarchy(decomposition):
    return PHMergeTree(decomposition.branches, decomposition.edges)


def trace_to_root(tree, branch_id):
    branch_id = int(branch_id)
    trace = []
    seen = set()
    current = branch_id
    while current in tree.edge_by_child and current not in seen:
        seen.add(current)
        edge_id, edge = tree.edge_by_child[current]
        trace.append((current, edge_id, edge))
        current = int(edge.parent_id)
    return trace, current


def query_pair_merge(tree, branch_u, branch_v):
    branch_u = int(branch_u)
    branch_v = int(branch_v)
    if branch_u == branch_v:
        branch = tree.branches.get(branch_u)
        saddle = branch.birth_coord if branch is not None else (0, 0)
        level = branch.birth if branch is not None else 0.0
        return float(level), tuple(saddle), -1

    trace_u, root_u = trace_to_root(tree, branch_u)
    trace_v, root_v = trace_to_root(tree, branch_v)
    if root_u != root_v:
        return None, None, None

    ancestors_u = {branch_u: []}
    path = []
    current = branch_u
    for _node, edge_id, edge in trace_u:
        path.append((edge_id, edge))
        current = int(edge.parent_id)
        ancestors_u[current] = list(path)

    path_v = []
    current = branch_v
    if current in ancestors_u:
        full_path = ancestors_u[current]
    else:
        full_path = None
        for _node, edge_id, edge in trace_v:
            path_v.append((edge_id, edge))
            current = int(edge.parent_id)
            if current in ancestors_u:
                full_path = ancestors_u[current] + path_v
                break
    if not full_path:
        return None, None, None

    bottleneck_edge_id, bottleneck = min(
        full_path,
        key=lambda item: float(item[1].merge_level),
    )
    return (
        float(bottleneck.merge_level),
        tuple(bottleneck.saddle_coord),
        int(bottleneck_edge_id),
    )


def query_ph_relation(tree, branch_u, branch_v, tau=0.5):
    branch_u = int(branch_u)
    branch_v = int(branch_v)
    u = tree.branches.get(branch_u)
    v = tree.branches.get(branch_v)
    if u is None or v is None:
        return PHRelationQuery(ABSENT, 0.0, None, -1)
    if float(u.birth) < tau or float(v.birth) < tau:
        return PHRelationQuery(ABSENT, 0.0, None, -1)

    merge_level, saddle_coord, edge_id = query_pair_merge(tree, branch_u, branch_v)
    if merge_level is None:
        return PHRelationQuery(ABSENT, 0.0, None, -1)
    state = CONNECTED if float(merge_level) >= tau else SEPARATED
    return PHRelationQuery(state, float(merge_level), saddle_coord, int(edge_id))


def build_relation_candidates(edges_a, edges_b, matches):
    a_to_match = {}
    b_to_match = {}
    for match_index, match in enumerate(matches):
        a_to_match[int(match['entity_a'].branch_id)] = match_index
        b_to_match[int(match['entity_b'].branch_id)] = match_index

    candidates = {}
    for edge in edges_a:
        child = int(edge.child_branch_id)
        parent = int(edge.parent_branch_id)
        if child not in a_to_match or parent not in a_to_match:
            continue
        key = tuple(sorted((a_to_match[child], a_to_match[parent])))
        if key[0] == key[1]:
            continue
        candidates.setdefault(key, {'sources': []})['edge_a'] = edge
        candidates[key]['sources'].append('A')
    for edge in edges_b:
        child = int(edge.child_branch_id)
        parent = int(edge.parent_branch_id)
        if child not in b_to_match or parent not in b_to_match:
            continue
        key = tuple(sorted((b_to_match[child], b_to_match[parent])))
        if key[0] == key[1]:
            continue
        candidates.setdefault(key, {'sources': []})['edge_b'] = edge
        candidates[key]['sources'].append('B')
    return candidates


def detect_ph_relation_conflicts(
    relation_candidates,
    tree_a,
    tree_b,
    matches,
    tau=0.5,
    collect_cc_agreement=False,
    pred_a=None,
    pred_b=None,
    presence_conflict_mode='strict',
):
    if presence_conflict_mode not in ('strict', 'connectivity_outcome'):
        raise ValueError(
            'Unsupported PH presence conflict mode: {}'.format(presence_conflict_mode)
        )
    conflicts = []
    diagnostics = _new_relation_tree_stats()
    diagnostics['num_relation_questions'] = len(relation_candidates)
    cc_a = cc_b = None
    if collect_cc_agreement and pred_a is not None and pred_b is not None:
        cc_a, _ = label(pred_a, structure=np.ones((3, 3), dtype=np.uint8))
        cc_b, _ = label(pred_b, structure=np.ones((3, 3), dtype=np.uint8))

    for canonical_pair, edge_refs in relation_candidates.items():
        left_id, right_id = canonical_pair
        left_match = matches[left_id]
        right_match = matches[right_id]
        left_a = left_match['entity_a']
        right_a = right_match['entity_a']
        left_b = left_match['entity_b']
        right_b = right_match['entity_b']
        query_a = query_ph_relation(
            tree_a, left_a.branch_id, right_a.branch_id, tau=tau
        )
        query_b = query_ph_relation(
            tree_b, left_b.branch_id, right_b.branch_id, tau=tau
        )
        pair_present_a = _pair_present(left_a, right_a, tau)
        pair_present_b = _pair_present(left_b, right_b, tau)
        _accumulate_presence_question_stats(
            diagnostics, pair_present_a, pair_present_b
        )
        _accumulate_query_stats(diagnostics, query_a, query_b, tau)
        connected_a = query_a.state == CONNECTED
        connected_b = query_b.state == CONNECTED
        diagnostics['relation_question_records'].append({
            'canonical_pair': canonical_pair,
            'left_match': left_match,
            'right_match': right_match,
            'query_a': query_a,
            'query_b': query_b,
            'connected_a': connected_a,
            'connected_b': connected_b,
            'pair_present_a': pair_present_a,
            'pair_present_b': pair_present_b,
            'source': '+'.join(sorted(set(edge_refs.get('sources', [])))) or '',
            'candidate_id': '{}-{}'.format(canonical_pair[0], canonical_pair[1]),
        })
        if cc_a is not None and cc_b is not None:
            _accumulate_cc_agreement(
                diagnostics, query_a, query_b, left_a, right_a, left_b, right_b, cc_a, cc_b
            )
        if presence_conflict_mode == 'strict' and (
            query_a.state == ABSENT or query_b.state == ABSENT
        ):
            _accumulate_absent_stats(diagnostics, query_a, query_b)
            if cc_a is not None and cc_b is not None:
                _accumulate_old_cc_overlap(
                    diagnostics,
                    False,
                    pair_present_a,
                    pair_present_b,
                    left_a,
                    right_a,
                    left_b,
                    right_b,
                    cc_a,
                    cc_b,
                )
            continue
        if presence_conflict_mode == 'connectivity_outcome' and (
            query_a.state == ABSENT or query_b.state == ABSENT
        ):
            _accumulate_absent_stats(diagnostics, query_a, query_b)
        if presence_conflict_mode == 'strict':
            structural_conflict = query_a.state != query_b.state
        else:
            structural_conflict = connected_a != connected_b
        if cc_a is not None and cc_b is not None:
            _accumulate_old_cc_overlap(
                diagnostics,
                structural_conflict,
                pair_present_a,
                pair_present_b,
                left_a,
                right_a,
                left_b,
                right_b,
                cc_a,
                cc_b,
            )
        if not structural_conflict:
            if (
                presence_conflict_mode == 'connectivity_outcome'
                and pair_present_a != pair_present_b
            ):
                diagnostics['num_presence_disagreement_nonconflict'] += 1
            continue

        if connected_a:
            conflict_type = 'a_connected_b_separate'
            saddle = query_a.saddle_coord
            connected_side = 'A'
        else:
            conflict_type = 'a_separate_b_connected'
            saddle = query_b.saddle_coord
            connected_side = 'B'
        structural_conflict_type = (
            'pure_relation'
            if pair_present_a and pair_present_b
            else 'presence_involved'
        )
        diagnostics['num_relation_conflicts'] += 1
        diagnostics['num_total_structural_conflicts'] += 1
        if structural_conflict_type == 'pure_relation':
            diagnostics['num_pure_relation_conflicts'] += 1
        else:
            diagnostics['num_presence_involved_conflicts'] += 1
        diagnostics['num_A_connected_B_separated'] += int(
            conflict_type == 'a_connected_b_separate'
        )
        diagnostics['num_A_separated_B_connected'] += int(
            conflict_type == 'a_separate_b_connected'
        )
        diagnostics['conflict_merge_level_A_sum'] += float(query_a.merge_level)
        diagnostics['conflict_merge_level_B_sum'] += float(query_b.merge_level)
        conflicts.append({
            'canonical_pair': canonical_pair,
            'left_match': left_match,
            'right_match': right_match,
            'edge_a': edge_refs.get('edge_a'),
            'edge_b': edge_refs.get('edge_b'),
            'query_a': query_a,
            'query_b': query_b,
            'connected_a': connected_a,
            'connected_b': connected_b,
            'conflict_type': conflict_type,
            'structural_conflict_type': structural_conflict_type,
            'connected_side': connected_side,
            'saddle_coord': saddle,
            'source': '+'.join(sorted(set(edge_refs.get('sources', [])))) or '',
            'candidate_id': '{}-{}'.format(canonical_pair[0], canonical_pair[1]),
            'pair_present_a': pair_present_a,
            'pair_present_b': pair_present_b,
        })
    return conflicts, diagnostics


def _new_relation_tree_stats():
    return {
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


def _accumulate_query_stats(stats, query_a, query_b, tau):
    if query_a.state != ABSENT and query_b.state != ABSENT:
        stats['num_valid_relation_queries'] += 1
        stats['merge_level_A_sum'] += float(query_a.merge_level)
        stats['merge_level_B_sum'] += float(query_b.merge_level)
        stats['abs_g_A_sum'] += abs(float(query_a.merge_level) - tau)
        stats['abs_g_B_sum'] += abs(float(query_b.merge_level) - tau)
    if query_a.state == CONNECTED:
        stats['num_connected_A'] += 1
    elif query_a.state == SEPARATED:
        stats['num_separated_A'] += 1
    if query_b.state == CONNECTED:
        stats['num_connected_B'] += 1
    elif query_b.state == SEPARATED:
        stats['num_separated_B'] += 1


def _accumulate_absent_stats(stats, query_a, query_b):
    if query_a.state == ABSENT:
        stats['num_absent_A'] += 1
    if query_b.state == ABSENT:
        stats['num_absent_B'] += 1


def _pair_present(branch_u, branch_v, tau):
    return float(branch_u.birth) >= tau and float(branch_v.birth) >= tau


def _accumulate_presence_question_stats(stats, pair_present_a, pair_present_b):
    if pair_present_a and pair_present_b:
        stats['num_pair_present_both'] += 1
    elif not pair_present_a and not pair_present_b:
        stats['num_pair_absent_both'] += 1
    elif not pair_present_a:
        stats['num_pair_absent_a'] += 1
    else:
        stats['num_pair_absent_b'] += 1


def _accumulate_old_cc_overlap(
    stats,
    structural_conflict,
    pair_present_a,
    pair_present_b,
    left_a,
    right_a,
    left_b,
    right_b,
    cc_a,
    cc_b,
):
    cc_connected_a = _connected(cc_a, left_a.birth_coord, right_a.birth_coord)
    cc_connected_b = _connected(cc_b, left_b.birth_coord, right_b.birth_coord)
    old_cc_conflict = cc_connected_a != cc_connected_b
    stats['old_cc_conflicts'] += int(old_cc_conflict)
    stats['new_conflicts'] += int(structural_conflict)
    if old_cc_conflict and structural_conflict:
        stats['old_cc_retained_conflicts'] += 1
    elif old_cc_conflict:
        stats['old_cc_lost_conflicts'] += 1
    elif structural_conflict:
        stats['new_extra_conflicts'] += 1
    if old_cc_conflict:
        if not structural_conflict:
            stats['old_cc_to_nonconflict'] += 1
        elif pair_present_a and pair_present_b:
            stats['old_cc_to_pure_relation'] += 1
        else:
            stats['old_cc_to_presence_involved'] += 1


def _accumulate_cc_agreement(
    stats, query_a, query_b, left_a, right_a, left_b, right_b, cc_a, cc_b
):
    if query_a.state != ABSENT and query_b.state != ABSENT:
        stats['valid_compare_count'] += 1
        stats['ph_cc_agree_A'] += int(
            (query_a.state == CONNECTED) == _connected(cc_a, left_a.birth_coord, right_a.birth_coord)
        )
        stats['ph_cc_agree_B'] += int(
            (query_b.state == CONNECTED) == _connected(cc_b, left_b.birth_coord, right_b.birth_coord)
        )


def _connected(labels, coord_a, coord_b):
    ay, ax = coord_a
    by, bx = coord_b
    label_a = int(labels[ay, ax])
    return label_a > 0 and label_a == int(labels[by, bx])
