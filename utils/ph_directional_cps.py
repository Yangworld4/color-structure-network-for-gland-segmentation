import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils.ph_correspondence import build_persistent_relation_conflict_events
from utils.ph_event_projection import (
    build_event_projection_conflict_events,
    update_event_projection_oracle_state,
)
from utils.ph_mask_only import build_ph_conflict_events
from utils.structural_diagnostics import _component_error_sets, _extract_components


PH_DIRECTIONAL_FIELDS = [
    'epoch',
    'ph_enabled',
    'ph_conflict_events',
    'ph_conflict_pixel_ratio',
    'a2b_events',
    'b2a_events',
    'a2b_pixels',
    'b2a_pixels',
    'mean_reliability_a',
    'mean_reliability_b',
    'mean_abs_reliability_gap',
    'mean_weight_a2b',
    'mean_weight_b2a',
    'loss_cps_original_estimate',
    'loss_cps_directional',
    'loss_a_from_b',
    'loss_b_from_a',
    'effective_a2b_pixel_ratio',
    'effective_b2a_pixel_ratio',
    'val_dice_a',
    'val_dice_b',
    'val_jaccard_a',
    'val_jaccard_b',
    'both_struct_correct_ratio',
    'a_only_struct_correct_ratio',
    'b_only_struct_correct_ratio',
    'both_struct_wrong_ratio',
    'structural_complementary_ratio',
    'merge_error_a',
    'merge_error_b',
    'split_error_a',
    'split_error_b',
    'oracle_routable_events',
    'routing_correct_events',
    'routing_wrong_events',
    'routing_accuracy',
    'harmful_routing_events',
    'harmful_routing_rate',
    'mean_weight_correct_direction',
    'mean_weight_wrong_direction',
    'merge_routing_accuracy',
    'split_routing_accuracy',
    'ph_pairs_a',
    'ph_pairs_b',
    'matched_ph_pairs',
    'unmatched_ph_a',
    'unmatched_ph_b',
    'ph_pair_match_rate',
    'matched_connectivity_conflicts',
    'matched_merge_conflicts',
    'matched_split_conflicts',
    'unmatched_conflict_candidates',
    'mean_match_score',
    'mean_match_overlap',
    'mean_birth_distance',
    'mean_death_distance',
    'directional_events_from_matched_pairs',
    'matched_conflict_oracle_events',
    'matched_routing_correct',
    'matched_routing_wrong',
    'matched_routing_accuracy',
    'matched_harmful_routing_rate',
    'ph_entities_a',
    'ph_entities_b',
    'raw_ph_branches_a',
    'raw_ph_branches_b',
    'matched_entities',
    'matched_branches',
    'unmatched_entities_a',
    'unmatched_entities_b',
    'unmatched_branches_a',
    'unmatched_branches_b',
    'support_candidate_pairs',
    'mean_birth_match_distance',
    'mean_matched_support_iou',
    'median_matched_support_iou',
    'mean_matched_birth_distance',
    'mean_support_area_a',
    'mean_support_area_b',
    'direct_merge_edges_a',
    'direct_merge_edges_b',
    'aligned_relation_candidates',
    'connectivity_conflicts',
    'mean_conflict_roi_pixels',
    'ph_extract_time',
    'support_build_time',
    'support_overlap_time',
    'entity_match_time',
    'relation_time',
    'roi_time',
    'total_ph_time',
    'entity_match_rate',
    'mean_core_iou',
    'mean_centroid_distance',
    'relation_candidates',
    'persistent_connectivity_conflicts',
    'a_separate_b_connected',
    'a_connected_b_separate',
    'existence_conflict_a_only',
    'existence_conflict_b_only',
    'merge_tree_cc_consistency_rate',
    'conflict_pixel_ratio',
    'directional_events_from_persistent_conflicts',
    'entity_match_oracle_pairs',
    'entity_match_correct_pairs',
    'entity_match_precision',
    'oracle_single_branch_correct_conflicts',
    'oracle_both_wrong_conflicts',
    'oracle_both_correct_conflicts',
    'single_branch_correct_conflict_rate',
]


PH_CORRESPONDENCE_FIELDS = [
    'epoch',
    'raw_ph_branches_a',
    'raw_ph_branches_b',
    'support_candidate_pairs',
    'matched_branches',
    'unmatched_a',
    'unmatched_b',
    'mean_support_iou',
    'median_support_iou',
    'mean_birth_distance',
    'mean_support_area_a',
    'mean_support_area_b',
    'direct_edges_a',
    'direct_edges_b',
    'aligned_relation_candidates',
    'connectivity_conflicts',
    'conflict_pixel_ratio',
    'ph_time',
    'support_time',
    'support_overlap_time',
    'matching_time',
    'relation_time',
    'roi_time',
    'total_time',
    'matched_same_gt_structure',
    'matched_different_gt_structure',
    'entity_match_precision',
    'entity_match_recall',
    'mean_ph_branches_per_gt_gland_A',
    'mean_ph_branches_per_gt_gland_B',
    'median_ph_branches_per_gt_gland_A',
    'median_ph_branches_per_gt_gland_B',
    'p90_ph_branches_per_gt_gland_A',
    'p90_ph_branches_per_gt_gland_B',
    'conflict_both_correct',
    'conflict_A_only_correct',
    'conflict_B_only_correct',
    'conflict_both_wrong',
    'complementary_conflict_rate',
]

PH_RELATION_TREE_DIAGNOSTIC_FIELDS = [
    'epoch',
    'num_branches_A',
    'num_branches_B',
    'num_direct_edges_A',
    'num_direct_edges_B',
    'num_matched_branches',
    'num_relation_candidates',
    'num_valid_relation_queries',
    'num_absent_A',
    'num_absent_B',
    'num_connected_A',
    'num_separated_A',
    'num_connected_B',
    'num_separated_B',
    'num_relation_conflicts',
    'num_A_connected_B_separated',
    'num_A_separated_B_connected',
    'mean_merge_level_A',
    'mean_merge_level_B',
    'mean_abs_g_A',
    'mean_abs_g_B',
    'mean_conflict_merge_level_A',
    'mean_conflict_merge_level_B',
    'num_roi',
    'conflict_pixel_ratio',
    'valid_compare_count',
    'ph_cc_agree_A',
    'ph_cc_agree_B',
    'ph_cc_agreement_rate_A',
    'ph_cc_agreement_rate_B',
    'ph_extract',
    'birth_matching',
    'candidate_build',
    'relation_tree_query',
    'roi',
    'routing',
    'total',
]


PERSISTENT_RELATION_EVENT_FIELDS = [
    'epoch',
    'image_id',
    'entity_i_a',
    'entity_i_b',
    'entity_j_a',
    'entity_j_b',
    'core_iou_i',
    'core_iou_j',
    'birth_i_a',
    'birth_i_b',
    'birth_j_a',
    'birth_j_b',
    'mu_a',
    'mu_b',
    'connected_a',
    'connected_b',
    'saddle_x_a',
    'saddle_y_a',
    'saddle_x_b',
    'saddle_y_b',
    'conflict_type',
    'roi_pixels',
    'reliability_a',
    'reliability_b',
    'routing_direction',
    'routing_weight',
    'oracle_direction',
    'routing_correct',
]

PH_RELATION_TREE_EVENT_FIELDS = [
    'epoch',
    'sample_id',
    'source',
    'A_branch_u',
    'A_branch_v',
    'B_branch_u',
    'B_branch_v',
    'birth_u_A',
    'birth_v_A',
    'birth_u_B',
    'birth_v_B',
    'merge_level_A',
    'merge_level_B',
    'g_A',
    'g_B',
    'state_A',
    'state_B',
    'saddle_row',
    'saddle_col',
    'routing_direction',
    'routing_weight',
    'roi_pixels',
]

PH_PRESENCE_CONFLICT_DIAGNOSTIC_FIELDS = [
    'epoch',
    'num_relation_questions',
    'num_pair_present_both',
    'num_pair_absent_a',
    'num_pair_absent_b',
    'num_pair_absent_both',
    'num_pure_relation_conflicts',
    'num_presence_involved_conflicts',
    'num_total_structural_conflicts',
    'num_presence_disagreement_nonconflict',
    'pure_relation_conflict_pixel_ratio',
    'presence_involved_conflict_pixel_ratio',
    'total_structural_conflict_pixel_ratio',
    'a_connected_b_not_count',
    'b_connected_a_not_count',
    'old_cc_conflicts',
    'new_conflicts',
    'old_cc_retained_conflicts',
    'old_cc_lost_conflicts',
    'new_extra_conflicts',
    'old_cc_recall',
    'new_vs_old_jaccard',
    'old_cc_to_pure_relation',
    'old_cc_to_presence_involved',
    'old_cc_to_nonconflict',
    'old_cc_candidate_missing',
]

PH_PRESENCE_CONFLICT_EVENT_FIELDS = [
    'epoch',
    'sample_id',
    'candidate_id',
    'source_event_side',
    'a_u_id',
    'a_v_id',
    'b_u_id',
    'b_v_id',
    'birth_a_u',
    'birth_a_v',
    'birth_b_u',
    'birth_b_v',
    'pair_present_a',
    'pair_present_b',
    'merge_level_a',
    'merge_level_b',
    'connected_a',
    'connected_b',
    'semantic_state_a',
    'semantic_state_b',
    'conflict_type',
    'connected_side',
    'saddle_x',
    'saddle_y',
    'roi_pixels',
]


def new_ph_directional_epoch_state(epoch, enabled):
    return {
        'epoch': int(epoch),
        'ph_enabled': int(bool(enabled)),
        'ph_conflict_events': 0,
        'ph_conflict_pixels': 0,
        'total_unlabeled_pixels': 0,
        'a2b_events': 0,
        'b2a_events': 0,
        'a2b_pixels': 0,
        'b2a_pixels': 0,
        'reliability_a_sum': 0.0,
        'reliability_b_sum': 0.0,
        'abs_reliability_gap_sum': 0.0,
        'weight_a2b_sum': 0.0,
        'weight_b2a_sum': 0.0,
        'event_weight_count': 0,
        'loss_cps_original_estimate_sum': 0.0,
        'loss_cps_directional_sum': 0.0,
        'loss_a_from_b_sum': 0.0,
        'loss_b_from_a_sum': 0.0,
        'loss_batches': 0,
        'oracle_routable_events': 0,
        'routing_correct_events': 0,
        'routing_wrong_events': 0,
        'harmful_routing_events': 0,
        'weight_correct_direction_sum': 0.0,
        'weight_wrong_direction_sum': 0.0,
        'merge_oracle_routable_events': 0,
        'merge_routing_correct_events': 0,
        'split_oracle_routable_events': 0,
        'split_routing_correct_events': 0,
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
        'matched_conflict_oracle_events': 0,
        'matched_routing_correct': 0,
        'matched_routing_wrong': 0,
        'matched_harmful_routing_events': 0,
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
        'support_iou_sum': 0.0,
        'support_iou_count': 0,
        'support_iou_values': [],
        'mean_support_area_a_sum': 0.0,
        'mean_support_area_b_sum': 0.0,
        'support_area_batches': 0,
        'direct_merge_edges_a': 0,
        'direct_merge_edges_b': 0,
        'aligned_relation_candidates': 0,
        'connectivity_conflicts': 0,
        'mean_conflict_roi_pixels_sum': 0.0,
        'mean_conflict_roi_pixels_count': 0,
        'ph_extract_time_sum': 0.0,
        'support_build_time_sum': 0.0,
        'support_overlap_time_sum': 0.0,
        'entity_match_time_sum': 0.0,
        'relation_time_sum': 0.0,
        'roi_time_sum': 0.0,
        'total_ph_time_sum': 0.0,
        'ph_timing_batches': 0,
        'core_iou_sum': 0.0,
        'centroid_distance_sum': 0.0,
        'relation_candidates': 0,
        'persistent_connectivity_conflicts': 0,
        'a_separate_b_connected': 0,
        'a_connected_b_separate': 0,
        'existence_conflict_a_only': 0,
        'existence_conflict_b_only': 0,
        'merge_tree_cc_consistency_sum': 0.0,
        'merge_tree_cc_consistency_count': 0,
        'directional_events_from_persistent_conflicts': 0,
        'entity_match_oracle_pairs': 0,
        'entity_match_correct_pairs': 0,
        'oracle_single_branch_correct_conflicts': 0,
        'oracle_both_wrong_conflicts': 0,
        'oracle_both_correct_conflicts': 0,
        'conflict_both_correct': 0,
        'conflict_A_only_correct': 0,
        'conflict_B_only_correct': 0,
        'conflict_both_wrong': 0,
        'complementary_conflict_rate': 0.0,
        'raw_ph_branches_A': 0,
        'raw_ph_branches_B': 0,
        'merge_events_A': 0,
        'merge_events_B': 0,
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
        'A_separate_B_connected': 0,
        'A_connected_B_separate': 0,
        'conflict_roi_count': 0,
        'event_projection_roi_size_sum': 0.0,
        'event_projection_roi_sizes': [],
        'event_projection_ph_extract_time_sum': 0.0,
        'event_projection_cc_time_sum': 0.0,
        'event_projection_event_query_time_sum': 0.0,
        'event_projection_roi_time_sum': 0.0,
        'event_projection_routing_time_sum': 0.0,
        'event_projection_total_ph_directional_time_sum': 0.0,
        'event_projection_oracle_valid_conflicts': 0,
        'event_projection_A_correct_B_wrong': 0,
        'event_projection_A_wrong_B_correct': 0,
        'event_projection_routing_correct': 0,
        'event_projection_routing_wrong': 0,
        'event_projection_false_merge_conflicts': 0,
        'event_projection_false_split_conflicts': 0,
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
        'birth_matching_time_sum': 0.0,
        'candidate_build_time_sum': 0.0,
        'relation_tree_query_time_sum': 0.0,
        'routing_time_sum': 0.0,
        'total_ph_directional_time_sum': 0.0,
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
    }


def compute_ph_directional_cps_losses(
    prediction_a,
    prediction_b,
    dilation=1,
    match_radius=16.0,
    unmatched_mode='vanilla',
    detector_mode='persistent_relation',
    debug_timing=False,
    batch_index=None,
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    presence_conflict_mode='strict',
):
    with torch.no_grad():
        prob_a = torch.softmax(prediction_a.detach(), dim=1)[:, 1]
        prob_b = torch.softmax(prediction_b.detach(), dim=1)[:, 1]
        if (
            detector_mode == 'persistent_relation'
            and correspondence_mode == 'event_projection'
        ):
            pred_a = prediction_a.detach().argmax(dim=1)
            pred_b = prediction_b.detach().argmax(dim=1)
            _ph_mask_np, events, ph_stats = build_event_projection_conflict_events(
                prob_a.cpu().numpy(),
                prob_b.cpu().numpy(),
                pred_a.cpu().numpy(),
                pred_b.cpu().numpy(),
                dilation=dilation,
                debug_timing=debug_timing,
                batch_index=batch_index,
            )
        elif detector_mode == 'persistent_relation':
            _ph_mask_np, events, ph_stats = build_persistent_relation_conflict_events(
                prob_a.cpu().numpy(),
                prob_b.cpu().numpy(),
                dilation=dilation,
                unmatched_mode=unmatched_mode,
                debug_timing=debug_timing,
                batch_index=batch_index,
                correspondence_mode=correspondence_mode,
                relation_backend=relation_backend,
                presence_conflict_mode=presence_conflict_mode,
            )
        else:
            _ph_mask_np, events, ph_stats = build_ph_conflict_events(
                prob_a.cpu().numpy(),
                prob_b.cpu().numpy(),
                dilation=dilation,
                use_matching=True,
                match_radius=match_radius,
                unmatched_mode=unmatched_mode,
            )
        import time
        routing_start = time.perf_counter()
        weight_a_from_b, weight_b_from_a, routing_stats = _build_directional_weights(
            prob_a,
            prob_b,
            events,
        )
        ph_stats['routing_time'] = time.perf_counter() - routing_start
        ph_stats['total_ph_directional_time'] = (
            float(ph_stats.get('total_ph_directional_time', ph_stats.get('total_ph_time', 0.0)))
            + ph_stats['routing_time']
        )

    pseudo_a = prediction_a.detach().argmax(dim=1)
    pseudo_b = prediction_b.detach().argmax(dim=1)
    loss_map_a_from_b = 0.5 * F.cross_entropy(
        prediction_a, pseudo_b, reduction='none'
    )
    loss_map_b_from_a = 0.5 * F.cross_entropy(
        prediction_b, pseudo_a, reduction='none'
    )
    num_pixels = float(prediction_a.shape[0] * prediction_a.shape[2] * prediction_a.shape[3])
    loss_a_from_b = (loss_map_a_from_b * weight_a_from_b).sum() / num_pixels
    loss_b_from_a = (loss_map_b_from_a * weight_b_from_a).sum() / num_pixels
    original_estimate = loss_map_a_from_b.mean() + loss_map_b_from_a.mean()
    total = loss_a_from_b + loss_b_from_a
    stats = {
        **ph_stats,
        **routing_stats,
        'total_unlabeled_pixels': int(num_pixels),
    }
    return {
        '1_to_2': loss_b_from_a,
        '2_to_1': loss_a_from_b,
        'total': total,
        'original_estimate': original_estimate,
        'loss_a_from_b': loss_a_from_b,
        'loss_b_from_a': loss_b_from_a,
        'stats': stats,
    }


def update_ph_directional_loss_state(state, losses):
    stats = losses['stats']
    state['ph_conflict_events'] += int(stats['events'])
    conflict_pixels = int(stats['ph_conflict_pixels'])
    total_pixels = int(stats['total_unlabeled_pixels'])
    state['ph_conflict_pixels'] += conflict_pixels
    state['total_unlabeled_pixels'] += total_pixels
    state['a2b_events'] += int(stats['a2b_events'])
    state['b2a_events'] += int(stats['b2a_events'])
    state['a2b_pixels'] += int(stats['a2b_pixels'])
    state['b2a_pixels'] += int(stats['b2a_pixels'])
    state['reliability_a_sum'] += float(stats['reliability_a_sum'])
    state['reliability_b_sum'] += float(stats['reliability_b_sum'])
    state['abs_reliability_gap_sum'] += float(stats['abs_reliability_gap_sum'])
    state['weight_a2b_sum'] += float(stats['weight_a2b_sum'])
    state['weight_b2a_sum'] += float(stats['weight_b2a_sum'])
    state['event_weight_count'] += int(stats['event_weight_count'])
    for key in (
        'ph_pairs_a',
        'ph_pairs_b',
        'matched_ph_pairs',
        'unmatched_ph_a',
        'unmatched_ph_b',
        'matched_connectivity_conflicts',
        'matched_merge_conflicts',
        'matched_split_conflicts',
        'unmatched_conflict_candidates',
        'directional_events_from_matched_pairs',
        'ph_entities_a',
        'ph_entities_b',
        'raw_ph_branches_a',
        'raw_ph_branches_b',
        'matched_entities',
        'matched_branches',
        'unmatched_entities_a',
        'unmatched_entities_b',
        'unmatched_branches_a',
        'unmatched_branches_b',
        'support_candidate_pairs',
        'direct_merge_edges_a',
        'direct_merge_edges_b',
        'aligned_relation_candidates',
        'connectivity_conflicts',
        'relation_candidates',
        'persistent_connectivity_conflicts',
        'a_separate_b_connected',
        'a_connected_b_separate',
        'existence_conflict_a_only',
        'existence_conflict_b_only',
        'directional_events_from_persistent_conflicts',
        'raw_ph_branches_A',
        'raw_ph_branches_B',
        'merge_events_A',
        'merge_events_B',
        'valid_source_events_A',
        'valid_source_events_B',
        'invalid_source_events_A',
        'invalid_source_events_B',
        'unmatched_target_events_A2B',
        'unmatched_target_events_B2A',
        'queried_events_A2B',
        'queried_events_B2A',
        'same_relation_events',
        'structural_conflict_events',
        'A_separate_B_connected',
        'A_connected_B_separate',
        'conflict_roi_count',
        'num_valid_relation_queries',
        'num_absent_A',
        'num_absent_B',
        'num_connected_A',
        'num_separated_A',
        'num_connected_B',
        'num_separated_B',
        'num_relation_conflicts',
        'num_A_connected_B_separated',
        'num_A_separated_B_connected',
        'valid_compare_count',
        'ph_cc_agree_A',
        'ph_cc_agree_B',
        'num_relation_questions',
        'num_pair_present_both',
        'num_pair_absent_a',
        'num_pair_absent_b',
        'num_pair_absent_both',
        'num_pure_relation_conflicts',
        'num_presence_involved_conflicts',
        'num_total_structural_conflicts',
        'num_presence_disagreement_nonconflict',
        'pure_relation_conflict_pixels',
        'presence_involved_conflict_pixels',
        'old_cc_conflicts',
        'new_conflicts',
        'old_cc_retained_conflicts',
        'old_cc_lost_conflicts',
        'new_extra_conflicts',
        'old_cc_to_pure_relation',
        'old_cc_to_presence_involved',
        'old_cc_to_nonconflict',
        'old_cc_candidate_missing',
    ):
        state[key] += int(stats.get(key, 0))
    for key in (
        'match_score_sum',
        'match_overlap_sum',
        'birth_distance_sum',
        'death_distance_sum',
        'core_iou_sum',
        'centroid_distance_sum',
        'merge_tree_cc_consistency_sum',
        'merge_level_A_sum',
        'merge_level_B_sum',
        'abs_g_A_sum',
        'abs_g_B_sum',
        'conflict_merge_level_A_sum',
        'conflict_merge_level_B_sum',
    ):
        state[key] += float(stats.get(key, 0.0))
    for key in (
        'mean_conflict_roi_pixels_sum',
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
        'routing_time',
        'total_ph_directional_time',
    ):
        if key == 'mean_conflict_roi_pixels_sum':
            state[key] += float(stats.get(key, 0.0))
        else:
            state[key + '_sum'] += float(stats.get(key, 0.0))
    state['mean_conflict_roi_pixels_count'] += int(
        stats.get('mean_conflict_roi_pixels_count', 0)
    )
    state['support_iou_sum'] += float(stats.get('support_iou_sum', 0.0))
    state['support_iou_count'] += int(stats.get('support_iou_count', 0))
    state['support_iou_values'].extend(stats.get('support_iou_values', []))
    state['event_projection_roi_sizes'].extend(stats.get('roi_sizes', []))
    state['event_projection_roi_size_sum'] += float(sum(stats.get('roi_sizes', [])))
    for key in (
        'ph_extract_time',
        'cc_time',
        'event_query_time',
        'roi_time',
        'routing_time',
        'total_ph_directional_time',
    ):
        state['event_projection_' + key + '_sum'] += float(stats.get(key, 0.0))
    state['mean_support_area_a_sum'] += float(stats.get('mean_support_area_a', 0.0))
    state['mean_support_area_b_sum'] += float(stats.get('mean_support_area_b', 0.0))
    state['support_area_batches'] += 1
    state['ph_timing_batches'] += 1
    state['merge_tree_cc_consistency_count'] += int(
        stats.get('merge_tree_cc_consistency_count', 0)
    )
    state['loss_cps_original_estimate_sum'] += float(
        losses['original_estimate'].detach().item()
    )
    state['loss_cps_directional_sum'] += float(losses['total'].detach().item())
    state['loss_a_from_b_sum'] += float(losses['loss_a_from_b'].detach().item())
    state['loss_b_from_a_sum'] += float(losses['loss_b_from_a'].detach().item())
    state['loss_batches'] += 1


def update_ph_directional_vanilla_state(state, cps_losses, prediction_a):
    num_pixels = int(prediction_a.shape[0] * prediction_a.shape[2] * prediction_a.shape[3])
    state['total_unlabeled_pixels'] += num_pixels
    state['loss_cps_original_estimate_sum'] += float(cps_losses['total'].detach().item())
    state['loss_cps_directional_sum'] += float(cps_losses['total'].detach().item())
    state['loss_a_from_b_sum'] += float(cps_losses['2_to_1'].detach().item())
    state['loss_b_from_a_sum'] += float(cps_losses['1_to_2'].detach().item())
    state['loss_batches'] += 1


def finalize_ph_directional_epoch_state(state):
    event_count = state['event_weight_count']
    loss_batches = state['loss_batches']
    total_pixels = state['total_unlabeled_pixels']
    match_denominator = state['matched_ph_pairs'] or state['matched_entities']
    return {
        'epoch': state['epoch'],
        'ph_enabled': state['ph_enabled'],
        'ph_conflict_events': state['ph_conflict_events'],
        'ph_conflict_pixel_ratio': _ratio(state['ph_conflict_pixels'], total_pixels),
        'a2b_events': state['a2b_events'],
        'b2a_events': state['b2a_events'],
        'a2b_pixels': state['a2b_pixels'],
        'b2a_pixels': state['b2a_pixels'],
        'mean_reliability_a': _ratio(state['reliability_a_sum'], event_count),
        'mean_reliability_b': _ratio(state['reliability_b_sum'], event_count),
        'mean_abs_reliability_gap': _ratio(
            state['abs_reliability_gap_sum'], event_count
        ),
        'mean_weight_a2b': _ratio(state['weight_a2b_sum'], event_count),
        'mean_weight_b2a': _ratio(state['weight_b2a_sum'], event_count),
        'loss_cps_original_estimate': _ratio(
            state['loss_cps_original_estimate_sum'], loss_batches
        ),
        'loss_cps_directional': _ratio(
            state['loss_cps_directional_sum'], loss_batches
        ),
        'loss_a_from_b': _ratio(state['loss_a_from_b_sum'], loss_batches),
        'loss_b_from_a': _ratio(state['loss_b_from_a_sum'], loss_batches),
        'effective_a2b_pixel_ratio': _ratio(state['a2b_pixels'], total_pixels),
        'effective_b2a_pixel_ratio': _ratio(state['b2a_pixels'], total_pixels),
        'oracle_routable_events': state['oracle_routable_events'],
        'routing_correct_events': state['routing_correct_events'],
        'routing_wrong_events': state['routing_wrong_events'],
        'routing_accuracy': _ratio(
            state['routing_correct_events'], state['oracle_routable_events']
        ),
        'harmful_routing_events': state['harmful_routing_events'],
        'harmful_routing_rate': _ratio(
            state['harmful_routing_events'], state['oracle_routable_events']
        ),
        'mean_weight_correct_direction': _ratio(
            state['weight_correct_direction_sum'], state['routing_correct_events']
        ),
        'mean_weight_wrong_direction': _ratio(
            state['weight_wrong_direction_sum'], state['routing_wrong_events']
        ),
        'merge_routing_accuracy': _ratio(
            state['merge_routing_correct_events'],
            state['merge_oracle_routable_events'],
        ),
        'split_routing_accuracy': _ratio(
            state['split_routing_correct_events'],
            state['split_oracle_routable_events'],
        ),
        'ph_pairs_a': state['ph_pairs_a'],
        'ph_pairs_b': state['ph_pairs_b'],
        'matched_ph_pairs': state['matched_ph_pairs'],
        'unmatched_ph_a': state['unmatched_ph_a'],
        'unmatched_ph_b': state['unmatched_ph_b'],
        'ph_pair_match_rate': _ratio(
            2.0 * state['matched_ph_pairs'],
            state['ph_pairs_a'] + state['ph_pairs_b'],
        ),
        'matched_connectivity_conflicts': state['matched_connectivity_conflicts'],
        'matched_merge_conflicts': state['matched_merge_conflicts'],
        'matched_split_conflicts': state['matched_split_conflicts'],
        'unmatched_conflict_candidates': state['unmatched_conflict_candidates'],
        'mean_match_score': _ratio(
            state['match_score_sum'], match_denominator
        ),
        'mean_match_overlap': _ratio(
            state['match_overlap_sum'], state['matched_ph_pairs']
        ),
        'mean_birth_distance': _ratio(
            state['birth_distance_sum'], match_denominator
        ),
        'mean_death_distance': _ratio(
            state['death_distance_sum'], state['matched_ph_pairs']
        ),
        'directional_events_from_matched_pairs': state[
            'directional_events_from_matched_pairs'
        ],
        'matched_conflict_oracle_events': state['matched_conflict_oracle_events'],
        'matched_routing_correct': state['matched_routing_correct'],
        'matched_routing_wrong': state['matched_routing_wrong'],
        'matched_routing_accuracy': _ratio(
            state['matched_routing_correct'],
            state['matched_conflict_oracle_events'],
        ),
        'matched_harmful_routing_rate': _ratio(
            state['matched_harmful_routing_events'],
            state['matched_conflict_oracle_events'],
        ),
        'ph_entities_a': state['ph_entities_a'],
        'ph_entities_b': state['ph_entities_b'],
        'raw_ph_branches_a': state['raw_ph_branches_a'],
        'raw_ph_branches_b': state['raw_ph_branches_b'],
        'matched_entities': state['matched_entities'],
        'matched_branches': state['matched_branches'],
        'unmatched_entities_a': state['unmatched_entities_a'],
        'unmatched_entities_b': state['unmatched_entities_b'],
        'unmatched_branches_a': state['unmatched_branches_a'],
        'unmatched_branches_b': state['unmatched_branches_b'],
        'support_candidate_pairs': state['support_candidate_pairs'],
        'mean_birth_match_distance': _ratio(
            state['birth_distance_sum'], state['matched_entities']
        ),
        'mean_matched_support_iou': _ratio(
            state['support_iou_sum'], state['support_iou_count']
        ),
        'median_matched_support_iou': (
            float(np.median(state['support_iou_values']))
            if state['support_iou_values'] else 0.0
        ),
        'mean_matched_birth_distance': _ratio(
            state['birth_distance_sum'], state['matched_entities']
        ),
        'mean_support_area_a': _ratio(
            state['mean_support_area_a_sum'], state['support_area_batches']
        ),
        'mean_support_area_b': _ratio(
            state['mean_support_area_b_sum'], state['support_area_batches']
        ),
        'direct_merge_edges_a': state['direct_merge_edges_a'],
        'direct_merge_edges_b': state['direct_merge_edges_b'],
        'aligned_relation_candidates': state['aligned_relation_candidates'],
        'connectivity_conflicts': state['connectivity_conflicts'],
        'mean_conflict_roi_pixels': _ratio(
            state['mean_conflict_roi_pixels_sum'],
            state['mean_conflict_roi_pixels_count'],
        ),
        'ph_extract_time': _ratio(
            state['ph_extract_time_sum'], state['ph_timing_batches']
        ),
        'support_build_time': _ratio(
            state['support_build_time_sum'], state['ph_timing_batches']
        ),
        'support_overlap_time': _ratio(
            state['support_overlap_time_sum'], state['ph_timing_batches']
        ),
        'entity_match_time': _ratio(
            state['entity_match_time_sum'], state['ph_timing_batches']
        ),
        'relation_time': _ratio(
            state['relation_time_sum'], state['ph_timing_batches']
        ),
        'roi_time': _ratio(
            state['roi_time_sum'], state['ph_timing_batches']
        ),
        'total_ph_time': _ratio(
            state['total_ph_time_sum'], state['ph_timing_batches']
        ),
        'entity_match_rate': _ratio(
            2.0 * state['matched_entities'],
            state['ph_entities_a'] + state['ph_entities_b'],
        ),
        'mean_core_iou': _ratio(state['core_iou_sum'], state['matched_entities']),
        'mean_centroid_distance': _ratio(
            state['centroid_distance_sum'], state['matched_entities']
        ),
        'relation_candidates': state['relation_candidates'],
        'persistent_connectivity_conflicts': state[
            'persistent_connectivity_conflicts'
        ],
        'a_separate_b_connected': state['a_separate_b_connected'],
        'a_connected_b_separate': state['a_connected_b_separate'],
        'existence_conflict_a_only': state['existence_conflict_a_only'],
        'existence_conflict_b_only': state['existence_conflict_b_only'],
        'merge_tree_cc_consistency_rate': _ratio(
            state['merge_tree_cc_consistency_sum'],
            state['merge_tree_cc_consistency_count'],
        ),
        'conflict_pixel_ratio': _ratio(state['ph_conflict_pixels'], total_pixels),
        'directional_events_from_persistent_conflicts': state[
            'directional_events_from_persistent_conflicts'
        ],
        'entity_match_oracle_pairs': state['entity_match_oracle_pairs'],
        'entity_match_correct_pairs': state['entity_match_correct_pairs'],
        'entity_match_precision': _ratio(
            state['entity_match_correct_pairs'], state['entity_match_oracle_pairs']
        ),
        'oracle_single_branch_correct_conflicts': state[
            'oracle_single_branch_correct_conflicts'
        ],
        'oracle_both_wrong_conflicts': state['oracle_both_wrong_conflicts'],
        'oracle_both_correct_conflicts': state['oracle_both_correct_conflicts'],
        'single_branch_correct_conflict_rate': _ratio(
            state['oracle_single_branch_correct_conflicts'],
            (
                state['oracle_single_branch_correct_conflicts']
                + state['oracle_both_wrong_conflicts']
                + state['oracle_both_correct_conflicts']
            ),
        ),
    }


def correspondence_diagnostics_row(ph_state):
    row = finalize_ph_directional_epoch_state(ph_state)
    return {
        'epoch': row['epoch'],
        'raw_ph_branches_a': row['raw_ph_branches_a'],
        'raw_ph_branches_b': row['raw_ph_branches_b'],
        'support_candidate_pairs': row['support_candidate_pairs'],
        'matched_branches': row['matched_branches'],
        'unmatched_a': row['unmatched_branches_a'],
        'unmatched_b': row['unmatched_branches_b'],
        'mean_support_iou': row['mean_matched_support_iou'],
        'median_support_iou': row['median_matched_support_iou'],
        'mean_birth_distance': row['mean_matched_birth_distance'],
        'mean_support_area_a': row['mean_support_area_a'],
        'mean_support_area_b': row['mean_support_area_b'],
        'direct_edges_a': row['direct_merge_edges_a'],
        'direct_edges_b': row['direct_merge_edges_b'],
        'aligned_relation_candidates': row['aligned_relation_candidates'],
        'connectivity_conflicts': row['connectivity_conflicts'],
        'conflict_pixel_ratio': row['conflict_pixel_ratio'],
        'ph_time': row['ph_extract_time'],
        'support_time': row['support_build_time'],
        'support_overlap_time': row['support_overlap_time'],
        'matching_time': row['entity_match_time'],
        'relation_time': row['relation_time'],
        'roi_time': row['roi_time'],
        'total_time': row['total_ph_time'],
        'matched_same_gt_structure': ph_state.get('entity_match_correct_pairs', 0),
        'matched_different_gt_structure': (
            ph_state.get('entity_match_oracle_pairs', 0)
            - ph_state.get('entity_match_correct_pairs', 0)
        ),
        'entity_match_precision': row['entity_match_precision'],
        'entity_match_recall': ph_state.get('entity_match_recall', 0.0),
        'mean_ph_branches_per_gt_gland_A': ph_state.get('mean_ph_branches_per_gt_gland_A', 0.0),
        'mean_ph_branches_per_gt_gland_B': ph_state.get('mean_ph_branches_per_gt_gland_B', 0.0),
        'median_ph_branches_per_gt_gland_A': ph_state.get('median_ph_branches_per_gt_gland_A', 0.0),
        'median_ph_branches_per_gt_gland_B': ph_state.get('median_ph_branches_per_gt_gland_B', 0.0),
        'p90_ph_branches_per_gt_gland_A': ph_state.get('p90_ph_branches_per_gt_gland_A', 0.0),
        'p90_ph_branches_per_gt_gland_B': ph_state.get('p90_ph_branches_per_gt_gland_B', 0.0),
        'conflict_both_correct': ph_state.get('conflict_both_correct', 0),
        'conflict_A_only_correct': ph_state.get('conflict_A_only_correct', 0),
        'conflict_B_only_correct': ph_state.get('conflict_B_only_correct', 0),
        'conflict_both_wrong': ph_state.get('conflict_both_wrong', 0),
        'complementary_conflict_rate': ph_state.get('complementary_conflict_rate', 0.0),
    }


def relation_tree_diagnostics_row(ph_state):
    row = finalize_ph_directional_epoch_state(ph_state)
    valid_queries = ph_state.get('num_valid_relation_queries', 0)
    conflicts = ph_state.get('num_relation_conflicts', 0)
    valid_compare = ph_state.get('valid_compare_count', 0)
    return {
        'epoch': row['epoch'],
        'num_branches_A': ph_state.get('raw_ph_branches_a', 0),
        'num_branches_B': ph_state.get('raw_ph_branches_b', 0),
        'num_direct_edges_A': ph_state.get('direct_merge_edges_a', 0),
        'num_direct_edges_B': ph_state.get('direct_merge_edges_b', 0),
        'num_matched_branches': ph_state.get('matched_branches', 0),
        'num_relation_candidates': ph_state.get('relation_candidates', 0),
        'num_valid_relation_queries': valid_queries,
        'num_absent_A': ph_state.get('num_absent_A', 0),
        'num_absent_B': ph_state.get('num_absent_B', 0),
        'num_connected_A': ph_state.get('num_connected_A', 0),
        'num_separated_A': ph_state.get('num_separated_A', 0),
        'num_connected_B': ph_state.get('num_connected_B', 0),
        'num_separated_B': ph_state.get('num_separated_B', 0),
        'num_relation_conflicts': conflicts,
        'num_A_connected_B_separated': ph_state.get('num_A_connected_B_separated', 0),
        'num_A_separated_B_connected': ph_state.get('num_A_separated_B_connected', 0),
        'mean_merge_level_A': _ratio(ph_state.get('merge_level_A_sum', 0.0), valid_queries),
        'mean_merge_level_B': _ratio(ph_state.get('merge_level_B_sum', 0.0), valid_queries),
        'mean_abs_g_A': _ratio(ph_state.get('abs_g_A_sum', 0.0), valid_queries),
        'mean_abs_g_B': _ratio(ph_state.get('abs_g_B_sum', 0.0), valid_queries),
        'mean_conflict_merge_level_A': _ratio(
            ph_state.get('conflict_merge_level_A_sum', 0.0), conflicts
        ),
        'mean_conflict_merge_level_B': _ratio(
            ph_state.get('conflict_merge_level_B_sum', 0.0), conflicts
        ),
        'num_roi': ph_state.get('mean_conflict_roi_pixels_count', 0),
        'conflict_pixel_ratio': row['conflict_pixel_ratio'],
        'valid_compare_count': valid_compare,
        'ph_cc_agree_A': ph_state.get('ph_cc_agree_A', 0),
        'ph_cc_agree_B': ph_state.get('ph_cc_agree_B', 0),
        'ph_cc_agreement_rate_A': _ratio(ph_state.get('ph_cc_agree_A', 0), valid_compare),
        'ph_cc_agreement_rate_B': _ratio(ph_state.get('ph_cc_agree_B', 0), valid_compare),
        'ph_extract': _ratio(ph_state.get('ph_extract_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'birth_matching': _ratio(ph_state.get('birth_matching_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'candidate_build': _ratio(ph_state.get('candidate_build_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'relation_tree_query': _ratio(ph_state.get('relation_tree_query_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'roi': _ratio(ph_state.get('roi_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'routing': _ratio(ph_state.get('routing_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
        'total': _ratio(ph_state.get('total_ph_directional_time_sum', 0.0), ph_state.get('ph_timing_batches', 0)),
    }


def presence_conflict_diagnostics_row(ph_state):
    total_pixels = ph_state.get('total_unlabeled_pixels', 0)
    old_cc_conflicts = ph_state.get('old_cc_conflicts', 0)
    new_conflicts = ph_state.get('new_conflicts', 0)
    retained = ph_state.get('old_cc_retained_conflicts', 0)
    union = old_cc_conflicts + new_conflicts - retained
    return {
        'epoch': ph_state['epoch'],
        'num_relation_questions': ph_state.get('num_relation_questions', 0),
        'num_pair_present_both': ph_state.get('num_pair_present_both', 0),
        'num_pair_absent_a': ph_state.get('num_pair_absent_a', 0),
        'num_pair_absent_b': ph_state.get('num_pair_absent_b', 0),
        'num_pair_absent_both': ph_state.get('num_pair_absent_both', 0),
        'num_pure_relation_conflicts': ph_state.get('num_pure_relation_conflicts', 0),
        'num_presence_involved_conflicts': ph_state.get('num_presence_involved_conflicts', 0),
        'num_total_structural_conflicts': ph_state.get('num_total_structural_conflicts', 0),
        'num_presence_disagreement_nonconflict': ph_state.get('num_presence_disagreement_nonconflict', 0),
        'pure_relation_conflict_pixel_ratio': _ratio(
            ph_state.get('pure_relation_conflict_pixels', 0), total_pixels
        ),
        'presence_involved_conflict_pixel_ratio': _ratio(
            ph_state.get('presence_involved_conflict_pixels', 0), total_pixels
        ),
        'total_structural_conflict_pixel_ratio': _ratio(
            ph_state.get('ph_conflict_pixels', 0), total_pixels
        ),
        'a_connected_b_not_count': ph_state.get('num_A_connected_B_separated', 0),
        'b_connected_a_not_count': ph_state.get('num_A_separated_B_connected', 0),
        'old_cc_conflicts': old_cc_conflicts,
        'new_conflicts': new_conflicts,
        'old_cc_retained_conflicts': retained,
        'old_cc_lost_conflicts': ph_state.get('old_cc_lost_conflicts', 0),
        'new_extra_conflicts': ph_state.get('new_extra_conflicts', 0),
        'old_cc_recall': _ratio(retained, old_cc_conflicts),
        'new_vs_old_jaccard': _ratio(retained, union),
        'old_cc_to_pure_relation': ph_state.get('old_cc_to_pure_relation', 0),
        'old_cc_to_presence_involved': ph_state.get('old_cc_to_presence_involved', 0),
        'old_cc_to_nonconflict': ph_state.get('old_cc_to_nonconflict', 0),
        'old_cc_candidate_missing': ph_state.get('old_cc_candidate_missing', 0),
    }


@torch.no_grad()
def update_ph_directional_oracle_state(
    state,
    logits_a,
    logits_b,
    gt,
    min_area=20,
    match_iou=0.1,
    dilation=1,
    match_radius=16.0,
    unmatched_mode='vanilla',
    detector_mode='persistent_relation',
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    presence_conflict_mode='strict',
):
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    if detector_mode == 'persistent_relation' and correspondence_mode == 'event_projection':
        pred_a = logits_a.detach().argmax(dim=1)
        pred_b = logits_b.detach().argmax(dim=1)
        _mask, events, _stats = build_event_projection_conflict_events(
            prob_a.cpu().numpy(),
            prob_b.cpu().numpy(),
            pred_a.cpu().numpy(),
            pred_b.cpu().numpy(),
            dilation=dilation,
        )
        weights = _event_routing_weights(prob_a, prob_b, events)
        update_event_projection_oracle_state(state, prob_a, prob_b, gt, events, weights)
        return
    if detector_mode == 'persistent_relation':
        _mask, events, _stats = build_persistent_relation_conflict_events(
            prob_a.cpu().numpy(),
            prob_b.cpu().numpy(),
            dilation=dilation,
            unmatched_mode=unmatched_mode,
            correspondence_mode=correspondence_mode,
            relation_backend=relation_backend,
            collect_cc_agreement=(relation_backend == 'ph_tree'),
            presence_conflict_mode=presence_conflict_mode,
        )
        if relation_backend == 'ph_tree':
            for key in (
                'valid_compare_count',
                'ph_cc_agree_A',
                'ph_cc_agree_B',
                'old_cc_conflicts',
                'new_conflicts',
                'old_cc_retained_conflicts',
                'old_cc_lost_conflicts',
                'new_extra_conflicts',
                'old_cc_to_pure_relation',
                'old_cc_to_presence_involved',
                'old_cc_to_nonconflict',
                'old_cc_candidate_missing',
            ):
                state[key] += int(_stats.get(key, 0))
        _update_persistent_relation_oracle_state(
            state,
            prob_a,
            prob_b,
            gt,
            events,
        )
        return
    _mask, events, _stats = build_ph_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        use_matching=False,
    )
    _update_oracle_from_events(
        state,
        prob_a,
        prob_b,
        gt,
        events,
        min_area,
        match_iou,
        prefix='',
    )
    _matched_mask, matched_events, _matched_stats = build_ph_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        use_matching=True,
        match_radius=match_radius,
        unmatched_mode=unmatched_mode,
    )
    _update_oracle_from_events(
        state,
        prob_a,
        prob_b,
        gt,
        [event for event in matched_events if event.get('matched_pair', False)],
        min_area,
        match_iou,
        prefix='matched_',
    )


@torch.no_grad()
def save_ph_match_examples(
    output_dir,
    epoch,
    logits_a,
    logits_b,
    gt,
    images_a=None,
    images_b=None,
    dilation=1,
    match_radius=16.0,
    unmatched_mode='vanilla',
    min_area=20,
    match_iou=0.1,
    detector_mode='persistent_relation',
    max_examples=3,
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    presence_conflict_mode='strict',
):
    if images_a is None or images_b is None:
        return
    output_path = Path(output_dir) / 'ph_match_examples' / 'epoch_{:03d}'.format(epoch)
    output_path.mkdir(parents=True, exist_ok=True)
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    if detector_mode == 'persistent_relation':
        _mask, events, _stats = build_persistent_relation_conflict_events(
            prob_a.cpu().numpy(),
            prob_b.cpu().numpy(),
            dilation=dilation,
            unmatched_mode=unmatched_mode,
            correspondence_mode=correspondence_mode,
            relation_backend=relation_backend,
            presence_conflict_mode=presence_conflict_mode,
        )
    else:
        _mask, events, _stats = build_ph_conflict_events(
            prob_a.cpu().numpy(),
            prob_b.cpu().numpy(),
            dilation=dilation,
            use_matching=True,
            match_radius=match_radius,
            unmatched_mode=unmatched_mode,
        )
    weights = _event_routing_weights(prob_a, prob_b, events)
    pred_a = prob_a.cpu().numpy() > 0.5
    pred_b = prob_b.cpu().numpy() > 0.5
    gt_np = gt.detach().cpu().numpy() > 0
    saved = 0
    per_sample = {}
    for event_index, event in enumerate(events):
        if saved >= max_examples:
            break
        if not event.get('matched_pair', False):
            continue
        sample = int(event['sample_index'])
        if sample not in per_sample:
            per_sample[sample] = _structural_correct_sets(
                pred_a[sample],
                pred_b[sample],
                gt_np[sample],
                min_area=min_area,
                match_iou=match_iou,
            )
        oracle = _oracle_direction_for_event(event, per_sample[sample])
        routing, routing_weight = _predicted_direction(weights[id(event)])
        canvas = _make_match_canvas(
            images_a[sample],
            images_b[sample],
            pred_a[sample],
            pred_b[sample],
            event,
            routing,
            routing_weight,
            oracle,
        )
        canvas.save(output_path / 'match_{:03d}.png'.format(event_index))
        saved += 1


@torch.no_grad()
def write_persistent_relation_event_rows(
    writer,
    epoch,
    image_ids,
    logits_a,
    logits_b,
    gt,
    dilation=1,
    unmatched_mode='vanilla',
    correspondence_mode='birth_nearest',
    relation_backend='cc',
    presence_conflict_mode='strict',
):
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    _mask, events, _stats = build_persistent_relation_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        unmatched_mode=unmatched_mode,
        correspondence_mode=correspondence_mode,
        relation_backend=relation_backend,
        collect_cc_agreement=(relation_backend == 'ph_tree'),
        presence_conflict_mode=presence_conflict_mode,
    )
    if not events:
        return
    weights = _event_routing_weights(prob_a, prob_b, events)
    gt_np = gt.detach().cpu().numpy() > 0
    gt_labels_by_sample = {}
    for event in events:
        sample = int(event['sample_index'])
        if sample not in gt_labels_by_sample:
            gt_labels_by_sample[sample], _count = _label_numpy(gt_np[sample])
        gt_labels = gt_labels_by_sample[sample]
        oracle = _persistent_event_oracle_direction(event, gt_labels)
        routing, routing_weight = _predicted_direction(weights[id(event)])
        writer.writerow(_persistent_event_row(
            epoch,
            image_ids[sample],
            event,
            weights[id(event)],
            routing,
            routing_weight,
            oracle,
        ))


@torch.no_grad()
def write_ph_relation_tree_event_rows(
    writer,
    epoch,
    image_ids,
    logits_a,
    logits_b,
    dilation=1,
    unmatched_mode='vanilla',
    correspondence_mode='birth_nearest',
    presence_conflict_mode='strict',
):
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    _mask, events, _stats = build_persistent_relation_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        unmatched_mode=unmatched_mode,
        correspondence_mode=correspondence_mode,
        relation_backend='ph_tree',
        collect_cc_agreement=True,
        presence_conflict_mode=presence_conflict_mode,
    )
    if not events:
        return
    weights = _event_routing_weights(prob_a, prob_b, events)
    for event in events:
        sample = int(event['sample_index'])
        routing, routing_weight = _predicted_direction(weights[id(event)])
        writer.writerow(_ph_relation_tree_event_row(
            epoch,
            image_ids[sample],
            event,
            routing,
            routing_weight,
        ))


@torch.no_grad()
def write_ph_presence_conflict_event_rows(
    writer,
    epoch,
    image_ids,
    logits_a,
    logits_b,
    dilation=1,
    unmatched_mode='vanilla',
    correspondence_mode='birth_nearest',
    presence_conflict_mode='strict',
):
    prob_a = torch.softmax(logits_a.detach(), dim=1)[:, 1]
    prob_b = torch.softmax(logits_b.detach(), dim=1)[:, 1]
    _mask, events, _stats = build_persistent_relation_conflict_events(
        prob_a.cpu().numpy(),
        prob_b.cpu().numpy(),
        dilation=dilation,
        unmatched_mode=unmatched_mode,
        correspondence_mode=correspondence_mode,
        relation_backend='ph_tree',
        collect_cc_agreement=True,
        presence_conflict_mode=presence_conflict_mode,
    )
    for event in events:
        sample = int(event['sample_index'])
        writer.writerow(_ph_presence_conflict_event_row(
            epoch,
            image_ids[sample],
            event,
        ))


def _ph_presence_conflict_event_row(epoch, image_id, event):
    left = event['left_match']
    right = event['right_match']
    left_a = left['entity_a']
    left_b = left['entity_b']
    right_a = right['entity_a']
    right_b = right['entity_b']
    query_a = event.get('relation_query_a')
    query_b = event.get('relation_query_b')
    saddle_y, saddle_x = event['saddle_coord_a'] if event['connected_a'] else event['saddle_coord_b']
    return {
        'epoch': epoch,
        'sample_id': image_id,
        'candidate_id': event.get('candidate_id', ''),
        'source_event_side': event.get('relation_source', ''),
        'a_u_id': left_a.entity_id,
        'a_v_id': right_a.entity_id,
        'b_u_id': left_b.entity_id,
        'b_v_id': right_b.entity_id,
        'birth_a_u': left_a.birth,
        'birth_a_v': right_a.birth,
        'birth_b_u': left_b.birth,
        'birth_b_v': right_b.birth,
        'pair_present_a': int(event.get('pair_present_a', False)),
        'pair_present_b': int(event.get('pair_present_b', False)),
        'merge_level_a': event['mu_a'],
        'merge_level_b': event['mu_b'],
        'connected_a': int(event['connected_a']),
        'connected_b': int(event['connected_b']),
        'semantic_state_a': query_a.state if query_a is not None else '',
        'semantic_state_b': query_b.state if query_b is not None else '',
        'conflict_type': event.get('structural_conflict_type', ''),
        'connected_side': event.get('connected_side', ''),
        'saddle_x': saddle_x,
        'saddle_y': saddle_y,
        'roi_pixels': event['roi_size'],
    }


def _ph_relation_tree_event_row(epoch, image_id, event, routing, routing_weight):
    left = event['left_match']
    right = event['right_match']
    left_a = left['entity_a']
    left_b = left['entity_b']
    right_a = right['entity_a']
    right_b = right['entity_b']
    query_a = event.get('relation_query_a')
    query_b = event.get('relation_query_b')
    saddle_y, saddle_x = event['saddle_coord_a'] if event['connected_a'] else event['saddle_coord_b']
    merge_a = float(event['mu_a'])
    merge_b = float(event['mu_b'])
    return {
        'epoch': epoch,
        'sample_id': image_id,
        'source': event.get('relation_source', ''),
        'A_branch_u': left_a.entity_id,
        'A_branch_v': right_a.entity_id,
        'B_branch_u': left_b.entity_id,
        'B_branch_v': right_b.entity_id,
        'birth_u_A': left_a.birth,
        'birth_v_A': right_a.birth,
        'birth_u_B': left_b.birth,
        'birth_v_B': right_b.birth,
        'merge_level_A': merge_a,
        'merge_level_B': merge_b,
        'g_A': merge_a - 0.5,
        'g_B': merge_b - 0.5,
        'state_A': query_a.state if query_a is not None else '',
        'state_B': query_b.state if query_b is not None else '',
        'saddle_row': saddle_y,
        'saddle_col': saddle_x,
        'routing_direction': routing,
        'routing_weight': routing_weight,
        'roi_pixels': event['roi_size'],
    }


def _persistent_event_row(epoch, image_id, event, weights, routing, routing_weight, oracle):
    left = event['left_match']
    right = event['right_match']
    left_a = left['entity_a']
    left_b = left['entity_b']
    right_a = right['entity_a']
    right_b = right['entity_b']
    saddle_a_y, saddle_a_x = event['saddle_coord_a']
    saddle_b_y, saddle_b_x = event['saddle_coord_b']
    return {
        'epoch': epoch,
        'image_id': image_id,
        'entity_i_a': left_a.entity_id,
        'entity_i_b': left_b.entity_id,
        'entity_j_a': right_a.entity_id,
        'entity_j_b': right_b.entity_id,
        'core_iou_i': left['core_iou'],
        'core_iou_j': right['core_iou'],
        'birth_i_a': left_a.birth,
        'birth_i_b': left_b.birth,
        'birth_j_a': right_a.birth,
        'birth_j_b': right_b.birth,
        'mu_a': event['mu_a'],
        'mu_b': event['mu_b'],
        'connected_a': int(event['connected_a']),
        'connected_b': int(event['connected_b']),
        'saddle_x_a': saddle_a_x,
        'saddle_y_a': saddle_a_y,
        'saddle_x_b': saddle_b_x,
        'saddle_y_b': saddle_b_y,
        'conflict_type': event['conflict_type'],
        'roi_pixels': event['roi_size'],
        'reliability_a': weights['reliability_a'],
        'reliability_b': weights['reliability_b'],
        'routing_direction': routing,
        'routing_weight': routing_weight,
        'oracle_direction': oracle or '',
        'routing_correct': int(oracle is not None and routing == oracle),
    }


def _persistent_event_oracle_direction(event, gt_labels):
    left_gt = _entity_gt_label(event['left_match']['entity_a'], gt_labels)
    right_gt = _entity_gt_label(event['right_match']['entity_a'], gt_labels)
    gt_connected = left_gt > 0 and left_gt == right_gt
    a_correct = bool(event['connected_a']) == gt_connected
    b_correct = bool(event['connected_b']) == gt_connected
    if a_correct and not b_correct:
        return 'a2b'
    if b_correct and not a_correct:
        return 'b2a'
    return None


def _make_match_canvas(image_a, image_b, pred_a, pred_b, event, routing, routing_weight, oracle):
    panels = [
        _tensor_to_image(image_a),
        _tensor_to_image(image_b),
        _mask_to_image(pred_a),
        _mask_to_image(pred_b),
        _match_panel(pred_a.shape, event),
    ]
    width, height = panels[0].size
    label_height = 42
    canvas = Image.new('RGB', (width * len(panels), height + label_height), 'white')
    draw = ImageDraw.Draw(canvas)
    labels = ('RGB', 'Stain', 'Pred A', 'Pred B', 'PH Match')
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
        draw.text((index * width + 4, height + 4), labels[index], fill=(0, 0, 0))
    text = (
        'score={:.3f} dist={:.3f} birthA={:.3f} birthB={:.3f} route={}({:.3f}) oracle={}'
    ).format(
        float(_event_match_score(event)),
        float(_event_match_overlap(event)),
        float(_event_pair_like(event, 'a').get('birth', 0.0)),
        float(_event_pair_like(event, 'b').get('birth', 0.0)),
        routing,
        float(routing_weight),
        oracle or 'none',
    )
    draw.text((4, height + 22), text[:120], fill=(0, 0, 0))
    return canvas


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


def _match_panel(shape, event):
    height, width = shape
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    y_slice, x_slice = event['roi_slice']
    roi = event['roi']
    ys, xs = np.nonzero(roi)
    if ys.size:
        box = (
            int(x_slice.start + xs.min()),
            int(y_slice.start + ys.min()),
            int(x_slice.start + xs.max()),
            int(y_slice.start + ys.max()),
        )
        draw.rectangle(box, outline=(255, 0, 0), width=2)
    pair_a = event.get('pair_a', {})
    pair_b = event.get('pair_b', {})
    if not pair_a:
        pair_a = _event_pair_like(event, 'a')
    if not pair_b:
        pair_b = _event_pair_like(event, 'b')
    for pair, color in ((pair_a, (0, 120, 255)), (pair_b, (255, 120, 0))):
        if 'birth_coord' in pair:
            _draw_point(draw, pair['birth_coord'], color)
        if 'death_coord' in pair:
            _draw_point(draw, pair['death_coord'], (255, 0, 0))
    if 'birth_coord' in pair_a and 'birth_coord' in pair_b:
        ay, ax = pair_a['birth_coord']
        by, bx = pair_b['birth_coord']
        draw.line((ax, ay, bx, by), fill=(0, 160, 0), width=1)
    return image


def _event_match_score(event):
    if 'match_score' in event:
        return event['match_score']
    if 'left_match' in event and 'right_match' in event:
        return 0.5 * (event['left_match']['score'] + event['right_match']['score'])
    return 0.0


def _event_match_overlap(event):
    if 'match_overlap' in event:
        return event['match_overlap']
    if 'left_match' in event and 'right_match' in event:
        return 0.5 * (event['left_match']['core_iou'] + event['right_match']['core_iou'])
    return 0.0


def _event_pair_like(event, branch):
    if 'left_match' not in event:
        return {}
    entity = event['left_match']['entity_a' if branch == 'a' else 'entity_b']
    return {
        'birth_coord': entity.birth_coord,
        'death_coord': entity.death_coord,
        'birth': entity.birth,
        'death': entity.death,
    }


def _draw_point(draw, coord, color):
    y, x = coord
    radius = 3
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def _update_oracle_from_events(
    state,
    prob_a,
    prob_b,
    gt,
    events,
    min_area,
    match_iou,
    prefix='',
):
    if not events:
        return
    weights = _event_routing_weights(prob_a, prob_b, events)
    pred_a = prob_a.cpu().numpy() > 0.5
    pred_b = prob_b.cpu().numpy() > 0.5
    gt_np = gt.detach().cpu().numpy() > 0

    per_sample = {}
    for event in events:
        index = int(event['sample_index'])
        if index not in per_sample:
            per_sample[index] = _structural_correct_sets(
                pred_a[index],
                pred_b[index],
                gt_np[index],
                min_area,
                match_iou,
            )
        oracle_direction = _oracle_direction_for_event(
            event,
            per_sample[index],
        )
        if oracle_direction is None:
            continue
        event_type = event['event_type']
        predicted_direction, predicted_weight = _predicted_direction(
            weights[id(event)]
        )
        if prefix == 'matched_':
            state['matched_conflict_oracle_events'] += 1
        else:
            state['oracle_routable_events'] += 1
            if event_type == 'merge':
                state['merge_oracle_routable_events'] += 1
            elif event_type == 'split':
                state['split_oracle_routable_events'] += 1
        if predicted_direction == oracle_direction:
            if prefix == 'matched_':
                state['matched_routing_correct'] += 1
            else:
                state['routing_correct_events'] += 1
                state['weight_correct_direction_sum'] += predicted_weight
                if event_type == 'merge':
                    state['merge_routing_correct_events'] += 1
                elif event_type == 'split':
                    state['split_routing_correct_events'] += 1
        else:
            if prefix == 'matched_':
                state['matched_routing_wrong'] += 1
            else:
                state['routing_wrong_events'] += 1
            if predicted_direction != 'abstain':
                if prefix == 'matched_':
                    state['matched_harmful_routing_events'] += 1
                else:
                    state['harmful_routing_events'] += 1
            if prefix != 'matched_':
                state['weight_wrong_direction_sum'] += predicted_weight


def _update_persistent_relation_oracle_state(state, prob_a, prob_b, gt, events):
    if not events:
        return
    weights = _event_routing_weights(prob_a, prob_b, events)
    gt_np = gt.detach().cpu().numpy() > 0
    seen_matches = set()
    for event in events:
        sample = int(event['sample_index'])
        gt_labels, _count = _label_numpy(gt_np[sample])
        left_match = event['left_match']
        right_match = event['right_match']
        for match in (left_match, right_match):
            key = (
                sample,
                match['entity_a'].entity_id,
                match['entity_b'].entity_id,
            )
            if key in seen_matches:
                continue
            seen_matches.add(key)
            state['entity_match_oracle_pairs'] += 1
            if _same_gt_identity(match['entity_a'], match['entity_b'], gt_labels):
                state['entity_match_correct_pairs'] += 1

        left_gt = _entity_gt_label(left_match['entity_a'], gt_labels)
        right_gt = _entity_gt_label(right_match['entity_a'], gt_labels)
        gt_connected = left_gt > 0 and left_gt == right_gt
        a_correct = bool(event['connected_a']) == gt_connected
        b_correct = bool(event['connected_b']) == gt_connected
        if a_correct and not b_correct:
            oracle_direction = 'a2b'
            state['oracle_single_branch_correct_conflicts'] += 1
            state['conflict_A_only_correct'] += 1
        elif b_correct and not a_correct:
            oracle_direction = 'b2a'
            state['oracle_single_branch_correct_conflicts'] += 1
            state['conflict_B_only_correct'] += 1
        elif a_correct and b_correct:
            oracle_direction = None
            state['oracle_both_correct_conflicts'] += 1
            state['conflict_both_correct'] += 1
        else:
            oracle_direction = None
            state['oracle_both_wrong_conflicts'] += 1
            state['conflict_both_wrong'] += 1
        if oracle_direction is None:
            continue
        predicted_direction, _predicted_weight = _predicted_direction(
            weights[id(event)]
        )
        state['matched_conflict_oracle_events'] += 1
        if predicted_direction == oracle_direction:
            state['matched_routing_correct'] += 1
        else:
            state['matched_routing_wrong'] += 1
            if predicted_direction != 'abstain':
                state['matched_harmful_routing_events'] += 1
    total = (
        state['conflict_both_correct']
        + state['conflict_A_only_correct']
        + state['conflict_B_only_correct']
        + state['conflict_both_wrong']
    )
    state['complementary_conflict_rate'] = (
        (state['conflict_A_only_correct'] + state['conflict_B_only_correct']) / total
        if total else 0.0
    )


def _label_numpy(mask):
    from scipy.ndimage import label
    return label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def _same_gt_identity(entity_a, entity_b, gt_labels):
    label_a = _entity_gt_label(entity_a, gt_labels)
    label_b = _entity_gt_label(entity_b, gt_labels)
    return label_a > 0 and label_a == label_b


def _entity_gt_label(entity, gt_labels):
    y, x = entity.birth_coord
    return int(gt_labels[y, x])


def _build_directional_weights(prob_a, prob_b, events):
    shape = prob_a.shape
    device = prob_a.device
    weight_a_from_b = torch.ones(shape, dtype=torch.float32, device=device)
    weight_b_from_a = torch.ones(shape, dtype=torch.float32, device=device)
    a2b_strength = torch.zeros(shape, dtype=torch.float32, device=device)
    b2a_strength = torch.zeros(shape, dtype=torch.float32, device=device)
    conflict_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    event_weights = _event_routing_weights(prob_a, prob_b, events)
    stats = {
        'a2b_events': 0,
        'b2a_events': 0,
        'reliability_a_sum': 0.0,
        'reliability_b_sum': 0.0,
        'abs_reliability_gap_sum': 0.0,
        'weight_a2b_sum': 0.0,
        'weight_b2a_sum': 0.0,
        'event_weight_count': 0,
    }

    for event in events:
        values = event_weights[id(event)]
        sample = int(event['sample_index'])
        roi_slice = event['roi_slice']
        roi = torch.from_numpy(event['roi']).to(device=device, dtype=torch.bool)
        y_slice, x_slice = roi_slice
        event_a2b = float(values['w_a2b'])
        event_b2a = float(values['w_b2a'])
        if not event.get('force_abstain', False):
            stats['reliability_a_sum'] += float(values['reliability_a'])
            stats['reliability_b_sum'] += float(values['reliability_b'])
            stats['abs_reliability_gap_sum'] += abs(
                float(values['reliability_a']) - float(values['reliability_b'])
            )
            stats['weight_a2b_sum'] += event_a2b
            stats['weight_b2a_sum'] += event_b2a
            stats['event_weight_count'] += 1
            if event_a2b > 0.0:
                stats['a2b_events'] += 1
            if event_b2a > 0.0:
                stats['b2a_events'] += 1

        local_conflict = conflict_mask[sample, y_slice, x_slice]
        local_conflict |= roi
        conflict_mask[sample, y_slice, x_slice] = local_conflict
        if event_a2b > 0.0:
            local = a2b_strength[sample, y_slice, x_slice]
            a2b_strength[sample, y_slice, x_slice] = torch.where(
                roi,
                torch.maximum(local, torch.full_like(local, event_a2b)),
                local,
            )
        if event_b2a > 0.0:
            local = b2a_strength[sample, y_slice, x_slice]
            b2a_strength[sample, y_slice, x_slice] = torch.where(
                roi,
                torch.maximum(local, torch.full_like(local, event_b2a)),
                local,
            )

    a2b_pixels = a2b_strength > b2a_strength
    b2a_pixels = b2a_strength > a2b_strength
    weight_a_from_b[conflict_mask] = 0.0
    weight_b_from_a[conflict_mask] = 0.0
    weight_b_from_a[a2b_pixels] = a2b_strength[a2b_pixels]
    weight_a_from_b[b2a_pixels] = b2a_strength[b2a_pixels]
    stats['ph_conflict_pixels'] = int(conflict_mask.sum().item())
    stats['a2b_pixels'] = int(a2b_pixels.sum().item())
    stats['b2a_pixels'] = int(b2a_pixels.sum().item())
    return weight_a_from_b, weight_b_from_a, stats


def _event_routing_weights(prob_a, prob_b, events):
    weights = {}
    for event in events:
        if event.get('force_abstain', False):
            weights[id(event)] = {
                'reliability_a': 0.0,
                'reliability_b': 0.0,
                'w_a2b': 0.0,
                'w_b2a': 0.0,
            }
            continue
        sample = int(event['sample_index'])
        y_slice, x_slice = event['roi_slice']
        roi = torch.from_numpy(event['roi']).to(
            device=prob_a.device,
            dtype=torch.bool,
        )
        roi_a = prob_a[sample, y_slice, x_slice][roi]
        roi_b = prob_b[sample, y_slice, x_slice][roi]
        reliability_a = _binary_reliability(roi_a)
        reliability_b = _binary_reliability(roi_b)
        delta = reliability_a - reliability_b
        weights[id(event)] = {
            'reliability_a': reliability_a,
            'reliability_b': reliability_b,
            'w_a2b': max(delta, 0.0),
            'w_b2a': max(-delta, 0.0),
        }
    return weights


def _binary_reliability(probability):
    if probability.numel() == 0:
        return 0.0
    probability = probability.detach().clamp(1e-8, 1.0 - 1e-8)
    entropy = -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )
    reliability = 1.0 - float(entropy.mean().item()) / math.log(2.0)
    return min(max(reliability, 0.0), 1.0)


def _structural_correct_sets(mask_a, mask_b, mask_gt, min_area, match_iou):
    components_gt, _cc_gt = _extract_components(mask_gt, min_area)
    components_a, _cc_a = _extract_components(mask_a, min_area)
    components_b, _cc_b = _extract_components(mask_b, min_area)
    errors_a = _component_error_sets(components_a, components_gt, match_iou)
    errors_b = _component_error_sets(components_b, components_gt, match_iou)
    return {
        'components_gt': components_gt,
        'a_correct': errors_a['struct_correct_gt'],
        'b_correct': errors_b['struct_correct_gt'],
    }


def _oracle_direction_for_event(event, structural):
    y_slice, x_slice = event['roi_slice']
    roi = event['roi']
    has_a2b = False
    has_b2a = False
    for gt_index, component in enumerate(structural['components_gt']):
        if not (component['mask'][y_slice, x_slice] & roi).any():
            continue
        a_correct = gt_index in structural['a_correct']
        b_correct = gt_index in structural['b_correct']
        if a_correct and not b_correct:
            has_a2b = True
        elif b_correct and not a_correct:
            has_b2a = True
    if has_a2b and not has_b2a:
        return 'a2b'
    if has_b2a and not has_a2b:
        return 'b2a'
    return None


def _predicted_direction(weight):
    w_a2b = float(weight['w_a2b'])
    w_b2a = float(weight['w_b2a'])
    if w_a2b > w_b2a:
        return 'a2b', w_a2b
    if w_b2a > w_a2b:
        return 'b2a', w_b2a
    return 'abstain', 0.0


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0
