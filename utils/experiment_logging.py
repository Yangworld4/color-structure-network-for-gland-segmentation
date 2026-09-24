"""Run output paths, logging setup, and CSV schemas.

Keeping these details here leaves main.py focused on the training workflow.
"""

import logging
import os
import time

METRIC_FIELDS = [
    'iteration', 'epoch',
    'train_loss_sup1', 'train_loss_sup2', 'train_loss_cps',
    'train_loss_total', 'student12_disagreement',
    'student12_dice_similarity', 'student1_entropy', 'student2_entropy',
    'pseudo_dice_student1', 'pseudo_dice_student2',
    'pseudo_jaccard_student1', 'pseudo_jaccard_student2',
]
MOTIVATION_METRIC_FIELDS = [
    'epoch', 'agreement', 'soft_disagreement', 'ww_given_agreement',
    'hc90ww_given_ww', 'total_pixels', 'agreement_pixels', 'ww_pixels',
    'hc90ww_pixels',
]

NATIVE_TRAINING_FIELDS = [
    'experiment', 'epoch', 'num_native_tokens', 'num_agreement_tokens',
    'num_contradiction_tokens', 'contradiction_rate',
    'num_hc_contradiction_tokens', 'hc_contradiction_rate',
    'mean_CPS_weight', 'loss_sup', 'loss_cps_raw', 'loss_cps_weighted',
    'loss_native_ref', 'loss_total', 'mean_pA_token_on_HC',
    'mean_pB_token_on_HC', 'UNI_FG_HC_count', 'UNI_BG_HC_count',
    'mean_S_cur_contradiction', 'mean_S_opp_contradiction',
    'mean_absolute_D', 'mean_relative_gap', 'relative_gap_p10',
    'relative_gap_p25', 'relative_gap_p50', 'relative_gap_p75',
    'relative_gap_p90', 'mean_adaptive_weight', 'weight_p10',
    'weight_p25', 'weight_p50', 'weight_p75', 'weight_p90',
    'fraction_weight_0_0.2', 'fraction_weight_0.2_0.4',
    'fraction_weight_0.4_0.6', 'fraction_weight_0.6_0.8',
    'fraction_weight_0.8_1.0', 'gap_Q1_mean_gap', 'gap_Q1_mean_weight',
    'gap_Q2_mean_gap', 'gap_Q2_mean_weight', 'gap_Q3_mean_gap',
    'gap_Q3_mean_weight', 'gap_Q4_mean_gap', 'gap_Q4_mean_weight',
    'mean_D_contradiction', 'D_p10', 'D_p25', 'D_p50', 'D_p75', 'D_p90',
    'mean_contradiction_weight', 'contradiction_weight_p10',
    'contradiction_weight_p25', 'contradiction_weight_p50',
    'contradiction_weight_p75', 'contradiction_weight_p90',
    'fraction_weight_0_0.1', 'fraction_weight_0.1_0.2',
    'fraction_weight_0.2_0.3', 'fraction_weight_0.3_0.4',
    'fraction_weight_0.4_0.5', 'fraction_zero_weight',
    'D_Q1_mean_D', 'D_Q1_mean_weight', 'D_Q2_mean_D', 'D_Q2_mean_weight',
    'D_Q3_mean_D', 'D_Q3_mean_weight', 'D_Q4_mean_D', 'D_Q4_mean_weight',
    'regionwise_candidate_pixels', 'fine_selected_pixels',
    'fine_selected_over_regionwise', 'R1_same_consensus',
    'R2_opposite_consensus', 'R3_student_disagreement',
    'fine_suppression_coverage', 'global_suppression_coverage',
    'mean_applied_pixel_weight', 'num_risk_pixels', 'risk_pixel_fraction',
    'mean_D_risk', 'D_risk_p10', 'D_risk_p25', 'D_risk_p50',
    'D_risk_p75', 'D_risk_p90', 'mean_w_risk', 'mean_r_risk',
    'w_risk_p10', 'w_risk_p50', 'w_risk_p90', 'r_risk_p10',
    'r_risk_p50', 'r_risk_p90', 'mean_w_plus_r', 'reverse_fraction',
    'loss_cps_after_suppression', 'loss_reverse', 'loss_unsup_total',
    'mean_q_risk', 'q_risk_p10', 'q_risk_p25', 'q_risk_p50',
    'q_risk_p75', 'q_risk_p90', 'mean_CPS_weight_risk',
    'mean_opposite_weight_risk', 'risk_budget', 'risk_weight_sum_max_error',
    'loss_risk_severity',
]


def _experiment_name(args, split_sup_mark, split_unsup_mark):
    experiment_name = (
        'CPS-l={}-e={}-s={}-g={}-b={}-cw={}-w={}-{}-{}-{}'.format(
            args.lr,
            args.num_epochs,
            args.step_size,
            args.gamma,
            args.batch_size,
            args.unsup_weight,
            args.warm_up_duration,
            split_sup_mark,
            split_unsup_mark,
            args.input1,
        )
    )
    experiment_name += '-input_mode={}'.format(args.input_mode)
    experiment_name += '-cps_mode={}'.format(
        getattr(args, 'cps_mode', 'standard')
    )
    return experiment_name


def build_output_paths(args, datasets, use_manifests):
    split_sup_mark = (
        str(len(datasets['train_sup_unique']))
        if use_manifests
        else str(args.sup_mark)
    )
    split_unsup_mark = (
        str(len(datasets['train_unsup']))
        if use_manifests
        else str(args.unsup_mark)
    )
    experiment_name = _experiment_name(
        args, split_sup_mark, split_unsup_mark
    )
    dataset_dir_name = os.path.basename(os.path.normpath(args.path_dataset))

    trained_models = os.path.join(
        args.path_trained_models, dataset_dir_name, experiment_name
    )
    segmentation_results = os.path.join(
        args.path_seg_results, dataset_dir_name, experiment_name
    )
    os.makedirs(trained_models, exist_ok=True)
    os.makedirs(segmentation_results, exist_ok=True)
    return trained_models, segmentation_results


def create_run_outputs(args, trained_models_path):
    log_path = os.path.join(trained_models_path, 'train.log')
    metrics_path = os.path.join(trained_models_path, 'metrics.csv')
    diagnostic_log_path = os.path.join(
        trained_models_path,
        'diagnostics_{}.log'.format(time.strftime('%Y%m%d-%H%M%S')),
    )

    logger = logging.getLogger('fdcl.train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    log_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    logger.addHandler(log_handler)
    logger.info(
        'Training started | dataset=%s | labeled_ratio=%s | network=%s '
        '| input_mode=%s',
        args.dataset_name,
        args.labeled_ratio,
        args.network,
        args.input_mode,
    )
    logger.info('Arguments: %s', vars(args))

    print('Training log:', log_path)
    print('Metrics CSV:', metrics_path)

    # CPS Diagnostic Analysis
    diagnostic_logger = logging.getLogger('fdcl.diagnostics')
    diagnostic_logger.setLevel(logging.INFO)
    diagnostic_logger.handlers.clear()
    diagnostic_logger.propagate = False
    diagnostic_handler = logging.FileHandler(
        diagnostic_log_path, mode='w', encoding='utf-8'
    )
    diagnostic_handler.setFormatter(logging.Formatter('%(message)s'))
    diagnostic_logger.addHandler(diagnostic_handler)
    print('Diagnostic log:', diagnostic_log_path)
    return logger, metrics_path, diagnostic_logger


def log_epoch(logger, args, metrics):
    logger.info(
        'epoch=%d/%d | learning_rate=%.8g | average_train_loss=%.6f '
        '| val_dice_student1=%.6f | val_dice_student2=%.6f '
        '| val_jaccard_student1=%.6f | val_jaccard_student2=%.6f '
        '| student12_disagreement=%.6f '
        '| student12_dice_similarity=%.6f '
        '| student1_correct_student2_wrong=%.6f '
        '| student2_correct_student1_wrong=%.6f '
        '| student12_both_correct=%.6f '
        '| student12_both_wrong=%.6f '
        '| complementary_error=%.6f '
        '| best_val_jaccard1=%.6f | best_val_jaccard2=%.6f '
        '| best_val_jaccard=%.6f | time=%.2fs',
        metrics['epoch'],
        args.num_epochs,
        metrics['learning_rate'],
        metrics['average_train_loss'],
        metrics['val_dice_student1'],
        metrics['val_dice_student2'],
        metrics['val_jaccard_student1'],
        metrics['val_jaccard_student2'],
        metrics['student12_disagreement'],
        metrics['student12_dice_similarity'],
        metrics['student1_correct_student2_wrong'],
        metrics['student2_correct_student1_wrong'],
        metrics['student12_both_correct'],
        metrics['student12_both_wrong'],
        metrics['complementary_error'],
        metrics['best_val_jaccard1'],
        metrics['best_val_jaccard2'],
        metrics['best_val_jaccard'],
        metrics['epoch_seconds'],
    )

