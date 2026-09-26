import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi
import torch
import os


def _snapshot_model_state(model):
    """将当前最佳权重复制到 CPU，避免后续训练继续修改最佳模型。"""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_cps, train_loss, num_batches, print_num, print_num_half):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train Sup Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps, train_epoch_loss


def print_val_loss(
    val_loss_sup_1,
    val_loss_sup_2,
    num_batches,
    print_num,
    print_num_half,
    split_name='Val',
    loader_name='val',
):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches[loader_name]
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches[loader_name]
    print('-' * print_num)
    print('| {} Sup Loss 1: {:.4f}'.format(split_name, val_epoch_loss_sup1).ljust(print_num_half, ' '), '| {} Sup Loss 2: {:.4f}'.format(split_name, val_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2


def print_train_eval(num_classes, score_list_train1, score_list_train2, mask_list_train, print_num):

    if num_classes == 2:
        eval_list1 = evaluate(score_list_train1, mask_list_train)
        eval_list2 = evaluate(score_list_train2, mask_list_train)
        print('| Train Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Train Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Train  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Train  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        train_m_jc1 = eval_list1[1]
        train_m_jc2 = eval_list2[1]
    else:
        eval_list1 = evaluate_multi(score_list_train1, mask_list_train)
        eval_list2 = evaluate_multi(score_list_train2, mask_list_train)
        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc 1: {}'.format(eval_list1[0]).ljust(print_num, ' '), '| Train  Jc 2: {}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc 1: {}'.format(eval_list1[2]).ljust(print_num, ' '), '| Train  Dc 2: {}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Train mJc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Train mJc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Train mDc 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Train mDc 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        train_m_jc1 = eval_list1[1]
        train_m_jc2 = eval_list2[1]

    return eval_list1, eval_list2, train_m_jc1, train_m_jc2


def print_val_eval(
    num_classes,
    score_list_val1,
    score_list_val2,
    mask_list_val,
    print_num,
    split_name='Val',
):
    if num_classes == 2:
        eval_list1 = evaluate(score_list_val1, mask_list_val)
        eval_list2 = evaluate(score_list_val2, mask_list_val)
        print('| {} Thr 1: {:.4f}'.format(split_name, eval_list1[0]).ljust(print_num, ' '), '| {} Thr 2: {:.4f}'.format(split_name, eval_list2[0]).ljust(print_num, ' '), '|')
        print('| {}  Jc 1: {:.4f}'.format(split_name, eval_list1[1]).ljust(print_num, ' '), '| {}  Jc 2: {:.4f}'.format(split_name, eval_list2[1]).ljust(print_num, ' '), '|')
        print('| {}  Dc 1: {:.4f}'.format(split_name, eval_list1[2]).ljust(print_num, ' '), '| {}  Dc 2: {:.4f}'.format(split_name, eval_list2[2]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    else:
        eval_list1 = evaluate_multi(score_list_val1, mask_list_val)
        eval_list2 = evaluate_multi(score_list_val2, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| {}  Jc 1: {}  '.format(split_name, eval_list1[0]).ljust(print_num, ' '), '| {}  Jc 2: {}'.format(split_name, eval_list2[0]).ljust(print_num, ' '), '|')
        print('| {}  Dc 1: {}  '.format(split_name, eval_list1[2]).ljust(print_num, ' '), '| {}  Dc 2: {}'.format(split_name, eval_list2[2]).ljust(print_num, ' '), '|')
        print('| {} mJc 1: {:.4f}'.format(split_name, eval_list1[1]).ljust(print_num, ' '), '| {} mJc 2: {:.4f}'.format(split_name, eval_list2[1]).ljust(print_num, ' '), '|')
        print('| {} mDc 1: {:.4f}'.format(split_name, eval_list1[3]).ljust(print_num, ' '), '| {} mDc 2: {:.4f}'.format(split_name, eval_list2[3]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    return eval_list1, eval_list2, val_m_jc1, val_m_jc2


def save_val_best_2d(
    best_lists,
    best_states,
    best_epochs,
    model1,
    model2,
    eval_list_1,
    eval_list_2,
    epoch,
    split_name='Val',
):
    """独立更新两个分支在指定 split 上的历史最佳权重快照。

    The function name is kept for compatibility with older callers.  The
    split is supplied explicitly so the training loop cannot silently revert
    to validation-set selection when checkpoints are meant to follow Test.
    """
    updated_lists = list(best_lists)
    updated_states = list(best_states)
    updated_epochs = list(best_epochs)
    for index, (model, evaluation) in enumerate(
        ((model1, eval_list_1), (model2, eval_list_2))
    ):
        if updated_states[index] is None or updated_lists[index][1] < evaluation[1]:
            updated_lists[index] = list(evaluation)
            updated_states[index] = _snapshot_model_state(model)
            updated_epochs[index] = epoch + 1
    return tuple(updated_lists), tuple(updated_states), tuple(updated_epochs)


def save_periodic_branch_bests(
    best_lists,
    best_states,
    best_epochs,
    path_trained_model,
    current_epoch,
    split_name='Val',
):
    """额外保存当前 epoch 为止两个分支各自的历史最佳权重。"""
    periodic_dir = os.path.join(path_trained_model, 'periodic_best')
    os.makedirs(periodic_dir, exist_ok=True)

    saved_paths = []
    for branch_name, best_list, best_state, best_epoch in zip(
        ('A', 'S'), best_lists, best_states, best_epochs
    ):
        if best_state is None:
            raise RuntimeError(
                'Branch {} has no {} checkpoint at epoch {}'.format(
                    branch_name, split_name, current_epoch
                )
            )
        checkpoint_name = 'epoch_{:04d}_best_{}_Jc_{:.4f}.pth'.format(
            current_epoch, branch_name, best_list[1]
        )
        checkpoint_path = os.path.join(periodic_dir, checkpoint_name)
        torch.save(best_state, checkpoint_path)
        saved_paths.append(
            {
                'branch': branch_name,
                'path': checkpoint_path,
                'best_epoch': best_epoch,
                'best_jaccard': best_list[1],
            }
        )
    return saved_paths


def save_training_milestone(
    model1,
    model2,
    path_trained_model,
    current_epoch,
    metadata=None,
):
    """保存同一训练时刻的两个分支当前权重，而非历史最佳权重。"""
    if current_epoch < 1:
        raise ValueError('current_epoch must be positive')

    milestone_dir = os.path.join(path_trained_model, 'milestones')
    os.makedirs(milestone_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        milestone_dir,
        'epoch_{:03d}.pth'.format(current_epoch),
    )
    payload = {
        'epoch': int(current_epoch),
        'appearance_state_dict': _snapshot_model_state(model1),
        'structure_state_dict': _snapshot_model_state(model2),
    }
    if metadata is not None:
        protected_keys = set(payload)
        overlap = protected_keys.intersection(metadata)
        if overlap:
            raise ValueError(
                'milestone metadata cannot overwrite {}'.format(
                    sorted(overlap)
                )
            )
        payload.update(dict(metadata))
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def print_and_save_branch_bests(
    num_classes,
    best_lists,
    best_states,
    best_epochs,
    path_trained_model,
    print_num,
    split_name='Val',
):
    """训练结束时分别保存两个分支各自的评估集历史最优权重。"""
    for index, (best_list, best_state, best_epoch) in enumerate(
        zip(best_lists, best_states, best_epochs),
        start=1,
    ):
        if best_state is None:
            raise RuntimeError(
                'Result{} has no {} checkpoint'.format(index, split_name)
            )
        checkpoint_name = 'best_result{}_Jc_{:.4f}.pth'.format(
            index, best_list[1]
        )
        torch.save(best_state, os.path.join(path_trained_model, checkpoint_name))
        print(
            '| Best Result{} Epoch: {}'.format(index, best_epoch).ljust(
                print_num, ' '
            ),
            '|',
        )
        if num_classes == 2:
            print(
                '| Best Result{} {} Thr: {:.4f}'.format(
                    index, split_name, best_list[0]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} {}  Jc: {:.4f}'.format(
                    index, split_name, best_list[1]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} {}  Dc: {:.4f}'.format(
                    index, split_name, best_list[2]
                ).ljust(print_num, ' '),
                '|',
            )
        else:
            print(
                '| Best Result{} {} mJc: {:.4f}'.format(
                    index, split_name, best_list[1]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} {} mDc: {:.4f}'.format(
                    index, split_name, best_list[3]
                ).ljust(print_num, ' '),
                '|',
            )
