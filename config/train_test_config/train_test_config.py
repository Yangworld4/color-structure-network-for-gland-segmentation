import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi
import torch
import os
from PIL import Image
import torchio as tio


def _snapshot_model_state(model):
    """将当前最佳权重复制到 CPU，避免后续训练继续修改最佳模型。"""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def print_train_loss_sup(train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss

def print_train_loss_MT(train_loss_sup_1, train_loss_cps, train_loss, num_batches, print_num, print_num_half, print_num_minus):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Sup  Loss: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_cps, train_epoch_loss

def print_train_loss_ConResNet(train_loss_seg, train_loss_res, train_loss, num_batches, print_num, print_num_half, print_num_minus):
    train_epoch_loss_seg = train_loss_seg / num_batches['train_sup']
    train_epoch_loss_res = train_loss_res / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Seg  Loss: {:.4f}'.format(train_epoch_loss_seg).ljust(print_num_half, ' '), '| Train Res Loss: {:.4f}'.format(train_epoch_loss_res).ljust(print_num_half, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_seg, train_epoch_loss_res, train_epoch_loss


def print_train_loss_EM(train_loss_sup_1, train_loss_cps, train_loss, num_batches, print_num, print_num_minus):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train  Sup  Loss: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_minus, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_minus, ' '), '|')
    print('| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_cps, train_epoch_loss


def print_train_loss(train_loss_sup_1, train_loss_sup_2, train_loss_cps, train_loss, num_batches, print_num, print_num_half):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train SUP Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps, train_epoch_loss

def print_val_loss_sup(val_loss, num_batches, print_num, print_num_minus):
    val_epoch_loss = val_loss / num_batches['val']
    print('-' * print_num)
    print('| Val Loss: {:.4f}'.format(val_epoch_loss).ljust(print_num_minus, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss

def print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss 1: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_half, ' '), '| Val Sup Loss 2: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2

def print_val_loss_ConResNet(val_loss_seg, val_loss_res, num_batches, print_num, print_num_half):
    val_epoch_loss_seg = val_loss_seg / num_batches['val']
    val_epoch_loss_res = val_loss_res / num_batches['val']
    print('-' * print_num)
    print('| Val Seg Loss: {:.4f}'.format(val_epoch_loss_seg).ljust(print_num_half, ' '), '| Val Res Loss: {:.4f}'.format(val_epoch_loss_res).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_seg, val_epoch_loss_res

def print_train_eval_sup(num_classes, score_list_train, mask_list_train, print_num):

    if num_classes == 2:
        eval_list = evaluate(score_list_train, mask_list_train)
        print('| Train Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    else:
        eval_list = evaluate_multi(score_list_train, mask_list_train)

        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc: {}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc: {}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Train mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Train mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        train_m_jc = eval_list[1]

    return eval_list, train_m_jc

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

def print_val_eval_sup(num_classes, score_list_val, mask_list_val, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_val, mask_list_val)
        print('| Val Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    else:
        eval_list = evaluate_multi(score_list_val, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Val mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Val mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')
        val_m_jc = eval_list[1]
    return eval_list, val_m_jc

def print_val_eval(num_classes, score_list_val1, score_list_val2, mask_list_val, print_num):
    if num_classes == 2:
        eval_list1 = evaluate(score_list_val1, mask_list_val)
        eval_list2 = evaluate(score_list_val2, mask_list_val)
        print('| Val Thr 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Thr 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val  Jc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    else:
        eval_list1 = evaluate_multi(score_list_val1, mask_list_val)
        eval_list2 = evaluate_multi(score_list_val2, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc 1: {}  '.format(eval_list1[0]).ljust(print_num, ' '), '| Val  Jc 2: {}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc 1: {}  '.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 2: {}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Val mJc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val mJc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val mDc 1: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Val mDc 2: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    return eval_list1, eval_list2, val_m_jc1, val_m_jc2

def save_val_best_sup_2d(num_classes, best_list, model, score_list_val, name_list_val, eval_list, path_trained_model, path_seg_results, palette, model_name):

    if num_classes == 2:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

            score_list_val = torch.softmax(score_list_val, dim=1)
            pred_results = score_list_val[:, 1, :, :].cpu().numpy()
            pred_results[pred_results > eval_list[0]] = 1
            pred_results[pred_results <= eval_list[0]] = 0

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))

    else:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

            pred_results = torch.max(score_list_val, 1)[1]
            pred_results = pred_results.cpu().numpy()

            assert len(name_list_val) == pred_results.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))

    return best_list

def save_val_best_sup_3d(num_classes, best_list, model, score_list_val, mask_list_val, eval_list, path_trained_model, path_seg_results, path_mask_results, model_name, format):

    if num_classes == 2:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    else:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    return best_list

def save_val_best_2d(
    best_lists,
    best_states,
    best_epochs,
    model1,
    model2,
    eval_list_1,
    eval_list_2,
    epoch,
):
    """独立更新 Result1 和 Result2 的验证集历史最佳权重快照。"""
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
                'Branch {} has no validation checkpoint at epoch {}'.format(
                    branch_name, current_epoch
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
):
    """训练结束时分别保存两个分支各自的验证集历史最优权重。"""
    for index, (best_list, best_state, best_epoch) in enumerate(
        zip(best_lists, best_states, best_epochs),
        start=1,
    ):
        if best_state is None:
            raise RuntimeError('Result{} has no validation checkpoint'.format(index))
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
                '| Best Result{} Val Thr: {:.4f}'.format(
                    index, best_list[0]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} Val  Jc: {:.4f}'.format(
                    index, best_list[1]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} Val  Dc: {:.4f}'.format(
                    index, best_list[2]
                ).ljust(print_num, ' '),
                '|',
            )
        else:
            print(
                '| Best Result{} Val mJc: {:.4f}'.format(
                    index, best_list[1]
                ).ljust(print_num, ' '),
                '|',
            )
            print(
                '| Best Result{} Val mDc: {:.4f}'.format(
                    index, best_list[3]
                ).ljust(print_num, ' '),
                '|',
            )


def save_val_best_3d(num_classes, best_model, best_list, best_result, model1, model2, score_list_val_1, score_list_val_2, mask_list_val, eval_list_1, eval_list_2, path_trained_model, path_seg_results, path_mask_results, format):

    if eval_list_1[1] < eval_list_2[1]:
        if best_list[1] < eval_list_2[1]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'

            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result2', best_list[1])))

        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[1] < eval_list_1[1]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'

            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result1', best_list[1])))

        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    return best_list, best_model, best_result

def draw_pred_sup(num_classes, mask_train_sup, mask_val, pred_train_sup, outputs_val, train_eval_list, val_eval_list):


    mask_image_train_sup = mask_train_sup[0, :, :].data.cpu().numpy()
    mask_image_val = mask_val[0, :, :].data.cpu().numpy()

    if num_classes == 2:
        pred_image_train_sup = pred_train_sup[0, 1, :, :].data.cpu().numpy()
        pred_image_train_sup[pred_image_train_sup > train_eval_list[0]] = 1
        pred_image_train_sup[pred_image_train_sup <= train_eval_list[0]] = 0

        pred_image_val = outputs_val[0, 1, :, :].data.cpu().numpy()
        pred_image_val[pred_image_val > val_eval_list[0]] = 1
        pred_image_val[pred_image_val <= val_eval_list[0]] = 0

    else:
        pred_image_train_sup = torch.max(pred_train_sup, 1)[1]
        pred_image_train_sup = pred_image_train_sup[0, :, :].cpu().numpy()

        pred_image_val = torch.max(outputs_val, 1)[1]
        pred_image_val = pred_image_val[0, :, :].cpu().numpy()

    return mask_image_train_sup, pred_image_train_sup, mask_image_val, pred_image_val


def print_best_sup(num_classes, best_val_list, print_num):
    if num_classes == 2:
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:
        np.set_printoptions(precision=4, suppress=True)
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

def print_best(num_classes, best_val_list, best_model, best_result, path_trained_model, print_num):
    # 兼容旧调用：新训练传入 CPU 权重快照，旧代码可能仍传入模型对象。
    best_state = (
        best_model
        if isinstance(best_model, dict)
        else _snapshot_model_state(best_model)
    )

    if num_classes == 2:

        torch.save(best_state, os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:

        torch.save(best_state, os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        np.set_printoptions(precision=4, suppress=True)
        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

def print_test_eval(num_classes, score_list_test, mask_list_test, print_num):
    if num_classes == 2:
        eval_list = evaluate(score_list_test, mask_list_test)
        print('| Test Thr: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Jc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
    else:
        eval_list = evaluate_multi(score_list_test, mask_list_test)
        np.set_printoptions(precision=4, suppress=True)
        print('| Test  Jc: {}  '.format(eval_list[0]).ljust(print_num, ' '), '|')
        print('| Test  Dc: {}  '.format(eval_list[2]).ljust(print_num, ' '), '|')
        print('| Test mJc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
        print('| Test mDc: {:.4f}'.format(eval_list[3]).ljust(print_num, ' '), '|')

    return eval_list


def save_test_2d(num_classes, score_list_test, name_list_test, threshold, path_seg_results, palette):

    if num_classes == 2:
        score_list_test = torch.softmax(score_list_test, dim=1)
        pred_results = score_list_test[:, 1, ...].cpu().numpy()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

    else:
        pred_results = torch.max(score_list_test, 1)[1]
        pred_results = pred_results.cpu().numpy()

        assert len(name_list_test) == pred_results.shape[0]

        for i in range(len(name_list_test)):
            color_results = Image.fromarray(pred_results[i].astype(np.uint8), mode='P')
            color_results.putpalette(palette)
            color_results.save(os.path.join(path_seg_results, name_list_test[i]))

def save_test_3d(num_classes, score_test, name_test, threshold, path_seg_results, affine):

    if num_classes == 2:
        score_list_test = torch.softmax(score_test, dim=0)
        pred_results = score_list_test[1, ...].cpu()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))

    else:
        pred_results = torch.max(score_test, 0)[1]
        pred_results = pred_results.cpu()
        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))
