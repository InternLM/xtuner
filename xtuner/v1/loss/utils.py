import torch
from mmengine.dist import dist

from xtuner.utils.device import get_device


DEVICE = get_device()


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def cal_global_grad_tokens(labels: list[torch.Tensor], sp_mesh=None):
    # calculate global token number which is used for loss scaling
    rank_grad_tokens = torch.tensor(0, dtype=torch.int64, device=DEVICE)
    for label in labels:
        rank_grad_tokens += (label >= 0).sum()
    dist.all_reduce(rank_grad_tokens)
    if sp_mesh:
        # data in different sp ranks are replicated
        global_grad_tokens = rank_grad_tokens / sp_mesh.size()
    else:
        global_grad_tokens = rank_grad_tokens
    return global_grad_tokens


def cal_global_sum_loss_weight(
    labels_list: list[torch.Tensor], num_tokens_list: list[torch.Tensor], loss_reduction: str, sp_mesh=None
):
    assert len(labels_list) == len(num_tokens_list), "labels and num_tokens must have the same length"

    batch_loss_weights = []
    global_sum_loss_weight = torch.tensor(0, dtype=torch.float32, device=DEVICE)
    for labels, num_tokens in zip(labels_list, num_tokens_list):
        labels_list_ = torch.split(labels, num_tokens.tolist(), dim=1)
        loss_weights_list = []
        for _labels in labels_list_:
            num_effective_tokens = (_labels >= 0).sum().item()
            loss_weight = len2weight(num_effective_tokens, loss_reduction)
            loss_weights_list.append(torch.full(_labels.shape, loss_weight, device=_labels.device))
        loss_weights = torch.cat(loss_weights_list, dim=1)
        batch_loss_weights.append(loss_weights)
        global_sum_loss_weight += loss_weights.sum()
    dist.all_reduce(global_sum_loss_weight)
    if sp_mesh:
        # data in different sp ranks are replicated
        global_sum_loss_weight = global_sum_loss_weight / sp_mesh.size()
    else:
        global_sum_loss_weight = global_sum_loss_weight
    return global_sum_loss_weight, batch_loss_weights
