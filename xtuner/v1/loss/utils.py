from typing import Any

import torch
from mmengine.dist import dist
from torch.distributed.device_mesh import DeviceMesh

from xtuner.utils.device import get_device
from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel


DEVICE = get_device()


def sp_split(
    tensor,
    sp_mesh: DeviceMesh,
    split_dim: int,
    padding_value: Any,
):
    if sp_mesh.size() == 1:
        return tensor
    tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
    tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
    return tensor


def sp_gather(tensor, sp_mesh: DeviceMesh, dim: int):
    if sp_mesh.size() == 1:
        return tensor
    tensor_list = [torch.empty_like(tensor) for _ in range(sp_mesh.size())]
    dist.all_gather(tensor_list, tensor, group=sp_mesh)
    return torch.cat(tensor_list, dim=dim)


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
            loss_weight = torch.full(_labels.shape, loss_weight, device=_labels.device)
            loss_weight[_labels == -100] = 0.0
            loss_weights_list.append(loss_weight)
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
