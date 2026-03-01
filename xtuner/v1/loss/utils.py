from typing import Any

import torch
from mmengine.dist import dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn import functional as dist_functional

from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def sp_split(
    tensor,
    sp_mesh: DeviceMesh | None,
    split_dim: int,
    padding_value: Any,
):
    if tensor is None or sp_mesh is None or sp_mesh.size() == 1:
        return tensor
    tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
    tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
    return tensor


def sp_gather(tensor, sp_mesh: DeviceMesh | None, dim: int):
    if tensor is None or sp_mesh is None or sp_mesh.size() == 1:
        return tensor
    tensor_list = dist.all_gather(tensor, group=sp_mesh.get_group())
    return torch.cat(tensor_list, dim=dim)


def sp_gather_autograd(tensor, sp_mesh: DeviceMesh | None, dim: int):
    """Differentiable version of sp_gather.

    Use this when gradients need to flow back through the gather operation,
    e.g., for hidden_states in chunk_linear mode.

    Args:
        tensor: Input tensor to gather.
        sp_mesh: Sequence parallel device mesh.
        dim: Dimension to gather along.

    Returns:
        Gathered tensor with gradient support.
    """
    if tensor is None or sp_mesh is None or sp_mesh.size() == 1:
        return tensor
    tensor_list = dist_functional.all_gather(tensor, group=sp_mesh.get_group())
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


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
