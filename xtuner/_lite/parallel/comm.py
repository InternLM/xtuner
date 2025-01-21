# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import (_get_pg_default_device,
                                                _object_to_tensor,
                                                _tensor_to_object)


# Modified from https://github.com/microsoft/DeepSpeed/blob/ffd0a0e3ef24bfd00c2e5f35019d2674cc01ec14/deepspeed/sequence/layer.py#L15  # noqa: E501
def _all_to_all(
    input: Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input: Input tensor
        sp_group: Sequence parallel process group
        scatter_dim: Scatter dimension
        gather_dim: Gather dimension
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, sp_group: dist.ProcessGroup,
                scatter_dim: int, gather_dim: int):
        ctx.sp_group = sp_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(sp_group)
        output = _all_to_all(input, ctx.world_size, sp_group, scatter_dim,
                             gather_dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple:
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.sp_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input: Tensor,
    sp_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    """Convenience function to apply the all-to-all operation with scatter and
    gather dimensions.

    Notes:
        We have wrapped the `torch.distributed.all_to_all` function to
        enable automatic differentiation of the all-to-all operation.

    Args:
        input: The input tensor for which all-to-all communication is performed
        sp_group: The sequence parallel process group.
        scatter_dim: The dimension along which the input tensor is scattered
            (default: 2).
        gather_dim: The dimension along which the output tensor is gathered
            (default: 1).

    Returns:
        The output tensor after the all-to-all communication.
    """
    return _AllToAll.apply(input, sp_group, scatter_dim, gather_dim)


def all_to_all_list(object_list, group=None):
    current_device = _get_pg_default_device(group)
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    tensor_list, size_list = zip(
        *
        [_object_to_tensor(obj, current_device, group) for obj in object_list])
    tensor_list = list(tensor_list)
    size_list = torch.cat(size_list)
    buffer = [None] * world_size

    dist.all_gather_object(buffer, size_list, group=group)
    size_this_rank = []
    for size_list in buffer:
        size_this_rank.append(size_list[rank])

    target_tensor_list = [
        torch.empty(size.item(), dtype=torch.uint8, device=current_device)
        for size in size_this_rank
    ]
    dist.all_to_all(target_tensor_list, tensor_list, group=group)

    for i in range(len(target_tensor_list)):
        obj_view = target_tensor_list[i].type(torch.uint8)
        target_tensor_list[i] = _tensor_to_object(obj_view, size_this_rank[i],
                                                  group)

    return target_tensor_list


def barrier():
    if not dist.is_available():
        return

    rank = dist.get_rank()
    if rank == 0:
        objects = [1]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0)
    return
