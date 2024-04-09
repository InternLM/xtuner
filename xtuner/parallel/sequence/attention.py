# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .setup_distributed import (get_sequence_parallel_group,
                                get_sequence_parallel_world_size)


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
        input: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, process_group: dist.ProcessGroup,
                scatter_dim: int, gather_dim: int):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input, ctx.world_size, process_group, scatter_dim,
                             gather_dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple:
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
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
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input, process_group, scatter_dim, gather_dim)


def pre_process_for_sequence_parallel_attn(query_states, key_states,
                                           value_states):
    sequence_parallel_world_size = get_sequence_parallel_world_size()
    n_head = query_states.shape[2]
    assert n_head % sequence_parallel_world_size == 0, \
        ('The number of attention heads should be divisible by '
         f'sequence_parallel_world_size. But got n_head = {n_head} and '
         f'sequence_parallel_world_size = {sequence_parallel_world_size}.')

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    sequence_parallel_group = get_sequence_parallel_group()
    query_states = all_to_all(query_states, sequence_parallel_group, 2, 1)
    key_states = all_to_all(key_states, sequence_parallel_group, 2, 1)
    value_states = all_to_all(value_states, sequence_parallel_group, 2, 1)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    sequence_parallel_group = get_sequence_parallel_group()
    output = all_to_all(attn_output, sequence_parallel_group, 1, 2)
    return output


def sequence_parallel_wrapper(local_attn):

    def sequence_parallel_attn(query_states, key_states, value_states, *args,
                               **kwargs):
        training = kwargs.pop('training', True)
        enable_sequence_parallel = (
            dist.is_initialized() and get_sequence_parallel_world_size() > 1
            and training)
        if enable_sequence_parallel:
            query_states, key_states, value_states = \
                pre_process_for_sequence_parallel_attn(
                    query_states, key_states, value_states)

        out = local_attn(query_states, key_states, value_states, *args,
                         **kwargs)

        if enable_sequence_parallel:
            out = post_process_for_sequence_parallel_attn(out).contiguous()

        return out

    return sequence_parallel_attn
