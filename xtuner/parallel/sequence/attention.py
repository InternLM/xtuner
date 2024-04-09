# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .setup_distributed import (get_sequence_parallel_group,
                                get_sequence_parallel_world_size)


def all_to_all_scatter_nhead(input):
    # bs, seq, nhead, dim ==>
    # bs, seq * sp_world_size, nhead / sp_world_size, dim
    sp_world_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    bs, seq, nhead, dim = input.shape
    input_t = input.reshape(bs, seq, sp_world_size, nhead // sp_world_size,
                            dim)
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=sp_group)
    output = output.transpose(0, 1)
    return output.reshape(bs, seq * sp_world_size, nhead // sp_world_size, dim)


def all_to_all_scatter_seq(input):
    # bs, seq * sp_world_size, nhead / sp_world_size, dim ==>
    # bs, seq, nhead, dim
    sp_world_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    bs, seq, nhead, dim = input.shape
    input_t = input.reshape(bs, sp_world_size, seq // sp_world_size, nhead,
                            dim)
    input_t = input_t.transpose(0, 1).contiguous()
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=sp_group)
    output = output.permute(1, 2, 0, 3, 4)
    return output.reshape(bs, seq // sp_world_size, nhead * sp_world_size, dim)


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: Tensor, scatter_seq) -> Tensor:
        ctx.scatter_seq = scatter_seq
        ctx.input_shape = input.shape
        if scatter_seq:
            return all_to_all_scatter_seq(input)
        return all_to_all_scatter_nhead(input)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None]:
        grad = _SeqAllToAll.apply(*grad_output, not ctx.scatter_seq)
        return (grad, None)


def pre_process_for_sequence_parallel_attn(query_states, key_states,
                                           value_states):
    sequence_parallel_world_size = get_sequence_parallel_world_size()
    n_head = query_states.shape[2]
    assert n_head % sequence_parallel_world_size == 0, \
        ('The number of attention heads should be divisible by '
         f'sequence_parallel_world_size. But got n_head = {n_head} and '
         f'sequence_parallel_world_size = {sequence_parallel_world_size}.')

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    query_states = _SeqAllToAll.apply(query_states, False)
    key_states = _SeqAllToAll.apply(key_states, False)
    value_states = _SeqAllToAll.apply(value_states, False)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    output = _SeqAllToAll.apply(attn_output, True)
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
