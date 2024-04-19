# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist

from .comm import all_to_all
from .setup_distributed import (get_sequence_parallel_group,
                                get_sequence_parallel_world_size)


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           scatter_dim=2,
                                           gather_dim=1):
    sequence_parallel_world_size = get_sequence_parallel_world_size()
    n_head = query_states.shape[2]
    assert n_head % sequence_parallel_world_size == 0, \
        ('The number of attention heads should be divisible by '
         f'sequence_parallel_world_size. But got n_head = {n_head} and '
         f'sequence_parallel_world_size = {sequence_parallel_world_size}.')

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    sequence_parallel_group = get_sequence_parallel_group()
    query_states = all_to_all(
        query_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    key_states = all_to_all(
        key_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    value_states = all_to_all(
        value_states,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            scatter_dim=1,
                                            gather_dim=2):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    sequence_parallel_group = get_sequence_parallel_group()
    output = all_to_all(
        attn_output,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
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
