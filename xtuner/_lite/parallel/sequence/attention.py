# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist

from ..comm import all_to_all
from ..setup import get_sp_mesh


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           sp_mesh,
                                           scatter_dim=2,
                                           gather_dim=1):
    sp_size = sp_mesh.size()
    n_head = query_states.shape[2]
    assert n_head % sp_size == 0, \
        ('The number of attention heads should be divisible by '
         f'sequence_parallel_world_size. But got n_head = {n_head} and '
         f'sequence_parallel_world_size = {sp_size}.')

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    sp_group = sp_mesh.get_group()
    query_states = all_to_all(
        query_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
    key_states = all_to_all(
        key_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
    value_states = all_to_all(
        value_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            sp_mesh,
                                            scatter_dim=1,
                                            gather_dim=2):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    sp_group = sp_mesh.get_group()
    output = all_to_all(
        attn_output, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim)
    return output


def sequence_parallel_wrapper(local_attn):

    def sequence_parallel_attn(query_states, key_states, value_states, *args,
                               **kwargs):
        training = kwargs.pop('training', True)
        sp_mesh = kwargs.pop('sp_mesh', None)

        if sp_mesh:
            sp_size = sp_mesh.size()
        else:
            sp_size = get_sp_mesh().size()

        enable_sequence_parallel = sp_size > 1
        if enable_sequence_parallel:
            query_states, key_states, value_states = \
                pre_process_for_sequence_parallel_attn(
                    query_states, key_states, value_states, sp_mesh)

        out = local_attn(query_states, key_states, value_states, *args,
                         **kwargs)

        if enable_sequence_parallel:
            out = post_process_for_sequence_parallel_attn(out, sp_mesh).contiguous()

        return out

    return sequence_parallel_attn
