# Copyright (c) OpenMMLab. All rights reserved.
import torch.distributed as dist

from .comm import (all_to_all, gather_forward_split_backward,
                   split_forward_gather_backward)
from .setup_distributed import (get_sequence_parallel_group,
                                get_sequence_parallel_inner_group,
                                get_sequence_parallel_inner_world_size,
                                get_sequence_parallel_world_size)


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           scatter_dim=2,
                                           gather_dim=1):
    sp_size = get_sequence_parallel_world_size()
    sp_inner_size = get_sequence_parallel_inner_world_size()
    b, s_div_sp, h, d = query_states.shape
    assert (h * sp_inner_size) % sp_size == 0, \
        ('The number of attention heads should be divisible by '
         '(sequence_parallel_world_size // sequence_parallel_inner_world_size)'
         f'. But got n_head = {h}, sequence_parallel_world_size = '
         f'{sp_size} and sequence_parallel_inner_world_size = '
         f'{sp_inner_size}.')

    # print(f'rank {dist.get_rank()} {(b, s_div_sp, sp_inner_size, h // sp_inner_size, sp_inner_size, d // sp_inner_size)}')

    # (b, s_div_sp, h, d) -> (b, s_div_sp, sp/insp, h*insp/sp, insp, d/insp) ->
    # (b, s_div_sp, sp/insp, insp, h*insp/sp, d/insp) -> (b, s_div_sp, insp*h, d/insp)
    query_states = query_states.view(
        b, s_div_sp, sp_size // sp_inner_size, h * sp_inner_size // sp_size,
        sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 4)
    key_states = key_states.view(b, s_div_sp, sp_size // sp_inner_size,
                                 h * sp_inner_size // sp_size, sp_inner_size,
                                 d // sp_inner_size).transpose(3, 4).flatten(
                                     2, 4)
    value_states = value_states.view(
        b, s_div_sp, sp_size // sp_inner_size, h * sp_inner_size // sp_size,
        sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 4)
    # query_states = query_states.view(
    #     b, s_div_sp, sp_inner_size, h // sp_inner_size,
    #     sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 4)
    # key_states = key_states.view(
    #     b, s_div_sp, sp_inner_size, h // sp_inner_size,
    #     sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 4)
    # value_states = value_states.view(
    #     b, s_div_sp, sp_inner_size, h // sp_inner_size,
    #     sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 4)

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

    query_states = gather_forward_split_backward(
        query_states, -1, get_sequence_parallel_inner_group())
    key_states = gather_forward_split_backward(
        key_states, -1, get_sequence_parallel_inner_group())
    value_states = gather_forward_split_backward(
        value_states, -1, get_sequence_parallel_inner_group())

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            scatter_dim=1,
                                            gather_dim=2):
    sp_size = get_sequence_parallel_world_size()
    sp_inner_size = get_sequence_parallel_inner_world_size()
    b, s, h_mul_insp_div_sp, d = attn_output.shape
    h = h_mul_insp_div_sp * sp_size // sp_inner_size
    s_div_sp = s // sp_size
    attn_output = split_forward_gather_backward(
        attn_output, -1, get_sequence_parallel_inner_group())

    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    sequence_parallel_group = get_sequence_parallel_group()
    output = all_to_all(
        attn_output,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)
    output = output.view(b, s_div_sp, sp_size // sp_inner_size, sp_inner_size,
                         h * sp_inner_size // sp_size,
                         d // sp_inner_size).transpose(3, 4).flatten(
                             2, 3).flatten(3, 4)
    # output = output.view(
    #     b, s_div_sp, sp_inner_size, sp_inner_size,
    #     h // sp_inner_size, d // sp_inner_size).transpose(3, 4).flatten(2, 3).flatten(3, 4)
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
