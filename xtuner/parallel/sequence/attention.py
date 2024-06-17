# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.distributed as dist

from .comm import (all_to_all, gather_forward_split_backward,
                   split_forward_gather_backward)
from .setup_distributed import (get_inner_sequence_parallel_group,
                                get_inner_sequence_parallel_world_size,
                                get_sequence_parallel_group,
                                get_sequence_parallel_world_size,
                                init_inner_sequence_parallel,
                                is_inner_sequence_parallel_initialized)


def pre_process_for_sequence_parallel_attn(query_states,
                                           key_states,
                                           value_states,
                                           scatter_dim=2,
                                           gather_dim=1):
    b, s_div_sp, h, d = query_states.shape
    sp = get_sequence_parallel_world_size()

    if not is_inner_sequence_parallel_initialized():
        insp = sp // math.gcd(h, sp)
        init_inner_sequence_parallel(insp)
    else:
        insp = get_inner_sequence_parallel_world_size()

    def pre_process_for_inner_sp(q, k, v):
        if scatter_dim != 2 and gather_dim != 1:
            raise NotImplementedError(
                'Currently only `scatter_dim == 2` and `gather_dim == 1` '
                f'is supported. But got scatter_dim = {scatter_dim} and '
                f'gather_dim = {gather_dim}.')

        # (b, s_div_sp, h, d) ->
        # (b, s_div_sp, sp/insp, h*insp/sp, insp, d/insp) ->
        # (b, s_div_sp, sp/insp, insp, h*insp/sp, d/insp) ->
        # (b, s_div_sp, insp*h, d/insp)
        q = q.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)
        k = k.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)
        v = v.view(b, s_div_sp, sp // insp, h * insp // sp, insp,
                   d // insp).transpose(3, 4).flatten(2, 4)

        return q, k, v

    def post_process_for_inner_sp(q, k, v):
        # (b, s, insp*h/sp, d/insp) -> (b, s, insp*h/sp, d)
        q = gather_forward_split_backward(q, -1,
                                          get_inner_sequence_parallel_group())
        k = gather_forward_split_backward(k, -1,
                                          get_inner_sequence_parallel_group())
        v = gather_forward_split_backward(v, -1,
                                          get_inner_sequence_parallel_group())

        return q, k, v

    assert (h * insp) % sp == 0, \
        ('The number of attention heads should be divisible by '
         '(sequence_parallel_world_size // sequence_parallel_inner_world_size)'
         f'. But got n_head = {h}, sequence_parallel_world_size = '
         f'{sp} and sequence_parallel_inner_world_size = {insp}.')

    if insp > 1:
        query_states, key_states, value_states = pre_process_for_inner_sp(
            query_states, key_states, value_states)

    # (b, s_div_sp, insp*h, d/insp) -> (b, s, insp*h/sp, d/insp)
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

    if insp > 1:
        query_states, key_states, value_states = post_process_for_inner_sp(
            query_states, key_states, value_states)

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(attn_output,
                                            scatter_dim=1,
                                            gather_dim=2):
    sp = get_sequence_parallel_world_size()
    insp = get_inner_sequence_parallel_world_size()
    b, s, h_mul_insp_div_sp, d = attn_output.shape
    h = h_mul_insp_div_sp * sp // insp
    s_div_sp = s // sp

    if insp > 1:
        # (b, s, insp*h/sp, d) -> (b, s, insp*h/sp, d/insp)
        attn_output = split_forward_gather_backward(
            attn_output, -1, get_inner_sequence_parallel_group())

    # (b, s, insp*h/sp, d/insp) -> (b, s_div_sp, insp*h, d/insp)
    sequence_parallel_group = get_sequence_parallel_group()
    output = all_to_all(
        attn_output,
        sequence_parallel_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim)

    if insp > 1:
        # (b, s_div_sp, insp*h, d/insp) ->
        # (b, s_div_sp, sp/insp, insp, h*insp/sp, d/insp) ->
        # (b, s_div_sp, sp/insp, h*insp/sp, insp, d/insp) ->
        # (b, s_div_sp, h, d)
        output = output.view(b, s_div_sp, sp // insp, insp, h * insp // sp,
                             d // insp).transpose(3, 4).reshape(
                                 b, s_div_sp, h, d)

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
