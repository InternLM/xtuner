# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.distributed.device_mesh import DeviceMesh

from ..comm import all_to_all


def pre_process_for_sequence_parallel_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sp_mesh: DeviceMesh,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    sp_size = sp_mesh.size()
    n_head = query_states.shape[2]
    assert n_head % sp_size == 0, (
        "The number of attention heads should be divisible by "
        f"sequence_parallel_world_size. But got n_head = {n_head} and "
        f"sequence_parallel_world_size = {sp_size}."
    )

    # (b, s // sp_world_size, nd, dim) -> (b, s, nd // sp_world_size, dim)
    sp_group = sp_mesh.get_group()
    query_states = all_to_all(
        query_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim
    )
    key_states = all_to_all(
        key_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim
    )
    value_states = all_to_all(
        value_states, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim
    )

    return query_states, key_states, value_states


def post_process_for_sequence_parallel_attn(
    attn_output: torch.Tensor, sp_mesh: DeviceMesh, scatter_dim=1, gather_dim=2
):
    # (b, s, nd // sp_world_size, dim) -> (b, s // sp_world_size, nd, dim)
    sp_group = sp_mesh.get_group()
    output = all_to_all(
        attn_output, sp_group, scatter_dim=scatter_dim, gather_dim=gather_dim
    )
    return output
