from typing import Tuple

import torch
from torch import nn
from torch.distributed._functional_collectives import all_gather_tensor_autograd, reduce_scatter_tensor_autograd
from torch.distributed._tensor import DeviceMesh


def attn_column_parallel_forward(
    hidden_states: torch.Tensor, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, tp_mesh: DeviceMesh
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = all_gather_tensor_autograd(hidden_states, gather_dim=1, group=tp_mesh.get_group())

    query_states = q_proj(hidden_states)
    key_states = k_proj(hidden_states)
    value_states = v_proj(hidden_states)

    return query_states, key_states, value_states


def attn_row_parallel_forward(attn_output: torch.Tensor, o_proj: nn.Linear, tp_mesh: DeviceMesh) -> torch.Tensor:
    output = o_proj(attn_output)
    output = reduce_scatter_tensor_autograd(output, reduceOp="sum", scatter_dim=1, group=tp_mesh.get_group())
    return output
