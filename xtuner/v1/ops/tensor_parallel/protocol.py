from typing import Protocol, Tuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh


class AttnColwiseParallelProtocol(Protocol):
    def __call__(
        self, hidden_states: torch.Tensor, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, tp_mesh: DeviceMesh
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class AttnRowwiseParallelProtocol(Protocol):
    def __call__(self, attn_output: torch.Tensor, o_proj: nn.Linear, tp_mesh: DeviceMesh) -> torch.Tensor: ...


def cpu_column_parallel_forward(
    hidden_states: torch.Tensor, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, tp_mesh: DeviceMesh
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("cpu_column_parallel_forward is not implemented")


def cpu_attn_row_parallel_forward(attn_output: torch.Tensor, o_proj: nn.Linear, tp_mesh: DeviceMesh) -> torch.Tensor:
    raise NotImplementedError("cpu_attn_row_parallel_forward is not implemented")
