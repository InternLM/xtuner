from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.ops import group_gemm


GroupedLinearParallelStyle = Literal["column", "row"]


class GroupedLinear(nn.Module):
    # TODO:Missng example docs
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        expert_tp_mesh: DeviceMesh | None = None,
        parallel_style: GroupedLinearParallelStyle | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts

        self.ep_mesh = ep_mesh
        self.expert_tp_mesh = expert_tp_mesh
        self.parallel_style: GroupedLinearParallelStyle | None = parallel_style
        self.ep_size = ep_mesh.size() if ep_mesh is not None else 1
        self.tp_size = expert_tp_mesh.size() if expert_tp_mesh is not None else 1
        self.ep_rank = ep_mesh.get_local_rank() if ep_mesh is not None else 0
        self.tp_rank = expert_tp_mesh.get_local_rank() if expert_tp_mesh is not None else 0
        self.tp_enabled = self.expert_tp_mesh is not None and self.tp_size > 1 and self.parallel_style is not None
        if self.expert_tp_mesh is not None and self.expert_tp_mesh.size() > 1 and self.parallel_style is None:
            raise ValueError("parallel_style must be set when expert_tp_mesh size is greater than 1.")
        if self.num_routed_experts % self.ep_size != 0:
            raise ValueError(
                f"num_routed_experts ({self.num_routed_experts}) must be divisible by ep_size ({self.ep_size})."
            )

        self.local_num_routed_experts = self.num_routed_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.local_num_routed_experts
        self.local_expert_end = self.local_expert_start + self.local_num_routed_experts
        self.local_in_features = in_features
        self.local_out_features = out_features
        if self.tp_enabled:
            if self.parallel_style == "column":
                if out_features % self.tp_size != 0:
                    raise ValueError(f"out_features ({out_features}) must be divisible by tp_size ({self.tp_size}).")
                self.local_out_features = out_features // self.tp_size
            elif self.parallel_style == "row":
                if in_features % self.tp_size != 0:
                    raise ValueError(f"in_features ({in_features}) must be divisible by tp_size ({self.tp_size}).")
                self.local_in_features = in_features // self.tp_size
            else:
                raise ValueError(f"Unsupported parallel_style: {self.parallel_style}.")

            # TODO: use DTensor instead of Tensor? for weight load?
            weight = torch.empty(
                self.local_num_routed_experts * self.local_out_features,
                self.local_in_features,
            )
            self.weight = nn.Parameter(weight)
        else:
            weight = torch.empty(num_routed_experts * out_features, in_features)
            if self.ep_mesh is not None and self.ep_mesh.size() > 1:
                self.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))
            else:
                self.weight = nn.Parameter(weight)

        self.moe_bias = moe_bias
        if self.moe_bias:
            if self.tp_enabled:
                bias_out_features = self.local_out_features if self.parallel_style == "column" else self.out_features
                self.bias = nn.Parameter(torch.zeros(self.local_num_routed_experts, bias_out_features))
            else:
                bias = torch.zeros(num_routed_experts, out_features)
                if self.ep_mesh is not None and self.ep_mesh.size() > 1:
                    self.bias = nn.Parameter(distribute_tensor(bias, ep_mesh, [Shard(0)]))
                else:
                    self.bias = nn.Parameter(torch.zeros(num_routed_experts, out_features))

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        weight = weight.view(-1, self.local_out_features, self.local_in_features)
        out = group_gemm(x, weight, tokens_per_expert)

        if self.moe_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            if self.tp_enabled and self.parallel_style == "row" and self.tp_rank != 0:
                return out
            out = out + bias.repeat_interleave(tokens_per_expert, dim=0)  # TODO: 无法 compile
        return out


def build_grouped_linear(
    in_features: int,
    out_features: int,
    num_routed_experts: int,
    moe_bias: bool = False,
    ep_mesh: DeviceMesh | None = None,
    expert_tp_mesh: DeviceMesh | None = None,
    parallel_style: GroupedLinearParallelStyle | None = None,
    float8_cfg: Float8Config | None = None,
):
    """Build a grouped linear layer with optional float8 support."""
    if float8_cfg is None or float8_cfg.scaling_granularity_gemm is None:
        return GroupedLinear(
            in_features,
            out_features,
            num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            parallel_style=parallel_style,
        )
    elif float8_cfg.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE:
        if expert_tp_mesh is not None and expert_tp_mesh.size() > 1:
            raise NotImplementedError("Tile-wise float8 grouped linear does not support expert TP sharding yet.")
        return TileWiseFloat8GroupedLinear(
            in_features, out_features, num_routed_experts, moe_bias=moe_bias, ep_mesh=ep_mesh
        )
    else:
        raise NotImplementedError(f"Unsupported float8 scaling granularity: {float8_cfg.scaling_granularity_gemm}")
