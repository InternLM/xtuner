import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.ops import grouped_gemm


class GroupedLinear(nn.Module):
    # TODO:Missng example docs
    def __init__(
        self, in_features: int, out_features: int, num_routed_experts: int, ep_mesh: DeviceMesh | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts
        weight = torch.empty(num_routed_experts * out_features, in_features)

        self.ep_mesh = ep_mesh
        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            self.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))
        else:
            self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        weight = weight.view(-1, self.out_features, self.in_features)
        out = grouped_gemm(x, weight, tokens_per_expert.cpu(), trans_a=False, trans_b=True)
        return out


def build_grouped_linear(
    in_features: int, out_features: int, num_routed_experts: int, ep_mesh: DeviceMesh | None = None, float8_cfg=None
):
    """Build a grouped linear layer with optional float8 support."""
    if float8_cfg is None:
        return GroupedLinear(in_features, out_features, num_routed_experts, ep_mesh=ep_mesh)
    elif float8_cfg.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE:
        return TileWiseFloat8GroupedLinear(in_features, out_features, num_routed_experts, ep_mesh=ep_mesh)
    else:
        raise NotImplementedError(f"Unsupported float8 scaling granularity: {float8_cfg.scaling_granularity_gemm}")
