import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.ops import grouped_gemm


class GroupedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_routed_experts: int, ep_mesh: DeviceMesh | None):
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
