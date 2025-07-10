import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear
from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear
from xtuner.v1.float8.float8_tensor import ScalingGranularity


class _Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)


def build_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    device=None,
    dtype=None,
    float8_cfg=None,
) -> nn.Module:
    """Build a linear layer with optional float8 support."""
    if float8_cfg is None:
        return _Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif float8_cfg.scaling_granularity_gemm is ScalingGranularity.TILEWISE:
        return TileWiseFloat8Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif float8_cfg.scaling_granularity_gemm is ScalingGranularity.TENSORWISE:
        return TensorWiseFloat8Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Unsupported float8 scaling granularity: {float8_cfg.scaling_granularity_gemm}")
