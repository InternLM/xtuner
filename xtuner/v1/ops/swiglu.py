from typing import Protocol

import torch


class SwigluProtocol(Protocol):
    def __call__(self, fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor: ...


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    from torch.nn import functional as F

    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def get_swiglu() -> SwigluProtocol:
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return npu_swiglu
    else:
        return native_swiglu


swiglu = get_swiglu()
