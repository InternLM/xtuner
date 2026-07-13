from functools import partial

import torch
from torch.nn import functional as F


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def native_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    """GPT-OSS clipped SwiGLU: clamped pre-activations plus GPT-OSS's ``up + 1`` residual term.

    Args:
        fused_x (torch.Tensor): Concatenated ``[gate, up]`` pre-activations.
        split_dim (int): Dimension the gate/up halves are concatenated along. Defaults to ``-1``.
        alpha (float): Sigmoid steepness of the GLU. Defaults to ``1.702``.
        limit (float): Clamp bound. Defaults to ``7``.

    Returns:
        torch.Tensor: Activated tensor, half the size of ``fused_x`` along ``split_dim``.
    """
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def native_clamped_swiglu(fused_x: torch.Tensor, split_dim=-1, limit=10) -> torch.Tensor:
    """Plain SwiGLU on clamped pre-activations, as used by DeepSeek-V4.

    Shares the asymmetric clamp with :func:`native_clipped_swiglu` (``gate`` bounded from above
    only, ``up`` bounded both ways) but keeps the product ``silu(gate) * up``. GPT-OSS's extra
    ``up + 1`` residual term is *not* part of DeepSeek-V4: its reference ``Expert.forward`` and
    HF's ``DeepseekV4MLP.forward`` both compute ``act_fn(gate) * up``, so the two variants are
    kept as separate activations rather than one parameterised function.

    Args:
        fused_x (torch.Tensor): Concatenated ``[gate, up]`` pre-activations.
        split_dim (int): Dimension the gate/up halves are concatenated along. Defaults to ``-1``.
        limit (float): Clamp bound (DeepSeek-V4's ``swiglu_limit``). Defaults to ``10``.

    Returns:
        torch.Tensor: Activated tensor, half the size of ``fused_x`` along ``split_dim``.
    """
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def native_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    if approximate is not None:
        return F.gelu(x, approximate=approximate)
    return F.gelu(x)


def npu_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    import torch_npu

    if approximate is not None:
        return torch_npu.npu_gelu(x, approximate=approximate)
    return torch_npu.npu_gelu(x)


def npu_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    raise NotImplementedError


def npu_clamped_swiglu(fused_x: torch.Tensor, split_dim=-1, limit=10) -> torch.Tensor:
    raise NotImplementedError


def native_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


act_fn_type_map_cuda = {
    "swiglu": native_swiglu,
    "clipped_swiglu": native_clipped_swiglu,
    "clamped_swiglu": native_clamped_swiglu,
    "gelu": native_gelu,
    "gelu_pytorch_tanh": partial(native_gelu, approximate="tanh"),
    "silu": native_silu,
}
act_fn_type_map_npu = {
    "swiglu": npu_swiglu,
    "clipped_swiglu": npu_clipped_swiglu,
    "clamped_swiglu": npu_clamped_swiglu,
    "gelu": npu_gelu,
    "gelu_pytorch_tanh": partial(npu_gelu, approximate="tanh"),
    "silu": native_silu,
}


def get_act_fn(act_type):
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return act_fn_type_map_npu[act_type]
    else:
        return act_fn_type_map_cuda[act_type]
