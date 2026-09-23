from functools import partial
from typing import Callable

import torch
from torch.nn import functional as F


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def native_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def native_clamped_swiglu(fused_x: torch.Tensor, split_dim=-1, limit=10.0) -> torch.Tensor:
    """GLM-5.3-Flash's clamped SwiGLU: ``silu(clamp_max(gate, limit)) *
    clamp(up, -limit, limit)``.

    Distinct from :func:`native_clipped_swiglu` (GPT-OSS's ``(up+1) * gate * sigmoid(alpha*gate)``)
    -- same activation family, different formula, so this is a separate function rather than a
    parameterization of the existing one.
    """
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def get_gated_act_fn(hidden_act: str, swiglu_limit: float | None = None) -> Callable[..., torch.Tensor]:
    """Build the gated activation for MLPs that keep gate/up in **separate**
    projections.

    The fused-``gate_up`` counterpart is :class:`~xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEActFnConfig`
    (routed experts); this is its unfused twin, used by ``DenseMLP`` / ``MoEMLP``. Resolving the
    variant once here keeps the formula -- and the branch -- out of every MLP's forward.

    Args:
        hidden_act (str): Elementwise activation applied to the gate, as named in
            ``transformers.activations.ACT2CLS``.
        swiglu_limit (float | None): ``None`` gives the plain ``act_fn(gate) * up`` every model
            uses. A float selects GLM-5.3-Flash's clamped SwiGLU,
            ``act_fn(clamp_max(gate, limit)) * clamp(up, -limit, limit)``.

    Returns:
        Callable[..., torch.Tensor]: ``(gate, up) -> activated``.
    """
    act_fn = get_act_fn(hidden_act)
    if swiglu_limit is None:
        return lambda gate, up: act_fn(gate) * up
    return lambda gate, up: act_fn(gate.clamp(max=swiglu_limit)) * up.clamp(min=-swiglu_limit, max=swiglu_limit)


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
    "clamped_swiglu": native_clamped_swiglu,
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
