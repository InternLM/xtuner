from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from .device import get_device
from .misc import clean_param_name


DEVICE = get_device()


def init_params(param: torch.Tensor, init_fn: Callable[[torch.Tensor], torch.Tensor | None]):
    """Initialize a single model parameter tensor, supporting both regular
    tensors and DTensors.

    Args:
        param (torch.Tensor): The parameter tensor to be initialized.
        init_fn (Callable[[torch.Tensor], torch.Tensor | None]): **in-place** initialization function to be applied.
    """
    assert not param.is_meta, "Internal Error. Found meta tensor during initialize model weight"
    device = param.device

    if isinstance(param, DTensor):
        # DTensors with `_StridedShard` at the rightmost mesh dim (e.g. InterleavedShard for
        # per-expert column-parallel MoE weights) cannot go through ``full_tensor()`` /
        # ``distribute_tensor`` — both depend on ``redistribute`` which has no path for that
        # layout. Initialize on the local tensor directly. This changes the random seed
        # distribution vs. "init full then scatter" but is the only path that works.
        from .interleaved_shard import has_interleaved_placement

        if has_interleaved_placement(param):
            init_fn(param._local_tensor)
        else:
            full_param = torch.empty_like(param.full_tensor(), device=device)
            init_fn(full_param)
            param.copy_(distribute_tensor(full_param, param.device_mesh, param.placements))
    else:
        init_fn(param)


def default_init_weights(module: nn.Module) -> set[str]:
    initialized_params: set[str] = set()

    def _init_weights_recursive(name: str, module: nn.Module, seen: set | None = None):
        if seen is None:
            seen = set()

        if id(module) in seen:
            return

        _default_init_atom(name, module)
        for child_name, child in module.named_children():
            child_name = f"{child_name}" if name == "" else f"{name}.{child_name}"
            _init_weights_recursive(child_name, child, seen)

        seen.add(id(module))

    def _default_init_atom(name: str, module: nn.Module):
        if hasattr(module, "bias") and module.bias is not None:
            bias = cast(torch.Tensor, module.bias)
            init_params(bias, nn.init.zeros_)
            initialized_params.add(clean_param_name(f"{name}.bias"))

        if hasattr(module, "weight") and module.weight is not None:
            weight = cast(torch.Tensor, module.weight)
            if "norm" in name:
                init_params(weight, nn.init.ones_)
            else:
                init_params(weight, partial(nn.init.normal_, mean=0.0, std=0.02))
            initialized_params.add(clean_param_name(f"{name}.weight"))

    _init_weights_recursive("", module)
    return initialized_params
