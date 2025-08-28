from typing import Callable, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from xtuner.v1.utils import get_device


DEVICE = get_device()


def init_params(param: torch.Tensor, init_fn: Callable[[torch.Tensor], torch.Tensor | None]):
    assert not param.is_meta, "Internal Error. Found meta tensor during initialize model weight"
    device = param.device

    if isinstance(param, DTensor):
        full_param = torch.empty_like(param.full_tensor(), device=device)
        ret = init_fn(full_param)
        if ret is not None:
            full_param = ret
        param.copy_(distribute_tensor(full_param, param.device_mesh, param.placements))
    else:
        ret = init_fn(param)


def default_init_weights(module: nn.Module) -> set[str]:
    initialized_params: set[str] = set()

    def _init_weights_recursive(name: str, module: nn.Module, seen: set | None = None):
        if seen is None:
            seen = set()

        if id(module) in seen:
            return

        _init_weights_default(name, module)
        for child_name, child in module.named_children():
            child_name = f"{child_name}" if name == "" else f"{name}.{child_name}"
            _init_weights_recursive(child_name, child, seen)

        seen.add(id(module))

    def _init_weights_default(name: str, module: nn.Module):
        is_norm = "norm" in name

        if hasattr(module, "bias") and module.bias is not None:
            device = DEVICE if module.bias.is_meta else module.bias.device
            bias = cast(torch.Tensor, module.bias)

            if isinstance(bias, DTensor):
                replicate_bias = torch.empty_like(bias.full_tensor(), device=device)  # type: ignore

                replicate_bias.zero_()
                bias.copy_(distribute_tensor(replicate_bias, device_mesh=bias.device_mesh, placements=bias.placements))
            else:
                bias.zero_()
            initialized_params.add(f"{name}.bias")

        if hasattr(module, "weight") and module.weight is not None:
            device = DEVICE if module.weight.is_meta else module.weight.device
            weight = cast(torch.Tensor, module.weight)

            if isinstance(weight, DTensor):
                replicate_weight = torch.empty_like(weight.full_tensor(), device=device)  # type: ignore

                if is_norm:
                    replicate_weight.fill_(1.0)
                else:
                    replicate_weight.normal_(mean=0.0, std=0.02)
                weight.copy_(
                    distribute_tensor(replicate_weight, device_mesh=weight.device_mesh, placements=weight.placements)
                )
            else:
                weight = torch.empty_like(weight, device=device)  # type: ignore

                if is_norm:
                    weight.fill_(1.0)
                else:
                    weight.normal_(mean=0.0, std=0.02)
            initialized_params.add(f"{name}.weight")

    _init_weights_recursive("", module)
    return initialized_params
