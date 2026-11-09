"""CPU model destinations with bounded pinned storage, retaining Module/FSDP move hooks."""

from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from .pinned_memory import allocate_pinned_like


def offload_to_pinned_cpu(module: nn.Module) -> None:
    """Move CUDA parameters and buffers into bounded pinned CPU destinations.

    Keep Module._apply responsible for Parameter/gradient identity and FSDP hooks.
    CUDA copies remain nonblocking; the caller must synchronize exactly as after
    Module.to(cpu, non_blocking=True). Unsupported tensor subclasses/layouts use
    the original Tensor.to path rather than losing their representation.

    Args:
        module (nn.Module): Model to move, including any fully_shard children.
    """
    fsdp_params: dict[int, Any] = {}
    for child in module.modules():
        if isinstance(child, FSDPModule):
            # Match FSDPModule._apply: plan only after its sharded parameters are registered.
            child.reshard()
            group = child._get_fsdp_state()._fsdp_param_group
            if group is not None:
                with torch.no_grad():
                    for sharded in group.fsdp_params:
                        # HF loading can materialize ordinary descendants with recurse=False,
                        # bypassing the owning FSDP parent's _apply hook. Refresh before reading
                        # the padded backing: it may still be meta or refer to old CUDA storage.
                        sharded.reset_sharded_param()
                        fsdp_params[id(sharded.sharded_param)] = sharded

    tensors: dict[int, torch.Tensor] = {}
    for param in module.parameters():
        tensors[id(param)] = param
        if param.grad is not None:
            tensors[id(param.grad)] = param.grad
    for buffer in module.buffers():
        # Preserve the original autograd behavior for the unusual differentiable-buffer case.
        if not buffer.requires_grad:
            tensors[id(buffer)] = buffer

    sources: list[torch.Tensor] = []
    selected: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for identity, tensor in tensors.items():
        local = tensor._local_tensor if isinstance(tensor, DTensor) else tensor
        if (
            local.device.type != "cuda"
            or type(local) not in (torch.Tensor, nn.Parameter)
            or local.layout != torch.strided
            or local.is_quantized
        ):
            continue
        fsdp_param = fsdp_params.get(identity)
        # FSDP retains a padded backing even for empty/uneven local shards. Allocate/copy that backing,
        # not just the visible shard, so reset_sharded_param cannot allocate a second, unpooled CPU copy.
        source = (
            fsdp_param._sharded_param_data.view(fsdp_param.padded_sharded_param_size)
            if fsdp_param is not None
            else local
        )
        sources.append(source)
        selected.append((identity, tensor, local))

    destinations = allocate_pinned_like(sources, copy_non_blocking=True)
    converted: dict[int, torch.Tensor] = {}
    padded: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for (identity, tensor, local), destination in zip(selected, destinations):
            fsdp_param = fsdp_params.get(identity)
            view = destination
            if fsdp_param is not None:
                padded[identity] = destination.view(-1)
                dim = fsdp_param.fsdp_placement.dim
                view = destination.narrow(dim, 0, local.shape[dim])
            converted[identity] = (
                DTensor(view, tensor._spec, requires_grad=False) if isinstance(tensor, DTensor) else view
            )

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        identity = id(tensor)
        target = converted.get(identity)
        if target is None:
            return tensor.to(device="cpu", non_blocking=True)
        if identity in padded:
            # reset_sharded_param recognizes this exact backing and keeps the narrow logical DTensor view.
            # Do this only as _apply reaches the parameter, not while preallocating destinations.
            fsdp_params[identity]._sharded_param_data = padded[identity]
        return target

    module._apply(convert)
