"""Read-only FSDP2 binding for UltraEP expert projections.

This module is the only UltraEP integration layer that imports private FSDP2 types.  UltraEP does not own FSDP's all-
gather storage: native weight sync reads the current unsharded parameter view and refreshes its pointer pool.
Consequently the binding records identity and validates the view, while leaving FSDP allocation and resharding methods
untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import torch
from torch import nn
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.distributed.tensor import DTensor


_LOG = logging.getLogger(__name__)
_FSDP_PARAM_ATTR = "_xtuner_ultraep_fsdp_param"
_OWNER_ATTR = "_xtuner_ultraep_fsdp_owner"
_PROJECTION_ATTR = "_xtuner_ultraep_projection"
UltraEPFSDPBinding: TypeAlias = tuple[FSDPParam, ...]


def _resolve_targets(
    fsdp_root: nn.Module,
    targets: Sequence[tuple[str, tuple[nn.Module, nn.Module]]],
) -> list[tuple[FSDPParam, nn.Module, nn.Module]]:
    by_identity: dict[tuple[int, str], tuple[FSDPParam, nn.Module]] = {}
    for owner in fsdp_root.modules():
        state = _get_module_fsdp_state(owner)
        if state is None:
            continue
        groups = getattr(cast(Any, state), "_fsdp_param_groups", None)
        if groups is None:
            # PyTorch 2.9 exposes the same group through the singular
            # compatibility field; newer FSDP2 releases use a list.
            group = getattr(cast(Any, state), "_fsdp_param_group", None)
            groups = () if group is None else (group,)
        for group in groups:
            for fsdp_param in group.fsdp_params:
                key = (id(fsdp_param._module_info.module), fsdp_param._module_info.param_name)
                if key in by_identity:
                    raise RuntimeError("UltraEP found duplicate FSDP parameter identity")
                by_identity[key] = fsdp_param, owner

    selected: list[tuple[FSDPParam, nn.Module, nn.Module]] = []
    for layer_label, projections in targets:
        for projection_name, projection in zip(("fused_w1w3", "fused_w2"), projections, strict=True):
            if hasattr(projection, _FSDP_PARAM_ATTR):
                raise RuntimeError(f"UltraEP FSDP binding is already installed for {layer_label}.{projection_name}")
            match = by_identity.get((id(projection), "weight"))
            if match is None:
                raise RuntimeError(f"UltraEP could not find FSDPParam for {layer_label}.{projection_name}.weight")
            fsdp_param, owner = match
            if fsdp_param.fsdp_placement.dim != 0:
                raise RuntimeError(f"UltraEP requires dim-0 FSDP sharding for {layer_label}.{projection_name}")
            if fsdp_param.sharded_state is not ShardedState.SHARDED or fsdp_param.all_gather_outputs:
                raise RuntimeError(f"UltraEP binding must be installed before AllGather for {layer_label}")
            expected_dtype = fsdp_param.mp_policy.param_dtype or fsdp_param.sharded_param.dtype
            if expected_dtype is not torch.bfloat16:
                raise RuntimeError(
                    f"UltraEP FSDP binding requires BF16 parameters for {layer_label}.{projection_name}"
                )
            if hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather"):
                raise RuntimeError(f"UltraEP does not support FSDP post-AllGather extensions for {layer_label}")
            selected.append((fsdp_param, owner, projection))

    if len({id(item[0]) for item in selected}) != len(selected):
        raise RuntimeError("UltraEP expert targets must map to distinct FSDPParams")
    return selected


def install_ultraep_fsdp_binding(
    *,
    fsdp_root: nn.Module,
    targets: Sequence[tuple[str, tuple[nn.Module, nn.Module]]],
) -> tuple[FSDPParam, ...]:
    """Record FSDP parameter identities for the routed expert projections.

    Unlike MoonEP this function does not replace FSDP's allocation methods or force a landing address.  UltraEP
    refreshes native pointers from the current parameter view on every weight-sync call.
    """
    if not (torch.__version__.startswith("2.9.") or torch.__version__.startswith("2.12.")):
        _LOG.warning("UltraEP FSDP binding is untested with torch %s", torch.__version__)
    selected = _resolve_targets(fsdp_root, targets)
    for fsdp_param, owner, projection in selected:
        setattr(projection, _FSDP_PARAM_ATTR, fsdp_param)
        setattr(fsdp_param, _OWNER_ATTR, owner)
        setattr(fsdp_param, _PROJECTION_ATTR, projection)
    return tuple(item[0] for item in selected)


def fsdp_binding_installed(projections: tuple[object, object]) -> bool:
    """Return whether both expert projections have the identity seam
    installed."""
    states = tuple(hasattr(projection, _FSDP_PARAM_ATTR) for projection in projections)
    if any(states) and not all(states):
        raise RuntimeError("UltraEP FSDP binding is only partially installed for an expert layer")
    return all(states)


def fsdp_current_unsharded_expert_parameters(
    projections: tuple[object, object],
) -> tuple[nn.Parameter, nn.Parameter]:
    """Return the two current FSDP Parameters inside their unsharded window."""
    result: list[nn.Parameter] = []
    for projection in projections:
        fsdp_param = getattr(projection, _FSDP_PARAM_ATTR, None)
        if fsdp_param is None:
            raise RuntimeError("UltraEP FSDP binding is not installed")
        if fsdp_param.sharded_state is not ShardedState.UNSHARDED:
            raise RuntimeError("UltraEP expert weight was read outside its FSDP unsharded window")
        parameter = getattr(fsdp_param._module_info.module, fsdp_param._module_info.param_name)
        if parameter is not fsdp_param.unsharded_param or not isinstance(parameter, nn.Parameter):
            raise RuntimeError("UltraEP observed an unexpected FSDP Parameter switch")
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        if not local.is_contiguous() or local.numel() != fsdp_param._orig_size.numel():
            raise RuntimeError("UltraEP FSDP unsharded expert view has unexpected shape or layout")
        result.append(parameter)
    return result[0], result[1]


def writeback_fsdp_unsharded_expert_gradients(
    parameters: tuple[nn.Parameter, nn.Parameter],
    reduced_gradients: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Bind completed BF16 staging gradients to the current FSDP Parameters."""
    with torch.no_grad():
        for parameter, staging in zip(parameters, reduced_gradients, strict=True):
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            if staging.dtype is not torch.bfloat16:
                raise TypeError(f"UltraEP staging must be BF16, got {staging.dtype}")
            grad = staging.reshape(local.shape)
            if isinstance(parameter, DTensor):
                grad = DTensor.from_local(
                    grad,
                    parameter.device_mesh,
                    parameter.placements,
                    run_check=False,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
            parameter.grad = grad


def uninstall_ultraep_fsdp_binding(fsdp_params: tuple[FSDPParam, ...]) -> None:
    """Remove identity attributes after the FSDP root has returned to idle."""
    for fsdp_param in fsdp_params:
        owner = getattr(fsdp_param, _OWNER_ATTR, None)
        state = _get_module_fsdp_state(owner) if owner is not None else None
        if state is not None and state._training_state.name != "IDLE":
            raise RuntimeError("UltraEP FSDP binding may only be removed at an idle boundary")
        projection = getattr(fsdp_param, _PROJECTION_ATTR, None)
        if projection is not None and hasattr(projection, _FSDP_PARAM_ATTR):
            delattr(projection, _FSDP_PARAM_ATTR)
        if hasattr(fsdp_param, _OWNER_ATTR):
            delattr(fsdp_param, _OWNER_ATTR)
        if hasattr(fsdp_param, _PROJECTION_ATTR):
            delattr(fsdp_param, _PROJECTION_ATTR)


__all__ = [
    "fsdp_current_unsharded_expert_parameters",
    "fsdp_binding_installed",
    "install_ultraep_fsdp_binding",
    "uninstall_ultraep_fsdp_binding",
    "writeback_fsdp_unsharded_expert_gradients",
]
