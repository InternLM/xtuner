"""Version-pinned FSDP2 landing adapter for MoonEP expert weights.

This is the only XTuner module allowed to know about ``_fully_shard``
internals.  The rest of the MoonEP integration only sees installation,
current-view, and uninstallation functions from this module.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
from torch.distributed.tensor import DTensor


_TARGET_TORCH_VERSION = "2.12.1+cu132"
_BINDING_ATTR = "_xtuner_moonep_landing"
_OWNER_ATTR = "_xtuner_moonep_fsdp_owner"
_PROJECTION_ATTR = "_xtuner_moonep_projection"
_FSDP_PARAM_ATTR = "_xtuner_moonep_fsdp_param"


def _init_direct_all_gather_outputs(
    fsdp_param: FSDPParam,
    all_gather_input_numels: list[int],
    all_gather_input_dtypes: list[torch.dtype],
    world_size: int,
    device: torch.device,
    force_recreate: bool = False,
) -> None:
    """Point FSDP's final per-parameter unpack at the fixed VMM landing."""
    del force_recreate
    landing = getattr(fsdp_param, _BINDING_ATTR)
    if (
        len(all_gather_input_numels) != 1
        or len(all_gather_input_dtypes) != 1
        or all_gather_input_numels[0] * world_size != landing.numel()
        or all_gather_input_dtypes[0] is not landing.dtype
        or device != landing.device
    ):
        raise RuntimeError("MoonEP direct landing no longer matches FSDP AllGather metadata")
    fsdp_param.all_gather_outputs = [landing.view(-1)]


def _keep_direct_all_gather_storage(fsdp_param: FSDPParam) -> None:
    """FSDP must not resize/free runtime-owned, non-resizable VMM storage."""
    del fsdp_param


def install_fsdp_vmm_landing(
    *,
    fsdp_root: nn.Module,
    targets: Sequence[tuple[str, tuple[nn.Module, nn.Module], tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[FSDPParam, ...]:
    """Bind routed expert FSDPParams to their two-generation VMM landings.

    Each target is ``(layer_fqn, projections, landings)``.
    Matching uses the original module and parameter-name identities recorded
    by FSDP, never an FQN guess.
    """
    if torch.__version__ != _TARGET_TORCH_VERSION:
        raise RuntimeError(
            f"MoonEP direct FSDP landing requires torch {_TARGET_TORCH_VERSION}, got {torch.__version__}"
        )

    by_identity: dict[tuple[int, str], tuple[FSDPParam, nn.Module]] = {}
    for fsdp_owner in fsdp_root.modules():
        state = _get_module_fsdp_state(fsdp_owner)
        if state is None:
            continue
        # The 2.12 runtime has the plural list; its bundled type stub still
        # exposes only the deprecated singular compatibility property.
        for param_group in cast(Any, state)._fsdp_param_groups:
            for fsdp_param in param_group.fsdp_params:
                key = (id(fsdp_param._module_info.module), fsdp_param._module_info.param_name)
                if key in by_identity:
                    raise RuntimeError("MoonEP found duplicate FSDP parameter identity")
                by_identity[key] = fsdp_param, fsdp_owner

    selected: list[tuple[FSDPParam, nn.Module, nn.Module, torch.Tensor]] = []
    for layer_fqn, projections, landings in targets:
        for projection_name, projection, landing in zip(
            ("fused_w1w3", "fused_w2"), projections, landings, strict=True
        ):
            if hasattr(projection, _FSDP_PARAM_ATTR):
                raise RuntimeError(f"MoonEP direct landing is already installed for {layer_fqn}.{projection_name}")
            match = by_identity.get((id(projection), "weight"))
            if match is None:
                raise RuntimeError(f"MoonEP could not find FSDPParam for {layer_fqn}.{projection_name}.weight")
            fsdp_param, fsdp_owner = match
            expected_dtype = fsdp_param.mp_policy.param_dtype or fsdp_param.sharded_param.dtype
            shard_world_size = fsdp_param.mesh_info.shard_mesh_size
            unpadded_numel = fsdp_param._orig_size.numel()
            gathered_numel = fsdp_param.padded_sharded_param_size.numel() * shard_world_size
            if fsdp_param.fsdp_placement.dim != 0 or not landing.is_contiguous():
                raise RuntimeError(f"MoonEP requires contiguous dim-0 FSDP layout for {layer_fqn}.{projection_name}")
            if unpadded_numel != gathered_numel:
                raise RuntimeError(
                    f"MoonEP direct landing does not support FSDP padding for {layer_fqn}.{projection_name}"
                )
            if landing.numel() != unpadded_numel or landing.dtype is not expected_dtype:
                raise RuntimeError(f"MoonEP VMM landing metadata mismatch for {layer_fqn}.{projection_name}")
            if landing.device != fsdp_param.device or shard_world_size <= 1:
                raise RuntimeError(
                    f"MoonEP direct landing requires multi-rank CUDA FSDP for {layer_fqn}.{projection_name}"
                )
            if hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather"):
                raise RuntimeError(f"MoonEP direct landing does not support post-AllGather extensions: {layer_fqn}")
            if fsdp_param.sharded_state is not ShardedState.SHARDED or fsdp_param.all_gather_outputs:
                raise RuntimeError(f"MoonEP direct landing must be installed before first AllGather: {layer_fqn}")
            if any(
                name in fsdp_param.__dict__
                for name in ("init_all_gather_outputs", "alloc_all_gather_outputs", "free_unsharded_param")
            ):
                raise RuntimeError(f"MoonEP refuses an already customized FSDPParam: {layer_fqn}")
            selected.append((fsdp_param, fsdp_owner, projection, landing))

    if len({id(item[0]) for item in selected}) != len(selected):
        raise RuntimeError("MoonEP routed expert targets must map to distinct FSDPParams")

    for fsdp_param, fsdp_owner, projection, landing in selected:
        setattr(projection, _FSDP_PARAM_ATTR, fsdp_param)
        setattr(fsdp_param, _BINDING_ATTR, landing)
        setattr(fsdp_param, _OWNER_ATTR, fsdp_owner)
        setattr(fsdp_param, _PROJECTION_ATTR, projection)
        fsdp_param.init_all_gather_outputs = types.MethodType(  # type: ignore[method-assign]
            _init_direct_all_gather_outputs, fsdp_param
        )
        fsdp_param.alloc_all_gather_outputs = types.MethodType(  # type: ignore[method-assign]
            _keep_direct_all_gather_storage, fsdp_param
        )
        fsdp_param.free_unsharded_param = types.MethodType(  # type: ignore[method-assign]
            _keep_direct_all_gather_storage, fsdp_param
        )
    return tuple(item[0] for item in selected)


def fsdp_current_unsharded_expert_parameters(
    projections: tuple[nn.Module, nn.Module],
) -> tuple[nn.Parameter, nn.Parameter]:
    """Return the current FSDP leaf Parameters without starting an
    AllGather."""
    current_parameters: list[nn.Parameter] = []
    for projection in projections:
        fsdp_param = getattr(projection, _FSDP_PARAM_ATTR, None)
        if fsdp_param is None:
            raise RuntimeError("MoonEP direct FSDP landing is not installed for this expert projection")
        if fsdp_param.sharded_state is not ShardedState.UNSHARDED:
            raise RuntimeError("MoonEP expert weight was read outside its FSDP unsharded window")
        registered = getattr(fsdp_param._module_info.module, fsdp_param._module_info.param_name)
        if registered is not fsdp_param.unsharded_param:
            raise RuntimeError("MoonEP observed an unexpected FSDP Parameter switch")
        if not isinstance(registered, nn.Parameter):
            raise RuntimeError("MoonEP expected FSDP to expose an unsharded Parameter")
        local = registered.to_local() if isinstance(registered, DTensor) else registered
        landing = getattr(fsdp_param, _BINDING_ATTR)
        if local.data_ptr() != landing.data_ptr() or local.numel() != landing.numel():
            raise RuntimeError("MoonEP unsharded FSDP view no longer aliases its VMM landing")
        current_parameters.append(registered)
    return current_parameters[0], current_parameters[1]


def accumulate_fsdp_unsharded_expert_gradients(
    parameters: tuple[nn.Parameter, nn.Parameter],
    local_gradients: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Hand completed home gradients to the Parameters consumed by FSDP."""
    with torch.no_grad():
        for parameter, local_gradient in zip(parameters, local_gradients, strict=True):
            local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            local_gradient = local_gradient.reshape(local_parameter.shape)
            if isinstance(parameter, DTensor):
                gradient: torch.Tensor = DTensor.from_local(
                    local_gradient,
                    parameter.device_mesh,
                    parameter.placements,
                    run_check=False,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
            else:
                gradient = local_gradient

            # The first producer transfers storage ownership without a copy;
            # Domino's later producers use the same accumulation semantics as
            # native AccumulateGrad.
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)


def uninstall_fsdp_vmm_landing(fsdp_params: tuple[FSDPParam, ...]) -> None:
    """Restore native instance methods and release every FSDP VMM reference."""
    owners = {getattr(fsdp_param, _OWNER_ATTR) for fsdp_param in fsdp_params}
    for owner in owners:
        state = _get_module_fsdp_state(owner)
        if state is None or state._training_state.name != "IDLE":
            raise RuntimeError("MoonEP direct landing may only be removed at an idle FSDP boundary")
        # Public FSDPModule operation: swap modules back to their sharded
        # Parameters before removing the VMM-backed unsharded Parameter.
        owner.reshard()

    projections = {getattr(fsdp_param, _PROJECTION_ATTR) for fsdp_param in fsdp_params}
    for fsdp_param in fsdp_params:
        if fsdp_param.sharded_state is not ShardedState.SHARDED:
            raise RuntimeError("MoonEP failed to reshard a bound FSDP parameter")
        fsdp_param.all_gather_outputs.clear()
        fsdp_param._unsharded_inner_tensors.clear()
        if hasattr(fsdp_param, "_unsharded_param"):
            del fsdp_param._unsharded_param
        for method_name in ("init_all_gather_outputs", "alloc_all_gather_outputs", "free_unsharded_param"):
            delattr(fsdp_param, method_name)
        for attr_name in (_BINDING_ATTR, _OWNER_ATTR, _PROJECTION_ATTR):
            delattr(fsdp_param, attr_name)
    for projection in projections:
        delattr(projection, _FSDP_PARAM_ATTR)


__all__ = [
    "accumulate_fsdp_unsharded_expert_gradients",
    "fsdp_current_unsharded_expert_parameters",
    "install_fsdp_vmm_landing",
    "uninstall_fsdp_vmm_landing",
]
