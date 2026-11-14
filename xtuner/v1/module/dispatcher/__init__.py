from __future__ import annotations

import os
from typing import Any, Literal, Protocol


XTUNER_DISPATCHER_DEBUG = os.getenv("XTUNER_DISPATCHER_DEBUG", "0") == "1"

import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.utils import get_logger, log_rank0

from .agrs import MoEAGRSDispatcher
from .base import (
    CombineResult,
    DispacherInterface,
    DispatchResult,
    ExpertWeightLayout,
    NaiveDispatcher,
    PostCombineResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
)
from .torch_all2all import TorchAll2AllDispatcher


logger = get_logger()


class EPExecutionRuntime(Protocol):
    """Model-scoped EP execution lifecycle: four boundaries ``MoE`` calls
    unconditionally, with no shared implementation (MoonEP, or the no-op
    Adapter below).

    ``bind_layer`` returns this backend's per-layer Dispatcher, or ``None`` to
    let ``build_dispatcher`` fall back to a generic Adapter.
    """

    def bind_layer(self, *, layer_fqn: str, projections: tuple[nn.Module, nn.Module]) -> Any: ...

    def validate_before_fsdp(self, fsdp_config: object) -> None: ...

    def install_after_fsdp(self, *, fsdp_root: nn.Module, execution_order: list[str]) -> None: ...

    def close(self) -> None: ...


class NoEPExecutionRuntime:
    """The "no model-scoped EP execution runtime" Adapter (not ``None``).

    Every EP-runtime lifecycle boundary on ``MoE`` is called unconditionally;
    for a backend without one, each boundary is a no-op and ``bind_layer``
    returns ``None`` so ``build_dispatcher`` falls back to a generic Adapter.
    """

    def bind_layer(self, *, layer_fqn: str, projections: tuple[nn.Module, nn.Module]) -> None:
        del layer_fqn, projections
        return None

    def validate_before_fsdp(self, fsdp_config: object) -> None:
        del fsdp_config

    def install_after_fsdp(self, *, fsdp_root: nn.Module, execution_order: list[str]) -> None:
        del fsdp_root, execution_order

    def close(self) -> None:
        return


def build_ep_execution_runtime(config: Any, ep_mesh: DeviceMesh | None) -> EPExecutionRuntime:
    """Single place a model-scoped EP execution backend is selected.

    ``config`` is a ``MoEConfig``; it is typed loosely to avoid a model-layer
    import cycle.
    """
    if config.dispatcher == "moonep":
        from .moonep import MoonEPModelRuntime

        assert ep_mesh is not None
        return MoonEPModelRuntime(
            ep_group=ep_mesh.get_group(),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            intra_layer_micro_batch=config.intra_layer_micro_batch,
            staging_reference=config.moonep_staging_reference,
            num_sms=config.moonep_num_sms,
        )
    return NoEPExecutionRuntime()


# TODO: (yehaochen) This interface declaration does not follow the Liskov Substitution Principle.
# Maybe we should find a better way to handle the dispatchers.
def build_dispatcher(
    dispatcher: Literal["deepep", "all2all", "agrs", "moonep"] | None,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup | None = None,
    tp_group: dist.ProcessGroup | None = None,
    ep_tp_group: dist.ProcessGroup | None = None,
    *,
    ep_runtime: EPExecutionRuntime | None = None,
    layer_fqn: str | None = None,
    projections: tuple[nn.Module, nn.Module] | None = None,
) -> DispacherInterface:
    # A model-scoped EP execution runtime binds its own per-layer Dispatcher.
    if ep_runtime is not None and layer_fqn is not None and projections is not None:
        bound = ep_runtime.bind_layer(layer_fqn=layer_fqn, projections=projections)
        if bound is not None:
            return bound  # type: ignore[return-value]

    if ep_group is None or ep_group.size() == 1:
        if dispatcher is not None:
            log_rank0.warning(f"{dispatcher} will not be used because the ep group is None.")
        return NaiveDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            tp_group=tp_group,
        )  # type: ignore[return-value]

    if dispatcher is None:
        dispatcher = "all2all"

    if dispatcher == "deepep":
        # TODO: remove ignore
        from .deepep import DeepEPDispatcher  # type: ignore[attr-defined]

        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        # When expert TP is enabled, fuse EP dispatch + TP replication into a single DeepEP
        # collective: the dispatcher operates on the combined (ep × tp) group with each
        # physical expert virtualized into ``tp_size`` copies (see ``DeepEPDispatcher``).
        tp_size = tp_group.size() if tp_group is not None else 1
        if tp_size > 1:
            assert ep_tp_group is not None, (
                "DeepEPDispatcher with expert TP requires the combined (ep × tp) process group; "
                "pass ``ep_tp_group`` from ``ep_tp_mesh._flatten().get_group()``."
            )
            process_group = ep_tp_group
        else:
            process_group = ep_group
        # TODO: remove type ignore here
        return DeepEPDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            tp_size=tp_size,
        )  # type: ignore
    elif dispatcher == "all2all":
        assert ep_group is not None, "TorchAll2AllDispatcher requires a non-null ep_group."
        return TorchAll2AllDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            tp_group=tp_group,
        )  # type: ignore[return-value]
    elif dispatcher == "agrs":
        assert ep_group is not None, "MoEAGRSDispatcher requires a non-null process group."
        return MoEAGRSDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
        )  # type: ignore[return-value]
    else:
        raise ValueError(
            f"Unknown dispatcher name: {dispatcher}, name must be one of 'deepep', 'all2all', 'agrs', or 'moonep'."
        )


__all__ = [
    "DispacherInterface",
    "NaiveDispatcher",
    "TorchAll2AllDispatcher",
    "MoEAGRSDispatcher",
    "EPExecutionRuntime",
    "NoEPExecutionRuntime",
    "build_dispatcher",
    "build_ep_execution_runtime",
    "PreDispatchResult",
    "DispatchResult",
    "ExpertWeightLayout",
    "PostDispatchResult",
    "PreCombineResult",
    "CombineResult",
    "PostCombineResult",
]
