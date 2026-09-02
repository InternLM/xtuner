import os
from typing import Literal


XTUNER_DISPATCHER_DEBUG = os.getenv("XTUNER_DISPATCHER_DEBUG", "0") == "1"

import torch.distributed as dist
from torch import nn

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


# TODO: (yehaochen) This interface declaration does not follow the Liskov Substitution Principle.
# Maybe we should find a better way to handle the dispatchers.
def build_dispatcher(
    dispatcher: Literal["deepep", "all2all", "agrs", "moonep"] | None,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup | None = None,
    tp_group: dist.ProcessGroup | None = None,
    ep_tp_group: dist.ProcessGroup | None = None,
    *,
    moonep_runtime=None,
    layer_fqn: str | None = None,
    projections: tuple[nn.Module, nn.Module] | None = None,
) -> DispacherInterface:
    if dispatcher == "moonep":
        if ep_group is None or ep_group.size() not in (2, 4, 8):
            raise ValueError("MoonEP requires ep_size in {2, 4, 8}")
        if n_routed_experts % ep_group.size():
            raise ValueError("MoonEP requires n_routed_experts divisible by ep_size")
        if moonep_runtime is None or layer_fqn is None or projections is None:
            raise ValueError("MoonEP runtime, layer_fqn, and expert projections are required")
        return moonep_runtime.build_dispatcher(
            layer_fqn=layer_fqn,
            projections=projections,
        )  # type: ignore[return-value]

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
    "build_dispatcher",
    "PreDispatchResult",
    "DispatchResult",
    "ExpertWeightLayout",
    "PostDispatchResult",
    "PreCombineResult",
    "CombineResult",
    "PostCombineResult",
]
