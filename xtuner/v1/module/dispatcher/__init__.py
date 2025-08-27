import os
from typing import Literal


XTUNER_DISPATCHER_DEBUG = os.getenv("XTUNER_DISPATCHER_DEBUG", "0") == "1"

import torch.distributed as dist

from xtuner.v1.utils import get_logger

from .base import (
    CombineResult,
    DispacherInterface,
    DispatchResult,
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
    dispatcher: Literal["deepep", "all2all"] | None,
    n_routed_experts: int,
    ep_group: dist.ProcessGroup | None = None,
    training_dtype: Literal["bf16", "fp8"] = "bf16",
    generate_dtype: Literal["bf16", "fp8"] = "bf16",
) -> DispacherInterface:
    if ep_group is None or ep_group.size() == 1:
        if dispatcher is not None:
            logger.warning(f"{dispatcher} will not be used because the ep group is None.")
        return NaiveDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]

    if dispatcher is None:
        dispatcher = "all2all"

    if dispatcher == "deepep":
        # TODO: remove ignore
        from .deepep import DeepEPDispatcher  # type: ignore[attr-defined]

        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        # TODO: remove type ignore here
        return DeepEPDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore
    elif dispatcher == "all2all":
        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        return TorchAll2AllDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown dispatcher name: {dispatcher}, name must be one of 'deepep' or 'all2all'.")


__all__ = [
    "DispacherInterface",
    "NaiveDispatcher",
    "TorchAll2AllDispatcher",
    "build_dispatcher",
    "PreDispatchResult",
    "DispatchResult",
    "PostDispatchResult",
    "PreCombineResult",
    "CombineResult",
    "PostCombineResult",
]
