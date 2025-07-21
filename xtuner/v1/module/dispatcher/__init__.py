from typing import Literal

import torch.distributed as dist

from xtuner.v1.config.base_model import GenerateConfig
from xtuner.v1.utils import get_logger

from .base import DecodingDispatchResult, DispacherInterface, NaiveDispatcher, PrefillingDispatchResult
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
    if ep_group is None:
        if dispatcher is not None:
            logger.warning(f"{dispatcher} will not be used because the ep group is None.")
        return NaiveDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    if (ep_group is not None and ep_group.size() == 1) or dispatcher is None:
        return NaiveDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
    if dispatcher == "deepep":
        from .deepep import DeepEPDispatcher

        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        return DeepEPDispatcher(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )  # type: ignore[return-value]
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
    "DecodingDispatchResult",
    "build_dispatcher",
    "PrefillingDispatchResult",
]
