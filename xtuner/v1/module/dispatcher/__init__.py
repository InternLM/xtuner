from typing import cast

import torch.distributed as dist

from xtuner.v1.config import MoEConfig
from xtuner.v1.utils import get_logger

from .base import DecodingDispatchResult, DispacherInterface, NaiveDispatcher, PrefillingDispatchResult
from .torch_all2all import TorchAll2AllDispatcher


logger = get_logger()


# TODO: (yehaochen) This interface declaration does not follow the Liskov Substitution Principle.
# Maybe we should find a better way to handle the dispatchers.
def build_dispatcher(config: MoEConfig, ep_group: dist.ProcessGroup | None = None) -> DispacherInterface:
    dispatcher = config.dispatcher
    if ep_group is None:
        if dispatcher is not None:
            logger.warning(f"{dispatcher} will not be used because the ep group is None.")
        return NaiveDispatcher(config=config)  # type: ignore[return-value]
    elif (ep_group is not None and ep_group.size() == 1) or dispatcher is None:
        return NaiveDispatcher(config=config, process_group=ep_group)  # type: ignore[return-value]
    if dispatcher == "deepep":
        from .deepep import DeepEPDispatcher

        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        return DeepEPDispatcher(config=config, process_group=ep_group)  # type: ignore[return-value]
    elif dispatcher == "all2all":
        assert ep_group is not None, "DeepEPDispatcher requires a non-null process group."
        return TorchAll2AllDispatcher(config=config, process_group=ep_group)  # type: ignore[return-value]
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
