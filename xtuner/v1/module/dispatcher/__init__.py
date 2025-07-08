from typing import Literal

from .base import DecodingDispatchResult, DispacherInterface, NaiveDispacher, PrefillingDispatchResult
from .torch_all2all import TorchAll2AllDispatcher


# TODO: (yehaochen) This interface declaration does not follow the Liskov Substitution Principle.
# Maybe we should find a better way to handle the dispatchers.
def get_dispatcher(name: Literal["deepep", "naive", "all2all"]) -> type[DispacherInterface]:
    if name == "deepep":
        from .deepep import DeepEPDispatcher

        return DeepEPDispatcher  # type: ignore[return-value]
    elif name == "naive":
        return NaiveDispacher  # type: ignore[return-value]
    elif name == "all2all":
        return TorchAll2AllDispatcher  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown dispatcher name: {name}, name must be one of 'deepep', 'naive', or 'all2all'.")
