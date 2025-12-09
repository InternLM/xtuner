from types import FunctionType
from typing import Callable, Generic, cast

import torch

from .device import get_device
from .logger import get_logger
from .misc import FunctionEnum, get_function_type
from .type_helper import P, T


logger = get_logger()


TARGET_DEVICE = get_device()


class MaybeCompile(Generic[P, T]):
    """A decorator class that can conditionally apply `torch.compile` to a **Top(Module) level function**.

    XTuner adopts a runtime compile strategy, which applies different compile strategies based on different model
    configurations, rather than using a decorator-based approach. However, this also makes it difficult to compile some
    module-level functions at runtime. This is because during the runtime phase, other modules have already completed
    the import of the target function, and at this point, compiling the function in the original module with in-place
    modification can no longer take effect in the target modules.

    Therefore, for these module-level functions, XTuner provides the MaybeCompile decorator class. By first decorating
    the target function with this class, it becomes possible to modify the calling behavior of all modules for that
    function at runtime.
    """

    def __init__(self, func: Callable[P, T]):
        assert isinstance(func, FunctionType), f"MaybeCompile can only be used to decorate functions, but got {func}"
        assert (function_type := get_function_type(func)) == FunctionEnum.TOP_LEVEL_FUNCTION, (
            "MaybeCompile can only be used to decorate top(module) level function, "
            f"but got type {function_type}: {func}"
        )
        func = cast(Callable[P, T], func)
        self.origin_func = func
        self.func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Apply the decorator with optional torch.compile arguments."""
        return self.func(*args, **kwargs)

    def enable_compile(self, **compile_options) -> None:
        """Enable torch.compile with the given arguments."""
        if not is_compiled_function(self.func):
            self.func = torch.compile(self.origin_func, **compile_options)

    def disable_compile(self) -> None:
        """Disable torch.compile, reverting to the original function."""
        logger.info(f"Disabling torch.compile for function {self.origin_func.__name__}")
        self.func = self.origin_func


def is_compiled_function(func: Callable) -> bool:
    """Check if a function has been compiled using torch.compile."""
    return hasattr(func, "get_compiler_config")


# Create a singleton instance
maybe_compile = MaybeCompile
