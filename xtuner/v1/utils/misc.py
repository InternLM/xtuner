import os
import socket
import sys
import threading
from functools import reduce
from math import lcm
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from pathlib import Path
from types import FunctionType
from typing import Annotated

import torch
from huggingface_hub import constants
from mmengine import is_installed

import transformers
from transformers import AutoConfig

from .enum_helper import StrEnum
from .logger import get_logger


HF_PATCH_MODULES_CACHE_PREFIX = "modules_cache"

logger = get_logger()
XTUNER_DETERMINISTIC = os.getenv("XTUNER_DETERMINISTIC") == "true"


def set_deterministic():
    if XTUNER_DETERMINISTIC:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


# https://github.com/python/cpython/issues/82300#issuecomment-2169035092
if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:

    class SharedMemory(_mpshm.SharedMemory):
        __lock = threading.Lock()

        def __init__(
            self, name: str | None = None, create: bool = False, size: int = 0, *, track: bool = False
        ) -> None:
            self._track = track

            # if tracking, normal init will suffice
            if track:
                return super().__init__(name=name, create=create, size=size)

            # lock so that other threads don't attempt to use the
            # register function during this time
            with self.__lock:
                # temporarily disable registration during initialization
                orig_register = _mprt.register
                _mprt.register = self.__tmp_register

                # initialize; ensure original register function is
                # re-instated
                try:
                    super().__init__(name=name, create=create, size=size)
                finally:
                    _mprt.register = orig_register

        @staticmethod
        def __tmp_register(*args, **kwargs) -> None:
            return

        def unlink(self) -> None:
            # TODO: (yehaochen) Even though accessing some non-public attributes is unsafe, no better solution has
            # been found yet, not sure if it works properly across Python 3.10-Python 3.12 versions.
            if _mpshm._USE_POSIX and self._name:  # type: ignore[attr-defined]
                _mpshm._posixshmem.shm_unlink(self._name)  # type: ignore[attr-defined]
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")  # type: ignore[attr-defined]


def get_padding_length(length: int, divisors: list[int]) -> int:
    """Calculate the padding length needed to make the input length divisible
    by divisors.

    Args:
        length: The input length to be padded.
        divisors: A list of integers that the length should be divisible by.

    Returns:
        The padding length needed to make the input length divisible by all
        divisors. Returns 0 if already divisible.

    Raises:
        ValueError: If divisors list is empty or contains zero.
    """
    if not divisors:
        raise ValueError("Divisors list cannot be empty")
    if 0 in divisors:
        raise ValueError("Divisors cannot contain zero")

    divisors_lcm: int = reduce(lcm, divisors)

    if length % divisors_lcm == 0:
        return 0

    padding = (divisors_lcm - (length % divisors_lcm)) % divisors_lcm
    return padding


def record_git_info(staged_path: Path, unstaged_path: Path) -> Annotated[str, "Commit"]:
    if not is_installed("git"):
        return "null"

    from git import InvalidGitRepositoryError, Repo

    import xtuner

    try:
        repo = Repo(str(Path(next(iter(xtuner.__path__))).parent))
    except InvalidGitRepositoryError:
        return "null"

    commit = str(repo.commit())
    staged = repo.git.diff("--cached")
    unstaged = repo.git.diff()

    with staged_path.open("w") as f:
        f.write(staged)

    with unstaged_path.open("w") as f:
        f.write(unstaged)
    return commit


def is_hf_model_path(path: str | Path) -> tuple[bool, Exception | None]:
    try:
        AutoConfig.from_pretrained(path, trust_remote_code=True)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        return False, e
    else:
        return True, None


def monkey_patch_hf_modules_cache():
    # When using remote_code in HF for tokenizer, config, etc., e.g., `AutoConfig.from_pretrained(hf_model_path,
    # trust_remote_code=True)`, the hf_model_path will be copied to HF_MODULES_CACHE. If multiple processes read/write
    # this directory simultaneously, it will cause conflicts. Therefore, we need to set HF_MODULES_CACHE to a unique
    # temporary directory (identified w/ hostname + pid) for the current process.
    hostname = socket.gethostname()
    pid = os.getpid()
    modules_cache = os.path.join(constants.HF_HOME, f"{HF_PATCH_MODULES_CACHE_PREFIX}_{hostname}_{pid}")
    os.environ["HF_MODULES_CACHE"] = modules_cache
    transformers.utils.hub.HF_MODULES_CACHE = modules_cache
    # At import time, Python creates a new name HF_MODULES_CACHE in the dynamic_module_utils module's namespace,
    # binding it to the object that transformers.utils.HF_MODULES_CACHE pointed to at that moment.
    # Therefore, we need to set transformers.dynamic_module_utils.HF_MODULES_CACHE to the new modules_cache as well.
    transformers.dynamic_module_utils.HF_MODULES_CACHE = modules_cache
    transformers.utils.HF_MODULES_CACHE = modules_cache
    logger.info(f"set HF_MODULES_CACHE to {modules_cache} for current process (hostname={hostname}, pid={pid})")


class FunctionEnum(StrEnum):
    TOP_LEVEL_FUNCTION = "top_level_function"
    CLASS_LEVEL_FUNCTION = "class_level_function"
    LOCAL_FUNCTION = "local_function"


def get_function_type(func: FunctionType):
    if not isinstance(func, FunctionType):
        raise TypeError(f"Expected a function or method, but got {type(func)}")

    qualname = func.__qualname__
    parts = qualname.split(".")
    # 检查是否恰好有两部分，且没有 <locals> 标记
    if len(parts) != 1:
        if "<locals>" in parts:
            return FunctionEnum.LOCAL_FUNCTION
        else:
            return FunctionEnum.CLASS_LEVEL_FUNCTION
    else:
        return FunctionEnum.TOP_LEVEL_FUNCTION


def get_function_full_qualname(function: FunctionType) -> str:
    """Get the full qualified name of a function, including module and class
    names if applicable.

    Args:
        function: The function to get the qualified name for.

    Returns:
        The full qualified name of the function.
    """
    if isinstance(function, FunctionType):
        module_name = function.__module__
        qualname = function.__qualname__
    # For `xtuner.v1.utils.compile.MaybeCompile`, using attributes check to avoid from circular import.
    elif hasattr(function, "origin_func"):
        module_name = function.origin_func.__module__
        qualname = function.origin_func.__qualname__
    else:
        raise TypeError(f"Expected a `function` or `MaybeCompile`, but got {type(function)}")

    full_qualname = f"{module_name}.{qualname}"
    return full_qualname


def clean_param_name(name: str) -> str:
    if "_checkpoint_wrapped_module." in name:
        name = name.replace("_checkpoint_wrapped_module.", "")
    if "_orig_mod." in name:
        name = name.replace("_orig_mod.", "")
    return name


_TRIM_MEMORY_WARNED = False


def trim_memory() -> bool:
    """Try to return free heap pages to OS.

    Best-effort only: on platforms without `malloc_trim` (or when unavailable),
    this will fail. We log the failure once per process to avoid spamming.
    """
    global _TRIM_MEMORY_WARNED
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0)
    except Exception as e:
        if not _TRIM_MEMORY_WARNED:
            _logger = get_logger()
            _logger.warning(f" >>>>>>>>> [trim_memory] Failed to trim memory: {e} <<<<<<<<")
            _TRIM_MEMORY_WARNED = True
        return False
