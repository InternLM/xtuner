import os
import sys
import threading
from functools import reduce
from math import lcm
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from pathlib import Path
from typing import Annotated

from huggingface_hub import constants
from mmengine import is_installed

import transformers
from transformers import AutoConfig
from xtuner.v1.utils import get_logger


HF_PATCH_MODULES_CACHE_PREFIX = "modules_pid_"

logger = get_logger()
XTUNER_DETERMINISTIC = os.getenv("XTUNER_DETERMINISTIC") == "true"

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


def is_hf_model_path(path: str | Path) -> bool:
    try:
        AutoConfig.from_pretrained(path, trust_remote_code=True)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.debug(f"Model path {path} is not a valid HuggingFace model path. Error: {e}")
        return False
    else:
        return True


def monkey_patch_hf_modules_cache():
    # 如果在hf中tokenizer、config等使用remote_code，例如 `AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)`，
    # 会将hf_model_path 拷贝到 HF_MODULES_CACHE 中。 如果单机八卡机器上多个进程同时读写此目录，会导致冲突。
    # 因此需要将 HF_MODULES_CACHE 设置为当前进程的临时目录。
    modules_cache = os.path.join(constants.HF_HOME, f"{HF_PATCH_MODULES_CACHE_PREFIX}{os.getpid()}")
    os.environ["HF_MODULES_CACHE"] = modules_cache
    transformers.utils.hub.HF_MODULES_CACHE = modules_cache
    # 在 import 时刻，Python 会在 dynamic_module_utils 模块的命名空间中创建一个新的名字 HF_MODULES_CACHE，
    # 并将其绑定到 transformers.utils.HF_MODULES_CACHE 当时所指向的对象。
    # 因此，需要将 transformers.dynamic_module_utils.HF_MODULES_CACHE 也设置为新的 modules_cache。
    transformers.dynamic_module_utils.HF_MODULES_CACHE = modules_cache
    transformers.utils.HF_MODULES_CACHE = modules_cache
    logger.info(f"set HF_MODULES_CACHE to {modules_cache} for current process {os.getpid()}")
