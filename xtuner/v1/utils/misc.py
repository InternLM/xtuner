import os
import sys
import threading
from functools import reduce
from math import lcm
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm


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
    by divisors."""
    if not divisors:
        raise ValueError("Divisors list cannot be empty")
    if 0 in divisors:
        raise ValueError("Divisors cannot contain zero")

    divisors_lcm: int = reduce(lcm, divisors)

    if length % divisors_lcm == 0:
        return 0

    padding = (divisors_lcm - (length % divisors_lcm)) % divisors_lcm
    return padding
