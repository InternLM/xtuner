import fcntl
import logging
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from packaging import version
from torch.distributed.checkpoint import FileSystemWriter, Metadata, SavePlan, SavePlanner
from torch.distributed.checkpoint._extension import (
    StreamTransformExtension,
)
from torch.distributed.checkpoint.filesystem import FileSystem
from torch.distributed.checkpoint.staging import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint.storage import (
    WriteResult,
)
from torch.futures import Future


logger = logging.getLogger(__name__)


# PyTorch 2.7+ introduced _extensions parameter for FileSystemWriter
_TORCH_DCP_FSWRITER_HAS_EXTENSIONS = version.parse(torch.__version__) >= version.parse("2.7.0")


def _compare_write_results(write_results: list[WriteResult], other_write_results: list[WriteResult]) -> bool:
    """Compare two lists of WriteResults for equality.

    Args:
        write_results: First list of WriteResults to compare.
        other_write_results: Second list of WriteResults to compare.

    Returns:
        True if both lists have the same length and all elements are equal,
        False otherwise.
    """

    # Both the plans should have the same number of items
    if len(write_results) != len(other_write_results):
        return False

    # Both the plans should have the same write items.
    for write_item, other_write_item in zip(write_results, other_write_results):
        # Write item type should be same
        if write_item != other_write_item:
            return False

    return True


def _contains_new_write_results(results: list[list[WriteResult]]) -> bool:
    return any(delta_result for delta_result in results)


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _release_write_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _get_file_write_lock_slots() -> int:
    return max(0, int(os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_SLOTS", "1")))


def _get_file_write_lock_key(path: Union[str, os.PathLike]) -> str:
    path = Path(path)
    parts = path.parts
    if "checkpoints" in parts:
        run_root = Path(*parts[: parts.index("checkpoints")])
    else:
        run_root = path.parent
    return str(run_root).replace(os.sep, "_")


def _get_file_write_lock_dir() -> str:
    return "/dev/shm/xtuner_dcp_write_locks"


def _acquire_file_write_lock(slots: int, lock_key: str) -> tuple[int, int, float]:
    os.makedirs(_get_file_write_lock_dir(), exist_ok=True)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    start_slot = rank % slots
    start = time.time()
    while True:
        for offset in range(slots):
            slot = (start_slot + offset) % slots
            lock_path = os.path.join(_get_file_write_lock_dir(), f"{lock_key}.file-slot{slot}.lock")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd, slot, time.time() - start
            except BlockingIOError:
                os.close(fd)
        time.sleep(0.05)


class _FileLockingFileSystem(FileSystem):
    def __init__(self, slots: int, lock_key: str) -> None:
        self._slots = slots
        self._lock_key = lock_key

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str):
        if self._slots <= 0 or "w" not in mode:
            with super().create_stream(path, mode) as stream:
                yield stream
            return

        lock_fd, lock_slot, waited = _acquire_file_write_lock(self._slots, self._lock_key)
        if waited >= 1.0:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            logger.info(
                "[DCP async_save] acquired file write slot "
                f"{lock_slot}/{self._slots} after {waited:.2f}s rank={rank} file={path}"
            )
        try:
            with super().create_stream(path, mode) as stream:
                yield stream
        finally:
            _release_write_lock(lock_fd)


class XtunerCacheWriter(FileSystemWriter):
    # Save write results for the current rank as computed by `write_data` API
    # Cached on the local rank.
    _cache_write_results: dict[str, list[WriteResult]] = {}

    # Collection of all the write results from all the ranks.
    # This is the ``results`` input to the `finish` API.
    # Cached on the coordinator rank.
    _cached_all_write_results: dict[str, list[list[WriteResult]]] = {}

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Optional[Sequence[StreamTransformExtension]] = None,
        enable_write_result_caching: bool = False,
        cache_key_prefix: str = "",
    ) -> None:
        # Build kwargs conditionally to support both PyTorch 2.6 and 2.7+
        kwargs: dict[str, Any] = dict()
        if _TORCH_DCP_FSWRITER_HAS_EXTENSIONS:
            kwargs["_extensions"] = _extensions
        super().__init__(
            path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
            **kwargs,
        )
        file_write_lock_slots = _get_file_write_lock_slots()
        if file_write_lock_slots > 0:
            self.fs = _FileLockingFileSystem(file_write_lock_slots, _get_file_write_lock_key(path))
            if _is_rank0():
                logger.info(f"[DCP async_save] file-level write lock enabled slots={file_write_lock_slots}")

        self._enable_write_result_caching = enable_write_result_caching
        self._cached_write_results_key = cache_key_prefix + self.__class__.__name__

    def stage(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Stage tensors into a reusable shared-memory cache for async process
        saves.

        PyTorch's default BlockingAsyncStager creates a pinned CPU cache when cache_staged_state_dict=True. For async
        process checkpointing that cache is later handed to a subprocess; if it is not backed by shared memory,
        multiprocessing has to create/copy shared files during handoff. Creating the long-lived cache directly in shm
        lets later checkpoints reuse the same storage and update it in place.
        """

        # The staged state_dict is already CPU-resident. Disable DCP's
        # overlapping CPU loader so the checkpoint process does not create CUDA
        # streams or touch GPU memory while writing files.
        self.per_thread_copy_ahead = 0

        if not self.cache_staged_state_dict:
            staged_state_dict = _create_cpu_state_dict(state_dict, share_memory=True)
            return _copy_state_dict(state_dict, staged_state_dict, type_check=self.type_check)

        if self.state_dict_cache is None:
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                logger.info("[DCP async_save] creating shared-memory staged cache")
            self.state_dict_cache = _create_cpu_state_dict(
                state_dict,
                share_memory=True,
            )

        return _copy_state_dict(state_dict, self.state_dict_cache, type_check=self.type_check)

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        all_writes_fut = super().write_data(plan, planner)

        if self._enable_write_result_caching:
            all_writes_fut = self._get_write_future_with_caching(all_writes_fut)
        return all_writes_fut

    def _get_write_future_with_caching(self, all_writes_fut):
        new_fut: Future[list[WriteResult]] = Future()
        all_writes_fut.wait()

        if self._cached_write_results_key not in XtunerCacheWriter._cache_write_results:
            # Case 1: If the write results are not cached,.............
            XtunerCacheWriter._cache_write_results[self._cached_write_results_key] = all_writes_fut.value()
            new_fut.set_result(all_writes_fut.value())
        elif _compare_write_results(
            all_writes_fut.value(), XtunerCacheWriter._cache_write_results[self._cached_write_results_key]
        ):
            # Case 2: equal
            new_fut.set_result([])
        else:
            # Case 3: not equal
            XtunerCacheWriter._cache_write_results[self._cached_write_results_key] = all_writes_fut.value()
            new_fut.set_result(all_writes_fut.value())

        return new_fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self._enable_write_result_caching:
            results = self._get_results_from_caching(results)

        super().finish(metadata, results)

    def _get_results_from_caching(self, results: list[list[WriteResult]]):
        if self._cached_write_results_key not in XtunerCacheWriter._cached_all_write_results:
            # Case 1:
            XtunerCacheWriter._cached_all_write_results[self._cached_write_results_key] = results
        elif not _contains_new_write_results(results):
            # Case 2: no new
            results = XtunerCacheWriter._cached_all_write_results[self._cached_write_results_key]
        else:
            # Case 3: not equal TODO: merge
            XtunerCacheWriter._cached_all_write_results[self._cached_write_results_key] = results

        return results
