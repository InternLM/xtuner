import fcntl
import logging
import os
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from packaging import version
from torch.distributed.checkpoint import FileSystemWriter, Metadata, SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint._extension import (
    StreamTransformExtension,
)
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
    return max(0, int(os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_SLOTS", "0")))


def _get_file_write_lock_key() -> str:
    return os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_KEY", "default")


def _get_file_write_lock_dir() -> str:
    return os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_DIR", "/dev/shm/xtuner_dcp_write_locks")


def _acquire_file_write_lock(slots: int) -> tuple[int, int, float]:
    lock_dir = _get_file_write_lock_dir()
    lock_key = _get_file_write_lock_key()
    os.makedirs(lock_dir, exist_ok=True)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    start_slot = rank % slots
    start = time.time()
    while True:
        for offset in range(slots):
            slot = (start_slot + offset) % slots
            lock_path = os.path.join(lock_dir, f"{lock_key}.file-slot{slot}.lock")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd, slot, time.time() - start
            except BlockingIOError:
                os.close(fd)
        time.sleep(0.05)


class _FileLevelWriteLockContext(AbstractContextManager):
    def __init__(self, stream_cm, slots: int, file_name: Any) -> None:
        self._stream_cm = stream_cm
        self._slots = slots
        self._file_name = file_name
        self._lock_fd: int | None = None
        self._lock_slot: int | None = None
        self._stream = None

    def __enter__(self):
        self._lock_fd, self._lock_slot, waited = _acquire_file_write_lock(self._slots)
        wait_log_threshold = float(os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_LOG_WAIT_SECONDS", "1.0"))
        if waited >= wait_log_threshold:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            logger.info(
                "[DCP async_save] acquired file write slot "
                f"{self._lock_slot}/{self._slots} after {waited:.2f}s rank={rank} file={self._file_name}"
            )
        self._stream = self._stream_cm.__enter__()
        return self._stream

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        try:
            return self._stream_cm.__exit__(exc_type, exc, tb)
        finally:
            if self._lock_fd is not None:
                _release_write_lock(self._lock_fd)
                self._lock_fd = None


def _wrap_create_stream_with_file_write_lock(create_stream, slots: int):
    def _create_stream(file_name, mode="rb", *args, **kwargs):
        stream_cm = create_stream(file_name, mode, *args, **kwargs)
        if slots <= 0 or "w" not in mode:
            return stream_cm
        return _FileLevelWriteLockContext(stream_cm, slots=slots, file_name=file_name)

    return _create_stream


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
        self._enable_write_result_caching = enable_write_result_caching
        self._cached_write_results_key = cache_key_prefix + self.__class__.__name__

    def stage(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Stage tensors into a reusable shared-memory cache for async process saves.

        PyTorch's default BlockingAsyncStager creates a pinned CPU cache when
        cache_staged_state_dict=True. For async process checkpointing that cache
        is later handed to a subprocess; if it is not backed by shared memory,
        multiprocessing has to create/copy shared files during handoff. Creating
        the long-lived cache directly in shm lets later checkpoints reuse the
        same storage and update it in place.
        """

        share_memory_cache = os.environ.get("ASYNC_DCP_SHARE_MEMORY", "0") not in {"0", "false", "False"}
        if not share_memory_cache:
            return super().stage(state_dict)

        if not self.cache_staged_state_dict:
            staged_state_dict = _create_cpu_state_dict(state_dict, share_memory=True)
            return _copy_state_dict(state_dict, staged_state_dict, type_check=self.type_check)

        if self.state_dict_cache is None:
            pin_shared_cache = os.environ.get("ASYNC_DCP_SHARE_MEMORY_PINNED", "0") not in {
                "0",
                "false",
                "False",
            }
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(
                    "[DCP async_save] creating shared-memory staged cache "
                    f"pin_memory={pin_shared_cache}"
                )
            self.state_dict_cache = _create_cpu_state_dict(
                state_dict,
                pin_memory=pin_shared_cache,
                share_memory=True,
            )

        return _copy_state_dict(state_dict, self.state_dict_cache, type_check=self.type_check)

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        file_write_lock_slots = _get_file_write_lock_slots()
        if file_write_lock_slots > 0 and _is_rank0():
            logger.info(
                "[DCP async_save] file-level write lock enabled "
                f"slots={file_write_lock_slots} key={_get_file_write_lock_key()}"
            )

        original_create_stream = None
        if file_write_lock_slots > 0:
            original_create_stream = self.fs.create_stream
            self.fs.create_stream = _wrap_create_stream_with_file_write_lock(
                original_create_stream,
                slots=file_write_lock_slots,
            )

        try:
            all_writes_fut = super().write_data(plan, planner)
        finally:
            if original_create_stream is not None:
                self.fs.create_stream = original_create_stream

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
