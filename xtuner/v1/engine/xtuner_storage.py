import os
from typing import Optional, Union
from collections.abc import Sequence

from torch.distributed.checkpoint import SavePlan, SavePlanner
from torch.distributed.checkpoint import FileSystemWriter, Metadata
from torch.distributed.checkpoint._extension import (
    StreamTransformExtension,
)
from torch.distributed.checkpoint.storage import (
    WriteResult,
)
from torch.futures import Future


def _compare_write_results(write_results: list[WriteResult], other_write_results: list[WriteResult]) -> bool:
    """
    Compare two lists of WriteResults for equality.

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
        super().__init__(
            path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
            _extensions=_extensions,
        )
        self._enable_write_result_caching = enable_write_result_caching
        self._cached_write_results_key = cache_key_prefix + self.__class__.__name__


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

        if self._cached_write_results_key not in XTunerCacheWriter._cache_write_results:
            # Case 1: If the write results are not cached,.............
            XTunerCacheWriter._cache_write_results[self._cached_write_results_key] = all_writes_fut.value()
            new_fut.set_result(all_writes_fut.value())
        elif _compare_write_results(all_writes_fut.value(), XTunerCacheWriter._cache_write_results[self._cached_write_results_key]):
            # Case 2: equal 
            new_fut.set_result([])
        else:
            # Case 3: not equal
            XTunerCacheWriter._cache_write_results[self._cached_write_results_key] = all_writes_fut.value()
            new_fut.set_result(all_writes_fut.value())

        return new_fut

    
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self._enable_write_result_caching:
            results = self._get_results_from_caching(results)

        super().finish(metadata, results)

    
    def _get_results_from_caching(self, results: list[list[WriteResult]]):
        if self._cached_write_results_key not in XTunerCacheWriter._cached_all_write_results:
            # Case 1:
            XTunerCacheWriter._cached_all_write_results[self._cached_write_results_key] = results
        elif not _contains_new_write_results(results):
            # Case 2: no new
            results = XTunerCacheWriter._cached_all_write_results[self._cached_write_results_key]
        else:
            # Case 3: not equal TODO: merge 
            XTunerCacheWriter._cached_all_write_results[self._cached_write_results_key] = results
        
        return results