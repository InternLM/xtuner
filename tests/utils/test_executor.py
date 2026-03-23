from concurrent.futures import as_completed

import numpy as np
import pytest

from xtuner.v1.utils.executor import SharedPoolExecutor


def _add_offset(item: int, *, offset: int, scale: np.ndarray) -> int:
    return item * int(scale[0]) + offset


def _raise_on(item: int, *, threshold: int) -> int:
    if item == threshold:
        raise ValueError(f"item == {threshold}")
    return item


class TestSharedPoolExecutor:
    def test_map_basic(self):
        items = list(range(20))
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 10, "scale": np.array([2], dtype=np.int64)},
            max_workers=4,
            mp_context="fork",
        ) as pool:
            results = list(pool.map(items))

        assert results == [i * 2 + 10 for i in items]

    def test_map_preserves_order(self):
        items = list(range(100))
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 0, "scale": np.array([1], dtype=np.int64)},
            max_workers=8,
            mp_context="fork",
        ) as pool:
            results = list(pool.map(items))

        assert results == items

    def test_submit_basic(self):
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 1, "scale": np.array([3], dtype=np.int64)},
            max_workers=2,
            mp_context="fork",
        ) as pool:
            futures = [pool.submit(i) for i in range(10)]
            results = [f.result() for f in futures]

        assert results == [i * 3 + 1 for i in range(10)]

    def test_submit_as_completed(self):
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 0, "scale": np.array([1], dtype=np.int64)},
            max_workers=4,
            mp_context="fork",
        ) as pool:
            futures = [pool.submit(i) for i in range(20)]
            results = sorted(f.result() for f in as_completed(futures))

        assert results == list(range(20))

    def test_exception_propagates(self):
        with SharedPoolExecutor(
            fn=_raise_on,
            partial_kwargs={"threshold": 5},
            max_workers=2,
            mp_context="fork",
        ) as pool:
            futures = [pool.submit(i) for i in range(10)]
            with pytest.raises(ValueError, match="item == 5"):
                for f in futures:
                    f.result()

    def test_ndarray_shared(self):
        big = np.arange(1000, dtype=np.int64)
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 0, "scale": big[:1]},
            max_workers=4,
            mp_context="fork",
        ) as pool:
            results = list(pool.map(list(range(10))))

        assert results == [i * 0 for i in range(10)]

    def test_empty_iterable(self):
        with SharedPoolExecutor(
            fn=_add_offset,
            partial_kwargs={"offset": 0, "scale": np.array([1], dtype=np.int64)},
            max_workers=2,
            mp_context="fork",
        ) as pool:
            results = list(pool.map([]))

        assert results == []
