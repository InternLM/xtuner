import random
from typing import Any
from unittest.mock import patch

import pytest
import torch

from xtuner.v1.utils.pinned_memory import allocate_pinned_like, plan_pinned_slabs


class TestPinnedMemory:
    def test_plan_bounds_and_empty(self) -> None:
        assert plan_pinned_slabs([0, 6, 4, 3, 7, 12, 1], 10) == [[5], [4, 3, 0], [1, 2], [6]]
        assert plan_pinned_slabs([]) == []
        assert plan_pinned_slabs([0, 0]) == [[0, 1]]
        with pytest.raises(ValueError):
            plan_pinned_slabs([-1])
        with pytest.raises(ValueError):
            plan_pinned_slabs([1], 0)

    def test_plan_matches_reference_best_fit_ties(self) -> None:
        rng = random.Random(42)
        for _ in range(1000):
            cap = rng.randint(1, 128)
            sizes = [rng.randint(0, cap * 2) for _ in range(rng.randint(0, 100))]
            expected: list[list[int]] = []
            used: list[int] = []
            for index in sorted(range(len(sizes)), key=lambda index: (-sizes[index], index)):
                size = sizes[index]
                fitting = [i for i, total in enumerate(used) if total <= cap and total + size <= cap]
                if fitting:
                    best = min(fitting, key=lambda i: (cap - used[i] - size, i))
                    expected[best].append(index)
                    used[best] += size
                else:
                    expected.append([index])
                    used.append(size)
            assert plan_pinned_slabs(sizes, cap) == expected

    def test_layout_zeroing_and_nonoverlap(self) -> None:
        tensors = [
            torch.empty(3, 5).t(),
            torch.empty(0, 3),
            torch.empty(6, 8)[:, ::2],
            torch.empty(7, dtype=torch.float64),
        ]
        original = torch.empty
        allocated = []

        def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            if kwargs.pop("pin_memory", False):
                allocated.append(args[0])
            return original(*args, **kwargs)

        with patch.object(torch, "empty", empty):
            views = allocate_pinned_like(tensors, zero=True)
        assert len(allocated) == 2
        spans = {}
        for source, view in zip(tensors, views):
            assert view.shape == source.shape and view.dtype == source.dtype
            assert view.stride() == torch.empty_like(source).stride()
            assert torch.count_nonzero(view) == 0
            spans.setdefault(view.untyped_storage().data_ptr(), []).append(
                (view.storage_offset(), view.storage_offset() + view.numel())
            )
        for ranges in spans.values():
            ranges.sort()
            assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        assert allocate_pinned_like([]) == []

    def test_copies_preserve_layout_and_do_not_capture_autograd(self) -> None:
        tensors = [torch.arange(15.0).reshape(3, 5).t().requires_grad_(), torch.ones(1, 2).as_strided((1, 2), (99, 1))]
        original = torch.empty

        def empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            kwargs.pop("pin_memory", None)
            return original(*args, **kwargs)

        with patch.object(torch, "empty", empty):
            views = allocate_pinned_like(tensors, copy_non_blocking=True)
        for source, view in zip(tensors, views):
            torch.testing.assert_close(view, source)
            assert view.stride() == torch.empty_like(source).stride()
            assert not view.requires_grad and view.grad_fn is None
        with pytest.raises(ValueError):
            allocate_pinned_like(tensors, zero=True, copy_non_blocking=True)
