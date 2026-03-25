"""Unit tests for PresetSampler."""

import numpy as np
import pytest

from xtuner.v1.datasets.preset_sampler import PresetSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePackDataset:
    """Minimal stand-in: PresetSampler only calls len() on the dataset."""

    def __init__(self, num_packs: int):
        self._num_packs = num_packs

    def __len__(self) -> int:
        return self._num_packs


def _i64(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_single_rank():
    """Single-rank sampler yields all indices in the supplied order."""
    dataset = _FakePackDataset(5)
    global_order = _i64(2, 0, 4, 1, 3)

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=1)
    assert list(sampler) == [2, 0, 4, 1, 3]


def test_round_down():
    """global_order is truncated so its length is a multiple of global_batch_size."""
    dataset = _FakePackDataset(10)
    global_order = np.arange(7, dtype=np.int64)  # 7 items, batch_size=4 → use first 4

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=4)
    yielded = list(sampler)

    assert len(yielded) == 4
    assert yielded == [0, 1, 2, 3]
    assert all(0 <= idx < 10 for idx in yielded)


def test_invalid_order_out_of_range():
    """global_order containing an out-of-range index raises ValueError."""
    dataset = _FakePackDataset(5)

    with pytest.raises(ValueError, match="out of range"):
        PresetSampler(dataset, global_order=_i64(0, 1, 99), global_batch_size=1)


def test_load_from_npy_file_mmap(tmp_path):
    """PresetSampler accepts a .npy file path and keeps mmap-backed order."""
    dataset = _FakePackDataset(5)
    order = [2, 0, 1, 3, 4]

    npy_path = str(tmp_path / "sampler_order.npy")
    np.save(npy_path, np.array(order, dtype=np.int64))

    sampler = PresetSampler(dataset, global_order=npy_path, global_batch_size=1)
    assert list(sampler) == order
    assert isinstance(sampler.global_order, np.ndarray)
    assert isinstance(sampler.global_order, np.memmap)


def test_state_dict_resume():
    """Restoring a state dict causes __iter__ to resume from the saved offset."""
    dataset = _FakePackDataset(6)
    global_order = _i64(0, 1, 2, 3, 4, 5)

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=1)

    # Simulate 3 consumed samples (get_state_dict takes global consumed count as `step`).
    state = sampler.get_state_dict(step=3)
    assert state["step"] == 3

    sampler2 = PresetSampler(dataset, global_order=global_order, global_batch_size=1)
    sampler2.load_state_dict(state)

    assert list(sampler2) == [3, 4, 5]


def test_state_dict_world_size_mismatch():
    """Mismatched world_size in state dict logs a warning but does not raise."""
    dataset = _FakePackDataset(4)
    global_order = _i64(0, 1, 2, 3)

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=1)
    state = sampler.get_state_dict(step=0)
    state["world_size"] = 99  # force mismatch

    sampler.load_state_dict(state)  # must not raise


def test_repeated_packs():
    """global_order may reference the same pack more than once (over-sampling)."""
    dataset = _FakePackDataset(3)
    global_order = _i64(0, 0, 1, 1, 2, 2)

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=1)
    yielded = list(sampler)

    assert len(yielded) == global_order.size
    assert len(set(yielded)) < len(yielded)  # duplicates present


def test_len():
    """__len__ returns the per-rank number of samples."""
    dataset = _FakePackDataset(6)
    global_order = _i64(0, 1, 2, 3, 4, 5)

    sampler = PresetSampler(dataset, global_order=global_order, global_batch_size=1)
    assert len(sampler) == 6


def test_round_down_too_short_raises():
    """If global_order is shorter than global_batch_size*world_size, round-down yields 0."""
    dataset = _FakePackDataset(10)
    global_order = _i64(0, 1, 2)  # len 3, batch 4 * world 1 → 0

    with pytest.raises(ValueError, match="round down"):
        PresetSampler(dataset, global_order=global_order, global_batch_size=4)
