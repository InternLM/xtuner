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


def _write_order_npy(tmp_path, name: str, arr: np.ndarray) -> str:
    p = tmp_path / name
    np.save(str(p), arr)
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_single_rank(tmp_path):
    """Single-rank sampler yields all indices in the file order."""
    dataset = _FakePackDataset(5)
    order = _i64(2, 0, 4, 1, 3)
    path = _write_order_npy(tmp_path, "order.npy", order)

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    assert list(sampler) == [2, 0, 4, 1, 3]


def test_round_down(tmp_path):
    """Sampler order is truncated so its length is a multiple of global_batch_size."""
    dataset = _FakePackDataset(10)
    order = np.arange(7, dtype=np.int64)
    path = _write_order_npy(tmp_path, "order.npy", order)

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=4)
    yielded = list(sampler)

    assert len(yielded) == 4
    assert yielded == [0, 1, 2, 3]
    assert all(0 <= idx < 10 for idx in yielded)


def test_invalid_order_out_of_range(tmp_path):
    """Order file containing an out-of-range index raises ValueError."""
    dataset = _FakePackDataset(5)
    path = _write_order_npy(tmp_path, "bad.npy", _i64(0, 1, 99))

    with pytest.raises(ValueError, match="out of range"):
        PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)


def test_load_from_npy_file_mmap(tmp_path):
    """PresetSampler loads a .npy path and keeps mmap-backed order."""
    dataset = _FakePackDataset(5)
    order_list = [2, 0, 1, 3, 4]
    path = _write_order_npy(tmp_path, "sampler_order.npy", np.array(order_list, dtype=np.int64))

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    assert list(sampler) == order_list
    assert isinstance(sampler.global_order, np.ndarray)
    assert isinstance(sampler.global_order, np.memmap)


def test_state_dict_resume(tmp_path):
    """Restoring a state dict causes __iter__ to resume from the saved offset."""
    dataset = _FakePackDataset(6)
    order = _i64(0, 1, 2, 3, 4, 5)
    path = _write_order_npy(tmp_path, "order.npy", order)

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)

    state = sampler.get_state_dict(step=3)
    assert state["step"] == 3

    sampler2 = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    sampler2.load_state_dict(state)

    assert list(sampler2) == [3, 4, 5]


def test_state_dict_world_size_mismatch(tmp_path):
    """Mismatched world_size in state dict logs a warning but does not raise."""
    dataset = _FakePackDataset(4)
    path = _write_order_npy(tmp_path, "order.npy", _i64(0, 1, 2, 3))

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    state = sampler.get_state_dict(step=0)
    state["world_size"] = 99

    sampler.load_state_dict(state)


def test_repeated_packs(tmp_path):
    """Order may reference the same pack more than once (over-sampling)."""
    dataset = _FakePackDataset(3)
    order = _i64(0, 0, 1, 1, 2, 2)
    path = _write_order_npy(tmp_path, "order.npy", order)

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    yielded = list(sampler)

    assert len(yielded) == order.size
    assert len(set(yielded)) < len(yielded)


def test_len(tmp_path):
    """__len__ returns the per-rank number of samples."""
    dataset = _FakePackDataset(6)
    path = _write_order_npy(tmp_path, "order.npy", _i64(0, 1, 2, 3, 4, 5))

    sampler = PresetSampler(dataset, sampler_config_path=path, global_batch_size=1)
    assert len(sampler) == 6


def test_round_down_too_short_raises(tmp_path):
    """If order length < global_batch_size*world_size, round-down yields 0."""
    dataset = _FakePackDataset(10)
    path = _write_order_npy(tmp_path, "order.npy", _i64(0, 1, 2))

    with pytest.raises(ValueError, match="round down"):
        PresetSampler(dataset, sampler_config_path=path, global_batch_size=4)
