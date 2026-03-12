"""Unit tests for CustomPackDataset."""

import json
import os

import numpy as np
import pytest

from xtuner.v1.datasets.custom_pack import CustomPackDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal fake JsonlDataset compatible with CustomPackDataset."""

    def __init__(self, samples: list[list[int]], name: str = "fake"):
        self.name = name
        self._samples = samples
        # Identity mapping: logical index == raw index (no filtering applied).
        self.sampled = list(range(len(samples)))
        self.num_tokens = np.array([len(s) for s in samples], dtype=np.int64)

    def __getitem__(self, s_idx: int) -> dict:
        ids = self._samples[s_idx]
        return {"input_ids": ids, "labels": list(ids), "num_tokens": len(ids)}

    def __len__(self) -> int:
        return len(self._samples)


def _write_jsonl_pack(path: str, packs: list[list[list[int]]]) -> None:
    with open(path, "w") as f:
        for pack in packs:
            f.write(json.dumps({"samples": pack}) + "\n")


def _write_npy_pack(base_dir: str, packs: list[list[list[int]]]) -> None:
    flat: list[list[int]] = []
    boundaries: list[int] = [0]
    for pack in packs:
        flat.extend(pack)
        boundaries.append(len(flat))
    np.save(
        os.path.join(base_dir, "pack_boundaries.npy"),
        np.array(boundaries, dtype=np.int64),
    )
    np.save(
        os.path.join(base_dir, "pack_samples.npy"),
        np.array(flat, dtype=np.int64).reshape(-1, 4),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_jsonl_pack(tmp_path):
    """JSONL pack config with exact-length packs is loaded correctly."""
    ds0 = _FakeDataset([[i] * 256 for i in range(3)], name="ds0")
    ds1 = _FakeDataset([[i + 10] * 256 for i in range(3)], name="ds1")

    packs = [
        [[0, 0, 0, 256]],   # ds0, sample 0, full
        [[0, 1, 0, 256]],   # ds0, sample 1, full
        [[1, 2, 0, 256]],   # ds1, sample 2, full
    ]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0, ds1], config_path, pack_max_length=256)
    assert len(dataset) == 3

    items0 = dataset[0]
    assert len(items0) == 1
    assert items0[0]["num_tokens"] == 256
    assert items0[0]["input_ids"] == ds0[0]["input_ids"]

    items1 = dataset[1]
    assert items1[0]["input_ids"] == ds0[1]["input_ids"]

    items2 = dataset[2]
    assert items2[0]["input_ids"] == ds1[2]["input_ids"]


def test_valid_npy_pack(tmp_path):
    """NPY CSR pack config with exact-length packs is loaded correctly."""
    ds0 = _FakeDataset([[i] * 256 for i in range(3)], name="ds0")

    packs = [
        [[0, 0, 0, 256]],
        [[0, 1, 0, 256]],
    ]
    _write_npy_pack(str(tmp_path), packs)

    dataset = CustomPackDataset([ds0], str(tmp_path), pack_max_length=256)
    assert len(dataset) == 2
    assert dataset[0][0]["input_ids"] == ds0[0]["input_ids"]
    assert dataset[1][0]["input_ids"] == ds0[1]["input_ids"]


def test_token_end_zero_sentinel(tmp_path):
    """token_end=0 is resolved to the sample's full length."""
    ds0 = _FakeDataset([[i] * 128 for i in range(2)])

    # token_end=0 → take all 128 tokens
    packs = [[[0, 0, 0, 0]]]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=128)
    items = dataset[0]
    assert items[0]["num_tokens"] == 128
    assert items[0]["input_ids"] == ds0[0]["input_ids"]


def test_short_pack_error(tmp_path):
    """Pack with fewer tokens than pack_max_length raises ValueError (strategy='error')."""
    ds0 = _FakeDataset([[i] * 100 for i in range(2)])

    packs = [[[0, 0, 0, 100]]]   # 100 tokens but pack_max_length=256
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    with pytest.raises(ValueError, match="total tokens"):
        CustomPackDataset([ds0], config_path, pack_max_length=256, short_pack_strategy="error")


def test_short_pack_skip(tmp_path):
    """Pack with fewer tokens is dropped (strategy='skip'); valid packs remain."""
    ds0 = _FakeDataset([[i] * 256 for i in range(3)])

    packs = [
        [[0, 0, 0, 256]],   # valid
        [[0, 1, 0, 100]],   # short: only 100 tokens
        [[0, 2, 0, 256]],   # valid
    ]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=256, short_pack_strategy="skip")
    assert len(dataset) == 2


def test_short_pack_padding(tmp_path):
    """Pack with fewer tokens is padded to pack_max_length (strategy='padding')."""
    ds0 = _FakeDataset([[i] * 100 for i in range(2)])

    packs = [[[0, 0, 0, 100]]]   # 100 tokens, pack_max_length=256
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=256, short_pack_strategy="padding")
    assert len(dataset) == 1

    items = dataset[0]
    total_tokens = sum(item["num_tokens"] for item in items)
    assert total_tokens == 256

    pad_item = items[-1]
    assert pad_item["num_tokens"] == 156   # 256 - 100
    assert pad_item["input_ids"] == [0] * 156
    assert pad_item["labels"] == [-100] * 156


def test_long_pack_error(tmp_path):
    """Pack with more tokens than pack_max_length raises ValueError (strategy='error')."""
    ds0 = _FakeDataset([[i] * 300 for i in range(2)])

    packs = [[[0, 0, 0, 300]]]   # 300 tokens but pack_max_length=256
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    with pytest.raises(ValueError, match="total tokens"):
        CustomPackDataset([ds0], config_path, pack_max_length=256, long_pack_strategy="error")


def test_long_pack_skip(tmp_path):
    """Pack with more tokens than pack_max_length is dropped (strategy='skip')."""
    ds0 = _FakeDataset([[i] * 300 for i in range(3)])

    packs = [
        [[0, 0, 0, 256]],   # valid (takes first 256 tokens)
        [[0, 1, 0, 300]],   # long
        [[0, 2, 0, 256]],   # valid
    ]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=256, long_pack_strategy="skip")
    assert len(dataset) == 2


def test_long_pack_truncate(tmp_path):
    """Last slice of a too-long pack is truncated so total == pack_max_length."""
    ds0 = _FakeDataset([[i] * 300 for i in range(2)])

    packs = [[[0, 0, 0, 300]]]   # 300 tokens, pack_max_length=256
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=256, long_pack_strategy="truncate")
    assert len(dataset) == 1

    items = dataset[0]
    total_tokens = sum(item["num_tokens"] for item in items)
    assert total_tokens == 256

    assert items[0]["num_tokens"] == 256
    assert items[0]["input_ids"] == ds0[0]["input_ids"][:256]


def test_hard_error_invalid_dataset_id(tmp_path):
    """dataset_id out of range raises ValueError regardless of pack strategy."""
    ds0 = _FakeDataset([[i] * 256 for i in range(2)])

    packs = [[[5, 0, 0, 256]]]   # dataset_id=5 but only one dataset
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    with pytest.raises(ValueError, match="dataset_id"):
        CustomPackDataset(
            [ds0], config_path, pack_max_length=256,
            short_pack_strategy="skip", long_pack_strategy="skip",
        )


def test_hard_error_invalid_sample_idx(tmp_path):
    """sample_idx out of range raises ValueError."""
    ds0 = _FakeDataset([[i] * 256 for i in range(2)])   # 2 samples

    packs = [[[0, 99, 0, 256]]]   # sample_idx=99 but only 2 samples
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    with pytest.raises(ValueError, match="sample_idx"):
        CustomPackDataset([ds0], config_path, pack_max_length=256)


@pytest.mark.parametrize(
    "bad_range",
    [
        [0, 0, -1, 100],   # negative token_start
        [0, 0, 0, 300],    # token_end > num_tokens (sample has 256 tokens)
        [0, 0, 100, 50],   # token_start >= token_end
    ],
)
def test_hard_error_invalid_token_range(tmp_path, bad_range):
    """Invalid token ranges always raise ValueError."""
    ds0 = _FakeDataset([[i] * 256 for i in range(2)])

    packs = [[bad_range]]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    with pytest.raises(ValueError):
        CustomPackDataset([ds0], config_path, pack_max_length=256)


def test_state_dict(tmp_path):
    """get_state_dict returns {} and load_state_dict is a no-op."""
    ds0 = _FakeDataset([[i] * 256 for i in range(2)])

    packs = [[[0, 0, 0, 256]], [[0, 1, 0, 256]]]
    config_path = str(tmp_path / "pack_config.jsonl")
    _write_jsonl_pack(config_path, packs)

    dataset = CustomPackDataset([ds0], config_path, pack_max_length=256)
    assert dataset.get_state_dict() == {}
    dataset.load_state_dict({})   # must not raise
    assert len(dataset) == 2      # unchanged
