"""Unit tests for CustomPackDataset."""

import json

import numpy as np
import pytest

from xtuner.v1.datasets.custom_pack import (
    CustomPackDataset,
    _load_pack_config_jsonl,
    _load_pack_config_parquet,
)
from xtuner.v1.datasets.jsonl import save_mixed_dict_to_parquet


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal fake JsonlDataset with .path and .num_tokens attributes."""

    def __init__(self, samples: list[list[int]], path: str):
        self.path = path
        self._samples = samples
        self.num_tokens = np.array([len(s) for s in samples], dtype=np.int64)

    def __getitem__(self, s_idx: int) -> dict:
        ids = self._samples[s_idx]
        return {"input_ids": ids, "labels": list(ids), "num_tokens": len(ids)}

    def __len__(self) -> int:
        return len(self._samples)


def _write_jsonl_pack(path: str, packs: list[list[list]]) -> None:
    """Write packs as JSONL. Each slice: [dataset_path, s_idx, c_start, c_end, tok_off]."""
    with open(path, "w") as f:
        for pack in packs:
            f.write(json.dumps({"samples": pack}) + "\n")


def _write_parquet_pack(path: str, packs: list[list[list[int]]], paths: list[str]) -> None:
    """Write packs as Parquet. Each slice: [path_id, s_idx, c_start, c_end, tok_off]."""
    flat: list[list[int]] = []
    boundaries: list[int] = [0]
    for pack in packs:
        flat.extend(pack)
        boundaries.append(len(flat))
    save_mixed_dict_to_parquet(
        {
            "boundaries": np.array(boundaries, dtype=np.int64),
            "samples": flat,
            "paths": paths,  # type: ignore[dict-item]
        },
        path,
    )


# ---------------------------------------------------------------------------
# Feature 1: loader tests (test loader functions directly)
# ---------------------------------------------------------------------------


class TestLoadPackConfigJsonl:
    def test_basic_load(self, tmp_path):
        """JSONL loader returns correct pack count and slice fields."""
        packs = [
            [["ds0.jsonl", 0, -1, -1, 0], ["ds0.jsonl", 1, -1, -1, 0]],
            [["ds1.jsonl", 5, 10, 200, 3]],
        ]
        config_path = str(tmp_path / "pack_config.jsonl")
        _write_jsonl_pack(config_path, packs)

        result = _load_pack_config_jsonl(config_path)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        ds_path, s_idx, c_start, c_end, tok_off = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0

        ds_path, s_idx, c_start, c_end, tok_off = result[1][0]
        assert ds_path == "ds1.jsonl"
        assert s_idx == 5
        assert c_start == 10
        assert c_end == 200
        assert tok_off == 3

    def test_invalid_slice_length(self, tmp_path):
        """Slice with wrong element count raises ValueError."""
        packs = [[["ds0.jsonl", 0, -1, -1]]]  # 4 elements instead of 5
        config_path = str(tmp_path / "pack_config.jsonl")
        _write_jsonl_pack(config_path, packs)

        with pytest.raises(ValueError, match="slice must be"):
            _load_pack_config_jsonl(config_path)


class TestLoadPackConfigParquet:
    def test_basic_load(self, tmp_path):
        """Parquet loader returns correct pack count, slice count, and field values."""
        paths = ["ds0.jsonl", "ds1.jsonl"]
        packs = [
            [[0, 0, -1, -1, 0], [0, 1, -1, -1, 0]],
            [[1, 5, 10, 200, 3]],
        ]
        config_path = str(tmp_path / "pack_config.parquet")
        _write_parquet_pack(config_path, packs, paths)

        result = _load_pack_config_parquet(config_path)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        ds_path, s_idx, c_start, c_end, tok_off = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0

        ds_path, s_idx, c_start, c_end, tok_off = result[1][0]
        assert ds_path == "ds1.jsonl"
        assert s_idx == 5
        assert c_start == 10
        assert c_end == 200
        assert tok_off == 3

# ---------------------------------------------------------------------------
# Feature 2: validation tests (test CustomPackDataset.__init__ / _validate_pack)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_pack_exact_length(self, tmp_path):
        """Pack with total tokens == pack_max_length is accepted."""
        ds = _FakeDataset([[1] * 128, [2] * 128], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0], ["/ds.jsonl", 1, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256)
        assert len(dataset) == 1

    def test_invalid_dataset_path(self, tmp_path):
        """Unknown dataset_path raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/unknown.jsonl", 0, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="dataset_path"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    def test_invalid_sample_idx(self, tmp_path):
        """sample_idx out of range raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")  # only 1 sample
        packs = [["/ds.jsonl", 99, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="sample_idx"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    @pytest.mark.parametrize("char_start,char_end", [(-5, 100), (100, 50), (-1, 100)])
    def test_invalid_char_range(self, tmp_path, char_start, char_end):
        """Invalid char range raises ValueError (unless both are -1)."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, char_start, char_end, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="char range"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    def test_short_pack_error(self, tmp_path):
        """Pack shorter than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_path, pack_max_length=256, short_pack_strategy="error")

    def test_short_pack_padding(self, tmp_path):
        """Pack shorter than pack_max_length is accepted with strategy='padding'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256, short_pack_strategy="padding")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256

    def test_long_pack_error(self, tmp_path):
        """Pack longer than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_path, pack_max_length=256, long_pack_strategy="error")

    def test_long_pack_truncate(self, tmp_path):
        """Pack longer than pack_max_length is truncated to pack_max_length with strategy='truncate'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256, long_pack_strategy="truncate")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256
