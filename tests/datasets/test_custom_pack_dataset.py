"""Unit tests for CustomPackDataset."""

import json

import numpy as np
import pytest

from xtuner.v1.datasets.custom_pack import (
    _load_pack_config_jsonl,
    _load_pack_config_parquet,
)
from xtuner.v1.datasets.jsonl import save_mixed_dict_to_parquet


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


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

        # pack 0, slice 0
        ds_path, s_idx, c_start, c_end, tok_off = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0

        # pack 1, slice 0 – LongText style
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

        # pack 0, slice 0
        ds_path, s_idx, c_start, c_end, tok_off = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0

        # pack 1, slice 0 – LongText style, path_id 1 -> "ds1.jsonl"
        ds_path, s_idx, c_start, c_end, tok_off = result[1][0]
        assert ds_path == "ds1.jsonl"
        assert s_idx == 5
        assert c_start == 10
        assert c_end == 200
        assert tok_off == 3

