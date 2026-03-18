"""Unit tests for CustomPackDataset."""

import json
import os

import numpy as np
import pytest

from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.custom_pack import (
    CustomPackDataset,
    _load_pack_config_jsonl,
    _load_pack_config_parquet,
)
from xtuner.v1.datasets.jsonl import JsonlDataset, save_mixed_dict_to_parquet
from xtuner.v1.datasets.utils import CachableTokenizeFunction


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal fake JsonlDataset with .path and .num_tokens attributes.

    Args:
        samples (list[list[int]]): Token id lists, one per sample.
        path (str): Dataset path used for pack config matching.
        long_text_meta (dict[int, tuple[int, int, int]] | None): Optional mapping from
            sample_idx to (char_start, char_end, token_start_offset) for LongTextDataItem samples.
    """

    def __init__(
        self,
        samples: list[list[int]],
        path: str,
        long_text_meta: dict[int, tuple[int, int, int]] | None = None,
    ):
        self.path = path
        self._samples = samples
        self.num_tokens = np.array([len(s) for s in samples], dtype=np.int64)
        self._long_text_meta = long_text_meta or {}

    def __getitem__(self, s_idx: int) -> dict:
        ids = self._samples[s_idx]
        if s_idx in self._long_text_meta:
            char_start, char_end, tok_off = self._long_text_meta[s_idx]
            return {
                "input_ids": ids,
                "labels": list(ids),
                "num_tokens": len(ids),
                "char_start": char_start,
                "char_end": char_end,
                "token_start_offset": tok_off,
            }
        return {"input_ids": ids, "labels": list(ids), "num_tokens": len(ids)}

    def __len__(self) -> int:
        return len(self._samples)


def _write_jsonl_pack(path: str, packs: list[list[list]]) -> None:
    """Write packs as JSONL. Each slice: [dataset_path, s_idx, c_start, c_end, tok_off, tok_end]."""
    with open(path, "w") as f:
        for pack in packs:
            f.write(json.dumps({"samples": pack}) + "\n")


def _write_parquet_pack(path: str, packs: list[list[list[int]]], paths: list[str]) -> None:
    """Write packs as Parquet. Each slice: [path_id, s_idx, c_start, c_end, tok_off, tok_end]."""
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
        """JSONL loader returns correct pack count and slice fields including token_end_offset."""
        packs = [
            [["ds0.jsonl", 0, -1, -1, 0, 100], ["ds0.jsonl", 1, -1, -1, 0, 200]],
            [["ds1.jsonl", 5, 10, 200, 3, 50]],
        ]
        config_path = str(tmp_path / "pack_config.jsonl")
        _write_jsonl_pack(config_path, packs)

        result = _load_pack_config_jsonl(config_path)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        ds_path, s_idx, c_start, c_end, tok_off, tok_end = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0
        assert tok_end == 100

        ds_path, s_idx, c_start, c_end, tok_off, tok_end = result[1][0]
        assert ds_path == "ds1.jsonl"
        assert s_idx == 5
        assert c_start == 10
        assert c_end == 200
        assert tok_off == 3
        assert tok_end == 50

    def test_invalid_slice_length(self, tmp_path):
        """Slice with wrong element count raises ValueError."""
        packs = [[["ds0.jsonl", 0, -1, -1, 0]]]  # 5 elements instead of 6
        config_path = str(tmp_path / "pack_config.jsonl")
        _write_jsonl_pack(config_path, packs)

        with pytest.raises(ValueError, match="slice must be"):
            _load_pack_config_jsonl(config_path)


class TestLoadPackConfigParquet:
    def test_basic_load(self, tmp_path):
        """Parquet loader returns correct pack count, slice count, and field values including token_end_offset."""
        paths = ["ds0.jsonl", "ds1.jsonl"]
        packs = [
            [[0, 0, -1, -1, 0, 100], [0, 1, -1, -1, 0, 200]],
            [[1, 5, 10, 200, 3, 50]],
        ]
        config_path = str(tmp_path / "pack_config.parquet")
        _write_parquet_pack(config_path, packs, paths)

        result = _load_pack_config_parquet(config_path)

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        ds_path, s_idx, c_start, c_end, tok_off, tok_end = result[0][0]
        assert ds_path == "ds0.jsonl"
        assert s_idx == 0
        assert c_start == -1
        assert c_end == -1
        assert tok_off == 0
        assert tok_end == 100

        ds_path, s_idx, c_start, c_end, tok_off, tok_end = result[1][0]
        assert ds_path == "ds1.jsonl"
        assert s_idx == 5
        assert c_start == 10
        assert c_end == 200
        assert tok_off == 3
        assert tok_end == 50

# ---------------------------------------------------------------------------
# Feature 2: validation tests (test CustomPackDataset.__init__ / _validate_pack)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_pack_exact_length(self, tmp_path):
        """Pack with total tokens == pack_max_length is accepted."""
        ds = _FakeDataset([[1] * 128, [2] * 128], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 128], ["/ds.jsonl", 1, -1, -1, 0, 128]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256)
        assert len(dataset) == 1

    def test_invalid_dataset_path(self, tmp_path):
        """Unknown dataset_path raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/unknown.jsonl", 0, -1, -1, 0, 256]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="dataset_path"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    def test_invalid_sample_idx(self, tmp_path):
        """sample_idx out of range raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")  # only 1 sample
        packs = [["/ds.jsonl", 99, -1, -1, 0, 256]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="sample_idx"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    @pytest.mark.parametrize("char_start,char_end", [(-5, 100), (100, 50), (-1, 100)])
    def test_invalid_char_range(self, tmp_path, char_start, char_end):
        """Invalid char range raises ValueError (unless both are -1)."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, char_start, char_end, 0, 256]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="char range"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    def test_short_pack_error(self, tmp_path):
        """Pack shorter than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 100]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_path, pack_max_length=256, short_pack_strategy="error")

    def test_short_pack_padding(self, tmp_path):
        """Pack shorter than pack_max_length is accepted with strategy='padding'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 100]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256, short_pack_strategy="padding")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256

    def test_long_pack_error(self, tmp_path):
        """Pack longer than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 300]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_path, pack_max_length=256, long_pack_strategy="error")

    def test_long_pack_truncate(self, tmp_path):
        """Pack longer than pack_max_length is truncated to pack_max_length with strategy='truncate'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 300]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256, long_pack_strategy="truncate")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256

    def test_invalid_token_end_offset(self, tmp_path):
        """token_end_offset <= token_start_offset raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 50, 30]]  # tok_end (30) < tok_start (50)
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="token_end_offset"):
            CustomPackDataset([ds], config_path, pack_max_length=256)

    def test_invalid_token_end_offset_equal(self, tmp_path):
        """token_end_offset == token_start_offset raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 50, 50]]  # tok_end == tok_start
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        with pytest.raises(ValueError, match="token_end_offset"):
            CustomPackDataset([ds], config_path, pack_max_length=256)


# ---------------------------------------------------------------------------
# Feature 3: __getitem__ tests
# ---------------------------------------------------------------------------


class TestGetitem:
    def test_dataitem_jsonl(self, tmp_path):
        """DataItem case (char_start==-1) via JSONL config: returns correct input_ids/labels/num_tokens."""
        ids0 = list(range(128))
        ids1 = list(range(128, 256))
        ds = _FakeDataset([ids0, ids1], path="/ds.jsonl")
        packs = [["/ds.jsonl", 0, -1, -1, 0, 128], ["/ds.jsonl", 1, -1, -1, 0, 128]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256)
        items = dataset[0]

        assert len(items) == 2
        assert items[0]["input_ids"] == ids0
        assert items[0]["labels"] == ids0
        assert items[0]["num_tokens"] == 128
        assert items[1]["input_ids"] == ids1
        assert items[1]["num_tokens"] == 128

    def test_dataitem_parquet(self, tmp_path):
        """DataItem case (char_start==-1) via Parquet config: returns correct items."""
        ids0 = [1] * 100
        ids1 = [2] * 156
        ds = _FakeDataset([ids0, ids1], path="/ds.jsonl")
        paths = ["/ds.jsonl"]
        packs = [[[0, 0, -1, -1, 0, 100], [0, 1, -1, -1, 0, 156]]]
        config_path = str(tmp_path / "packs.parquet")
        _write_parquet_pack(config_path, packs, paths)

        dataset = CustomPackDataset([ds], config_path, pack_max_length=256)
        items = dataset[0]

        assert len(items) == 2
        assert items[0]["input_ids"] == ids0
        assert items[1]["input_ids"] == ids1
        assert sum(it["num_tokens"] for it in items) == 256

    def test_longtextdataitem(self, tmp_path):
        """LongTextDataItem case: consistency check passes and item is returned as-is."""
        ids = list(range(128))
        # token_start_offset=5, len(ids)=128, so token_end_offset=5+128=133
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (100, 600, 5)})
        packs = [["/ds.jsonl", 0, 100, 600, 5, 133]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=128)
        items = dataset[0]

        assert len(items) == 1
        assert items[0]["input_ids"] == ids
        assert items[0]["char_start"] == 100
        assert items[0]["char_end"] == 600
        assert items[0]["token_start_offset"] == 5

    def test_mixed_dataitem_and_longtextdataitem(self, tmp_path):
        """Mixed pack: DataItem and LongTextDataItem slices coexist correctly."""
        ids0 = [1] * 100  # plain DataItem
        ids1 = [2] * 128  # LongTextDataItem, token_start_offset=0, token_end_offset=128
        ds = _FakeDataset([ids0, ids1], path="/ds.jsonl", long_text_meta={1: (0, 512, 0)})
        packs = [["/ds.jsonl", 0, -1, -1, 0, 100], ["/ds.jsonl", 1, 0, 512, 0, 128]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=228)
        items = dataset[0]

        assert len(items) == 2
        assert "char_start" not in items[0]
        assert items[1]["char_start"] == 0
        assert sum(it["num_tokens"] for it in items) == 228

    def test_consistency_check_error_dataitem_got_longtext(self, tmp_path):
        """Consistency check raises ValueError when DataItem expected but LongTextDataItem returned."""
        ids = [1] * 128
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (0, 512, 0)})
        # pack config says char_start==-1 (DataItem), but dataset returns LongTextDataItem
        packs = [["/ds.jsonl", 0, -1, -1, 0, 128]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=128)
        with pytest.raises(ValueError, match="LongTextDataItem"):
            dataset[0]

    def test_consistency_check_error_longtext_fields_mismatch(self, tmp_path):
        """Consistency check raises ValueError when LongTextDataItem fields don't match pack config."""
        ids = [1] * 128
        # dataset returns char_start=100, but pack config expects char_start=999 (valid range, but mismatches)
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (100, 600, 5)})
        packs = [["/ds.jsonl", 0, 999, 1600, 5, 133]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=128)
        with pytest.raises(ValueError, match="mismatch"):
            dataset[0]

    def test_plain_tokenizefn_token_start_offset_applied(self, tmp_path):
        """Plain DataItem: input_ids[token_start_offset:token_end_offset] slicing is applied."""
        ids = list(range(200))
        ds = _FakeDataset([ids], path="/ds.jsonl")
        # Take tokens [50:150] — 100 tokens out of the 200-token sample
        packs = [["/ds.jsonl", 0, -1, -1, 50, 150]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=100)
        items = dataset[0]

        assert len(items) == 1
        assert items[0]["input_ids"] == ids[50:150]
        assert items[0]["labels"] == ids[50:150]
        assert items[0]["num_tokens"] == 100

    def test_longtextdataitem_no_extra_truncation(self, tmp_path):
        """LongTextDataItem is pre-truncated at tokenize time; __getitem__ does not apply extra slicing."""
        ids = list(range(128))  # already pre-truncated 128 tokens
        # token_start_offset=10, token_end_offset=138: tok_end - tok_off = 128 = len(ids)
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (0, 1000, 10)})
        packs = [["/ds.jsonl", 0, 0, 1000, 10, 138]]
        config_path = str(tmp_path / "packs.jsonl")
        _write_jsonl_pack(config_path, [packs])

        dataset = CustomPackDataset([ds], config_path, pack_max_length=128)
        items = dataset[0]

        assert len(items) == 1
        assert items[0]["input_ids"] == ids  # returned as-is, not re-sliced
        assert items[0]["num_tokens"] == 128


# ---------------------------------------------------------------------------
# Helpers for Feature 4 tests
# ---------------------------------------------------------------------------


class _CountingTokenizeFn(CachableTokenizeFunction):
    """Tokenize function that returns preset num_tokens per line (by call order)."""

    def __init__(self, num_tokens_per_line: list[int]):
        super().__init__(tokenizer=None)
        self._num_tokens = num_tokens_per_line
        self._idx = 0

    def __call__(self, item, **kwargs):
        n = self._num_tokens[self._idx % len(self._num_tokens)]
        self._idx += 1
        return {"num_tokens": n}

    def hash(self) -> str:
        return "counting_test_hash"


def _write_jsonl(path: str, lines: list[dict]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# Feature 4: disable_filter tests
# ---------------------------------------------------------------------------


class TestDisableFilter:
    def test_zero_token_filter_applied_by_default(self, tmp_path, monkeypatch):
        """Without disable_filter, samples with num_tokens==0 are excluded."""
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        jsonl_path = str(tmp_path / "data.jsonl")
        # 3 lines; tokenize fn returns [0, 5, 5] tokens
        _write_jsonl(jsonl_path, [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        tokenize_fn = _CountingTokenizeFn([0, 5, 5])

        ds = JsonlDataset(jsonl_path, tokenize_fn=tokenize_fn)
        # Sample with 0 tokens should be filtered out
        assert len(ds) == 2

    def test_disable_filter_skips_zero_token_filter(self, tmp_path, monkeypatch):
        """disable_filter=True keeps samples with num_tokens==0."""
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        jsonl_path = str(tmp_path / "data.jsonl")
        _write_jsonl(jsonl_path, [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        tokenize_fn = _CountingTokenizeFn([0, 5, 5])

        ds = JsonlDataset(jsonl_path, tokenize_fn=tokenize_fn, disable_filter=True)
        assert len(ds) == 3

    def test_disable_filter_skips_max_length_filter(self, tmp_path, monkeypatch):
        """disable_filter=True keeps samples that exceed max_length."""
        monkeypatch.setenv("XTUNER_TOKENIZE_WORKERS", "1")
        jsonl_path = str(tmp_path / "data.jsonl")
        _write_jsonl(jsonl_path, [{"text": "a"}, {"text": "b"}])
        # line 0: 100 tokens (within limit), line 1: 200 tokens (over limit)
        tokenize_fn = _CountingTokenizeFn([100, 200])

        ds_filtered = JsonlDataset(jsonl_path, tokenize_fn=tokenize_fn, max_length=150)
        ds_unfiltered = JsonlDataset(
            jsonl_path,
            tokenize_fn=_CountingTokenizeFn([100, 200]),
            max_length=150,
            disable_filter=True,
        )
        assert len(ds_filtered) == 1
        assert len(ds_unfiltered) == 2


# ---------------------------------------------------------------------------
# Feature 4: DataloaderConfig forces settings for pack_level='custom'
# ---------------------------------------------------------------------------


class TestDataloaderConfigCustomMode:
    def test_forces_dataset_settings(self):
        """_force_custom_pack_settings forces sample_ratio=1.0, enable_sequential_sampler=True,
        disable_filter=True regardless of original config values."""
        dc = DatasetConfig(
            anno_path="/fake/ds.jsonl",
            sample_ratio=0.5,
            enable_sequential_sampler=False,
            disable_filter=False,
        )
        config_list = [{"dataset": dc, "tokenize_fn": None}]  # type: ignore[list-item]
        forced = DataloaderConfig._force_custom_pack_settings(config_list)

        assert len(forced) == 1
        result_dc = forced[0]["dataset"]
        assert result_dc.sample_ratio == 1.0
        assert result_dc.enable_sequential_sampler is True
        assert result_dc.disable_filter is True
