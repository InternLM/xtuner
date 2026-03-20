"""Unit tests for CustomPackDataset."""

import json
import multiprocessing
import os
import time

import numpy as np
import psutil
import pytest

from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.custom_pack import CustomPackDataset, load_config
from xtuner.v1.datasets.jsonl import JsonlDataset
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


def _write_npy_pack_dir(directory: str, packs: list[list[list[int]]], paths: list[str]) -> None:
    """Write packs to NPY directory format.

    Each slice: [path_id, s_idx, c_start, c_end, tok_off, tok_end].
    """
    os.makedirs(directory, exist_ok=True)
    flat = [row for pack in packs for row in pack]
    boundaries: list[int] = [0]
    for pack in packs:
        boundaries.append(boundaries[-1] + len(pack))
    samples_arr = np.array(flat, dtype=np.int64).reshape(-1, 6) if flat else np.empty((0, 6), dtype=np.int64)
    np.save(os.path.join(directory, "boundaries.npy"), np.array(boundaries, dtype=np.int64))
    np.save(os.path.join(directory, "samples.npy"), samples_arr)
    np.save(os.path.join(directory, "paths.npy"), np.array(paths, dtype=object))


# ---------------------------------------------------------------------------
# Feature 7: load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    @pytest.mark.parametrize("mmap", [False, True])
    def test_basic_load(self, tmp_path, mmap):
        """load_config returns correct boundaries, samples, and paths from NPY directory."""
        paths = ["ds0.jsonl", "ds1.jsonl"]
        packs = [
            [[0, 0, -1, -1, 0, 100], [0, 1, -1, -1, 0, 200]],
            [[1, 5, 10, 200, 3, 50]],
        ]
        config_dir = str(tmp_path / "pack_dir")
        _write_npy_pack_dir(config_dir, packs, paths)

        result = load_config(config_dir, mmap=mmap)

        boundaries = result["boundaries"]
        samples = result["samples"]
        result_paths = result["paths"]

        assert list(boundaries) == [0, 2, 3]
        assert samples.shape == (3, 6)
        assert list(samples[0]) == [0, 0, -1, -1, 0, 100]
        assert list(samples[1]) == [0, 1, -1, -1, 0, 200]
        assert list(samples[2]) == [1, 5, 10, 200, 3, 50]
        assert list(result_paths) == ["ds0.jsonl", "ds1.jsonl"]


# ---------------------------------------------------------------------------
# Feature 2: validation tests (test CustomPackDataset.__init__ / _validate_arrays)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_pack_exact_length(self, tmp_path):
        """Pack with total tokens == pack_max_length is accepted."""
        ds = _FakeDataset([[1] * 128, [2] * 128], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 128], [0, 1, -1, -1, 0, 128]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=256)
        assert len(dataset) == 1

    def test_invalid_dataset_path(self, tmp_path):
        """Unknown dataset_path raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 256]]]  # path_id 0 -> "/unknown.jsonl"
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/unknown.jsonl"])

        with pytest.raises(ValueError, match="dataset_path"):
            CustomPackDataset([ds], config_dir, pack_max_length=256)

    def test_invalid_sample_idx(self, tmp_path):
        """sample_idx out of range raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")  # only 1 sample
        packs = [[[0, 99, -1, -1, 0, 256]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="sample_idx"):
            CustomPackDataset([ds], config_dir, pack_max_length=256)

    @pytest.mark.parametrize("char_start,char_end", [(-5, 100), (100, 50), (-1, 100)])
    def test_invalid_char_range(self, tmp_path, char_start, char_end):
        """Invalid char range raises ValueError (unless both are -1)."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [[[0, 0, char_start, char_end, 0, 256]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="char range"):
            CustomPackDataset([ds], config_dir, pack_max_length=256)

    def test_short_pack_error(self, tmp_path):
        """Pack shorter than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 100]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_dir, pack_max_length=256, short_pack_strategy="error")

    def test_short_pack_padding(self, tmp_path):
        """Pack shorter than pack_max_length is accepted with strategy='padding'."""
        ds = _FakeDataset([[1] * 100], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 100]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=256, short_pack_strategy="padding")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256

    def test_long_pack_error(self, tmp_path):
        """Pack longer than pack_max_length raises ValueError with strategy='error'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 300]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="total tokens"):
            CustomPackDataset([ds], config_dir, pack_max_length=256, long_pack_strategy="error")

    def test_long_pack_truncate(self, tmp_path):
        """Pack longer than pack_max_length is truncated to pack_max_length with strategy='truncate'."""
        ds = _FakeDataset([[1] * 300], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 300]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=256, long_pack_strategy="truncate")
        assert len(dataset) == 1
        items = dataset[0]
        assert sum(it["num_tokens"] for it in items) == 256

    def test_invalid_token_end_offset(self, tmp_path):
        """token_end_offset <= token_start_offset raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 50, 30]]]  # tok_end (30) < tok_start (50)
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="token_end_offset"):
            CustomPackDataset([ds], config_dir, pack_max_length=256)

    def test_invalid_token_end_offset_equal(self, tmp_path):
        """token_end_offset == token_start_offset raises ValueError."""
        ds = _FakeDataset([[1] * 256], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 50, 50]]]  # tok_end == tok_start
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        with pytest.raises(ValueError, match="token_end_offset"):
            CustomPackDataset([ds], config_dir, pack_max_length=256)


# ---------------------------------------------------------------------------
# Feature 3 & 6: __getitem__ tests
# ---------------------------------------------------------------------------


class TestGetitem:
    def test_dataitem_basic(self, tmp_path):
        """DataItem case (char_start==-1): returns correct input_ids/labels/num_tokens."""
        ids0 = list(range(128))
        ids1 = list(range(128, 256))
        ds = _FakeDataset([ids0, ids1], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 128], [0, 1, -1, -1, 0, 128]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=256)
        items = dataset[0]

        assert len(items) == 2
        assert items[0]["input_ids"] == ids0
        assert items[0]["labels"] == ids0
        assert items[0]["num_tokens"] == 128
        assert items[1]["input_ids"] == ids1
        assert items[1]["num_tokens"] == 128

    def test_dataitem_two_packs(self, tmp_path):
        """Two packs each with one DataItem slice."""
        ids0 = [1] * 128
        ids1 = [2] * 128
        ds = _FakeDataset([ids0, ids1], path="/ds.jsonl")
        packs = [[[0, 0, -1, -1, 0, 128]], [[0, 1, -1, -1, 0, 128]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=128)
        assert len(dataset) == 2
        assert dataset[0][0]["input_ids"] == ids0
        assert dataset[1][0]["input_ids"] == ids1

    def test_longtextdataitem(self, tmp_path):
        """LongTextDataItem case: consistency check passes and item is returned as-is."""
        ids = list(range(128))
        # token_start_offset=5, len(ids)=128, so token_end_offset=5+128=133
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (100, 600, 5)})
        packs = [[[0, 0, 100, 600, 5, 133]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=128)
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
        packs = [[[0, 0, -1, -1, 0, 100], [0, 1, 0, 512, 0, 128]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=228)
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
        packs = [[[0, 0, -1, -1, 0, 128]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=128)
        with pytest.raises(ValueError, match="LongTextDataItem"):
            dataset[0]

    def test_consistency_check_error_longtext_fields_mismatch(self, tmp_path):
        """Consistency check raises ValueError when LongTextDataItem fields don't match pack config."""
        ids = [1] * 128
        # dataset returns char_start=100, but pack config expects char_start=999
        ds = _FakeDataset([ids], path="/ds.jsonl", long_text_meta={0: (100, 600, 5)})
        packs = [[[0, 0, 999, 1600, 5, 133]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=128)
        with pytest.raises(ValueError, match="mismatch"):
            dataset[0]

    def test_plain_tokenizefn_token_start_offset_applied(self, tmp_path):
        """Plain DataItem: input_ids[token_start_offset:token_end_offset] slicing is applied."""
        ids = list(range(200))
        ds = _FakeDataset([ids], path="/ds.jsonl")
        # Take tokens [50:150] — 100 tokens out of the 200-token sample
        packs = [[[0, 0, -1, -1, 50, 150]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=100)
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
        packs = [[[0, 0, 0, 1000, 10, 138]]]
        config_dir = str(tmp_path / "packs")
        _write_npy_pack_dir(config_dir, packs, ["/ds.jsonl"])

        dataset = CustomPackDataset([ds], config_dir, pack_max_length=128)
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


# ---------------------------------------------------------------------------
# Feature 8: stress test helpers (module-level for multiprocessing compatibility)
# ---------------------------------------------------------------------------

# Use a smaller scale for CI; the helpers support any num_slices for real stress runs.
_STRESS_NUM_SLICES = int(os.environ.get("STRESS_NUM_SLICES", 500_000))
_STRESS_PACK_MAX_LENGTH = int(os.environ.get("STRESS_PACK_MAX_LENGTH", 32768))


def generate_stress_pack_config(num_slices: int, pack_max_length: int, out_dir: str) -> int:
    """Generate boundaries.npy, samples.npy, paths.npy for stress testing.

    Token length per slice is drawn uniformly from [200, 16000]. Packs are
    filled greedily: a new pack starts whenever adding the next slice would
    exceed pack_max_length.

    Args:
        num_slices (int): Total number of slices to generate.
        pack_max_length (int): Maximum token count per pack.
        out_dir (str): Output NPY directory path.

    Returns:
        int: Number of packs generated.
    """
    rng = np.random.default_rng(42)
    token_lengths = rng.integers(200, 16001, size=num_slices, dtype=np.int64)

    boundaries: list[int] = [0]
    running = 0
    for i in range(num_slices):
        tl = int(token_lengths[i])
        if running > 0 and running + tl > pack_max_length:
            boundaries.append(i)
            running = 0
        running += tl
    boundaries.append(num_slices)

    samples = np.stack(
        [
            np.zeros(num_slices, dtype=np.int64),  # path_id
            np.arange(num_slices, dtype=np.int64),  # sample_idx
            np.full(num_slices, -1, dtype=np.int64),  # char_start
            np.full(num_slices, -1, dtype=np.int64),  # char_end
            np.zeros(num_slices, dtype=np.int64),  # tok_start
            token_lengths,  # tok_end
        ],
        axis=1,
    )

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "boundaries.npy"), np.array(boundaries, dtype=np.int64))
    np.save(os.path.join(out_dir, "samples.npy"), samples)
    np.save(os.path.join(out_dir, "paths.npy"), np.array(["mock://stress"], dtype=object))
    return len(boundaries) - 1


class _MockDataset:
    """Minimal dataset satisfying the JsonlDataset interface without file I/O.

    __getitem__ always returns a synthetic DataItem with 16001 zero-valued tokens,
    which covers any tok_end up to 16000 as produced by generate_stress_pack_config.

    Args:
        path (str): Dataset path for pack config matching.
        n_samples (int): Number of logical samples (used for len()).
    """

    _MOCK_IDS: list[int] = [0] * 16001

    def __init__(self, path: str, n_samples: int):
        self.path = path
        self.num_tokens = np.zeros(n_samples, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.num_tokens)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self._MOCK_IDS, "labels": self._MOCK_IDS, "num_tokens": len(self._MOCK_IDS)}


def _stress_worker_fn(
    rank: int,
    num_ranks: int,
    config_dir: str,
    pack_max_length: int,
    num_slices: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker: build CustomPackDataset (mmap=True), benchmark __getitem__ on random indices."""
    mock_ds = _MockDataset("mock://stress", num_slices)
    proc = psutil.Process()

    mem_before = proc.memory_full_info()
    t0 = time.perf_counter()
    ds = CustomPackDataset(
        [mock_ds],
        config_dir,
        pack_max_length,
        short_pack_strategy="padding",
        long_pack_strategy="truncate",
        mmap=True,
    )
    init_time = time.perf_counter() - t0
    mem_after = proc.memory_full_info()

    n_packs = len(ds)
    rng = np.random.default_rng(rank)
    indices = rng.choice(n_packs, size=min(200, n_packs), replace=False).tolist()

    t0 = time.perf_counter()
    for i in indices:
        items = ds[i]
        assert len(items) > 0
    getitem_time = time.perf_counter() - t0

    result_queue.put(
        {
            "rank": rank,
            "n_packs": n_packs,
            "init_time_s": round(init_time, 4),
            "rss_delta_mb": round((mem_after.rss - mem_before.rss) / 1024**2, 1),
            "pss_delta_mb": round((mem_after.pss - mem_before.pss) / 1024**2, 1),
            "getitem_time_s": round(getitem_time, 4),
            "n_items": len(indices),
        }
    )


def _measure_load_rss(use_mmap: bool, config_dir: str, result_queue: multiprocessing.Queue) -> None:
    """Subprocess: call load_config and report RSS and PSS deltas before any data access."""
    from xtuner.v1.datasets.custom_pack import load_config as _load_config

    proc = psutil.Process()
    mem_before = proc.memory_full_info()
    t0 = time.perf_counter()
    config = _load_config(config_dir, mmap=use_mmap)
    elapsed = time.perf_counter() - t0
    mem_after = proc.memory_full_info()
    result_queue.put(
        {
            "rss_delta": mem_after.rss - mem_before.rss,
            "pss_delta": mem_after.pss - mem_before.pss,
            "elapsed_s": round(elapsed, 4),
        }
    )
    _ = config  # keep alive until measured


# ---------------------------------------------------------------------------
# Feature 8: stress tests
# ---------------------------------------------------------------------------


class TestStress:
    def test_generate_stress_pack_config(self, tmp_path):
        """generate_stress_pack_config produces a valid NPY directory."""
        config_dir = str(tmp_path / "stress")
        n_packs = generate_stress_pack_config(10_000, 32768, config_dir)

        config = load_config(config_dir, mmap=False)
        assert len(config["boundaries"]) == n_packs + 1
        assert config["samples"].shape == (10_000, 6)
        assert list(config["paths"]) == ["mock://stress"]
        assert int(config["boundaries"][-1]) == 10_000

    def test_multiprocess_getitem(self, tmp_path):
        """8 processes independently build CustomPackDataset (mmap=True) and call __getitem__.

        Run with 2 Billion slices (set STRESS_NUM_SLICES=2000000000), the output is like:
        ```
        [Rank 0] n_packs=589147374 init=82.361s rss_delta=91554.0MB pss_delta=11182.2MB getitem=0.024s n_items=200
        [Rank 1] n_packs=589147374 init=82.352s rss_delta=91554.0MB pss_delta=11327.0MB getitem=0.024s n_items=200
        [Rank 2] n_packs=589147374 init=82.374s rss_delta=91554.0MB pss_delta=11372.4MB getitem=0.024s n_items=200
        [Rank 3] n_packs=589147374 init=82.308s rss_delta=91554.0MB pss_delta=11399.8MB getitem=0.024s n_items=200
        [Rank 4] n_packs=589147374 init=82.317s rss_delta=91554.0MB pss_delta=11419.3MB getitem=0.024s n_items=200
        [Rank 5] n_packs=589147374 init=82.325s rss_delta=91554.0MB pss_delta=11433.9MB getitem=0.024s n_items=200
        [Rank 6] n_packs=589147374 init=83.762s rss_delta=91554.0MB pss_delta=20910.8MB getitem=0.026s n_items=200
        [Rank 7] n_packs=589147374 init=82.317s rss_delta=91554.0MB pss_delta=11445.3MB getitem=0.024s n_items=200
        ```
        """
        config_dir = str(tmp_path / "stress")
        generate_stress_pack_config(_STRESS_NUM_SLICES, _STRESS_PACK_MAX_LENGTH, config_dir)

        ctx = multiprocessing.get_context("fork")
        result_queue: multiprocessing.Queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_stress_worker_fn,
                args=(rank, 8, config_dir, _STRESS_PACK_MAX_LENGTH, _STRESS_NUM_SLICES, result_queue),
            )
            for rank in range(8)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"Worker rank process exited with code {p.exitcode}"

        results = [result_queue.get() for _ in range(8)]
        assert all(r["n_items"] > 0 for r in results)
        for r in sorted(results, key=lambda x: x["rank"]):
            print(
                f"[Rank {r['rank']}] n_packs={r['n_packs']} init={r['init_time_s']:.3f}s "
                f"rss_delta={r['rss_delta_mb']:.1f}MB pss_delta={r['pss_delta_mb']:.1f}MB "
                f"getitem={r['getitem_time_s']:.3f}s n_items={r['n_items']}"
            )

    def test_mmap_memory_saving(self, tmp_path):
        """mmap=True uses less RSS than mmap=False before any data access.

        Run with 2 Billion slices (set STRESS_NUM_SLICES=2000000000), the output is like:
        ```
        mmap=True  RSS delta: 0.2MB  PSS delta: 0.4MB
        mmap=False RSS delta: 96047.9MB  PSS delta: 96047.9MB
        ```
        """
        config_dir = str(tmp_path / "stress")
        generate_stress_pack_config(_STRESS_NUM_SLICES, _STRESS_PACK_MAX_LENGTH, config_dir)

        ctx = multiprocessing.get_context("fork")
        q: multiprocessing.Queue = ctx.Queue()

        p_mmap = ctx.Process(target=_measure_load_rss, args=(True, config_dir, q))
        p_mmap.start()
        p_mmap.join(timeout=30)
        assert p_mmap.exitcode == 0
        result_mmap = q.get()

        p_nommap = ctx.Process(target=_measure_load_rss, args=(False, config_dir, q))
        p_nommap.start()
        p_nommap.join(timeout=1800)
        assert p_nommap.exitcode == 0
        result_nommap = q.get()

        rss_mmap = result_mmap["rss_delta"]
        pss_mmap = result_mmap["pss_delta"]
        elapsed_mmap = result_mmap["elapsed_s"]
        rss_nommap = result_nommap["rss_delta"]
        pss_nommap = result_nommap["pss_delta"]
        elapsed_nommap = result_nommap["elapsed_s"]

        print(
            f"\nmmap=True  RSS delta: {rss_mmap / 1024**2:.1f}MB  PSS delta: {pss_mmap / 1024**2:.1f}MB  elapsed: {elapsed_mmap:.4f}s"
            f"\nmmap=False RSS delta: {rss_nommap / 1024**2:.1f}MB  PSS delta: {pss_nommap / 1024**2:.1f}MB  elapsed: {elapsed_nommap:.4f}s"
        )
        assert rss_nommap > rss_mmap, (
            f"mmap=False should use more RSS than mmap=True: "
            f"nommap={rss_nommap / 1024**2:.1f}MB vs mmap={rss_mmap / 1024**2:.1f}MB"
        )
        assert pss_nommap > pss_mmap, (
            f"mmap=False should use more PSS than mmap=True: "
            f"nommap={pss_nommap / 1024**2:.1f}MB vs mmap={pss_mmap / 1024**2:.1f}MB"
        )
