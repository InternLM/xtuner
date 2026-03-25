"""Integration test: JsonlDataset + PresetPackDataset + PresetSampler via DataloaderConfig.build."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.packing import get_pack_infos_by_hard_split
from xtuner.v1.datasets.preset_pack import PresetPackDataset
from xtuner.v1.datasets.sampler import LengthGroupedSampler


def get_pack_config_by_simple_hard_split(
    inds: np.ndarray, dataset_id: int, num_tokens: np.ndarray, pack_max_length: int, pack_workers: int = 1
):
    """Like packing.get_pack_infos_by_hard_split: walk inds order, hard-split at pack_max_length -> preset NPY."""
    _ = pack_workers
    rows: list[list[int]] = []
    boundaries: list[int] = [0]
    cur: list[list[int]] = []
    cur_sum = 0

    def close():
        nonlocal cur, cur_sum
        if cur_sum != pack_max_length:
            return
        rows.extend(cur)
        boundaries.append(len(rows))
        cur, cur_sum = [], 0

    for s_idx in inds.astype(np.int64, copy=False).tolist():
        pos, L = 0, int(num_tokens[s_idx])
        while pos < L:
            room = pack_max_length - cur_sum
            if room == 0:
                close()
                room = pack_max_length
            take = min(L - pos, room)
            cur.append([dataset_id, s_idx, -1, -1, pos, pos + take])
            cur_sum += take
            pos += take
            close()
    samples = np.asarray(rows, dtype=np.int64).reshape(-1, 6) if rows else np.empty((0, 6), dtype=np.int64)
    return {"boundaries": np.asarray(boundaries, dtype=np.int64), "samples": samples}


def get_pack_config_from_pack_infos_by_hard_split(
    pack_infos: dict[str, np.ndarray], path_id: int, num_tokens: np.ndarray
) -> dict[str, np.ndarray]:
    """Same keys as ``get_pack_config_by_simple_hard_split``; built from ``get_pack_infos_by_hard_split`` output."""
    npack = int(pack_infos["dataset_id"].shape[0])
    rows: list[list[int]] = []
    boundaries: list[int] = [0]
    cu, ix = pack_infos["indices_cu_len"], pack_infos["indices"]
    starts, ends = pack_infos["start_offset"], pack_infos["end_offset"]
    for item in range(npack):
        i0 = 0 if item == 0 else int(cu[item - 1])
        i1 = int(cu[item])
        indices = ix[i0:i1]
        s_off, e_off = int(starts[item]), int(ends[item])
        for i, idx in enumerate(indices):
            idx_i = int(idx)
            L = int(num_tokens[idx_i])
            st = 0 if i else s_off
            ed = L if i < len(indices) - 1 else e_off
            rows.append([path_id, idx_i, -1, -1, st, ed])
        boundaries.append(len(rows))
    samples = np.asarray(rows, dtype=np.int64).reshape(-1, 6) if rows else np.empty((0, 6), dtype=np.int64)
    return {"boundaries": np.asarray(boundaries, dtype=np.int64), "samples": samples}


def test_hard_split_pack_config_matches_packing_implementation():
    rng = np.random.RandomState(42)
    num_tokens = rng.randint(8, 25, size=24).astype(np.int64)
    inds = rng.permutation(24).astype(np.int64)
    pack_max_length = 32
    assert int(num_tokens.sum()) >= 2 * pack_max_length

    simple = get_pack_config_by_simple_hard_split(inds, 0, num_tokens, pack_max_length, 1)
    infos = get_pack_infos_by_hard_split(inds, 0, num_tokens, pack_max_length, pack_workers=1)
    ref = get_pack_config_from_pack_infos_by_hard_split(infos, 0, num_tokens)

    np.testing.assert_array_equal(simple["boundaries"], ref["boundaries"])
    np.testing.assert_array_equal(simple["samples"], ref["samples"])


class _StubJsonl:
    """Minimal stand-in for ``JsonlDataset``: only ``path`` / ``__len__`` are used by ``PresetPackDataset`` init."""

    def __init__(self, path: str, n_samples: int) -> None:
        self.path = path
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        raise AssertionError("not used in this test")


def test_preset_pack_longest_matches_hard_pack_and_length_grouped_sampler(tmp_path: Path) -> None:
    rng = np.random.RandomState(7)
    num_tokens = rng.randint(5, 19, size=20).astype(np.int64)
    inds = rng.permutation(20).astype(np.int64)
    pack_max_length = 24
    anno = str((tmp_path / "stub.jsonl").resolve())
    (tmp_path / "stub.jsonl").write_text("", encoding="utf-8")

    infos = get_pack_infos_by_hard_split(inds, 0, num_tokens, pack_max_length, pack_workers=1)
    ref_longest = infos["longest"]
    cfg = get_pack_config_from_pack_infos_by_hard_split(infos, 0, num_tokens)

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    np.save(pack_dir / "boundaries.npy", cfg["boundaries"])
    np.save(pack_dir / "samples.npy", cfg["samples"])
    (pack_dir / "paths.json").write_text(json.dumps([anno]), encoding="utf-8")

    ds = PresetPackDataset(
        [_StubJsonl(anno, len(num_tokens))],
        pack_config_path=str(pack_dir),
        pack_max_length=pack_max_length,
        mmap=True,
    )
    np.testing.assert_array_equal(ds.longest, ref_longest)
    assert isinstance(ds._samples, np.memmap) or hasattr(ds._samples, "filename")

    LengthGroupedSampler(ds, global_batch_size=2, dp_mesh=None, seed=123)


def test_preset_dataloader_build_is_deterministic(tmp_path: Path) -> None:
    tokenizer_path = os.environ["QWEN3_MOE_PATH"]
    # TODO: multiple jsonl files with ConcatDataset
    # TODO: jsonl file with sample_ratio
    jsonl_path = tmp_path / "data.jsonl"
    records = [
        {"messages": [{"role": "pretrain", "content": f"This is an example of pretrain text sample {i}. " + "xtuner " * (3 + (i % 5))}]}
        for i in range(12)
    ]
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    anno = str(jsonl_path.resolve())

    # TODO: use JsonlDataset to cache first and get num_tokens from cache
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tok_fn = PretrainTokenizeFunctionConfig().build(tokenizer)
    num_tokens = np.array([tok_fn(r)["num_tokens"] for r in records], dtype=np.int64)

    pack_max_length = 32
    inds = np.random.RandomState(0).permutation(len(records))

    # TODO: use hard pack logic, compare with current hard pack logic
    cfg = get_pack_config_by_simple_hard_split(inds, 0, num_tokens, pack_max_length, 1)
    assert len(cfg["boundaries"]) >= 3, "need >=2 full packs"

    print(f"\n\n--------------- original samples: {len(records)}, after pack: {len(cfg['boundaries']) - 1}\n\n")

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    np.save(pack_dir / "boundaries.npy", cfg["boundaries"])
    np.save(pack_dir / "samples.npy", cfg["samples"])
    (pack_dir / "paths.json").write_text(json.dumps([anno]), encoding="utf-8")

    # TODO: a dist version of this test
    # TODO: use group by length logic
    num_packs = int(cfg["boundaries"].shape[0] - 1)
    order_path = tmp_path / "order.npy"
    order = np.arange(num_packs, dtype=np.int64)
    np.save(order_path, order)

    dataloader_cfg = DataloaderConfig(
        dataset_config_list=[
            {
                "dataset": DatasetConfig(name="preset", anno_path=anno),
                "tokenize_fn": PretrainTokenizeFunctionConfig(),
            }
        ],
        pack_level="preset",
        pack_config_path=str(pack_dir),
        sampler_config_path=str(order_path),
        pack_max_length=pack_max_length,
        pad_token_id=None,
        num_workers=0,
    )

    def _build(seed: int, shuffle: bool = False):
        return dataloader_cfg.build(
            tokenizer,
            dp_mesh=None,
            global_batch_size=2,
            micro_batch_size=1,
            seed=seed,
            shuffle=shuffle,
        )

    # seed and shuffle should not affect the output in preset mode
    dl_a = _build(1, shuffle=False)
    dl_b = _build(2, shuffle=True)

    _cnt = 0
    for batch_a, batch_b in zip(dl_a, dl_b):
        assert len(batch_a) == len(batch_b)
        for x, y in zip(batch_a, batch_b):
            _cnt += 1
            assert torch.equal(x["seq_ctx"].input_ids, y["seq_ctx"].input_ids)
            assert torch.equal(x["shifted_labels"], y["shifted_labels"])
    
    print(f"--------------- packed samples: {_cnt}, sampler len: {len(order)}")
