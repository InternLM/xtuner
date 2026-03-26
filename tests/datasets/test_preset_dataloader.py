"""Integration test: JsonlDataset + PresetPackDataset + PresetSampler via DataloaderConfig.build."""

import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig, build_datasets
from xtuner.v1.datasets.packing import get_pack_infos_by_hard_split
from xtuner.v1.datasets.preset_pack import PresetPackDataset
from xtuner.v1.datasets.sampler import LengthGroupedSampler, get_length_grouped_indices


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


def get_sampler_config(
    order_path: Path,
    *,
    mode: Literal["sequential", "length_grouped"],
    num_packs: int,
    longest: np.ndarray | list | tuple | None = None,
    global_batch_size: int = 2,
    world_size: int = 1,
    seed: int = 0,
    epoch: int = 0,
) -> np.ndarray:
    """Build pack index order and persist to ``order_path`` (``.npy``).

    Args:
        order_path: Output path for a 1-D int64 array of pack indices (same format as training).
        mode:
            ``sequential`` — ``0..num_packs-1`` (legacy / PresetSampler baseline).
            ``length_grouped`` — same megabatch + sort + group shuffle logic as
            :class:`LengthGroupedSampler` via :func:`get_length_grouped_indices`.
        num_packs: Number of valid packs (length of the order permutation).
        longest: Per-pack max token length; required when ``mode=='length_grouped'``.
        global_batch_size: Same as dataloader ``global_batch_size`` (drives megabatch sizing).
        world_size: Same as distributed world size (``group_size`` in length-grouped logic).
        seed / epoch: Match :class:`LengthGroupedSampler` (`seed + epoch` for both RNGs).

    Returns:
        The saved order array.
    """
    if mode == "sequential":
        order = np.arange(num_packs, dtype=np.int64)
    elif mode == "length_grouped":
        if longest is None:
            raise ValueError("longest is required when mode is 'length_grouped'")
        longest_arr = np.asarray(longest)
        if longest_arr.shape[0] != num_packs:
            raise ValueError(f"len(longest)={longest_arr.shape[0]} must equal num_packs={num_packs}")
        mega_batch_mult = min(
            num_packs // (global_batch_size * LengthGroupedSampler.GROUP_BATCH_FACTOR),
            LengthGroupedSampler.MAX_GROUP_BATCH_SIZE,
        )
        if mega_batch_mult == 0:
            mega_batch_mult = 1
        group_batch_size = mega_batch_mult * global_batch_size
        group_size = world_size
        torch_generator = torch.Generator()
        torch_generator.manual_seed(seed + epoch)
        random_generator = random.Random()
        random_generator.seed(seed + epoch)
        order_list = get_length_grouped_indices(
            max_lengths=longest_arr,
            group_batch_size=group_batch_size,
            group_size=group_size,
            torch_generator=torch_generator,
            random_generator=random_generator,
        )
        order = np.asarray(order_list, dtype=np.int64)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    np.save(order_path, order)
    return order


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


@pytest.mark.parametrize(
    "sampler_case",
    [
        pytest.param("preset_seq_sampler", id="preset_seq_sampler"),
        pytest.param("preset_group_sampler", id="preset_group_sampler"),
        pytest.param("length_grouped", id="pack_and_group_sampler_consistent"),
    ],
)
def test_preset_dataloader(tmp_path: Path, sampler_case: str) -> None:
    tokenizer_path = os.environ["QWEN3_MOE_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # step 1: prepare jsonl and its num_tokens
    sample_num = 12
    # TODO: multiple jsonl files with ConcatDataset
    # TODO: jsonl file with sample_ratio
    jsonl_path = tmp_path / "data.jsonl"
    records = [
        {"messages": [{"role": "pretrain", "content": f"This is an example of pretrain text sample {i}. " + "xtuner " * (3 + (i % 5))}]}
        for i in range(sample_num)
    ]
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    anno = str(jsonl_path.resolve())

    # TODO: use JsonlDataset to cache first and get num_tokens from cache
    tok_fn = PretrainTokenizeFunctionConfig().build(tokenizer)
    num_tokens = np.array([tok_fn(r)["num_tokens"] for r in records], dtype=np.int64)

    inds = np.random.RandomState(0).permutation(sample_num)

    # step 2: prepare pack config
    pack_max_length = 32
    # TODO: use hard pack logic, compare with current hard pack logic
    cfg = get_pack_config_by_simple_hard_split(inds, 0, num_tokens, pack_max_length, 1)
    assert len(cfg["boundaries"]) >= 3, "need >=2 full packs"

    print(f"\n\n--------------- original samples: {len(records)}, after pack: {len(cfg['boundaries']) - 1}\n\n")

    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    np.save(pack_dir / "boundaries.npy", cfg["boundaries"])
    np.save(pack_dir / "samples.npy", cfg["samples"])
    (pack_dir / "paths.json").write_text(json.dumps([anno]), encoding="utf-8")

    num_packs = int(cfg["boundaries"].shape[0] - 1)
    pack_level = "preset"

    # step 3: prepare sampler config (.npy order; length_grouped matches LengthGroupedSampler internals)
    global_batch_size = 2
    dp_size = 1
    seed_ref = 1
    seed_ref2 = 2
    micro_batch_size = global_batch_size // dp_size
    # TODO: a dist version of this test
    order_path = tmp_path / "order.npy"
    dataset_config_list = [
        {
            "dataset": DatasetConfig(name="preset", anno_path=anno),
            "tokenize_fn": PretrainTokenizeFunctionConfig(),
        }
    ]
    if sampler_case == "preset_seq_sampler":
        get_sampler_config(order_path, mode="sequential", num_packs=num_packs)

        sampler_type = "preset"
        group_by_length = False
    elif sampler_case == "preset_group_sampler":
        forced = DataloaderConfig._force_preset_pack_settings(dataset_config_list)
        pack_ds_for_longest = PresetPackDataset(
            build_datasets(forced, tokenizer),
            pack_config_path=str(pack_dir),
            pack_max_length=pack_max_length,
        )
        get_sampler_config(
            order_path,
            mode="length_grouped",
            num_packs=num_packs,
            longest=pack_ds_for_longest.longest,
            global_batch_size=global_batch_size,
            world_size=dp_size,
            seed=seed_ref,
            epoch=0,
        )

        sampler_type = "preset"
        group_by_length = False
    else:  # length_grouped
        sampler_type = "none"
        group_by_length = True

    # step 4: compose dataloader config
    dataloader_cfg = DataloaderConfig(
        dataset_config_list=dataset_config_list,
        pack_level=pack_level,
        sampler_type=sampler_type,
        group_by_length=group_by_length,
        pack_config_path=str(pack_dir),
        sampler_config_path=str(order_path),
        pack_max_length=pack_max_length,
        pad_token_id=None,
        num_workers=0,
    )

    # step 5: build dataloader
    def _build(seed: int, shuffle: bool = False):
        return dataloader_cfg.build(
            tokenizer,
            dp_mesh=None,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            seed=seed,
            shuffle=shuffle,
        )

    if sampler_case in ("preset_seq_sampler", "preset_group_sampler"):
        # PresetSampler follows order.npy; seed/shuffle should not change batches.
        dl_a = _build(seed_ref, shuffle=False)
        dl_b = _build(seed_ref2, shuffle=True)
    else:
        # LengthGroupedSampler: same seed + shuffle=True on repeated builds should match.
        dl_a = _build(seed_ref, shuffle=True)
        dl_b = _build(seed_ref, shuffle=True)

    # step 6: verify dataloader output is consistent
    _cnt = 0
    for batch_a, batch_b in zip(dl_a, dl_b):
        assert len(batch_a) == len(batch_b)
        for x, y in zip(batch_a, batch_b):
            _cnt += 1
            assert torch.equal(x["seq_ctx"].input_ids, y["seq_ctx"].input_ids)
            assert torch.equal(x["shifted_labels"], y["shifted_labels"])

    print(f"--------------- packed samples: {_cnt}, original packed samples: {num_packs}")
