"""Integration test: DataloaderConfig.build with preset pack / HardPackDataset and samplers."""

import json
import os
import random
from dataclasses import dataclass
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
            idx_i = int(idx)  # TODO: ConcatDataset.get_sample_idx_from_idx
            L = int(num_tokens[idx_i])
            st = 0 if i else s_off
            ed = L if i < len(indices) - 1 else e_off
            # TODO: ConcatDataset.get_dataset_id_from_idx (path_id)
            rows.append([path_id, idx_i, -1, -1, st, ed])
        boundaries.append(len(rows))
    samples = np.asarray(rows, dtype=np.int64).reshape(-1, 6) if rows else np.empty((0, 6), dtype=np.int64)
    return {"boundaries": np.asarray(boundaries, dtype=np.int64), "samples": samples}


def test_simple_hard_preset_pack_config_matches_buildin_hard_pack():
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


@dataclass(frozen=True)
class CreateDataloaderConfigResult:
    dataloader_cfg: DataloaderConfig
    global_batch_size: int
    micro_batch_size: int
    seed_ref: int
    sampler_case: str
    num_packs_ref: int | None


def create_dataloader_config(
    tmp_path: Path,
    tokenizer,
    pack_case: str,
    sampler_case: str,
    *,
    sample_num: int = 12,
    pack_max_length: int = 32,
    global_batch_size: int = 2,
    dp_size: int = 1,
    seed_ref: int = 1,
    num_workers: int = 0,
) -> CreateDataloaderConfigResult:
    """Prepare jsonl + optional preset NPY / order, then return :class:`DataloaderConfig` and build metadata.

    ``seed`` controls the sample permutation for preset packing (``preset_pack*``). For comparisons with
    ``HardPackDataset``, set it equal to the ``seed`` passed to :meth:`DataloaderConfig.build` so both use the same
    ``numpy.random.RandomState`` sequence as in ``HardPackDataset._compute_pack_infos``.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)

    if pack_case == "hard_pack" and sampler_case in ("preset_seq_sampler", "preset_group_sampler"):
        pytest.skip("PresetSampler 需要 pack_level='preset' 与 PresetPackDataset，与 HardPackDataset 不兼容")

    # TODO: multiple jsonl files with ConcatDataset
    # TODO: jsonl file with sample_ratio
    jsonl_path = tmp_path / "data.jsonl"
    records = [
        {"messages": [{"role": "pretrain", "content": f"This is an example of pretrain text sample {i}. " + "xtuner " * (3 + (i % 5))}]}
        for i in range(sample_num)
    ]
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    anno = str(jsonl_path.resolve())

    dataset_config_list = [
        {
            "dataset": DatasetConfig(anno_path=anno, disable_filter=True, sample_ratio=1.0),
            "tokenize_fn": PretrainTokenizeFunctionConfig(),
        }
    ]

    # TODO: use JsonlDataset to cache first and get num_tokens from cache
    tok_fn = PretrainTokenizeFunctionConfig().build(tokenizer)
    num_tokens = np.array([tok_fn(r)["num_tokens"] for r in records], dtype=np.int64)

    pack_dir: Path | None = None
    num_packs_ref: int | None = None
    pack_level: Literal["preset", "hard"]

    if pack_case == "hard_pack":
        assert int(num_tokens.sum()) >= 2 * pack_max_length, "need enough tokens for multiple packs"
        pack_level = "hard"
        num_packs_ref = None
        print(f"\n\n--------------- original samples: {len(records)}, pack: HardPackDataset in DataloaderConfig.build\n\n")
    elif pack_case in ("preset_pack_simple", "preset_pack"):
        inds = np.arange(sample_num, dtype=np.int64)
        np.random.RandomState(seed_ref).shuffle(inds)  # type: ignore[arg-type]
        if pack_case == "preset_pack_simple":
            cfg = get_pack_config_by_simple_hard_split(inds, 0, num_tokens, pack_max_length, 1)
        else:
            infos = get_pack_infos_by_hard_split(inds, 0, num_tokens, pack_max_length, pack_workers=8)
            cfg = get_pack_config_from_pack_infos_by_hard_split(infos, 0, num_tokens)
        assert len(cfg["boundaries"]) >= 3, "need >=2 full packs"

        print(f"\n\n--------------- original samples: {len(records)}, after pack: {len(cfg['boundaries']) - 1}\n\n")

        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        np.save(pack_dir / "boundaries.npy", cfg["boundaries"])
        np.save(pack_dir / "samples.npy", cfg["samples"])
        (pack_dir / "paths.json").write_text(json.dumps([anno]), encoding="utf-8")

        num_packs_ref = int(cfg["boundaries"].shape[0] - 1)
        pack_level = "preset"
    else:
        raise ValueError(f"unknown pack_case: {pack_case!r}")

    micro_batch_size = global_batch_size // dp_size
    # TODO: a dist version of this test
    order_path = tmp_path / "order.npy"
    if sampler_case == "preset_seq_sampler":
        assert num_packs_ref is not None
        get_sampler_config(order_path, mode="sequential", num_packs=num_packs_ref)

        sampler_type = "preset"
        group_by_length = False
    elif sampler_case == "preset_group_sampler":
        assert pack_dir is not None and num_packs_ref is not None
        forced = DataloaderConfig._force_preset_pack_settings(dataset_config_list)
        pack_ds_for_longest = PresetPackDataset(
            build_datasets(forced, tokenizer),
            pack_config_path=str(pack_dir),
            pack_max_length=pack_max_length,
        )
        get_sampler_config(
            order_path,
            mode="length_grouped",
            num_packs=num_packs_ref,
            longest=pack_ds_for_longest.longest,
            global_batch_size=global_batch_size,
            world_size=dp_size,
            seed=seed_ref,
            epoch=0,
        )

        sampler_type = "preset"
        group_by_length = False
    else:  # group_sampler
        sampler_type = "none"
        group_by_length = True

    dataloader_cfg = DataloaderConfig(
        dataset_config_list=dataset_config_list,
        pack_level=pack_level,
        sampler_type=sampler_type,
        group_by_length=group_by_length,
        pack_config_path=str(pack_dir) if pack_dir is not None else None,
        sampler_config_path=str(order_path) if pack_level == "preset" else None,
        pack_max_length=pack_max_length,
        round_up=False,  # preset sampler does not support round up
        pad_token_id=None,
        num_workers=num_workers,
    )

    return CreateDataloaderConfigResult(
        dataloader_cfg=dataloader_cfg,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        seed_ref=seed_ref,
        sampler_case=sampler_case,
        num_packs_ref=num_packs_ref,
    )


class _StubJsonl:
    """Minimal stand-in for ``JsonlDataset``: only ``path`` / ``__len__`` are used by ``PresetPackDataset`` init."""

    def __init__(self, path: str, n_samples: int) -> None:
        self.path = path
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        raise AssertionError("not used in this test")


def test_preset_pack_longest_matches_buildin_hard_pack(tmp_path: Path) -> None:
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

    # check LengthGroupedSampler can be initialized with PresetPackDataset
    LengthGroupedSampler(ds, global_batch_size=2, dp_mesh=None, seed=123)


@pytest.mark.parametrize(
    "pack_case",
    [
        pytest.param("preset_pack_simple", id="preset_pack_simple"),
        pytest.param("preset_pack", id="preset_pack"),
        pytest.param("hard_pack", id="hard_pack"),
    ],
)
@pytest.mark.parametrize(
    "sampler_case",
    [
        pytest.param("preset_seq_sampler", id="preset_seq_sampler"),
        pytest.param("preset_group_sampler", id="preset_group_sampler"),
        pytest.param("group_sampler", id="pack_and_group_sampler_consistent"),
    ],
)
def test_various_dataloader_configs_consistent(tmp_path: Path, pack_case: str, sampler_case: str) -> None:
    tokenizer_path = os.environ["QWEN3_MOE_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    bundle = create_dataloader_config(
        tmp_path,
        tokenizer,
        pack_case,
        sampler_case,
        seed_ref=1,
    )
    dataloader_cfg = bundle.dataloader_cfg
    global_batch_size = bundle.global_batch_size
    micro_batch_size = bundle.micro_batch_size
    seed_ref = bundle.seed_ref
    seed_ref2 = seed_ref + 1
    num_packs_ref = bundle.num_packs_ref

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

    _pack_info = num_packs_ref if num_packs_ref is not None else "hard (runtime)"
    print(f"--------------- packed samples: {_cnt}, original packed samples: {_pack_info}")


def test_preset_pack_sampler_matches_buildin_hard_pack_group_sampler(tmp_path: Path) -> None:
    """Preset 打包 + PresetSampler(与 LengthGroupedSampler 同序的 order.npy) 应与运行时 HardPack + LengthGroupedSampler 一致。"""
    tokenizer_path = os.environ["QWEN3_MOE_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    seed = 1
    common = dict(
        sample_num=12,
        pack_max_length=32,
        global_batch_size=3,
        dp_size=1,
        seed_ref=seed,
    )
    pre = create_dataloader_config(tmp_path / "preset_run", tokenizer, "preset_pack", "preset_group_sampler", **common)
    hard = create_dataloader_config(tmp_path / "hard_run", tokenizer, "hard_pack", "group_sampler", **common)

    dl_pre = pre.dataloader_cfg.build(
        tokenizer,
        dp_mesh=None,
        global_batch_size=pre.global_batch_size,
        micro_batch_size=pre.micro_batch_size,
        seed=seed,
        shuffle=True,
    )
    dl_hard = hard.dataloader_cfg.build(
        tokenizer,
        dp_mesh=None,
        global_batch_size=hard.global_batch_size,
        micro_batch_size=hard.micro_batch_size,
        seed=seed,
        shuffle=True,
    )

    batches_pre = list(dl_pre)
    batches_hard = list(dl_hard)
    assert len(batches_pre) == len(batches_hard)
    for batch_pre, batch_hard in zip(batches_pre, batches_hard):
        assert len(batch_pre) == len(batch_hard)
        for x, y in zip(batch_pre, batch_hard):
            assert torch.equal(x["seq_ctx"].input_ids, y["seq_ctx"].input_ids)
            assert torch.equal(x["shifted_labels"], y["shifted_labels"])
