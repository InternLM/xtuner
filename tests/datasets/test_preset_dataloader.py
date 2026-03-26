"""Integration test: DataloaderConfig.build with preset pack / HardPackDataset and samplers."""

import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoTokenizer

from xtuner.v1.utils.device import get_device

from itertools import chain

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig, get_dataloader_state, load_dataloader_state
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.packing import get_pack_infos_by_hard_split
from xtuner.v1.datasets.preset_pack import PresetPackDataset
from xtuner.v1.datasets.sampler import LengthGroupedSampler
from xtuner.v1.datasets.utils import (
    concat_cumulative_sizes_from_lengths,
    get_dataset_id_and_sample_idx_from_idx,
    get_longest,
    get_pack_config_from_pack_infos_by_hard_split,
    get_sampler_config,
)

DEVICE = get_device()


def assert_dataloader_batches_seq_and_labels_equal(batches_a, batches_b) -> int:
    """对比两份 dataloader 产出：逐 batch、逐样本检查 ``seq_ctx.input_ids`` 与 ``shifted_labels`` 一致。

    Returns:
        参与比较的样本对数量（所有 batch 内 ``zip`` 后的元素对总数）。
    """
    assert len(batches_a) == len(batches_b)
    n = 0
    for batch_a, batch_b in zip(batches_a, batches_b):
        assert len(batch_a) == len(batch_b)
        for x, y in zip(batch_a, batch_b):
            n += 1
            assert torch.equal(x["seq_ctx"].input_ids, y["seq_ctx"].input_ids)
            assert torch.equal(x["shifted_labels"], y["shifted_labels"])
    return n


def get_pack_config_by_simple_hard_split(
    inds: np.ndarray,
    dataset_id: int,
    num_tokens: np.ndarray,
    pack_max_length: int,
    pack_workers: int = 1,
    *,
    concat_cumulative_sizes: np.ndarray | None = None,
):
    """Like packing.get_pack_infos_by_hard_split: walk inds order, hard-split at pack_max_length -> preset NPY.

    If ``concat_cumulative_sizes`` is set (``ConcatDataset.cumulative_sizes``), ``inds`` / ``num_tokens`` use
    the flat concatenated index space; each row's ``[path_id, sample_idx, ...]`` uses the mapped sub-dataset id
    and local index (same convention as :func:`get_dataset_id_and_sample_idx_from_idx`).
    """
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
        s_idx_i = int(s_idx)
        if concat_cumulative_sizes is not None:
            path_id, local_idx = get_dataset_id_and_sample_idx_from_idx(s_idx_i, concat_cumulative_sizes)
        else:
            path_id, local_idx = dataset_id, s_idx_i
        pos, L = 0, int(num_tokens[s_idx_i])
        while pos < L:
            room = pack_max_length - cur_sum
            if room == 0:
                close()
                room = pack_max_length
            take = min(L - pos, room)
            cur.append([path_id, local_idx, -1, -1, pos, pos + take])
            cur_sum += take
            pos += take
            close()
    boundaries_arr = np.asarray(boundaries, dtype=np.int64)
    samples = np.asarray(rows, dtype=np.int64).reshape(-1, 6) if rows else np.empty((0, 6), dtype=np.int64)
    return {"boundaries": boundaries_arr, "samples": samples, "longest": get_longest(boundaries_arr, samples)}


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
    np.testing.assert_array_equal(simple["longest"], ref["longest"])


def test_get_dataset_id_and_sample_idx_matches_torch_concat_dataset():
    from torch.utils.data import ConcatDataset

    class _LenDS(torch.utils.data.Dataset):
        def __init__(self, n: int) -> None:
            self._n = n

        def __len__(self) -> int:
            return self._n

        def __getitem__(self, idx: int):
            return idx

    lens = [3, 7, 4]
    cu = concat_cumulative_sizes_from_lengths(lens)
    ds = ConcatDataset([_LenDS(n) for n in lens])
    assert len(ds) == int(cu[-1])
    for flat in range(len(ds)):
        sub_id, sample_idx = get_dataset_id_and_sample_idx_from_idx(flat, cu)
        assert ds[flat] == sample_idx
        # sub-dataset identity: cumulative prefix
        assert flat < int(cu[sub_id])
        prev = 0 if sub_id == 0 else int(cu[sub_id - 1])
        assert flat >= prev


@pytest.mark.parametrize("pack_workers", [1, 8])
def test_preset_pack_config_with_multiple_datasets_matches_concat_dataset_reference(pack_workers: int) -> None:
    """``ConcatDataset``-style flat indices: preset samples rows use per-jsonl path_id + local sample_idx."""
    rng = np.random.RandomState(202)

    n0, n1 = 11, 15
    tok0 = rng.randint(6, 18, size=n0).astype(np.int64)
    tok1 = rng.randint(6, 18, size=n1).astype(np.int64)
    num_tokens = np.concatenate([tok0, tok1])
    cu = concat_cumulative_sizes_from_lengths([n0, n1])

    pack_max_length = 28
    assert int(num_tokens.sum()) >= 2 * pack_max_length

    # you can filter or do sample_ratio here to get new inds
    inds = rng.permutation(n0 + n1).astype(np.int64)

    simple = get_pack_config_by_simple_hard_split(
        inds, 0, num_tokens, pack_max_length, 1, concat_cumulative_sizes=cu
    )

    infos = get_pack_infos_by_hard_split(
        inds, 0, num_tokens, pack_max_length, pack_workers=pack_workers
    )
    ref = get_pack_config_from_pack_infos_by_hard_split(infos, 0, num_tokens, concat_cumulative_sizes=cu)

    np.testing.assert_array_equal(simple["boundaries"], ref["boundaries"])
    np.testing.assert_array_equal(simple["samples"], ref["samples"])
    np.testing.assert_array_equal(simple["longest"], ref["longest"])


class _StubJsonl:
    """Minimal stand-in for ``JsonlDataset``: only ``path`` / ``__len__`` are used by ``PresetPackDataset`` init."""

    def __init__(self, path: str, n_samples: int) -> None:
        self.path = path
        self._n = n_samples

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        raise AssertionError("not used in this test")


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

    # Step 1: jsonl
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
            "dataset": DatasetConfig(anno_path=anno, disable_filter=True, sample_ratio=1.0, enable_mmap_shared=True),
            "tokenize_fn": PretrainTokenizeFunctionConfig(),
        }
    ]

    # TODO: use JsonlDataset to cache first and get num_tokens from cache
    tok_fn = PretrainTokenizeFunctionConfig().build(tokenizer)
    num_tokens = np.array([tok_fn(r)["num_tokens"] for r in records], dtype=np.int64)

    # Step 2: pack config
    pack_dir: Path | None = None
    num_packs_ref: int | None = None
    pack_level: Literal["preset", "hard"]
    longest: np.ndarray | None = None

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
        longest = cfg["longest"]

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

    # Step 3: sampler config
    micro_batch_size = global_batch_size // dp_size
    order_path = tmp_path / "order.npy"
    if sampler_case == "preset_seq_sampler":
        assert num_packs_ref is not None
        get_sampler_config(order_path, mode="sequential", num_packs=num_packs_ref)

        sampler_type = "preset"
        group_by_length = False
    elif sampler_case == "preset_group_sampler":
        assert pack_dir is not None and num_packs_ref is not None
        get_sampler_config(
            order_path,
            mode="length_grouped",
            num_packs=num_packs_ref,
            longest=longest,
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
    batches_a = list(dl_a)
    batches_b = list(dl_b)
    _cnt = assert_dataloader_batches_seq_and_labels_equal(batches_a, batches_b)

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
    assert_dataloader_batches_seq_and_labels_equal(batches_pre, batches_hard)


class TestPresetPackSamplerMatchesHardPackGroupSamplerDist(DistributedTestBase):
    """分布式下 Preset 打包 + PresetSampler 与 HardPack + LengthGroupedSampler 各 rank 局部 batch 一致。"""

    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        os.environ["LOCAL_WORLD_SIZE"] = str(self.world_size)
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        return ret

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_preset_pack_sampler_matches_buildin_hard_pack_group_sampler_dist(self):
        self.create_pg(DEVICE)
        rank = dist.get_rank()
        ws = dist.get_world_size()

        tokenizer_path = os.environ["QWEN3_MOE_PATH"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if rank == 0:
            tmp_root = tempfile.mkdtemp(prefix="xtuner_preset_dist_")
        else:
            tmp_root = None
        tmp_list: list[str | None] = [tmp_root]
        dist.broadcast_object_list(tmp_list, src=0)
        tmp_path = Path(tmp_list[0])  # type: ignore[arg-type]

        seed = 1
        global_batch_size = ws
        common = dict(
            sample_num=36,
            pack_max_length=32,
            global_batch_size=global_batch_size,
            dp_size=ws,
            seed_ref=seed,
            num_workers=0,
        )

        if rank == 0:
            pre = create_dataloader_config(tmp_path / "preset_run", tokenizer, "preset_pack", "preset_group_sampler", **common)
            hard = create_dataloader_config(tmp_path / "hard_run", tokenizer, "hard_pack", "group_sampler", **common)
            bundle_list: list = [pre, hard]
        else:
            bundle_list = [None, None]
        dist.broadcast_object_list(bundle_list, src=0)
        pre, hard = bundle_list[0], bundle_list[1]
        assert isinstance(pre, CreateDataloaderConfigResult)
        assert isinstance(hard, CreateDataloaderConfigResult)
        num_packs_ref = pre.num_packs_ref

        dist.barrier()
        dp_mesh = init_device_mesh(DEVICE, (ws,))

        dl_pre = pre.dataloader_cfg.build(
            tokenizer,
            dp_mesh=dp_mesh,
            global_batch_size=pre.global_batch_size,
            micro_batch_size=pre.micro_batch_size,
            seed=seed,
            shuffle=True,
        )
        dl_hard = hard.dataloader_cfg.build(
            tokenizer,
            dp_mesh=dp_mesh,
            global_batch_size=hard.global_batch_size,
            micro_batch_size=hard.micro_batch_size,
            seed=seed,
            shuffle=True,
        )

        batches_pre = list(dl_pre)
        batches_hard = list(dl_hard)
        self.assertEqual(len(batches_pre), len(batches_hard))

        n_pre = len(batches_pre)
        n_list = [n_pre]
        dist.broadcast_object_list(n_list, src=0)
        self.assertEqual(n_pre, n_list[0])

        _cnt = assert_dataloader_batches_seq_and_labels_equal(batches_pre, batches_hard)

        _pack_info = num_packs_ref if num_packs_ref is not None else "hard (runtime)"
        print(f"[RANK {rank}]--------------- packed samples: {_cnt}, original packed samples: {_pack_info}")

        dist.barrier()


class TestPresetPackGroupSamplerResumeDist(DistributedTestBase):
    """分布式 ``preset_pack`` + ``preset_group_sampler``：中段 checkpoint 后 resume，batch 与连续跑一致。"""

    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        os.environ["LOCAL_WORLD_SIZE"] = str(self.world_size)
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        return ret

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_preset_pack_group_sampler_resume_matches_continuous_dist(self):
        self.create_pg(DEVICE)
        rank = dist.get_rank()
        ws = dist.get_world_size()

        tokenizer_path = os.environ["QWEN3_MOE_PATH"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if rank == 0:
            tmp_root = tempfile.mkdtemp(prefix="xtuner_preset_resume_dist_")
        else:
            tmp_root = None
        tmp_list: list[str | None] = [tmp_root]
        dist.broadcast_object_list(tmp_list, src=0)
        tmp_path = Path(tmp_list[0])  # type: ignore[arg-type]

        # 1. Build dataloader
        seed = 1
        global_batch_size = ws
        half_step = 5
        common = dict(
            sample_num=200,
            pack_max_length=32,
            global_batch_size=global_batch_size,
            dp_size=ws,
            seed_ref=seed,
            num_workers=2,
        )

        if rank == 0:
            pre = create_dataloader_config(
                tmp_path / "preset_run", tokenizer, "preset_pack", "preset_group_sampler", **common
            )
            bundle_list: list = [pre]
        else:
            bundle_list = [None]
        dist.broadcast_object_list(bundle_list, src=0)
        pre = bundle_list[0]
        assert isinstance(pre, CreateDataloaderConfigResult)

        dist.barrier()
        dp_mesh = init_device_mesh(DEVICE, (ws,))

        def _build():
            return pre.dataloader_cfg.build(
                tokenizer,
                dp_mesh=dp_mesh,
                global_batch_size=pre.global_batch_size,
                micro_batch_size=pre.micro_batch_size,
                seed=seed,
                shuffle=True,
            )

        dl = _build()

        # 2. Consume data at [0, half_step)
        data_iter = iter(dl)
        consumed_samples = 0
        data_list = []
        for _ in range(half_step):
            batch = next(data_iter)
            data_list.append(batch)
            consumed_samples += len(batch)

        consumed_samples_list = [None for _ in range(ws)]
        dist.all_gather_object(consumed_samples_list, consumed_samples)
        global_consumed_samples = sum(int(x) for x in consumed_samples_list if x is not None)

        # 3. Get ckpt state
        # dataloader_state = get_dataloader_state(dl, global_consumed_samples)
        dataloader_state = dl.get_state_dict(global_consumed_samples)

        # 4. Continue to consume data at [half_step, 2*half_step)
        expected_batches = []
        for _ in range(half_step):
            expected_batches.append(next(data_iter))

        first_flat = list(chain(*data_list))
        expected_flat = list(chain(*expected_batches))

        all_data_list = [None for _ in range(ws)]
        dist.all_gather_object(all_data_list, first_flat)
        all_expected = [None for _ in range(ws)]
        dist.all_gather_object(all_expected, expected_flat)
        all_data_gathered = list(chain(*zip(*all_data_list)))
        all_expected_gathered = list(chain(*zip(*all_expected)))

        # 5. save ckpt
        ckpt_path = tmp_path / "preset_resume.ckpt"
        if rank == 0:
            with ckpt_path.open("wb") as f:
                pickle.dump(
                    {
                        "dataloader_state": dataloader_state,
                        "data_list": all_data_gathered,
                        "expected_data": all_expected_gathered,
                        "consumed_samples": consumed_samples,
                    },
                    f,
                )

        dist.barrier()

        # 6. Resume from ckpt
        dl2 = _build()
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        # load_dataloader_state(dl2, ckpt["dataloader_state"])
        dl2.load_state_dict(ckpt["dataloader_state"])

        resume_iter = iter(dl2)
        # 7. Continue to consume data at [half_step, 2*half_step)
        resume_batches = [next(resume_iter) for _ in range(half_step)]

        # 8. Verify the resume result is consistent in each rank
        assert_dataloader_batches_seq_and_labels_equal(expected_batches, resume_batches)

        # 9. Gather the resume result in all ranks and verify the result is consistent
        resume_flat = list(chain(*resume_batches))
        all_resume = [None for _ in range(ws)]
        dist.all_gather_object(all_resume, resume_flat)
        all_resume_gathered = list(chain(*zip(*all_resume)))
        if rank == 0:
            assert len(ckpt["expected_data"]) == len(all_resume_gathered)
            for s_exp, s_res in zip(ckpt["expected_data"], all_resume_gathered):
                assert_dataloader_batches_seq_and_labels_equal([[s_exp]], [[s_res]])

        dist.barrier()
