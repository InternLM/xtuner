from pathlib import Path
import os
import pickle

import torch

from xtuner.v1.datasets import build_dataloader, build_datasets, get_dataloader_state, load_dataloader_state, FTDPTokenizeFnConfig, DatasetConfig, DataloaderConfig
from xtuner.v1.train.toy_tokenizer import UTF8ByteTokenizer
from torch.multiprocessing import spawn, get_context
from torch.distributed.device_mesh import init_device_mesh
import pytest

import jsonlines
import random
from itertools import repeat, chain




class RandomDataset:
    def __init__(self, size: int, **kwargs):
        self.size = size
        self.num_tokens = [random.randint(50, 4096) for _ in range(size)]

    def __len__(self):
        return self.size


    def __getitem__(self, idx: int):
        num_tokens = self.num_tokens[idx]
        return {"input_ids": [1] * num_tokens, "labels": [1] * num_tokens, "num_tokens": num_tokens}


def _is_batch_same(batch1, batch2):
    assert len(batch1) == len(batch2)
    for b1, b2 in zip(batch1, batch2):
        if isinstance(b1, dict):
            if "seq_ctx" in b1:
                if (b1["seq_ctx"].seq_lens_k.tolist() != b2["seq_ctx"].seq_lens_k.tolist()):
                    return False
            elif "input_ids" in b1:
                if b1["input_ids"] != b2["input_ids"]:
                    return False
            else:
                __import__('ipdb').set_trace()
                raise RuntimeError("XTuner test bug")
        elif isinstance(b1, list):
            for item1, item2 in zip(b1, b2):
                if (item1["input_ids"] != item2["input_ids"]):
                    return False
    return True


def _dulicate_test_data(dup_path: Path, dup_times: int = 5):
    dup_path.parent.mkdir(parents=True, exist_ok=True)
    openai_data = Path(__file__).parent.parent / "resource" / "openai_sft.jsonl"
    with jsonlines.open(openai_data, "r") as reader:
        samples = list(reader)

    samples = list(chain(*repeat(samples, times=dup_times)))
    with jsonlines.open(dup_path, "w") as writer:
        writer.write_all(samples)


def _create_fake_dataset(tmp_path: Path, dataset_num: int = 5, max_depth: int = 3, dup_times: int = 5):
    if max_depth <= 0:
        _dulicate_test_data(tmp_path / "data.jsonl", dup_times=dup_times)
        return

    for _ in range(dataset_num):
        _create_fake_dataset(tmp_path / str(max_depth), 1, max_depth - 1, dup_times)



def test_consistant():
    dataloader_config = DataloaderConfig(
        pack_chunk_size=10,
    )
    datasets = [RandomDataset(size=random.randint(5, 20)) for i in range(100)]

    # 1. test consistent with same seed
    dataloader1_1 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=16,
        micro_batch_size=8,
        seed=10,
    )
    dataloader1_2 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=16,
        micro_batch_size=8,
        seed=10,
    )

    for batch1, batch2 in zip(dataloader1_1, dataloader1_2):
        assert _is_batch_same(batch1, batch2)

    # 2. test inconsistent with different seed
    dataloader2_1 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=16,
        micro_batch_size=8,
        seed=11,
    )

    different = False
    for batch1, batch3 in zip(dataloader1_1, dataloader2_1):
        if not _is_batch_same(batch1, batch3):
            different = True
            break
    if not different:
        raise AssertionError("Dataloaders with different seeds should be different")


    # 3. test different pack worker return same result
    dataloader_config3_1 = DataloaderConfig(
        pack_chunk_size=10,
        pack_workers=4,
    )

    dataloader3_1 = build_dataloader(
        dataloader_config=dataloader_config3_1,
        datasets=datasets,
        global_batch_size=16,
        micro_batch_size=8,
        seed=10,
    )
    for batch1, batch4 in zip(dataloader1_1, dataloader3_1):
        for b1, b4 in zip(batch1, batch4):
            assert (b1["seq_ctx"].input_ids == b4["seq_ctx"].input_ids).all()

    # 4. test different num_workers return same result
    dataloader_config4_1 = DataloaderConfig(
        pack_chunk_size=10,
        num_workers=4,
    )
    dataloader4_1 = build_dataloader(
        dataloader_config=dataloader_config4_1,
        datasets=datasets,
        global_batch_size=16,
        micro_batch_size=8,
        seed=10,
    )
    for batch1, batch4 in zip(dataloader1_1, dataloader4_1):
        assert _is_batch_same(batch1, batch4)


@pytest.mark.parametrize(
    "pack_level,num_workers,group_by_length,pack_workers",
    [
        ("hard", 0, False, 1),
        ("hard", 0, True, 1),
        ("hard", 4, True, 1),
        ("hard", 4, True, 3),
        ("none", 0, False, 1),
        ("soft", 0, True, 1),
        ("soft", 4, True, 1),
        ("soft", 4, True, 3),
    ]
)
def test_dataloader_resume_single_process(tmp_path, pack_level, num_workers, group_by_length, pack_workers):
    RESUME_ITER = 10
    AFTER_RESUME_ITER = 10
    BATCH_SIZE = 8
    GLOBAL_BATCH_SIZE = BATCH_SIZE

    data_dir1 = tmp_path / "data1"
    _create_fake_dataset(data_dir1 / f"depth1", dataset_num=3, max_depth=1, dup_times=3)
    _create_fake_dataset(data_dir1 / f"depth2", dataset_num=3, max_depth=2, dup_times=2)
    _create_fake_dataset(data_dir1 / f"depth3", dataset_num=3, max_depth=3, dup_times=3)

    # 1. Test normal resume after consuming RESUME_ITER batches
    tokenizer = UTF8ByteTokenizer()

    dataset_configs = [
        {
            "dataset": DatasetConfig(anno_path=str(data_dir1)),
            "tokenize_fn": FTDPTokenizeFnConfig(max_length=1024)
        },
    ]

    dataloader_config = DataloaderConfig(
        pack_max_length=1024,
        pack_level=pack_level,
        num_workers=num_workers,
        group_by_length=group_by_length,
        pack_workers=pack_workers,
    )

    datasets = build_datasets(
        dataset_config=dataset_configs,
        tokenizer=tokenizer,
    )
    dataloader1 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=BATCH_SIZE,
        seed=10,
    )

    print(f"dataloader1 length: {len(dataloader1)}")
    assert len(dataloader1) > 10

    dataloader_iter = iter(dataloader1)
    consumed_sample = 0
    for _ in range(RESUME_ITER):
        batch = next(dataloader_iter)
        consumed_sample += len(batch)

    dataloader_state = get_dataloader_state(dataloader1, consumed_sample)
    expected_data = []
    for _ in range(AFTER_RESUME_ITER):
        batch = next(dataloader_iter)
        consumed_sample += len(batch)
        expected_data.append(batch)

    new_dataloader1 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=BATCH_SIZE,
        seed=10,
    )
    load_dataloader_state(new_dataloader1, dataloader_state)
    new_dataloader_iter = iter(new_dataloader1)

    resume_data = []
    for _ in range(AFTER_RESUME_ITER):
        resume_data.append(next(new_dataloader_iter))

    for b1, b2 in zip(expected_data, resume_data):
        assert _is_batch_same(b1, b2)

    # 2. Test resume after consuming multiple epochs
    while True:
        try:
            batch = next(dataloader_iter)
            consumed_sample += len(batch)
        except StopIteration:
            break


    dataloader_iter = iter(dataloader1)

    for batch in range(RESUME_ITER):
        batch = next(dataloader_iter)
        consumed_sample += len(batch)

    dataloader_state = get_dataloader_state(dataloader1, consumed_sample)

    expected_data = []
    for _ in range(AFTER_RESUME_ITER):
        expected_data.append(next(dataloader_iter))

    new_dataloader2 = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=GLOBAL_BATCH_SIZE,
        micro_batch_size=BATCH_SIZE,
        seed=10,
    )
    load_dataloader_state(new_dataloader2, dataloader_state)
    new_dataloader_iter2 = iter(new_dataloader2)

    resume_data = []
    for _ in range(AFTER_RESUME_ITER):
        resume_data.append(next(new_dataloader_iter2))

    for b1, b2 in zip(expected_data, resume_data):
        assert _is_batch_same(b1, b2)


def _test_resume_spmd(
    rank: int,
    world_size: int,
    dataloader_config: DataloaderConfig,
    dataset_configs: list[dict],
    global_batch_size: int,
    micro_batch_size: int,
    step:int,
    seed: int,
    save_path: Path,
    dataloader_state: dict | None = None,
    consumed_samples: int = 0,
):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"


    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    data_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,)
    )
    tokenizer = UTF8ByteTokenizer()

    datasets = build_datasets(
        dataset_config=dataset_configs,
        tokenizer=tokenizer,
    )
    dataloader = build_dataloader(
        dataloader_config=dataloader_config,
        datasets=datasets,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        seed=seed,
        dp_mesh=data_mesh,
    )

    if dataloader_state is not None:
        load_dataloader_state(dataloader, dataloader_state)

    data_iter = iter(dataloader)
    data_list = []
    for _ in range(step):
        batch = next(data_iter)
        data_list.append(batch)
        consumed_samples += len(batch)

    consumed_samples_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(consumed_samples_list, consumed_samples)
    global_consumed_samples = sum(consumed_samples_list)

    expected_data = []

    for _ in range(step):
        batch = next(data_iter)
        expected_data.append(batch)

    dataloader_state = get_dataloader_state(dataloader, global_consumed_samples)

    all_data_list = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_data_list, list(chain(*data_list)))

    all_expected_data = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_expected_data, list(chain(*expected_data)))

    all_data_list = list(chain(*zip(*all_data_list)))
    all_expected_data = list(chain(*zip(*all_expected_data)))
    # [
    #    [[0 2 4 6 8], [10, 12, 14, 16]],
    #    [[1 3 5 7 9], [11, 13, 15, 17]]
    # ]
    #
    # ch
    # 第一维，world size
    # 第二维， batch
    # 第三维， sample in batch
    # 0 4 8 12
    # 1 5 9 13
    # 2 6 10 14
    # 3 7 11 15


    if rank == 0:
        with save_path.open("wb") as f:
            f.write(
                pickle.dumps(
                    {
                        "dataloader_state": dataloader_state,
                        "data_list": all_data_list,
                        "expected_data": all_expected_data,
                        "consumed_samples": consumed_samples
                    }
                )
            )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

@pytest.mark.parametrize(
    "pack_level,num_workers,group_by_length",
    [
        ("hard", 0, True),
        ("hard", 0, False),
        ("hard", 4, True),
        ("none", 0, False),
        ("soft", 0, True),
        ("soft", 4, True),
        ("soft", 4, True),
    ]
)
def test_dataloader_resume_multi_process(tmp_path, pack_level, num_workers, group_by_length):
    TOTAL_STEP = 10
    BATCH_SIZE = 8

    data_dir1 = tmp_path / "data1"
    _create_fake_dataset(data_dir1 / f"depth1", dataset_num=3, max_depth=1, dup_times=10)
    _create_fake_dataset(data_dir1 / f"depth2", dataset_num=3, max_depth=2, dup_times=8)
    _create_fake_dataset(data_dir1 / f"depth3", dataset_num=3, max_depth=3, dup_times=9)

    # 1. Test resuming with the same world size
    dataloader_config = DataloaderConfig(
        pack_max_length=1024,
        pack_level=pack_level,
        num_workers=num_workers,
        group_by_length=group_by_length,
        collator="fake_collator"
    )
    dataset_configs = [
        {
            "dataset": DatasetConfig(anno_path=str(data_dir1)),
            "tokenize_fn": FTDPTokenizeFnConfig(max_length=1024)
        },
    ]

    ctx = get_context("spawn")
    world_size = 2
    save_path1 = tmp_path / "dataloader_state.pkl"
    spawn(
        _test_resume_spmd,
        args=(
            world_size,
            dataloader_config,
            dataset_configs,
            16,
            BATCH_SIZE,
            TOTAL_STEP,
            10,
            save_path1,
            None,
            0,
        ),
        nprocs=2,
        join=True,
    )
    with save_path1.open("rb") as f:
        result1 = pickle.load(f)

    expected_data = result1["expected_data"]

    # 2. tet Rsume with same world size
    save_path2 = tmp_path / "dataloader_state2.pkl"
    spawn(
        _test_resume_spmd,
        args=(
            world_size,
            dataloader_config,
            dataset_configs,
            16,
            BATCH_SIZE,
            TOTAL_STEP,
            10,
            save_path2,
            result1["dataloader_state"],
            result1["consumed_samples"],
        ),
        nprocs=world_size,
        join=True,
    )
    with save_path2.open("rb") as f:
        result2 = pickle.load(f)

    resume_data = result2["data_list"]

    for b1, b2 in zip(expected_data, resume_data):
        assert _is_batch_same(b1, b2)

    world_size = 4
    save_path3 = tmp_path / "dataloader_state3.pkl"
    spawn(
        _test_resume_spmd,
        args=(
            world_size,
            dataloader_config,
            dataset_configs,
            16,
            BATCH_SIZE,
            TOTAL_STEP,
            10,
            save_path3,
            result1["dataloader_state"],
            result1["consumed_samples"],
        ),
        nprocs=world_size,
        join=True,
    )
    with save_path3.open("rb") as f:
        result3 = pickle.load(f)

    resume_data = result3["data_list"]

    for b1, b2 in zip(expected_data, resume_data):
        assert _is_batch_same(b1, b2)


if __name__ == "__main__":
    test_dataloader_resume_single_process(tmp_path=Path("/tmp/test_dataloader_resume_single_process"), pack_level="hard", num_workers=0, group_by_length=False, pack_workers=1)
