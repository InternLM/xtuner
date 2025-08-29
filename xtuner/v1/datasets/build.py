import copy
import os
from functools import partial
from typing import Iterable

import torch
from mmengine.dist import get_rank
from mmengine.fileio import list_dir_or_file
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, SequentialSampler

from xtuner.v1.config import DatasetConfigList
from xtuner.v1.utils import get_logger

from ..config.data import DataloaderConfig, DatasetConfig
from ..datasets.collator import ColateItem
from .packing import ExpandSoftPackDataset, SoftPackDataset
from .sampler import LengthGroupedSampler, ParallelSampler


logger = get_logger()


# TODO: (huanghaian) Fix the return type hint static check
# TODO: (huanghaian) Moving arguments to dataset config
def build_datasets(
    dataset_config: DatasetConfigList,
    tokenizer,
    model_cfg: dict | None = None,
):
    datasets = []
    assert len(dataset_config) > 0
    for config in dataset_config:
        _dataset_config = config["dataset"]
        assert isinstance(_dataset_config, DatasetConfig)
        _tokenize_fn_name = config["tokenize_fn"]
        anno_path = _dataset_config.anno_path
        if os.path.isfile(anno_path):
            all_anno_path = [anno_path]
        else:
            all_anno_path = [
                os.path.join(anno_path, f)
                for f in list_dir_or_file(anno_path, suffix=".jsonl", list_dir=False, recursive=True)
            ]
        all_anno_path.sort()
        for anno_path in all_anno_path:
            _dataset_config = copy.deepcopy(_dataset_config)
            _dataset_config.anno_path = anno_path
            anno_name = os.path.basename(anno_path)  # for debug
            _tokenize_fn = _tokenize_fn_name.build(tokenizer, anno_name=anno_name)
            _dataset = _dataset_config.build(_tokenize_fn)
            if get_rank() == 0:
                logger.info(
                    f"[Dataset] (Original) {_dataset_config.name}/{os.path.basename(anno_path)}: {len(_dataset)} samples."
                )
            datasets.append(_dataset)

    return datasets


def build_dataloader(
    dataloader_config: DataloaderConfig,
    datasets: list,
    global_batch_size: int,
    micro_batch_size: int,
    seed: int,
    dp_mesh: DeviceMesh | None = None,
    shuffle: bool = True,
) -> Iterable[list[ColateItem]]:
    assert isinstance(datasets, list), "datasets must be a list of datasets."

    if dataloader_config.pack_level != "none" and get_rank == 0:
        num_tokens = sum(dset.num_tokens.sum() for dset in datasets)
        logger.debug(f"[Dataset] {num_tokens} tokens.")

    if dataloader_config.pack_level == "soft":
        logger.info("[Dataset] Start packing data of SoftPackDataset.")
        dataset = SoftPackDataset(
            datasets,
            target=dataloader_config.pack_max_length,
            blend=dataloader_config.global_pack,
            seed=seed,
        )
    elif dataloader_config.pack_level == "expand_soft":
        logger.info("[Dataset] Start packing data of ExpandSoftPackDataset.")
        dataset = ExpandSoftPackDataset(
            datasets,
            target=dataloader_config.pack_max_length,
            blend=dataloader_config.global_pack,
            pack_extra_buffer_size=dataloader_config.pack_extra_buffer_size,
            seed=seed,
        )
    elif dataloader_config.pack_level == "hard":
        raise NotImplementedError
    else:
        dataset = ConcatDataset(datasets)  # type: ignore

    if dataloader_config.pack_level != "none" and get_rank() == 0:
        ori_samples = sum([len(dset) for dset in datasets])
        packed_samples = len(dataset)
        logger.info(f"[Dataset] (Original) {ori_samples} samples.")
        logger.info(f"[Dataset] (Packed) {packed_samples} samples.")

    sampler: LengthGroupedSampler | ParallelSampler | RandomSampler | SequentialSampler
    if dp_mesh is not None:
        if dataloader_config.group_by_length:
            assert shuffle, "Currently only shuffling is supported for LengthGroupedSampler."
            sampler = LengthGroupedSampler(dataset, dp_mesh, global_batch_size, seed=seed)
        else:
            sampler = ParallelSampler(dataset, dp_mesh, global_batch_size, shuffle=shuffle)
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            # TODO: SequentialSampler 可能有点问题，训练莫名其妙卡住
            sampler = SequentialSampler(dataset)

    ctx = torch.multiprocessing.get_context("fork")
    # Using `fork` here since `torchrun` uses the spawn method by default.
    # The unpickling process of spawn method creates a new process which reimports all heavy dependencies like `torch`.
    # Additionally, `torch.compile` will also be re-executed during this process, as it's commonly
    # used as a decorator within the imported module.

    # For example, when deserializing a dataset like `SoftPackDataset`,
    # all of its dependency chain including `xtuner`, `xtuner.datasets`
    # will be imported during unpickling. This import process happens during deserialization,
    # not serialization, and is very slow and inefficient.
    # Using forkserver avoids these redundant imports and improves performance.
    collator = partial(
        dataloader_config.build_collator(),
        pack_max_length=dataloader_config.pack_max_length,
        padding_token_idx=dataloader_config.pad_token_id if dataloader_config.pad_token_id is not None else 0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=dataloader_config.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=sampler,
        collate_fn=collator,
        multiprocessing_context=ctx if dataloader_config.num_workers > 0 else None,
        persistent_workers=dataloader_config.num_workers > 0,
    )
    return dataloader
