import copy
import os

import torch
from mmengine.dist import get_rank
from mmengine.fileio import list_dir_or_file
from torch.utils.data import ConcatDataset, DataLoader

from xtuner.v1.utils import get_logger

from ..config.data_config import DataloaderConfig, DatasetConfig
from .packing import ExpandSoftPackDataset, SoftPackDataset
from .sampler import LengthGroupedSampler, ParallelSampler


logger = get_logger()


def build_datasets(dataset_config: DatasetConfig, tokenizer):
    meta_datas = dataset_config.meta_datas

    datasets = []
    for name, meta_data in meta_datas.items():
        annotation = meta_data["annotation"]
        if os.path.isfile(annotation):
            all_annotation = [annotation]
        else:
            all_annotation = list(list_dir_or_file(annotation, list_dir=False, suffix=".jsonl", recursive=True))
        for annotation_path in all_annotation:
            meta_data_ = copy.deepcopy(meta_data)
            meta_data_["annotation"] = annotation_path

            tokenizer_fn_args = dataset_config.tokenizer_fn_args
            if tokenizer_fn_args is None:
                tokenizer_fn_args = {}

            tokenize_fn = dataset_config.tokenizer_fn(meta_data_, tokenizer, **tokenizer_fn_args)

            dataset_args = dataset_config.dataset_args
            if dataset_args is None:
                dataset_args = {}

            _dataset = dataset_config.dataset_class(
                annotation_path,
                sample_ratio=meta_data_.get("sample_ratio", 1.0),
                tokenize_fn=tokenize_fn,
                **dataset_args,
            )

            if get_rank() == 0:
                logger.info(
                    f"[Dataset] (Original) {name}/{os.path.basename(annotation_path)}: {len(_dataset)} samples."
                )
            datasets.append(_dataset)
    return datasets


def build_dataloader(dataloader_config: DataloaderConfig, datasets, dp_mesh):
    assert isinstance(datasets, list), "datasets must be a list of datasets."

    if dataloader_config.pack_level != "none" and get_rank == 0:
        num_tokens = sum(dset.num_tokens.sum() for dset in datasets)
        logger.debug(f"[Dataset] {num_tokens} tokens.")

    if dataloader_config.pack_level == "soft":
        logger.info("[Dataset] Start packing data of SoftPackDataset.")
        dataset = SoftPackDataset(
            datasets, target=dataloader_config.pack_max_length, blend=dataloader_config.global_pack
        )
    elif dataloader_config.pack_level == "expand_soft":
        logger.info("[Dataset] Start packing data of ExpandSoftPackDataset.")
        dataset = ExpandSoftPackDataset(
            datasets,
            target=dataloader_config.pack_max_length,
            blend=dataloader_config.global_pack,
            pack_extra_buffer_size=dataloader_config.pack_extra_buffer_size,
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

    if dataloader_config.group_by_length:
        sampler = LengthGroupedSampler(dataset, dp_mesh, dataloader_config.global_batch_size)
    else:
        sampler = ParallelSampler(dataset, dp_mesh, dataloader_config.global_batch_size, shuffle=True)  # type: ignore

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
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.mirco_batch_size,
        num_workers=dataloader_config.num_workers,
        # Ensure to round up or drop last based on the `global_batch_size`,
        # if you want to replace a custom sampler.
        sampler=sampler,
        collate_fn=dataloader_config.collator_fn,
        multiprocessing_context=ctx if dataloader_config.num_workers > 0 else None,
        persistent_workers=dataloader_config.num_workers > 0,
    )
    return dataloader
