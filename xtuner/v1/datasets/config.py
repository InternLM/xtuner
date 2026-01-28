import copy
import os
import pydoc
from functools import partial
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional, Protocol, Union, runtime_checkable

import torch
from cyclopts import Parameter
from mmengine.dist import get_rank
from mmengine.fileio import list_dir_or_file
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from typing_extensions import TypedDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.utils import get_logger, profile_time

from ..datasets.collator import ColateItem
from .collator import (
    fake_collator,
    intern_s1_vl_sft_collator,
    qwen3_vl_sft_collator,
    sft_llm_collator,
)
from .dataloader import BaseDataloader, Dataloader
from .jsonl import JsonlDataset
from .packing import ExpandSoftPackDataset, HardPackDataset, MLLMPretrainHybridPackDataset, _LegacySoftPackDataset
from .sampler import LengthGroupedSampler, ParallelSampler
from .utils import CachableTokenizeFunction, tokenizer_xxhash
from .vlm_jsonl import VLMJsonlDataset


logger = get_logger()


# TODO: Enhance the configurable fields of dataset config
class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="forbid")
    anno_path: Annotated[str | Path, Parameter(group="dataset")]
    cache_dir: str | Path | None = None
    cache_tag: str | None = None
    name: Annotated[str, Parameter(group="dataset")] = "default"
    class_name: Annotated[str, Parameter(group="dataset")] = "JsonlDataset"
    sample_ratio: Annotated[float, Parameter(group="dataset")] = 1.0
    enable_sequential_sampler: Annotated[bool, Parameter(group="dataset")] = False
    media_root: Annotated[str | None, Parameter(group="dataset")] = ""

    def build(
        self,
        tokenize_fn: Optional["CachableTokenizeFunction"] = None,
    ) -> "JsonlDataset":
        if self.class_name == "JsonlDataset":
            return JsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=self.anno_path,
                sample_ratio=self.sample_ratio,
                enable_sequential_sampler=self.enable_sequential_sampler,
                name=self.name,
                cache_dir=self.cache_dir,
                cache_tag=self.cache_tag,
            )
        elif self.class_name == "VLMJsonlDataset":
            return VLMJsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=self.anno_path,
                sample_ratio=self.sample_ratio,
                enable_sequential_sampler=self.enable_sequential_sampler,
                name=self.name,
                media_root=self.media_root,
                cache_dir=self.cache_dir,
                cache_tag=self.cache_tag,
            )
        else:
            raise ValueError(f"Unsupported class_name: {self.class_name}")


@runtime_checkable
class BaseTokenizeFnConfig(Protocol):
    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str | None = None, **kwargs
    ) -> "CachableTokenizeFunction":
        """Build the tokenize function."""
        raise NotImplementedError


class DatasetCombine(TypedDict):
    dataset: DatasetConfig
    tokenize_fn: BaseTokenizeFnConfig


DatasetConfigList = list[DatasetCombine]
DatasetConfigListAdatper = TypeAdapter(DatasetConfigList, config=ConfigDict(arbitrary_types_allowed=True))


# TODO: (huanghaian) Fix the return type hint static check
# TODO: (huanghaian) Moving arguments to dataset config
def build_datasets(
    dataset_config: DatasetConfigList, tokenizer: PreTrainedTokenizer, tokenizer_hash: str | None = None
) -> list[JsonlDataset]:
    datasets: list[JsonlDataset] = []
    assert len(dataset_config) > 0

    if tokenizer_hash is None:
        tokenizer_hash = tokenizer_xxhash(tokenizer)[:16]

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
            _tokenize_fn = _tokenize_fn_name.build(tokenizer, tokenizer_hash=tokenizer_hash, anno_name=anno_name)
            _dataset = _dataset_config.build(_tokenize_fn)
            if get_rank() == 0:
                logger.info(
                    f"[Dataset] (Original) {_dataset_config.name}/{os.path.basename(anno_path)}: {len(_dataset)} samples."
                )
            datasets.append(_dataset)

    return datasets


# TODO: Removed in version 1.1.0
def build_dataloader(
    dataloader_config: "DataloaderConfig",
    datasets: list[JsonlDataset],
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

    dataset: (
        ExpandSoftPackDataset
        | _LegacySoftPackDataset
        | ConcatDataset
        | HardPackDataset
        | MLLMPretrainHybridPackDataset
    )
    if dataloader_config.pack_level == "soft":
        logger.info("[Dataset] Start packing data of ExpandSoftPackDataset.")
        dataset = ExpandSoftPackDataset(
            datasets,
            pack_max_length=dataloader_config.pack_max_length,
            pack_chunk_size=dataloader_config.pack_chunk_size,
            pack_workers=dataloader_config.pack_workers,
            global_pack=dataloader_config.global_pack,
            pack_extra_buffer_size=dataloader_config.pack_extra_buffer_size,
            seed=seed,
        )
    elif dataloader_config.pack_level == "mllm_hybrid":
        logger.info("[Dataset] Start packing data of MLLMPretrainHybridPackDataset.")
        dataset = MLLMPretrainHybridPackDataset(
            datasets,
            pack_max_length=dataloader_config.pack_max_length,
            pack_chunk_size=dataloader_config.pack_chunk_size,
            pack_workers=dataloader_config.pack_workers,
            global_pack=dataloader_config.global_pack,
            pack_extra_buffer_size=dataloader_config.pack_extra_buffer_size,
            seed=seed,
        )
    elif dataloader_config.pack_level == "hard":
        logger.info("[Dataset] Start packing data of HardPackDataset.")
        dataset = HardPackDataset(
            datasets,
            pack_max_length=dataloader_config.pack_max_length,
            global_pack=dataloader_config.global_pack,
            seed=seed,
        )
    elif dataloader_config.pack_level == "none":
        dataset = ConcatDataset(datasets)  # type: ignore
    elif dataloader_config.pack_level == "__legacy":
        logger.info("[Dataset] Start packing data of _LegacySoftPackDataset.")
        dataset = _LegacySoftPackDataset(
            datasets,
            pack_max_length=dataloader_config.pack_max_length,
            global_pack=dataloader_config.global_pack,
            seed=seed,
        )
    else:
        raise NotImplementedError(f"Unsupported pack level: {dataloader_config.pack_level}")

    if dataloader_config.pack_level in ("mllm_hybrid", "soft", "__legacy") and get_rank() == 0:
        ori_samples = sum([len(dset) for dset in datasets])
        packed_samples = len(dataset)
        logger.info(f"[Dataset] (Original) {ori_samples} samples.")
        logger.info(f"[Dataset] (Packed) {packed_samples} samples.")

    sampler: LengthGroupedSampler | ParallelSampler | RandomSampler | SequentialSampler
    if dataloader_config.group_by_length:
        assert shuffle, "Currently only shuffling is supported for LengthGroupedSampler."
        assert isinstance(
            dataset, (ExpandSoftPackDataset, _LegacySoftPackDataset, HardPackDataset, MLLMPretrainHybridPackDataset)
        ), (
            "Internal Error, LengthGroupedSampler requires ExpandSoftPackDataset or _LegacySoftPackDataset, "
            f"but got {type(dataset)}"
        )
        sampler = LengthGroupedSampler(
            dataset=dataset, dp_mesh=dp_mesh, global_batch_size=global_batch_size, seed=seed
        )
    else:
        sampler = ParallelSampler(
            dataset=dataset, dp_mesh=dp_mesh, global_batch_size=global_batch_size, shuffle=shuffle, seed=seed
        )

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
    dataloader = TorchDataLoader(
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


class BaseDataloaderConfig(BaseModel):
    def build(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dp_mesh: DeviceMesh,
        global_batch_size: int,
        micro_batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> BaseDataloader:
        raise NotImplementedError


class DataloaderConfig(BaseDataloaderConfig):
    model_config = ConfigDict(title="Dataloader config for xtuner", extra="forbid", arbitrary_types_allowed=True)

    dataset_config_list: DatasetConfigList | None = None

    collator: Annotated[
        Literal["sft_llm_collator", "intern_s1_vl_sft_collator", "qwen3_vl_sft_collator", "fake_collator"] | str,
        Parameter(help="collator func name"),
    ] = "sft_llm_collator"
    pack_to_max_length: Annotated[bool, Parameter(help="whether to pack to max length")] = True
    pack_level: Annotated[
        Literal["soft", "none", "__legacy", "hard", "mllm_hybrid"], Parameter(help="__legacy is only for debug")
    ] = "soft"
    pack_max_length: Annotated[int, Parameter(help="pack max length")] = 32768
    pack_chunk_size: Annotated[int, Parameter(help="pack chunk size")] = 10000
    pack_workers: Annotated[int, Parameter(help="pack workers")] = 8
    global_pack: Annotated[bool, Parameter(help="enable or disable global pack mode")] = True
    group_by_length: Annotated[bool, Parameter(help="enable or disable group by length mode")] = True
    pack_extra_buffer_size: Annotated[
        int, Parameter(help="pack extra buffer size when pack_level is expand_soft model")
    ] = 100
    num_workers: Annotated[int, Parameter(help="dataloader num workers")] = 0
    pad_token_id: Annotated[int | None, Parameter(help="padding token id")] = None
    tokenizer_hash: Annotated[str | None, Parameter(help="tokenizer hash")] = None

    def build_collator(self):
        if self.collator == "sft_llm_collator":
            return sft_llm_collator
        elif self.collator == "intern_s1_vl_sft_collator":
            return intern_s1_vl_sft_collator
        elif self.collator == "qwen3_vl_sft_collator":
            return qwen3_vl_sft_collator
        elif self.collator == "fake_collator":
            return fake_collator  # for RL
        else:
            collator = pydoc.locate(self.collator)
            if collator is None:
                raise ImportError(f"Cannot locate collator: {self.collator}")
            return collator

    @model_validator(mode="before")
    @classmethod
    def _infer_group_by_length(cls, data) -> None:
        if "pack_level" in data and "group_by_length" not in data:
            if data["pack_level"] == "none":
                data["group_by_length"] = False
            else:
                data["group_by_length"] = True

        if "group_by_length" in data and "pack_level" in data:
            if data["pack_level"] == "none" and data["group_by_length"] is True:
                raise ValueError("group_by_length must be False when pack_level is none.")
        return data

    def build(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dp_mesh: DeviceMesh | None,
        global_batch_size: int,
        micro_batch_size: int,
        seed: int,
        shuffle: bool = True,
        total_step: int | None = None,
    ) -> Dataloader:
        if self.dataset_config_list is None:
            raise ValueError("dataset_config_list is required.")

        with profile_time("[Build Datasets]"):
            datasets = build_datasets(self.dataset_config_list, tokenizer, tokenizer_hash=self.tokenizer_hash)

        assert isinstance(datasets, list), "datasets must be a list of datasets."

        if self.pack_level != "none" and get_rank == 0:
            num_tokens = sum(dset.num_tokens.sum() for dset in datasets)
            logger.debug(f"[Dataset] {num_tokens} tokens.")

        with profile_time("[Pack Datasets]"):
            dataset: (
                ExpandSoftPackDataset
                | _LegacySoftPackDataset
                | ConcatDataset
                | HardPackDataset
                | MLLMPretrainHybridPackDataset
            )
            if self.pack_level == "soft":
                logger.info("[Dataset] Start packing data of ExpandSoftPackDataset.")
                dataset = ExpandSoftPackDataset(
                    datasets,
                    pack_max_length=self.pack_max_length,
                    pack_chunk_size=self.pack_chunk_size,
                    pack_workers=self.pack_workers,
                    global_pack=self.global_pack,
                    pack_extra_buffer_size=self.pack_extra_buffer_size,
                    seed=seed,
                )
            elif self.pack_level == "mllm_hybrid":
                logger.info("[Dataset] Start packing data of MLLMPretrainHybridPackDataset.")
                dataset = MLLMPretrainHybridPackDataset(
                    datasets,
                    pack_max_length=self.pack_max_length,
                    pack_chunk_size=self.pack_chunk_size,
                    pack_workers=self.pack_workers,
                    global_pack=self.global_pack,
                    pack_extra_buffer_size=self.pack_extra_buffer_size,
                    seed=seed,
                )
            elif self.pack_level == "hard":
                logger.info("[Dataset] Start packing data of HardPackDataset.")
                dataset = HardPackDataset(
                    datasets,
                    pack_max_length=self.pack_max_length,
                    global_pack=self.global_pack,
                    seed=seed,
                )
            elif self.pack_level == "none":
                dataset = ConcatDataset(datasets)  # type: ignore
            elif self.pack_level == "__legacy":
                logger.info("[Dataset] Start packing data of _LegacySoftPackDataset.")
                dataset = _LegacySoftPackDataset(
                    datasets,
                    pack_max_length=self.pack_max_length,
                    global_pack=self.global_pack,
                    seed=seed,
                )
            else:
                raise NotImplementedError(f"Unsupported pack level: {self.pack_level}")

        if self.pack_level in ("mllm_hybrid", "soft", "__legacy") and get_rank() == 0:
            ori_samples = sum([len(dset) for dset in datasets])
            packed_samples = len(dataset)
            logger.info(f"[Dataset] (Original) {ori_samples} samples.")
            logger.info(f"[Dataset] (Packed) {packed_samples} samples.")

        sampler: LengthGroupedSampler | ParallelSampler | RandomSampler | SequentialSampler
        if self.group_by_length:
            assert shuffle, "Currently only shuffling is supported for LengthGroupedSampler."
            assert isinstance(dataset, (ExpandSoftPackDataset, _LegacySoftPackDataset, HardPackDataset)), (
                "Internal Error, LengthGroupedSampler requires ExpandSoftPackDataset or _LegacySoftPackDataset, "
                f"but got {type(dataset)}"
            )
            sampler = LengthGroupedSampler(
                dataset=dataset, dp_mesh=dp_mesh, global_batch_size=global_batch_size, seed=seed
            )
        else:
            sampler = ParallelSampler(
                dataset=dataset, dp_mesh=dp_mesh, global_batch_size=global_batch_size, shuffle=shuffle, seed=seed
            )

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
            self.build_collator(),
            pack_max_length=self.pack_max_length,
            pack_to_max_length=self.pack_to_max_length,
            padding_token_idx=self.pad_token_id if self.pad_token_id is not None else 0,
        )
        dataloader = Dataloader(
            dataset,
            batch_size=micro_batch_size,
            num_workers=self.num_workers,
            # Ensure to round up or drop last based on the `global_batch_size`,
            # if you want to replace a custom sampler.
            sampler=sampler,
            collate_fn=collator,
            multiprocessing_context=ctx if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )
        return dataloader
