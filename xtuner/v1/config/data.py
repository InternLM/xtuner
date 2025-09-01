from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Protocol, runtime_checkable

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import TypedDict


if TYPE_CHECKING:
    from xtuner.v1.datasets import (
        CachableTokenizeFunction,
        JsonlDataset,
    )


# TODO: Enhance the configurable fields of dataset config
class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    anno_path: Annotated[str | Path, Parameter(group="dataset")]
    cache_dir: str | Path | None = None
    cache_tag: str | None = None
    name: Annotated[str, Parameter(group="dataset")] = "default"
    class_name: Annotated[str, Parameter(group="dataset")] = "JsonlDataset"
    sample_ratio: Annotated[float, Parameter(group="dataset")] = 1.0
    media_root: Annotated[str, Parameter(group="dataset")] = ""

    def build(
        self,
        tokenize_fn: Optional["CachableTokenizeFunction"] = None,
    ) -> "JsonlDataset":
        if self.class_name == "JsonlDataset":
            from xtuner.v1.datasets import JsonlDataset

            return JsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=self.anno_path,
                sample_ratio=self.sample_ratio,
                name=self.name,
                cache_dir=self.cache_dir,
                cache_tag=self.cache_tag,
            )
        elif self.class_name == "VLMJsonlDataset":
            from xtuner.v1.datasets import VLMJsonlDataset

            return VLMJsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=self.anno_path,
                sample_ratio=self.sample_ratio,
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


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(title="Base dataloader config for xtuner", extra="allow")
    collator: Literal["sft_llm_collator", "sft_vllm_collator", "fake_collator"] = "sft_llm_collator"
    pack_level: Annotated[str, Parameter()] = "soft"  # TODO: (huanghaian) Only provide 1 pad level
    pack_max_length: Annotated[int, Parameter()] = 32768
    global_pack: Annotated[bool, Parameter()] = True
    group_by_length: Annotated[bool, Parameter()] = True
    pack_extra_buffer_size: Annotated[int, Parameter()] = 100
    num_workers: Annotated[int, Parameter()] = 0
    pad_token_id: Annotated[int | None, Parameter()] = None

    def build_collator(self):
        from xtuner.v1.datasets import fake_collator, sft_llm_collator, sft_vllm_collator

        if self.collator == "sft_llm_collator":
            return sft_llm_collator
        elif self.collator == "sft_vllm_collator":
            return sft_vllm_collator
        elif self.collator == "fake_collator":
            return fake_collator  # for RL
        else:
            raise ValueError(f"Unsupported collator: {self.collator}")


class DatasetCombine(TypedDict):
    dataset: DatasetConfig
    tokenize_fn: BaseTokenizeFnConfig


DatasetConfigList = list[DatasetCombine]
DatasetConfigListAdatper = TypeAdapter(DatasetConfigList, config=ConfigDict(arbitrary_types_allowed=True))
