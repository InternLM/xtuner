from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, Protocol, TypedDict, runtime_checkable

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, TypeAdapter


if TYPE_CHECKING:
    from xtuner.v1.datasets import CachableTokenizeFunction, FtdpTokenizeFunction, JsonlDataset


class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    anno_path: Annotated[str | Path, Parameter(group="dataset")]
    class_name: Annotated[str, Parameter(group="dataset")] = "JsonlDataset"
    name: Annotated[str, Parameter(group="dataset")] = "default"
    sample_ratio: Annotated[float, Parameter(group="dataset")] = 1.0
    media_root: Annotated[str | None, Parameter(group="dataset")] = None

    def build(
        self,
        tokenize_fn: Optional["CachableTokenizeFunction"] = None,
        max_length: int | None = None,
        cache_dir: str | None = None,
        cache_tag: str | None = None,
    ) -> "JsonlDataset":
        if self.class_name == "JsonlDataset":
            from xtuner.v1.datasets import JsonlDataset

            return JsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=self.anno_path,
                sample_ratio=self.sample_ratio,
                name=self.name,
                max_length=max_length,
                cache_dir=cache_dir,
                cache_tag=cache_tag,
            )
        else:
            raise NotImplementedError


@runtime_checkable
class BaseTokenizeFnConfig(Protocol):
    def build(self, tokenizer, tokenizer_hash: str | None = None, **kwargs) -> "CachableTokenizeFunction":
        """Build the tokenize function."""
        raise NotImplementedError


# TODO: Maybe rename
class FTDPTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    chat_template: Annotated[str, Parameter(group="tokenize_fn")] = "internlm2"
    hash: Annotated[str | None, Parameter(group="tokenize_fn")] = None

    def build(self, tokenizer, tokenizer_hash: str | None = None, **kwargs) -> "FtdpTokenizeFunction":
        from xtuner.v1.datasets import FtdpTokenizeFunction

        return FtdpTokenizeFunction(
            tokenizer, chat_template=self.chat_template, hash=self.hash, tokenizer_hash=tokenizer_hash
        )


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(title="Base dataloader config for xtuner", extra="allow")
    pack_level: Annotated[str, Parameter()] = "soft"
    pack_max_length: Annotated[int, Parameter()] = 32768
    max_length: Annotated[int, Parameter()] = 4096
    global_pack: Annotated[bool, Parameter()] = True
    group_by_length: Annotated[bool, Parameter()] = True
    pack_extra_buffer_size: Annotated[int, Parameter()] = 100
    num_workers: Annotated[int, Parameter()] = 0
    padding_token_idx: Annotated[int, Parameter()] = 0
    cache_dir: Annotated[str | None, Parameter()] = None
    cache_tag: Annotated[str | None, Parameter()] = None

    def model_post_init(self, __context) -> None:
        assert self.pack_max_length >= self.max_length, (
            f"pack_max_length {self.pack_max_length} must be larger than max_length {self.max_length}"
        )


class DatasetCombine(TypedDict):
    dataset: DatasetConfig
    tokenize_fn: BaseTokenizeFnConfig


DatasetConfigList = list[DatasetCombine]
DatasetConfigListAdatper = TypeAdapter(DatasetConfigList, config=ConfigDict(arbitrary_types_allowed=True))
