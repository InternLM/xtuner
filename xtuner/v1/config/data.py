from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict


if TYPE_CHECKING:
    from xtuner.v1.datasets import CachableTokenizeFunction, FtdpTokenizeFunction, JsonlDataset


class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    anno_path: str
    class_name: str = "JsonlDataset"
    name: str = "default"
    sample_ratio: float = 1.0
    media_root: str | None = None

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


class FTDPTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    chat_template: str = "internlm2"
    hash: str | None = None

    def build(self, tokenizer, tokenizer_hash: str | None = None, **kwargs) -> "FtdpTokenizeFunction":
        from xtuner.v1.datasets import FtdpTokenizeFunction

        return FtdpTokenizeFunction(
            tokenizer, chat_template=self.chat_template, hash=self.hash, tokenizer_hash=tokenizer_hash
        )


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(title="Base dataloader config for xtuner", extra="allow")
    pack_level: str = "soft"
    pack_max_length: int = 32768
    max_length: int = 4096
    global_pack: bool = True
    group_by_length: bool = True
    pack_extra_buffer_size: int = 100
    num_workers: int = 0
    padding_token_idx: int = 0
    cache_dir: str | None = None
    cache_tag: str | None = None

    def model_post_init(self, __context) -> None:
        assert self.pack_max_length >= self.max_length, (
            f"pack_max_length {self.pack_max_length} must be larger than max_length {self.max_length}"
        )
