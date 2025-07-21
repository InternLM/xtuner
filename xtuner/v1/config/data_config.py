from typing import TypedDict

from pydantic import BaseModel, ConfigDict
from typing_extensions import Required


class MetaItem(TypedDict, total=False):
    annotation: Required[str]
    sample_ratio: float


class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    meta_datas: dict[str, MetaItem]
    dataset_args: dict | None = None
    tokenizer_fn_args: dict | None = None


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

    def model_post_init(self, __context) -> None:
        assert self.pack_max_length > self.max_length, (
            f"pack_max_length {self.pack_max_length} must be larger than max_length {self.max_length}"
        )
