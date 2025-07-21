from pydantic import BaseModel, ConfigDict

from ..datasets.collator import sft_llm_collator
from ..datasets.ftdp import FtdpTokenizeFunction
from ..datasets.jsonl import JsonlDataset


class DatasetConfig(BaseModel):
    model_config = ConfigDict(title="Base dataset config for xtuner", extra="allow")
    meta_datas: dict
    dataset_class = JsonlDataset
    dataset_args: dict | None = None
    tokenizer_fn = FtdpTokenizeFunction
    tokenizer_fn_args: dict | None = None


class DataloaderConfig(BaseModel):
    model_config = ConfigDict(title="Base dataloader config for xtuner", extra="allow")
    global_batch_size: int
    collator_fn = sft_llm_collator
    pack_level: str = "softpack"
    pack_max_length: int = 32768
    global_pack: bool = True
    group_by_length: bool = True
    pack_extra_buffer_size: int = 100
    num_workers: int = 0
    mirco_batch_size: int = 0
