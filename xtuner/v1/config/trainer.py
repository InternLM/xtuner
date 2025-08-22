from pathlib import Path
from typing import Annotated, TypedDict

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.base_model import TransformerConfig

from .data import DataloaderConfig, DatasetConfigList
from .fsdp import FSDPConfig
from .optim import LRConfig, OptimConfig


class ResumeConfig(TypedDict):
    resume_from: str | Path | None
    load_optimizer: bool
    load_dataset: bool
    load_scheduler: bool


class TrainerConfig(BaseModel):
    model_config = ConfigDict(title="Trainer config", extra="allow", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig
    load_from: str | Path | None = None
    tokenizer_path: str | Path
    dataset_cfg: Annotated[DatasetConfigList, Parameter(show_default=False)]
    dataloader_cfg: DataloaderConfig
    optim_cfg: OptimConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig | None = None
    global_batch_size: int | None
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    sp_size: int = 1
    total_step: int | None = None
    epoch_num: int | None = None
    resume: ResumeConfig | None = None
    strict_load: bool = True
    hf_interval: int | None = None
    hf_max_keep: int | None = None
    profile_step: int | None = None
    profile_time: bool = True
    profile_memory: bool = False
    seed: int = 42
    dist_backend: str = "cpu:gloo,cuda:nccl"
    debug: bool = False
    chunked_loss: Annotated[bool, Parameter(group="model")] = False

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self
