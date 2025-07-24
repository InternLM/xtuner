from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, model_validator

from .data import DataloaderConfig, DatasetConfig
from .engine import EngineConfig


class ResumeConfig(TypedDict):
    resume_from: str | Path | None
    load_optimizer: bool
    load_dataset: bool
    load_scheduler: bool


class TrainerConfig(BaseModel):
    hf_path: str | Path | None = None
    engine: EngineConfig
    dataset_config: DatasetConfig
    dataloader_config: DataloaderConfig
    resume_config: ResumeConfig
    global_batch_size: int
    total_step: int | None
    epoch_num: int | None
    sp_size: int = 1
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    tokenizer: str | Path
    seed: int = 42
    # data

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self
