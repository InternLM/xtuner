import sys
from pathlib import Path
from typing import Dict, List, Sized, cast

import ray
import torch
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from ray.actor import ActorClass

from transformers import AutoTokenizer
from xtuner.v1.config import DataloaderConfig
from xtuner.v1.config.trainer import ResumeConfig
from xtuner.v1.datasets.build import build_dataloader, build_datasets
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, log_format


@ray.remote
class TrainingController:
    """Controller for managing training processes in the Ray framework."""

    def __init__(
        self,
        workers: List[ActorClass],
        dataset_config: List[Dict],
        dataloader_config: DataloaderConfig,
        tokenizer: str | Path,
        global_batch_size: int,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        sp_size: int = 1,
        total_step: int | None = None,
        epoch_num: int | None = None,
        resume_config: ResumeConfig | None = None,
        seed: int = 42,
        debug: bool = False,
    ):
        self.workers = workers

        self._global_batch_size = global_batch_size
        self._micro_batch_size: int | None = None
        self._dataset_config = dataset_config
        self._dataloader_config = dataloader_config
        self._total_step = total_step
        self._epoch_num = epoch_num
        self._cur_step = 0

        assert epoch_num is not None or total_step is not None, "`epoch_num` or `total_step` should be set"
        assert epoch_num is None or total_step is None, (
            f"`epoch_num`: {epoch_num}, `total_step`: {total_step} should not be set at the same time"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # TODO: Unify the name of `cfg` and `config`
        self.sp_size = sp_size
        self.debug = debug

        self._set_deterministic()
        self._set_random_seed(seed)

        if work_dir is None:
            work_dir = Path.cwd() / "work_dir"

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if log_dir is None:
            log_dir = work_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.logger = self._init_logger(log_dir)

        if not work_dir.exists():
            self.logger.info(f"Creating work directory: {work_dir}")

        self.work_dir = work_dir

        self.tokenizer.model_max_length = dataloader_config.max_length
        self._dataloader = self.build_dataloader(
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            tokenizer=self.tokenizer,
            global_batch_size=global_batch_size,
            micro_batch_size=global_batch_size,
            seed=seed,
        )

    @property
    def global_batch_size(self):
        return self._global_batch_size

    @property
    def total_step(self):
        if self._total_step is None:
            assert isinstance(self._dataloader, Sized), (
                f"`epoch_num` should be set for a Mapped dataset, but got {self._dataloader.dataset}"
            )
            self._total_step = len(self._dataloader) * cast(int, self._epoch_num)
        return self._total_step

    @property
    def cur_step(self):
        return self._cur_step

    def _init_logger(self, work_dir: Path):
        logger = get_logger()
        logger.add(work_dir / f"rank{get_rank()}.log", format=log_format(), backtrace=True, catch=True)
        logger.add(sys.stderr, format=log_format(rank=get_rank()))
        return logger

    def data_iter(self):
        data_iter = iter(self._dataloader)
        for i in range(self.total_step):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataloader)
                data = next(data_iter)
            yield data

    def build_dataloader(
        self,
        dataloader_config: DataloaderConfig,
        dataset_config: List[Dict],
        tokenizer: AutoTokenizer,
        global_batch_size: int,
        micro_batch_size: int,
        seed: int,
        resume_config: ResumeConfig | None = None,
    ):
        # TODO: Support resume
        # 1. load dataloader state
        # 2. set cur step
        # TODO(hha): 如何传入 model_cfg 到 dataset 中
        datasets = build_datasets(dataset_config, tokenizer)  # type: ignore
        return build_dataloader(
            dataloader_config=dataloader_config,
            datasets=datasets,
            dp_mesh=None,
            seed=seed,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        )

    def fit(self):
        """Start the training process."""
        self.logger.info(f"Starting training with {len(self.workers)} workers.")
        self.logger.info(f"Total steps: {self.total_step}, Global batch size: {self.global_batch_size}")

        for step, data in enumerate(self.data_iter()):
            self._cur_step = step
            self.logger.info(f"Training step {step + 1}/{self.total_step}")

            # Distribute data to workers and perform training
            ray.get([worker.train_step.remote(data_batches=data) for worker in self.workers])
            # Process results from workers if needed

        self.logger.info("Training completed.")

    def save(self):
        ...
        # TODO: save latest information in `meta`

    def save_hf(self): ...

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)
