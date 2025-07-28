import sys
import time
from pathlib import Path
from typing import Dict, List, Sized, cast

import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

from transformers import AutoTokenizer
from xtuner.utils.device import get_device, get_torch_device
from xtuner.v1.config import DataloaderConfig, EngineConfig, LRConfig
from xtuner.v1.config.trainer import ResumeConfig
from xtuner.v1.data_proto import CELossContext
from xtuner.v1.datasets.build import build_dataloader, build_datasets
from xtuner.v1.engine import build_engine
from xtuner.v1.engine.utils import cal_global_grad_tokens
from xtuner.v1.model.base import ModelItem
from xtuner.v1.utils import XTUNER_DETERMINISTIC, ParallelConfigException, get_logger, log_format


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device()


class Trainer:
    META_PATH = "meta"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,  # Huggingface model path or saved trainer_path
        engine_config: EngineConfig,
        dataset_config: List[Dict],
        dataloader_config: DataloaderConfig,
        loss_ctx: CELossContext,
        lr_config: LRConfig,
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
        self.engine_config = engine_config
        self.sp_size = sp_size
        self.debug = debug
        self.seed = seed

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

        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        self.work_dir = work_dir
        self.data_mesh = self._init_data_mesh(engine_config)
        self.sp_mesh = self.data_mesh["sp"]

        self.tokenizer.model_max_length = dataloader_config.max_length

        self._dataloader = self.build_dataloader(
            dataset_config=dataset_config,
            dataloader_config=dataloader_config,
            dp_mesh=self.data_mesh["dp"],
            tokenizer=self.tokenizer,
            global_batch_size=global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seed=seed,
        )

        self._engine = self.build_engine(model_path, engine_config, resume_config)
        self._lr_scheduler = self.build_lr_scheduler(lr_config)
        self.loss_ctx = loss_ctx
        # TODO: TMP hardcode here

    def fit(self):
        time_before_get_data = time.time()
        for data_batch in self.data_iter():
            DEVICE_MODULE.reset_peak_memory_stats()

            time_before_train_step = time.time()
            data_time = time_before_train_step - time_before_get_data

            global_grad_tokens = cal_global_grad_tokens([i["labels"] for i in data_batch], self.sp_mesh)
            grad_accumulation_steps = self._engine.grad_accumulation_steps(len(data_batch))

            for data in data_batch:
                seq_ctx = data["seq_ctx"]
                labels = data["labels"]
                seq_ctx.to(DEVICE)
                # build_item 是一个自定义方法和接口
                loss_ctx = self.loss_ctx.build_forward_item(
                    seq_ctx=seq_ctx,
                    labels=labels,
                    grad_accumulation_steps=grad_accumulation_steps,
                    global_grad_tokens=global_grad_tokens,
                )
                del data["labels"]  # type: ignore
                data = cast(ModelItem, data)
                data["loss_ctx"] = loss_ctx

            loss_log, other_log = self._engine.train_step(data_batch)
            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)
            self._lr_scheduler.step()
            time_after_train_step = time.time()
            step_time = time_after_train_step - time_before_train_step

            self._cur_step += 1

            step_consumed_tokens = other_log["consumed_tokens"]
            tgs = step_consumed_tokens / step_time
            lr = self._lr_scheduler.get_last_lr()[0]
            total_loss = loss_log["total_loss"]
            reduced_llm_loss = loss_log["reduced_llm_loss"]

            max_memory = DEVICE_MODULE.max_memory_allocated()

            self.logger.info(
                f"Step {self.cur_step}/{self.total_step} data_time: {data_time:.4f} lr: {lr:.6f} "
                f"text_tokens: {step_consumed_tokens} "
                f"total_loss: {total_loss:.3f} "
                f"reduced_llm_loss: {reduced_llm_loss:.3f} "
                f"max_memory: {max_memory / (1024**3):.2f} GB "
                f"grad_norm: {grad_norm:.3f} tgs: {tgs:.1f} "
            )
            time_before_get_data = time.time()

    @property
    def world_size(self) -> int:
        return get_world_size()

    @property
    def rank(self) -> int:
        return get_rank()

    @property
    def micro_batch_size(self) -> int:
        if self._micro_batch_size is None:
            micro_batch_size = self.global_batch_size / self.data_mesh["dp"].size()
            if not micro_batch_size.is_integer():
                raise ParallelConfigException(
                    f"Global batch size {self.global_batch_size} must be divisible by "
                    f"data parallel size {self.data_mesh['dp'].size()}. "
                    "Please adjust the global batch size."
                )
            self._micro_batch_size = int(micro_batch_size)

        return self._micro_batch_size

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
        # Logging system maybe need better design
        logger = get_logger()
        logger.remove()
        logger.add(work_dir / f"rank{get_rank()}.log", format=log_format(), backtrace=True, catch=True)
        logger.add(sys.stderr, format=log_format(rank=get_rank()))
        return logger

    def _init_data_mesh(
        self,
        engine_config: EngineConfig,
    ):
        tp_size = engine_config.fsdp.tp_size
        sp_size = self.sp_size
        if self.world_size % tp_size != 0:
            raise ParallelConfigException(
                f"Found tp_size {self.engine_config.fsdp.tp_size}, world_size {self.world_size}."
                "tensor parallel size must be a divisor of world size."
            )

        if self.world_size % sp_size != 0:
            raise ParallelConfigException(
                f"Found sp_size {self.sp_size}, world_size {self.world_size}."
                "sequence parallel size must be a divisor of world size."
            )

        if self.world_size % (tp_size * sp_size) != 0:
            raise ParallelConfigException(
                f"Found tp_size {tp_size}, sp_size {sp_size}, world_size {self.world_size}."
                "`tp_size * sp_size` size must be a divisor of world size."
            )

        dp_size = self.world_size // (tp_size * sp_size)

        # TODO: fsdp_config could be None
        device = str(DEVICE) if not self.engine_config.fsdp.cpu_offload else "cpu"

        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size, tp_size),
            mesh_dim_names=("dp", "sp", "tp"),
        )
        return data_mesh

    def data_iter(self):
        data_iter = iter(self._dataloader)
        for i in range(self.total_step):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataloader)
                data = next(data_iter)
            yield data

    def build_engine(self, model_path: str | Path | None, engine_config: EngineConfig | None, resume_config):
        # TODO: yehaochen
        assert engine_config is not None, "Engine config should not be None"
        engine = build_engine(engine_config)
        if model_path is not None:
            engine.from_hf(str(model_path))
        return engine

    def build_dataloader(
        self,
        dataloader_config: DataloaderConfig,
        dataset_config: List[Dict],
        tokenizer: AutoTokenizer,
        dp_mesh: DeviceMesh,
        global_batch_size: int,
        micro_batch_size: int,
        seed,
        resume_config: ResumeConfig | None = None,
    ):
        # TODO: Support resume
        # 1. load dataloader state
        # 2. set cur step
        # TODO(hha): 如何传入 model_cfg 到 dataset 中
        datasets = build_datasets(dataset_config, tokenizer)
        return build_dataloader(
            dataloader_config=dataloader_config,
            datasets=datasets,
            dp_mesh=dp_mesh,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            seed=seed,
        )

    def build_lr_scheduler(self, lr_cfg: LRConfig) -> torch.optim.lr_scheduler.LRScheduler:
        total_step = self.total_step
        warmup_steps = int(lr_cfg.warmup_ratio * total_step)

        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        warmup_scheduler = LambdaLR(self._engine.optimizer, warmup_fn)

        scheduler: torch.optim.lr_scheduler.LRScheduler
        if lr_cfg.lr_type == "linear":
            scheduler = LinearLR(
                self._engine.optimizer,
                start_factor=1.0,
                end_factor=lr_cfg.lr_min / self._engine.optimizer.defaults["lr"],
                total_iters=total_step - warmup_steps,
            )
        elif lr_cfg.lr_type == "cosine":
            scheduler = CosineAnnealingLR(
                self._engine.optimizer, T_max=total_step - warmup_steps, eta_min=lr_cfg.lr_min
            )
        elif lr_cfg.lr_type == "constant":
            scheduler = LambdaLR(self._engine.optimizer, lambda x: 1.0)
        else:
            raise ValueError(f"Unsupported lr type: {lr_cfg.lr_type}")
        lr_scheduler = SequentialLR(
            optimizer=self._engine.optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_steps],
        )
        return lr_scheduler

    def save(self):
        ...
        # TODO: save latest information in `meta`

    def save_hf(self): ...

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)
