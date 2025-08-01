import os
import sys
import time
from pathlib import Path
from typing import Sized, cast

import torch
import torch.distributed as dist
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
from torch.distributed import init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing_extensions import Self

from transformers import AutoTokenizer
from xtuner.utils.device import get_device, get_torch_device
from xtuner.v1.config import DataloaderConfig, DatasetConfigList, FSDPConfig, LRConfig, OptimConfig
from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.config.trainer import ResumeConfig, TrainerConfig
from xtuner.v1.datasets.build import build_dataloader, build_datasets
from xtuner.v1.engine.utils import cal_global_grad_tokens
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.interns1 import InternS1Config
from xtuner.v1.utils import XTUNER_DETERMINISTIC, ParallelConfigException, get_logger, log_format


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device()


class Trainer:
    META_PATH = "meta"
    config: TrainerConfig | None

    def __init__(
        self,
        *,
        load_from: str | Path | None = None,  # Huggingface model path or saved trainer_path
        model_cfg: TransformerConfig,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig | None = None,
        dataset_cfg: DatasetConfigList,
        dataloader_cfg: DataloaderConfig,
        loss_ctx: CELossContext | None = None,
        lr_cfg: LRConfig,
        tokenizer_path: str | Path,
        global_batch_size: int | None,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        sp_size: int = 1,
        total_step: int | None = None,
        epoch_num: int | None = None,
        resume_config: ResumeConfig | None = None,
        strict_load: bool = True,
        seed: int = 42,
        debug: bool = False,
        backend: str = "cpu:gloo,cuda:nccl",
    ):
        self._micro_batch_size: int | None = None
        self._dataset_config = dataset_cfg
        self._dataloader_config = dataloader_cfg
        self._total_step = total_step
        self._epoch_num = epoch_num
        self._cur_step = 0

        assert epoch_num is not None or total_step is not None, "`epoch_num` or `total_step` should be set"
        assert epoch_num is None or total_step is None, (
            f"`epoch_num`: {epoch_num}, `total_step`: {total_step} should not be set at the same time"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig()
        self.fsdp_config = fsdp_cfg
        self.optim_config = optim_cfg

        self.sp_size = sp_size
        self.debug = debug
        self.seed = seed

        self._init_dist(backend)
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
        self.data_mesh = self._init_data_mesh(
            fsdp_cfg.tp_size,
            sp_size,
        )
        self.sp_mesh = self.data_mesh["sp"]

        if global_batch_size is None:
            global_batch_size = self.data_mesh["dp"].size()
        self._global_batch_size = global_batch_size

        self.tokenizer.model_max_length = dataloader_cfg.max_length

        self._dataloader = self.build_dataloader(
            dataset_config=dataset_cfg,
            dataloader_config=dataloader_cfg,
            dp_mesh=self.data_mesh["dp"],
            tokenizer=self.tokenizer,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seed=seed,
        )

        self.loss_ctx = loss_ctx
        if isinstance(load_from, str):
            load_from = Path(load_from)

        self._engine = self.build_engine(
            model_path=load_from,
            model_config=model_cfg,
            optim_config=optim_cfg,
            fsdp_config=fsdp_cfg,
            resume_config=resume_config,
            strict=strict_load,
        )
        self._lr_scheduler = self.build_lr_scheduler(lr_cfg)
        # TODO: (huanghaian) The impl of CELossContext should be decoupled with config
        if loss_ctx is None:
            self.loss_ctx = CELossContext()
        # TODO: TMP hardcode here

    @classmethod
    def from_config(cls, config: TrainerConfig) -> Self:
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config: TrainerConfig instance containing all configuration parameters

        Returns:
            Trainer instance initialized with the provided config
        """
        self = cls(
            load_from=config.load_from,
            model_cfg=config.model_cfg,
            optim_cfg=config.optim_cfg,
            fsdp_cfg=config.fsdp_cfg,
            dataset_cfg=config.dataset_cfg,
            dataloader_cfg=config.dataloader_cfg,
            lr_cfg=config.lr_cfg,
            tokenizer_path=config.tokenizer_path,
            global_batch_size=config.global_batch_size,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            sp_size=config.sp_size,
            total_step=config.total_step,
            epoch_num=config.epoch_num,
            resume_config=config.resume,
            strict_load=config.strict_load,
            seed=config.seed,
            debug=config.debug,
        )
        self.config = config
        return self

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
                labels.to(DEVICE)
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
        tp_size: int,
        sp_size: int,
    ):
        if self.world_size % tp_size != 0:
            raise ParallelConfigException(
                f"Found tp_size {tp_size}, world_size {self.world_size}."
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
        device = str(DEVICE) if self.fsdp_config.cpu_offload else "cpu"

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

    def build_engine(
        self,
        model_path: Path | None,
        model_config: TransformerConfig,
        optim_config: OptimConfig,
        fsdp_config: FSDPConfig,
        resume_config: ResumeConfig | None = None,
        strict: bool = True,
    ):
        from xtuner.v1.engine import InternS1TrainEngine, MoETrainEngine

        # TODO: yehaochen
        if isinstance(model_config, MoEConfig):
            engine = MoETrainEngine(
                optim_cfg=optim_config,
                fsdp_cfg=fsdp_config,
                model_cfg=model_config,
            )
        # TODO: 太 hard code 了
        elif isinstance(model_config, InternS1Config):
            engine = InternS1TrainEngine(
                optim_cfg=optim_config,
                fsdp_cfg=fsdp_config,
                model_cfg=model_config,
            )
        else:
            raise NotImplementedError

        if model_path is not None:
            engine.from_hf(hf_path=model_path, strict=strict)
        return engine

    def build_dataloader(
        self,
        dataloader_config: DataloaderConfig,
        dataset_config: DatasetConfigList,
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

    def _init_dist(self, backend: str):
        if not dist.is_initialized():
            init_process_group(backend=backend)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
