import contextlib
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Sized, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine import is_installed, load
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
from pydantic import BaseModel
from torch.distributed import init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing_extensions import NotRequired, Self, TypedDict

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from xtuner.utils.device import get_device, get_torch_device
from xtuner.v1.config import DataloaderConfig, DatasetConfigList, FSDPConfig, LRConfig, OptimConfig
from xtuner.v1.config.base_model import TransformerConfig
from xtuner.v1.config.trainer import ResumeConfig, TrainerConfig
from xtuner.v1.datasets.build import build_dataloader, build_datasets
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.interns1 import InternS1BaseConfig
from xtuner.v1.profiler import profilling_memory, profilling_time
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_logger,
    is_hf_model_path,
    log_format,
    record_git_info,
)

from .toy_tokenizer import UTF8ByteTokenizer


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device()


logger = get_logger()


class GitInfo(TypedDict):
    commit: str
    staged: str
    unstaged: str


class ExpHistory(TypedDict):
    begin: int
    timestamp: str
    git_info: GitInfo
    end: NotRequired[int]
    comment: NotRequired[str]


class ExpInfo(BaseModel):
    history: list[ExpHistory]
    exp_dir: str
    latest_checkpoint: str | None = None
    hf_checkpoint_list: list[str] = []


class XTunerMeta(BaseModel):
    exps: list[ExpInfo]

    @property
    def latest_checkpoint(self) -> str | None:
        for exp in self.exps:
            if exp.latest_checkpoint is not None:
                return exp.latest_checkpoint
        return None

    @property
    def latest_exp(self) -> ExpInfo:
        return self.exps[-1]


class Trainer:
    """Trainer class for fine-tuning transformer models with FSDP support.

    This class provides a high-level interface for training transformer models
    with configurable distributed training, optimization, and checkpointing.
    It supports various training configurations including sequence parallelism,
    tensor parallelism, and data parallelism.

    Args:
        load_from (str | Path | None): Path to Huggingface model or saved trainer checkpoint.
        model_cfg (TransformerConfig | InternS1BaseConfig): Configuration for the transformer model architecture.
        optim_cfg (OptimConfig): Configuration for the optimizer.
        fsdp_cfg (FSDPConfig | None): Configuration for Fully Sharded Data Parallel (FSDP).
        dataset_cfg (DatasetConfigList): Configuration for training datasets.
        dataloader_cfg (DataloaderConfig): Configuration for the data loader.
        loss_ctx (CELossContext | None): Context for the cross-entropy loss function.
        lr_cfg (LRConfig): Configuration for the learning rate scheduler.
        tokenizer_path (str | Path | None): Path to the tokenizer.
        global_batch_size (int | None): Global batch size for training.
        work_dir (Path | str | None): Directory for saving experiment outputs.
        log_dir (Path | str | None): Directory for log files.
        sp_size (int): Sequence parallel size.
        total_step (int | None): Total training steps.
        epoch_num (int | None): Number of training epochs.
        resume_config (ResumeConfig | None): Configuration for resuming training.
        strict_load (bool): Whether to strictly load model weights.
        hf_interval (int | None): Interval for saving Huggingface format checkpoints.
        hf_max_keep (int | None): Maximum number of Huggingface checkpoints to keep.
        profile_step (int | None): Step to perform profiling.
        profile_time (bool): Whether to profile training time.
        profile_memory (bool): Whether to profile memory usage.
        intra_layer_micro_batch (int): Intra-layer micro batch size.
        seed (int): Random seed for reproducibility.
        debug (bool): Whether to enable debug mode.
        backend (str): Backend for distributed training.
    """

    config: TrainerConfig | None
    _META_PATH = ".xtuner"
    _PROFILE_TIME_PATH = "profilling_time"
    _PROFILE_MEMORY_PATH = "profilling_memory"

    def __init__(
        self,
        *,
        load_from: str | Path | None = None,  # Huggingface model path or saved trainer_path
        # TODO: InternS1BaseConfig 是组合配置，后续应该专门写一个组合 base model cfg，就可以通用
        model_cfg: TransformerConfig | InternS1BaseConfig,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig | None = None,
        dataset_cfg: DatasetConfigList,
        dataloader_cfg: DataloaderConfig,
        loss_ctx: CELossContext | None = None,
        lr_cfg: LRConfig,
        tokenizer_path: str | Path | None = None,
        global_batch_size: int | None,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        sp_size: int = 1,
        total_step: int | None = None,
        epoch_num: int | None = None,
        resume_config: ResumeConfig | None = None,
        strict_load: bool = True,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        profile_step: int | None = None,
        profile_time: bool = True,
        profile_memory: bool = False,
        intra_layer_micro_batch: int = 1,
        seed: int = 42,
        debug: bool = False,
        backend: str | None = None,
    ):
        self._micro_batch_size: int | None = None
        self._dataset_config = dataset_cfg
        self._dataloader_config = dataloader_cfg
        self._total_step = total_step
        self._epoch_num = epoch_num
        self._cur_step = 0

        self._profile_step = profile_step
        self._profile_time = profile_time
        self._profile_memory = profile_memory
        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from
        self._load_from_hf = load_from is not None and is_hf_model_path(load_from)

        if not self._load_from_hf:
            assert hf_interval is None and hf_max_keep is None, (
                "`hf_interval` and `hf_max_keep` should be None when `load_from` is not a Huggingface model path, "
            )

        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval

        assert epoch_num is not None or total_step is not None, "`epoch_num` or `total_step` should be set"
        assert epoch_num is None or total_step is None, (
            f"`epoch_num`: {epoch_num}, `total_step`: {total_step} should not be set at the same time"
        )

        if tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = UTF8ByteTokenizer()
            logger.info(f"Using toy tokenizer: {self.tokenizer}!")

        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig()

        self._fsdp_config = fsdp_cfg
        self._optim_config = optim_cfg

        self._sp_size = sp_size
        self._debug = debug
        self._seed = seed

        self._consumed_tokens = 0

        self._init_dist(backend)
        self._set_deterministic()
        self._set_random_seed(seed)

        if work_dir is None:
            work_dir = Path.cwd() / "work_dir"

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        self._work_dir = work_dir
        self._meta = self._init_xtuner_meta(work_dir, resume_config is not None)

        if log_dir is None:
            log_dir = self.exp_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.logger = self._init_logger(log_dir)

        self.data_mesh = self._init_data_mesh(
            fsdp_cfg.tp_size,
            sp_size,
        )
        self.sp_mesh = self.data_mesh["sp"]

        if global_batch_size is None:
            global_batch_size = self.data_mesh["dp"].size()
        self._global_batch_size = global_batch_size

        self._resolve_config_conflicts(self.tokenizer, model_cfg, dataloader_cfg)

        self._dataloader = self.build_dataloader(
            dataset_config=dataset_cfg,
            dataloader_config=dataloader_cfg,
            dp_mesh=self.data_mesh["dp"],
            tokenizer=self.tokenizer,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seed=seed,
        )

        if isinstance(load_from, str):
            load_from = Path(load_from)

        self._engine = self.build_engine(
            model_path=load_from,
            model_config=model_cfg,
            optim_config=optim_cfg,
            fsdp_config=fsdp_cfg,
            resume_config=resume_config,
            strict=strict_load,
            intra_layer_micro_batch=intra_layer_micro_batch,
        )
        self._lr_scheduler = self.build_lr_scheduler(lr_cfg)
        # TODO: (huanghaian) The impl of CELossContext should be decoupled with config
        if loss_ctx is None:
            self.loss_ctx = CELossContext()
        else:
            self.loss_ctx = loss_ctx
        # TODO: TMP hardcode here
        #
        if debug:
            self._register_debug_hook()

        if self._load_from is not None and is_hf_model_path(self._load_from) and self._hf_interval is None:
            self._hf_interval = self.total_step

    @classmethod
    def from_config(cls, config: TrainerConfig) -> Self:
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config (TrainerConfig): TrainerConfig instance containing all configuration parameters.

        Returns:
            Self: Trainer instance initialized with the provided config.
        """
        if config.chunked_loss:
            if is_installed("liger_kernel"):
                loss_class = "liger_cross_entropy"
            else:
                loss_class = "chunk_cross_entropy"
        else:
            loss_class = "cross_entropy"

        loss_ctx = CELossContext(loss_class=loss_class)
        self = cls(
            load_from=config.load_from,
            model_cfg=config.model_cfg,
            optim_cfg=config.optim_cfg,
            fsdp_cfg=config.fsdp_cfg,
            dataset_cfg=config.dataset_cfg,
            dataloader_cfg=config.dataloader_cfg,
            loss_ctx=loss_ctx,
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
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            profile_step=config.profile_step,
            profile_time=config.profile_time,
            profile_memory=config.profile_memory,
            intra_layer_micro_batch=config.intra_layer_micro_batch,
            seed=config.seed,
            backend=config.dist_backend,
            debug=config.debug,
        )
        self.config = config
        return self

    def fit(self):
        """Run the training loop.

        This method executes the main training loop, iterating through the dataset and performing training steps. It
        handles data loading, forward pass, backward pass, optimization, logging, and checkpointing.
        """
        train_begin = time.time()
        time_before_get_data = time.time()
        for data_batch in self._data_iter():
            DEVICE_MODULE.reset_peak_memory_stats()

            time_before_train_step = time.time()
            data_time = time_before_train_step - time_before_get_data

            data_batch = self.loss_ctx.build_list_ctx(data_batch, self.data_mesh, DEVICE)
            with self._maybe_profilling():
                loss_log, other_log = self._engine.train_step(data_batch)

            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)
            self._lr_scheduler.step()
            self._cur_step += 1

            time_after_train_step = time.time()
            step_time = time_after_train_step - time_before_train_step
            step_consumed_tokens = other_log["consumed_tokens"]
            self._consumed_tokens += step_consumed_tokens

            self._log_step(
                loss_log=loss_log,
                step_consumed_tokens=step_consumed_tokens,
                total_consumed_tokens=self._consumed_tokens,
                data_time=data_time,
                step_time=step_time,
                train_time=time_after_train_step - train_begin,
                grad_norm=grad_norm,
            )

            time_before_get_data = time.time()
            self._maybe_save_hf()
            self._maybe_save()

    @property
    def world_size(self) -> int:
        """Get the total number of processes in the distributed training group.

        Returns:
            int: Total number of processes.
        """
        return get_world_size()

    @property
    def rank(self) -> int:
        """Get the rank of the current process in the distributed training
        group.

        Returns:
            int: Rank of the current process.
        """
        return get_rank()

    @property
    def micro_batch_size(self) -> int:
        """Calculate the micro batch size per data parallel rank.

        Returns:
            int: Micro batch size for the current rank.
        """
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
    def global_batch_size(self) -> int:
        """Get the global batch size across all data parallel ranks.

        Returns:
            int: Global batch size.
        """
        return self._global_batch_size

    @property
    def total_step(self) -> int:
        """Calculate the total number of training steps.

        Returns:
            int: Total training steps.
        """
        if self._total_step is None:
            assert isinstance(self._dataloader, Sized), (
                f"`epoch_num` should be set for a Mapped dataset, but got {self._dataloader.dataset}"
            )
            self._total_step = len(self._dataloader) * cast(int, self._epoch_num)
        return self._total_step

    @property
    def cur_step(self) -> int:
        """Get the current training step.

        Returns:
            int: Current step number.
        """
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
                f"Found sp_size {self._sp_size}, world_size {self.world_size}."
                "sequence parallel size must be a divisor of world size."
            )

        if self.world_size % (tp_size * sp_size) != 0:
            raise ParallelConfigException(
                f"Found tp_size {tp_size}, sp_size {sp_size}, world_size {self.world_size}."
                "`tp_size * sp_size` size must be a divisor of world size."
            )

        dp_size = self.world_size // (tp_size * sp_size)

        # TODO: fsdp_config could be None
        device = str(DEVICE) if self._fsdp_config.cpu_offload else "cpu"

        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size, tp_size),
            mesh_dim_names=("dp", "sp", "tp"),
        )
        return data_mesh

    def build_engine(
        self,
        model_path: Path | None,
        model_config: TransformerConfig | InternS1BaseConfig,
        optim_config: OptimConfig,
        fsdp_config: FSDPConfig,
        resume_config: ResumeConfig | None = None,
        intra_layer_micro_batch: int = 1,
        strict: bool = True,
    ):
        """Build the training engine for the transformer model.

        Args:
            model_path (Path | None): Path to the model checkpoint or None for new initialization.
            model_config (TransformerConfig | InternS1BaseConfig): Model configuration.
            optim_config (OptimConfig): Optimizer configuration.
            fsdp_config (FSDPConfig): FSDP configuration for distributed training.
            resume_config (ResumeConfig | None): Resume configuration for continuing training.
            intra_layer_micro_batch (int): Intra-layer micro batch size for gradient accumulation.
            strict (bool): Whether to strictly load model weights.

        Returns:
            TrainEngine: Initialized training engine.
        """
        from xtuner.v1.engine import InternS1TrainEngine, TrainEngine

        if isinstance(model_config, InternS1BaseConfig):
            engine = InternS1TrainEngine(
                optim_cfg=optim_config,
                fsdp_cfg=fsdp_config,
                model_cfg=model_config,
                intra_layer_micro_batch=intra_layer_micro_batch,
            )
        else:
            engine = TrainEngine(  # type: ignore
                optim_cfg=optim_config,
                fsdp_cfg=fsdp_config,
                model_cfg=model_config,
            )
        if model_path is not None:
            engine.from_hf(hf_path=model_path, strict=strict)
        else:
            engine.init_model_weights()
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
        """Build the dataloader for training.

        Args:
            dataloader_config (DataloaderConfig): Configuration for the data loader.
            dataset_config (DatasetConfigList): Configuration for training datasets.
            tokenizer (AutoTokenizer): Tokenizer for processing text data.
            dp_mesh (DeviceMesh): Device mesh for data parallelism.
            global_batch_size (int): Global batch size across all ranks.
            micro_batch_size (int): Micro batch size per rank.
            seed: Random seed for reproducibility.
            resume_config (ResumeConfig | None): Resume configuration for continuing training.

        Returns:
            DataLoader: Configured dataloader for training.
        """
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
        """Build the learning rate scheduler.

        Args:
            lr_cfg (LRConfig): Configuration for the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: Configured learning rate scheduler.
        """
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

    def _maybe_save(self):
        ...
        # TODO: save latest information in `meta`

    @property
    def work_dir(self) -> Path:
        """Get the working directory for the trainer.

        Returns:
            Path: Working directory path.
        """
        return self._work_dir

    @property
    def exp_dir(self) -> Path:
        """Get the experiment directory for the current run.

        Returns:
            Path: Experiment directory path.
        """
        return Path(self._meta.latest_exp.exp_dir)

    @property
    def meta(self) -> XTunerMeta:
        """Get the XTuner metadata for tracking experiments.

        Returns:
            XTunerMeta: Experiment metadata tracker.
        """
        return self._meta

    def _data_iter(self):
        data_iter = iter(self._dataloader)
        for i in range(self.total_step):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(self._dataloader)
                data = next(data_iter)
            yield data

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)

    def _init_dist(self, backend: str | None = None):
        if backend is None:
            if torch.accelerator.current_accelerator().type == "cuda":
                backend = "cpu:gloo,cuda:nccl"
            elif torch.accelerator.current_accelerator().type == "npu":
                backend = "cpu:gloo,npu:hccl"
            else:
                raise NotImplementedError

        if not dist.is_initialized():
            init_process_group(backend=backend)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))

    def _init_xtuner_meta(self, work_dir: Path, resume: bool) -> XTunerMeta:
        if not work_dir.exists():
            if self.rank == 0:
                work_dir.mkdir(parents=True, exist_ok=True)

        meta_path = work_dir / self._META_PATH
        if not meta_path.exists() and self.rank == 0:
            meta = XTunerMeta(exps=[])
            with open(meta_path, "w") as f:
                f.write(meta.model_dump_json(indent=2))
        dist.barrier()

        meta = cast(XTunerMeta, XTunerMeta.model_validate(load(meta_path, file_format="json")))

        resume = resume and bool(meta.exps)

        if resume:
            latest_exp = meta.exps[-1]
            latest_exp_history = latest_exp.history[-1]

            begin = cast(int, latest_exp_history.get("end") or latest_exp_history["begin"])
            exp_dir = Path(latest_exp.exp_dir)
            git_dir = exp_dir / f"git-info-begin-{begin}"

            if not git_dir and self.rank == 0:
                git_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"

            commit = record_git_info(staged_path, unstaged_path)
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            timestamp_list = [datetime.now().strftime("%Y%m%d%H%M%S")]
            dist.broadcast_object_list(timestamp_list, src=0)
            timestamp = timestamp_list[0]
            new_exp_history = ExpHistory(
                begin=begin,
                timestamp=timestamp,
                git_info=git_info,
            )
            latest_exp.history.append(new_exp_history)
        else:
            timestamp_list = [datetime.now().strftime("%Y%m%d%H%M%S")]
            dist.broadcast_object_list(timestamp_list, src=0)
            timestamp = timestamp_list[0]
            exp_dir = work_dir / timestamp
            git_dir = Path(f"{exp_dir}/git-info-begin-{0}")

            if not git_dir.exists() and self.rank == 0:
                git_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"
            _commit_tmp: list[str | None]
            if self.rank == 0:
                commit = record_git_info(staged_path, unstaged_path)
                _commit_tmp = [commit]
            else:
                _commit_tmp = [None]
            dist.broadcast_object_list(_commit_tmp, src=0)
            commit = cast(str, _commit_tmp[0])
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            new_history = ExpHistory(
                begin=0,
                timestamp=timestamp,
                git_info=git_info,
            )
            new_exp = ExpInfo(
                history=[new_history],
                exp_dir=str(exp_dir),
                latest_checkpoint=None,
            )
            meta.exps.append(new_exp)
        return meta

    @contextmanager
    def _maybe_profilling(self):
        """Check if profiling is enabled and perform profiling if necessary."""
        if self._profile_step is not None and self._cur_step == self._profile_step:
            with contextlib.ExitStack() as stack:
                if self._profile_time:
                    time_dir = self.work_dir / self._PROFILE_TIME_PATH / f"step-{self._cur_step}"
                    stack.enter_context(profilling_time(time_dir))

                if self._profile_memory:
                    memory_dir = self.work_dir / self._PROFILE_MEMORY_PATH / f"step-{self._cur_step}"
                    stack.enter_context(profilling_memory(memory_dir))
                yield
        else:
            yield

    def _log_step(
        self,
        loss_log: dict,
        step_consumed_tokens: int,
        total_consumed_tokens: int,
        data_time: float,
        step_time: float,
        train_time: float,
        grad_norm: float,
    ):
        """Log the training step information."""
        tgs = step_consumed_tokens / step_time
        e2e_tgs = total_consumed_tokens / train_time
        lr = self._lr_scheduler.get_last_lr()[0]
        total_loss = loss_log["total_loss"]
        reduced_llm_loss = loss_log["reduced_llm_loss"]

        max_memory = DEVICE_MODULE.max_memory_allocated()  # type: ignore[attr-defined]
        reserved_memory = DEVICE_MODULE.max_memory_reserved()  # type: ignore[attr-defined]

        self.logger.info(
            f"Step {self.cur_step}/{self.total_step} data_time: {data_time:.4f} lr: {lr:.6f} time: {step_time:.4f} "
            f"text_tokens: {step_consumed_tokens} "
            f"total_loss: {total_loss:.3f} "
            f"reduced_llm_loss: {reduced_llm_loss:.3f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"grad_norm: {grad_norm:.3f} "
            f"tgs: {tgs:.1f} "
            f"e2e_tgs: {e2e_tgs:.1f} "
        )
        DEVICE_MODULE.reset_peak_memory_stats()  # type: ignore[attr-defined]

    def _maybe_save_hf(self):
        if self._hf_interval is None:
            return

        assert self._load_from_hf, (
            "Only support saving to Huggingface format when loading from Huggingface! "
            "You meet this error means `load_from` of trainer is not a Huggingface model path."
        )

        if self.cur_step % self._hf_interval != 0 and self.cur_step != self.total_step:
            return

        save_hf_path = self.exp_dir / f"hf-{self.cur_step}"
        self.meta.latest_exp.hf_checkpoint_list.append(str(save_hf_path))

        if self._hf_max_keep is not None and len(self.meta.latest_exp.hf_checkpoint_list) > self._hf_max_keep:
            deleted_hf_checkpoints = self.meta.latest_exp.hf_checkpoint_list[: -self._hf_max_keep]
            self.meta.latest_exp.hf_checkpoint_list = self.meta.latest_exp.hf_checkpoint_list[-self._hf_max_keep :]
            for hf_dir in deleted_hf_checkpoints:
                if self.rank == 0:
                    rmtree(hf_dir)

        self._engine.save_hf(str(save_hf_path))
        meta_path = self.work_dir / self._META_PATH

        if self.rank == 0:
            with meta_path.open("w") as f:
                f.write(self.meta.model_dump_json(indent=2))

    def _register_debug_hook(self):
        """Register a debug hook function to be called at the end of each
        training step."""

        def _detect_nan(module: nn.Module, output):
            if isinstance(output, torch.Tensor):
                if output.isnan().any():
                    logger.warning(f"Detect NaN in output of module {module.__class__.__name__}")
            elif isinstance(output, (tuple, list)):
                for item in output:
                    _detect_nan(module, item)
            elif isinstance(output, dict):
                for value in output.values():
                    _detect_nan(module, value)

        def module_debug_forward_hook(module, input, output):
            """Debug hook to print module name and input/output shapes."""
            _detect_nan(module, output)

        for model in self._engine.model.modules():
            if isinstance(model, nn.Module):
                model.register_forward_hook(module_debug_forward_hook)

    def _resolve_config_conflicts(
        self,
        tokenizer: PreTrainedTokenizer,
        model_cfg: TransformerConfig | InternS1BaseConfig,
        dataloader_cfg: DataloaderConfig,
    ):
        if hasattr(tokenizer, "pad_token_id"):
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = tokenizer.eos_token_id

        if not isinstance(pad_token_id, int):
            logger.warning(
                f"Tokenizer pad_token_id is {pad_token_id}, which is not an integer. Setting pad_token_id to 0."
            )

        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]

        assert isinstance(pad_token_id, int), f"pad_token_id should be an integer, but got {pad_token_id}"

        # TODO: 后续配置会统一，因此不会有很多种情况
        if isinstance(model_cfg, InternS1BaseConfig):
            if model_cfg.text_config.pad_token_id != pad_token_id:
                logger.warning(
                    f"Model pad_token_id {model_cfg.text_config.pad_token_id} is different from tokenizer "
                    f"pad_token_id {pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
                )
                model_cfg.text_config.pad_token_id = pad_token_id

        elif model_cfg.pad_token_id != pad_token_id:
            logger.warning(
                f"Model pad_token_id {model_cfg.pad_token_id} is different from tokenizer pad_token_id "
                f"{pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
            )
            model_cfg.pad_token_id = pad_token_id

        if dataloader_cfg.pad_token_id is None:
            dataloader_cfg.pad_token_id = pad_token_id
        elif dataloader_cfg.pad_token_id != pad_token_id:
            logger.warning(
                f"Dataloader pad_token_id {dataloader_cfg.pad_token_id} is different from tokenizer "
                f"pad_token_id {pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
            )
            dataloader_cfg.pad_token_id = pad_token_id
