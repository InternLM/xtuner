import contextlib
import gc
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree
from typing import Annotated, Literal, Sized, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from cyclopts import Parameter
from mmengine import load
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator
from torch.distributed import init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing_extensions import NotRequired, Self, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1._writer import get_writer
from xtuner.v1.config import FSDPConfig, LRConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.config import BaseDataloaderConfig, DataloaderConfig, DatasetConfigList
from xtuner.v1.engine import TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeConfigProtocol, VisionComposeTrainEngine
from xtuner.v1.loss import CELossConfig
from xtuner.v1.loss.ce_loss import CELossContextInputItem
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.utils import ModelForwardExtraLogInfo
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.profiler.prober import ProberList
from xtuner.v1.profiler.prober_utils import setup_prober_list
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_logger,
    is_hf_model_path,
    log_format,
    record_git_info,
)
from xtuner.v1.utils.check_health import check_health
from xtuner.v1.utils.device import get_device, get_torch_device_module

from .toy_tokenizer import UTF8ByteTokenizer


# TODO: Move DEVICE to `xtuner.utils.device`
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


logger = get_logger()


class GitInfo(TypedDict):
    commit: str | None
    staged: str
    unstaged: str


class ExpHistory(TypedDict):
    begin: int
    timestamp: str
    git_info: GitInfo
    end: NotRequired[int]
    comment: NotRequired[str]


class ExpInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: list[ExpHistory]
    exp_dir: str
    hf_checkpoint_list: list[str] = []
    checkpoint_list: list[str] = []
    snap_checkpoint_list: list[str] = []
    cur_step: int = 0
    cur_epoch: int = 0
    consumed_tokens: int = 0
    consumed_samples: int = 0

    @property
    def latest_checkpoint(self) -> str | None:
        # compare checkpoint_list and snap_checkpoint_list, return the latest checkpoint
        latest_ckp = None
        if self.checkpoint_list:
            latest_ckp = self.checkpoint_list[-1]
        if self.snap_checkpoint_list:
            snap_ckp = self.snap_checkpoint_list[-1]
            latest_ckp = self._get_latest_checkpoint(latest_ckp, snap_ckp)
        return latest_ckp

    def _get_latest_checkpoint(self, ckp1: str | None, ckp2: str | None) -> str | None:
        if ckp1 is None:
            return ckp2
        if ckp2 is None:
            return ckp1
        # compare the timestamp of ckp1 and ckp2, return the latest one
        # ckp path is like: checkpoints/epoch-1-step-20 or checkpoints/snapshot-epoch-3-step-50
        step1 = int(ckp1.split("-")[-1])
        step2 = int(ckp2.split("-")[-1])
        return ckp1 if step1 > step2 else ckp2


class XTunerMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
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

    def get_exp_by_checkpoint(self, checkpoint: str) -> ExpInfo | None:
        for exp in self.exps:
            for cp in exp.checkpoint_list:
                if cp == checkpoint:
                    return exp
        return None


class ResumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_from: str | Path | None = None
    auto_resume: bool = False
    load_optimizer_states: bool = True
    load_optimizer_args: bool = True
    load_dataset: bool = True
    load_scheduler: bool = True


class TrainerConfig(BaseModel):
    model_config = ConfigDict(
        title="Trainer config",
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )
    model_cfg: TransformerConfig | VisionComposeConfigProtocol
    load_from: str | Path | None = None
    tokenizer_path: str | Path | None = None
    dataset_cfg: Annotated[DatasetConfigList | None, Parameter(show_default=False)] = (
        None  # TODO: Removed in version 1.1.0
    )
    dataloader_cfg: BaseDataloaderConfig
    optim_cfg: OptimConfig
    lr_cfg: LRConfig
    loss_cfg: CELossConfig = CELossConfig()
    fsdp_cfg: FSDPConfig | None = None
    global_batch_size: int | None
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    sp_size: int = 1
    total_step: int | None = None
    total_epoch: int | None = None
    resume_cfg: ResumeConfig | None = None
    strict_load: bool = True
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    skip_checkpoint_validation: bool = False  # Suggest enabled if fsdp_size is larger than 512
    snapshot_interval: int | None = None
    check_health_interval: int | None = None
    hf_interval: int | None = None
    hf_max_keep: int | None = None
    exp_tracker: Literal["tensorboard", "jsonl"] = "jsonl"
    profile_step: list[int] | int | None = None
    profile_time: bool = True
    profile_memory: bool = False
    intra_layer_micro_batch: int = 1
    seed: int = 42
    dist_backend: str | None = None
    debug: bool = False
    debug_skip_save: bool = False
    prober_list: list[str] = []
    do_clip: bool = True
    grad_norm_dtype: torch.dtype = torch.float32

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self

    @field_serializer("grad_norm_dtype")
    def serialize_dtype(self, value: torch.dtype) -> str:
        return str(value)

    @field_validator("grad_norm_dtype", mode="before")
    @classmethod
    def deserialize_dtype(cls, value: str) -> torch.dtype:
        if "bfloat16" in value:
            return torch.bfloat16
        elif "float32" in value:
            return torch.float32
        elif "float64" in value:
            return torch.float64
        else:
            raise ValueError()


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
        loss_cfg (CELossConfig | None): Config for the cross-entropy loss function.
        lr_cfg (LRConfig): Configuration for the learning rate scheduler.
        tokenizer_path (str | Path | None): Path to the tokenizer.
        global_batch_size (int | None): Global batch size for training.
        work_dir (Path | str | None): Directory for saving experiment outputs.
        log_dir (Path | str | None): Directory for log files.
        sp_size (int): Sequence parallel size.
        total_step (int | None): Total training steps.
        total_epoch (int | None): Number of training epochs.
        resume_cfg (ResumeConfig | None): Configuration for resuming training.
        strict_load (bool): Whether to strictly load model weights.
        checkpoint_interval (int | None): Interval for saving checkpoints.
        checkpoint_maxkeep (int | None): Maximum number of checkpoints to keep.
        hf_interval (int | None): Interval for saving Huggingface format checkpoints.
        hf_max_keep (int | None): Maximum number of Huggingface checkpoints to keep.
        profile_step (list[int] | int | None): Step to perform profiling.
        profile_time (bool): Whether to profile training time.
        profile_memory (bool): Whether to profile memory usage.
        intra_layer_micro_batch (int): Intra-layer micro batch size.
        seed (int): Random seed for reproducibility.
        debug (bool): Whether to enable debug mode.
        backend (str): Backend for distributed training.
    """

    config: TrainerConfig | None
    _META_PATH = ".xtuner"
    _PROFILE_TIME_PATH = "profiling_time"
    _PROFILE_MEMORY_PATH = "profiling_memory"
    _EXP_TRACKING_PATH = "exp_tracking"
    _CHECKPOINT_DIR = "checkpoints"

    _SAVE_OPTIMIZER_DIR = "optimizer"
    _SAVE_MODEL_DIR = "model"
    _SAVE_DATALOADER_DIR = "dataloader"
    _SAVE_SCHEDULER_DIR = "lr_scheduler"
    _SAVE_TRAIN_STATE_PATH = "train_state.json"
    _DEFAULT_LOG_DIR = "logs"

    def __init__(
        self,
        *,
        load_from: str | Path | None = None,  # Huggingface model path or saved trainer_path
        model_cfg: TransformerConfig | VisionComposeConfigProtocol,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig | None = FSDPConfig(),
        dataset_cfg: DatasetConfigList | None = None,  # TODO: Removed in version 1.1.0
        dataloader_cfg: DataloaderConfig,
        loss_cfg: CELossConfig | None = CELossConfig(),
        lr_cfg: LRConfig,
        tokenizer_path: str | Path | None = None,
        global_batch_size: int | None,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        sp_size: int = 1,
        total_step: int | None = None,
        total_epoch: int | None = None,
        resume_cfg: ResumeConfig | None = ResumeConfig(),
        strict_load: bool = True,
        checkpoint_interval: int | None = -1,
        checkpoint_maxkeep: int | None = -1,
        skip_checkpoint_validation: bool = False,  # Suggest enabled if fsdp_size is larger than 512
        snapshot_interval: int | None = None,
        check_health_interval: int | None = None,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        exp_tracker: Literal["tensorboard", "jsonl"] = "jsonl",
        profile_step: list[int] | int | None = None,
        profile_time: bool = True,
        profile_memory: bool = False,
        intra_layer_micro_batch: int = 1,
        seed: int = 42,
        debug: bool = False,
        backend: str | None = None,
        debug_skip_save: bool = False,
        prober_list: list[str] = [],
        do_clip: bool = True,
        grad_norm_dtype: torch.dtype = torch.float32,
        trainer_cfg: TrainerConfig | None = None,
    ):
        self._do_clip = do_clip
        self._grad_norm_dtype = grad_norm_dtype
        self._dataloader_config = dataloader_cfg

        self._total_step = total_step
        self._total_epoch = total_epoch
        self._cur_epoch = 1
        self._cur_step = 0

        self._trainer_cfg = trainer_cfg

        self._micro_batch_size: int | None = None
        if skip_checkpoint_validation:
            patch_default_save_plan()

        if isinstance(profile_step, int):
            profile_step = [profile_step]
        self._profile_step = profile_step
        self._profile_time = profile_time
        self._profile_memory = profile_memory
        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from
        self._load_from_hf = load_from is not None and is_hf_model_path(load_from)
        self._can_save_hf = model_cfg.hf_config is not None or self._load_from_hf

        if not self._can_save_hf:
            assert hf_interval is None and hf_max_keep is None, (
                "`hf_interval` and `hf_max_keep` should be None when `load_from` is not a Huggingface model path, "
            )

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_maxkeep = checkpoint_maxkeep
        self._snapshot_interval = snapshot_interval
        self._check_health_interval = check_health_interval
        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval

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
        self._consumed_samples = 0

        self._train_time = 0
        self._train_time_offset = 0

        self._init_dist(backend)
        self._try_bind_numa()
        self._set_deterministic()
        self._set_random_seed(seed)
        self._setup_env()

        if resume_cfg is None:
            resume_cfg = ResumeConfig()

        self._work_dir = self._resolve_work_dir(work_dir)
        self._meta = self._init_xtuner_meta(self.work_dir, auto_resume=resume_cfg.auto_resume)
        self._log_dir = self._resolve_log_dir(log_dir)
        self._resume_cfg = self._resolve_resume_cfg(resume_cfg)

        self.logger, log_dir = self._init_logger(self._log_dir)
        self._exp_tracker = self._init_tracker(
            exp_tracker, self._log_dir / f"{self._EXP_TRACKING_PATH}/rank{self.rank}"
        )

        self.data_mesh = self._init_data_mesh(
            fsdp_cfg.tp_size,
            sp_size,
        )
        self.sp_mesh = self.data_mesh["sp"]

        if global_batch_size is None:
            global_batch_size = self.data_mesh["dp"].size()
        self._global_batch_size = global_batch_size

        self._resolve_config_conflicts(self.tokenizer, model_cfg, dataloader_cfg)

        if dataset_cfg is not None:  # TODO: Removed in version 1.1.0
            # For backward compatibility, reserve the dataset_cfg interface, remove it later
            if dataloader_cfg.dataset_config_list is not None:
                logger.warning("Outside dataset_cfg will override inner dataset_config_list")
            dataloader_cfg.dataset_config_list = dataset_cfg

        self._dataloader = dataloader_cfg.build(
            tokenizer=self.tokenizer,
            dp_mesh=self.data_mesh["dp"],
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seed=seed,
            total_step=total_step,
        )

        # streaming dataloader may override `total_step`, so we may move this check after `build_dataloader` later.
        assert total_epoch is not None or total_step is not None, "`total_epoch` or `total_step` should be set"
        assert total_epoch is None or total_step is None, (
            f"`total_epoch`: {total_epoch}, `total_step`: {total_step} should not be set at the same time"
        )

        if isinstance(load_from, str):
            load_from = Path(load_from)

        self._engine = self.build_engine(
            model_path=load_from,
            model_config=model_cfg,
            optim_config=optim_cfg,
            fsdp_config=fsdp_cfg,
            resume_cfg=resume_cfg,
            strict=strict_load,
            intra_layer_micro_batch=intra_layer_micro_batch,
        )
        self._lr_cfg = lr_cfg
        self._lr_scheduler = self.build_lr_scheduler(lr_cfg, self.total_step)

        if loss_cfg is None:
            loss_cfg = CELossConfig()
        self.loss_cfg = loss_cfg

        # TODO: TMP hardcode here
        #
        if debug:
            self._register_debug_hook()

        if self._can_save_hf and self._hf_interval is None:
            self._hf_interval = self.total_step

        if debug_skip_save:
            self._hf_interval = None
            self._checkpoint_interval = None
            self._snapshot_interval = None

        if self._resume_cfg.resume_from is not None:
            self._resume()

        setup_prober_list(self.exp_dir, self._profile_step, self._engine.model, prober_list)

    @classmethod
    def from_config(cls, config: TrainerConfig) -> Self:
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config (TrainerConfig): TrainerConfig instance containing all configuration parameters.

        Returns:
            Self: Trainer instance initialized with the provided config.
        """
        self = cls(
            load_from=config.load_from,
            model_cfg=config.model_cfg,
            optim_cfg=config.optim_cfg,
            fsdp_cfg=config.fsdp_cfg,
            dataset_cfg=config.dataset_cfg,
            dataloader_cfg=config.dataloader_cfg,
            loss_cfg=config.loss_cfg,
            lr_cfg=config.lr_cfg,
            tokenizer_path=config.tokenizer_path,
            global_batch_size=config.global_batch_size,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            sp_size=config.sp_size,
            total_step=config.total_step,
            total_epoch=config.total_epoch,
            resume_cfg=config.resume_cfg,
            strict_load=config.strict_load,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_maxkeep=config.checkpoint_maxkeep,
            skip_checkpoint_validation=config.skip_checkpoint_validation,
            snapshot_interval=config.snapshot_interval,
            check_health_interval=config.check_health_interval,
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            exp_tracker=config.exp_tracker,
            profile_step=config.profile_step,
            profile_time=config.profile_time,
            profile_memory=config.profile_memory,
            intra_layer_micro_batch=config.intra_layer_micro_batch,
            seed=config.seed,
            backend=config.dist_backend,
            debug=config.debug,
            debug_skip_save=config.debug_skip_save,
            prober_list=config.prober_list,
            do_clip=config.do_clip,
            grad_norm_dtype=config.grad_norm_dtype,
            trainer_cfg=config,
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
            ProberList.set_step(self._cur_step + 1)
            DEVICE_MODULE.reset_peak_memory_stats()

            time_before_train_step = time.time()
            data_time = time_before_train_step - time_before_get_data

            seq_ctx_list: list[SequenceContext] = []
            loss_ctx_input_list: list[CELossContextInputItem] = []
            for data in data_batch:
                seq_ctx = data["seq_ctx"].to(DEVICE)
                loss_ctx_input = CELossContextInputItem(shifted_labels=data["shifted_labels"]).to(DEVICE)
                if self.sp_mesh.size() > 1:
                    seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
                    loss_ctx_input = loss_ctx_input.sp_split(self.sp_mesh)
                seq_ctx_list.append(seq_ctx)
                loss_ctx_input_list.append(loss_ctx_input)

            del data_batch

            LossContext = self.loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list,
                self.loss_cfg,
                cu_seq_lens_list=[seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list],
                sp_mesh=self.sp_mesh,
            )
            engine_input = []
            for seq_ctx, loss_kwargs in zip(seq_ctx_list, batches_loss_kwargs):
                loss_ctx = LossContext(
                    loss_cfg=self.loss_cfg,
                    loss_kwargs=loss_kwargs,
                )
                engine_input.append(
                    ModelItem(
                        seq_ctx=seq_ctx,
                        loss_ctx=loss_ctx,
                    )
                )

            with self._maybe_profiling():
                loss_log, other_log = self._engine.train_step(engine_input)

            grad_norm = self._engine.clip_grad_norm(do_clip=self._do_clip, dtype=self._grad_norm_dtype)
            self._engine.step_optimizer(grad_norm)
            time_after_train_step = time.time()
            ProberList.after_step()
            step_time = time_after_train_step - time_before_train_step
            step_consumed_tokens = other_log["consumed_tokens"]

            extra_info = other_log.get("extra_info", {})
            if isinstance(extra_info, ModelForwardExtraLogInfo):
                extra_info_dict = extra_info.get()
            else:
                extra_info_updated = ModelForwardExtraLogInfo(extra_info)
                extra_info_dict = extra_info_updated.get()
            loss_log.update(extra_info_dict)

            if "maxvio" in other_log:
                loss_log["maxvio"] = other_log["maxvio"]
            loss_log["efficient_attn_ratio"] = other_log["efficient_attn_ratio"]

            self._cur_step += 1
            self._consumed_tokens += step_consumed_tokens
            self._train_time = time_after_train_step - train_begin

            # TODO: This log should be move before lr_scheduler.step, but for CI BC, keep it temporarily
            self._log_step(
                loss_log=loss_log,
                step_consumed_tokens=step_consumed_tokens,
                total_consumed_tokens=self._consumed_tokens,
                data_time=data_time,
                step_time=step_time,
                train_time=self._train_time + self._train_time_offset,
                grad_norm=grad_norm.item(),
            )

            self._lr_scheduler.step()
            self._maybe_check_health()
            self._maybe_save_hf()
            ckpt_saved = self._maybe_save(is_snapshot=False)
            if not ckpt_saved:
                _ = self._maybe_save(is_snapshot=True)

            time_before_get_data = time.time()

            if self.cur_step % 50 == 0:
                gc.collect()

        # TODO: Should use flush rather than close
        self._exp_tracker.close()
        self.logger.info(f"Training finished in {time.time() - train_begin:.2f} seconds")

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
                f"`total_epoch` should be set for a Mapped dataset, but got {self._dataloader.dataset}"
            )
            self._total_step = len(self._dataloader) * cast(int, self._total_epoch)
        return self._total_step

    @property
    def cur_step(self) -> int:
        """Get the current training step.

        Returns:
            int: Current step number.
        """
        return self._cur_step

    def _init_logger(self, log_dir: Path):
        # Logging system maybe need better design
        logger = get_logger()
        logger.remove()
        logger.add(log_dir / f"rank{get_rank()}.log", format=log_format(), backtrace=True, catch=True)
        logger.add(sys.stderr, format=log_format(rank=get_rank()))
        return logger, log_dir

    def _init_tracker(self, exp_tracker: Literal["tensorboard", "jsonl"], log_dir: Path):
        writer = get_writer(writer_type=exp_tracker, log_dir=log_dir)
        return writer

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
                f"Found sp_size {sp_size}, world_size {self.world_size}."
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
        model_config: TransformerConfig | VisionComposeConfigProtocol,
        optim_config: OptimConfig,
        fsdp_config: FSDPConfig,
        resume_cfg: ResumeConfig,
        intra_layer_micro_batch: int = 1,
        strict: bool = True,
    ):
        """Build the training engine for the transformer model.

        Args:
            model_path (Path | None): Path to the model checkpoint or None for new initialization.
            model_config (TransformerConfig | VisionComposeConfigProtocol): Model configuration.
            optim_config (OptimConfig): Optimizer configuration.
            fsdp_config (FSDPConfig): FSDP configuration for distributed training.
            resume_cfg (ResumeConfig | None): Resume configuration for continuing training.
            intra_layer_micro_batch (int): Intra-layer micro batch size for gradient accumulation.
            strict (bool): Whether to strictly load model weights.

        Returns:
            TrainEngine: Initialized training engine.
        """
        if isinstance(model_config, VisionComposeConfigProtocol):
            engine = VisionComposeTrainEngine(
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
                intra_layer_micro_batch=intra_layer_micro_batch,
            )
        if model_path is not None and (model_config.dcp_ignore_frozen_params or resume_cfg.resume_from is None):
            engine.from_hf(hf_path=model_path, strict=strict)
        elif resume_cfg.resume_from is None:
            engine.init_model_weights()

        if model_path is not None:
            engine.model.set_hf(model_path)
        return engine

    def build_lr_scheduler(self, lr_cfg: LRConfig, scheduler_step: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the learning rate scheduler.

        Args:
            lr_cfg (LRConfig): Configuration for the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: Configured learning rate scheduler.
        """
        if lr_cfg.warmup_ratio < 1:
            warmup_steps = int(lr_cfg.warmup_ratio * scheduler_step)
        else:
            warmup_steps = int(lr_cfg.warmup_ratio)

        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        warmup_scheduler = LambdaLR(self._engine.optimizer, warmup_fn)

        scheduler: torch.optim.lr_scheduler.LRScheduler
        if lr_cfg.lr_type == "linear":
            scheduler = LinearLR(
                self._engine.optimizer,
                start_factor=1.0,
                end_factor=lr_cfg.lr_min / self._engine.optimizer.defaults["lr"],
                total_iters=scheduler_step - warmup_steps,
            )
        elif lr_cfg.lr_type == "cosine":
            scheduler = CosineAnnealingLR(
                self._engine.optimizer, T_max=scheduler_step - warmup_steps, eta_min=lr_cfg.lr_min
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

    def _maybe_check_health(self):
        if (
            (self._check_health_interval is not None and self.cur_step % self._check_health_interval == 0)
            or (self._checkpoint_interval is not None and self.cur_step % self._checkpoint_interval == 0)
            or (self._snapshot_interval is not None and self.cur_step % self._snapshot_interval == 0)
        ):
            if not check_health():
                raise RuntimeError("Health check failed, exit training")

    def _maybe_save(self, is_snapshot: bool = False) -> bool:
        ckp_interval = self._checkpoint_interval if not is_snapshot else self._snapshot_interval
        if ckp_interval is None:
            return False

        if ckp_interval == -1:  # only save at the end of training
            if self._cur_step != self.total_step:
                return False
        else:
            if self.cur_step % ckp_interval != 0 and (is_snapshot or self._cur_step != self.total_step):
                # if is_snapshot, only save at interval
                # else save at interval or at the end of training
                return False

        checkpoint_path = self._get_checkpoint_path(epoch=self._cur_epoch, step=self.cur_step, is_snapshot=is_snapshot)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        meta_path = self.work_dir / self._META_PATH

        optimizer_path = (
            checkpoint_path / self._SAVE_OPTIMIZER_DIR
            if self._resume_cfg.load_optimizer_states or self._resume_cfg.load_optimizer_args
            else None
        )
        model_path = checkpoint_path / self._SAVE_MODEL_DIR
        dataloader_path = checkpoint_path / self._SAVE_DATALOADER_DIR
        scheduler_path = checkpoint_path / self._SAVE_SCHEDULER_DIR
        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH

        # Save model and optimizer
        self._engine.save_dcp(
            model_dir=model_path,
            optimizer_dir=optimizer_path,
        )

        # Save dataloader
        self._save_dataloader(dataloader_path)

        # Save scheduler
        if self.rank == 0:
            lr_scheduler_state = self._lr_scheduler.state_dict()
            torch.save(lr_scheduler_state, scheduler_path)

        # Save trainer config
        if self._trainer_cfg is not None and self.rank == 0:
            # TODO: Maybe we need a better way to serialize and deserialize config, rather than using pickle
            config_path = checkpoint_path / "trainer_config.json"
            config_bin = checkpoint_path / "trainer_config.bin"
            with config_path.open("w") as f:
                f.write(self._trainer_cfg.model_dump_json(indent=2))

            with config_bin.open("wb") as f:
                pickle.dump(self._trainer_cfg, f)

        dist.barrier()

        # Save train state
        if self.rank == 0:
            with train_state_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "cur_step": self.cur_step,
                            "cur_epoch": self._cur_epoch,
                            "consumed_samples": self._consumed_samples,
                            "consumed_tokens": self._consumed_tokens,
                            "train_time_offset": self._train_time + self._train_time_offset,
                        }
                    )
                )

        # Update meta
        current_exp = self.meta.latest_exp
        ckp_list = current_exp.checkpoint_list if not is_snapshot else current_exp.snap_checkpoint_list
        ckp_list.append(str(checkpoint_path))
        current_exp.cur_step = self.cur_step
        current_exp.cur_epoch = self._cur_epoch
        current_exp.consumed_samples = self._consumed_samples
        current_exp.consumed_tokens = int(self._consumed_tokens)
        current_exp.history[-1]["end"] = self.cur_step

        # Delete checkpoints and update meta's checkpoint_list
        ckp_maxkeep = self._checkpoint_maxkeep if not is_snapshot else 1
        if ckp_maxkeep is not None and ckp_maxkeep > 0 and len(ckp_list) > ckp_maxkeep:
            ckp_pop_num = len(ckp_list) - ckp_maxkeep
            for _ in range(ckp_pop_num):
                deleted_ckp = ckp_list.pop(0)
                if self.rank == 0 and Path(deleted_ckp).exists():
                    rmtree(deleted_ckp)

        # Save meta, must after deleting checkpoints to ensure the checkpoint_list is updated in the meta file
        if self.rank == 0:
            with meta_path.open("w") as f:
                f.write(self.meta.model_dump_json(indent=2))

        dist.barrier()
        return True

    def _save_dataloader(self, dataloader_path: Path | str):
        _gathered_list = [None for _ in range(self.data_mesh["dp"].size())]
        dist.all_gather_object(_gathered_list, self._consumed_samples, group=self.data_mesh["dp"].get_group())
        global_consumed_samples = sum(_gathered_list)  # type: ignore[arg-type]

        if self.rank == 0:
            dataloader_state = self._dataloader.get_state_dict(global_consumed_samples)
            torch.save(dataloader_state, dataloader_path)

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
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def checkpoint_dir(self) -> Path:
        """Get the path to the latest checkpoint.

        Returns:
            Path | None: Path to the latest checkpoint or None if not available.
        """
        return self.exp_dir / self._CHECKPOINT_DIR

    @property
    def meta(self) -> XTunerMeta:
        """Get the XTuner metadata for tracking experiments.

        Returns:
            XTunerMeta: Experiment metadata tracker.
        """
        return self._meta

    def _data_iter(self):
        data_iter = iter(self._dataloader)
        while self._cur_step < self.total_step:
            # dist.breakpoint(skip=14)
            try:
                data = next(data_iter)
                self._consumed_samples += len(data)
            except StopIteration:
                self._cur_epoch += 1
                self._dataloader.set_epoch(self._cur_epoch)
                data_iter = iter(self._dataloader)
                data = next(data_iter)
            yield data

    def _get_checkpoint_path(self, epoch: int, step: int, is_snapshot: bool = False) -> Path:
        prefix = "snapshot-" if is_snapshot else "ckpt-"
        # TODO: epoch在不同rank间可能不一致，在这个问题下使用 epoch 会出错, 待解决。
        #       先使用 step 作为 checkpoint 的命名。
        # return self.checkpoint_dir / f"{prefix}epoch-{epoch}-step-{step}"
        return self.checkpoint_dir / f"{prefix}step-{step}"

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            logger.info("Setting deterministic algorithms")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)

    def _try_bind_numa(self):
        if str(DEVICE) != "cuda":
            logger.info("Current device is not cuda, skip numa binding.")
            return

        try:
            import numa
            from numa import memory, schedule

            numa_node_num = numa.info.get_max_node() + 1
            total_GPU_per_node = DEVICE_MODULE.device_count()

            # return while total_GPU_per_node is larger than numa_node_num or is not divisible by numa_node_num
            if total_GPU_per_node <= numa_node_num:
                return
            if total_GPU_per_node % numa_node_num != 0:
                return
            # return while the number of processes is smaller than one node GPUs num
            if self.world_size < total_GPU_per_node:
                return

            local_rank = self.rank % total_GPU_per_node
            # compute numa id for each locak rank
            per_numa = total_GPU_per_node // numa_node_num
            numa_id = local_rank // per_numa

            # bind numa node
            schedule.run_on_nodes(numa_id)
            memory.set_membind_nodes(numa_id)
        except Exception:
            logger.info(f"Rank: {self.rank} failed to bind process to numa node.")
            return  # try_bind_numa should not raise exception
        else:
            logger.info(f"Rank: {self.rank} success bind process to numa node: {numa_id}")

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

    def _init_xtuner_meta(self, work_dir: Path, auto_resume: bool) -> XTunerMeta:
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

        if auto_resume and meta.exps:
            latest_exp = meta.exps[-1]
            latest_exp_history = latest_exp.history[-1]

            begin = cast(int, latest_exp_history.get("end") or latest_exp_history["begin"])
            exp_dir = Path(latest_exp.exp_dir)
            git_dir = exp_dir / f"git-info-begin-{begin}"

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"
            if not git_dir.exists() and self.rank == 0:
                git_dir.mkdir(parents=True, exist_ok=True)
                commit = record_git_info(staged_path, unstaged_path)
                _commit_tmp = [commit]
            else:
                _commit_tmp = [None]  # type: ignore[list-item]
            dist.broadcast_object_list(_commit_tmp, src=0)
            commit = cast(str, _commit_tmp[0])
            dist.barrier()

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
            if self.rank == 0:
                commit = record_git_info(staged_path, unstaged_path)
                _commit_tmp = [commit]
            else:
                _commit_tmp = [None]  # type: ignore[list-item]
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
            )
            meta.exps.append(new_exp)
        return meta

    @contextmanager
    def _maybe_profiling(self):
        """Check if profiling is enabled and perform profiling if necessary."""
        if self._profile_step is not None and self._cur_step in self._profile_step:
            with contextlib.ExitStack() as stack:
                if self._profile_time:
                    time_dir = self.exp_dir / self._PROFILE_TIME_PATH / f"step-{self._cur_step}"
                    stack.enter_context(profiling_time(time_dir))

                if self._profile_memory:
                    memory_dir = self.exp_dir / self._PROFILE_MEMORY_PATH / f"step-{self._cur_step}"
                    stack.enter_context(profiling_memory(memory_dir))
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

        remaining_steps = self.total_step - self.cur_step
        avg_tokens_per_step = total_consumed_tokens / self.cur_step
        remaining_tokens = remaining_steps * avg_tokens_per_step
        eta_seconds = remaining_tokens / tgs
        eta_hms = str(timedelta(seconds=int(eta_seconds)))

        loss_log_list = [f"{k}: {v:.8f}" for k, v in loss_log.items()]
        loss_log_str = ", ".join(loss_log_list)

        max_memory = DEVICE_MODULE.max_memory_allocated()  # type: ignore[attr-defined]
        reserved_memory = DEVICE_MODULE.max_memory_reserved()  # type: ignore[attr-defined]

        self.logger.info(
            f"Epoch {self._cur_epoch} Step {self.cur_step}/{self.total_step} data_time: {data_time:.4f} lr: {lr:.6e} time: {step_time:.4f} "
            f"text_tokens: {step_consumed_tokens} "
            f"total_consumed_tokens: {total_consumed_tokens} "
            f"{loss_log_str} "
            f"grad_norm: {grad_norm:.8f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"tgs: {tgs:.1f} "
            f"e2e_tgs: {e2e_tgs:.1f} "
            f"eta: {eta_hms} "
        )

        log_scalars = {
            "lr": lr,
            "time/data_time": round(data_time, 4),
            "time/step_time": round(step_time, 4),
            "time/train_time": round(train_time, 4),
            "time/eta_seconds": round(eta_seconds, 1),
            "runtime_info/text_tokens": step_consumed_tokens,
            "runtime_info/total_consumed_tokens": total_consumed_tokens,
            "runtime_info/tgs": tgs,
            "runtime_info/e2e_tgs": e2e_tgs,
            "memory/max_memory_GB": round(max_memory / (1024**3), 3),
            "memory/reserved_memory_GB": round(reserved_memory / (1024**3), 3),
            "grad_norm": grad_norm,
        }
        log_scalars.update({f"loss/{k}": v for k, v in loss_log.items()})
        self._exp_tracker.add_scalars(tag_scalar_dict=log_scalars, global_step=self.cur_step)

        DEVICE_MODULE.reset_peak_memory_stats()  # type: ignore[attr-defined]

    def _maybe_save_hf(self):
        if self._hf_interval is None:
            return

        assert self._can_save_hf, "Model does not support saving in Huggingface format."

        if self.cur_step % self._hf_interval != 0 and self.cur_step != self.total_step:
            return

        save_hf_path = self.exp_dir / f"hf-{self.cur_step}"
        latest_hf_link = self.exp_dir / "hf-latest"

        self.meta.latest_exp.hf_checkpoint_list.append(str(save_hf_path))

        if self._hf_max_keep is not None and len(self.meta.latest_exp.hf_checkpoint_list) > self._hf_max_keep:
            deleted_hf_checkpoints = self.meta.latest_exp.hf_checkpoint_list[: -self._hf_max_keep]
            self.meta.latest_exp.hf_checkpoint_list = self.meta.latest_exp.hf_checkpoint_list[-self._hf_max_keep :]
            for hf_dir in deleted_hf_checkpoints:
                if self.rank == 0 and Path(hf_dir).exists():
                    rmtree(hf_dir)

        self._engine.save_hf(str(save_hf_path))
        if isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer.save_pretrained(str(save_hf_path))
        # 将 latest_hf_link 指向 save_hf_path
        if self.rank == 0:
            latest_hf_link.unlink(missing_ok=True)
            latest_hf_link.symlink_to(save_hf_path.absolute(), target_is_directory=True)

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

    def _resolve_work_dir(self, work_dir: Path | str | None) -> Path:
        if work_dir is None:
            work_dir = Path.cwd() / "work_dir"

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        return work_dir

    def _resolve_log_dir(self, log_dir: Path | str | None) -> Path:
        if log_dir is None:
            log_dir = self.exp_dir / self._DEFAULT_LOG_DIR
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        return log_dir

    def _resolve_config_conflicts(
        self,
        tokenizer: PreTrainedTokenizer,
        model_cfg: TransformerConfig | VisionComposeConfigProtocol,
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

        # Model's pad_token_id only affects the embedding module which acts specially for pad token.
        # Model's pad_token_id may be different from tokenizer's pad_token_id.
        # Note: Qwen3 Model's pad_token_id is None, which is different from Qwen tokenizer's pad_token_id.
        # if isinstance(model_cfg, VisionComposeConfigProtocol):
        #     if model_cfg.text_config.pad_token_id != pad_token_id:
        #         logger.warning(
        #             f"Model pad_token_id {model_cfg.text_config.pad_token_id} is different from tokenizer "
        #             f"pad_token_id {pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
        #         )
        #         model_cfg.text_config.pad_token_id = pad_token_id

        # elif model_cfg.pad_token_id != pad_token_id:
        #     logger.warning(
        #         f"Model pad_token_id {model_cfg.pad_token_id} is different from tokenizer pad_token_id "
        #         f"{pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
        #     )
        #     model_cfg.pad_token_id = pad_token_id

        if dataloader_cfg.pad_token_id is None:
            dataloader_cfg.pad_token_id = pad_token_id
        elif dataloader_cfg.pad_token_id != pad_token_id:
            logger.warning(
                f"Dataloader pad_token_id {dataloader_cfg.pad_token_id} is different from tokenizer "
                f"pad_token_id {pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
            )
            dataloader_cfg.pad_token_id = pad_token_id

    def _resolve_resume_cfg(self, resume_cfg: ResumeConfig):
        latest_checkpoint = self.meta.latest_exp.latest_checkpoint
        if latest_checkpoint is not None and resume_cfg.auto_resume:
            resume_cfg.resume_from = Path(latest_checkpoint)
        return resume_cfg

    def _resume(self):
        resume_cfg = self._resume_cfg

        if (resume_from := resume_cfg.resume_from) is None:
            logger.info("No checkpoint to resume from.")
            return

        if isinstance(resume_from, str):
            resume_from = Path(resume_from)
        logger.info(f"Resume from checkpoint: {resume_from}")

        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path {resume_from} does not exist.")

        model_path = resume_from / self._SAVE_MODEL_DIR
        optimizer_path = (
            resume_from / self._SAVE_OPTIMIZER_DIR
            if self._resume_cfg.load_optimizer_states or self._resume_cfg.load_optimizer_args
            else None
        )

        self._engine.load_dcp(
            model_dir=model_path,
            optimizer_dir=optimizer_path,
            load_states=self._resume_cfg.load_optimizer_states,
            load_args=self._resume_cfg.load_optimizer_args,
        )

        if resume_cfg.load_dataset:
            dataloader_path = resume_from / self._SAVE_DATALOADER_DIR
            self._resume_dataloader(dataloader_path)

        train_state_path = resume_from / self._SAVE_TRAIN_STATE_PATH

        with train_state_path.open("r") as f:
            train_state = json.load(f)

        self._cur_step = train_state["cur_step"]
        self._cur_epoch = train_state["cur_epoch"]
        self._consumed_samples = train_state["consumed_samples"]
        self._consumed_tokens = train_state["consumed_tokens"]
        self._train_time_offset = train_state["train_time_offset"]

        if resume_cfg.load_scheduler:
            scheduler_path = resume_from / self._SAVE_SCHEDULER_DIR
            if not scheduler_path.exists():
                raise FileNotFoundError(f"Scheduler path {scheduler_path} does not exist.")
            lr_scheduler_state = torch.load(scheduler_path, map_location=DEVICE)
            self._lr_scheduler.load_state_dict(lr_scheduler_state)
        else:
            assert self.total_step > self._cur_step
            scheduler_step = self.total_step - self._cur_step
            self._lr_scheduler = self.build_lr_scheduler(self._lr_cfg, scheduler_step)

    def _resume_dataloader(self, dataloader_path: Path):
        if not dataloader_path.exists():
            raise FileNotFoundError(f"Dataloader path {dataloader_path} does not exist.")
        dataloader_state = torch.load(dataloader_path, map_location=DEVICE)
        self._dataloader.load_state_dict(dataloader_state)

    def _setup_env(self):
        gc.disable()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        log_str = "\n============XTuner Training Environment============\n"
        env = {
            "XTUNER_DETERMINISTIC": os.getenv("XTUNER_DETERMINISTIC"),
            "XTUNER_FILE_OPEN_CONCURRENCY": os.getenv("XTUNER_FILE_OPEN_CONCURRENCY"),
            "XTUNER_TOKENIZE_CHUNK_SIZE": os.getenv("XTUNER_TOKENIZE_CHUNK_SIZE"),
            "XTUNER_TOKENIZE_WORKERS": os.getenv("XTUNER_TOKENIZE_WORKERS"),
            "XTUNER_ACTIVATION_OFFLOAD": os.getenv("XTUNER_ACTIVATION_OFFLOAD"),
            "XTUNER_USE_FA3": os.getenv("XTUNER_USE_FA3"),
            "XTUNER_DISPATCHER_DEBUG": os.getenv("XTUNER_DISPATCHER_DEBUG"),
            "XTUNER_ROUTER_DEBUG": os.getenv("XTUNER_ROUTER_DEBUG"),
            "XTUNER_DECORD_VIDEO_THREADS": os.getenv("XTUNER_DECORD_VIDEO_THREADS"),
            "XTUNER_USE_CUTLASS_GROUP_GEMM": os.getenv("XTUNER_USE_CUTLASS_GROUP_GEMM"),
            "GROUPED_GEMM_USE_CUTLASS": os.getenv("GROUPED_GEMM_USE_CUTLASS"),
            "XTUNER_USE_NATIVE_RMSNORM": os.getenv("XTUNER_USE_NATIVE_RMSNORM"),
        }

        for k, v in env.items():
            log_str += f"{k}: {v}\n"
        log_str += "=================================================="
        logger.info(log_str)
