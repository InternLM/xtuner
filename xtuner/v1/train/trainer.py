import contextlib
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import copy, rmtree
from typing import Sized, TypedDict, cast

import torch
import torch.distributed as dist
from mmengine import load
from mmengine.dist import get_rank, get_world_size
from mmengine.runner import set_random_seed
from pydantic import BaseModel
from torch.distributed import init_process_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing_extensions import NotRequired, Self

from transformers import AutoTokenizer
from xtuner.utils.device import get_device, get_torch_device
from xtuner.v1.config import DataloaderConfig, DatasetConfigList, FSDPConfig, LRConfig, OptimConfig
from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.config.trainer import ResumeConfig, TrainerConfig
from xtuner.v1.datasets.build import build_dataloader, build_datasets
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.interns1 import InternS1Config
from xtuner.v1.profiler import profilling_memory, profilling_time
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_logger,
    is_hf_model_path,
    log_format,
    record_git_info,
)


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
    META_PATH = ".xtuner"
    config: TrainerConfig | None
    profile_time_path = "profilling_time"
    profile_memory_path = "profilling_memory"

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
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        profile_step: int | None = None,
        profile_time: bool = True,
        profile_memory: bool = False,
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

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

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
        )
        self._lr_scheduler = self.build_lr_scheduler(lr_cfg)
        # TODO: (huanghaian) The impl of CELossContext should be decoupled with config
        if loss_ctx is None:
            self.loss_ctx = CELossContext()
        else:
            self.loss_ctx = loss_ctx
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
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            seed=config.seed,
            debug=config.debug,
        )
        self.config = config
        return self

    def fit(self):
        train_begin = time.time()
        time_before_get_data = time.time()
        for data_batch in self.data_iter():
            DEVICE_MODULE.reset_peak_memory_stats()

            time_before_train_step = time.time()
            data_time = time_before_train_step - time_before_get_data

            data_batch = self.loss_ctx.build_list_ctx(data_batch, self.data_mesh, DEVICE)
            with self.maybe_profilling():
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
            self.maybe_save()

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

    def maybe_save(self):
        ...
        # TODO: save latest information in `meta`

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def exp_dir(self) -> Path:
        return Path(self._meta.latest_exp.exp_dir)

    @property
    def meta(self) -> XTunerMeta:
        return self._meta

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        set_random_seed(seed)

    def _init_dist(self, backend: str):
        if not dist.is_initialized():
            init_process_group(backend=backend)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))

    def _init_xtuner_meta(self, work_dir: Path, resume: bool) -> XTunerMeta:
        if not work_dir.exists():
            if self.rank == 0:
                work_dir.mkdir(parents=True, exist_ok=True)

        meta_path = work_dir / self.META_PATH
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
            commit = record_git_info(staged_path, unstaged_path)
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
    def maybe_profilling(self):
        """Check if profiling is enabled and perform profiling if necessary."""
        if self._profile_step is not None and self._cur_step == self._profile_step:
            with contextlib.ExitStack() as stack:
                if self._profile_time:
                    time_dir = self.work_dir / self.profile_time_path / f"step-{self._cur_step}"
                    stack.enter_context(profilling_time(time_dir))

                if self._profile_memory:
                    memory_dir = self.work_dir / self.profile_memory_path / f"step-{self._cur_step}"
                    stack.enter_context(profilling_memory(memory_dir))
                yield
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

        self.logger.info(
            f"Step {self.cur_step}/{self.total_step} data_time: {data_time:.4f} lr: {lr:.6f} time: {step_time:.4f} "
            f"text_tokens: {step_consumed_tokens} "
            f"total_loss: {total_loss:.3f} "
            f"reduced_llm_loss: {reduced_llm_loss:.3f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"grad_norm: {grad_norm:.3f} "
            f"tgs: {tgs:.1f} "
            f"e2e_tgs: {e2e_tgs:.1f} "
        )

    def _maybe_save_hf(self):
        assert self._load_from_hf, (
            "Only support saving to Huggingface format when loading from Huggingface! "
            "You meet this error means `load_from` of trainer is not a Huggingface model path."
        )
        if self._hf_interval is None:
            return

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

        for file in cast(Path, self._load_from).iterdir():
            if file.suffix != ".safetensors":
                # Copy the model config and tokenizer files to the save path
                target_path = save_hf_path / file.name
                if self.rank == 0:
                    copy(file, target_path)

        meta_path = self.work_dir / self.META_PATH

        if self.rank == 0:
            with meta_path.open("w") as f:
                f.write(self.meta.model_dump_json(indent=2))
