import contextlib
import json
import math
import os
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Sequence,
    TypedDict,
    cast,
)


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

import numpy as np
import ray
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import NotRequired

from transformers import AutoTokenizer
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.engine.train_engine import TrainEngine, TrainStepInfo
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import BaseLossContext, CELossConfig, LogProbConfig
from xtuner.v1.loss.ce_loss import CELossContext, LMHeadLossContext
from xtuner.v1.loss.mtp_loss import MTPLossConfig, MTPLossContext
from xtuner.v1.loss.utils import sp_gather, sp_split
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.rl.loss import (
    BaseRLLossConfig,
    BaseRLLossContext,
    explained_variance,
    finalize_train_policy_metrics,
    kl_divergence_per_token,
    kl_penalty,
)
from xtuner.v1.rl.utils import SingleAcceleratorWorker
from xtuner.v1.rl.weight_update import WeightUpdater
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_device,
    get_logger,
    get_torch_device_module,
    ray_method,
    set_deterministic,
)
from xtuner.v1.utils.activation_offload import OffloadManager
from xtuner.v1.utils.fsdp import release_deferred_fsdp_all_gathers
from xtuner.v1.utils.nccl_process_group import resume_nccl_process_groups, suspend_nccl_process_groups

from ..advantage import BaseTokenLevelAdvantageConfig, TokenLevelAdvantageEstimator, normalize_advantages
from ..rollout_is import merge_rollout_is_metrics
from .critic_config import CriticWorkerConfig, KLRewardConfig


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def calculate_entropy(
    shifted_labels_list: Sequence[torch.Tensor],
    old_logprobs_list: Sequence[torch.Tensor | None],
    global_grad_tokens: torch.Tensor,
) -> torch.Tensor | None:
    if len(old_logprobs_list) == 0 or old_logprobs_list[0] is None:
        return None
    sum_entropy: torch.Tensor | None = None
    for i, shifted_labels in enumerate(shifted_labels_list):
        mask = shifted_labels != -100
        assert old_logprobs_list[i] is not None
        entropy = -(cast(torch.Tensor, old_logprobs_list[i]) * mask).sum()
        sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
    sum_entropy = cast(torch.Tensor, sum_entropy)
    dist.all_reduce(sum_entropy, op=dist.ReduceOp.SUM)
    avg_sum_entropy = sum_entropy / global_grad_tokens if global_grad_tokens > 0 else torch.tensor(0.0)
    return avg_sum_entropy


class WorkerConfig(BaseModel):
    """Training worker configuration for XTuner RL.

    Configuration for RL training workers managing model training, optimization,
    and distributed computing in reinforcement learning workflows.

    Args:
        model_cfg (TransformerConfig): Model architecture configuration.
        optim_cfg (OptimConfig): Optimizer configuration for training.
        loss_cfg (BaseRLLossConfig): Loss function configuration for RL training.
        lr_cfg (LRConfig): Learning rate scheduler configuration.
        fsdp_cfg (FSDPConfig): Fully Sharded Data Parallel configuration.
        load_from (str | Path): Path to load the main model from.
        optimizer_steps (int): Number of optimizer steps per training iteration. Defaults to 1.
        sp_size (int): Sequence parallel size for distributed training. Defaults to 1.
        pack_max_length (int): Maximum sequence length for data packing.
        ref_load_from (str | Path | None): Path to load reference model from.
            If None, uses same as load_from. Defaults to None.
        ref_model_fsdp_cfg (FSDPConfig | None): FSDP configuration for reference model.
            Defaults to None.
        log_dir (str | Path | None): Directory for training logs. Defaults to None.
        update_weight_bucket_size_in_gb (float): Bucket size used when syncing
            updated weights to rollout workers. Defaults to 0.5.
        seed (int | None): Training worker random seed. When None, the RL
            trainer seed is used. Defaults to None.
        offload_rollout_routed_experts (bool): Keep rollout routed experts on
            CPU and move each layer's slice to device during forward. Defaults to False.

    **Examples:**

    Example configuration for Basic worker::

        config = WorkerConfig(
            model_cfg=TransformerConfig(model_name="llama2-7b"),
            optim_cfg=OptimConfig(optimizer="adamw"),
            loss_cfg=GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}),
            lr_cfg=LRConfig(lr=1e-5),
            fsdp_cfg=FSDPConfig(),
            load_from="meta-llama/Llama-2-7b-hf",
            pack_max_length=2048,
        )

    .. note::
       When ``use_kl_loss=True`` in loss_cfg, a reference model will be loaded
       for KL divergence computation during training.
    """

    model_config = ConfigDict(title="Worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: BaseRLLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path  # TODO: 把 actor 和 ref 配置分离
    optimizer_steps: int = 1
    # Total learning-rate scheduler steps. One step happens per applied
    # optimizer update, i.e. roughly `total_train_steps * optimizer_steps`.
    # Defaults high enough that a `constant` schedule (the common RL choice)
    # never runs off the end of a decay curve.
    scheduler_steps: int = 1_000_000
    sp_size: int = 1
    pack_max_length: int
    ref_load_from: str | Path | None = None
    ref_model_fsdp_cfg: FSDPConfig | None = None
    log_dir: str | Path | None = None
    update_weight_bucket_size_in_gb: float = 0.5  # 512MB
    seed: None | int = None  # if None, use RLTrainer seed
    profile_step: list[int] | int | None = None  # 1-based global RL train_step ids to profile.
    profile_time: bool = True
    profile_memory: bool = False
    free_rollout_routed_experts_in_worker: bool = True  # 默认不需要用户配置
    offload_rollout_routed_experts: bool = False

    # PPO. Both stay unset for group-baseline algorithms, which keep the
    # original actor-only behavior.
    critic_cfg: CriticWorkerConfig | None = None
    advantage_cfg: BaseTokenLevelAdvantageConfig | None = None
    kl_reward_cfg: KLRewardConfig | None = None

    # sft config
    sft_dataloader_cfg: DataloaderConfig | None = None
    sft_global_batch_size: int = -1
    rollout_steps_per_sft: int = 1
    sft_loss_cfg: CELossConfig = CELossConfig()

    def build(self, placement_group: "PlacementGroup"):
        """Build training workers and controller from this config and placement
        group."""
        # import here to avoid circular import
        from xtuner.v1.rl.trainer.controller import TrainingController
        from xtuner.v1.rl.utils import AutoAcceleratorWorkers

        TrainingWorkerCls = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                }
            }
        )(TrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorkerCls, self, placement_group)
        ray.wait([w.ready.remote() for w in train_workers])
        return TrainingController(workers=train_workers)


class WorkerInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.LongTensor
    advantages: torch.Tensor
    rollout_logprobs: torch.Tensor | None
    # Per-token rewards for PPO, packed and zero-padded alongside the labels.
    # Absent for group-baseline algorithms, which need no value function.
    token_rewards: NotRequired[torch.Tensor]


class WorkerTrainLogItem(TypedDict, total=False):
    step_consumed_tokens: int
    efficient_attn_ratio: float
    grad_norm: float
    max_memory: float
    reserved_memory: float
    lr: float


class WorkerLogItem(TypedDict):
    train_entropy: float
    rollout_entropy: NotRequired[float]
    mismatch_metrics: NotRequired[dict[str, float]]
    rollout_is_metrics: NotRequired[dict[str, float]]
    train_metrics: List[WorkerTrainLogItem]
    sft_train_metrics: NotRequired[dict[str, float]]
    critic_train_metrics: NotRequired[List[WorkerTrainLogItem]]
    critic_metrics: NotRequired[dict[str, float]]
    kl_reward_mean: NotRequired[float]


class PPOPhase(str, Enum):
    """Which model currently owns accelerator memory in a PPO worker.

    Actor and critic are full models that cannot both be resident on a
    colocated setup, so the worker moves one in while the other is out. The
    phase is tracked explicitly because external callers (`offload_model`,
    `onload_model`, `save_hf`) would otherwise be able to fault the wrong model
    in mid-step.
    """

    ALL_OFFLOADED = "all_offloaded"
    CRITIC_TRAIN = "critic_train"
    ACTOR_TRAIN = "actor_train"
    ACTOR_READY = "actor_ready"


class TrainingWorker(SingleAcceleratorWorker):
    _SAVE_WEIGHTS_DIR = "weights"
    _SAVE_SCHEDULER_PATH = "lr_scheduler.pt"
    _SAVE_CRITIC_WEIGHTS_DIR = "critic_weights"
    _SAVE_CRITIC_SCHEDULER_PATH = "critic_lr_scheduler.pt"
    _SAVE_CRITIC_HF_DIR = "critic"
    _SAVE_SFT_DATALOADER_DIR = "sft_dataloader"
    _SAVE_SFT_TRAIN_STATE_PATH = "sft_train_state.json"

    def __init__(
        self,
        worker_cfg: WorkerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(worker_cfg, rank, master_addr, master_port, world_size, accelerator)
        self.config = cast(WorkerConfig, self.config)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        self.rank = rank

        log_dir = worker_cfg.log_dir
        self.log_dir = None
        if log_dir is not None:
            self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        else:
            self.logger = get_logger()

        self._set_deterministic()
        self._set_random_seed(worker_cfg.seed)

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]

        self._init_sft(worker_cfg)

        if not worker_cfg.fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = self._build_engine(worker_cfg)

        self._critic_engine: TrainEngine | None = None
        self._critic_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._critic_optimizer_updates = 0
        self._advantage_estimator: TokenLevelAdvantageEstimator | None = None
        self._ppo_phase = PPOPhase.ACTOR_READY
        if worker_cfg.critic_cfg is not None:
            self._init_ppo(worker_cfg)

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss or worker_cfg.kl_reward_cfg is not None:
            self._has_ref = True
            if worker_cfg.ref_load_from is None:
                worker_cfg.ref_load_from = worker_cfg.load_from
            self._ref_model = self._build_ref_model(
                worker_cfg.model_cfg, worker_cfg.ref_load_from, worker_cfg.ref_model_fsdp_cfg
            )

        self._optimizer_steps = worker_cfg.optimizer_steps
        self._lr_scheduler = worker_cfg.lr_cfg.build(self._engine.optimizer, worker_cfg.scheduler_steps)
        self._optimizer_updates = 0
        profile_step = worker_cfg.profile_step
        if isinstance(profile_step, int):
            profile_step = [profile_step]
        self._profile_step = set(profile_step or [])
        self._profile_time = worker_cfg.profile_time
        self._profile_memory = worker_cfg.profile_memory
        self._global_train_step = 0

        if worker_cfg.loss_cfg.chunk_size is not None:
            mode = "chunk"
        else:
            mode = "eager"
        self.logprob_cfg = LogProbConfig(chunk_size=worker_cfg.loss_cfg.chunk_size, mode=mode)
        self.mtp_config = None
        if isinstance(worker_cfg.model_cfg, BaseComposeConfig):
            if hasattr(worker_cfg.model_cfg.text_config, "mtp_config"):
                self.mtp_config = worker_cfg.model_cfg.text_config.mtp_config
        elif hasattr(worker_cfg.model_cfg, "mtp_config"):
            # 非 compose 模型的 mtp_config 直接放在了 model_cfg 中
            self.mtp_config = worker_cfg.model_cfg.mtp_config

        self.update_weighter = WeightUpdater(
            rank=self.rank,
            logger=self.logger,
            config=self.config,
            engine=self._engine,
        )

    @ray_method
    def suspend_train_nccl_process_groups(self) -> dict[str, object]:
        """Suspend train-side NCCL PG backends after weight sync and train
        offload."""

        include_default = (
            os.getenv(
                "XTUNER_SUSPEND_TRAIN_NCCL_INCLUDE_DEFAULT",
                os.getenv("XTUNER_DESTROY_TRAIN_NCCL_INCLUDE_DEFAULT", "0"),
            )
            == "1"
        )
        details = suspend_nccl_process_groups(include_default=include_default)
        errors = [str(detail["error"]) for detail in details if "error" in detail]
        DEVICE_MODULE.empty_cache()
        result = {
            "rank": self.rank,
            "suspended": sum("error" not in detail and not detail.get("skipped", False) for detail in details),
            "skipped": sum(bool(detail.get("skipped", False)) for detail in details),
            "details": details,
            "errors": errors,
        }
        return result

    @ray_method
    def resume_train_nccl_process_groups(self) -> dict[str, object]:
        """Resume train-side NCCL PG backends before FSDP2 runs again."""

        details = resume_nccl_process_groups()
        errors = [str(detail["error"]) for detail in details if "error" in detail]
        DEVICE_MODULE.empty_cache()
        result = {
            "rank": self.rank,
            "resumed": sum("error" not in detail for detail in details),
            "details": details,
            "errors": errors,
        }
        return result

    @ray_method
    def bind_rollout_weight_update(self, *args, **kwargs):
        return self.update_weighter.bind_rollout_weight_update(*args, **kwargs)

    @ray_method
    def weight_update(self, **kwargs):
        return self.update_weighter.weight_update(**kwargs)

    def _init_sft(self, worker_cfg: WorkerConfig):
        self._sft_dataloader_config = worker_cfg.sft_dataloader_cfg
        self._sft_dataloader: Dataloader | None = None
        self._sft_dataloader_iter: Iterable | None = None
        self._sft_loss_cfg: CELossConfig | None = None
        self._rollout_steps_per_sft = worker_cfg.rollout_steps_per_sft

        self._rollout_step = 0
        self._sft_cur_epoch = 0
        self._sft_total_consumed_tokens = 0

        if self._sft_dataloader_config is not None:
            assert worker_cfg.sft_global_batch_size > 0, "sft_global_batch_size must be greater than 0"
            assert worker_cfg.seed is not None, "seed must be set when sft_dataloader_config is not None"
            tokenizer = AutoTokenizer.from_pretrained(worker_cfg.load_from, trust_remote_code=True)
            self._sft_dataloader = self._sft_dataloader_config.build(
                tokenizer=tokenizer,
                dp_mesh=self.data_mesh["dp"],
                global_batch_size=worker_cfg.sft_global_batch_size,
                micro_batch_size=1,
                seed=worker_cfg.seed,
            )
            self.logger.info(f"Sft Dataloader len: {len(self._sft_dataloader)}")

            sft_loss_cfg = worker_cfg.sft_loss_cfg
            if worker_cfg.sft_loss_cfg is None:
                sft_loss_cfg = CELossConfig()
            self._sft_loss_cfg = sft_loss_cfg

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            self.logger.info("Setting deterministic algorithms of TrainingWorker.")
            set_deterministic()

    def _set_random_seed(self, seed: None | int):
        set_random_seed(seed)

    def _build_engine(self, worker_cfg: WorkerConfig) -> TrainEngine:
        engine = TrainEngine(  # type: ignore
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        if worker_cfg.load_from is not None:
            engine.from_hf(worker_cfg.load_from)

        if engine.model.compile_cfg is not None and self.rank == 0:
            self.logger.info(f"The `compile_cfg` of model is {json.dumps(engine.model.compile_cfg, indent=4)}")
        return engine

    def _init_ppo(self, worker_cfg: WorkerConfig) -> None:
        """Build the critic and validate the PPO configuration.

        Both models are parked on CPU afterwards; the phase machine faults in
        whichever one the current step needs.
        """
        critic_cfg = worker_cfg.critic_cfg
        assert critic_cfg is not None

        if worker_cfg.advantage_cfg is None:
            raise ValueError("advantage_cfg is required when critic_cfg is configured.")
        if worker_cfg.loss_cfg.use_kl_loss:
            # PPO folds the KL penalty into the token reward instead, so it
            # reaches the value targets through GAE. Allowing both would
            # penalize divergence twice.
            raise ValueError(
                "Loss-side KL is not supported with a PPO critic; use kl_reward_cfg instead, "
                "which lets the penalty flow through GAE into the value targets."
            )
        if self._sft_dataloader is not None:
            raise NotImplementedError("SFT interleaving is not supported with a PPO critic yet.")
        if worker_cfg.pack_max_length % worker_cfg.sp_size != 0:
            # GAE runs on the gathered full sequence. Keeping packs a multiple
            # of sp_size means the gather returns exactly pack_max_length and
            # no trailing shard padding has to be trimmed.
            raise ValueError(
                f"pack_max_length ({worker_cfg.pack_max_length}) must be divisible by "
                f"sp_size ({worker_cfg.sp_size}) when training a PPO critic."
            )

        self._advantage_estimator = worker_cfg.advantage_cfg.build()
        self._normalize_advantage = getattr(worker_cfg.advantage_cfg, "normalize_advantage", False)
        self._kl_reward_cfg = worker_cfg.kl_reward_cfg
        self._critic_warmup_steps = critic_cfg.warmup_steps

        # DCP must round-trip frozen params too, so a resumed critic keeps a
        # bit-exact backbone rather than silently re-initializing part of it.
        worker_cfg.model_cfg.dcp_ignore_frozen_params = False
        critic_cfg.model_cfg.dcp_ignore_frozen_params = False

        self._engine.put_model_to_device("cpu")
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()

        self._critic_engine = self._build_critic_engine(worker_cfg, critic_cfg)
        self._critic_scheduler = critic_cfg.lr_cfg.build(self._critic_engine.optimizer, critic_cfg.scheduler_steps)
        self._critic_engine.put_model_to_device("cpu")
        self._critic_engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._ppo_phase = PPOPhase.ALL_OFFLOADED

    def _build_critic_engine(self, worker_cfg: WorkerConfig, critic_cfg: CriticWorkerConfig) -> TrainEngine:
        if not critic_cfg.fsdp_cfg.torch_compile:
            critic_cfg.model_cfg.compile_cfg = False
        engine = TrainEngine(  # type: ignore[arg-type]
            optim_cfg=critic_cfg.optim_cfg,
            fsdp_cfg=critic_cfg.fsdp_cfg,
            model_cfg=critic_cfg.model_cfg,
        )

        if critic_cfg.load_mode == "init_from_actor":
            source = critic_cfg.load_from or worker_cfg.load_from
            if source is None:
                raise ValueError("An actor checkpoint path is required to initialize the critic backbone.")
            # The value model consumes the expected missing value-head key and
            # initializes just that tensor, so `strict` still catches a genuinely
            # incomplete backbone.
            engine.from_hf(source, strict=True)
        else:
            assert critic_cfg.load_from is not None
            weights_path = Path(critic_cfg.load_from)
            nested = weights_path / self._SAVE_CRITIC_WEIGHTS_DIR
            if nested.exists():
                weights_path = nested
            engine.load_dcp(weights_dir=weights_path, load_states=False, load_args=False, strict=True)
        return engine

    def _build_ref_model(
        self,
        ref_model_cfg: TransformerConfig | BaseComposeConfig,
        load_from: str | Path,
        ref_model_fsdp_cfg: FSDPConfig | None = None,
    ):
        # TODO: 需要重构，使得能更优雅的兼容 mllm
        model: BaseComposeModel | XtunerBaseModel
        with torch.device("meta"):
            model = ref_model_cfg.build()

        if isinstance(ref_model_cfg, BaseComposeConfig):
            assert ref_model_cfg.text_config.float8_cfg is None, "BaseComposeConfig does not support float8"
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model = model.fully_shard(ref_model_fsdp_cfg)
            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
        else:
            ref_model_cfg = cast(TransformerConfig, ref_model_cfg)
            if ref_model_cfg.float8_cfg is not None and ref_model_cfg.float8_cfg.enable_float8:
                float8_handler = Float8Handler(
                    scaling_granularity_gemm=ref_model_cfg.float8_cfg.scaling_granularity_gemm,
                    scaling_granularity_grouped_gemm=ref_model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
                )
            else:
                float8_handler = None
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model = model.fully_shard(ref_model_fsdp_cfg)  # type: ignore

            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
            if float8_handler is not None:
                # As the ref model is not updated, we only compute params' scales once
                float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)  # type: ignore
        model.to_device("cpu")  # type: ignore
        DEVICE_MODULE.empty_cache()
        return model

    def _init_data_mesh(
        self,
        sp_size: int,
    ):
        world_size = dist.get_world_size()
        if world_size % sp_size != 0:
            raise ParallelConfigException(
                f"Found sp_size {sp_size}, world_size {world_size}."
                "sequence parallel size must be a divisor of world size."
            )
        dp_size = world_size // sp_size

        # TODO: fsdp_config could be None
        device = str(DEVICE) if not self.config.fsdp_cfg.cpu_offload else "cpu"

        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        return data_mesh

    def _step_optimizer_and_scheduler(self, grad_norm: torch.Tensor) -> bool:
        """Step the optimizer, advancing the LR schedule only if it applied.

        A step skipped for a non-finite or over-threshold gradient norm must not
        consume a scheduler step, otherwise the learning rate would decay on
        updates that never happened.

        Args:
            grad_norm (torch.Tensor): The gradient norm for this step.

        Returns:
            bool: Whether the optimizer update was applied.
        """
        applied = self._engine.optimizer_step_will_apply(grad_norm)
        self._engine.step_optimizer(grad_norm)
        if applied:
            self._lr_scheduler.step()
            self._optimizer_updates += 1
        return applied

    @property
    def current_lr(self) -> float:
        """The learning rate currently in effect."""
        return float(self._engine.optimizer.param_groups[0]["lr"])

    @property
    def is_ppo(self) -> bool:
        """Whether this worker trains a PPO critic alongside the actor."""
        return self._critic_engine is not None

    def _set_ppo_phase(self, phase: PPOPhase) -> None:
        self._ppo_phase = phase
        self.logger.info(
            f"PPO phase={phase.value}, allocated={DEVICE_MODULE.memory_allocated() / (1024**3):.2f} GiB, "
            f"reserved={DEVICE_MODULE.memory_reserved() / (1024**3):.2f} GiB"
        )

    def _onload_critic(self) -> None:
        """Swap the actor out and the critic in."""
        if self._ppo_phase not in (PPOPhase.ALL_OFFLOADED, PPOPhase.ACTOR_READY):
            raise RuntimeError(f"Cannot onload the critic from phase {self._ppo_phase.value}.")
        assert self._critic_engine is not None
        self._engine.put_model_to_device("cpu")
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._critic_engine.put_model_to_device(DEVICE)
        self._set_ppo_phase(PPOPhase.CRITIC_TRAIN)

    def _offload_critic(self) -> None:
        if self._ppo_phase != PPOPhase.CRITIC_TRAIN:
            raise RuntimeError(f"Cannot offload the critic from phase {self._ppo_phase.value}.")
        assert self._critic_engine is not None
        self._critic_engine.put_optimizer_to_device("cpu")
        self._critic_engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)

    def _onload_actor(self) -> None:
        """Fault the actor in for training.

        `ACTOR_READY` means the actor is already resident from a previous
        phase (a warmup step skips the actor update, or the KL phase re-enters
        the actor on the next rollout), so it does not need another device
        transfer -- only a phase stamp. `ALL_OFFLOADED` does the transfer.
        """
        if self._ppo_phase not in (PPOPhase.ALL_OFFLOADED, PPOPhase.ACTOR_READY):
            raise RuntimeError(f"Cannot onload the actor from phase {self._ppo_phase.value}.")
        if self._ppo_phase == PPOPhase.ALL_OFFLOADED:
            self._engine.put_model_to_device(DEVICE)
        self._set_ppo_phase(PPOPhase.ACTOR_TRAIN)

    def _step_critic_optimizer_and_scheduler(self, grad_norm: torch.Tensor) -> bool:
        """Step the critic optimizer, advancing its schedule only if applied."""
        assert self._critic_engine is not None and self._critic_scheduler is not None
        applied = self._critic_engine.optimizer_step_will_apply(grad_norm)
        self._critic_engine.step_optimizer(grad_norm)
        if applied:
            self._critic_scheduler.step()
            self._critic_optimizer_updates += 1
        return applied

    def compute_actor_logprobs(
        self,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # precompute float8 dynamic scale only once
        self._engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        old_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
            assert loss_ctx is not None
            output = self._engine.forward_only(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
            old_logprobs_list.append(output["loss"])
        return old_logprobs_list

    def compute_ref_logprobs(
        self, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert self._has_ref
        self._ref_model.to_device(DEVICE)
        ref_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            with torch.no_grad():
                loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
                assert loss_ctx is not None
                ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                ref_logprobs_list.append(ref_output["loss"])
        self._ref_model.to_device("cpu")
        return ref_logprobs_list

    def _add_rollout_routed_experts(
        self, seq_ctx: SequenceContext, rollout_routed_experts: torch.Tensor | list[torch.Tensor | ray.ObjectRef]
    ):
        language_cfg = (
            self.config.model_cfg.text_config
            if isinstance(self.config.model_cfg, BaseComposeConfig)
            else self.config.model_cfg
        )

        to_free_routed_expert_refs: list[ray.ObjectRef] = []
        if isinstance(rollout_routed_experts, list):
            # list[n,l,e]
            out_rollout_routed_expert = []
            for rollout_routed_expert in rollout_routed_experts:
                if isinstance(rollout_routed_expert, torch.Tensor):
                    rollout_routed_experts_tensor = torch.randint(
                        low=0,
                        high=language_cfg.n_routed_experts,
                        size=(
                            rollout_routed_expert.size(0),
                            language_cfg.num_hidden_layers,
                            language_cfg.num_experts_per_tok,
                        ),
                    )
                    out_rollout_routed_expert.append(rollout_routed_experts_tensor)
                else:
                    rollout_routed_expert_refs = rollout_routed_expert
                    if isinstance(rollout_routed_expert_refs, ray.ObjectRef):
                        rollout_routed_expert = ray.get(rollout_routed_expert_refs)
                    elif isinstance(rollout_routed_expert_refs, list):
                        _rollout_routed_expert = []
                        _rollout_routed_expert_refs = rollout_routed_expert_refs
                        for rollout_routed_expert_ref in rollout_routed_expert_refs:
                            rollout_routed_expert = ray.get(rollout_routed_expert_ref)  # np
                            _rollout_routed_expert.append(rollout_routed_expert)
                        rollout_routed_expert = np.concatenate(_rollout_routed_expert, axis=0)[1:, ...]
                        rollout_routed_expert_refs = _rollout_routed_expert_refs
                    else:
                        raise ValueError(
                            f"Invalid rollout_routed_expert_refs type: {type(rollout_routed_expert_refs)}"
                        )
                    # Some agent loops export routed-expert refs from the rollout trace store.
                    # Those refs may be shared by multiple trainable segments and replicated
                    # train workers, so they must be released by the trainer after all workers
                    # finish consuming the batch.
                    if self.config.free_rollout_routed_experts_in_worker:
                        if self.sp_mesh is None or self.sp_mesh.size() == 1:
                            ray.internal.free(rollout_routed_expert_refs, local_only=False)
                        else:
                            if self.sp_mesh.get_local_rank() == 0:
                                # only free once of sp mesh
                                to_free_routed_expert_refs.append(rollout_routed_expert_refs)
                    rollout_routed_expert = torch.as_tensor(rollout_routed_expert, dtype=torch.long)
                    rollout_routed_expert = rollout_routed_expert.reshape(
                        -1, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok
                    )
                    out_rollout_routed_expert.append(rollout_routed_expert)

            seq_ctx.rollout_routed_experts = torch.cat(out_rollout_routed_expert, dim=0)  # max_len,l,e
        else:
            assert isinstance(rollout_routed_experts, torch.Tensor), (
                f"padding experts should be a dummy tensor, bug got {type(rollout_routed_experts)}"
            )
            rollout_routed_experts_tensor = torch.randint(
                low=0,
                high=language_cfg.n_routed_experts,
                size=(
                    self.config.pack_max_length,
                    language_cfg.num_hidden_layers,
                    language_cfg.num_experts_per_tok,
                ),
            )
            seq_ctx.rollout_routed_experts = rollout_routed_experts_tensor

        assert seq_ctx.input_ids is not None, "input_ids is None"
        assert seq_ctx.rollout_routed_experts.size(0) == seq_ctx.input_ids.size(1), (
            f"rollout_routed_experts.size(0) {seq_ctx.rollout_routed_experts.size(0)} != input_ids.size(1) {seq_ctx.input_ids.size(1)}"
        )

        if self.config.free_rollout_routed_experts_in_worker and self.sp_mesh is not None and self.sp_mesh.size() > 1:
            dist.barrier()
            for free_routed_expert_refs in to_free_routed_expert_refs:
                ray.internal.free(free_routed_expert_refs, local_only=False)
            del to_free_routed_expert_refs

    @contextmanager
    def _maybe_profiling(self, global_train_step: int, phase: str):
        if global_train_step not in self._profile_step:
            yield
            return

        if self.log_dir is not None:
            profile_home = self.log_dir.parent
        else:
            profile_home = Path(os.environ.get("WORK_DIR", "."))

        with contextlib.ExitStack() as stack:
            if self._profile_time:
                time_dir = profile_home / "profiling_time" / phase / f"global-step-{global_train_step}"
                stack.enter_context(profiling_time(time_dir))
            if self._profile_memory:
                memory_dir = profile_home / "profiling_memory" / phase / f"global-step-{global_train_step}"
                stack.enter_context(profiling_memory(memory_dir))
            yield

    @ray_method
    def fit(self, data_batches: list[WorkerInputItem], rollout_idx: int) -> WorkerLogItem:
        # NOTE: sglang会清除logger handle, 重新创建
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        if self.is_ppo:
            return self._fit_ppo(data_batches, rollout_idx)
        loss_cfg: BaseRLLossConfig = self.config.loss_cfg
        num_batches = len(data_batches)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps)
        if num_batches < self._optimizer_steps:
            self.logger.info(
                f"Optimizer only step once because num_batches {num_batches} < optimizer_steps {self._optimizer_steps}."
            )

        # Update seq_ctx: pixel_values, rollout_routed_experts
        # Init loss_ctx: shifted_labels, advantages, rollout_logprobs
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_list: list[BaseRLLossContext] = []
        mtp_loss_ctx_list: list[list[MTPLossContext]] = []
        prepare_inputs_begin = time.perf_counter()
        for data in data_batches:
            # update seq_ctx
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None:
                if not isinstance(pixel_values, np.ndarray):
                    assert isinstance(pixel_values, list), (
                        f"pixel_values should be list of tensor, got {type(pixel_values)}"
                    )
                    pixel_values = ray.get(list(pixel_values))
                    pixel_values = [torch.as_tensor(pixel_value) for pixel_value in pixel_values]
                    pixel_values = torch.cat(pixel_values, dim=0)
                    seq_ctx.pixel_values = pixel_values
                else:
                    raise NotImplementedError("The case where pixel_values is a numpy array is not implemented yet.")

            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                self._add_rollout_routed_experts(seq_ctx, rollout_routed_experts)

            seq_ctx.offload_rollout_routed_experts = self.config.offload_rollout_routed_experts
            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(self.sp_mesh)

            # init loss_ctx
            shifted_labels = data["shifted_labels"].to(DEVICE)
            advantages = data["advantages"].to(DEVICE)
            rollout_logprobs = data.get("rollout_logprobs", None)
            rollout_logprobs = rollout_logprobs.to(DEVICE) if rollout_logprobs is not None else None
            loss_ctx = loss_cfg.build(
                data={
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                },
                sp_mesh=self.sp_mesh,
            )

            seq_ctx_list.append(seq_ctx)
            assert loss_ctx is not None
            loss_ctx_list.append(loss_ctx)
            if self.mtp_config is not None:
                mtp_loss_ctxs_per_batch: list[MTPLossContext] = []
                for mtp_idx in range(self.mtp_config.num_layers):
                    mtp_loss_cfg = MTPLossConfig(
                        **loss_cfg.model_dump(include={"mode", "chunk_size"}),
                        mtp_depth=mtp_idx + 1,
                        detach_mtp_lm_head_weight=self.mtp_config.detach_mtp_lm_head_weight,
                    )
                    mtp_ctx = mtp_loss_cfg.build(
                        data={
                            "shifted_labels": shifted_labels,
                            "seq_ctx": seq_ctx,
                            "logprobs": rollout_logprobs,
                        },
                        sp_mesh=self.sp_mesh,
                    )
                    if mtp_ctx is not None:
                        mtp_loss_ctxs_per_batch.append(mtp_ctx)
                mtp_loss_ctx_list.append(mtp_loss_ctxs_per_batch)
        self.logger.debug(
            f"Rank{self.rank} Rollout {rollout_idx} prepare_inputs elapsed="
            f"{time.perf_counter() - prepare_inputs_begin:.4f}s"
        )

        del data_batches

        # When sp_mesh.size() > 1, get the sp_split shifted_labels and rollout_logprobs
        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        rollout_logprobs_list = [loss_ctx.loss_kwargs.rollout_logprobs for loss_ctx in loss_ctx_list]

        # compute old logprobs
        old_logprobs_list = self.compute_actor_logprobs(seq_ctx_list, shifted_labels_list)
        for old_logprobs, loss_ctx in zip(old_logprobs_list, loss_ctx_list):
            loss_ctx.loss_kwargs.old_logprobs = old_logprobs

        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": [], "sft_train_metrics": {}}
        logger_msg = f"Rollout {rollout_idx}: "

        # compute entropy
        rank_grad_tokens: torch.Tensor | None = None
        for shifted_labels in shifted_labels_list:
            mask = shifted_labels != -100
            grad_tokens = mask.sum()
            rank_grad_tokens = grad_tokens if rank_grad_tokens is None else rank_grad_tokens + grad_tokens
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        avg_sum_entropy = calculate_entropy(shifted_labels_list, old_logprobs_list, global_grad_tokens)
        avg_rollout_entropy = calculate_entropy(shifted_labels_list, rollout_logprobs_list, global_grad_tokens)

        assert avg_sum_entropy is not None
        worker_log_item["train_entropy"] = avg_sum_entropy.item()
        logger_msg += f"avg entropy: {avg_sum_entropy:.4f}"
        if avg_rollout_entropy is not None:
            worker_log_item["rollout_entropy"] = avg_rollout_entropy.item()
            logger_msg += f", avg rollout entropy: {avg_rollout_entropy:.4f}"

        # compute rollout importance sampling metrics
        all_rollout_is_metrics = []
        all_mismatch_metrics = []
        for i, loss_ctx in enumerate(loss_ctx_list):
            if loss_ctx.loss_kwargs.rollout_logprobs is not None:
                # calculate importance sampling weights
                num_tokens = seq_ctx_list[i].seq_lens_q
                mismatch_metrics, rollout_is_metrics = loss_ctx.compute_rollout_is(self.sp_mesh, num_tokens)
                all_rollout_is_metrics.append(rollout_is_metrics)
                all_mismatch_metrics.append(mismatch_metrics)

        if len(all_mismatch_metrics) > 0:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if len(mismatch_metrics) > 0:
                worker_log_item["mismatch_metrics"] = mismatch_metrics
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"

        if len(all_rollout_is_metrics) > 0:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if len(rollout_is_metrics) > 0:
                worker_log_item["rollout_is_metrics"] = rollout_is_metrics
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"

        if self.rank == 0:
            self.logger.info(logger_msg)

        only_calc_mismatch_ratio = os.environ.get("ONLY_CALC_MISMATCH_RATIO", "0") == "1"
        if only_calc_mismatch_ratio:
            # This early-return path skips the MTP forward. FSDP may have
            # prefetched the MTP module already, leaving its all-gather result
            # cached until we release it explicitly. The normal training branch
            # can leave the same residue, but it is outside the peak-memory
            # path and can be ignored.
            self._release_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/only_calc_mismatch")
            DEVICE_MODULE.empty_cache()
            return worker_log_item

        # compute reference logprobs
        ref_logprobs_list: list[torch.Tensor] | None = None
        if self._has_ref:
            ref_logprobs_list = self.compute_ref_logprobs(seq_ctx_list, shifted_labels_list)

            for i, loss_ctx in enumerate(loss_ctx_list):
                loss_ctx.loss_kwargs.ref_logprobs = ref_logprobs_list[i]

            kl_div_sum: torch.Tensor | None = None
            for i, shifted_labels in enumerate(shifted_labels_list):
                mask = shifted_labels != -100
                kl_div = kl_penalty(
                    cast(torch.Tensor, old_logprobs_list[i]),
                    cast(torch.Tensor, ref_logprobs_list[i]),
                    loss_weights=mask,
                    kl_penalty="low_var_kl",
                )
                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div

            kl_div_sum = cast(torch.Tensor, kl_div_sum)
            dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
            avg_kl_div = kl_div_sum / global_grad_tokens if global_grad_tokens > 0 else 0
            self.logger.info(f"Rollout {rollout_idx}: avg KL divergence: {avg_kl_div:.4f}")

        worker_log_item["train_metrics"].extend(
            self._train_actor(
                seq_ctx_list,
                loss_ctx_list,
                mtp_loss_ctx_list,
                rollout_idx=rollout_idx,
                iters_per_step=iters_per_step,
            )
        )

        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            train_step_info = self._fit_sft()
            engine_logs_info = train_step_info["logs_info"]
            worker_log_item["sft_train_metrics"] = {
                **engine_logs_info,
                **train_step_info["extra_info"].get(),
                "efficient_attn_ratio": train_step_info["efficient_attn_ratio"],
            }

        return worker_log_item

    def _train_actor(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[BaseRLLossContext],
        mtp_loss_ctx_list: list[list[MTPLossContext]],
        *,
        rollout_idx: int,
        iters_per_step: int,
    ) -> list[WorkerTrainLogItem]:
        """Run the actor's optimizer steps over prepared loss contexts.

        Splits the batch into `iters_per_step`-sized chunks, calibrates each
        chunk's loss globally, then runs forward, backward and one optimizer
        update per chunk.

        Shared by the group-baseline path and the PPO path, which differ only in
        how advantages reach `loss_ctx_list`.

        Args:
            seq_ctx_list (list[SequenceContext]): Per-pack sequence contexts.
            loss_ctx_list (list[BaseRLLossContext]): Per-pack policy loss
                contexts, with `old_logprobs` already attached.
            mtp_loss_ctx_list (list[list[MTPLossContext]]): Per-pack MTP loss
                contexts, outer index batch and inner index depth. Empty when
                MTP is disabled.
            rollout_idx (int): Current rollout index, for logging.
            iters_per_step (int): Micro-batches per optimizer update.

        Returns:
            list[WorkerTrainLogItem]: One log item per optimizer step.
        """
        loss_cfg: BaseRLLossConfig = self.config.loss_cfg
        train_metrics: list[WorkerTrainLogItem] = []

        # compute batched loss context
        batched_loss_ctx_list: list[BaseRLLossContext] = []
        batched_mtp_loss_ctx_list: list[list[MTPLossContext]] = []
        LossContext = loss_cfg.loss_ctx_cls
        for i in range(0, len(loss_ctx_list), iters_per_step):
            batches_loss_ctx = loss_ctx_list[i : i + iters_per_step]
            batched_loss_ctx_list.extend(
                LossContext.build_batches(batches_loss_ctx)  # type: ignore[arg-type]
            )

            if self.mtp_config is not None:
                batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
                cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in batches_seq_ctx]
                # mtp_loss_ctx_list: list[list[MTPLossContext]], outer=batch, inner=mtp_depth
                num_mtp_depths = len(mtp_loss_ctx_list[0]) if mtp_loss_ctx_list else 0
                for mtp_idx in range(num_mtp_depths):
                    depth_mtp_loss_ctxs: list[LMHeadLossContext] = [
                        mtp_loss_ctx_list[j][mtp_idx]
                        for j in range(i, min(i + iters_per_step, len(mtp_loss_ctx_list)))
                    ]
                    batched_mtp_depth_ctxs = cast(
                        list[MTPLossContext],
                        MTPLossContext.build_batches(
                            depth_mtp_loss_ctxs,
                            cu_seq_lens_list=cu_seq_lens_list,
                            sp_mesh=self.sp_mesh,
                        ),
                    )
                    # Append each depth's batched ctx to the corresponding batch index
                    for batch_offset, mtp_ctx in enumerate(batched_mtp_depth_ctxs):
                        global_batch_idx = i + batch_offset
                        if global_batch_idx >= len(batched_mtp_loss_ctx_list):
                            batched_mtp_loss_ctx_list.append([mtp_ctx])
                        else:
                            batched_mtp_loss_ctx_list[global_batch_idx].append(mtp_ctx)

        # train optimizer steps
        for i in range(0, len(seq_ctx_list), iters_per_step):
            global_train_step = self._global_train_step + 1
            batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
            batches_loss_ctx = batched_loss_ctx_list[i : i + iters_per_step]

            engine_input = [
                ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                for seq_ctx, loss_ctx in zip(batches_seq_ctx, batches_loss_ctx)
            ]

            if self.mtp_config is not None:
                batches_mtp_loss_ctxs = batched_mtp_loss_ctx_list[i : i + iters_per_step]
                engine_input = [
                    ModelItem(
                        seq_ctx=seq_ctx,
                        loss_ctx=cast(
                            dict[str, BaseLossContext],
                            {"mtp": mtp_loss_ctx_depths, "lm": loss_ctx},
                        ),
                    )
                    for seq_ctx, loss_ctx, mtp_loss_ctx_depths in zip(
                        batches_seq_ctx, batches_loss_ctx, batches_mtp_loss_ctxs
                    )
                ]

            train_step_begin = time.perf_counter()
            with self._maybe_profiling(global_train_step, "train_step"):
                train_step_info = self._engine.train_step(
                    data_batches=engine_input,
                )
                OffloadManager().clear()
            self.logger.debug(
                f"Rank{self.rank} Rollout {rollout_idx} GlobalStep {global_train_step} "
                f"train_step[{i}].engine_train_step elapsed={time.perf_counter() - train_step_begin:.4f}s"
            )
            grad_norm = self._engine.clip_grad_norm()
            self._step_optimizer_and_scheduler(grad_norm)

            engine_logs_info = cast(dict[str, float], train_step_info.pop("logs_info"))  # type: ignore[misc]
            engine_extra_info = train_step_info.pop("extra_info")  # type: ignore[misc]

            if isinstance(engine_extra_info, ModelForwardExtraLogInfo):
                extra_info_dict = engine_extra_info.get()
            else:
                extra_info_dict = cast(dict, engine_extra_info)

            extra_info_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in extra_info_dict.items()
                if isinstance(v, (torch.Tensor, int, float))
            }
            extra_info_dict = finalize_train_policy_metrics(extra_info_dict, DEVICE)
            train_step_info.pop("total_loss")  # type: ignore[misc]
            max_memory = DEVICE_MODULE.max_memory_allocated() / (1024**3)  # type: ignore[attr-defined]
            reserved_memory = DEVICE_MODULE.max_memory_reserved() / (1024**3)  # type: ignore[attr-defined]

            train_log_item = WorkerTrainLogItem(
                **engine_logs_info,  # type: ignore[typeddict-item]
                **train_step_info,
                **extra_info_dict,
                grad_norm=grad_norm.item(),
                max_memory=max_memory,
                reserved_memory=reserved_memory,
                lr=self.current_lr,
            )
            train_metrics.append(train_log_item)

            # Extract logs_info for logging
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in train_log_item.items()
                if not key.startswith("reduced_train_policy_") and key != "max_ratio"
            )
            log_str = f"Rank{self.rank} Rollout {rollout_idx} Step {i}: " + log_str
            self.logger.info(log_str)
            self._global_train_step = global_train_step
        return train_metrics

    def _fit_ppo(self, data_batches: list[WorkerInputItem], rollout_idx: int) -> WorkerLogItem:
        """Run one PPO step: critic forward, GAE, critic training, actor training.

        Actor and critic never occupy the accelerator at the same time, so the
        step is strictly sequenced: the critic is faulted in to produce frozen
        values and consume them for its own regression, then swapped out before
        the actor trains on the resulting advantages.

        Args:
            data_batches (list[WorkerInputItem]): Packed batches for this rank,
                each carrying `token_rewards`.
            rollout_idx (int): Current rollout index.

        Returns:
            WorkerLogItem: Actor and critic metrics for this step.
        """
        critic_cfg = self.config.critic_cfg
        critic_engine = self._critic_engine
        estimator = self._advantage_estimator
        assert critic_cfg is not None and critic_engine is not None and estimator is not None

        prepared = self._prepare_ppo_inputs(data_batches)
        del data_batches

        worker_log_item: WorkerLogItem = {
            "train_entropy": 0.0,
            "train_metrics": [],
            "critic_train_metrics": [],
            "sft_train_metrics": {},
        }

        in_warmup = self._rollout_step < self._critic_warmup_steps
        if in_warmup:
            self.logger.info(
                f"Critic warmup {self._rollout_step + 1}/{self._critic_warmup_steps}: "
                "training the critic only, the actor is frozen this step."
            )

        # --- KL reward phase ----------------------------------------------
        # The penalty must be folded into the reward before GAE runs, so that
        # it reaches the value targets rather than only the policy gradient.
        if self._kl_reward_cfg is not None:
            self._apply_kl_reward(prepared, worker_log_item)

        # --- Critic phase -------------------------------------------------
        self._onload_critic()
        # The critic routes independently, so it must not consume the actor's
        # recorded rollout expert choices.
        critic_seq_ctx_list = [seq_ctx.copy(rollout_routed_experts=None) for seq_ctx in prepared["seq_ctx_list"]]
        if self.sp_mesh.size() > 1:
            critic_seq_ctx_list = [seq_ctx.split(self.sp_mesh) for seq_ctx in critic_seq_ctx_list]

        old_values_list = self._compute_critic_values(critic_seq_ctx_list)
        advantages_list, returns_list = self._compute_ppo_targets(
            old_values_list,
            prepared["token_rewards_list"],
            prepared["action_mask_list"],
            prepared["cu_seq_lens_list"],
        )

        critic_loss_ctx_list: list[BaseLossContext] = []
        for returns, action_mask, old_values in zip(returns_list, prepared["action_mask_list"], old_values_list):
            critic_loss_ctx = critic_cfg.loss_cfg.build(
                data={"returns": returns, "value_mask": action_mask, "old_values": old_values},
                sp_mesh=self.sp_mesh,
            )
            if critic_loss_ctx is None:
                raise RuntimeError("The critic loss config did not build a loss context.")
            critic_loss_ctx_list.append(critic_loss_ctx)

        critic_logs, critic_metrics = self._train_critic(
            critic_seq_ctx_list, critic_loss_ctx_list, rollout_idx=rollout_idx
        )
        worker_log_item["critic_train_metrics"] = critic_logs
        if critic_metrics:
            worker_log_item["critic_metrics"] = critic_metrics

        # Release every critic-only tensor before the actor is faulted in.
        del critic_loss_ctx_list, critic_seq_ctx_list, old_values_list, returns_list
        DEVICE_MODULE.empty_cache()
        self._offload_critic()
        self._onload_actor()

        # --- Actor phase --------------------------------------------------
        if in_warmup:
            # Skip the policy update entirely: no forward, no optimizer step,
            # and no scheduler step, so the actor's schedule stays aligned with
            # the updates it actually receives.
            self.logger.info("Critic warmup: skipping the actor update.")
        else:
            worker_log_item["train_metrics"] = self._train_actor_on_advantages(
                prepared["seq_ctx_list"],
                prepared["shifted_labels_list"],
                advantages_list,
                prepared["rollout_logprobs_list"],
                rollout_idx=rollout_idx,
                worker_log_item=worker_log_item,
            )

        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._set_ppo_phase(PPOPhase.ACTOR_READY)
        self._rollout_step += 1
        return worker_log_item

    def _apply_kl_reward(self, prepared: dict[str, list], worker_log_item: WorkerLogItem) -> None:
        """Subtract a per-token KL penalty from the token rewards, in place.

        Runs before GAE so the penalty is discounted and bootstrapped like any
        other reward, and so the critic learns to predict it. Adding KL to the
        loss instead would leave the value function unaware of it.

        Both the actor and the reference model are transient here: each is
        faulted in, used for one forward pass, and evicted before the critic
        phase needs the accelerator.
        """
        kl_cfg = self._kl_reward_cfg
        assert kl_cfg is not None

        action_mask_list = prepared["action_mask_list"]
        token_rewards_list = prepared["token_rewards_list"]
        shifted_labels_list = prepared["shifted_labels_list"]

        if kl_cfg.coef == 0.0:
            return

        # Behavior log probabilities: either recomputed by the current policy
        # or reused from the inference engine.
        if kl_cfg.behavior_logprobs == "old":
            self._onload_actor()
            behavior_list = self._compute_full_logprobs(
                self._actor_forward_logprobs, prepared["seq_ctx_list"], shifted_labels_list
            )
            self._engine.put_model_to_device("cpu")
            DEVICE_MODULE.empty_cache()
            self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)
        else:
            behavior_list = []
            for rollout_logprobs, labels in zip(prepared["rollout_logprobs_list"], shifted_labels_list):
                if rollout_logprobs is None:
                    raise ValueError(
                        "kl_reward_cfg.behavior_logprobs='rollout' needs rollout logprobs, but the "
                        "rollout engine did not return them."
                    )
                behavior_list.append(rollout_logprobs.reshape(labels.shape))

        ref_list = self._compute_full_logprobs(
            self._ref_forward_logprobs, prepared["seq_ctx_list"], shifted_labels_list
        )

        kl_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
        token_count = torch.zeros((), dtype=torch.float64, device=DEVICE)
        for index, (behavior, ref, action_mask) in enumerate(zip(behavior_list, ref_list, action_mask_list)):
            kl = kl_divergence_per_token(behavior, ref, kl_cfg.kl_type)
            kl = torch.where(action_mask, kl, torch.zeros_like(kl))
            token_rewards_list[index] = token_rewards_list[index] - kl_cfg.coef * kl
            kl_sum += kl.sum().double()
            token_count += action_mask.sum().double()

        stats = torch.stack((kl_sum, token_count))
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() > 0:
            worker_log_item["kl_reward_mean"] = float((stats[0] / stats[1]).item())

    def _compute_full_logprobs(
        self,
        forward_fn,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run a log-probability forward, returning full-sequence results.

        Under sequence parallelism the model consumes shards, so the labels are
        split for the forward and the outputs gathered back: the KL penalty is
        applied to unsplit token rewards.
        """
        if self.sp_mesh.size() == 1:
            return forward_fn(seq_ctx_list, shifted_labels_list)

        split_seq_ctx_list = [seq_ctx.split(self.sp_mesh) for seq_ctx in seq_ctx_list]
        split_labels_list = [
            sp_split(labels, sp_mesh=self.sp_mesh, split_dim=1, padding_value=self.config.loss_cfg.ignore_idx)
            for labels in shifted_labels_list
        ]
        sharded = forward_fn(split_seq_ctx_list, split_labels_list)
        return [
            sp_gather(logprobs, self.sp_mesh, dim=1).reshape(labels.shape)
            for logprobs, labels in zip(sharded, shifted_labels_list)
        ]

    def _actor_forward_logprobs(
        self, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return self.compute_actor_logprobs(seq_ctx_list, shifted_labels_list)

    def _ref_forward_logprobs(
        self, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return self.compute_ref_logprobs(seq_ctx_list, shifted_labels_list)

    def _prepare_ppo_inputs(self, data_batches: list[WorkerInputItem]) -> dict[str, list]:
        """Materialize packed inputs and split the per-token PPO tensors.

        `token_rewards` and `action_mask` stay unsplit: GAE runs over the whole
        sequence, so sequence parallelism is handled by gathering values rather
        than by sharding the targets.
        """
        seq_ctx_list: list[SequenceContext] = []
        shifted_labels_list: list[torch.Tensor] = []
        rollout_logprobs_list: list[torch.Tensor | None] = []
        token_rewards_list: list[torch.Tensor] = []
        action_mask_list: list[torch.Tensor] = []
        cu_seq_lens_list: list[torch.Tensor] = []

        ignore_idx = self.config.loss_cfg.ignore_idx
        for data in data_batches:
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None:
                if not isinstance(pixel_values, list):
                    raise NotImplementedError(f"Unsupported pixel_values type: {type(pixel_values)}")
                materialized = ray.get(list(pixel_values))
                seq_ctx.pixel_values = torch.cat([torch.as_tensor(value) for value in materialized], dim=0)

            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                self._add_rollout_routed_experts(seq_ctx, rollout_routed_experts)
            seq_ctx.offload_rollout_routed_experts = self.config.offload_rollout_routed_experts

            if "token_rewards" not in data:
                raise ValueError("PPO worker input is missing `token_rewards`.")

            shifted_labels = data["shifted_labels"].to(DEVICE)
            # The action mask is derived rather than transferred: it is exactly
            # the set of tokens the policy loss trains on.
            action_mask = (shifted_labels != ignore_idx).to(DEVICE)
            token_rewards = data["token_rewards"].to(DEVICE).float()
            if token_rewards.shape != shifted_labels.shape:
                raise ValueError(
                    f"token_rewards shape {tuple(token_rewards.shape)} does not match labels "
                    f"{tuple(shifted_labels.shape)}."
                )

            rollout_logprobs = data.get("rollout_logprobs")
            rollout_logprobs = rollout_logprobs.to(DEVICE) if rollout_logprobs is not None else None

            # Capture the packed boundaries before any sequence-parallel split.
            cu_seq_lens_list.append(seq_ctx.cu_seq_lens_q.clone())
            seq_ctx_list.append(seq_ctx.to(DEVICE))
            shifted_labels_list.append(shifted_labels)
            action_mask_list.append(action_mask)
            token_rewards_list.append(token_rewards)
            rollout_logprobs_list.append(rollout_logprobs)

        return {
            "seq_ctx_list": seq_ctx_list,
            "shifted_labels_list": shifted_labels_list,
            "rollout_logprobs_list": rollout_logprobs_list,
            "token_rewards_list": token_rewards_list,
            "action_mask_list": action_mask_list,
            "cu_seq_lens_list": cu_seq_lens_list,
        }

    @torch.no_grad()
    def _compute_critic_values(self, critic_seq_ctx_list: list[SequenceContext]) -> list[torch.Tensor]:
        """Run the frozen critic forward, returning full-sequence values.

        Under sequence parallelism each rank holds one shard, so the shards are
        all-gathered: GAE is a sequential recursion over the whole trajectory
        and cannot be evaluated on a shard. Every rank then computes the same
        targets redundantly, which is far cheaper than the communication a
        distributed scan would need.
        """
        critic_engine = self._critic_engine
        assert critic_engine is not None
        critic_engine._maybe_precompute_float8_dynamic_scale_for_fsdp()

        values_list: list[torch.Tensor] = []
        for seq_ctx in critic_seq_ctx_list:
            output = critic_engine.model(seq_ctx=seq_ctx, loss_ctx=None)  # type: ignore[call-overload]
            values = output["logits"]
            if values is None or values.size(-1) != 1:
                raise RuntimeError("The critic must return scalar logits shaped [batch, seq, 1].")
            values = values.squeeze(-1).float()
            if self.sp_mesh.size() > 1:
                values = sp_gather(values, self.sp_mesh, dim=1)
            values_list.append(values.detach())
            output.free_nongrad_feature()
        return values_list

    def _compute_ppo_targets(
        self,
        values_list: list[torch.Tensor],
        token_rewards_list: list[torch.Tensor],
        action_mask_list: list[torch.Tensor],
        cu_seq_lens_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute advantages and returns, optionally normalizing advantages."""
        estimator = self._advantage_estimator
        assert estimator is not None

        advantages_list: list[torch.Tensor] = []
        returns_list: list[torch.Tensor] = []
        for values, token_rewards, action_mask, cu_seq_lens in zip(
            values_list, token_rewards_list, action_mask_list, cu_seq_lens_list
        ):
            if values.shape != action_mask.shape:
                raise RuntimeError(
                    f"Critic values {tuple(values.shape)} do not match the action mask "
                    f"{tuple(action_mask.shape)}. With sp_size > 1 this means the gathered "
                    "sequence length differs from pack_max_length."
                )
            advantages, returns = estimator.compute(values, token_rewards, action_mask, cu_seq_lens)
            advantages_list.append(advantages)
            returns_list.append(returns)

        if self._normalize_advantage and advantages_list:
            # Normalize over the whole global batch at once so every token sees
            # the same affine transform.
            sizes = [advantages.numel() for advantages in advantages_list]
            flat = torch.cat([advantages.reshape(-1) for advantages in advantages_list])
            flat_mask = torch.cat([mask.reshape(-1) for mask in action_mask_list])
            flat = normalize_advantages(flat, flat_mask)
            advantages_list = [
                chunk.reshape(reference.shape) for chunk, reference in zip(flat.split(sizes), advantages_list)
            ]

        return advantages_list, returns_list

    def _train_critic(
        self,
        critic_seq_ctx_list: list[SequenceContext],
        critic_loss_ctx_list: list[BaseLossContext],
        *,
        rollout_idx: int,
    ) -> tuple[list[WorkerTrainLogItem], dict[str, float]]:
        """Train the critic for the configured number of passes."""
        critic_engine = self._critic_engine
        critic_cfg = self.config.critic_cfg
        assert critic_engine is not None and critic_cfg is not None

        train_logs: list[WorkerTrainLogItem] = []
        accumulated: dict[str, float] = {}
        critic_engine.put_optimizer_to_device(DEVICE)
        num_batches = len(critic_seq_ctx_list)
        iters_per_step = math.ceil(num_batches / critic_cfg.optimizer_steps_per_pass) if num_batches else 1

        for pass_index in range(critic_cfg.num_passes):
            order = self._ppo_pass_order(num_batches, rollout_idx, pass_index)
            for start in range(0, num_batches, iters_per_step):
                indices = order[start : start + iters_per_step]
                if not indices:
                    continue
                chunk = [critic_loss_ctx_list[index] for index in indices]

                rank_tokens = sum(ctx.loss_kwargs.value_mask.sum() for ctx in chunk)
                global_tokens = cast(torch.Tensor, rank_tokens).clone()
                dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
                if global_tokens.item() == 0:
                    continue

                batched = critic_cfg.loss_cfg.loss_ctx_cls.build_batches(chunk)  # type: ignore[arg-type]
                engine_input = [
                    ModelItem(seq_ctx=critic_seq_ctx_list[index], loss_ctx={"lm": loss_ctx})
                    for index, loss_ctx in zip(indices, batched)
                ]
                train_step_info = critic_engine.train_step(engine_input)
                grad_norm = critic_engine.clip_grad_norm()
                self._step_critic_optimizer_and_scheduler(grad_norm)

                log_item, step_metrics = self._build_critic_log(train_step_info, grad_norm)
                train_logs.append(log_item)
                for key, value in step_metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + value

        return train_logs, self._finalize_critic_metrics(accumulated)

    def _ppo_pass_order(self, num_batches: int, rollout_idx: int, pass_index: int) -> list[int]:
        """Batch order for one critic pass; reshuffled after the first."""
        if pass_index == 0:
            return list(range(num_batches))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed or 0) + rollout_idx * 1_000_003 + pass_index)
        return torch.randperm(num_batches, generator=generator).tolist()

    def _build_critic_log(
        self, train_step_info: TrainStepInfo, grad_norm: torch.Tensor
    ) -> tuple[WorkerTrainLogItem, dict[str, float]]:
        logs_info = cast(dict[str, float], train_step_info.pop("logs_info"))  # type: ignore[misc]
        extra_info = train_step_info.pop("extra_info")  # type: ignore[misc]
        if isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info_dict = extra_info.get()
        else:
            extra_info_dict = cast(dict, extra_info)
        extra_info_dict = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in extra_info_dict.items()
            if isinstance(value, (torch.Tensor, int, float))
        }
        # Split off the reducible critic sums; the rest is per-step logging.
        critic_sums = {
            key: float(value) for key, value in extra_info_dict.items() if key.startswith("reduced_critic_")
        }
        for key in critic_sums:
            extra_info_dict.pop(key)
        train_step_info.pop("total_loss", None)  # type: ignore[misc]

        assert self._critic_engine is not None
        log_item = WorkerTrainLogItem(
            **logs_info,  # type: ignore[typeddict-item]
            **train_step_info,
            **extra_info_dict,
            grad_norm=float(grad_norm.item()),
            lr=float(self._critic_engine.optimizer.param_groups[0]["lr"]),
        )
        return log_item, critic_sums

    def _finalize_critic_metrics(self, accumulated: dict[str, float]) -> dict[str, float]:
        """Reduce critic sums across ranks into interpretable metrics."""
        if not accumulated:
            return {}
        keys = sorted(accumulated)
        totals = torch.tensor([accumulated[key] for key in keys], dtype=torch.float64, device=DEVICE)
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        reduced = dict(zip(keys, totals.tolist()))

        count = reduced.get("reduced_critic_valid_count", 0.0)
        if count <= 0:
            return {}
        metrics = {
            "critic/value_mean": reduced.get("reduced_critic_value_sum", 0.0) / count,
            "critic/return_mean": reduced.get("reduced_critic_return_sum", 0.0) / count,
            "critic/value_mse": reduced.get("reduced_critic_error_square_sum", 0.0) / count,
        }
        variance = explained_variance(
            return_square_sum=reduced.get("reduced_critic_return_square_sum", 0.0),
            return_sum=reduced.get("reduced_critic_return_sum", 0.0),
            error_square_sum=reduced.get("reduced_critic_error_square_sum", 0.0),
            count=count,
        )
        if variance is not None:
            metrics["critic/explained_variance"] = variance
        if "reduced_critic_clip_count" in reduced:
            metrics["critic/clip_frac"] = reduced["reduced_critic_clip_count"] / count
        return metrics

    def _train_actor_on_advantages(
        self,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
        advantages_list: list[torch.Tensor],
        rollout_logprobs_list: list[torch.Tensor | None],
        *,
        rollout_idx: int,
        worker_log_item: WorkerLogItem,
    ) -> list[WorkerTrainLogItem]:
        """Build actor loss contexts from GAE advantages and run the updates."""
        loss_cfg: BaseRLLossConfig = self.config.loss_cfg

        # Split now: the loss context owns the sharded view, while GAE needed
        # the full sequence.
        split_seq_ctx_list = seq_ctx_list
        if self.sp_mesh.size() > 1:
            split_seq_ctx_list = [seq_ctx.split(self.sp_mesh) for seq_ctx in seq_ctx_list]

        loss_ctx_list: list[BaseRLLossContext] = []
        for shifted_labels, advantages, rollout_logprobs in zip(
            shifted_labels_list, advantages_list, rollout_logprobs_list
        ):
            loss_ctx = loss_cfg.build(
                data={
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                },
                sp_mesh=self.sp_mesh,
            )
            if loss_ctx is None:
                raise RuntimeError("The actor loss config did not build a loss context.")
            loss_ctx_list.append(loss_ctx)

        sharded_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        sharded_logprobs_list = [loss_ctx.loss_kwargs.rollout_logprobs for loss_ctx in loss_ctx_list]

        old_logprobs_list = self.compute_actor_logprobs(split_seq_ctx_list, sharded_labels_list)
        for old_logprobs, loss_ctx in zip(old_logprobs_list, loss_ctx_list):
            loss_ctx.loss_kwargs.old_logprobs = old_logprobs

        rank_grad_tokens = sum((labels != loss_cfg.ignore_idx).sum() for labels in sharded_labels_list)
        global_grad_tokens = cast(torch.Tensor, rank_grad_tokens).clone()
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        train_entropy = calculate_entropy(sharded_labels_list, old_logprobs_list, global_grad_tokens)
        rollout_entropy = calculate_entropy(sharded_labels_list, sharded_logprobs_list, global_grad_tokens)
        if train_entropy is not None:
            worker_log_item["train_entropy"] = float(train_entropy.item())
        if rollout_entropy is not None:
            worker_log_item["rollout_entropy"] = float(rollout_entropy.item())

        all_mismatch_metrics = []
        all_rollout_is_metrics = []
        for seq_ctx, loss_ctx in zip(split_seq_ctx_list, loss_ctx_list):
            if loss_ctx.loss_kwargs.rollout_logprobs is not None:
                mismatch_metrics, rollout_is_metrics = loss_ctx.compute_rollout_is(self.sp_mesh, seq_ctx.seq_lens_q)
                all_mismatch_metrics.append(mismatch_metrics)
                all_rollout_is_metrics.append(rollout_is_metrics)
        if all_mismatch_metrics:
            merged = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if merged:
                worker_log_item["mismatch_metrics"] = merged
        if all_rollout_is_metrics:
            merged = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if merged:
                worker_log_item["rollout_is_metrics"] = merged

        self._engine.put_optimizer_to_device(DEVICE)
        num_batches = len(split_seq_ctx_list)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps) if num_batches else 1
        return self._train_actor(
            split_seq_ctx_list,
            loss_ctx_list,
            [],
            rollout_idx=rollout_idx,
            iters_per_step=iters_per_step,
        )

    def _fit_sft(self):
        self.logger.info(f"Train SFT after {self._rollout_step} RL steps")
        if self._sft_dataloader_iter is None:
            self._sft_dataloader_iter = iter(self._sft_dataloader)

        time_before_get_data = time.time()
        data_batch = self._next_sft_data_batch()
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        DEVICE_MODULE.reset_peak_memory_stats()

        train_step_info, grad_norm = self._train_one_step_sft(data_batch)

        time_after_train_step = time.time()
        step_time = time_after_train_step - time_before_train_step
        step_consumed_tokens = train_step_info["step_consumed_tokens"]

        reduced_step_consumed_tokens = self._reduce_number_across_rank(step_consumed_tokens)
        self._sft_total_consumed_tokens += reduced_step_consumed_tokens

        self._sft_log_step(
            train_step_info=train_step_info,
            local_step_consumed_tokens=step_consumed_tokens,
            step_consumed_tokens=reduced_step_consumed_tokens,
            total_consumed_tokens=self._sft_total_consumed_tokens,
            data_time=data_time,
            step_time=step_time,
            grad_norm=grad_norm,
        )

        return train_step_info

    def _next_sft_data_batch(self):
        try:
            data = next(self._sft_dataloader_iter)  # type: ignore[assignment]
        except StopIteration:
            self._sft_cur_epoch += 1
            self._sft_dataloader.set_epoch(self._sft_cur_epoch)
            self._sft_dataloader_iter = iter(self._sft_dataloader)
            data = next(self._sft_dataloader_iter)
        return data

    def _train_one_step_sft(self, data_batch):
        seq_ctx_list: list[SequenceContext] = []
        loss_cfg: CELossConfig = self._sft_loss_cfg
        loss_ctx_list: list[CELossContext] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx = loss_cfg.build(data={"shifted_labels": data["shifted_labels"]}, sp_mesh=self.sp_mesh)
            loss_ctx_list.append(loss_ctx)

        del data_batch

        cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
        loss_ctx_list = CELossContext.build_batches(
            loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=self.sp_mesh
        )

        engine_input = [
            ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]

        train_step_info = self._engine.train_step(engine_input)
        grad_norm = self._engine.clip_grad_norm()
        self._step_optimizer_and_scheduler(grad_norm)
        return train_step_info, grad_norm

    def _sft_log_step(
        self,
        train_step_info: TrainStepInfo,
        local_step_consumed_tokens: int,
        step_consumed_tokens: int,
        total_consumed_tokens: int,
        data_time: float,
        step_time: float,
        grad_norm: torch.Tensor,
    ):
        tgs = local_step_consumed_tokens / step_time
        logs_info = train_step_info.get("logs_info", {})
        log_items = [f"{k}: {v:.8f}" for k, v in logs_info.items() if "loss" in k]
        log_items.append(f"total_loss: {train_step_info['total_loss']:.8f}")
        loss_log_str = ", ".join(log_items)

        max_memory = DEVICE_MODULE.max_memory_allocated()  # type: ignore[attr-defined]
        reserved_memory = DEVICE_MODULE.max_memory_reserved()  # type: ignore[attr-defined]

        self.logger.info(
            f"Rank{self.rank} Step {self._rollout_step}: data_time: {data_time:.4f} time: {step_time:.4f} "
            f"text_tokens: {local_step_consumed_tokens} "
            f"step_consumed_tokens: {step_consumed_tokens} "
            f"total_consumed_tokens: {total_consumed_tokens} "
            f"efficient_attn_ratio: {train_step_info['efficient_attn_ratio']:.4f} "
            f"{loss_log_str} "
            f"grad_norm: {grad_norm:.8f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"tgs: {tgs:.4f}"
        )

    def _reduce_number_across_rank(self, rank_number: int) -> int:
        _gathered_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(_gathered_list, rank_number)
        reduced_number = sum(_gathered_list)  # type: ignore[arg-type]
        return reduced_number

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        """Export actor weights, and critic weights for a PPO worker."""
        if self.is_ppo and self._ppo_phase != PPOPhase.ACTOR_READY:
            raise RuntimeError(f"Cannot save a PPO HF checkpoint from phase {self._ppo_phase.value}.")

        self._engine.save_hf(hf_dir, save_dtype)
        if self._critic_engine is not None:
            critic_hf_dir = Path(hf_dir) / self._SAVE_CRITIC_HF_DIR
            self._onload_critic()
            try:
                self._critic_engine.save_hf(str(critic_hf_dir), save_dtype)
            finally:
                self._offload_critic()
                self._onload_actor()
                self._set_ppo_phase(PPOPhase.ACTOR_READY)

    @ray_method
    def get_data_replicate_size(self) -> int:
        """Get the data replicate size for the training worker."""
        # tp and pp will affect the data replicate size in engine
        # sp will affect the data replicate size in worker
        return self._engine.data_replicate_size * self.sp_mesh.size()

    @ray_method
    def get_model_cfg(self):
        model_cfg = self._engine.model_cfg
        return model_cfg

    def _clear_cublas_workspaces(self) -> None:
        clear_fn = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_fn is None:
            self.logger.warning("torch._C._cuda_clearCublasWorkspaces is unavailable")
            return
        try:
            clear_fn()
            DEVICE_MODULE.empty_cache()
        except Exception as e:
            self.logger.warning(f"Failed to clear cuBLAS workspaces: {e}")

    def _release_deferred_fsdp_all_gathers(self, log_tag: str) -> None:
        comm_states, param_groups = release_deferred_fsdp_all_gathers(self._engine.model)
        if comm_states or param_groups:
            self.logger.debug(
                f"[{log_tag}] released deferred FSDP all-gathers: "
                f"comm_states={comm_states}, param_groups={param_groups}"
            )

    @ray_method
    def offload_model(self):
        if self.is_ppo:
            if self._ppo_phase == PPOPhase.CRITIC_TRAIN:
                raise RuntimeError("Cannot offload the actor while the critic owns the accelerator.")
            self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)
        model_moved = self._engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._clear_cublas_workspaces()
        if not model_moved:
            self.logger.info("Skip model offload because model placement is unchanged.")
            return
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        optimizer_moved = self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        if not optimizer_moved:
            if getattr(self.config.optim_cfg, "swap_optimizer", False):
                self.logger.info(
                    "Skip optimizer offload because swap_optimizer=True; optimizer states are already CPU-resident."
                )
            else:
                self.logger.info("Skip optimizer offload because optimizer state is empty.")
            return
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def onload_model(self):
        if self.is_ppo and self._ppo_phase == PPOPhase.CRITIC_TRAIN:
            raise RuntimeError("The actor and critic cannot be resident at the same time.")
        model_moved = self._engine.put_model_to_device(DEVICE)
        if self.is_ppo:
            self._set_ppo_phase(PPOPhase.ACTOR_READY)
        if not model_moved:
            self.logger.info("Skip model onload because model placement is unchanged.")

    @ray_method
    def onload_optimizer(self):
        optimizer_moved = self._engine.put_optimizer_to_device(DEVICE)
        if not optimizer_moved:
            if getattr(self.config.optim_cfg, "swap_optimizer", False):
                self.logger.info(
                    "Skip optimizer onload because swap_optimizer=True; optimizer states stay on CPU and are swapped per step."
                )
            else:
                self.logger.info("Skip optimizer onload because optimizer state is empty.")

    @ray_method
    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training worker."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        weights_path = checkpoint_path / self._SAVE_WEIGHTS_DIR

        # Save model and optimizer
        self._engine.save_dcp(
            weights_dir=weights_path,
            save_optimizer=not no_save_optimizer,
        )

        # The scheduler state is replicated across ranks, so rank 0 owns it.
        if self.rank == 0:
            torch.save(
                {
                    "lr_scheduler": self._lr_scheduler.state_dict(),
                    "optimizer_updates": self._optimizer_updates,
                },
                checkpoint_path / self._SAVE_SCHEDULER_PATH,
            )

        if self._critic_engine is not None:
            if no_save_optimizer:
                raise ValueError("A PPO checkpoint must include the critic optimizer state.")
            self._critic_engine.save_dcp(
                weights_dir=checkpoint_path / self._SAVE_CRITIC_WEIGHTS_DIR,
                save_optimizer=True,
            )
            if self.rank == 0:
                assert self._critic_scheduler is not None
                torch.save(
                    {
                        "lr_scheduler": self._critic_scheduler.state_dict(),
                        "optimizer_updates": self._critic_optimizer_updates,
                    },
                    checkpoint_path / self._SAVE_CRITIC_SCHEDULER_PATH,
                )

        # Save sft dataloader
        if self._sft_dataloader is not None:
            sft_dataloader_path = checkpoint_path / self._SAVE_SFT_DATALOADER_DIR
            dataloader_state = self._sft_dataloader.get_state_dict()
            total_consumed_samples = dataloader_state["total_consumed_samples"]
            if self.rank != 0:
                return

            torch.save(dataloader_state, sft_dataloader_path)

            train_state_path = checkpoint_path / self._SAVE_SFT_TRAIN_STATE_PATH
            with train_state_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "cur_step": self._rollout_step,
                            "cur_epoch": self._sft_cur_epoch,
                            "total_consumed_samples": total_consumed_samples,
                            "total_consumed_tokens": self._sft_total_consumed_tokens,
                        }
                    )
                )

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training worker from the checkpoint."""
        resume_from = load_checkpoint_cfg.checkpoint_path
        if resume_from is None:
            return
        if isinstance(resume_from, str):
            resume_from = Path(resume_from)
        self.logger.info(f"Resume from checkpoint: {resume_from}")

        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path {resume_from} does not exist.")

        weights_path = resume_from / self._SAVE_WEIGHTS_DIR
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint at {resume_from} has no '{self._SAVE_WEIGHTS_DIR}/' directory.")

        self._engine.load_dcp(
            weights_dir=weights_path,
            load_states=load_checkpoint_cfg.load_optimizer_states,
            load_args=load_checkpoint_cfg.load_optimizer_args,
        )

        if load_checkpoint_cfg.load_scheduler:
            scheduler_path = resume_from / self._SAVE_SCHEDULER_PATH
            if scheduler_path.exists():
                scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=False)
                self._lr_scheduler.load_state_dict(scheduler_state["lr_scheduler"])
                self._optimizer_updates = int(scheduler_state["optimizer_updates"])
                self.logger.info(f"Resumed LR scheduler at {self._optimizer_updates} optimizer updates")
            else:
                # Checkpoints written before the RL scheduler existed have no
                # such file; keeping the fresh schedule is the safe fallback.
                self.logger.warning(f"No LR scheduler state at {scheduler_path}; starting the schedule from step 0.")

        if self._critic_engine is not None:
            critic_weights_path = resume_from / self._SAVE_CRITIC_WEIGHTS_DIR
            if not critic_weights_path.exists():
                raise FileNotFoundError(
                    f"PPO resume requires a '{self._SAVE_CRITIC_WEIGHTS_DIR}/' directory in {resume_from}."
                )
            self._critic_engine.load_dcp(
                weights_dir=critic_weights_path,
                load_states=load_checkpoint_cfg.load_optimizer_states,
                load_args=load_checkpoint_cfg.load_optimizer_args,
                strict=True,
            )
            critic_scheduler_path = resume_from / self._SAVE_CRITIC_SCHEDULER_PATH
            if load_checkpoint_cfg.load_scheduler and critic_scheduler_path.exists():
                assert self._critic_scheduler is not None
                critic_state = torch.load(critic_scheduler_path, map_location="cpu", weights_only=False)
                self._critic_scheduler.load_state_dict(critic_state["lr_scheduler"])
                self._critic_optimizer_updates = int(critic_state["optimizer_updates"])
            # Both models start parked; the phase machine faults in whichever
            # the next step needs.
            self._engine.put_model_to_device("cpu")
            self._engine.put_optimizer_to_device("cpu")
            self._critic_engine.put_model_to_device("cpu")
            self._critic_engine.put_optimizer_to_device("cpu")
            DEVICE_MODULE.empty_cache()
            self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)

        # Resume sft dataloader
        if self._sft_dataloader is not None:
            train_state_path = resume_from / self._SAVE_SFT_TRAIN_STATE_PATH
            if not train_state_path.exists():
                raise FileNotFoundError(f"Train state path {train_state_path} does not exist.")
            with train_state_path.open("r") as f:
                train_state = json.loads(f.read())
            self._rollout_step = train_state["cur_step"]
            self._sft_cur_epoch = train_state["cur_epoch"]
            self._sft_total_consumed_tokens = train_state["total_consumed_tokens"]
            self.logger.info(f"Resume sft train state from {train_state_path}")

            sft_dataloader_path = resume_from / self._SAVE_SFT_DATALOADER_DIR
            if not sft_dataloader_path.exists():
                raise FileNotFoundError(f"Dataloader path {sft_dataloader_path} does not exist.")
            dataloader_state = torch.load(sft_dataloader_path, map_location=DEVICE)
            self._sft_dataloader.load_state_dict(dataloader_state)
            self.logger.info(f"Resume sft dataloader from {sft_dataloader_path}")

    @ray_method
    def ready(self) -> bool:
        return True


TrainingWorkerClass = ActorClass[TrainingWorker]
TrainingWorkerProxy = ActorProxy[TrainingWorker]
