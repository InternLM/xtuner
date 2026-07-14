import contextlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Sequence,
    TypeAlias,
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
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.rl.loss import BaseRLLossConfig, BaseRLLossContext, finalize_train_policy_metrics, kl_penalty
from xtuner.v1.rl.ppo import action_gae, normalize_advantages
from xtuner.v1.rl.utils import SingleAcceleratorWorker
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

from ..rollout_is import merge_rollout_is_metrics
from .ppo_config import CriticWorkerConfig, PPOConfig
from .update_weighter import UpdateWeighter


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
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

    # Optional PPO/Critic path. Existing RL recipes leave both fields unset and
    # retain the original Actor-only behavior.
    critic_cfg: CriticWorkerConfig | None = None
    ppo_cfg: PPOConfig | None = None

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
    action_mask: NotRequired[torch.BoolTensor]
    actor_loss_mask: NotRequired[torch.BoolTensor]
    critic_loss_mask: NotRequired[torch.BoolTensor]
    token_rewards: NotRequired[torch.Tensor]
    actor_positive_scale: NotRequired[torch.Tensor]
    actor_negative_scale: NotRequired[torch.Tensor]


class WorkerTrainLogItem(TypedDict, total=False):
    step_consumed_tokens: int
    efficient_attn_ratio: float
    grad_norm: float


class WorkerLogItem(TypedDict):
    train_entropy: float
    rollout_entropy: NotRequired[float]
    mismatch_metrics: NotRequired[dict[str, float]]
    rollout_is_metrics: NotRequired[dict[str, float]]
    train_metrics: List[WorkerTrainLogItem]
    sft_train_metrics: NotRequired[dict[str, float]]
    critic_train_metrics: NotRequired[List[WorkerTrainLogItem]]


class PPOPhase(str, Enum):
    ALL_OFFLOADED = "all_offloaded"
    CRITIC_TRAIN = "critic_train"
    ACTOR_TRAIN = "actor_train"
    ACTOR_READY = "actor_ready"


class TrainingWorker(SingleAcceleratorWorker, UpdateWeighter):
    _SAVE_WEIGHTS_DIR = "weights"
    _SAVE_CRITIC_WEIGHTS_DIR = "critic_weights"
    _SAVE_PPO_STATE_DIR = "ppo_worker_state"
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
        self.config: WorkerConfig = worker_cfg
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

        if worker_cfg.critic_cfg is not None:
            worker_cfg.model_cfg.dcp_ignore_frozen_params = False
            worker_cfg.critic_cfg.model_cfg.dcp_ignore_frozen_params = False
        if not worker_cfg.fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = self._build_engine(worker_cfg)

        self._critic_engine: TrainEngine | None = None
        self._actor_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._critic_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._actor_optimizer_updates = 0
        self._critic_optimizer_updates = 0
        self._ppo_phase = PPOPhase.ACTOR_READY

        if worker_cfg.critic_cfg is not None:
            if worker_cfg.ppo_cfg is None:
                raise ValueError("ppo_cfg is required when critic_cfg is configured.")
            if worker_cfg.sp_size != 1:
                raise ValueError("The first PPO/Critic implementation supports actor sp_size=1 only.")
            if worker_cfg.loss_cfg.use_kl_loss:
                raise ValueError("KL loss must be disabled for the first PPO/Critic implementation.")
            if self._sft_dataloader is not None:
                raise ValueError("SFT interleaving is not supported by the first PPO/Critic implementation.")

            self._actor_scheduler = self._build_lr_scheduler(
                self._engine.optimizer,
                worker_cfg.lr_cfg,
                worker_cfg.scheduler_steps,
            )
            self._engine.put_model_to_device("cpu")
            self._engine.put_optimizer_to_device("cpu")
            DEVICE_MODULE.empty_cache()

            self._critic_engine = self._build_critic_engine(worker_cfg, worker_cfg.critic_cfg)
            self._critic_scheduler = self._build_lr_scheduler(
                self._critic_engine.optimizer,
                worker_cfg.critic_cfg.lr_cfg,
                worker_cfg.critic_cfg.scheduler_steps,
            )
            self._critic_engine.put_model_to_device("cpu")
            self._critic_engine.put_optimizer_to_device("cpu")
            DEVICE_MODULE.empty_cache()
            self._ppo_phase = PPOPhase.ALL_OFFLOADED

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss:
            self._has_ref = True
            if worker_cfg.ref_load_from is None:
                worker_cfg.ref_load_from = worker_cfg.load_from
            self._ref_model = self._build_ref_model(
                worker_cfg.model_cfg, worker_cfg.ref_load_from, worker_cfg.ref_model_fsdp_cfg
            )

        self._optimizer_steps = worker_cfg.optimizer_steps
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
        if worker_cfg.critic_cfg is not None and self.mtp_config is not None:
            raise ValueError("Actor MTP must be disabled for the first PPO/Critic implementation.")

        self._init_update_weighter()

    def _build_critic_engine(
        self,
        worker_cfg: WorkerConfig,
        critic_cfg: CriticWorkerConfig,
    ) -> TrainEngine:
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
                raise ValueError("Actor HF path is required to initialize the Critic backbone.")
            # The value-model loader consumes the expected Actor LM-head mismatch
            # and initializes only the scalar head. Keep the compose-level load
            # strict so a missing backbone/vision/projector tensor fails early.
            engine.from_hf(source, strict=True)
        else:
            assert critic_cfg.load_from is not None
            weights_path = Path(critic_cfg.load_from)
            nested_weights_path = weights_path / self._SAVE_CRITIC_WEIGHTS_DIR
            if nested_weights_path.exists():
                weights_path = nested_weights_path
            engine.load_dcp(
                weights_dir=weights_path,
                load_states=False,
                load_args=False,
                strict=True,
            )
        return engine

    @staticmethod
    def _build_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_cfg: LRConfig,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if lr_cfg.warmup_ratio < 1:
            warmup_steps = int(lr_cfg.warmup_ratio * total_steps)
        else:
            warmup_steps = int(lr_cfg.warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps)
        base_lr = float(optimizer.defaults["lr"])
        min_factor = float(lr_cfg.lr_min) / base_lr if base_lr > 0 else 1.0

        def lr_factor(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(step, 1) / warmup_steps
            if lr_cfg.lr_type == "constant":
                return 1.0
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            if lr_cfg.lr_type == "linear":
                return 1.0 - progress * (1.0 - min_factor)
            if lr_cfg.lr_type == "cosine":
                return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
            raise ValueError(f"Unsupported lr type: {lr_cfg.lr_type}")

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

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
                    # free obj store explicitly
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

        if self.sp_mesh is not None and self.sp_mesh.size() > 1:
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
        if self._critic_engine is not None:
            return self._fit_ppo(data_batches, rollout_idx)

        # NOTE: sglang会清除logger handle, 重新创建
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
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
            self.logger.debug(
                f"Rank{self.rank} Rollout {rollout_idx} GlobalStep {global_train_step} "
                f"train_step[{i}].engine_train_step elapsed={time.perf_counter() - train_step_begin:.4f}s"
            )
            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)

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

            train_log_item = WorkerTrainLogItem(
                **engine_logs_info,  # type: ignore[typeddict-item]
                **train_step_info,
                **extra_info_dict,
                grad_norm=grad_norm.item(),
            )
            worker_log_item["train_metrics"].append(train_log_item)

            # Extract logs_info for logging
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in train_log_item.items()
                if not key.startswith("reduced_train_policy_") and key != "max_ratio"
            )
            log_str = f"Rank{self.rank} Rollout {rollout_idx} Step {i}: " + log_str
            self.logger.info(log_str)
            self._global_train_step = global_train_step

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

    def _fit_ppo(self, data_batches: list[WorkerInputItem], rollout_idx: int) -> WorkerLogItem:
        """Run one Critic phase followed by one Actor phase on frozen targets."""
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        critic_engine = self._critic_engine
        critic_cfg = self.config.critic_cfg
        ppo_cfg = self.config.ppo_cfg
        assert critic_engine is not None and critic_cfg is not None and ppo_cfg is not None
        if self.sp_mesh.size() != 1:
            raise RuntimeError("The first PPO/Critic implementation supports TRAIN_SP_SIZE=1 only.")

        # Keep Actor inputs (especially rollout routed-expert tensors) on CPU
        # until the Critic model and all Critic targets have been offloaded.
        seq_ctx_list: list[SequenceContext] = []
        shifted_labels_list: list[torch.Tensor] = []
        action_mask_list: list[torch.Tensor] = []
        actor_loss_mask_list: list[torch.Tensor] = []
        critic_loss_mask_list: list[torch.Tensor] = []
        token_rewards_list: list[torch.Tensor] = []
        rollout_logprobs_list: list[torch.Tensor | None] = []
        positive_scale_list: list[torch.Tensor] = []
        negative_scale_list: list[torch.Tensor] = []

        for data in data_batches:
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None and not isinstance(pixel_values, torch.Tensor):
                if not isinstance(pixel_values, list):
                    raise NotImplementedError(f"Unsupported pixel_values type: {type(pixel_values)}")
                materialized = ray.get(list(pixel_values))
                seq_ctx.pixel_values = torch.cat([torch.as_tensor(value) for value in materialized], dim=0)

            # Resolve and release rollout ObjectRefs immediately, but keep the
            # resulting routed-expert tensor on CPU throughout the Critic phase.
            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                self._add_rollout_routed_experts(seq_ctx, rollout_routed_experts)

            required_fields = (
                "action_mask",
                "actor_loss_mask",
                "critic_loss_mask",
                "token_rewards",
                "actor_positive_scale",
                "actor_negative_scale",
            )
            missing = [field for field in required_fields if field not in data]
            if missing:
                raise ValueError(f"PPO worker input is missing fields: {missing}")

            shifted_labels = data["shifted_labels"].cpu()
            action_mask = data["action_mask"].cpu().bool()
            actor_loss_mask = data["actor_loss_mask"].cpu().bool()
            critic_loss_mask = data["critic_loss_mask"].cpu().bool()
            token_rewards = data["token_rewards"].cpu().float()
            positive_scale = data["actor_positive_scale"].cpu().float()
            negative_scale = data["actor_negative_scale"].cpu().float()
            rollout_logprobs = data.get("rollout_logprobs")
            rollout_logprobs = rollout_logprobs.cpu() if rollout_logprobs is not None else None

            expected_shape = shifted_labels.shape
            fields = {
                "action_mask": action_mask,
                "actor_loss_mask": actor_loss_mask,
                "critic_loss_mask": critic_loss_mask,
                "token_rewards": token_rewards,
                "actor_positive_scale": positive_scale,
                "actor_negative_scale": negative_scale,
            }
            for name, tensor in fields.items():
                if tensor.shape != expected_shape:
                    raise ValueError(f"{name} shape {tensor.shape} does not match labels {expected_shape}.")
            if torch.any(actor_loss_mask & ~action_mask) or torch.any(critic_loss_mask & ~action_mask):
                raise ValueError("Actor/Critic loss masks must be subsets of action_mask.")

            seq_ctx_list.append(seq_ctx)
            shifted_labels_list.append(shifted_labels)
            action_mask_list.append(action_mask)
            actor_loss_mask_list.append(actor_loss_mask)
            critic_loss_mask_list.append(critic_loss_mask)
            token_rewards_list.append(token_rewards)
            rollout_logprobs_list.append(rollout_logprobs)
            positive_scale_list.append(positive_scale)
            negative_scale_list.append(negative_scale)

        del data_batches
        worker_log_item: WorkerLogItem = {
            "train_entropy": 0.0,
            "train_metrics": [],
            "critic_train_metrics": [],
            "sft_train_metrics": {},
        }

        # Critic routes independently and owns no rollout routed-expert tensor.
        self._onload_critic()
        critic_seq_ctx_list = [seq_ctx.copy(rollout_routed_experts=None).to(DEVICE) for seq_ctx in seq_ctx_list]
        critic_engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        old_values_list: list[torch.Tensor] = []
        with torch.no_grad():
            for critic_seq_ctx in critic_seq_ctx_list:
                output = critic_engine.model(seq_ctx=critic_seq_ctx, loss_ctx=None)  # type: ignore[call-overload]
                values = output["logits"]
                if values is None or values.size(-1) != 1:
                    raise RuntimeError("Critic must return scalar logits with shape [batch, seq, 1].")
                # GAE is a sequential recurrence. Stage one value tensor to CPU
                # per pack instead of synchronizing CUDA once per action token.
                old_values_list.append(values.squeeze(-1).detach().float().to("cpu", copy=True))
                output.free_nongrad_feature()
        del critic_seq_ctx, output, values

        actor_advantages_list: list[torch.Tensor] = []
        critic_returns_list: list[torch.Tensor] = []
        for seq_ctx, old_values, token_rewards, action_mask, positive_scale, negative_scale in zip(
            seq_ctx_list,
            old_values_list,
            token_rewards_list,
            action_mask_list,
            positive_scale_list,
            negative_scale_list,
        ):
            boundaries = seq_ctx.cu_seq_lens_q
            actor_advantages = action_gae(
                old_values,
                token_rewards,
                action_mask,
                boundaries,
                gamma=ppo_cfg.actor_gamma,
                gae_lambda=ppo_cfg.actor_lambda,
            )
            critic_advantages = action_gae(
                old_values,
                token_rewards,
                action_mask,
                boundaries,
                gamma=ppo_cfg.critic_gamma,
                gae_lambda=ppo_cfg.critic_lambda,
            )
            actor_advantages = torch.where(
                actor_advantages > 0,
                actor_advantages * positive_scale,
                actor_advantages * negative_scale,
            )
            actor_advantages_list.append(actor_advantages.detach())
            critic_returns_list.append(
                torch.where(action_mask, critic_advantages + old_values, torch.zeros_like(old_values)).detach()
            )

        if ppo_cfg.normalize_actor_advantage:
            sizes = [advantages.numel() for advantages in actor_advantages_list]
            flat_advantages = torch.cat([advantages.reshape(-1) for advantages in actor_advantages_list]).to(DEVICE)
            flat_actor_mask = torch.cat([mask.reshape(-1) for mask in actor_loss_mask_list]).to(DEVICE)
            flat_advantages = normalize_advantages(flat_advantages, flat_actor_mask, distributed=True)
            actor_advantages_list = [
                chunk.reshape(reference.shape).cpu()
                for chunk, reference in zip(flat_advantages.split(sizes), actor_advantages_list)
            ]
            del flat_advantages, flat_actor_mask
        actor_advantages_list = [
            torch.where(mask, advantages, torch.zeros_like(advantages)).detach().cpu()
            for advantages, mask in zip(actor_advantages_list, actor_loss_mask_list)
        ]

        critic_loss_ctx_list: list[BaseLossContext] = []
        for returns, value_mask, old_values in zip(
            critic_returns_list,
            critic_loss_mask_list,
            old_values_list,
        ):
            critic_loss_ctx = critic_cfg.loss_cfg.build(
                data={
                    "returns": returns,
                    "value_mask": value_mask,
                    "old_values": old_values,
                },
                sp_mesh=self.sp_mesh,
            )
            if critic_loss_ctx is None:
                raise RuntimeError("Critic loss config did not build a loss context.")
            critic_loss_ctx_list.append(critic_loss_ctx)
        del actor_advantages, critic_advantages, action_mask, negative_scale, old_values, positive_scale, seq_ctx
        del critic_loss_ctx, token_rewards, returns, value_mask

        worker_log_item["critic_train_metrics"].extend(
            self._train_ppo_critic(
                critic_seq_ctx_list,
                critic_loss_ctx_list,
                rollout_idx,
            )
        )

        # Drop every Critic-only tensor before Actor parameters or routed-expert
        # tensors are brought to the accelerator.
        del (
            critic_loss_ctx_list,
            critic_returns_list,
            old_values_list,
            critic_seq_ctx_list,
        )
        DEVICE_MODULE.empty_cache()
        self._offload_critic()
        self._onload_actor()

        for seq_ctx in seq_ctx_list:
            seq_ctx.to(DEVICE)
        shifted_labels_list = [labels.to(DEVICE) for labels in shifted_labels_list]
        actor_advantages_list = [advantages.to(DEVICE) for advantages in actor_advantages_list]
        rollout_logprobs_list = [
            logprobs.to(DEVICE) if logprobs is not None else None for logprobs in rollout_logprobs_list
        ]

        loss_cfg: BaseRLLossConfig = self.config.loss_cfg
        actor_loss_ctx_list: list[BaseRLLossContext] = []
        for shifted_labels, advantages, rollout_logprobs in zip(
            shifted_labels_list,
            actor_advantages_list,
            rollout_logprobs_list,
        ):
            actor_loss_ctx = loss_cfg.build(
                data={
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                },
                sp_mesh=self.sp_mesh,
            )
            if actor_loss_ctx is None:
                raise RuntimeError("Actor loss config did not build a loss context.")
            actor_loss_ctx_list.append(actor_loss_ctx)

        old_logprobs_list = self.compute_actor_logprobs(seq_ctx_list, shifted_labels_list)
        all_rollout_is_metrics = []
        all_mismatch_metrics = []
        for seq_ctx, old_logprobs, actor_loss_ctx in zip(seq_ctx_list, old_logprobs_list, actor_loss_ctx_list):
            actor_loss_ctx.loss_kwargs.old_logprobs = old_logprobs.detach()
            if actor_loss_ctx.loss_kwargs.rollout_logprobs is not None:
                mismatch_metrics, rollout_is_metrics = actor_loss_ctx.compute_rollout_is(
                    self.sp_mesh,
                    seq_ctx.seq_lens_q,
                )
                all_mismatch_metrics.append(mismatch_metrics)
                all_rollout_is_metrics.append(rollout_is_metrics)

        rank_actor_tokens = sum((labels != -100).sum() for labels in shifted_labels_list)
        global_actor_tokens = cast(torch.Tensor, rank_actor_tokens).clone()
        dist.all_reduce(global_actor_tokens, op=dist.ReduceOp.SUM)
        avg_train_entropy = calculate_entropy(shifted_labels_list, old_logprobs_list, global_actor_tokens)
        avg_rollout_entropy = calculate_entropy(shifted_labels_list, rollout_logprobs_list, global_actor_tokens)
        if avg_train_entropy is not None:
            worker_log_item["train_entropy"] = float(avg_train_entropy.item())
        if avg_rollout_entropy is not None:
            worker_log_item["rollout_entropy"] = float(avg_rollout_entropy.item())
        if all_mismatch_metrics:
            worker_log_item["mismatch_metrics"] = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
        if all_rollout_is_metrics:
            worker_log_item["rollout_is_metrics"] = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)

        self._engine.put_optimizer_to_device(DEVICE)
        actor_order = list(range(len(seq_ctx_list)))
        actor_chunks = self._partition_indices(actor_order, self._optimizer_steps)
        for chunk_index, indices in enumerate(actor_chunks):
            chunk_contexts = [actor_loss_ctx_list[index] for index in indices]
            global_chunk_tokens = sum(
                (context.loss_kwargs.shifted_labels != loss_cfg.ignore_idx).sum() for context in chunk_contexts
            )
            global_chunk_tokens = cast(torch.Tensor, global_chunk_tokens).clone()
            dist.all_reduce(global_chunk_tokens, op=dist.ReduceOp.SUM)
            if global_chunk_tokens.item() == 0:
                continue
            batched_contexts = loss_cfg.loss_ctx_cls.build_batches(chunk_contexts)  # type: ignore[arg-type]
            engine_input = [
                ModelItem(seq_ctx=seq_ctx_list[index], loss_ctx={"lm": loss_ctx})
                for index, loss_ctx in zip(indices, batched_contexts)
            ]
            train_step_info = self._engine.train_step(engine_input)
            grad_norm = self._engine.clip_grad_norm()
            optimizer_stepped = self._optimizer_step_succeeds(self._engine, grad_norm)
            self._engine.step_optimizer(grad_norm)
            if optimizer_stepped:
                assert self._actor_scheduler is not None
                self._actor_scheduler.step()
                self._actor_optimizer_updates += 1
            log_item = self._build_ppo_train_log(
                train_step_info,
                grad_norm,
                lr=self._engine.optimizer.param_groups[0]["lr"],
                pass_index=0,
                chunk_index=chunk_index,
                policy_metrics=True,
            )
            worker_log_item["train_metrics"].append(log_item)
            self._global_train_step += 1

        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._set_ppo_phase(PPOPhase.ACTOR_READY)
        self._rollout_step += 1
        return worker_log_item

    def _train_ppo_critic(
        self,
        critic_seq_ctx_list: list[SequenceContext],
        critic_loss_ctx_list: list[BaseLossContext],
        rollout_idx: int,
    ) -> list[WorkerTrainLogItem]:
        """Train both Critic passes while keeping their tensor lifetime local."""
        critic_engine = self._critic_engine
        critic_cfg = self.config.critic_cfg
        assert critic_engine is not None and critic_cfg is not None

        train_logs: list[WorkerTrainLogItem] = []
        critic_engine.put_optimizer_to_device(DEVICE)
        for pass_index in range(critic_cfg.num_passes):
            order = self._ppo_pass_order(len(critic_seq_ctx_list), rollout_idx, pass_index)
            chunks = self._partition_indices(order, critic_cfg.optimizer_steps_per_pass)
            for chunk_index, indices in enumerate(chunks):
                chunk_contexts = [critic_loss_ctx_list[index] for index in indices]
                global_chunk_tokens = sum(context.loss_kwargs.value_mask.sum() for context in chunk_contexts)
                global_chunk_tokens = cast(torch.Tensor, global_chunk_tokens).clone()
                dist.all_reduce(global_chunk_tokens, op=dist.ReduceOp.SUM)
                if global_chunk_tokens.item() == 0:
                    continue
                batched_contexts = critic_cfg.loss_cfg.loss_ctx_cls.build_batches(chunk_contexts)  # type: ignore[attr-defined]
                engine_input = [
                    ModelItem(seq_ctx=critic_seq_ctx_list[index], loss_ctx={"lm": loss_ctx})
                    for index, loss_ctx in zip(indices, batched_contexts)
                ]
                train_step_info = critic_engine.train_step(engine_input)
                grad_norm = critic_engine.clip_grad_norm()
                optimizer_stepped = self._optimizer_step_succeeds(critic_engine, grad_norm)
                critic_engine.step_optimizer(grad_norm)
                if optimizer_stepped:
                    assert self._critic_scheduler is not None
                    self._critic_scheduler.step()
                    self._critic_optimizer_updates += 1
                train_logs.append(
                    self._build_ppo_train_log(
                        train_step_info,
                        grad_norm,
                        lr=critic_engine.optimizer.param_groups[0]["lr"],
                        pass_index=pass_index,
                        chunk_index=chunk_index,
                    )
                )
        return train_logs

    @staticmethod
    def _partition_indices(indices: list[int], max_updates: int) -> list[list[int]]:
        if not indices:
            return []
        num_updates = min(len(indices), max_updates)
        quotient, remainder = divmod(len(indices), num_updates)
        chunks = []
        offset = 0
        for chunk_index in range(num_updates):
            chunk_size = quotient + int(chunk_index < remainder)
            chunks.append(indices[offset : offset + chunk_size])
            offset += chunk_size
        return chunks

    def _ppo_pass_order(self, num_batches: int, rollout_idx: int, pass_index: int) -> list[int]:
        if pass_index == 0:
            return list(range(num_batches))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed or 0) + rollout_idx * 1_000_003 + pass_index)
        return torch.randperm(num_batches, generator=generator).tolist()

    @staticmethod
    def _optimizer_step_succeeds(engine: TrainEngine, grad_norm: torch.Tensor) -> bool:
        if not torch.isfinite(grad_norm):
            return False
        threshold = engine.optim_cfg.skip_grad_norm_threshold
        return threshold is None or bool(grad_norm <= threshold)

    def _build_ppo_train_log(
        self,
        train_step_info: TrainStepInfo,
        grad_norm: torch.Tensor,
        *,
        lr: float,
        pass_index: int,
        chunk_index: int,
        policy_metrics: bool = False,
    ) -> WorkerTrainLogItem:
        logs_info = cast(dict[str, float], train_step_info["logs_info"])
        extra_info = train_step_info["extra_info"]
        if isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info_dict = extra_info.get()
        else:
            extra_info_dict = cast(dict, extra_info)
        extra_info_dict = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in extra_info_dict.items()
            if isinstance(value, (torch.Tensor, int, float))
        }
        if policy_metrics:
            extra_info_dict = finalize_train_policy_metrics(extra_info_dict, DEVICE)
        return WorkerTrainLogItem(
            **logs_info,  # type: ignore[typeddict-item]
            **extra_info_dict,  # type: ignore[typeddict-item]
            step_consumed_tokens=train_step_info["step_consumed_tokens"],
            efficient_attn_ratio=train_step_info["efficient_attn_ratio"],
            grad_norm=float(grad_norm.item()),
            lr=float(lr),  # type: ignore[typeddict-unknown-key]
            pass_index=float(pass_index),  # type: ignore[typeddict-unknown-key]
            chunk_index=float(chunk_index),  # type: ignore[typeddict-unknown-key]
        )

    def _set_ppo_phase(self, phase: PPOPhase) -> None:
        self._ppo_phase = phase
        self.logger.info(
            f"PPO phase={phase.value}, allocated={DEVICE_MODULE.memory_allocated() / (1024**3):.2f} GiB, "
            f"reserved={DEVICE_MODULE.memory_reserved() / (1024**3):.2f} GiB"
        )

    def _onload_critic(self) -> None:
        if self._ppo_phase not in (PPOPhase.ALL_OFFLOADED, PPOPhase.ACTOR_READY):
            raise RuntimeError(f"Cannot onload Critic from phase {self._ppo_phase.value}.")
        assert self._critic_engine is not None
        self._engine.put_model_to_device("cpu")
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._critic_engine.put_model_to_device(DEVICE)
        self._set_ppo_phase(PPOPhase.CRITIC_TRAIN)

    def _offload_critic(self) -> None:
        if self._ppo_phase != PPOPhase.CRITIC_TRAIN:
            raise RuntimeError(f"Cannot offload Critic from phase {self._ppo_phase.value}.")
        assert self._critic_engine is not None
        self._critic_engine.put_optimizer_to_device("cpu")
        self._critic_engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)

    def _onload_actor(self) -> None:
        if self._ppo_phase != PPOPhase.ALL_OFFLOADED:
            raise RuntimeError(f"Cannot onload Actor from phase {self._ppo_phase.value}.")
        self._engine.put_model_to_device(DEVICE)
        self._set_ppo_phase(PPOPhase.ACTOR_TRAIN)

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
        self._engine.step_optimizer(grad_norm)
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
        self._engine.save_hf(hf_dir, save_dtype)

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

    @ray_method
    def offload_model(self):
        self._engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        if self._critic_engine is not None:
            if self._ppo_phase == PPOPhase.CRITIC_TRAIN:
                raise RuntimeError("External Actor offload is not allowed during the Critic phase.")
            self._set_ppo_phase(PPOPhase.ALL_OFFLOADED)
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def onload_model(self):
        if self._critic_engine is not None and self._ppo_phase == PPOPhase.CRITIC_TRAIN:
            raise RuntimeError("Actor and Critic cannot be onloaded at the same time.")
        self._engine.put_model_to_device(DEVICE)
        if self._critic_engine is not None:
            self._set_ppo_phase(PPOPhase.ACTOR_READY)

    @ray_method
    def onload_optimizer(self):
        self._engine.put_optimizer_to_device(DEVICE)

    @ray_method
    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training worker."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        if self._critic_engine is not None and no_save_optimizer:
            raise ValueError("PPO strict checkpoints require both Actor and Critic optimizer states.")
        weights_path = checkpoint_path / self._SAVE_WEIGHTS_DIR

        # Save model and optimizer
        self._engine.save_dcp(
            weights_dir=weights_path,
            save_optimizer=not no_save_optimizer,
        )

        if self._critic_engine is not None:
            critic_weights_path = checkpoint_path / self._SAVE_CRITIC_WEIGHTS_DIR
            self._critic_engine.save_dcp(
                weights_dir=critic_weights_path,
                save_optimizer=True,
            )
            assert self._actor_scheduler is not None and self._critic_scheduler is not None
            state_dir = checkpoint_path / self._SAVE_PPO_STATE_DIR
            if self.rank == 0:
                state_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            worker_state = {
                "actor_scheduler": self._actor_scheduler.state_dict(),
                "critic_scheduler": self._critic_scheduler.state_dict(),
                "actor_optimizer_updates": self._actor_optimizer_updates,
                "critic_optimizer_updates": self._critic_optimizer_updates,
                "global_train_step": self._global_train_step,
                "rollout_step": self._rollout_step,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                # Ray workers can see every device on the node. Touch only the
                # rank-local generator instead of initializing CUDA contexts
                # on all visible GPUs during checkpointing.
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_available() else None
                ),
            }
            torch.save(worker_state, state_dir / f"rank-{self.rank}.pt")
            dist.barrier()

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
            strict=self._critic_engine is not None,
        )

        if self._critic_engine is not None:
            if not (
                load_checkpoint_cfg.load_optimizer_states
                and load_checkpoint_cfg.load_optimizer_args
                and load_checkpoint_cfg.load_scheduler
            ):
                raise ValueError("PPO strict resume requires optimizer states, optimizer args, and schedulers.")
            critic_weights_path = resume_from / self._SAVE_CRITIC_WEIGHTS_DIR
            if not critic_weights_path.exists():
                raise FileNotFoundError(f"PPO checkpoint has no {self._SAVE_CRITIC_WEIGHTS_DIR}/ directory.")
            self._critic_engine.load_dcp(
                weights_dir=critic_weights_path,
                load_states=True,
                load_args=True,
                strict=True,
            )
            worker_state_path = resume_from / self._SAVE_PPO_STATE_DIR / f"rank-{self.rank}.pt"
            if not worker_state_path.exists():
                raise FileNotFoundError(f"PPO worker state does not exist: {worker_state_path}")
            worker_state = torch.load(worker_state_path, map_location="cpu", weights_only=False)
            assert self._actor_scheduler is not None and self._critic_scheduler is not None
            self._actor_scheduler.load_state_dict(worker_state["actor_scheduler"])
            self._critic_scheduler.load_state_dict(worker_state["critic_scheduler"])
            self._actor_optimizer_updates = int(worker_state["actor_optimizer_updates"])
            self._critic_optimizer_updates = int(worker_state["critic_optimizer_updates"])
            self._global_train_step = int(worker_state["global_train_step"])
            self._rollout_step = int(worker_state["rollout_step"])
            random.setstate(worker_state["python_rng_state"])
            np.random.set_state(worker_state["numpy_rng_state"])
            torch.set_rng_state(worker_state["torch_rng_state"])
            if torch.cuda.is_available() and worker_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(
                    worker_state["cuda_rng_state"],
                    device=torch.cuda.current_device(),
                )
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
