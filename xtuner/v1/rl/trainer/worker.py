from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, Field
from ray.actor import ActorClass, ActorProxy
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import NotRequired

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    DistillationTrainerAdapter,
    TrainTeacherManagerConfig,
)
from xtuner.v1.rl.loss import BaseRLLossConfig
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
from xtuner.v1.utils.nccl_process_group import resume_nccl_process_groups, suspend_nccl_process_groups

from .actor_worker import ActorWorker
from .critic_worker import CriticWorker, CriticWorkerConfig
from .pack import DPRankPackIndices, TokenBatchProto
from .role_worker import RoleName


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from xtuner.v1.rl.advantage.base import AdvantageEstimator

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def get_train_seq_ctx(
    input_ids: torch.LongTensor,
    position_ids: np.ndarray | None = None,
    multimodal_train_info: dict | None = None,
) -> SequenceContext:
    """Build a CPU ``SequenceContext`` for one training sample.

    Args:
        input_ids (torch.LongTensor): Model input tokens with shape ``(1, seq_len)``.
        position_ids (np.ndarray | None): Optional position ids. A 3D array triggers
            the VLM MRoPE layout; a prompt-only position segment is extended to cover
            the whole input.
        multimodal_train_info (dict | None): Optional multimodal payload with
            ``pixel_values``, ``image_grid_thw`` and ``num_img_tokens``.

    Returns:
        SequenceContext: The CPU sequence context of the sample.
    """
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    position_ids = _to_cpu_tensor(position_ids, dtype=torch.long)
    if position_ids is not None and len(position_ids.shape) == 3:
        # Match get_rope_index_3: response text continues from a single global
        # max(T, H, W), not per-axis maxima. Per-axis max diverges when the
        # prompt ends on image tokens (T≈0 while H/W are large). The extension
        # length is derived from the position deficit so samples whose positions
        # already cover the whole input (full-sequence agentic samples) need no
        # extra shape information.
        num_extend = input_ids.size(-1) - position_ids.size(-1)
        if num_extend > 0:
            max_value = position_ids.amax()
            response_position_ids = (
                (torch.arange(1, num_extend + 1, device=position_ids.device, dtype=position_ids.dtype) + max_value)
                .view(1, 1, -1)
                .expand(3, 1, -1)
            )
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        seq_ctx.position_ids = position_ids  # type: ignore[assignment]
        assert position_ids.size(-1) == input_ids.size(-1)

    if multimodal_train_info:
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = _to_cpu_tensor(multimodal_train_info.get("image_grid_thw"), dtype=torch.long)
        num_img_tokens = multimodal_train_info.get("num_img_tokens")
        if num_img_tokens is not None:
            seq_ctx.num_img_tokens = [num_img_tokens]
    return seq_ctx


def _to_cpu_tensor(value: np.ndarray | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def requires_role(role: RoleName):
    """Decorate a worker RPC so it only runs when ``role`` is currently on the
    GPU.

    Example: ``@requires_role("actor")`` on ``actor_fit`` makes the call fail
    fast when the critic is resident instead of training on the wrong engine.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if role not in self._roles():
                raise RuntimeError(f"{fn.__name__}: role {role!r} is not configured on this worker")
            if self._current_role != role:
                raise RuntimeError(f"{fn.__name__}: requires active role {role!r}, current is {self._current_role!r}")
            return fn(self, *args, **kwargs)

        return wrapper

    return deco


class WorkerConfig(BaseModel):
    """Training worker configuration for XTuner RL.

    Configuration for RL training workers managing model training, optimization,
    and distributed computing in reinforcement learning workflows.

    Args:
        model_cfg (TransformerConfig): Model architecture configuration.
        optim_cfg (OptimConfig): Optimizer configuration for training.
        loss_cfg (BaseRLLossConfig): Policy RL loss. Critic value loss belongs
            on ``critic_cfg``.
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
        pack_strategy (Literal["legacy", "greedy", "balance", "native", "lpt"]): Index-only
            packing strategy used by the controller-side packer. ``legacy`` exactly
            reproduces the previous greedy pack, interleaved DP allocation and
            optimizer-step grouping. ``lpt`` reproduces the baseline longest-processing-time
            packer. Defaults to "legacy".
        pack_extra_buffer_size (int): Candidate buffer used by greedy chunk packing;
            zero disables gap filling. Defaults to 1000.
        pack_seed (int | None): Random seed for length-grouped packing strategies such
            as "balance". Defaults to None.
        offload_rollout_routed_experts (bool): Keep rollout routed experts on
            CPU and move each layer's slice to device during forward. Defaults to False.
        critic_cfg (CriticWorkerConfig | None): Attached critic configuration.
            The critic reuses this worker's FSDP and sequence-parallel topology;
            do not configure those fields on the critic itself.

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
    scheduler_steps: int | None = None
    sp_size: int = 1
    pack_max_length: int
    ref_load_from: str | Path | None = None
    ref_model_fsdp_cfg: FSDPConfig | None = None
    log_dir: str | Path | None = None
    update_weight_bucket_size_in_gb: float = 0.5  # 512MB
    seed: None | int = None  # if None, use RLTrainer seed
    pack_strategy: Literal["legacy", "greedy", "balance", "native", "lpt"] = "legacy"
    pack_extra_buffer_size: int = Field(default=1000, ge=0)
    pack_seed: int | None = None
    distillation_config: DistillationConfig | None = None
    profile_step: list[int] | int | None = None  # 1-based global RL train_step ids to profile.
    profile_time: bool = True
    profile_memory: bool = False
    offload_rollout_routed_experts: bool = False
    train_teacher_manager_config: TrainTeacherManagerConfig | None = None

    # sft config
    sft_dataloader_cfg: DataloaderConfig | None = None
    sft_global_batch_size: int = -1
    rollout_steps_per_sft: int = 1
    sft_loss_cfg: CELossConfig = CELossConfig()

    # The critic is hosted in this actor process. A configured critic selects
    # the critic training flow; GAE is selected by the advantage estimator.
    critic_cfg: CriticWorkerConfig | None = None

    def build(
        self,
        placement_group: PlacementGroup,
        advantage_estimator: AdvantageEstimator | None = None,
        distillation: DistillationTrainerAdapter | None = None,
    ):
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
        controller = TrainingController(
            workers=train_workers,
            advantage_estimator=advantage_estimator,
            distillation=distillation,
            has_critic=self.critic_cfg is not None,
        )
        return controller


class TrainBatchAttr(TypedDict):
    """Attributes of one training batch, derived once by the controller from
    the lightweight metadata written by AgentLoop canonicalization.

    The controller scans the batch meta and passes this single structure across the
    Ray boundary so workers materialize packs uniformly: cross-rank collectives stay
    symmetric because every worker of one batch sees the same attributes.

    rollout_idx (int): Global rollout (train step) id used for logging.
    use_3d_position_ids (bool): Whether the batch uses 3D MRoPE position ids; padding
        packs mirror this layout.
    pack_loss_keys (list[str]): Batch-level optional loss inputs (e.g.
        ``rollout_logprobs``) that must exist on every pack, including all-padding
        packs, so cross-rank metric collectives stay uniform.
    has_routed_experts (bool): Whether the batch carries routed experts; padding
        positions then receive dummy expert ids.
    """

    rollout_idx: int
    use_3d_position_ids: bool
    pack_loss_keys: list[str]
    has_routed_experts: bool


class WorkerTrainLogItem(TypedDict, total=False):
    step_consumed_tokens: int
    efficient_attn_ratio: float
    grad_norm: float
    mtp_grad_parameter_coverage: float
    max_memory: float
    reserved_memory: float


class WorkerLogItem(TypedDict):
    train_entropy: float
    teacher_timings: NotRequired[dict[str, dict[str, float]]]
    rollout_entropy: NotRequired[float]
    mismatch_metrics: NotRequired[dict[str, float]]
    rollout_is_metrics: NotRequired[dict[str, float]]
    train_metrics: list[WorkerTrainLogItem]
    sft_train_metrics: NotRequired[dict[str, float]]
    packing_conversion_time_s: NotRequired[float]
    packing_pack_time_s: NotRequired[float]


@dataclass
class TrainSample:
    """Training tensors for one rollout sample, shared by the actor and the
    critic.

    ``values``, ``returns``, and ``advantages`` start as ``None`` and are filled by later stages.
    GRPO leaves ``values`` and ``returns`` as ``None``.
    """

    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    rollout_logprobs: torch.Tensor | None
    teacher_targets: dict[str, torch.Tensor]
    # Optional terminal score. Only worker-local GAE reads it; GRPO receives its final
    # advantages from the controller.
    reward_score: float | None = None

    values: torch.Tensor | None = None  # per-token critic values, written when the critic collects values
    returns: torch.Tensor | None = None  # GAE returns, written when the critic fits
    advantages: torch.Tensor | None = None  # per-token row written when the actor builds the loss


class TrainingWorker(SingleAcceleratorWorker):
    """The only object holding Ray resources: process group, data/sp meshes,
    the batch cache, and the pack primitives. At most one role resides on the
    GPU at any time. The only seq-side API is ``pack_seq_ctx``."""

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

        # Batch cache: the whole TrainSample list, resident for the batch's lifetime.
        self._cached_samples: list[TrainSample] | None = None
        self._cached_pack_plan: DPRankPackIndices | None = None
        self._cached_batch_attr: TrainBatchAttr | None = None

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]

        # torch.compile is gated by fsdp_cfg.torch_compile; normalize the role
        # model configs once, before the roles build their engines.
        role_model_cfgs = [worker_cfg.model_cfg]
        if worker_cfg.critic_cfg is not None:
            role_model_cfgs.append(worker_cfg.critic_cfg.model_cfg)
        if not worker_cfg.fsdp_cfg.torch_compile:
            for model_cfg in role_model_cfgs:
                model_cfg.compile_cfg = False

        # Roles are built in-process; the actor engine steps aside while the
        # critic engine is built so peak memory stays at one resident model.
        self.actor = ActorWorker(worker_cfg, host=self)
        self.critic: CriticWorker | None = None
        if worker_cfg.critic_cfg is not None:
            self.actor.offload()
            DEVICE_MODULE.empty_cache()
            try:
                self.critic = CriticWorker(worker_cfg.critic_cfg, host=self)
                self.critic.offload()
            finally:
                DEVICE_MODULE.empty_cache()
                self.actor.onload()
        self._current_role: RoleName | None = "actor"

    def _roles(self) -> dict[str, Any]:
        return {role.role_name: role for role in (self.actor, self.critic) if role is not None}

    @ray_method
    def switch_role(self, role: RoleName | None) -> None:
        """Idempotently make ``role`` the sole GPU resident; None = all
        offloaded (colocation yields to rollout).

        Declarative "ensure state", no history branching: offload every other role, onload the target, and let the
        granular onload/offload methods decide (via the residency flag / engine guards) whether anything actually
        moves.
        """
        roles = self._roles()
        if role is not None and role not in roles:
            raise KeyError(f"role is not registered: {role}")
        if self._current_role is not None and self._current_role != role:
            roles[self._current_role].offload()
        if role is not None:
            roles[role].onload()
        self._current_role = role

    @ray_method
    def load_train_batch(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
    ) -> None:
        """The only data entry point into the worker; holds no role/loss
        concepts: one ``ray.get`` -> the full ``TrainSample`` cache (seq side +
        static token fields, the base copy shared by both roles) + the pack
        plan (index-only) + batch_attr. ``pixel_values`` / routed experts
        stay in ref form in the cache and are resolved per pack.
        """
        rollout_items: list[RolloutState] = ray.get(rollout_item_refs[0])
        self._cached_samples = [
            self._get_train_sample(state, batch_attr["use_3d_position_ids"]) for state in rollout_items
        ]
        del rollout_items
        self._cached_pack_plan = pack_plan
        self._cached_batch_attr = batch_attr

    @ray_method
    def release_train_batch(self) -> None:
        """Actor fit releases on its own; the controller broadcasts this in its
        ``finally`` as a fallback (critic failure path).

        Safe to call when nothing is cached.
        """
        self._cached_samples = None
        self._cached_pack_plan = None
        self._cached_batch_attr = None

    @property
    def cached_samples(self) -> list[TrainSample] | None:
        """Samples cached by ``load_train_batch``, or ``None`` when no batch is
        loaded."""
        return self._cached_samples

    def get_cached_batch(self) -> tuple[list[TrainSample], DPRankPackIndices, TrainBatchAttr]:
        """Return the samples, the pack plan, and the batch attributes cached
        by ``load_train_batch``.

        Roles read this cache. Raises when the batch has not been loaded.

        Returns:
            tuple[list[TrainSample], DPRankPackIndices, TrainBatchAttr]: The cached
                samples, pack plan, and batch attributes.
        """
        samples = self._cached_samples
        pack_plan = self._cached_pack_plan
        batch_attr = self._cached_batch_attr
        if samples is None or pack_plan is None or batch_attr is None:
            raise RuntimeError("training batch is not loaded; call load_train_batch before training")
        return samples, pack_plan, batch_attr

    def _get_train_sample(self, state: RolloutState, use_3d_position_ids: bool) -> TrainSample:
        """Convert one rollout state into a ``TrainSample``.

        Drops the last input id and shifts labels and logprobs onto the prediction
        positions. ``values``, ``returns``, and ``advantages`` stay unset.
        """
        input_ids = cast(list[int], state.input_ids)
        labels = cast(list[int], state.labels)
        shifted_labels = labels[1:]

        input_ids_t = cast(torch.LongTensor, torch.tensor(input_ids[:-1], dtype=torch.int64).unsqueeze(0))
        shifted_labels_t = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

        rollout_logprobs: torch.Tensor | None = None
        if state.logprobs is not None:
            rollout_logprobs = torch.tensor(state.logprobs[1:], dtype=torch.float32).unsqueeze(0)

        position_ids = state.position_ids
        if use_3d_position_ids and (position_ids is None or position_ids.ndim != 3):
            seq_len = input_ids_t.size(-1)
            text_position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
            # Mixed batches use three-axis MRoPE position IDs. Repeat the normal text
            # positions on every axis so text samples have the same rank as the VLM
            # samples when their sequence contexts are packed together.
            position_ids = np.broadcast_to(text_position_ids, (3, 1, seq_len)).copy()
        multimodal_train_info = cast(dict | None, state.mm_info)
        seq_ctx = get_train_seq_ctx(input_ids_t, position_ids, multimodal_train_info)
        seq_ctx.rollout_routed_experts = state.routed_experts  # type: ignore[assignment] # n,layer*expert
        return TrainSample(
            seq_ctx=seq_ctx,
            shifted_labels=shifted_labels_t,
            rollout_logprobs=rollout_logprobs,
            teacher_targets=self.actor.get_rollout_teacher_targets(state, shifted_labels=shifted_labels),
            reward_score=float(state.reward["score"])
            if state.reward is not None and "score" in state.reward
            else None,
        )

    def pack_seq_ctx(self, pack: list[int]) -> SequenceContext:
        """Cat the cached samples' seq contexts of one pack and tail-pad to
        ``pack_max_length``.

        pixel_values / routed-experts refs stay in ref form inside the ctx and
        are materialized per pack by the role's ``_resolve_refs`` and released
        with the pack; every other field is fully materialized here.
        """
        samples = self._cached_samples
        if samples is None:
            raise RuntimeError("training batch is not loaded; call load_train_batch before training")
        batch_attr = self._cached_batch_attr
        assert batch_attr is not None
        real_tokens = 0
        seq_ctx_list: list[SequenceContext] = []
        for index in pack:
            seq_ctx = samples[index].seq_ctx
            assert seq_ctx.input_ids is not None
            real_tokens += seq_ctx.input_ids.shape[1]
            seq_ctx_list.append(seq_ctx)
        padding_len = self.config.pack_max_length - real_tokens
        assert padding_len >= 0, f"Pack tokens {real_tokens} exceed pack_max_length {self.config.pack_max_length}"
        if padding_len > 0:
            seq_ctx_list.append(
                self.pack_padding_seq_ctx(
                    padding_len,
                    use_3d_position_ids=batch_attr["use_3d_position_ids"],
                    has_routed_experts=batch_attr["has_routed_experts"],
                )
            )
        return SequenceContext.cat(seq_ctx_list)

    def cat_pack_fields(
        self,
        fields: dict[str, list[torch.Tensor] | None],
        padding_len: int,
    ) -> dict[str, torch.Tensor | None]:
        """Concatenate per-sample rows of each field and tail-pad to pack
        length.

        Args:
            fields (dict[str, list[torch.Tensor] | None]): Per-sample tensors by
                field name. ``None`` marks a field absent for the whole pack and
                is passed through as ``None``.
            padding_len (int): Number of tail padding positions to append.

        Returns:
            dict[str, torch.Tensor | None]: Packed tensors keyed by field name.
        """
        packed: dict[str, torch.Tensor | None] = {}
        for key, rows in fields.items():
            if rows is None:
                packed[key] = None
                continue
            tensor = rows[0] if len(rows) == 1 else torch.cat(rows, dim=1)
            if padding_len > 0:
                tensor = torch.cat([tensor, self.pack_padding_loss_input(key, padding_len)], dim=1)
            packed[key] = tensor
        return packed

    def pack_padding_loss_input(self, key: str, padding_len: int) -> torch.Tensor:
        """Build the padding rows of one packed loss-input tensor.

        Args:
            key (str): Loss-input field name, e.g. ``"shifted_labels"`` or
                ``"teacher_logprobs"``.
            padding_len (int): Number of padding positions to materialize.

        Returns:
            torch.Tensor: Padding rows of shape ``(1, padding_len[, top_k])``.
        """
        if key == "shifted_labels":
            return torch.full((1, padding_len), -100, dtype=torch.long)
        if key in ("advantages", "rollout_logprobs", "returns", "old_values"):
            return torch.zeros(1, padding_len, dtype=torch.float32)
        if key == "teacher_indices":
            return torch.full((1, padding_len), -1, dtype=torch.long)
        if key in ("teacher_logprobs", "target_token_ids"):
            distillation_config = self.config.distillation_config
            assert distillation_config is not None, f"Padding {key} requires distillation_config"
            top_k = distillation_config.loss_config.top_k
            if key == "teacher_logprobs":
                if top_k is None:
                    # Sampled teacher targets carry one logprob per position.
                    return torch.zeros(1, padding_len, dtype=torch.float32)
                return torch.zeros(1, padding_len, top_k, dtype=torch.float32)
            assert top_k is not None, f"Padding {key} requires loss_config.top_k"
            return torch.zeros(1, padding_len, top_k, dtype=torch.long)
        raise ValueError(f"Unknown pack loss input key: {key}")

    def pack_padding_seq_ctx(
        self,
        padding_len: int,
        use_3d_position_ids: bool,
        has_routed_experts: bool,
    ) -> SequenceContext:
        """Build the padding slot of one pack as a standalone sequence context.

        Args:
            padding_len (int): Number of padding positions to materialize.
            use_3d_position_ids (bool): Whether to emit 3-axis mrope position ids.
            has_routed_experts (bool): Whether to emit dummy routed-expert ids.

        Returns:
            SequenceContext: Padding context of length ``padding_len``.
        """
        # Reduce the attn calculation time by using multiple short sequence packs
        pad_tokens = tuple(torch.zeros(1, 1024, dtype=torch.long) for _ in range(padding_len // 1024))
        if padding_len % 1024 > 0:
            pad_tokens = pad_tokens + (torch.zeros(1, padding_len % 1024, dtype=torch.long),)
        pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore[arg-type]
        pad_seq_ctx.num_padding = padding_len
        if use_3d_position_ids:
            _position_ids_list = []
            for pad_token in pad_tokens:
                _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                _position_ids_list.append(_position_ids)
            pad_seq_ctx.position_ids = torch.cat(_position_ids_list, dim=-1)  # type: ignore[assignment]
        if has_routed_experts:
            language_cfg = (
                self.config.model_cfg.text_config
                if isinstance(self.config.model_cfg, BaseComposeConfig)
                else self.config.model_cfg
            )
            pad_seq_ctx.rollout_routed_experts = torch.randint(
                low=0, high=language_cfg.n_routed_experts, size=(padding_len, 1, 1)
            )
        return pad_seq_ctx

    def _add_sample_fields(self, field: str, values: list[float] | TokenBatchProto) -> None:
        """Write controller-delivered training inputs (scalar/token advantages,
        returns) onto the cached samples by sample index."""
        samples = self._cached_samples
        if samples is None:
            raise RuntimeError("training batch is not loaded; call load_train_batch before training")
        # Branch before iterating: iterating a union of lists gives mypy a
        # joined element type (object); separate loops keep precise types.
        if isinstance(values, TokenBatchProto):
            tensor_rows = values.chunk()
            if len(tensor_rows) != len(samples):
                raise ValueError(f"rows {len(tensor_rows)} do not match samples {len(samples)}")
            for sample, row in zip(samples, tensor_rows):
                self._write_sample_field(field, sample, row)
            return
        if len(values) != len(samples):
            raise ValueError(f"rows {len(values)} do not match samples {len(samples)}")
        for sample, scalar in zip(samples, values):
            self._write_sample_field(field, sample, None, scalar=float(scalar))

    def _write_sample_field(
        self,
        field: str,
        sample: TrainSample,
        value: torch.Tensor | None,
        *,
        scalar: float | None = None,
    ) -> None:
        # Add ``field`` onto ``sample`` with ``value`` as its content: either a
        # per-token tensor row or a scalar broadcast to the sequence length.
        shifted_labels = sample.shifted_labels
        if value is not None:
            flat = value.detach().reshape(-1).float()
            if flat.numel() != shifted_labels.size(1):
                raise ValueError(
                    f"Token {field} length mismatch: {flat.numel()} does not match "
                    f"shifted labels length {shifted_labels.size(1)}"
                )
            tensor = flat.unsqueeze(0)
        else:
            assert scalar is not None
            tensor = torch.full(shifted_labels.shape, scalar, dtype=torch.float32)
        if field == "advantages":
            sample.advantages = tensor.masked_fill(shifted_labels.eq(-100), 0.0)
        elif field == "returns":
            sample.returns = tensor
        else:
            raise ValueError(f"Unknown sample field: {field}")

    @requires_role("actor")
    @ray_method
    def actor_fit(self, advantages: list[float] | TokenBatchProto | None = None) -> WorkerLogItem:
        """Train the actor on the cached batch.

        GRPO scalar / session-GAE token advantages arrive here; the actor is the batch's last consumer and releases the
        cache itself.
        """
        conversion_begin = time.perf_counter()
        if advantages is not None:
            self._add_sample_fields("advantages", advantages)  # GRPO scalars / session-GAE token rows
            conversion_time_s = time.perf_counter() - conversion_begin
        else:
            conversion_time_s = 0.0
        try:
            worker_log_item = self.actor.fit()
        finally:
            self.release_train_batch()
        worker_log_item.setdefault("packing_conversion_time_s", conversion_time_s)
        return worker_log_item

    @requires_role("critic")
    @ray_method
    def critic_collect_values(self, return_values: bool) -> TokenBatchProto | None:
        assert self.critic is not None  # requires_role("critic") guarantees it
        return self.critic.collect_values(return_values)

    @requires_role("critic")
    @ray_method
    def critic_compute_segment_gae(self, **gae_kwargs) -> dict[str, float]:
        assert self.critic is not None  # requires_role("critic") guarantees it
        return self.critic.compute_segment_gae(**gae_kwargs)

    @requires_role("critic")
    @ray_method
    def critic_fit(self, token_returns: TokenBatchProto | None = None) -> WorkerLogItem:
        assert self.critic is not None  # requires_role("critic") guarantees it
        if token_returns is not None:
            self._add_sample_fields("returns", token_returns)  # session-GAE returns rows
        return self.critic.fit()

    @ray_method
    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False) -> None:
        """Save the DCP checkpoint of every configured role."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        for name, role in self._roles().items():
            role.save(checkpoint_path / name, no_save_optimizer)
        self.actor.save_sft_state(checkpoint_path)

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig) -> None:
        """Resume every configured role from the checkpoint."""
        resume_from = load_checkpoint_cfg.checkpoint_path
        if resume_from is None:
            return
        if isinstance(resume_from, str):
            resume_from = Path(resume_from)
        self.logger.info(f"Resume from checkpoint: {resume_from}")
        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path {resume_from} does not exist.")
        for name, role in self._roles().items():
            role.resume(resume_from / name, load_checkpoint_cfg)
        self.actor.resume_sft_state(resume_from)

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16) -> None:
        """Save every configured role in Hugging Face format under
        ``hf_dir``."""
        hf_path = Path(hf_dir)
        for name, role in self._roles().items():
            role.save_hf(hf_path / name, save_dtype)

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
        return self.actor.update_weighter.bind_rollout_weight_update(*args, **kwargs)

    @ray_method
    def weight_update(self, **kwargs):
        return self.actor.update_weighter.weight_update(**kwargs)

    @ray_method
    def get_ep_gather_count(self) -> int:
        iterator = self.actor.update_weighter.weight_iterator
        return 0 if iterator is None else iterator.ep_gather_count

    @ray_method
    def has_registered_weight_checkpoint(self) -> bool:
        return self.actor.update_weighter.has_registered_weight_checkpoint()

    @ray_method
    def get_data_replicate_size(self) -> int:
        """Get the data replicate size for the training worker."""
        # tp shards each sample across ranks; sp shards within a rank --
        # replicate size multiplies both.
        return self.config.fsdp_cfg.tp_size * self.sp_mesh.size()

    @ray_method
    def get_worker_cfg(self) -> WorkerConfig:
        """Expose the worker config for controller-side pack planning."""
        return self.config

    @ray_method
    def get_model_cfg(self) -> TransformerConfig | BaseComposeConfig:
        return self.config.model_cfg

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

    @ray_method
    def offload_model(self):
        self.actor.offload_model()
        DEVICE_MODULE.empty_cache()
        self._clear_cublas_workspaces()

    @ray_method
    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        self.actor.offload_optimizer()
        DEVICE_MODULE.empty_cache()

    @ray_method
    def onload_model(self):
        self.actor.onload_model()

    @ray_method
    def onload_optimizer(self):
        self.actor.onload_optimizer()

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            self.logger.info("Setting deterministic algorithms of TrainingWorker.")
            set_deterministic()

    def _set_random_seed(self, seed: None | int):
        set_random_seed(seed)

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

    @ray_method
    def ready(self) -> bool:
        return True


TrainingWorkerClass = ActorClass[TrainingWorker]
TrainingWorkerProxy = ActorProxy[TrainingWorker]
