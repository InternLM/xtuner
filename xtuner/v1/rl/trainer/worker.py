from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

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
from .pack import RankPlan
from .role_worker import RoleContext, RoleName
from .train_batch import TrainBatch


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from xtuner.v1.rl.advantage.base import AdvantageEstimator, TokenAdvantageEstimator

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


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
            packing strategy used by the controller-side packer. Defaults to "legacy".
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
        advantage_estimator: AdvantageEstimator | TokenAdvantageEstimator | None = None,
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
    """Attributes of one training batch, derived once by the controller.

    The controller scans the batch meta and passes this single structure across the
    Ray boundary so workers materialize packs uniformly: cross-rank collectives stay
    symmetric because every worker of one batch sees the same attributes.

    rollout_idx (int): Global rollout (train step) id used for logging.
    use_3d_position_ids (bool): Whether the batch uses 3D MRoPE position ids.
    pack_loss_keys (list[str]): Batch-level optional loss inputs (e.g.
        ``rollout_logprobs``) required on every pack, including padding packs.
    has_routed_experts (bool): Whether the batch carries routed experts.
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


class TrainingWorker(SingleAcceleratorWorker):
    """The Ray process host holding the actor and optional critic role.

    The worker owns the process group, data/SP mesh, one shared ``TrainBatch``
    and the named RPC surface. Roles own their engines, schedulers and placement
    facts; they receive the batch as an argument and never read a host cache.
    """

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

        # torch.compile is gated by fsdp_cfg.torch_compile; normalize the role
        # model configs once, before the roles build their engines.
        role_model_cfgs = [worker_cfg.model_cfg]
        if worker_cfg.critic_cfg is not None:
            role_model_cfgs.append(worker_cfg.critic_cfg.model_cfg)
        if not worker_cfg.fsdp_cfg.torch_compile:
            for model_cfg in role_model_cfgs:
                model_cfg.compile_cfg = False

        ctx = RoleContext(
            data_mesh=self.data_mesh,
            device=torch.device(DEVICE),
            rank=rank,
            seed=worker_cfg.seed or 0,
            logger=self.logger,
            log_dir=self.log_dir,
        )
        # Roles are built in-process; the actor engine steps aside while the
        # critic engine is built so peak memory stays at one resident model.
        self.actor = ActorWorker(worker_cfg, ctx)
        self.critic: CriticWorker | None = None
        if worker_cfg.critic_cfg is not None:
            self.actor.offload()
            DEVICE_MODULE.empty_cache()
            try:
                self.critic = CriticWorker(worker_cfg.critic_cfg, ctx, worker_cfg.fsdp_cfg)
                self.critic.offload()
            finally:
                DEVICE_MODULE.empty_cache()
                self.actor.onload()

        self._distillation = DistillationTrainerAdapter(worker_cfg.distillation_config)
        # The single worker-local training batch; roles borrow it by argument.
        self.batch: TrainBatch | None = None

    def _roles(self) -> dict[RoleName, Any]:
        return {role.role_name: role for role in (self.actor, self.critic) if role is not None}

    @ray_method
    def set_residency(self, model_role: RoleName | None, optimizer_role: RoleName | None) -> None:
        """Make the requested roles resident and every other role offloaded.

        Args:
            model_role (RoleName | None): Role whose model must be on the device.
            optimizer_role (RoleName | None): Role whose optimizer must be on the device.

        Returns:
            None
        """
        roles = self._roles()
        if model_role not in (*roles, None) or optimizer_role not in (*roles, None):
            raise ValueError("Requested role is not configured")
        for name, role in roles.items():
            if name != optimizer_role and role.optimizer_on_device:
                role.offload_optimizer()
            if name != model_role and role.model_on_device:
                role.offload_model()
        if model_role is not None:
            roles[model_role].onload_model()
        if optimizer_role is not None:
            roles[optimizer_role].onload_optimizer()

    def _resident_model_role(self) -> RoleName | None:
        return next((name for name, role in self._roles().items() if role.model_on_device), None)

    def _resident_optimizer_role(self) -> RoleName | None:
        return next((name for name, role in self._roles().items() if role.optimizer_on_device), None)

    @ray_method
    def load_train_batch(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        rank_plan: RankPlan,
        batch_attr: TrainBatchAttr,
    ) -> None:
        """Install one local training batch; the only data entry point.

        Args:
            rollout_item_refs (list[ray.ObjectRef]): Nested ref whose element zero wraps
                this DP rank's ``RolloutState`` list, shared by all replicas.
            rank_plan (RankPlan): Local pack indices addressing the states.
            batch_attr (TrainBatchAttr): Batch-level attributes derived by the controller.

        Returns:
            None
        """
        states = ray.get(rollout_item_refs[0])
        self.batch = TrainBatch(
            states,
            rank_plan,
            batch_attr,
            self._distillation,
            pack_max_length=self.config.pack_max_length,
            model_cfg=self.config.model_cfg,
            teacher_loss_keys=self.actor.teacher_loss_keys(),
            distillation_config=self.config.distillation_config,
        )

    @ray_method
    def release_train_batch(self) -> None:
        """Release the worker-local batch; remote refs stay owned by the
        trainer.

        Safe to call when nothing is loaded.

        Returns:
            None
        """
        if self.batch is not None:
            self.batch.close()
            self.batch = None

    def _require_batch(self) -> TrainBatch:
        if self.batch is None:
            raise RuntimeError("training batch is not loaded; call load_train_batch first")
        return self.batch

    @ray_method
    def install_advantages(self, payload: list[float] | Any) -> None:
        """Install group/scalar or token-row advantages into the local batch.

        Args:
            payload (list[float] | TokenBatchProto): Scalar or token rows per local sample.

        Returns:
            None
        """
        self._require_batch().install_advantages(payload)

    @ray_method
    def install_ppo_targets(self, payload: Any) -> None:
        """Install a pair of complete advantage/return rows per local sample.

        Args:
            payload: ``(TokenBatchProto, TokenBatchProto)`` for this DP rank.

        Returns:
            None
        """
        self._require_batch().install_ppo_targets(*payload)

    @ray_method
    def actor_fit(self) -> WorkerLogItem:
        """Train the actor on the installed batch.

        Returns:
            WorkerLogItem: Actor training metrics.
        """
        self.actor.require_train_residency()
        return self.actor.fit(self._require_batch())

    @ray_method
    def critic_collect_values(self, return_values: bool) -> Any:
        """Forward the critic over the installed batch.

        Args:
            return_values (bool): Return representative rows for session GAE.

        Returns:
            TokenBatchProto | None: Per-sample value rows when requested.
        """
        if self.critic is None:
            raise RuntimeError("critic role is not configured")
        self.critic.require_model_residency()
        return self.critic.collect_values(self._require_batch(), return_values=return_values)

    @ray_method
    def critic_compute_segment_targets(self, estimator: Any, contribute: bool) -> Any:
        """Compute and install segment GAE targets on the installed batch.

        Args:
            estimator (TokenAdvantageEstimator): Segment GAE estimator.
            contribute (bool): Whether this worker is the DP representative.

        Returns:
            TokenStats: Normalized action-token statistics.
        """
        if self.critic is None:
            raise RuntimeError("critic role is not configured")
        return self.critic.compute_segment_targets(self._require_batch(), estimator, contribute=contribute)

    @ray_method
    def critic_fit(self) -> WorkerLogItem:
        """Train the critic on the installed batch.

        Returns:
            WorkerLogItem: Critic training metrics.
        """
        if self.critic is None:
            raise RuntimeError("critic role is not configured")
        self.critic.require_train_residency()
        return self.critic.fit(self._require_batch())

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
        return {
            "rank": self.rank,
            "suspended": sum("error" not in detail and not detail.get("skipped", False) for detail in details),
            "skipped": sum(bool(detail.get("skipped", False)) for detail in details),
            "details": details,
            "errors": errors,
        }

    @ray_method
    def resume_train_nccl_process_groups(self) -> dict[str, object]:
        """Resume train-side NCCL PG backends before FSDP2 runs again."""

        details = resume_nccl_process_groups()
        errors = [str(detail["error"]) for detail in details if "error" in detail]
        DEVICE_MODULE.empty_cache()
        return {
            "rank": self.rank,
            "resumed": sum("error" not in detail for detail in details),
            "details": details,
            "errors": errors,
        }

    @ray_method
    def bind_rollout_weight_update(self, *args, **kwargs):
        return self.actor.update_weighter.bind_rollout_weight_update(*args, **kwargs)

    @ray_method
    def weight_update(self, **kwargs):
        # Weight sync only needs the actor model.
        self.set_residency("actor", self._resident_optimizer_role())
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
        """Offload the resident model while keeping the resident optimizer."""
        self.set_residency(None, self._resident_optimizer_role())
        DEVICE_MODULE.empty_cache()
        self._clear_cublas_workspaces()

    @ray_method
    def offload_optimizer(self):
        """Offload the resident optimizer while keeping the resident model."""
        self.set_residency(self._resident_model_role(), None)
        DEVICE_MODULE.empty_cache()

    @ray_method
    def onload_model(self):
        """Load the actor model for weight sync."""
        self.set_residency("actor", self._resident_optimizer_role())

    @ray_method
    def onload_optimizer(self):
        """Load the actor optimizer for training."""
        self.set_residency(self._resident_model_role() or "actor", "actor")

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
