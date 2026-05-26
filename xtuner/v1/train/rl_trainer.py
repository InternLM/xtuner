import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any, List, cast

import ray
import torch
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1._writer import get_writer
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.rl.advantage import BaseAdvantageConfig, GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    ProduceBatchResult,
    ProduceBatchStatus,
)
from xtuner.v1.rl.agent_loop_manager.producer import default_should_continue_fn
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.gateway.config import GatewayConfig
from xtuner.v1.rl.replay_buffer import (
    AsyncReplayBufferConfig,
    SyncReplayBufferConfig,
    _restore_nested_objectrefs,
    _snapshot_nested_objectrefs,
)
from xtuner.v1.rl.rollout.controller import RolloutControllerProxy
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.worker import WorkerConfig, WorkerLogItem
from xtuner.v1.rl.utils import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourceManager,
    asyncio_run,
    create_task,
    set_cpu_resource_manager,
    sort_rollout_state_for_deterministic,
)
from xtuner.v1.train.trainer import LoadCheckpointConfig, XTunerMeta
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, is_hf_model_path, set_deterministic, timer
from xtuner.v1.utils.device import get_device, get_torch_device_module
from xtuner.v1.utils.env_check import get_rollout_engine_version


# TODO: Move DEVICE to `xtuner.utils.device`
PG_READY_TIMEOUT = 30
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def check_fa3():
    if os.environ.get("XTUNER_USE_FA3", "0") != "1":
        return

    try:
        from xtuner.v1.ops.flash_attn import get_flash_attn_varlen

        get_flash_attn_varlen()
    except RuntimeError as e:
        raise RuntimeError(f"Flash attention v3 runtime error {e}, Please install it first or set XTUNER_USE_FA3=0.")


def bind_train_rollout(
    train_controller: TrainingController,
    rollout_controller: RolloutControllerProxy,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())  # type: ignore[attr-defined]
    train_controller.update_rollout_info(info_dict)
    return


def _parse_debug_rollout_step(path: Path) -> int:
    match = re.fullmatch(r"debug_rollout_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Unexpected debug rollout file name: {path}")
    return int(match.group(1))


class TrainInfo(TypedDict, total=False):
    data_info: dict[str, float]
    workers_log_item: list[WorkerLogItem]


@dataclass(frozen=True)
class RLThroughputBenchmark:
    """Throughput metrics exported by RL trainer.

    Keep this dataclass focused on concise, user-facing throughput signals.
    Large counters and intermediate rates used only for computation should stay
    as local variables in `_compute_benchmark_metrics`.

    Metrics:
        sgs means samples per GPU per second, and tgs means tokens per GPU per second.

        e2e_effective_sgs: Run-level E2E effective sample throughput per train
            worker/GPU.
            It uses cumulative training-consumed samples since RL training start
            divided by elapsed wall time from RL training start to current step
            log time and train worker count.
        e2e_effective_tgs: Per-train-worker run-level E2E effective token
            throughput. It uses cumulative training-consumed tokens since RL
            training start divided by run-level E2E elapsed time and train
            worker count.
        effective_sgs: Current step effective sample throughput per train
            worker/GPU. It
            uses samples consumed by the current training step divided by the
            full current step wall time, including rollout/get, prepare,
            training, sync/save/eval phases that run inside the step timer,
            and train worker count.
        effective_tgs: Per-train-worker current step effective token throughput.
            It uses tokens consumed by the current training step divided by the
            full current step wall time and train worker count.
        training_tgs: Per-train-worker training-only token throughput. It uses
            current step training-consumed tokens divided by `train_controller.fit`
            time and train worker count.
        rollout_sgs: Rollout sample throughput per rollout worker/GPU. It uses samples
            produced by the current rollout window divided by producer
            `produce_batch` wall time and rollout worker count.
        rollout_tgs: Per-rollout-worker rollout token throughput. It uses
            response tokens produced by the current rollout window divided by
            producer `produce_batch` wall time and rollout worker count.
    """

    e2e_effective_sgs: float
    e2e_effective_tgs: float
    effective_sgs: float
    effective_tgs: float
    training_tgs: float
    rollout_sgs: float
    rollout_tgs: float

    def to_scalars(self) -> dict[str, float]:
        return {f"throughput/{key}": value for key, value in asdict(self).items()}


def get_train_seq_ctx(
    input_ids: torch.LongTensor,
    position_ids: torch.Tensor | None = None,
    multimodal_train_info: dict | None = None,
    len_response_ids: int = 0,
):
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    if position_ids is not None and len(position_ids.shape) == 3:
        # qwen3vl 需要特殊处理，其余的不需要额外处理
        max_value = position_ids.max(dim=-1).values  # (3,1)
        response_position_ids = max_value.unsqueeze(-1).expand(-1, -1, len_response_ids) + torch.arange(
            1, len_response_ids + 1, device=max_value.device
        )
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        seq_ctx.position_ids = position_ids  # type: ignore[assignment]
        assert position_ids.size(-1) == input_ids.size(-1)

    if multimodal_train_info:
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = multimodal_train_info.get("image_grid_thw")
    return seq_ctx


def is_valid_for_training(group_data_items: list[RolloutState], logger) -> bool:
    """Checks if a group of rollout states is valid for a training step.

    Args:
        group_data_items: A list of RolloutState objects.

    Returns:
        True if the group is valid, False otherwise.

    NOTE: Why this check is needed:
    - For system fault tolerance, this check is performed at rollout / dataflow
      time, but we still do it here to ensure training data integrity.
    - 'filtered'/'failed': These items are fundamentally broken or incomplete and
      should not be used for training.
    - 'aborted': These items represent rollouts that were stopped
      prematurely. Using such partial data could lead the model to learn
      undesirable behaviors (e.g., stopping generation too early).
    - Empty response/response_ids: The model's generated response is the core
      of the training data for RL algorithms like PPO. If the response is
      missing, there is nothing to compute rewards on or to train the model with.
    """
    is_abort = any(item.status == Status.ABORTED for item in group_data_items)
    is_filtered = any(item.status == Status.FILTERED for item in group_data_items)
    is_failed = any(item.status == Status.FAILED for item in group_data_items)
    if is_filtered or is_failed or is_abort:
        logger.warning(
            f"Invalid dataflow group found during training, rollout state filtered: {is_filtered}, failed: {is_failed}, aborted: {is_abort}."
        )
        return False
    for item in group_data_items:
        response_valid = item.response is not None and len(item.response) > 0
        ids_valid = item.response_ids is not None and len(item.response_ids) > 0
        if not ids_valid:
            # NOTE: `response_ids` is the critical field for token-in-token-out mode, so we ensure it's not empty.
            logger.warning(
                "Invalid dataflow item found during training: no response or response_ids and skip this item."
            )
            return False
        if not response_valid:
            # NOTE: check valid response string for judger inputs
            logger.warning("Invalid dataflow item found during training: empty response string and skip this item.")
            return False
    return True


def _validate_sync_intervals(
    sync_weights_interval: int,
    checkpoint_interval: int | None,
    hf_interval: int | None,
    evaluate_step: int | None = None,
    enable_evaluate: bool = False,
) -> None:
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    for name, interval in (
        ("checkpoint_interval", checkpoint_interval),
        ("hf_interval", hf_interval),
    ):
        if interval is None or interval == -1:
            continue
        if interval <= 0:
            raise ValueError(f"{name} must be positive or -1/None to disable it, got {interval}.")
        if interval % sync_weights_interval != 0:
            raise ValueError(
                f"{name}={interval} must be a multiple of sync_weights_interval={sync_weights_interval}, "
                "because checkpoint/HF saves only run on weight-sync steps."
            )

    if enable_evaluate:
        if evaluate_step is None or evaluate_step <= 0:
            raise ValueError(f"evaluate_step must be positive when evaluation is enabled, got {evaluate_step}.")
        if evaluate_step % sync_weights_interval != 0:
            raise ValueError(
                f"evaluate_step={evaluate_step} must be a multiple of "
                f"sync_weights_interval={sync_weights_interval}, because evaluation only runs on weight-sync steps."
            )


class BaseRLTrainerConfig(BaseModel):
    """Base configuration shared by XTuner RL trainers.

    This base class defines the common training, rollout, evaluation, checkpoint, and logging fields used by both
    colocated and disaggregated RL trainers. Concrete trainer configs add their resource layout fields.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    tokenizer_path: str | Path
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: AgentLoopManagerConfig
    eval_agent_loop_manager_cfg: AgentLoopManagerConfig | None = None
    evaluator_config: EvaluatorConfig | None = None
    load_from: str | Path
    total_train_steps: int | None = None
    total_epochs: int | None = None
    train_batch_size: int
    advantage_estimator_config: BaseAdvantageConfig = Field(default_factory=GRPOAdvantageConfig)
    sync_weights_interval: int = 1
    gateway_config: GatewayConfig | None = None

    enable_evaluate: bool = True
    enable_initial_evaluate: bool = False
    evaluate_step: int = 1
    work_dir: Path | str | None = None
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    hf_interval: int | None = -1
    hf_max_keep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    log_dir: Path | str | None = None
    seed: int = 42
    debug_rollout: bool = False
    debug_rollout_dir: Path | str | None = None
    debug_train: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"

    @model_validator(mode="after")
    def _validate_sync_intervals(self):
        if self.debug_rollout and self.debug_train:
            raise ValueError("debug_rollout and debug_train cannot be enabled at the same time.")
        if self.debug_rollout and self.debug_rollout_dir is None:
            raise ValueError("debug_rollout_dir must be provided when debug_rollout=True.")
        if self.debug_train and self.debug_rollout_dir is None:
            raise ValueError("debug_rollout_dir must be provided when debug_train=True.")
        if not self.debug_train and self.total_train_steps is None and self.total_epochs is None:
            raise ValueError("Either total_train_steps or total_epochs must be provided.")
        if self.total_train_steps is not None and self.total_train_steps <= 0:
            raise ValueError(f"total_train_steps must be positive, got {self.total_train_steps}.")
        if self.total_epochs is not None and self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {self.total_epochs}.")
        _validate_sync_intervals(
            sync_weights_interval=self.sync_weights_interval,
            checkpoint_interval=self.checkpoint_interval,
            hf_interval=self.hf_interval,
            evaluate_step=self.evaluate_step,
            enable_evaluate=self.enable_evaluate,
        )
        return self


class RLColocateTrainerConfig(BaseRLTrainerConfig):
    """Configuration for the colocated RL trainer.

    ``RLColocateTrainerConfig`` runs training workers and rollout workers on a
    shared accelerator resource pool. It is typically used when rollout and
    training alternate on the same set of devices.

    Args:
        train_worker_cfg (WorkerConfig): Training worker configuration,
            including model, optimizer, loss, and FSDP settings.
        rollout_config (RolloutConfig): Rollout backend configuration.
        tokenizer_path (str | Path): Tokenizer path used by the agent loop
            sampler and rollout processing.
        replay_buffer_config (SyncReplayBufferConfig | AsyncReplayBufferConfig):
            Replay buffer configuration. Defaults to ``SyncReplayBufferConfig``.
        agent_loop_manager_cfg (AgentLoopManagerConfig): Agent loop manager
            configuration used for training rollout production.
        eval_agent_loop_manager_cfg (AgentLoopManagerConfig | None): Optional
            agent loop manager for evaluation. Defaults to None.
        evaluator_config (EvaluatorConfig | None): Optional evaluator
            configuration. Defaults to None.
        load_from (str | Path): Initial checkpoint or model path to load.
        total_train_steps (int | None): Total number of training steps.
            Defaults to None.
        total_epochs (int | None): Total number of dataset epochs. Defaults to
            None.
        train_batch_size (int): Number of rollout samples consumed per training
            step.
        advantage_estimator_config (BaseAdvantageConfig): Advantage estimator
            configuration. Defaults to ``GRPOAdvantageConfig``.
        sync_weights_interval (int): Interval, in train steps, for syncing
            weights from training to rollout. Defaults to 1.
        gateway_config (GatewayConfig | None): Optional gateway configuration.
            Defaults to None.
        enable_evaluate (bool): Whether to run evaluation. Defaults to True.
        enable_initial_evaluate (bool): Whether to evaluate before training.
            Defaults to False.
        evaluate_step (int): Evaluation interval in train steps. Defaults to 1.
        work_dir (Path | str | None): Directory for checkpoints and runtime
            state. Defaults to None.
        auto_resume (bool): Whether to resume automatically from ``work_dir``.
            Defaults to False.
        load_checkpoint_cfg (LoadCheckpointConfig): Checkpoint loading policy.
            Defaults to ``LoadCheckpointConfig()``.
        checkpoint_interval (int | None): Native checkpoint interval. Defaults
            to -1.
        checkpoint_maxkeep (int | None): Maximum number of native checkpoints
            to keep. Defaults to -1.
        hf_interval (int | None): Hugging Face checkpoint export interval.
            Defaults to -1.
        hf_max_keep (int | None): Maximum number of Hugging Face checkpoints to
            keep. Defaults to -1.
        checkpoint_no_save_optimizer (bool): Whether to skip optimizer states
            when saving checkpoints. Defaults to False.
        log_dir (Path | str | None): Directory for logs. Defaults to None.
        seed (int): Global random seed. Defaults to 66.
        debug_rollout (bool): Whether to enable rollout debugging. Defaults to
            False.
        skip_checkpoint_validation (bool): Whether to skip checkpoint
            validation. Defaults to False.
        exp_tracker (Literal["tensorboard", "jsonl"]): Experiment tracker type.
            Defaults to "tensorboard".
        resources (AcceleratorResourcesConfig): Shared accelerator resources
            used by both training and rollout workers.

    **Examples:**

    Example colocated trainer configuration::

        config = RLColocateTrainerConfig(
            train_worker_cfg=train_worker_cfg,
            rollout_config=rollout_config,
            tokenizer_path="Qwen/Qwen3-8B",
            agent_loop_manager_cfg=agent_loop_manager_cfg,
            load_from="Qwen/Qwen3-8B",
            total_train_steps=1000,
            train_batch_size=128,
            resources=AcceleratorResourcesConfig(num_workers=8),
        )
    """

    resources: AcceleratorResourcesConfig

    def build(self) -> "RLColocateTrainer":
        return RLColocateTrainer(self)


class RLDisaggregatedTrainerConfig(BaseRLTrainerConfig):
    """Configuration for the disaggregated RL trainer.

    ``RLDisaggregatedTrainerConfig`` uses separate accelerator resource pools
    for training and rollout. It is typically used when rollout production runs
    concurrently with training on dedicated devices.

    Args:
        train_worker_cfg (WorkerConfig): Training worker configuration,
            including model, optimizer, loss, and FSDP settings.
        rollout_config (RolloutConfig): Rollout backend configuration.
        tokenizer_path (str | Path): Tokenizer path used by the agent loop
            sampler and rollout processing.
        replay_buffer_config (SyncReplayBufferConfig | AsyncReplayBufferConfig):
            Replay buffer configuration. Defaults to ``SyncReplayBufferConfig``.
        agent_loop_manager_cfg (AgentLoopManagerConfig): Agent loop manager
            configuration used for training rollout production.
        eval_agent_loop_manager_cfg (AgentLoopManagerConfig | None): Optional
            agent loop manager for evaluation. Defaults to None.
        evaluator_config (EvaluatorConfig | None): Optional evaluator
            configuration. Defaults to None.
        load_from (str | Path): Initial checkpoint or model path to load.
        total_train_steps (int | None): Total number of training steps.
            Defaults to None.
        total_epochs (int | None): Total number of dataset epochs. Defaults to
            None.
        train_batch_size (int): Number of rollout samples consumed per training
            step.
        advantage_estimator_config (BaseAdvantageConfig): Advantage estimator
            configuration. Defaults to ``GRPOAdvantageConfig``.
        sync_weights_interval (int): Interval, in train steps, for syncing
            weights from training to rollout. Defaults to 1.
        gateway_config (GatewayConfig | None): Optional gateway configuration.
            Defaults to None.
        enable_evaluate (bool): Whether to run evaluation. Defaults to True.
        enable_initial_evaluate (bool): Whether to evaluate before training.
            Defaults to False.
        evaluate_step (int): Evaluation interval in train steps. Defaults to 1.
        work_dir (Path | str | None): Directory for checkpoints and runtime
            state. Defaults to None.
        auto_resume (bool): Whether to resume automatically from ``work_dir``.
            Defaults to False.
        load_checkpoint_cfg (LoadCheckpointConfig): Checkpoint loading policy.
            Defaults to ``LoadCheckpointConfig()``.
        checkpoint_interval (int | None): Native checkpoint interval. Defaults
            to -1.
        checkpoint_maxkeep (int | None): Maximum number of native checkpoints
            to keep. Defaults to -1.
        hf_interval (int | None): Hugging Face checkpoint export interval.
            Defaults to -1.
        hf_max_keep (int | None): Maximum number of Hugging Face checkpoints to
            keep. Defaults to -1.
        checkpoint_no_save_optimizer (bool): Whether to skip optimizer states
            when saving checkpoints. Defaults to False.
        log_dir (Path | str | None): Directory for logs. Defaults to None.
        seed (int): Global random seed. Defaults to 66.
        debug_rollout (bool): Whether to enable rollout debugging. Defaults to
            False.
        skip_checkpoint_validation (bool): Whether to skip checkpoint
            validation. Defaults to False.
        exp_tracker (Literal["tensorboard", "jsonl"]): Experiment tracker type.
            Defaults to "tensorboard".
        train_resources (AcceleratorResourcesConfig): Accelerator resources for
            training workers.
        rollout_resources (AcceleratorResourcesConfig): Accelerator resources
            for rollout workers.

    **Examples:**

    Example disaggregated trainer configuration::

        config = RLDisaggregatedTrainerConfig(
            train_worker_cfg=train_worker_cfg,
            rollout_config=rollout_config,
            tokenizer_path="Qwen/Qwen3-8B",
            agent_loop_manager_cfg=agent_loop_manager_cfg,
            load_from="Qwen/Qwen3-8B",
            total_train_steps=1000,
            train_batch_size=128,
            train_resources=AcceleratorResourcesConfig(num_workers=4),
            rollout_resources=AcceleratorResourcesConfig(num_workers=4),
        )
    """

    train_resources: AcceleratorResourcesConfig
    rollout_resources: AcceleratorResourcesConfig

    def build(self) -> "RLDisaggregatedTrainer":
        return RLDisaggregatedTrainer(self)


class BaseRLTrainer:
    _EXP_TRACKING_PATH = "exp_tracking"
    _CHECKPOINT_DIR = "checkpoints"
    _HF_DIR = "hf"
    _SAVE_TRAIN_STATE_PATH = "train_state.json"

    train_controller: TrainingController
    rollout_controller: RolloutControllerProxy
    _debug_train_files: dict[int, Path]

    def _init_common(self, cfg: BaseRLTrainerConfig, *, meta_path: str, logger_tag: str) -> None:
        check_fa3()
        self._init_work_dir_and_meta(cfg, meta_path)
        self._init_load_source(cfg)
        self._init_save_config(cfg)
        log_dir = self._init_logger(cfg, logger_tag)
        self._save_runtime_environment(log_dir)
        self._init_train_state(cfg)
        self._init_train_worker_config(cfg, log_dir)
        self._init_rollout_config(cfg, log_dir)
        self._ensure_rollout_http_concurrency(cfg)
        self._init_runtime_flags(cfg)
        self._advantage_estimator = cfg.advantage_estimator_config.build()
        self._cpu_resource_manager: CPUResourceManager | None = None
        self._num_workers = 1.0
        self._rollout_num_workers = 1.0
        self._benchmark_start_time_s: float | None = None
        self._benchmark_training_samples: int = 0
        self._benchmark_training_tokens: int = 0

        self._exp_tracker = get_writer(writer_type=cfg.exp_tracker, log_dir=log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = False

    def _init_work_dir_and_meta(self, cfg: BaseRLTrainerConfig, meta_path: str) -> None:
        work_dir = Path(cfg.work_dir) if cfg.work_dir else Path.cwd() / "work_dirs"
        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
        self._meta = XTunerMeta.build(work_dir, meta_path, cfg.auto_resume)
        self._meta_path = meta_path

    def _init_load_source(self, cfg: BaseRLTrainerConfig) -> None:
        self._load_from = Path(cfg.load_from) if isinstance(cfg.load_from, str) else cfg.load_from
        is_hf_path, error_info = is_hf_model_path(cfg.load_from) if cfg.load_from is not None else (False, "")
        self._load_from_hf = is_hf_path
        if not self._load_from_hf:
            raise NotImplementedError(error_info)

    def _init_save_config(self, cfg: BaseRLTrainerConfig) -> None:
        self._hf_max_keep = cfg.hf_max_keep
        self._hf_interval = cfg.hf_interval

        self._checkpoint_interval = cfg.checkpoint_interval
        self._checkpoint_maxkeep = cfg.checkpoint_maxkeep
        self._checkpoint_no_save_optimizer = cfg.checkpoint_no_save_optimizer
        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(cfg.auto_resume, cfg.load_checkpoint_cfg)

    def _init_logger(self, cfg: BaseRLTrainerConfig, logger_tag: str) -> Path:
        log_dir = self.exp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(log_dir=log_dir, tag=logger_tag)

        if cfg.skip_checkpoint_validation:
            patch_default_save_plan()
        return log_dir

    def _save_runtime_environment(self, log_dir: Path) -> None:
        if get_rank() != 0:
            return

        env_path = log_dir / "env.json"
        environment_variables = dict(os.environ)
        infer_engine_version = get_rollout_engine_version()
        environment_variables.update(infer_engine_version)
        with env_path.open("w") as f:
            json.dump(environment_variables, f, indent=2)

    def _init_train_state(self, cfg: BaseRLTrainerConfig) -> None:
        self._total_train_steps = cfg.total_train_steps or 0
        self._total_epochs = cfg.total_epochs
        self._cur_step = 0
        self._global_train_step = 0
        self._seed = cfg.seed
        self.train_batch_size = cfg.train_batch_size
        self._sync_weights_interval = cfg.sync_weights_interval
        set_deterministic()
        set_random_seed(cfg.seed)

    def _init_train_worker_config(self, cfg: BaseRLTrainerConfig, log_dir: Path) -> None:
        if cfg.train_worker_cfg.seed is None:
            self.logger.warning(f"RLTrainer seed {cfg.seed} is used as train worker seed.")
            cfg.train_worker_cfg.seed = cfg.seed
        cfg.train_worker_cfg.load_from = cfg.load_from
        cfg.train_worker_cfg.log_dir = log_dir
        self._train_worker_cfg = cfg.train_worker_cfg

    def _init_rollout_config(self, cfg: BaseRLTrainerConfig, log_dir: Path) -> None:
        cfg.rollout_config.worker_log_dir = log_dir
        if self._load_checkpoint_cfg.checkpoint_path is not None:
            cfg.rollout_config.skip_load_weights = True
            self.logger.info(
                f"Skip load rollout weights due to resume from checkpoint {self._load_checkpoint_cfg.checkpoint_path}"
            )
        self._rollout_config = cfg.rollout_config

    def _ensure_rollout_http_concurrency(self, cfg: BaseRLTrainerConfig) -> None:
        rollout_max_batch_size = cfg.rollout_config.rollout_max_batch_size_per_instance
        if rollout_max_batch_size is None or rollout_max_batch_size <= 0:
            return

        if isinstance(cfg, RLDisaggregatedTrainerConfig):
            rollout_worker_count = cfg.rollout_resources.num_workers
        elif isinstance(cfg, RLColocateTrainerConfig):
            rollout_worker_count = cfg.resources.num_workers
        else:
            rollout_worker_count = 1
        active_rollout_worker_count, _ = cfg.rollout_config.get_active_servers_count(rollout_worker_count)
        if active_rollout_worker_count <= 0:
            return

        tasks = cfg.agent_loop_manager_cfg.tasks
        task_cfgs = tasks if isinstance(tasks, list) else [tasks]
        total_weight = sum(task.weight for task in task_cfgs)
        if total_weight <= 0:
            return

        scheduled_http_requests = 0.0
        for task in task_cfgs:
            task_batch_size = cfg.train_batch_size * task.weight / total_weight
            over_sample_threshold = float(getattr(task.produce_strategy_config, "over_sample_threshold", 0.0))
            scheduled_http_requests += (
                task_batch_size * task.sampler_config.prompt_repeat_k * (1 + over_sample_threshold)
            )

        required_http_concurrency = math.ceil(scheduled_http_requests / active_rollout_worker_count)
        current_http_concurrency = math.ceil(rollout_max_batch_size * cfg.rollout_config.allow_over_concurrency_ratio)
        if current_http_concurrency >= required_http_concurrency:
            return

        new_ratio = required_http_concurrency / rollout_max_batch_size
        cfg.rollout_config.allow_over_concurrency_ratio = new_ratio
        self.logger.warning(
            "Increasing rollout_config.allow_over_concurrency_ratio because httpx max_connections is smaller "
            "than the expected per-worker rollout request concurrency: "
            f"max_connections={current_http_concurrency}, "
            f"required_connections={required_http_concurrency}"
        )

    def _init_runtime_flags(self, cfg: BaseRLTrainerConfig) -> None:
        self._enable_evaluate = cfg.enable_evaluate
        self._enable_initial_evaluate = cfg.enable_initial_evaluate and cfg.enable_evaluate
        self._evaluate_step = cfg.evaluate_step
        self._debug_rollout = cfg.debug_rollout
        self._debug_rollout_dir = Path(cfg.debug_rollout_dir) if cfg.debug_rollout_dir is not None else None
        self._debug_train = cfg.debug_train
        self._debug_train_files: dict[int, Path] = {}

    def _maybe_start_gateway(self, cfg: BaseRLTrainerConfig) -> None:
        if cfg.gateway_config is None or not cfg.gateway_config.auto_start:
            return
        # gateway 依赖 rollout controller，因此在 rollout controller 构建完成后统一启动。
        ray.get(self.rollout_controller.start_gateway.remote(cfg.gateway_config))

    def _build_agent_loop_components(self, cfg: BaseRLTrainerConfig, replay_buffer) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)
        self.agent_loop_manager = cfg.agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )

        if self._enable_evaluate:
            assert cfg.eval_agent_loop_manager_cfg is not None
            self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build(
                rollout_controller=self.rollout_controller,
                tokenizer=self.tokenizer,
                replay_buffer=replay_buffer,
                logger=self.logger,
                sync_weights_interval=cfg.sync_weights_interval,
            )

            total_eval_samples = len(self.eval_agent_loop_manager.data_sampler)
            assert cfg.evaluator_config is not None
            self.evaluator = cfg.evaluator_config.build(total_eval_samples=total_eval_samples)
        self._resolve_total_train_steps(cfg)

    def _resolve_total_train_steps(self, cfg: BaseRLTrainerConfig) -> None:
        if cfg.total_train_steps is not None:
            self._total_train_steps = cfg.total_train_steps
            return

        assert cfg.total_epochs is not None
        dataset_size = len(self.agent_loop_manager.data_sampler)
        self._total_train_steps = dataset_size // cfg.train_batch_size * cfg.total_epochs
        self.logger.info(
            "Resolved total_train_steps from total_epochs: "
            f"dataset_size={dataset_size}, train_batch_size={cfg.train_batch_size}, "
            f"total_epochs={cfg.total_epochs}, total_train_steps={self._total_train_steps}"
        )

    @property
    def exp_dir(self) -> Path:
        return Path(self._meta.latest_exp.exp_dir)

    def _resolve_load_checkpoint_cfg(
        self, auto_resume: bool, load_checkpoint_cfg: LoadCheckpointConfig
    ) -> LoadCheckpointConfig:
        """Resolve checkpoint path for auto-resume."""
        latest_checkpoint = self._meta.latest_exp.latest_checkpoint
        if latest_checkpoint is not None and auto_resume:
            load_checkpoint_cfg.checkpoint_path = Path(latest_checkpoint)
        return load_checkpoint_cfg

    def _resume_train_controller_and_state(self, checkpoint_path: Path | str) -> Path:
        # 子类只复用训练 worker 和 train_state 恢复，权重同步流程各自维护。
        self.logger.info(f"Resume train controller and state from {checkpoint_path}")
        checkpoint_path = Path(checkpoint_path)
        self.train_controller.resume(self._load_checkpoint_cfg)

        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH
        with train_state_path.open("r") as f:
            train_state = json.load(f)
        self._cur_step = train_state["cur_step"]
        return checkpoint_path

    async def _resume_agent_loop_manager(self, checkpoint_path: Path | str) -> int:
        self.logger.info(f"Resume agent_loop_manager from {checkpoint_path}")
        checkpoint_path = Path(checkpoint_path)
        # asyncio_run 只能出现在 trainer 的同步边界：
        # - colocate 的 __init__/fit/_sync_weights_and_save 仍是同步入口，可以显式包一层；
        # - disaggregated 的 _fit 已经在 asyncio_run 启动的事件循环里，内部必须全程 await。
        # 因此 agent_loop_manager / replay_buffer 的 save/resume 必须保持 async；如果它们内部再调用
        # asyncio_run，save/resume 会在 disaggregated 训练循环里触发 nested asyncio_run 失败。
        saved_model_step = await self.agent_loop_manager.resume(checkpoint_path)
        return saved_model_step

    async def _maybe_save_checkpoint(self, cur_step: int) -> None:
        """Save checkpoint if interval condition is met."""
        ckp_interval = self._checkpoint_interval
        if ckp_interval is None or ckp_interval == -1:
            return
        if cur_step % ckp_interval != 0:
            return

        checkpoint_path = self.exp_dir / self._CHECKPOINT_DIR / f"ckpt-step-{cur_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 1. Save sampler (dataloader) state
        self.logger.info(f"Saving sampler state to {checkpoint_path}")
        # 保持 manager checkpoint 的 async 调用链。
        # 是否 asyncio_run 只由 trainer 最外层同步入口统一决定。
        await self.agent_loop_manager.save(checkpoint_path, model_step=cur_step)

        # 2. Save DCP checkpoint (model + optimizer)
        self.logger.info(f"Saving DCP checkpoint to {checkpoint_path}")
        self.train_controller.save(str(checkpoint_path), self._checkpoint_no_save_optimizer)

        # 3. Save train state JSON
        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH
        with train_state_path.open("w") as f:
            json.dump({"cur_step": cur_step}, f)

        # 4. Update meta
        current_exp = self._meta.latest_exp
        current_exp.checkpoint_list.append(str(checkpoint_path))

        # 5. Prune old checkpoints
        ckp_maxkeep = self._checkpoint_maxkeep
        ckp_list = current_exp.checkpoint_list
        if ckp_maxkeep is not None and ckp_maxkeep > 0 and len(ckp_list) > ckp_maxkeep:
            for deleted in ckp_list[:-ckp_maxkeep]:
                if Path(deleted).exists():
                    rmtree(deleted, ignore_errors=True)
            current_exp.checkpoint_list = ckp_list[-ckp_maxkeep:]

        # 6. Persist meta to disk
        meta_path = self.exp_dir.parent / self._meta_path
        with meta_path.open("w") as f:
            f.write(self._meta.model_dump_json(indent=2))

    def _maybe_save_hf(self, cur_step: int):
        if self._hf_interval is None or self._hf_interval == -1:
            return

        if not self._load_from_hf:
            raise RuntimeError(
                "Only support saving to Huggingface format when loading from Huggingface! "
                "You meet this error means `load_from` of trainer is not a Huggingface model path."
            )

        if cur_step % self._hf_interval != 0 and cur_step != self._total_train_steps:
            return

        save_hf_path = self.exp_dir / self._HF_DIR / f"hf-step-{cur_step}"
        save_hf_path.mkdir(parents=True, exist_ok=True)

        # update meta
        current_exp = self._meta.latest_exp
        current_exp.hf_checkpoint_list.append(str(save_hf_path))

        # save hf
        self.logger.info(f"Saving Huggingface checkpoint to {save_hf_path}")
        hf_list = self._meta.latest_exp.hf_checkpoint_list
        if self._hf_max_keep is not None and self._hf_max_keep > 0 and len(hf_list) > self._hf_max_keep:
            for deleted in hf_list[: -self._hf_max_keep]:
                if Path(deleted).exists():
                    rmtree(deleted, ignore_errors=True)
            current_exp.hf_checkpoint_list = hf_list[-self._hf_max_keep :]
        self.train_controller.save_hf(str(save_hf_path))

        # save tokenizer
        if isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer.save_pretrained(str(save_hf_path))

    async def _run_initial_evaluate(self) -> None:
        eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
            self.evaluator.eval_batch_size,
            train_step=1,
            model_step=0,
        )
        if XTUNER_DETERMINISTIC:
            eval_produce_result.rollout_states = sort_rollout_state_for_deterministic(
                eval_produce_result.rollout_states
            )
        eval_metrics = self.evaluator.run(eval_produce_result.rollout_states)
        self.logger.info(f"Initial rollout evaluate scores {eval_metrics} and start training")
        tb_scores = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self._exp_tracker.add_scalars(tag_scalar_dict=tb_scores, global_step=0)

    def _train_one_batch(
        self,
        train_batch: list[list[RolloutState]],
        train_step: int,
        step_timer_dict: dict,
        *,
        offload_rollout_before_train: bool = False,
        onload_train_before_train: bool = False,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> TrainInfo:
        train_sample_count = sum(len(group) for group in train_batch)
        self.logger.info(f"generate {train_sample_count} samples for training")

        train_trajectory_dir = self.exp_dir / "train_rollout"
        train_trajectory_dir.mkdir(parents=True, exist_ok=True)
        train_trajectory_path = train_trajectory_dir / f"train_rollout_{train_step}.jsonl"
        self._save_trajectories(train_batch, train_trajectory_path)
        self.logger.info(f"Train step {train_step} train trajectories saved to {train_trajectory_path}")

        # 共卡需要先释放 rollout，再把训练 worker onload；非共卡不走这两个动作。
        if offload_rollout_before_train:
            ray.get(self.rollout_controller.offload.remote())
        if onload_train_before_train:
            with timer("onload", step_timer_dict):
                self.train_controller.onload(target="all")
                self.logger.info("Training controller loaded")

        with timer("prepare_data", step_timer_dict):
            data_batches, data_info = self._prepare_train_data(
                train_batch,
                self._train_worker_cfg.pack_max_length,
                raw_rewards_sum=raw_rewards_sum,
                raw_rewards_count=raw_rewards_count,
            )
        self.logger.info(f"Prepared {len(data_batches)} training data batches")

        with timer("training", step_timer_dict):
            workers_log_item: list[WorkerLogItem] = self.train_controller.fit(
                data_batches,
                pack_max_length=self._train_worker_cfg.pack_max_length,
                rollout_idx=train_step,
            )
        return {
            "data_info": data_info,
            "workers_log_item": workers_log_item,
        }

    async def _run_evaluation(self, train_step: int) -> dict[str, float]:
        eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
            self.evaluator.eval_batch_size,
            train_step=1,
            model_step=0,
        )
        if XTUNER_DETERMINISTIC:
            eval_produce_result.rollout_states = sort_rollout_state_for_deterministic(
                eval_produce_result.rollout_states
            )
        eval_batch = eval_produce_result.rollout_states
        eval_metrics = self.evaluator.run(eval_batch)
        eval_trajectory_dir = self.exp_dir / "eval_rollout"
        eval_trajectory_dir.mkdir(parents=True, exist_ok=True)
        eval_trajectory_path = eval_trajectory_dir / f"eval_rollout_{train_step}.jsonl"
        self._save_trajectories(eval_batch, eval_trajectory_path)
        self.logger.info(f"Train step {train_step} eval trajectories saved to {eval_trajectory_path}")
        return eval_metrics

    def _save_debug_rollout_batch(self, train_batch: list[list[RolloutState]], train_step: int) -> None:
        assert self._debug_rollout_dir is not None
        self._debug_rollout_dir.mkdir(parents=True, exist_ok=True)
        save_path = self._debug_rollout_dir / f"debug_rollout_{train_step}.pt"
        serializable_batch = [
            [cast(RolloutState, _snapshot_nested_objectrefs(rollout_state)) for rollout_state in group]
            for group in train_batch
        ]
        torch.save(serializable_batch, save_path)
        self.logger.info(f"Debug rollout batch for step {train_step} saved to {save_path}")

    def _list_debug_rollout_files(self, debug_rollout_dir: Path) -> dict[int, Path]:
        debug_files = {
            _parse_debug_rollout_step(path): path
            for path in sorted(debug_rollout_dir.glob("debug_rollout_*.pt"), key=_parse_debug_rollout_step)
        }
        if not debug_files:
            raise FileNotFoundError(f"No debug rollout files found in {debug_rollout_dir}")
        return debug_files

    def _load_debug_rollout_batch(self, train_step: int) -> list[list[RolloutState]]:
        debug_file = self._debug_train_files.get(train_step)
        if debug_file is None:
            raise FileNotFoundError(f"No debug rollout file found for train step {train_step}")
        train_batch = torch.load(debug_file, map_location="cpu", weights_only=False)
        train_batch = [
            [cast(RolloutState, _restore_nested_objectrefs(rollout_state)) for rollout_state in group]
            for group in train_batch
        ]
        self.logger.info(f"Loaded debug rollout batch for step {train_step} from {debug_file}")
        return cast(list[list[RolloutState]], train_batch)

    # TODO: simplify with Packer.pack_pad_dispatch()
    def _prepare_train_data(
        self,
        data_groups: list[list[RolloutState]],
        pack_max_length: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ):
        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []
        training_tokens = 0

        data_batches = []

        for j, group in enumerate(data_groups):
            if not is_valid_for_training(group, self.logger):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue

            is_vlm_model = "train_prompt_ids" in group[0].extra_fields
            if is_vlm_model:
                # TODO(hha): VLM, 不好的设计，后续要去掉
                prompt_ids = group[0].extra_fields["train_prompt_ids"]
            else:
                prompt_ids = group[0].prompt_ids
            assert prompt_ids is not None and len(prompt_ids) > 0, (
                f"Prompt ids cannot be None or empty in data: {group[0]}"
            )
            rewards = []
            for data in group:
                assert data.reward is not None and "score" in data.reward, (
                    f"Reward is missing or does not contain 'score' key in data: {data}"
                )
                rewards.append(data.reward["score"])

            rewards_list.extend(rewards)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            advantages = self._advantage_estimator.compute(rewards_tensor, group)

            prompt_repeat_k = len(group)
            for i in range(prompt_repeat_k):
                item = group[i].response
                logprobs: list[float] | None = None

                response_ids: List[int] = []
                if group[i].response_ids is not None:
                    resp_ids_raw = group[i].response_ids
                    if isinstance(resp_ids_raw, torch.Tensor):
                        response_ids = resp_ids_raw.flatten().tolist()
                    else:
                        response_ids = cast(List[int], resp_ids_raw)

                    logprobs = group[i].logprobs
                    if logprobs is not None:
                        assert len(logprobs) == len(response_ids), (
                            f"{len(logprobs)} vs {len(response_ids)}, data: {group[i]}"
                        )
                        # 只有 response 部分有 logprobs, 需要前面追加
                        logprobs = [0.0] * (len(prompt_ids) - 1) + logprobs  # type: ignore[arg-type]
                else:
                    assert item is not None, "response item cannot be None"
                    response_ids = self.tokenizer(item, return_tensors="pt")["input_ids"].flatten().tolist()

                # 返回的 routed_experts 不包括 eos 的值，实际上也不需要，需要减一
                # TODO: verl tool agent loop 是否需要？
                input_ids = prompt_ids + response_ids[:-1]

                prompt_len_list.append(len(prompt_ids))
                response_len_list.append(len(response_ids))

                # 根据 response_mask 计算 response_ids 对应的shifted_labels
                if not group[i].response_mask:
                    response_mask = [1] * len(response_ids)
                    response_labels = response_ids
                else:
                    assert len(group[i].response_mask) == len(response_ids), (  # type: ignore[arg-type]
                        f"{len(group[i].response_mask)} vs {len(response_ids)}"  # type: ignore[arg-type]
                    )
                    response_mask = cast(list[int], group[i].response_mask)
                    response_labels = [
                        response_id if mask_id != 0 else -100
                        for response_id, mask_id in zip(response_ids, response_mask)
                    ]
                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_labels
                shifted_labels_t = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                # 根据 response_mask 计算新的 advantages
                advatnages_val = advantages[i].item()
                actual_advantages = [advatnages_val] * len(prompt_ids) + [
                    0.0 if mask == 0 else advatnages_val for mask in response_mask
                ]
                advantages_list.extend(actual_advantages[:-1])

                assert len(input_ids) <= pack_max_length, f"{len(input_ids)} vs {pack_max_length}"
                training_tokens += len(input_ids)
                input_ids_t = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)

                if logprobs is not None:
                    rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                    assert rollout_logprobs.size() == shifted_labels_t.size(), (
                        f"{rollout_logprobs.size()} vs {shifted_labels_t.size()}"
                    )
                else:
                    rollout_logprobs = None

                position_ids = group[i].position_ids
                multimodal_train_info = group[i].mm_info
                multi_info_cast = cast(dict | None, multimodal_train_info)
                seq_ctx = get_train_seq_ctx(input_ids_t, position_ids, multi_info_cast, len(response_ids) - 1)  # type: ignore[arg-type]

                data_dict = {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels_t,
                    "advantage": actual_advantages,
                    "rollout_logprobs": rollout_logprobs,
                }

                seq_ctx.rollout_routed_experts = group[i].routed_experts  # n,layer*expert

                data_batches.append(data_dict)
        if not XTUNER_DETERMINISTIC:
            random.shuffle(data_batches)

        rewards_t = torch.tensor(rewards_list).float() if rewards_list else torch.tensor([0.0]).float()
        advantages_t = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_t = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_t = torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()

        raw_rewards_mean = raw_rewards_sum / raw_rewards_count if raw_rewards_count > 0 else rewards_t.mean().item()
        info_dict = {
            "batch_size": len(rewards_list),
            "training_samples": len(rewards_list),
            "training_tokens": training_tokens,
            "rewards/mean": rewards_t.mean().item(),
            "rewards/min": rewards_t.min().item(),
            "rewards/max": rewards_t.max().item(),
            "raw_rewards/mean": raw_rewards_mean,
            "advantages/mean": advantages_t.mean().item(),
            "advantages/min": advantages_t.min().item(),
            "advantages/max": advantages_t.max().item(),
            "response_len/mean": response_len_t.mean().item(),
            "response_len/min": response_len_t.min().item(),
            "response_len/max": response_len_t.max().item(),
            "response_len/std": response_len_t.std().item(),
            "prompt_len/mean": prompt_len_t.mean().item(),
            "prompt_len/min": prompt_len_t.min().item(),
            "prompt_len/max": prompt_len_t.max().item(),
        }
        return data_batches, info_dict

    def _compute_benchmark_metrics(
        self,
        data_info: dict[str, float],
        produce_result: ProduceBatchResult,
        step_timer_dict: dict,
        benchmark_end_time_s: float,
    ) -> RLThroughputBenchmark | None:
        benchmark_start_time_s = self._benchmark_start_time_s
        if benchmark_start_time_s is None:
            return None
        e2e_s = benchmark_end_time_s - benchmark_start_time_s
        step_s = step_timer_dict.get("step")
        training_s = step_timer_dict.get("training")
        rollout_s = produce_result.produce_time_s or step_timer_dict.get("produce_batch")
        if e2e_s <= 0 or step_s is None or step_s <= 0 or training_s is None or training_s <= 0:
            return None

        training_samples = float(data_info.get("training_samples", data_info.get("batch_size", 0.0)))
        training_tokens = float(data_info.get("training_tokens", 0.0))
        effective_samples = float(self._benchmark_training_samples)
        effective_tokens = float(self._benchmark_training_tokens)
        rollout_samples = float(produce_result.produced_samples)
        rollout_tokens = float(produce_result.produced_tokens)
        train_gpu_count = float(getattr(self, "_num_workers", 1.0))
        if train_gpu_count <= 0:
            train_gpu_count = 1.0
        rollout_gpu_count = float(getattr(self, "_rollout_num_workers", train_gpu_count))
        if rollout_gpu_count <= 0:
            rollout_gpu_count = 1.0

        e2e_effective_tokens_per_s = effective_tokens / e2e_s
        effective_tokens_per_s = training_tokens / step_s
        training_tokens_per_s = training_tokens / training_s
        rollout_samples_per_s = rollout_samples / rollout_s if rollout_s is not None and rollout_s > 0 else 0.0
        rollout_tokens_per_s = rollout_tokens / rollout_s if rollout_s is not None and rollout_s > 0 else 0.0
        return RLThroughputBenchmark(
            e2e_effective_sgs=effective_samples / e2e_s / train_gpu_count,
            e2e_effective_tgs=e2e_effective_tokens_per_s / train_gpu_count,
            effective_sgs=training_samples / step_s / train_gpu_count,
            effective_tgs=effective_tokens_per_s / train_gpu_count,
            training_tgs=training_tokens_per_s / train_gpu_count,
            rollout_sgs=rollout_samples_per_s / rollout_gpu_count,
            rollout_tgs=rollout_tokens_per_s / rollout_gpu_count,
        )

    def _log_step(
        self,
        train_step: int,
        step_timer_dict: dict,
        produce_result: ProduceBatchResult,
        train_info: TrainInfo,
        eval_info: dict[str, float],
    ):
        all_scalars = {}
        log_time_str = ""
        trajectory_str = ""
        throughput_str = ""
        eval_str = ""
        if step_timer_dict:
            all_scalars.update({f"time/{k}": v for k, v in step_timer_dict.items()})
            log_time_str = f"\nTrain step {train_step} finished and timing listed:\n"
            log_time_str += "\n".join([f" - {k:<25}: {v:.2f}s" for k, v in step_timer_dict.items()])

        if produce_result.group_gen_count is not None:
            all_scalars["timing/task_n"] = produce_result.group_gen_count
            all_scalars["timing/task_mean_s"] = produce_result.group_gen_mean_s
            all_scalars["timing/task_p50_s"] = produce_result.group_gen_p50_s
            all_scalars["timing/task_p99_s"] = produce_result.group_gen_p99_s
            all_scalars["timing/task_p99_p50_ratio"] = produce_result.group_gen_p99_p50_ratio
        if produce_result.group_gen_pause_time_s is not None:
            all_scalars["timing/pause_s"] = produce_result.group_gen_pause_time_s
        all_scalars["async/init_samples"] = produce_result.leftover_init
        all_scalars["async/completed_samples"] = produce_result.leftover_completed
        all_scalars["async/aborted_samples"] = produce_result.leftover_aborted
        all_scalars["async/expired_samples"] = produce_result.leftover_expired
        all_scalars["async/failed_samples"] = produce_result.leftover_failed
        all_scalars["async/filtered_samples"] = produce_result.leftover_filtered

        if train_info:
            data_info = train_info.get("data_info", {})
            training_samples = int(data_info.get("training_samples", data_info.get("batch_size", 0)))
            training_tokens = int(data_info.get("training_tokens", 0))
            self._benchmark_training_samples += training_samples
            self._benchmark_training_tokens += training_tokens
            benchmark_end_time_s = float(data_info.get("benchmark_end_time_s", time.perf_counter()))
            benchmark_data_info_keys = {
                "training_samples",
                "training_tokens",
                "benchmark_end_time_s",
            }
            response_data_info = {k: v for k, v in data_info.items() if k not in benchmark_data_info_keys}
            all_scalars.update({f"response/{k}": v for k, v in response_data_info.items()})
            throughput_benchmark = self._compute_benchmark_metrics(
                data_info,
                produce_result,
                step_timer_dict,
                benchmark_end_time_s,
            )
            if throughput_benchmark is not None:
                throughput_metrics = throughput_benchmark.to_scalars()
                all_scalars.update(throughput_metrics)
                throughput_str = f"\nTrain step {train_step} throughput statistics:\n"
                throughput_str += "\n".join(
                    [f"- {k.removeprefix('throughput/'):<25}: {v:.4f}" for k, v in throughput_metrics.items()]
                )
            trajectory_str = f"\nTrain step {train_step} data statistics:\n"
            trajectory_str += "\n".join([f"- {k:<25}: {v:.4f}" for k, v in response_data_info.items()])
            rank0_log_item = train_info["workers_log_item"][0]
            rank0_rollout_is_metrics = rank0_log_item.get("rollout_is_metrics", {})
            rank0_mismatch_metrics = rank0_log_item.get("mismatch_metrics", {})
            rank0_rollout_entropy = rank0_log_item.get("rollout_entropy", 0.0)
            all_scalars.update({f"rollout_is/{k}": v for k, v in rank0_rollout_is_metrics.items()})
            all_scalars.update({f"{k}": v for k, v in rank0_mismatch_metrics.items()})
            all_scalars.update({"entropy/rollout": rank0_rollout_entropy})
            all_scalars.update({"entropy/train": rank0_log_item["train_entropy"]})
            for worker_idx, log_item in enumerate(train_info["workers_log_item"]):
                if not self._display_all_workers_log and worker_idx > 0:
                    break
                mini_batch_metrics: dict[str, List[float]] = {}
                for mini_batch_log in log_item["train_metrics"]:
                    for k, v in mini_batch_log.items():
                        mini_batch_metrics.setdefault(k, []).append(cast(float, v))

                for key, value in mini_batch_metrics.items():
                    avg_value = sum(value) / len(value)
                    all_scalars.update({f"train_metrics/worker_{worker_idx}/step_avg_{key}": avg_value})

                rank_sft_log = log_item["sft_train_metrics"]
                for k, v in rank_sft_log.items():
                    all_scalars.update({f"sft_train_metrics/worker_{worker_idx}/{k}": v})

            self._log_mini_batch_metrics(train_info["workers_log_item"])

        if eval_info:
            all_scalars.update({f"eval/{k}": v for k, v in eval_info.items()})
            eval_str = " ".join([f"{k}: {v:.4f}" for k, v in eval_info.items()])

        self.logger.info(
            f"Train step {train_step}/{self._total_train_steps}{log_time_str} {trajectory_str} {throughput_str}"
        )
        if eval_str:
            self.logger.info(f"Eval: {eval_str}")
        self._exp_tracker.add_scalars(tag_scalar_dict=all_scalars, global_step=train_step)

    def _save_trajectories(self, data_groups: list[list[RolloutState]], save_path: Path) -> None:
        rewards = []
        response_len_list = []

        for group in data_groups:
            if not is_valid_for_training(group, self.logger):
                continue
            for data in group:
                assert data.reward is not None
                rewards.append(data.reward["score"])
                if data.response_ids is not None:
                    if isinstance(data.response_ids, torch.Tensor):
                        response_ids = data.response_ids.flatten().tolist()
                    else:
                        response_ids = data.response_ids
                    response_len_list.append(len(response_ids))
                elif data.response is not None:
                    response_ids = self.tokenizer.encode(data.response, add_special_tokens=False)
                    response_len_list.append(len(response_ids))

        rewards_tensor = torch.tensor(rewards).float() if rewards else torch.tensor([0.0]).float()
        response_lens = torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()

        _count = 0
        with open(save_path, "w", encoding="utf-8") as f:
            summary = {
                "reward_mean": rewards_tensor.mean().item(),
                "reward_std": rewards_tensor.std().item(),
                "reward_max": rewards_tensor.max().item(),
                "reward_min": rewards_tensor.min().item(),
                "response_len_mean": response_lens.mean().item(),
                "response_len_std": response_lens.std().item(),
                "response_len_max": response_lens.max().item(),
                "response_len_min": response_lens.min().item(),
                "total_len": len(rewards),
            }
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
            for group in data_groups:
                if not is_valid_for_training(group, self.logger):
                    continue
                for data in group:
                    assert data.reward is not None
                    ground_truth = None
                    if data.reward_model is not None:
                        ground_truth = data.reward_model.get("ground_truth")
                    item = {
                        "prompt": data.message,
                        "raw_prompt": data.extra_fields.get("raw_prompt", None),
                        "response": data.response,
                        "response_len": response_len_list[_count],
                        "label": ground_truth,
                        "reward": data.reward["score"],
                        "finish_reason": data.finish_reason,
                    }
                    json.dump(item, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                    _count += 1

    def _log_mini_batch_metrics(self, workers_log_item: List[WorkerLogItem]):
        train_start_step = self._global_train_step + 1
        for worker_idx, log_item in enumerate(workers_log_item):
            for step_idx, mini_batch_log in enumerate(log_item["train_metrics"]):
                if not self._display_all_workers_log and worker_idx > 0:
                    break
                current_global_step = train_start_step + step_idx

                metrics: dict[str, Any] = dict(mini_batch_log)

                self._exp_tracker.add_scalars(
                    tag_scalar_dict={f"train_metrics/worker_{worker_idx}/{k}": float(v) for k, v in metrics.items()},
                    global_step=current_global_step,
                )
        self._global_train_step += len(workers_log_item[0]["train_metrics"])


class RLColocateTrainer(BaseRLTrainer):
    _META_PATH = ".xtuner_rl_colocate_trainer"

    # 共卡 trainer 保留自己的资源编排、resume、主循环和权重同步；通用保存、日志仍在 BaseRLTrainer。
    def __init__(self, cfg: RLColocateTrainerConfig):
        self._init_common(cfg, meta_path=self._META_PATH, logger_tag="RLTrainer")
        self._num_workers = float(cfg.resources.num_workers)
        self._rollout_num_workers = float(cfg.resources.num_workers)

        self._pg = AutoAcceleratorWorkers.build_placement_group(cfg.resources)
        self._cpu_resource_manager = CPUResourceManager(self._pg)
        self._cpu_resource_manager.log_initial_snapshot()
        set_cpu_resource_manager(self._cpu_resource_manager)

        if self._debug_rollout:
            if self._rollout_config.skip_load_weights:
                self.logger.info(
                    "debug_rollout cannot be used with rollout_config.skip_load_weights=True. force set skip_load_weights to False"
                )
                self._rollout_config.skip_load_weights = False
            self.rollout_controller = self._rollout_config.build(self._pg)
            self._maybe_start_gateway(cfg)
            replay_buffer = cfg.replay_buffer_config.build()
            self._build_agent_loop_components(cfg, replay_buffer)
            self._cpu_resource_manager.log_registered_summary()
            self.logger.warning("Debug rollout mode is enabled. Only rollout workers will be started.")
            return

        self.train_controller = self._train_worker_cfg.build(self._pg)

        checkpoint_path = self._load_checkpoint_cfg.checkpoint_path
        if checkpoint_path is not None:
            checkpoint_path = self._resume_train_controller_and_state(checkpoint_path)

        if self._debug_train:
            assert self._debug_rollout_dir is not None
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, trust_remote_code=True)
            self._debug_train_files = self._list_debug_rollout_files(self._debug_rollout_dir)
            if cfg.total_train_steps is None:
                self._total_train_steps = max(self._debug_train_files)
            self.logger.warning(
                "Debug train mode is enabled. Only training workers will be started and rollout weights will not be synchronized."
            )
            return

        # Free trainer-side GPU memory before bringing up colocated rollout workers.
        # Backends like sglang may size KV cache against their own target utilization
        # instead of the trainer's transient footprint, which can cause init-time OOM.
        self.train_controller.offload(target="all")

        self.rollout_controller = self._rollout_config.build(self._pg)
        self._maybe_start_gateway(cfg)
        bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)

        replay_buffer = cfg.replay_buffer_config.build()
        self._build_agent_loop_components(cfg, replay_buffer)
        if checkpoint_path is not None:
            asyncio_run(self._resume_agent_loop_manager(checkpoint_path))

        self.train_controller.set_train_rollout_mode("colocate")
        self._cpu_resource_manager.log_registered_summary()

        if self._rollout_config.skip_load_weights:
            self._sync_weights_from_train_workers()

    def _sync_weights_from_train_workers(self) -> None:
        self.logger.info("Rollout workers skip load weights, update weights from train workers.")
        ray.get(self.rollout_controller.offload.remote())
        self.train_controller.onload(target="model")
        ray.get(self.rollout_controller.onload_weights.remote())
        self.train_controller.update_weights()
        self.train_controller.offload(target="model")
        ray.get(self.rollout_controller.onload_kvcache.remote())
        self.logger.info("Rollout workers updated weights from train workers.")

    def fit(self):
        self.logger.info("Start RL training")
        if self._cur_step >= self._total_train_steps:
            self.logger.info(f"Train steps {self._total_train_steps} reached, stop training")
            return

        if self._debug_train:
            self._fit_debug_train()
            return

        if self._enable_initial_evaluate and not self._debug_rollout:
            asyncio_run(self._run_initial_evaluate())

        self._benchmark_start_time_s = time.perf_counter()
        self._benchmark_training_samples = 0
        self._benchmark_training_tokens = 0

        init_train_step = self._cur_step + 1
        model_step = self._get_colocate_rollout_model_step(init_train_step)
        for train_step in range(init_train_step, self._total_train_steps + 1):
            self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                # 共卡路径一次调用内完成 rollout 生产和 replay buffer 消费。
                self.logger.info(
                    f"[Step {train_step}] start to generate rollout experience for train step {train_step} with model step {model_step}"
                )
                with timer("produce_batch", step_timer_dict):
                    produce_result: ProduceBatchResult = asyncio_run(
                        self.agent_loop_manager.produce_batch(
                            self.train_batch_size,
                            train_step=train_step,
                            model_step=model_step,
                        )
                    )
                if XTUNER_DETERMINISTIC:
                    produce_result.rollout_states = sort_rollout_state_for_deterministic(produce_result.rollout_states)
                train_batch = produce_result.rollout_states
                assert train_batch, (
                    "RLColocateTrainer expects agent_loop_manager.produce_batch() to return non-empty rollout_states."
                )

                if not self._debug_rollout:
                    train_log_info = self._train_one_batch(
                        train_batch,
                        train_step,
                        step_timer_dict,
                        offload_rollout_before_train=True,
                        onload_train_before_train=True,
                        raw_rewards_sum=produce_result.raw_rewards_sum,
                        raw_rewards_count=produce_result.raw_rewards_count,
                    )
                else:
                    self._save_debug_rollout_batch(train_batch, train_step)
                    train_log_info = {}

                if not self._debug_rollout:
                    weights_synced = self._sync_weights_and_save(train_step, step_timer_dict)
                    if weights_synced:
                        model_step = train_step

                    eval_log_info = {}
                    if weights_synced and self._enable_evaluate and train_step % self._evaluate_step == 0:
                        with timer("evaluation", step_timer_dict):
                            eval_log_info.update(asyncio_run(self._run_evaluation(train_step)))
                else:
                    eval_log_info = {}

            self._log_step(train_step, step_timer_dict, produce_result, train_log_info, eval_log_info)
            self._cur_step = train_step

    def _fit_debug_train(self) -> None:
        self._benchmark_start_time_s = time.perf_counter()
        self._benchmark_training_samples = 0
        self._benchmark_training_tokens = 0

        init_train_step = self._cur_step + 1
        for train_step in range(init_train_step, self._total_train_steps + 1):
            self.logger.info(f"Debug train step {train_step}/{self._total_train_steps} start")
            step_timer_dict: dict[str, float] = {}
            with timer("step", step_timer_dict):
                with timer("load_debug_rollout", step_timer_dict):
                    train_batch = self._load_debug_rollout_batch(train_step)
                train_log_info = self._train_one_batch(
                    train_batch,
                    train_step,
                    step_timer_dict,
                    offload_rollout_before_train=False,
                    onload_train_before_train=False,
                )
                eval_log_info: dict[str, float] = {}
                produce_result = ProduceBatchResult(rollout_states=train_batch)

            self._log_step(train_step, step_timer_dict, produce_result, train_log_info, eval_log_info)
            self._cur_step = train_step

    def _get_colocate_rollout_model_step(self, train_step: int) -> int:
        previous_step = train_step - 1
        return previous_step - (previous_step % self._sync_weights_interval)

    def _sync_weights_and_save(self, train_step: int, step_timer_dict: dict) -> bool:
        """Save state and switch colocated resources back to rollout
        workers."""
        should_sync_weights = train_step % self._sync_weights_interval == 0
        with timer("save_ckpt", step_timer_dict):
            self.train_controller.offload(target="optimizer")
            asyncio_run(self._maybe_save_checkpoint(train_step))
            self._maybe_save_hf(train_step)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        timer_name = "sync_weight" if should_sync_weights else "switch_to_rollout"
        with timer(timer_name, step_timer_dict):
            if should_sync_weights:
                bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
                ray.get(self.rollout_controller.onload_weights.remote())
                self.train_controller.update_weights()
                self.logger.info("Rollout workers update weights successfully in colocate mode")
                self.train_controller.offload(target="model")
            else:
                self.train_controller.offload(target="model")
                ray.get(self.rollout_controller.onload_weights.remote())
            ray.get(self.rollout_controller.onload_kvcache.remote())
        return should_sync_weights


class RLDisaggregatedTrainer(BaseRLTrainer):
    _META_PATH = ".xtuner_rl_disaggregated_trainer"

    def __init__(self, cfg: RLDisaggregatedTrainerConfig):
        self._init_common(cfg, meta_path=self._META_PATH, logger_tag="RLDisaggTrainer")
        self._num_workers = float(cfg.train_resources.num_workers)
        self._rollout_num_workers = float(cfg.rollout_resources.num_workers)

        self._train_pg, self._rollout_pg = self._build_disaggregated_placement_groups(
            train_resources=cfg.train_resources,
            rollout_resources=cfg.rollout_resources,
        )
        self._cpu_resource_manager = CPUResourceManager([self._train_pg, self._rollout_pg])
        self._cpu_resource_manager.log_initial_snapshot()
        set_cpu_resource_manager(self._cpu_resource_manager)
        self.train_controller = self._train_worker_cfg.build(self._train_pg)
        self.rollout_controller = self._rollout_config.build(self._rollout_pg)
        self._maybe_start_gateway(cfg)

        replay_buffer = cfg.replay_buffer_config.build()
        self._build_agent_loop_components(cfg, replay_buffer)
        # 在非共卡使用模式时，生产者和消费者并发执行
        # 为了让生产者和消费者配合，不能引入生产中的早停机制，否则生产不够，消费者会被阻塞
        # 所以 should_continue_fn 必须为 default_should_continue_fn
        for task_runner in self.agent_loop_manager.task_runners:
            if task_runner.produce_strategy.should_continue_fn is not default_should_continue_fn:
                raise ValueError(
                    "In disaggregated mode, should_continue_fn must be default, "
                    "because it does not allow early stopping in production."
                )
        bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
        self.train_controller.set_train_rollout_mode("disaggregated")

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            self._resume_from_checkpoint(self._load_checkpoint_cfg.checkpoint_path)

        self._cpu_resource_manager.log_registered_summary()

    def _build_disaggregated_placement_groups(
        self,
        train_resources: AcceleratorResourcesConfig,
        rollout_resources: AcceleratorResourcesConfig,
    ):
        pg_name_prefix = f"xtuner_rl_disagg_{self.exp_dir.name}"
        train_pg_name = f"{pg_name_prefix}_train"
        rollout_pg_name = f"{pg_name_prefix}_rollout"

        train_pg = AutoAcceleratorWorkers.build_placement_group(train_resources, name=train_pg_name)
        rollout_pg = AutoAcceleratorWorkers.build_placement_group(rollout_resources, name=rollout_pg_name)
        if train_pg.id == rollout_pg.id:
            raise RuntimeError(
                "RLDisaggregatedTrainer requires distinct placement groups for train and rollout, "
                f"but both resolved to the same placement group id={train_pg.id}. "
                "Please check placement-group naming and stale Ray cluster state."
            )

        self.logger.info(
            "Created disaggregated placement groups: "
            f"train={train_pg_name}(id={train_pg.id}), "
            f"rollout={rollout_pg_name}(id={rollout_pg.id})"
        )
        return train_pg, rollout_pg

    def _resume_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = self._resume_train_controller_and_state(checkpoint_path)
        saved_model_step = asyncio_run(self._resume_agent_loop_manager(checkpoint_path))
        assert self._cur_step == saved_model_step

        self.update_weights()
        asyncio_run(self.agent_loop_manager.continue_produce(model_step=saved_model_step))

    def fit(self):
        # 对外保留同步 fit 接口，内部用 async loop 组织 producer/consumer。
        return asyncio_run(self._fit())

    async def _fit(self):
        self.logger.info("Start RL disaggregated training")
        if self._cur_step >= self._total_train_steps:
            self.logger.info(f"Train steps {self._total_train_steps} reached, stop training")
            return

        if self._enable_initial_evaluate:
            await self._run_initial_evaluate()

        self._benchmark_start_time_s = time.perf_counter()
        self._benchmark_training_samples = 0
        self._benchmark_training_tokens = 0

        # 后台 producer 只负责持续往 replay buffer 写数据，前台 trainer 通过 get_batch 消费。
        producer_task = create_task(
            self.agent_loop_manager.produce_loop(
                batch_size=self.train_batch_size,
            )
        )
        try:
            # train_step 表示“下一步待完成训练”；空 expired 不算完成，所以必须用 while 支持重试同一步。
            train_step = self._cur_step + 1
            while train_step <= self._total_train_steps:
                self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
                step_timer_dict: dict[str, float] = {}
                train_log_info = {}
                eval_log_info = {}
                with timer("step", step_timer_dict):
                    with timer("get_batch", step_timer_dict):
                        produce_result = await self.agent_loop_manager.get_batch(
                            self.train_batch_size, train_step=train_step
                        )
                    if XTUNER_DETERMINISTIC:
                        produce_result.rollout_states = sort_rollout_state_for_deterministic(
                            produce_result.rollout_states
                        )

                    train_batch = produce_result.rollout_states
                    # EXPIRED_BATCH 分两类：空 batch 是控制面同步；非空 batch 仍然是可训练数据。
                    empty_expired_batch = produce_result.status == ProduceBatchStatus.EXPIRED_BATCH and not train_batch
                    if empty_expired_batch:
                        # 没有完成训练，能同步的只能是上一轮已经完成的 Model Step。
                        sync_model_step = train_step - 1
                        self.logger.info(
                            "Skip train step because rollout model is expired and a newer model already exists; "
                            f"sync completed model_step={sync_model_step} first."
                        )
                    else:
                        # 非空 expired 必须训练出当前 step 的新模型版本，否则 producer 没有更新权重可恢复。
                        assert train_batch, (
                            "RLDisaggregatedTrainer expects get_batch() to return non-empty rollout_states "
                            "unless status is empty EXPIRED_BATCH."
                        )
                        # 非共卡训练要求后台 producer 在训练当前 batch 时继续推进；
                        # 同步训练路径放到线程里执行，避免 ray.get / 文件写入阻塞事件循环。
                        train_log_info = await asyncio.to_thread(
                            self._train_one_batch,
                            train_batch,
                            train_step,
                            step_timer_dict,
                            raw_rewards_sum=produce_result.raw_rewards_sum,
                            raw_rewards_count=produce_result.raw_rewards_count,
                        )
                        sync_model_step = train_step

                    # 后续保存、同步、评测、恢复 producer 都以“已完成的 Model Step”为唯一口径。
                    need_sync = (
                        empty_expired_batch
                        or produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
                        or sync_model_step % self._sync_weights_interval == 0
                        or sync_model_step == self._total_train_steps
                    )

                    if need_sync:
                        # 同步前先暂停后台 producer，避免 save/sync 时还有 pending rollout 继续写 buffer。
                        with timer("pause_produce", step_timer_dict):
                            await self.agent_loop_manager.pause_produce(use_global_progress=True)

                        await self._sync_weights_and_save(sync_model_step, step_timer_dict)

                        if (
                            self._enable_evaluate
                            and sync_model_step > 0
                            and sync_model_step % self._evaluate_step == 0
                        ):
                            # eval 放在恢复 producer 前，避免后台生产抢占 rollout 资源。
                            with timer("evaluation", step_timer_dict):
                                eval_log_info.update(await self._run_evaluation(sync_model_step))

                        await self.agent_loop_manager.continue_produce(model_step=sync_model_step)

                if empty_expired_batch:
                    # 空 expired 没有完成训练，不能 log 成已完成 step，也不能推进 _cur_step。
                    continue
                self._log_step(train_step, step_timer_dict, produce_result, train_log_info, eval_log_info)
                self._cur_step = train_step
                train_step = self._cur_step + 1
        finally:
            self.agent_loop_manager.shutdown()
            await producer_task

    async def _sync_weights_and_save(self, model_step: int, step_timer_dict: dict):
        # 非共卡已经在 _fit 里暂停 producer；这里保持静止态下的 save -> bind -> update 顺序。
        with timer("save_ckpt", step_timer_dict):
            await self._maybe_save_checkpoint(model_step)
            self._maybe_save_hf(model_step)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        with timer("sync_weight", step_timer_dict):
            bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
            self.update_weights()

    def update_weights(self):
        # producer 的 pause/continue 由 AgentLoopManager 控制，避免这里提前恢复 rollout 影响 eval 顺序。
        self.train_controller.update_weights()
        self.logger.info("Rollout workers update weights successfully in disaggregated mode")
