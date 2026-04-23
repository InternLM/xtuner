import json
import os
import random
from pathlib import Path
from shutil import rmtree
from typing import Any, List, cast

import ray
import torch
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Literal, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1._writer import get_writer
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AgentLoopManagerStatus,
    ProduceBatchResult,
    ProduceBatchStatus,
)
from xtuner.v1.rl.agent_loop_manager.producer import default_should_continue_fn
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.gateway.config import GatewayConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.controller import RolloutControllerProxy
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.worker import WorkerConfig, WorkerLogItem
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run, create_task
from xtuner.v1.train.trainer import LoadCheckpointConfig, XTunerMeta
from xtuner.v1.utils import get_logger, is_hf_model_path, set_deterministic, timer
from xtuner.v1.utils.device import get_device, get_torch_device_module


# TODO: Move DEVICE to `xtuner.utils.device`
PG_READY_TIMEOUT = 30
TRAINER_RAY_GET_TIMEOUT = 5 * 3600  # 5 hour
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


def force_set_tokenize_workers(logger):
    # To avoid segmentation faults when setting num_workers for the dataloader
    # The root cause is the incompatibility between fork start method and ray's grpc.
    # The most fundamental solution is that all processes started in ray should
    # use spawn start method.
    tokenize_workers = os.environ.get("XTUNER_TOKENIZE_WORKERS", None)
    os.environ["XTUNER_TOKENIZE_WORKERS"] = "1"
    if tokenize_workers is not None and int(tokenize_workers) > 1:
        logger.warning(
            f"XTUNER_TOKENIZE_WORKERS is set to {tokenize_workers}, which may cause segmentation faults. Force set XTUNER_TOKENIZE_WORKERS to 1 to avoid this."
        )
    else:
        logger.info(
            f"Set XTUNER_TOKENIZE_WORKERS to {os.environ['XTUNER_TOKENIZE_WORKERS']} for safe tokenization in dataloader workers."
        )


def bind_train_rollout(
    train_controller: TrainingController,
    rollout_controller: RolloutControllerProxy,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())  # type: ignore[attr-defined]
    train_controller.update_rollout_info(info_dict)
    return


class TrainInfo(TypedDict, total=False):
    data_info: dict[str, float]
    workers_log_item: list[WorkerLogItem]


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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    tokenizer_path: str | Path
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: AgentLoopManagerConfig
    eval_agent_loop_manager_cfg: AgentLoopManagerConfig
    evaluator_config: EvaluatorConfig
    load_from: str | Path
    total_train_steps: int
    train_batch_size: int
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
    seed: int = 66
    debug_rollout: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"

    @model_validator(mode="after")
    def _validate_sync_intervals(self):
        _validate_sync_intervals(
            sync_weights_interval=self.sync_weights_interval,
            checkpoint_interval=self.checkpoint_interval,
            hf_interval=self.hf_interval,
            evaluate_step=self.evaluate_step,
            enable_evaluate=self.enable_evaluate,
        )
        return self


class RLColocateTrainerConfig(BaseRLTrainerConfig):
    resources: AcceleratorResourcesConfig

    def build(self) -> "RLColocateTrainer":
        return RLColocateTrainer(self)


class RLDisaggregatedTrainerConfig(BaseRLTrainerConfig):
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

    def _init_common(self, cfg: BaseRLTrainerConfig, *, meta_path: str, logger_tag: str) -> None:
        check_fa3()
        self._init_work_dir_and_meta(cfg, meta_path)
        self._init_load_source(cfg)
        self._init_save_config(cfg)
        log_dir = self._init_logger(cfg, logger_tag)
        self._init_train_state(cfg)
        self._init_train_worker_config(cfg, log_dir)
        self._init_rollout_config(cfg, log_dir)
        self._init_runtime_flags(cfg)

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
        self.logger = get_logger(log_dir=log_dir, tag=logger_tag)
        force_set_tokenize_workers(self.logger)

        if cfg.skip_checkpoint_validation:
            patch_default_save_plan()
        return log_dir

    def _init_train_state(self, cfg: BaseRLTrainerConfig) -> None:
        self._total_train_steps = cfg.total_train_steps
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

    def _init_runtime_flags(self, cfg: BaseRLTrainerConfig) -> None:
        self._enable_evaluate = cfg.enable_evaluate
        self._enable_initial_evaluate = cfg.enable_initial_evaluate
        self._evaluate_step = cfg.evaluate_step
        self._debug_rollout = cfg.debug_rollout

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

        self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )

        total_eval_samples = len(self.eval_agent_loop_manager.data_sampler)
        self.evaluator = cfg.evaluator_config.build(total_eval_samples=total_eval_samples)

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
        checkpoint_path = Path(checkpoint_path)
        self.train_controller.resume(self._load_checkpoint_cfg)

        train_state_path = checkpoint_path / self._SAVE_TRAIN_STATE_PATH
        with train_state_path.open("r") as f:
            train_state = json.load(f)
        self._cur_step = train_state["cur_step"]
        return checkpoint_path

    def _maybe_save_checkpoint(self, cur_step: int) -> None:
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
        self.agent_loop_manager.save(checkpoint_path, model_step=cur_step)

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
            data_batches, data_info = self._prepare_train_data(train_batch, self._train_worker_cfg.pack_max_length)
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
        eval_batch = eval_produce_result.rollout_states
        eval_metrics = self.evaluator.run(eval_batch)
        eval_trajectory_dir = self.exp_dir / "eval_rollout"
        eval_trajectory_dir.mkdir(parents=True, exist_ok=True)
        eval_trajectory_path = eval_trajectory_dir / f"eval_rollout_{train_step}.jsonl"
        self._save_trajectories(eval_batch, eval_trajectory_path)
        self.logger.info(f"Train step {train_step} eval trajectories saved to {eval_trajectory_path}")
        return eval_metrics

    # TODO: simplify with Packer.pack_pad_dispatch()
    def _prepare_train_data(self, data_groups: list[list[RolloutState]], pack_max_length: int):
        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

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
            advantages = (rewards_tensor - rewards_tensor.mean(0)) / (rewards_tensor.std(0) + 1e-8)

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
        random.shuffle(data_batches)

        rewards_t = torch.tensor(rewards_list).float() if rewards_list else torch.tensor([0.0]).float()
        advantages_t = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_t = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_t = torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()

        info_dict = {
            "batch_size": len(rewards_list),
            "rewards/mean": rewards_t.mean().item(),
            "rewards/min": rewards_t.min().item(),
            "rewards/max": rewards_t.max().item(),
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
            all_scalars["timing/pause_s"] = produce_result.group_gen_pause_time_s
        all_scalars["async/completed_samples"] = produce_result.leftover_completed
        all_scalars["async/aborted_samples"] = produce_result.leftover_aborted
        all_scalars["async/expired_samples"] = produce_result.leftover_expired

        if train_info:
            all_scalars.update({f"response/{k}": v for k, v in train_info.get("data_info", {}).items()})
            trajectory_str = f"\nTrain step {train_step} data statistics:\n"
            trajectory_str += "\n".join([f"- {k:<25}: {v:.4f}" for k, v in train_info.get("data_info", {}).items()])
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
                    rl_worker_log = mini_batch_log["loss_log"] | mini_batch_log["rl_other_log"]
                    for k, v in rl_worker_log.items():
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

        self.logger.info(f"Train step {train_step}/{self._total_train_steps}{log_time_str} {trajectory_str} ")
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

                metrics: dict[str, Any] = dict(mini_batch_log["loss_log"])
                metrics.update(mini_batch_log["rl_other_log"])

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

        self._pg = AutoAcceleratorWorkers.build_placement_group(cfg.resources)
        self.train_controller = self._train_worker_cfg.build(self._pg)
        self.rollout_controller = self._rollout_config.build(self._pg)
        self._maybe_start_gateway(cfg)

        replay_buffer = cfg.replay_buffer_config.build()
        self._build_agent_loop_components(cfg, replay_buffer)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            self._resume_from_checkpoint(self._load_checkpoint_cfg.checkpoint_path)
        else:
            self.train_controller.offload(target="all")

        if self._debug_rollout:
            self.logger.warning("Debug rollout mode is enabled, rollout will not be offloaded.")

    def _resume_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = self._resume_train_controller_and_state(checkpoint_path)

        self.logger.info(f"Resume sampler from {checkpoint_path}")
        self.agent_loop_manager.resume(checkpoint_path)

        bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
        self.logger.info("Rollout workers skip load weights, update weights from train workers.")
        self.train_controller.offload(target="optimizer")
        ray.get(self.rollout_controller.offload.remote())
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

        if self._enable_initial_evaluate and not self._debug_rollout:
            asyncio_run(self._run_initial_evaluate())

        init_train_step = self._cur_step + 1
        model_step = self._get_colocate_rollout_model_step(init_train_step)
        for train_step in range(init_train_step, self._total_train_steps + 1):
            self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                # 共卡路径一次调用内完成 rollout 生产和 replay buffer 消费。
                self.logger.info("start to generate rollout experience for training")
                produce_result: ProduceBatchResult = asyncio_run(
                    self.agent_loop_manager.produce_batch(
                        self.train_batch_size,
                        train_step=train_step,
                        model_step=model_step,
                    )
                )
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
                    )

                    weights_synced = self._sync_weights_and_save(train_step, step_timer_dict)
                    if weights_synced:
                        model_step = train_step

                    eval_log_info = {}
                    if weights_synced and self._enable_evaluate and train_step % self._evaluate_step == 0:
                        with timer("evaluation", step_timer_dict):
                            eval_log_info.update(asyncio_run(self._run_evaluation(train_step)))
                else:
                    train_log_info = {}
                    eval_log_info = {}

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
            self._maybe_save_checkpoint(train_step)
            self._maybe_save_hf(train_step)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        timer_name = "sync_weight" if should_sync_weights else "switch_to_rollout"
        with timer(timer_name, step_timer_dict):
            if should_sync_weights:
                bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
                ray.get(self.rollout_controller.onload_weights.remote())
                self.train_controller.update_weights()
                self.logger.info("Model weights synchronized successfully.")
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

        self._train_pg, self._rollout_pg = self._build_disaggregated_placement_groups(
            train_resources=cfg.train_resources,
            rollout_resources=cfg.rollout_resources,
        )
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

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            self._resume_from_checkpoint(self._load_checkpoint_cfg.checkpoint_path)

        if self._debug_rollout:
            self.logger.warning(
                "Debug rollout mode is enabled. Disaggregated training keeps rollout workers resident."
            )

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

        self.logger.info(f"Resume sampler from {checkpoint_path}")
        saved_model_step = self.agent_loop_manager.resume(checkpoint_path)
        assert self._cur_step == saved_model_step

        bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
        self.logger.info("Rollout workers skip load weights, update weights from train workers.")
        self.fake_update_weights()
        self.agent_loop_manager.continue_produce(model_step=saved_model_step)

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

        # 后台 producer 只负责持续往 replay buffer 写数据，前台 trainer 通过 get_batch 消费。
        producer_task = create_task(
            self.agent_loop_manager.produce_loop(
                batch_size=self.train_batch_size,
            )
        )
        try:
            for train_step in range(self._cur_step + 1, self._total_train_steps + 1):
                self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
                step_timer_dict: dict[str, float] = {}
                train_log_info = {}
                eval_log_info = {}
                with timer("step", step_timer_dict):
                    produce_result = await self.agent_loop_manager.get_batch(
                        self.train_batch_size, train_step=train_step
                    )
                    need_sync = (
                        produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
                        or train_step % self._sync_weights_interval == 0
                        or train_step == self._total_train_steps
                    )
                    if produce_result.status != ProduceBatchStatus.EXPIRED_BATCH:
                        train_batch = produce_result.rollout_states
                        assert train_batch, (
                            "RLDisaggregatedTrainer expects get_batch() to return non-empty rollout_states "
                            "unless status is EXPIRED_BATCH."
                        )
                        train_log_info = self._train_one_batch(train_batch, train_step, step_timer_dict)
                    else:
                        self.logger.info(
                            "Skip train step because rollout model is expired; prioritize weight sync first."
                        )

                    if need_sync:
                        # 同步前先暂停后台 producer，避免 save/sync 时还有 pending rollout 继续写 buffer。
                        with timer("pause_produce", step_timer_dict):
                            await self.agent_loop_manager.pause_produce(use_global_progress=True)

                        await self._sync_weights_and_save(train_step, step_timer_dict)

                        if self._enable_evaluate and train_step % self._evaluate_step == 0:
                            # eval 放在恢复 producer 前，避免后台生产抢占 rollout 资源。
                            with timer("evaluation", step_timer_dict):
                                eval_log_info.update(await self._run_evaluation(train_step))

                        self.agent_loop_manager.continue_produce(model_step=train_step)

                self._log_step(train_step, step_timer_dict, produce_result, train_log_info, eval_log_info)
                self._cur_step = train_step
        finally:
            self.agent_loop_manager._status = AgentLoopManagerStatus.FINISH
            self.agent_loop_manager._finish_event.set()
            await producer_task

    async def _sync_weights_and_save(self, train_step: int, step_timer_dict: dict):
        # 非共卡已经在 _fit 里暂停 producer；这里保持静止态下的 save -> bind -> update 顺序。
        with timer("save_ckpt", step_timer_dict):
            self._maybe_save_checkpoint(train_step)
            self._maybe_save_hf(train_step)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        with timer("sync_weight", step_timer_dict):
            bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
            self.fake_update_weights()

    def fake_update_weights(self):
        self.train_controller.update_weights()
        self.logger.info("Rollout workers updated weights through fake disaggregated sync.")
