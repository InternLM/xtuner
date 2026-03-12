import os
import random
from pathlib import Path
from typing import Any, List, Union, cast

import ray
import torch
from mmengine.dist import get_rank
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal, TypedDict

from transformers import AutoTokenizer
from xtuner.v1._writer import get_writer
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.rl.agent_loop import AgentLoopManagerConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import JudgerConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.controller import RolloutControllerProxy
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.controller import TrainingControllerProxy
from xtuner.v1.rl.trainer.worker import WorkerConfig, WorkerLogItem
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run
from xtuner.v1.train.trainer import XTunerMeta
from xtuner.v1.utils import get_logger, timer
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


def bind_train_rollout(
    train_controller: TrainingControllerProxy,
    rollout_controller: RolloutControllerProxy,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


class TrainInfo(TypedDict, total=False):
    data_info: dict[str, float]
    workers_log_item: list[WorkerLogItem]


def get_train_seq_ctx(
    input_ids: torch.LongTensor, multimodal_train_info: dict | None = None, len_response_ids: int = 0
):
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    if multimodal_train_info and len(multimodal_train_info) > 0:
        position_ids = multimodal_train_info.get("position_ids")  # (1,n) or (3,1,n)
        if position_ids is not None and len(position_ids.shape) == 3:
            # qwen3vl 需要特殊处理，其余的不需要额外处理
            max_value = position_ids.max(dim=-1).values  # (3,1)
            response_position_ids = max_value.unsqueeze(-1).expand(-1, -1, len_response_ids) + torch.arange(
                1, len_response_ids + 1, device=max_value.device
            )
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            seq_ctx.position_ids = position_ids  # type: ignore[assignment]
            assert position_ids.size(-1) == input_ids.size(-1)
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


class RLColocateTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    resources: AcceleratorResourcesConfig
    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    judger_config: JudgerConfig
    tokenizer_path: Union[str, Path]
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: AgentLoopManagerConfig
    eval_agent_loop_manager_cfg: AgentLoopManagerConfig
    evaluator_config: EvaluatorConfig
    load_from: Union[str, Path]
    rollout_steps: int
    global_batch_size: int

    enable_evaluate: bool = True
    enable_initial_evaluate: bool = False
    evaluate_step: int = 1
    work_dir: Union[Path, str, None] = None
    auto_resume: bool = False
    log_dir: Union[Path, str, None] = None
    seed: int = 66
    debug_rollout: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"

    def build(self) -> "RLColocateTrainer":
        return RLColocateTrainer(
            resources=self.resources,
            train_worker_cfg=self.train_worker_cfg,
            rollout_config=self.rollout_config,
            judger_config=self.judger_config,
            tokenizer_path=self.tokenizer_path,
            replay_buffer_config=self.replay_buffer_config,
            agent_loop_manager_cfg=self.agent_loop_manager_cfg,
            eval_agent_loop_manager_cfg=self.eval_agent_loop_manager_cfg,
            evaluator_config=self.evaluator_config,
            enable_evaluate=self.enable_evaluate,
            enable_initial_evaluate=self.enable_initial_evaluate,
            evaluate_step=self.evaluate_step,
            work_dir=self.work_dir,
            auto_resume=self.auto_resume,
            load_from=self.load_from,
            log_dir=self.log_dir,
            seed=self.seed,
            debug_rollout=self.debug_rollout,
            skip_checkpoint_validation=self.skip_checkpoint_validation,
            rollout_steps=self.rollout_steps,
            global_batch_size=self.global_batch_size,
            exp_tracker=self.exp_tracker,
        )


class RLColocateTrainer:
    _META_PATH = ".xtuner_rl_colocate_trainer"
    _EXP_TRACKING_PATH = "exp_tracking"

    # 弱化Trainer：Trainer中代码尽量少，尽量用componet来组织代码。
    # 目标是像torch一样，让用户自己写init 和 train loop，我们只提供组件。
    def __init__(
        self,
        *,
        resources: AcceleratorResourcesConfig,
        train_worker_cfg: WorkerConfig,
        rollout_config: RolloutConfig,
        judger_config: JudgerConfig,
        # Sampler config
        # sampler_config: SamplerConfig,
        tokenizer_path: str | Path,
        replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig,
        # agent loop config
        # agent_loop_config: AgentLoopConfig,
        # agent loop manager config
        # produce_strategy_config: ProduceStrategyConfig,
        agent_loop_manager_cfg: AgentLoopManagerConfig,
        # eval configs
        eval_agent_loop_manager_cfg: AgentLoopManagerConfig,
        evaluator_config: EvaluatorConfig,
        enable_evaluate: bool = True,
        enable_initial_evaluate: bool = False,
        evaluate_step: int = 1,
        # work_dir and resume
        work_dir: Path | str | None = None,
        auto_resume: bool = False,
        # others
        load_from: str | Path,
        log_dir: Path | str | None = None,
        seed: int = 66,
        debug_rollout: bool = False,
        skip_checkpoint_validation: bool = False,  # Suggest enabled if fsdp_size is larger than 512
        # steps
        rollout_steps: int,
        global_batch_size: int,
        # exp tracker
        exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard",
    ):
        check_fa3()

        # work_dir
        work_dir = Path(work_dir) if work_dir else Path.cwd() / "work_dirs"
        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
        self._meta = XTunerMeta.build(work_dir, self._META_PATH, auto_resume)

        # log
        log_dir = self.exp_dir / "logs"
        self.logger = get_logger(log_dir=log_dir, tag="RLTrainer")

        if skip_checkpoint_validation:
            patch_default_save_plan()

        # steps
        self._rollout_steps = rollout_steps
        # self._total_epochs = total_epochs  # TODO
        self._cur_step = 0
        self._global_train_step = 0
        self.global_batch_size = global_batch_size

        # main components
        self._pg = AutoAcceleratorWorkers.build_placement_group(resources)

        # override train worker config
        if train_worker_cfg.seed is None:
            self.logger.warning(f"RLTrainer seed {seed} is used as train worker seed.")
            train_worker_cfg.seed = seed
        train_worker_cfg.load_from = load_from
        train_worker_cfg.log_dir = log_dir
        self._train_worker_cfg = train_worker_cfg

        # override rollout config
        rollout_config.worker_log_dir = log_dir

        # build train controller and rollout controller
        self.train_controller = train_worker_cfg.build(self._pg)

        self.rollout_controller = rollout_config.build(self._pg)

        # build judger
        judger = judger_config.build()

        replay_buffer = replay_buffer_config.build()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # build agnet_loop_manager
        self.agent_loop_manager = agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            judger=judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )

        # build eval agent loop manager
        self.eval_agent_loop_manager = eval_agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            judger=judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )

        self._enable_evaluate = enable_evaluate
        self._enable_initial_evaluate = enable_initial_evaluate
        self._evaluate_step = evaluate_step

        # build evaluator
        total_eval_samples = len(self.eval_agent_loop_manager._data_sampler)
        self.evaluator = evaluator_config.build(total_eval_samples=total_eval_samples)

        # others
        if debug_rollout:
            self.logger.warning("Debug rollout mode is enabled, rollout will not be offloaded.")
        self._debug_rollout = debug_rollout
        self._exp_tracker = get_writer(writer_type=exp_tracker, log_dir=log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = False

    @property
    def exp_dir(self) -> Path:
        return Path(self._meta.latest_exp.exp_dir)

    def fit(self):
        self.logger.info("Start RL training")
        if self._cur_step >= self._rollout_steps:
            self.logger.info(f"Rollout steps {self._rollout_steps} reached, stop training")
            return

        if self._enable_initial_evaluate and not self._debug_rollout:
            # TODO: ray.get(self.rollout_controller.update_active_workers.remote())
            # TODO: ray.get(self.rollout_controller.restart.remote())
            eval_batch: list[list[RolloutState]] = asyncio_run(
                self.eval_agent_loop_manager.produce_batch(self.evaluator.eval_batch_size, rollout_step=0)
            )
            eval_metrics = self.evaluator.run(eval_batch)
            self.logger.info(f"Initial rollout evaluate scores {eval_metrics} and start training")

            tb_scores = {f"eval/{k}": v for k, v in eval_metrics.items()}
            self._exp_tracker.add_scalars(
                tag_scalar_dict=tb_scores,
                global_step=0,
            )

        for rollout_idx in range(self._cur_step + 1, self._rollout_steps + 1):
            self.logger.info(f"Rollout {rollout_idx}/{self._rollout_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                # 1. Rollout to generate experience
                # TODO: ray.get(self.rollout_controller.check_health.remote())
                self.logger.info("start to generate rollout experience for training")
                train_batch: list[list[RolloutState]] = asyncio_run(
                    self.agent_loop_manager.produce_batch(self.global_batch_size, rollout_step=rollout_idx)
                )
                self.logger.info(f"generate {len(train_batch) * len(train_batch[0])} samples for training")
                rollout_info = {}  # TODO: rollout info?
                # TODO: save train trajectory
                if not self._debug_rollout:
                    # TODO: ray.get(self.rollout_controller.pause_generation.remote())
                    ray.get(self.rollout_controller.offload.remote())

                if not self._debug_rollout:
                    with timer("onload", step_timer_dict):
                        ray.get(self.train_controller.onload.remote(target="all"))
                        self.logger.info("Training controller loaded")

                    # 2. Train on the generated experience
                    # TODO: simplify with Packer.pack_pad_dispatch()
                    # train_batch = Packer.pack_pad_dispatch(train_batch)
                    with timer("prepare_data", step_timer_dict):
                        data_batches, data_info = self._prepare_train_data(
                            train_batch, self._train_worker_cfg.pack_max_length
                        )
                    self.logger.info(f"Prepared {len(data_batches)} training data batches")

                    with timer("training", step_timer_dict):
                        workers_log_item: list[WorkerLogItem] = ray.get(
                            self.train_controller.fit.remote(
                                data_batches,
                                pack_max_length=self._train_worker_cfg.pack_max_length,
                                rollout_idx=rollout_idx,
                            )
                        )
                    train_log_info: TrainInfo = {
                        "data_info": data_info,
                        "workers_log_item": workers_log_item,
                    }

                    # 3. Synchronize weights and save checkpoints
                    self._sync_weights_and_save(rollout_idx, step_timer_dict)

                    # 4. Evaluate model performance
                    eval_log_info = {}
                    if self._enable_evaluate and rollout_idx % self._evaluate_step == 0:
                        with timer("evaluation", step_timer_dict):
                            # TODO: ray.get(self.rollout_controller.restart.remote())
                            eval_batch: list[list[RolloutState]] = asyncio_run(
                                self.eval_agent_loop_manager.produce_batch(
                                    self.evaluator.eval_batch_size, rollout_step=rollout_idx
                                )
                            )
                            eval_metrics = self.evaluator.run(eval_batch)
                            # TODO: save eval trajectory
                            eval_log_info.update(eval_metrics)
                else:
                    train_log_info = {}
                    eval_log_info = {}

            self._log_step(rollout_idx, step_timer_dict, rollout_info, train_log_info, eval_log_info)
            self._cur_step = rollout_idx

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
                if group[i].response_mask:
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
                multimodal_train_info = group[i].mm_info
                multi_info_cast = cast(dict | None, multimodal_train_info)
                seq_ctx = get_train_seq_ctx(input_ids_t, multi_info_cast, len(response_ids) - 1)  # type: ignore[arg-type]
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

    def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        """Synchronizes weights and saves checkpoints."""
        with timer("save_ckpt", step_timer_dict):
            ray.get(self.train_controller.offload.remote(target="optimizer"))
            # TODO:
            # self._maybe_save_hf()
            # self._maybe_save_checkpoint()

        with timer("sync_weight", step_timer_dict):
            bind_train_rollout(train_controller=self.train_controller, rollout_controller=self.rollout_controller)
            ray.get(self.rollout_controller.onload_weights.remote())
            ray.get(self.train_controller.update_weights.remote())
            self.logger.info("Model weights synchronized successfully.")
            ray.get(self.train_controller.offload.remote(target="model"))
            ray.get(self.rollout_controller.onload_kvcache.remote())

    def _log_step(
        self,
        rollout_idx: int,
        step_timer_dict: dict,
        rollout_info: dict,  # RolloutInfo,  # TODO
        train_info: TrainInfo,
        eval_info: dict[str, float],
    ):
        all_scalars = {}
        log_time_str = ""
        trajectory_str = ""
        eval_str = ""
        if step_timer_dict:
            all_scalars.update({f"time/{k}": v for k, v in step_timer_dict.items()})
            log_time_str = f"\nRollout {rollout_idx} finished and timing listed:\n"
            log_time_str += "\n".join([f" - {k:<25}: {v:.2f}s" for k, v in step_timer_dict.items()])

        if rollout_info:
            all_scalars.update(rollout_info.get("task_time", {}))
            all_scalars.update({f"async/{k}": v for k, v in rollout_info.get("replay_buffer_info", {}).items()})

        if train_info:
            all_scalars.update({f"response/{k}": v for k, v in train_info.get("data_info", {}).items()})
            trajectory_str = f"\nRollout {rollout_idx} data statistics:\n"
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

        self.logger.info(f"Rollout {rollout_idx}/{self._rollout_steps}{log_time_str} {trajectory_str} ")
        if eval_str:
            self.logger.info(f"Eval: {eval_str}")
        self._exp_tracker.add_scalars(tag_scalar_dict=all_scalars, global_step=rollout_idx)

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
