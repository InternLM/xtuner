import asyncio
import json
import math
from pathlib import Path
from typing import Union

import ray
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from transformers import AutoTokenizer
from xtuner.v1._writer import get_writer
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.data_proto import RolloutState, refresh_seq_staleness
from xtuner.v1.rl.agent_loop import (
    ColocatedAgentLoopManagerConfig,
    DisaggregatedMultiTaskAgentLoopManagerConfig,
    DisaggregatedSingleTaskAgentLoopManagerConfig,
    ProduceBatchResult,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.judger import JudgerConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer.worker import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run
from xtuner.v1.train.rl_colocate_trainer import (
    RLColocateTrainer,
    TrainInfo,
    check_fa3,
    force_set_tokenize_workers,
)
from xtuner.v1.train.trainer import LoadCheckpointConfig, XTunerMeta
from xtuner.v1.utils import is_hf_model_path, set_deterministic, timer


class DisaggregatedExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_batch_size: int
    total_train_steps: int
    trigger_parameter_sync_step: int = 1
    staleness_threshold: float = 0.0
    partial_rollout: bool = False
    completed_batch_timeout_s: float | None = 1800.0
    # 这几个字段决定的是 trainer 的“窗口节奏”，不是 producer 的 buffer policy。

    def model_post_init(self, __context) -> None:
        if self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be positive, got {self.train_batch_size}")
        if self.total_train_steps <= 0:
            raise ValueError(f"total_train_steps must be positive, got {self.total_train_steps}")
        if self.trigger_parameter_sync_step <= 0:
            raise ValueError(
                "trigger_parameter_sync_step must be positive, "
                f"got {self.trigger_parameter_sync_step}"
            )
        if self.staleness_threshold < 0:
            raise ValueError(f"staleness_threshold must be non-negative, got {self.staleness_threshold}")
        if self.completed_batch_timeout_s is not None and self.completed_batch_timeout_s <= 0:
            raise ValueError(
                "completed_batch_timeout_s must be positive when provided, "
                f"got {self.completed_batch_timeout_s}"
            )


class RLDisaggregatedTrainerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    train_resources: AcceleratorResourcesConfig
    rollout_resources: AcceleratorResourcesConfig
    train_worker_cfg: WorkerConfig
    rollout_config: RolloutConfig
    judger_config: JudgerConfig
    tokenizer_path: Union[str, Path]
    replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig = SyncReplayBufferConfig()
    agent_loop_manager_cfg: DisaggregatedMultiTaskAgentLoopManagerConfig | DisaggregatedSingleTaskAgentLoopManagerConfig
    eval_agent_loop_manager_cfg: ColocatedAgentLoopManagerConfig
    evaluator_config: EvaluatorConfig
    load_from: Union[str, Path]
    execution_config: DisaggregatedExecutionConfig

    enable_evaluate: bool = True
    enable_initial_evaluate: bool = False
    evaluate_step: int = 1
    work_dir: Union[Path, str, None] = None
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    hf_interval: int | None = -1
    hf_max_keep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    log_dir: Union[Path, str, None] = None
    seed: int = 66
    debug_rollout: bool = False
    skip_checkpoint_validation: bool = False
    exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard"

    def build(self) -> "RLDisaggregatedTrainer":
        return RLDisaggregatedTrainer(
            train_resources=self.train_resources,
            rollout_resources=self.rollout_resources,
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
            load_checkpoint_cfg=self.load_checkpoint_cfg,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_maxkeep=self.checkpoint_maxkeep,
            checkpoint_no_save_optimizer=self.checkpoint_no_save_optimizer,
            hf_interval=self.hf_interval,
            hf_max_keep=self.hf_max_keep,
            load_from=self.load_from,
            log_dir=self.log_dir,
            seed=self.seed,
            debug_rollout=self.debug_rollout,
            skip_checkpoint_validation=self.skip_checkpoint_validation,
            execution_config=self.execution_config,
            exp_tracker=self.exp_tracker,
        )


class RLDisaggregatedTrainer(RLColocateTrainer):
    _META_PATH = ".xtuner_rl_disaggregated_trainer"

    def __init__(
        self,
        *,
        train_resources: AcceleratorResourcesConfig,
        rollout_resources: AcceleratorResourcesConfig,
        train_worker_cfg: WorkerConfig,
        rollout_config: RolloutConfig,
        judger_config: JudgerConfig,
        tokenizer_path: str | Path,
        replay_buffer_config: SyncReplayBufferConfig | AsyncReplayBufferConfig,
        agent_loop_manager_cfg: DisaggregatedMultiTaskAgentLoopManagerConfig | DisaggregatedSingleTaskAgentLoopManagerConfig,
        eval_agent_loop_manager_cfg: ColocatedAgentLoopManagerConfig,
        evaluator_config: EvaluatorConfig,
        enable_evaluate: bool = True,
        enable_initial_evaluate: bool = False,
        evaluate_step: int = 1,
        work_dir: Path | str | None = None,
        auto_resume: bool = False,
        load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig(),
        checkpoint_interval: int | None = -1,
        checkpoint_maxkeep: int | None = -1,
        checkpoint_no_save_optimizer: bool = False,
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        load_from: str | Path,
        log_dir: Path | str | None = None,
        seed: int = 66,
        debug_rollout: bool = False,
        skip_checkpoint_validation: bool = False,
        execution_config: DisaggregatedExecutionConfig,
        exp_tracker: Literal["tensorboard", "jsonl"] = "tensorboard",
    ):
        check_fa3()

        work_dir = Path(work_dir) if work_dir else Path.cwd() / "work_dirs"
        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)
        self._meta = XTunerMeta.build(work_dir, self._META_PATH, auto_resume)

        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from
        is_hf_path, error_info = is_hf_model_path(load_from) if load_from is not None else (False, "")
        self._load_from_hf = is_hf_path
        if not self._load_from_hf:
            raise NotImplementedError(error_info)
        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_maxkeep = checkpoint_maxkeep
        self._checkpoint_no_save_optimizer = checkpoint_no_save_optimizer
        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(auto_resume, load_checkpoint_cfg)

        log_dir = self.exp_dir / "logs"
        self.logger = self._build_logger(log_dir)

        # force_set_tokenize_workers(self.logger)
        self.logger.warning(
            "Disaggregated weight sync is currently mocked. The real cross-device weight update path is expected "
            "to be provided by the follow-up weight sync module."
        )

        if skip_checkpoint_validation:
            patch_default_save_plan()

        self._execution_config = execution_config
        # 这里的 total_train_steps 指的是训练步，不是 rollout launch 次数。
        self._total_train_steps = execution_config.total_train_steps
        # 共享的 RLColocateTrainer 日志/保存逻辑仍然读取 _rollout_steps。
        # 对 disaggregated 来说两者语义都是“总训练步数”，这里保留一个兼容别名，
        # 避免在共享逻辑里继续散落一批 if hasattr(...) 分支。
        self._rollout_steps = self._total_train_steps
        self._cur_step = 0
        self._global_train_step = 0
        self._seed = seed
        set_deterministic()
        set_random_seed(seed)
        self.train_batch_size = execution_config.train_batch_size

        pg_name_prefix = f"disaggregated_{self.exp_dir.name}"
        # 训练和 rollout 明确分成两套 placement group，
        # 这是 disaggregated 和 colocated 在资源层面最本质的区别。
        self._train_pg = AutoAcceleratorWorkers.build_placement_group(
            train_resources, name=f"{pg_name_prefix}_train"
        )
        self._rollout_pg = AutoAcceleratorWorkers.build_placement_group(
            rollout_resources, name=f"{pg_name_prefix}_rollout"
        )

        if train_worker_cfg.seed is None:
            self.logger.warning(f"RLDisaggregatedTrainer seed {seed} is used as train worker seed.")
            train_worker_cfg.seed = seed
        train_worker_cfg.load_from = load_from
        train_worker_cfg.log_dir = log_dir
        train_worker_cfg.use_rollout_logprobs_as_old_logprobs = True
        self._train_worker_cfg = train_worker_cfg

        rollout_config.worker_log_dir = log_dir

        self.train_controller = train_worker_cfg.build(self._train_pg)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            ray.get(self.train_controller.resume.remote(self._load_checkpoint_cfg))
            train_state_path = Path(self._load_checkpoint_cfg.checkpoint_path) / self._SAVE_TRAIN_STATE_PATH
            with train_state_path.open("r") as f:
                train_state = json.load(f)
                self._cur_step = train_state["cur_step"]
            self.logger.warning(
                "Disaggregated weight sync is currently mocked. Rollout workers load weights from load_from after "
                "resume and are not updated from the resumed trainer checkpoint in this implementation."
            )

        self.rollout_controller = rollout_config.build(self._rollout_pg)

        judger = judger_config.build()
        replay_buffer = replay_buffer_config.build()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.agent_loop_manager = agent_loop_manager_cfg.build(
            rollout_controller=self.rollout_controller,
            judger=judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
            logger=self.logger,
        )
        self._log_disaggregated_replay_policy_interactions()

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

        total_eval_samples = len(self.eval_agent_loop_manager.data_sampler)
        self.evaluator = evaluator_config.build(total_eval_samples=total_eval_samples)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            self.logger.info(f"Resume sampler from {self._load_checkpoint_cfg.checkpoint_path}")
            self.agent_loop_manager.resume(self._load_checkpoint_cfg.checkpoint_path)

        if debug_rollout:
            self.logger.warning("Debug rollout mode is enabled, training and weight sync will be skipped.")
        self._debug_rollout = debug_rollout
        self._exp_tracker = get_writer(writer_type=exp_tracker, log_dir=log_dir / self._EXP_TRACKING_PATH)
        self._display_all_workers_log = False

    def _build_logger(self, log_dir: Path):
        from xtuner.v1.utils import get_logger

        return get_logger(log_dir=log_dir, tag="RLDisaggregatedTrainer")

    def _log_disaggregated_replay_policy_interactions(self) -> None:
        if self._execution_config.staleness_threshold <= 0:
            return

        for task in self.agent_loop_manager.task_runners:
            produce_strategy = task.produce_strategy
            tail_batch_stale_threshold = getattr(produce_strategy, "tail_batch_stale_threshold", 0)
            tail_batch_trigger_size = getattr(produce_strategy, "tail_batch_trigger_size", 0)
            if tail_batch_stale_threshold > 0 or tail_batch_trigger_size > 0:
                self.logger.warning(
                    "Disaggregated task %s uses staleness_threshold=%s together with "
                    "tail_batch_stale_threshold=%s and tail_batch_trigger_size=%s. "
                    "These knobs are coupled: aggressive tail-batch expiration can discard most window over-production, "
                    "while large expiration thresholds keep older leftovers eligible for training longer.",
                    task.task_name,
                    self._execution_config.staleness_threshold,
                    tail_batch_stale_threshold,
                    tail_batch_trigger_size,
                )

    def fit(self):
        asyncio_run(self._fit_async())

    async def _fit_async(self):
        self.logger.info("Start disaggregated RL training")
        if self._cur_step >= self._total_train_steps:
            self.logger.info(f"Train steps {self._total_train_steps} reached, stop training")
            return

        if self._enable_initial_evaluate and not self._debug_rollout:
            eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
                self.evaluator.eval_batch_size, rollout_step=0
            )
            eval_metrics = self.evaluator.run(eval_produce_result.rollout_states)
            self.logger.info(f"Initial rollout evaluate scores {eval_metrics} and start training")
            self._exp_tracker.add_scalars(
                tag_scalar_dict={f"eval/{k}": v for k, v in eval_metrics.items()},
                global_step=0,
            )

        while self._cur_step < self._total_train_steps:
            start_train_step = self._cur_step + 1
            remaining_train_steps = self._total_train_steps - self._cur_step
            window_train_steps = min(
                self._execution_config.trigger_parameter_sync_step,
                remaining_train_steps,
            )
            # 当前版本仍然是“按 window 同步推进”：
            # 一个 window 内先产数据，再按步取 batch 训练，最后统一做一次权重同步。
            await self._run_sync_window(start_train_step, window_train_steps)

    async def _run_sync_window(self, start_train_step: int, window_train_steps: int):
        end_train_step = start_train_step + window_train_steps - 1
        base_window_batch_size = self.train_batch_size * window_train_steps
        # required 是这个 window 真正训练要消耗的样本数；
        # target 则可能因为 staleness_threshold 被放大，用来给 replay buffer 留冗余。
        window_task_batch_sizes = self._get_window_required_batch_sizes(
            start_rollout_step=start_train_step,
            train_steps=window_train_steps,
        )
        target_task_batch_sizes = self._get_window_target_task_batch_sizes(window_task_batch_sizes)
        self.logger.info(
            f"Train window {start_train_step}-{end_train_step}/{self._total_train_steps} start, "
            f"train_batch_size={self.train_batch_size}, base_window_batch_size={base_window_batch_size}, "
            f"target_batch_size={sum(target_task_batch_sizes.values())}, "
            f"trigger_parameter_sync_step={self._execution_config.trigger_parameter_sync_step}, "
            f"staleness_threshold={self._execution_config.staleness_threshold}, "
            f"partial_rollout={self._execution_config.partial_rollout}"
        )

        produce_task = asyncio.create_task(
            # produce_task 会在整个 window 生命周期里后台持续往 replay buffer 里灌数据。
            self._produce_window_to_replay_buffer(
                required_task_batch_sizes=window_task_batch_sizes,
                target_task_batch_sizes=target_task_batch_sizes,
                rollout_step=start_train_step,
            )
        )

        last_step_timer_dict = {}
        last_produce_result: ProduceBatchResult | None = None
        last_train_log_info: TrainInfo = {}
        for train_step in range(start_train_step, end_train_step + 1):
            self.logger.info(f"Train step {train_step}/{self._total_train_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                with timer("wait_rollout", step_timer_dict):
                    # 每个 train_step 只从 replay buffer 取当前步所需的一个 train_batch。
                    produce_result = await self._get_completed_batch_or_raise(
                        produce_task=produce_task,
                        train_step=train_step,
                    )
                train_batch = produce_result.rollout_states
                self._refresh_train_batch_staleness(train_batch, train_step)
                if train_batch:
                    self.logger.info(f"Get {len(train_batch) * len(train_batch[0])} samples for training")
                train_trajectory_dir = self.exp_dir / "train_rollout"
                train_trajectory_dir.mkdir(parents=True, exist_ok=True)
                train_trajectory_path = train_trajectory_dir / f"train_rollout_{train_step}.jsonl"
                self._save_trajectories(train_batch, train_trajectory_path)
                self.logger.info(f"Train step {train_step} trajectories saved to {train_trajectory_path}")

                if not self._debug_rollout:
                    with timer("prepare_data", step_timer_dict):
                        data_batches, data_info = self._prepare_train_data(
                            train_batch, self._train_worker_cfg.pack_max_length
                        )
                    self.logger.info(f"Prepared {len(data_batches)} training data batches")

                    with timer("training", step_timer_dict):
                        workers_log_item = await asyncio.to_thread(
                            ray.get,
                            self.train_controller.fit.remote(
                                data_batches,
                                pack_max_length=self._train_worker_cfg.pack_max_length,
                                rollout_idx=train_step,
                            ),
                        )
                    train_log_info: TrainInfo = {
                        "data_info": data_info,
                        "workers_log_item": workers_log_item,
                    }
                else:
                    train_log_info = {}

            if train_step < end_train_step:
                self._log_step(train_step, step_timer_dict, produce_result, train_log_info, {})
            else:
                last_produce_result = produce_result
                last_train_log_info = train_log_info
            self._cur_step = train_step
            last_step_timer_dict = step_timer_dict

        with timer("rollout_window_drain", last_step_timer_dict):
            # 训练步都跑完后，再等待本 window 的 produce_task 收尾，
            # 确保 finally / cleanup 里的 leftovers 已经稳定写回 buffer。
            window_produce_result = await produce_task
        with timer("sync_and_save", last_step_timer_dict):
            self._sync_weights_and_save(end_train_step, last_step_timer_dict)

        eval_log_info = {}
        if self._enable_evaluate and end_train_step % self._evaluate_step == 0:
            with timer("evaluation", last_step_timer_dict):
                eval_produce_result = await self.eval_agent_loop_manager.produce_batch(
                    self.evaluator.eval_batch_size, rollout_step=end_train_step
                )
                eval_batch = eval_produce_result.rollout_states
                eval_metrics = self.evaluator.run(eval_batch)
                eval_trajectory_dir = self.exp_dir / "eval_rollout"
                eval_trajectory_dir.mkdir(parents=True, exist_ok=True)
                eval_trajectory_path = eval_trajectory_dir / f"eval_rollout_{end_train_step}.jsonl"
                self._save_trajectories(eval_batch, eval_trajectory_path)
                self.logger.info(f"Train step {end_train_step} eval trajectories saved to {eval_trajectory_path}")
                eval_log_info.update(eval_metrics)

        self._log_step(
            end_train_step,
            last_step_timer_dict,
            last_produce_result or window_produce_result,
            last_train_log_info,
            eval_log_info,
        )

    def _get_window_target_task_batch_sizes(self, window_task_batch_sizes: dict[str, int]) -> dict[str, int]:
        # staleness_threshold 的唯一职责就是把 required 放大成 target。
        # 这里不再叠加 strategy 侧的 over-sample 旋钮。
        return {
            task_name: math.ceil(task_batch_size * (1 + self._execution_config.staleness_threshold))
            for task_name, task_batch_size in window_task_batch_sizes.items()
        }

    def _get_window_required_batch_sizes(self, start_rollout_step: int, train_steps: int) -> dict[str, int]:
        return self.agent_loop_manager.get_window_task_batch_sizes(
            train_batch_size=self.train_batch_size,
            start_rollout_step=start_rollout_step,
            train_steps=train_steps,
        )

    async def _produce_window_to_replay_buffer(
        self,
        required_task_batch_sizes: dict[str, int],
        target_task_batch_sizes: dict[str, int],
        rollout_step: int,
    ) -> ProduceBatchResult:
        return await self.agent_loop_manager.produce_window_to_replay_buffer(
            required_task_batch_sizes=required_task_batch_sizes,
            target_task_batch_sizes=target_task_batch_sizes,
            rollout_step=rollout_step,
            enable_partial_rollout=self._execution_config.partial_rollout,
        )

    def _refresh_train_batch_staleness(self, train_batch: list[list[RolloutState]], train_step: int) -> None:
        # 训练前再刷新一次，是为了让真正进入训练的 batch 带上“本训练步视角”的 staleness。
        for group in train_batch:
            for item in group:
                refresh_seq_staleness(item, train_step)

    def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        with timer("save_ckpt", step_timer_dict):
            self._maybe_save_checkpoint(rollout_idx)
            self._maybe_save_hf(rollout_idx)

        ray.get(self.rollout_controller.recover_failed_workers.remote())
        with timer("sync_weight", step_timer_dict):
            self._mock_disaggregated_weight_sync(rollout_idx)

    def _mock_disaggregated_weight_sync(self, rollout_idx: int) -> None:
        self.logger.info(
            f"Mock disaggregated weight sync for rollout_idx={rollout_idx}. The real cross-device weight update path "
            "is expected to be provided by the follow-up weight sync module."
        )

    async def _get_completed_batch_or_raise(
        self,
        produce_task: asyncio.Task,
        train_step: int,
    ) -> ProduceBatchResult:
        poll_interval_s = 1.0
        # 当 producer 已经自然结束时，只允许一个很短的 drain 窗口，
        # 看 replay buffer 里能不能把下一步训练所需 batch 凑齐。
        drain_timeout_s = 2 * poll_interval_s + 0.5
        pending: set[asyncio.Task] = set()
        if produce_task.done():
            # 早检查：避免 producer 早就失败了，但 trainer 还静默继续吃旧缓存很多步。
            await produce_task
            return await self._get_completed_batch(
                train_step=train_step,
                poll_interval_s=poll_interval_s,
                max_wait_s=drain_timeout_s,
            )
        get_batch_task = asyncio.create_task(
            self._get_completed_batch(
                train_step=train_step,
                poll_interval_s=poll_interval_s,
                max_wait_s=self._execution_config.completed_batch_timeout_s,
            )
        )
        try:
            done, pending = await asyncio.wait(
                {get_batch_task, produce_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if produce_task in done:
                try:
                    # 这里先传播 producer 自己的异常/取消，再决定要不要 drain get_batch_task。
                    await produce_task
                except BaseException:
                    if not get_batch_task.done():
                        get_batch_task.cancel()
                        await asyncio.gather(get_batch_task, return_exceptions=True)
                    raise
                if get_batch_task.done():
                    # 最理想情况：producer 收尾时，这一步训练所需 batch 也已经准备好了。
                    return await get_batch_task
                try:
                    # producer 已经结束，但 get_batch_task 还差一点点；给一个很短的补齐窗口。
                    return await asyncio.wait_for(get_batch_task, timeout=drain_timeout_s)
                except (TimeoutError, asyncio.TimeoutError):
                    get_batch_task.cancel()
                    await asyncio.gather(get_batch_task, return_exceptions=True)
                    raise RuntimeError(
                        "Rollout producer finished before the next training batch became available. "
                        f"train_step={train_step}, train_batch_size={self.train_batch_size}, "
                        f"drain_timeout_s={drain_timeout_s}"
                    )

            return await get_batch_task
        finally:
            if get_batch_task in pending and not get_batch_task.done():
                get_batch_task.cancel()
                await asyncio.gather(get_batch_task, return_exceptions=True)

    async def _get_completed_batch(
        self,
        train_step: int,
        poll_interval_s: float,
        max_wait_s: float | None,
    ) -> ProduceBatchResult:
        return await self.agent_loop_manager.get_completed_batch(
            self.train_batch_size,
            rollout_step=train_step,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )
