from __future__ import annotations

import asyncio
import json
import time
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop import (
    AgentLoopConfig,
    IsValidSampleFn,
    collect_agent_loop_metrics,
    merge_metric_dicts,
)
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.cross_weight_update_rollout import CrossWeightUpdateRolloutConfig
from xtuner.v1.rl.utils import await_cancellable_tasks, await_owned_operation
from xtuner.v1.utils import get_logger

from .disagg_producer import (
    DisaggAsyncProduceStrategyConfig,
    DisaggProduceContext,
    DisaggProduceProgress,
    DisaggProduceStrategy,
    DisaggProduceStrategyConfig,
)
from .produce_utils import (
    _LEFTOVER_STATUSES,
    _MANAGER_STATE_PATH,
    _STATUS_POLL_INTERVAL_S,
    _TASK_CHECKPOINT_DIR,
    ProduceBatchResult,
    ProduceBatchStatus,
    _TaskRunner,
    _TaskSamplerView,
    allocate_task_batch_sizes,
    check_timeout_training_rerollout,
    get_pending_task_counts,
    manager_state_path,
    refresh_for_all_tasks,
    take_train_batch,
    task_checkpoint_path,
)
from .sampler import Sampler, SamplerConfig


if TYPE_CHECKING:
    from xtuner.v1.rl.distillation import RolloutTeacherScorerConfig


class DisaggTaskSpecConfig(BaseModel):
    """单个非共卡 RL 数据源配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    is_valid_sample_fn: IsValidSampleFn | None = None
    produce_strategy_config: DisaggProduceStrategyConfig = DisaggAsyncProduceStrategyConfig()
    sampler_config: SamplerConfig

    @model_validator(mode="after")
    def _check_timeout_training_rerollout(self) -> DisaggTaskSpecConfig:
        check_timeout_training_rerollout(self.agent_loop_config, self.produce_strategy_config)
        return self


class DisaggAgentLoopManagerConfig(BaseModel):
    """非共卡 rollout 后台生产侧配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[DisaggTaskSpecConfig] | DisaggTaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
        sync_weights_interval: int = 1,
        rollout_teacher_scorer_config: RolloutTeacherScorerConfig | None = None,
        *,
        cross_weight_update_rollout: CrossWeightUpdateRolloutConfig | None = None,
    ) -> DisaggAgentLoopManager:
        tasks = self.tasks if isinstance(self.tasks, list) else [self.tasks]
        if not tasks:
            raise ValueError("DisaggAgentLoopManagerConfig requires at least one task config.")

        seen_task_names: set[str] = set()
        task_runners: list[_TaskRunner] = []
        for order, task_cfg in enumerate(tasks):
            if task_cfg.task_name in seen_task_names:
                raise ValueError(f"Duplicate task_name found in DisaggAgentLoopManagerConfig: {task_cfg.task_name}")
            seen_task_names.add(task_cfg.task_name)

            agent_loop = task_cfg.agent_loop_config.build(
                rollout_controller=rollout_controller,
                judger=build_judger(task_cfg.judger_config) if task_cfg.judger_config is not None else None,
                logger=logger,
                is_valid_sample_fn=task_cfg.is_valid_sample_fn,
                rollout_teacher_scorer_config=rollout_teacher_scorer_config,
            )
            produce_strategy = task_cfg.produce_strategy_config.build(
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
                retain_live_tasks=cross_weight_update_rollout is not None,
            )
            sampler = task_cfg.sampler_config.build(tokenizer=tokenizer, replay_buffer=replay_buffer)
            task_runners.append(
                _TaskRunner(
                    task_name=task_cfg.task_name,
                    agent_loop=agent_loop,
                    produce_strategy=produce_strategy,
                    sampler=sampler,
                    weight=task_cfg.weight,
                    order=order,
                )
            )

        return DisaggAgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            rollout_controller=rollout_controller,
            logger=logger,
            cross_weight_update_rollout=cross_weight_update_rollout,
        )


class AgentLoopManagerStatus(Enum):
    """AgentLoopManager 的全局状态.

    按下面的路径流转：
    - 初始状态是 NORMAL
    - NORMAL -> UPDATE_WEIGHT_AND_ABORT
      - trainer 开始做权重同步前触发
    - UPDATE_WEIGHT_AND_ABORT -> NORMAL
      - 权重同步完成后调用 continue_product()
    - NORMAL -> EXPIRED_BATCH
      - 当前 rollout model 已经过旧
    - EXPIRED_BATCH -> UPDATE_WEIGHT_AND_ABORT
      - trainer 检测到过期后，进入权重同步阶段
    - 任意状态 -> FINISH
      - 训练结束

    这里有一个重要区分：
    - AgentLoopManagerStatus 是“后台 producer 的全局运行状态”
    - ProduceBatchStatus 是“单次调度调用的局部结果”
    """

    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def _aggregate_status(statuses: list[ProduceBatchStatus]) -> ProduceBatchStatus:
    if any(status == ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT for status in statuses):
        return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
    if any(status == ProduceBatchStatus.EXPIRED_BATCH for status in statuses):
        return ProduceBatchStatus.EXPIRED_BATCH
    return ProduceBatchStatus.NORMAL


class DisaggAgentLoopManager:
    """非共卡后台 producer / 前台 consumer 状态机。"""

    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    _rollout_controller: RolloutController
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger=None,
        cross_weight_update_rollout: CrossWeightUpdateRolloutConfig | None = None,
    ):
        if not task_runners:
            raise ValueError("DisaggAgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for DisaggAgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self._rollout_controller = rollout_controller
        self.cross_weight_update_rollout = cross_weight_update_rollout
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        self.logger = get_logger() if logger is None else logger
        self.task_names = [task.task_name for task in task_runners]

        # consumer 同步权重前置位；producer / strategy 直接观察 event。
        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()
        # The actual scheduling round. pause waits this Task; registering it has no await after the status
        # check, so a pause cannot miss a round that already started.
        self._produce_pass: asyncio.Task | None = None

        # rollout 侧当前模型版本；pause 清空 pending 后才能更新。
        self._model_step = 0

        # 跨 await 直接读 self._status，避免错过状态变化。
        self._status = AgentLoopManagerStatus.NORMAL

        # pause_produce 写入，下一次 get_batch 消费并清零。
        self._pause_time_s = 0.0

        # producer / consumer 共享绝对进度；对象引用保持稳定。
        self._produce_progress = DisaggProduceProgress.build(self.task_names)

    def _consume_pause_time(self) -> float:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0
        return pause_time_s

    async def _produce_batch_to_buffer(
        self, task_batch_sizes: dict[str, int], progress: DisaggProduceProgress
    ) -> ProduceBatchStatus:
        """One scheduling round: run the task strategies for the planned targets.

        Target planning, the future step and the manager status stay in ``produce_loop``; this method only reports
        the aggregated round result.
        """
        train_step = progress.producer_future_step
        expired_tasks = [
            task.task_name
            for task in self.task_runners
            if cast(DisaggProduceStrategy, task.produce_strategy).is_model_expired(train_step, self._model_step)
        ]
        if expired_tasks:
            self.logger.info(
                f"[DisaggAgentLoopManager][{self.name}] EXPIRED_BATCH: future_step={train_step}, tasks={expired_tasks}"
            )
            return ProduceBatchStatus.EXPIRED_BATCH
        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        produce_start = time.perf_counter()
        contexts = {
            task.task_name: DisaggProduceContext(
                agent_loop=task.agent_loop,
                sampler=task.sampler,
                replay_buffer=self.replay_buffer,
                task_batch_size=task_batch_sizes[task.task_name],
                task_name=task.task_name,
                train_step=train_step,
                model_step=self._model_step,
                progress=progress,
                update_event=self._update_event,
                stale_threshold=task.stale_threshold,
                token_stale_threshold=task.token_stale_threshold,
                expired_groups_retryable=task.expired_groups_retryable,
            )
            for task in active_tasks
        }

        async def run_strategy(task: _TaskRunner) -> ProduceBatchStatus:
            try:
                return await cast(DisaggProduceStrategy, task.produce_strategy).produce_batch(contexts[task.task_name])
            except (Exception, asyncio.CancelledError):
                # One failing scheduler stops the others from refilling; wait for them all before propagating.
                self._update_event.set()
                raise

        status = _aggregate_status(await await_cancellable_tasks(run_strategy(task) for task in active_tasks))
        # 原计时边界：目标规划、提前过期、异常路径都不计入生产耗时。
        progress.add_produce_time(time.perf_counter() - produce_start)
        return status

    def _pending_context(self, task: _TaskRunner) -> DisaggProduceContext:
        return DisaggProduceContext(
            agent_loop=task.agent_loop,
            sampler=task.sampler,
            replay_buffer=self.replay_buffer,
            task_batch_size=0,
            task_name=task.task_name,
            train_step=self._produce_progress.producer_future_step,
            model_step=self._model_step,
            progress=self._produce_progress,
            update_event=self._update_event,
            stale_threshold=task.stale_threshold,
            token_stale_threshold=task.token_stale_threshold,
            expired_groups_retryable=task.expired_groups_retryable,
        )

    async def pause_produce(self) -> float:
        # 先停止补采并等待本轮调度退出，避免恢复时清除 event，让尚未退出的旧轮次继续调度。
        # 此处不等全部 live Agent 自然结束；它们仍可继续工具/评分并交付。
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._update_event.set()
        if self._produce_pass is not None:
            await await_owned_operation(self._produce_pass)
        await self._rollout_controller.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            pause_time_s += await produce_strategy.pause_produce(self._pending_context(task))
        self._pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        # 与 pause_produce 成对：同步/评测完成后，用新 model_step 恢复后台 producer。
        self._model_step = model_step
        await self._rollout_controller.continue_generation.remote(model_step=model_step)  # type: ignore[attr-defined]
        self._update_event.clear()
        self._status = AgentLoopManagerStatus.NORMAL

    def shutdown(self) -> None:
        self._status = AgentLoopManagerStatus.FINISH
        self._update_event.set()
        self._finish_event.set()

    async def close(self) -> None:
        """Stop the producer state machine and wait for every retained Agent
        task to finish cleanup."""
        self.shutdown()
        errors: list[BaseException] = []
        if self._produce_pass is not None:
            try:
                await await_owned_operation(self._produce_pass)
            except (Exception, asyncio.CancelledError) as scheduling_failure:
                errors.append(scheduling_failure)
        # Close every registered complete group even if the scheduling round failed.
        results = await asyncio.gather(
            *[task.produce_strategy.close() for task in self.task_runners], return_exceptions=True
        )
        errors.extend(result for result in results if isinstance(result, BaseException))
        for cleanup_failure in errors:
            if not isinstance(cleanup_failure, asyncio.CancelledError):
                raise cleanup_failure

    async def _wait_for_status_exit(self, blocked_status: AgentLoopManagerStatus) -> None:
        while not self._finish_event.is_set() and self._status == blocked_status:
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    async def produce_loop(self, batch_size: int) -> None:
        # 后台持续生产；前台通过 get_batch 消费。
        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.FINISH:
                break
            if self._status in (AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT, AgentLoopManagerStatus.EXPIRED_BATCH):
                # 暂停/过期只能由 trainer 调用 continue_produce 恢复。
                await self._wait_for_status_exit(self._status)
                continue

            task_batch_sizes = self._produce_progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=self._produce_progress.producer_future_step,
                allocate_batch_sizes=lambda current_batch_size, future_step: allocate_task_batch_sizes(
                    self.task_runners,
                    current_batch_size,
                    future_step,
                ),
            )
            # 仅将原调用登记为实际 Task；登记前没有 await，pause 不会漏等本轮调度。
            self._produce_pass = asyncio.create_task(
                self._produce_batch_to_buffer(task_batch_sizes, self._produce_progress)
            )
            produce_status = await self._produce_pass

            if produce_status == ProduceBatchStatus.EXPIRED_BATCH:
                # 已生效的暂停/退出状态不能被旧轮次的 EXPIRED 覆盖。
                if self._status == AgentLoopManagerStatus.NORMAL:
                    self._status = AgentLoopManagerStatus.EXPIRED_BATCH
            elif produce_status == ProduceBatchStatus.NORMAL:
                # 正常完成一轮即推进，不附加 manager 状态或模型版本条件。
                self._produce_progress.advance_future_step()

            # 极快路径下主动让出事件循环。
            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        # 非共卡消费入口；空 batch 只表示已过期且已有更新模型可同步。
        progress = self._produce_progress
        progress.begin_consume(train_step)
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED, Status.EXPIRED],
        )
        task_batch_sizes = allocate_task_batch_sizes(self.task_runners, batch_size, train_step)
        current_model_step = train_step - 1

        while not self._finish_event.is_set():
            if self.cross_weight_update_rollout is not None:
                await self._rollout_controller.assert_request_gates_healthy.remote()  # type: ignore[attr-defined]
            if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                if current_model_step > self._model_step:
                    pause_time_s = self._consume_pause_time()
                    result = ProduceBatchResult(
                        rollout_states=[],
                        status=ProduceBatchStatus.EXPIRED_BATCH,
                    )
                    if pause_time_s > 0:
                        result.group_gen_pause_time_s = pause_time_s
                    return result
                # producer 已停且没有新模型可同步，立即暴露坏状态。
                if not await self.replay_buffer.is_ready(task_batch_sizes):
                    leftover_counts = await self.replay_buffer.count_statuses(self.task_names, _LEFTOVER_STATUSES)
                    raise RuntimeError(
                        "AgentLoopManager reached EXPIRED_BATCH without a newer model or a ready batch: "
                        f"train_step={train_step}, current_model_step={current_model_step}, "
                        f"rollout_model_step={self._model_step}, manager_status={self._status.name}, "
                        f"producer_future_step={progress.producer_future_step}, "
                        f"next_consumer_step={progress.next_consumer_step}, "
                        f"target_upto_future_step={progress.target_upto_future_step}, "
                        f"target_samples={progress.target_samples}, "
                        f"consumed_samples={progress.consumed_samples}, "
                        f"task_batch_sizes={task_batch_sizes}, "
                        f"leftover_status_counts={leftover_counts}"
                    )
            if await self.replay_buffer.is_ready(task_batch_sizes):
                result = await take_train_batch(
                    task_runners=self.task_runners,
                    replay_buffer=self.replay_buffer,
                    logger=self.logger,
                    manager_name=self.name,
                    task_batch_sizes=task_batch_sizes,
                    progress=progress,
                    current_train_step=train_step,
                    pause_time_s=self._consume_pause_time(),
                )
                if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                    # 有数据的 expired batch 仍需训练本 step。
                    result.status = ProduceBatchStatus.EXPIRED_BATCH
                if result.rollout_states:
                    progress.finish_consume(train_step)
                    await refresh_for_all_tasks(
                        task_runners=self.task_runners,
                        replay_buffer=self.replay_buffer,
                        logger=self.logger,
                        manager_name=self.name,
                        train_step=train_step + 1,
                        statuses=[Status.COMPLETED, Status.ABORTED, Status.EXPIRED],
                    )
                    # Only the real return path reaches the trainer's _log_step; draining on
                    # the EXPIRED_BATCH early return would drop counters without logging them.
                    window_metrics = merge_metric_dicts(
                        await asyncio.gather(
                            *(collect_agent_loop_metrics(task.agent_loop) for task in self.task_runners)
                        )
                    )
                    if window_metrics:
                        result.custom_scalars = window_metrics
                    return result
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

        return ProduceBatchResult(rollout_states=[])

    def _progress_state_without_replay_buffer(self, progress_state: dict) -> dict:
        progress_state = dict(progress_state)
        consumed_samples = dict(progress_state["consumed_samples"])
        task_names = list(consumed_samples)
        next_consumer_step = int(progress_state["next_consumer_step"])

        progress_state["producer_future_step"] = next_consumer_step
        progress_state["target_samples"] = dict(consumed_samples)
        progress_state["target_upto_future_step"] = max(0, next_consumer_step - 1)
        progress_state["raw_rewards_sum"] = {task_name: 0.0 for task_name in task_names}
        progress_state["raw_rewards_count"] = {task_name: 0 for task_name in task_names}
        progress_state["produced_samples"] = {task_name: 0 for task_name in task_names}
        progress_state["produced_tokens"] = {task_name: 0 for task_name in task_names}
        progress_state["produce_time_s"] = 0.0
        return progress_state

    async def save(
        self,
        checkpoint_path: Path | str,
        model_step: int,
        *,
        no_save_replay_buffer: bool = False,
    ) -> None:
        """保存暂停补采后的 sampler、buffer 和生产进度。

        不保存 live Agent/工具状态，三者也不是原子快照；冷恢复不保证样本序列和统计严格一致。
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = get_pending_task_counts(self.task_runners)
        if pending_task_counts:
            if self.cross_weight_update_rollout is None:
                raise RuntimeError(
                    "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                    f"{pending_task_counts}. Call pause_produce() first."
                )
            self.logger.warning(
                f"Non-exact rollout checkpoint: tracked group Tasks={pending_task_counts}. "
                "Live Agent/tool state is not saved; the sampler may be ahead of the buffer. "
                "Cold resume can lose unfinished groups and change sample order and metrics."
            )
        self._model_step = model_step
        for task in self.task_runners:
            checkpoint_dir = task_checkpoint_path(checkpoint_path, task.task_name)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            task.sampler.save(checkpoint_dir)
        if no_save_replay_buffer:
            self.logger.info(f"Skip saving replay buffer to {checkpoint_path}")
        else:
            await self.replay_buffer.save(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        progress_state = self._produce_progress.state_dict()
        if no_save_replay_buffer:
            progress_state = self._progress_state_without_replay_buffer(progress_state)
        with state_path.open("w") as f:
            json.dump(
                {
                    # 完整组 Task 存档格式标识；恢复时先验证，旧格式直接拒绝。
                    "group_task_lifecycle_version": 1,
                    "status": self._status.name,
                    "model_step": self._model_step,
                    "replay_buffer_saved": not no_save_replay_buffer,
                    **progress_state,
                },
                f,
            )

    async def resume(self, checkpoint_path: Path | str) -> int:
        """恢复非共卡 sampler、replay buffer 和后台生产进度。"""
        checkpoint_path = Path(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        # 在恢复任何数据前拒绝旧格式，不补字段、不转换。
        if manager_state.get("group_task_lifecycle_version") != 1:
            raise ValueError("Only complete-group checkpoint version 1 is supported")
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))

        if manager_state["replay_buffer_saved"]:
            # replay buffer 恢复是 async I/O，不能在已有 event loop 中再次嵌套 asyncio_run。
            await self.replay_buffer.resume(checkpoint_path)
        elif len(self.replay_buffer) > 0:
            raise RuntimeError("Cannot resume without replay buffer checkpoint into a non-empty buffer")
        else:
            self.logger.info(f"Skip replay buffer resume for checkpoint without replay buffer: {checkpoint_path}")
        saved_model_step = manager_state["model_step"]
        self._produce_progress.load_state_dict(manager_state)

        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()
        self._update_event.set()
        self._produce_pass = None
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._pause_time_s = 0.0
        self._model_step = saved_model_step
        if self.cross_weight_update_rollout is not None:
            self.logger.warning(
                "Non-exact rollout resume: live Agent/tool state is not restored. "
                "Unfinished groups can be lost; sample order and metrics may differ from uninterrupted training."
            )
        return saved_model_step
