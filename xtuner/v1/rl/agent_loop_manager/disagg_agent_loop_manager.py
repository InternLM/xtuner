import asyncio
import json
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop import AgentLoopConfig
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
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
    get_pending_task_counts,
    manager_state_path,
    refresh_for_all_tasks,
    take_train_batch,
    task_checkpoint_path,
)
from .sampler import Sampler, SamplerConfig


class DisaggTaskSpecConfig(BaseModel):
    """单个非共卡 RL 数据源配置。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    produce_strategy_config: DisaggProduceStrategyConfig = DisaggAsyncProduceStrategyConfig()
    sampler_config: SamplerConfig


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
    ) -> "DisaggAgentLoopManager":
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
            )
            produce_strategy = task_cfg.produce_strategy_config.build(
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
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
    ):
        if not task_runners:
            raise ValueError("DisaggAgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for DisaggAgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self._rollout_controller = rollout_controller
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
        self,
        task_batch_sizes: dict[str, int],
        progress: DisaggProduceProgress,
    ) -> ProduceBatchStatus:
        producer_train_step = progress.producer_future_step
        expired_tasks = []
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            if produce_strategy.is_model_expired(producer_train_step, self._model_step):
                expired_tasks.append(task.task_name)
        if expired_tasks:
            self.logger.info(
                f"[DisaggAgentLoopManager][{self.name}] EXPIRED_BATCH: "
                f"future_step={producer_train_step}, tasks={expired_tasks}"
            )
            return ProduceBatchStatus.EXPIRED_BATCH

        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        produce_start = time.perf_counter()
        produce_futures = []
        for task in active_tasks:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            produce_futures.append(
                produce_strategy.produce_batch(
                    DisaggProduceContext(
                        agent_loop=task.agent_loop,
                        sampler=task.sampler,
                        replay_buffer=self.replay_buffer,
                        task_batch_size=task_batch_sizes[task.task_name],
                        task_name=task.task_name,
                        train_step=producer_train_step,
                        model_step=self._model_step,
                        progress=progress,
                        update_event=self._update_event,
                        is_valid_sample_fn=task.is_valid_sample_fn,
                        stale_threshold=task.stale_threshold,
                    )
                )
            )
        produce_status = _aggregate_status(await asyncio.gather(*produce_futures))
        progress.add_produce_time(time.perf_counter() - produce_start)
        return produce_status

    async def pause_produce(self) -> float:
        # 非共卡显式刹车；共卡没有 public pause。
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._update_event.set()
        await self._rollout_controller.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            ctx = DisaggProduceContext(
                agent_loop=task.agent_loop,
                sampler=task.sampler,
                replay_buffer=self.replay_buffer,
                task_batch_size=0,
                task_name=task.task_name,
                train_step=self._produce_progress.producer_future_step,
                model_step=self._model_step,
                progress=self._produce_progress,
                update_event=self._update_event,
                is_valid_sample_fn=task.is_valid_sample_fn,
                stale_threshold=task.stale_threshold,
            )
            pause_time_s += await produce_strategy.pause_produce(ctx)
        self._pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        # 与 pause_produce 成对：同步/评测完成后，用新 model_step 恢复后台 producer。
        self._model_step = model_step
        await self._rollout_controller.continue_generation.remote()  # type: ignore[attr-defined]
        self._status = AgentLoopManagerStatus.NORMAL
        self._update_event.clear()

    def shutdown(self) -> None:
        self._status = AgentLoopManagerStatus.FINISH
        self._update_event.set()
        self._finish_event.set()

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
            produce_status = await self._produce_batch_to_buffer(task_batch_sizes, self._produce_progress)

            if produce_status == ProduceBatchStatus.EXPIRED_BATCH:
                # EXPIRED_BATCH 是 producer 自己检测出来的“立即停下”信号。
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
            elif produce_status == ProduceBatchStatus.NORMAL:
                # 只有正常完成一轮生产时，producer 自己维护的 train_step 才前进一步。
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
            statuses=[Status.COMPLETED, Status.ABORTED],
        )
        task_batch_sizes = allocate_task_batch_sizes(self.task_runners, batch_size, train_step)
        current_model_step = train_step - 1

        while not self._finish_event.is_set():
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
                        statuses=[Status.COMPLETED, Status.ABORTED],
                    )
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
        """保存非共卡 sampler、replay buffer 和后台生产进度。"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = get_pending_task_counts(self.task_runners)
        if pending_task_counts:
            raise RuntimeError(
                "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                f"{pending_task_counts}. Call pause_produce() first."
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
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))

        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        if manager_state.get("replay_buffer_saved", True):
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
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._pause_time_s = 0.0
        self._model_step = saved_model_step
        return saved_model_step
