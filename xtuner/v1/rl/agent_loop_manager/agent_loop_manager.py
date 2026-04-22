import asyncio
import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.agent_loop import AgentLoopConfig, AgentLoopSpec, get_agent_loop_rollout_ctl
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation, pause_generation
from xtuner.v1.rl.utils import asyncio_run
from xtuner.v1.utils import get_logger

from .producer import (
    GROUP_GENERATE_TIME_KEY,
    AsyncProduceStrategy,
    ProduceBatchStatus,
    ProduceProgress,
    ProduceStrategy,
    ProduceStrategyConfig,
    SyncProduceStrategyConfig,
)
from .sampler import Sampler, SamplerConfig


@dataclass
class ProduceBatchResult:
    """Result of a single ``produce_batch`` call.

    Args:
        rollout_states (list[list[RolloutState]]): Completed rollout groups retrieved from the replay buffer for training.
        group_gen_count (int | None): Number of generate-group calls finished in this batch (None if no generations ran).
        group_gen_mean_s (float | None): Mean wall-clock time per generate-group call, in seconds.
        group_gen_p50_s (float | None): Median (p50) generate-group time, in seconds.
        group_gen_p99_s (float | None): 99th percentile generate-group time, in seconds.
        group_gen_p99_p50_ratio (float | None): Ratio of p99 to p50, indicating tail-latency skew.
        group_gen_pause_time_s (float | None): Time spent in pause/cleanup phase (async strategy only), in seconds.
        leftover_completed (int): Number of completed groups remaining in the replay buffer after this batch.
        leftover_aborted (int): Number of aborted groups remaining in the replay buffer.
        leftover_expired (int): Number of expired groups remaining in the replay buffer.
    """

    rollout_states: list[list[RolloutState]]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    # per-group generation timing stats (all None if no generations occurred)
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    # leftover samples remaining in replay buffer after batch retrieval
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


@dataclass(frozen=True)
class _TaskRunner:
    task_name: str
    agent_loop: AgentLoopSpec
    produce_strategy: ProduceStrategy
    sampler: Sampler
    weight: float = 1.0
    order: int = 0


class _TaskSamplerView:
    def __init__(self, samplers: list[Sampler]):
        self._samplers = samplers

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self._samplers)


class AgentLoopManagerStatus(Enum):
    """AgentLoopManager 的全局状态.

    按下面的路径流转：
    - 初始状态是 NORMAL
    - NORMAL -> UPDATE_ABORT
      - trainer 开始做权重同步前触发
    - UPDATE_ABORT -> NORMAL
      - 权重同步完成后调用 continue_product()
    - NORMAL -> EXPIRED_BATCH
      - 当前 rollout model 已经过旧
    - EXPIRED_BATCH -> UPDATE_ABORT
      - trainer 检测到过期后，进入权重同步阶段
    - 任意状态 -> FINISH
      - 训练结束

    这里有一个重要区分：
    - AgentLoopManagerStatus 是“后台 producer 的全局运行状态”
    - ProduceBatchStatus 是“单次调度调用的局部结果”
    """

    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def _fill_produce_timing_stats(
    result: ProduceBatchResult, generate_times_s: list[float], pause_time_s: float = 0.0
) -> None:
    if not generate_times_s:
        if pause_time_s > 0:
            result.group_gen_pause_time_s = pause_time_s
        return
    sorted_times = sorted(generate_times_s)
    n = len(sorted_times)
    mean_s = sum(sorted_times) / n
    p50_s = sorted_times[n // 2]
    p99_s = sorted_times[int(n * 0.99)]
    ratio = p99_s / p50_s if p50_s > 0 else float("inf")
    result.group_gen_count = n
    result.group_gen_mean_s = mean_s
    result.group_gen_p50_s = p50_s
    result.group_gen_p99_s = p99_s
    result.group_gen_p99_p50_ratio = ratio
    result.group_gen_pause_time_s = pause_time_s


def _fill_group_timing_stats(
    result: ProduceBatchResult, rollout_states: list[list[RolloutState]], pause_time_s: float = 0.0
) -> None:
    generate_times: list[float] = []
    for group in rollout_states:
        if not group:
            continue
        group_time = getattr(group[0], "extra_fields", {}).get(GROUP_GENERATE_TIME_KEY)
        if group_time is not None:
            generate_times.append(group_time)

    _fill_produce_timing_stats(result, generate_times, pause_time_s=pause_time_s)


def _aggregate_status(statuses: list[ProduceBatchStatus]) -> ProduceBatchStatus:
    if any(status == ProduceBatchStatus.UPDATE_ABORT for status in statuses):
        return ProduceBatchStatus.UPDATE_ABORT
    if any(status == ProduceBatchStatus.EXPIRED_BATCH for status in statuses):
        return ProduceBatchStatus.EXPIRED_BATCH
    return ProduceBatchStatus.NORMAL


async def _produce_single_task_to_buffer(
    task_runner: _TaskRunner,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    train_step: int,
    model_step: int,
    update_event: asyncio.Event | None,
    progress: ProduceProgress,
    target_cumulative: int | None = None,
) -> ProduceBatchStatus:
    return await task_runner.produce_strategy.produce_batch(
        task_runner.agent_loop,
        task_runner.sampler,
        replay_buffer,
        batch_size,
        task_runner.task_name,
        train_step=train_step,
        model_step=model_step,
        update_event=update_event,
        progress=progress,
        target_cumulative=target_cumulative,
    )


class TaskSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    produce_strategy_config: ProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig


class AgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig] | TaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
        sync_weights_interval: int = 1,
    ) -> "AgentLoopManager":
        tasks = self.tasks if isinstance(self.tasks, list) else [self.tasks]
        if not tasks:
            raise ValueError("AgentLoopManagerConfig requires at least one task config.")

        seen_task_names: set[str] = set()
        task_runners: list[_TaskRunner] = []
        for order, task_cfg in enumerate(tasks):
            if task_cfg.task_name in seen_task_names:
                raise ValueError(f"Duplicate task_name found in AgentLoopManagerConfig: {task_cfg.task_name}")
            seen_task_names.add(task_cfg.task_name)

            agent_loop = task_cfg.agent_loop_config.build(
                rollout_controller=rollout_controller,
                judger=build_judger(task_cfg.judger_config) if task_cfg.judger_config is not None else None,
                logger=logger,
            )
            produce_strategy = task_cfg.produce_strategy_config.build(sync_weights_interval=sync_weights_interval)
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

        return AgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class AgentLoopManager:
    _TASK_CHECKPOINT_DIR = "tasks"
    _MANAGER_STATE_PATH = "agent_loop_manager_state.json"
    _STATUS_POLL_INTERVAL_S = 1.0

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        logger=None,
    ):
        if not task_runners:
            raise ValueError("AgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for AgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

        # 非共卡并发控制信号：consumer 在同步权重前置位，producer / strategy 应直接观察
        # event 状态并尽快停止继续发新 rollout；不要用额外布尔快照替代这个 event。
        self._update_event = asyncio.Event()

        self._finish_event = asyncio.Event()

        # 非共卡 producer 读取的 model_step：rollout 侧当前使用的是哪个 train_step 同步后的模型。
        # consumer 完成权重同步后通过 continue_produce 更新；已 schedule 的 pending task
        # 必须在 strategy 内绑定发起时的 model_step，不能在 task 完成时再读取最新值。
        self._model_step = 0

        # 非共卡 producer / consumer 共享的控制状态。produce_loop / get_batch 应直接读取
        # self._status，不要跨 await 缓存局部快照，避免错过同步、过期或结束状态变化。
        self._status = AgentLoopManagerStatus.NORMAL

        # pause_produce 写入、下一次 get batch 读取并清零的耗时指标。
        # 只用于消费侧日志/metrics；读写不构成生产正确性依赖。
        self._pause_time_s = 0.0

        # 非共卡 producer / consumer 共享的绝对累计进度。对象引用必须保持稳定；
        # consumer 原地更新字段，producer / strategy 需要字段值时直接读取 progress.xxx，
        # 不要把字段值复制成跨 await 使用的局部快照。
        self._produce_progress = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=1,
            consumed_samples={task.task_name: 0 for task in self.task_runners},
            target_samples={task.task_name: 0 for task in self.task_runners},
            target_upto_future_step=0,
        )

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        """Return the per-task batch sizes for the current train step.

        Subclasses may override this method to implement custom dynamic batch allocation policies. Returning 0 for a
        task effectively disables that task for the current produce_batch call.
        """
        if global_batch_size < 0:
            raise ValueError(f"global_batch_size must be non-negative, got {global_batch_size}")

        total_weight = sum(task.weight for task in self.task_runners)
        if total_weight <= 0:
            raise ValueError("Sum of task weights must be positive.")
        if global_batch_size == 0:
            return {task.task_name: 0 for task in self.task_runners}

        raw_allocations = [global_batch_size * task.weight / total_weight for task in self.task_runners]
        floor_allocations = [math.floor(raw) for raw in raw_allocations]
        remaining = global_batch_size - sum(floor_allocations)

        task_batch_sizes = {task.task_name: floor_allocations[idx] for idx, task in enumerate(self.task_runners)}
        if remaining <= 0:
            return task_batch_sizes

        ranked_tasks = sorted(
            enumerate(self.task_runners),
            key=lambda item: (
                -(raw_allocations[item[0]] - floor_allocations[item[0]]),
                item[1].order,
            ),
        )
        for idx, task in ranked_tasks[:remaining]:
            task_batch_sizes[task.task_name] += 1
        return task_batch_sizes

    def _validate_task_batch_sizes(self, task_batch_sizes: dict[str, int], global_batch_size: int) -> None:
        expected_task_names = {task.task_name for task in self.task_runners}
        actual_task_names = set(task_batch_sizes.keys())
        if actual_task_names != expected_task_names:
            missing_task_names = expected_task_names - actual_task_names
            extra_task_names = actual_task_names - expected_task_names
            raise ValueError(
                "Invalid task batch sizes returned by get_task_batch_sizes: "
                f"missing={sorted(missing_task_names)}, extra={sorted(extra_task_names)}"
            )

        negative_batch_sizes = {
            task_name: task_batch_size
            for task_name, task_batch_size in task_batch_sizes.items()
            if task_batch_size < 0
        }
        if negative_batch_sizes:
            raise ValueError(f"Task batch sizes must be non-negative, got {negative_batch_sizes}")

        total_batch_size = sum(task_batch_sizes.values())
        if total_batch_size != global_batch_size:
            raise ValueError(
                "Task batch sizes must sum to the requested global batch size, "
                f"got total={total_batch_size}, expected={global_batch_size}"
            )

    def _ensure_target_upto(self, batch_size: int, current_future_step: int) -> None:
        progress = self._produce_progress
        if current_future_step <= progress.target_upto_future_step:
            return

        for future_step in range(progress.target_upto_future_step + 1, current_future_step + 1):
            if len(self.task_runners) == 1:
                progress.target_samples[self.task_runners[0].task_name] += batch_size
            else:
                task_batch_sizes = self.get_task_batch_sizes(batch_size, future_step)
                self._validate_task_batch_sizes(task_batch_sizes, batch_size)
                for task_name, task_batch_size in task_batch_sizes.items():
                    progress.target_samples[task_name] += task_batch_size

        progress.target_upto_future_step = current_future_step

    def _any_task_model_expired(self, current_future_step: int) -> bool:
        expired_tasks = [
            task.task_name
            for task in self.task_runners
            if isinstance(task.produce_strategy, AsyncProduceStrategy)
            and task.produce_strategy.is_model_expired(current_future_step, self._model_step)
        ]
        if expired_tasks:
            self.logger.info(f"Expired future_step={current_future_step}, tasks={expired_tasks}")
            return True
        return False

    async def _refresh_for_all_tasks(self, train_step: int, statuses: list[Status]) -> None:
        for task in self.task_runners:
            stale_threshold = getattr(task.produce_strategy, "stale_threshold", None)
            if stale_threshold is None:
                # 同步生产没有跨权重版本的后台样本，只有异步 strategy 需要刷新并淘汰历史样本。
                continue
            await self.replay_buffer.refresh_staleness(
                task_name=task.task_name,
                current_train_step=train_step,
                stale_threshold=stale_threshold,
                statuses=statuses,
            )

    def _get_task_batch_sizes_for_step(self, batch_size: int, train_step: int) -> dict[str, int]:
        if len(self.task_runners) == 1:
            return {self.task_runners[0].task_name: batch_size}

        task_batch_sizes = self.get_task_batch_sizes(batch_size, train_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        return task_batch_sizes

    def _build_local_produce_progress(
        self,
        task_batch_sizes: dict[str, int],
        train_step: int,
    ) -> ProduceProgress:
        return ProduceProgress(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={task.task_name: 0 for task in self.task_runners},
            target_samples=dict(task_batch_sizes),
            target_upto_future_step=train_step,
        )

    @staticmethod
    def _aggregate_task_results(
        ordered_tasks: list[_TaskRunner], task_results: dict[str, ProduceBatchResult]
    ) -> ProduceBatchResult:
        rollout_states: list[list[RolloutState]] = []
        leftover_completed = 0
        leftover_aborted = 0
        leftover_expired = 0
        total_group_count = 0
        weighted_group_mean_sum = 0.0
        weighted_group_p50_sum = 0.0
        weighted_group_p99_sum = 0.0
        weighted_group_ratio_sum = 0.0
        total_pause_time_s = 0.0

        for task in ordered_tasks:
            result = task_results[task.task_name]
            rollout_states.extend(result.rollout_states)
            leftover_completed += result.leftover_completed
            leftover_aborted += result.leftover_aborted
            leftover_expired += result.leftover_expired
            if result.group_gen_count is not None and result.group_gen_mean_s is not None:
                total_group_count += result.group_gen_count
                weighted_group_mean_sum += result.group_gen_count * result.group_gen_mean_s
                weighted_group_p50_sum += result.group_gen_count * (result.group_gen_p50_s or 0.0)
                weighted_group_p99_sum += result.group_gen_count * (result.group_gen_p99_s or 0.0)
                weighted_group_ratio_sum += result.group_gen_count * (result.group_gen_p99_p50_ratio or 0.0)
                total_pause_time_s += result.group_gen_pause_time_s or 0.0

        aggregated = ProduceBatchResult(
            rollout_states=rollout_states,
            leftover_completed=leftover_completed,
            leftover_aborted=leftover_aborted,
            leftover_expired=leftover_expired,
            task_results={task.task_name: task_results[task.task_name] for task in ordered_tasks},
        )
        if total_group_count > 0:
            aggregated.group_gen_count = total_group_count
            aggregated.group_gen_mean_s = weighted_group_mean_sum / total_group_count
            aggregated.group_gen_p50_s = weighted_group_p50_sum / total_group_count
            aggregated.group_gen_p99_s = weighted_group_p99_sum / total_group_count
            aggregated.group_gen_p99_p50_ratio = weighted_group_ratio_sum / total_group_count
            aggregated.group_gen_pause_time_s = total_pause_time_s
        return aggregated

    async def _produce_batch_to_buffer(
        self,
        batch_size: int,
        progress: ProduceProgress,
        *,
        task_batch_sizes: dict[str, int] | None = None,
    ) -> ProduceBatchStatus:
        current_future_step = progress.producer_future_step
        model_step = self._model_step
        current_sizes = (
            self._get_task_batch_sizes_for_step(batch_size, current_future_step)
            if task_batch_sizes is None
            else task_batch_sizes
        )
        self._validate_task_batch_sizes(current_sizes, batch_size)

        if progress is self._produce_progress:
            # 只有后台生产循环使用全局 progress，需要在这里推进累计 target；
            # colocate 路径传入的是一次性本地 progress，不能污染全局计数。
            self._ensure_target_upto(batch_size, current_future_step)

        if self._any_task_model_expired(current_future_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        if len(self.task_runners) == 1:
            task = self.task_runners[0]
            self.logger.info(f"[AgentLoopManager][{self.name}] produce_to_buffer start batch={batch_size}")
            return await _produce_single_task_to_buffer(
                task_runner=task,
                replay_buffer=self.replay_buffer,
                batch_size=current_sizes[task.task_name],
                train_step=current_future_step,
                model_step=model_step,
                update_event=self._update_event,
                progress=progress,
            )

        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        statuses = await asyncio.gather(
            *[
                _produce_single_task_to_buffer(
                    task_runner=task,
                    replay_buffer=self.replay_buffer,
                    batch_size=current_sizes[task.task_name],
                    train_step=current_future_step,
                    model_step=model_step,
                    update_event=self._update_event,
                    progress=progress,
                )
                for task in active_tasks
            ]
        )
        return _aggregate_status(statuses)

    async def pause_produce(
        self,
        *,
        use_global_progress: bool,
        progress: ProduceProgress | None = None,
    ) -> float:
        # 这是 producer 的“显式刹车”接口。
        #
        # 设计动机：
        # - 旧 colocate 语义里，一次 produce_batch() 结束后就自然收尾；
        # - 非共卡后，producer 可能在后台持续运行，何时停下来必须交给 trainer 明确控制。
        #
        # 因此调用方必须显式说明是否使用全局 progress：
        # - use_global_progress=True：非共卡后台生产循环在权重同步点前暂停；
        # - use_global_progress=False：共卡同步 produce_batch 的本次调用收尾，使用本地 progress。
        # 返回值 `pause_time_s` 不是业务语义，而是日志/诊断信息，
        # 供训练侧在下一次消费 batch 时上报。
        # use_global_progress=False 模式会在下一次 produce_batch 入口通过 continue_produce 恢复；
        # use_global_progress=True 模式则由 trainer 在权重同步和评测完成后显式恢复。
        if use_global_progress:
            if progress is not None:
                raise ValueError("progress must not be provided when use_global_progress=True.")
            pause_progress = self._produce_progress
        else:
            if progress is None:
                raise ValueError("progress must be provided when use_global_progress=False.")
            pause_progress = progress

        # 合法参数确认后，统一拉起 manager 级暂停信号，阻止仍在运行的 produce_batch 继续调度新 rollout。
        self._update_event.set()
        self._status = AgentLoopManagerStatus.UPDATE_ABORT
        pause_time_s = 0.0
        for task in self.task_runners:
            pause_time_s += await task.produce_strategy.pause_produce(
                task.agent_loop,
                self.replay_buffer,
                task.task_name,
                progress=pause_progress,
            )
        self._pause_time_s = pause_time_s
        return pause_time_s

    async def _get_single_task_batch_from_buffer(
        self,
        task_runner: _TaskRunner,
        batch_size: int,
        train_step: int,
        consume_progress: ProduceProgress | None = None,
    ) -> ProduceBatchResult:
        result = ProduceBatchResult(rollout_states=[])
        batch_rollout_states: list[list[RolloutState]] = await self.replay_buffer.get(
            batch_size, task_runner.task_name, Status.COMPLETED
        )
        result.rollout_states = batch_rollout_states
        if consume_progress is not None:
            # get 已从 buffer 删除样本，立刻更新 consumed，避免 producer 短暂误判缺口。
            consume_progress.consumed_samples[task_runner.task_name] += len(batch_rollout_states)
        completed_sample_count, aborted_sample_count, expired_sample_count = await asyncio.gather(
            self.replay_buffer.count(task_name=task_runner.task_name, group_status=Status.COMPLETED),
            self.replay_buffer.count(task_name=task_runner.task_name, group_status=Status.ABORTED),
            self.replay_buffer.count(task_name=task_runner.task_name, group_status=Status.EXPIRED),
        )
        result.leftover_completed = completed_sample_count
        result.leftover_aborted = aborted_sample_count
        result.leftover_expired = expired_sample_count
        return result

    async def _get_batch_from_buffer(
        self,
        batch_size: int,
        train_step: int,
        consume_progress: ProduceProgress | None = None,
        task_batch_sizes: dict[str, int] | None = None,
    ) -> ProduceBatchResult:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0

        if len(self.task_runners) == 1:
            task = self.task_runners[0]
            result = await self._get_single_task_batch_from_buffer(
                task,
                batch_size,
                train_step,
                consume_progress=consume_progress,
            )
            _fill_group_timing_stats(result, result.rollout_states, pause_time_s=pause_time_s)
            return result

        if task_batch_sizes is None:
            task_batch_sizes = self._get_task_batch_sizes_for_step(batch_size, train_step)
        else:
            self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        active_tasks = [task for task in self.task_runners if task_batch_sizes[task.task_name] > 0]
        results = (
            await asyncio.gather(
                *[
                    self._get_single_task_batch_from_buffer(
                        task,
                        task_batch_sizes[task.task_name],
                        train_step,
                        consume_progress=consume_progress,
                    )
                    for task in active_tasks
                ]
            )
            if active_tasks
            else []
        )

        task_results = {task.task_name: result for task, result in zip(active_tasks, results)}
        for task in self.task_runners:
            if task.task_name not in task_results:
                task_results[task.task_name] = ProduceBatchResult(rollout_states=[])

        ordered_tasks = sorted(self.task_runners, key=lambda task: (task.task_name, task.order))
        aggregated = self._aggregate_task_results(ordered_tasks, task_results)
        aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
        _fill_group_timing_stats(aggregated, aggregated.rollout_states, pause_time_s=pause_time_s)
        return aggregated

    async def _is_batch_ready(self, batch_size: int, train_step: int) -> bool:
        if len(self.task_runners) == 1:
            task = self.task_runners[0]
            completed_count = await self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED)
            return completed_count >= batch_size

        task_batch_sizes = self._get_task_batch_sizes_for_step(batch_size, train_step)
        active_tasks = [task for task in self.task_runners if task_batch_sizes[task.task_name] > 0]
        if not active_tasks:
            return True

        completed_counts = await asyncio.gather(
            *[
                self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED)
                for task in active_tasks
            ]
        )
        return all(
            completed_count >= task_batch_sizes[task.task_name]
            for task, completed_count in zip(active_tasks, completed_counts)
        )

        # continue_produce 的语义是“producer 可以恢复工作了”。

    def continue_produce(self, model_step: int) -> None:
        #
        # 它和 pause_produce(use_global_progress=True) 是一对：
        # - pause_produce(...) 负责让 producer 停下来；
        # - continue_produce(...) 负责在同步/评测完成后解除暂停。
        #
        # 这里同步更新 `_model_step`，表示 rollout 侧接下来生成样本时，
        # 应把“当前正在使用的是哪一版权重”记录成这个版本号。
        self._status = AgentLoopManagerStatus.NORMAL
        self._model_step = model_step
        self._update_event.clear()

    async def _wait_for_status_exit(self, blocked_status: AgentLoopManagerStatus) -> None:
        while not self._finish_event.is_set() and self._status == blocked_status:
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        # `produce_batch()` 是保留给 colocate 路径的同步入口。
        #
        # 它虽然名字没变，但内部已经改成三段式：
        # 1. `_produce_batch_to_buffer()` 只负责生产，把结果写入 replay buffer
        # 2. `pause_produce()` 显式收尾 pending rollout
        # 3. `_get_batch_from_buffer()` 再把训练 batch 取出来
        #
        # 这也是为什么这里要求返回非空 batch：
        # - colocate 语义下，调用它就是为了拿一批可训练 completed groups
        # - 如果需要合法返回空 batch + 特殊状态，那应该走 disagg 的 `get_batch()`
        if batch_size <= 0:
            raise ValueError(f"produce_batch expects batch_size > 0, got {batch_size}")
        start = time.perf_counter()
        self.logger.info(f"[AgentLoopManager][{self.name}] produce_batch start batch={batch_size}")
        current_sizes = self._get_task_batch_sizes_for_step(batch_size, train_step)
        active_tasks = [task for task in self.task_runners if current_sizes[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        rollout_ctl = await get_agent_loop_rollout_ctl(active_tasks[0].agent_loop)
        await continue_generation(rollout_ctl)
        try:
            # 共卡路径不复用非共卡的 paused producer 状态机。
            # 即使 manager 是从 resume() 恢复出来、当前仍处在 UPDATE_ABORT，
            # produce_batch() 也应视作一次独立的同步生产过程，从干净状态开始。
            #
            # 共卡路径下，produce_batch() 对应 rollout worker 当前持有的权重版本。
            self.continue_produce(model_step=model_step)
            # 共卡 produce_batch 也是消费入口；生产前先刷新 buffer 中已有 completed / aborted。
            await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])
            local_progress = self._build_local_produce_progress(current_sizes, train_step)
            status = await self._produce_batch_to_buffer(
                batch_size=batch_size,
                progress=local_progress,
                task_batch_sizes=current_sizes,
            )
            await self.pause_produce(
                use_global_progress=False,
                progress=local_progress,
            )
            result = await self._get_batch_from_buffer(
                batch_size=batch_size,
                train_step=train_step,
                consume_progress=local_progress,
                task_batch_sizes=current_sizes,
            )
            result.status = status
            assert result.rollout_states, (
                "AgentLoopManager.produce_batch() must return non-empty rollout_states for colocated training. "
                "Use get_batch() for disaggregated empty/expired reads."
            )
        finally:
            await pause_generation(rollout_ctl)

        self.logger.info(
            f"[AgentLoopManager][{self.name}] produce_batch done "
            f"elapsed={time.perf_counter() - start:.3f}, completed_groups={len(result.rollout_states)}"
        )
        return result

    async def produce_loop(self, batch_size: int) -> None:
        # `produce_loop()` 是非共卡新增的后台生产循环。
        # batch_size 表示每个 future train_step 的目标生产规模；producer 需要它来推进累计目标，
        # 所以这个参数保留在后台生产入口，而不是从 get_batch() 的消费请求里推断。
        #
        # 和 colocate 最大的区别是：
        # - 它不直接把 batch 返回给 trainer
        # - 它只是持续把样本“喂”进 replay buffer
        # - trainer 前台通过 `get_batch()` 异步消费
        #
        # 因此这里的核心职责不是“凑出一批训练数据”，而是根据 manager 的全局状态机
        # 决定什么时候继续生产、什么时候暂停等待、什么时候彻底退出。
        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.FINISH:
                break
            if self._status == AgentLoopManagerStatus.UPDATE_ABORT:
                # trainer 已经发出了“准备同步权重”的信号。
                # producer 在这里阻塞等待 continue_produce()，而不是自己擅自恢复。
                await self._wait_for_status_exit(AgentLoopManagerStatus.UPDATE_ABORT)
                continue
            if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                # 当前 rollout 权重已经过旧。
                # 这里继续等待 trainer 完成同步，再通过 continue_produce() 恢复。
                await self._wait_for_status_exit(AgentLoopManagerStatus.EXPIRED_BATCH)
                continue

            rollout_ctl = await get_agent_loop_rollout_ctl(self.task_runners[0].agent_loop)
            await continue_generation(rollout_ctl)
            produce_status = await self._produce_batch_to_buffer(
                batch_size=batch_size,
                progress=self._produce_progress,
            )

            if produce_status == ProduceBatchStatus.EXPIRED_BATCH:
                # 注意：
                # - EXPIRED_BATCH 是 producer 在生产过程中自己检测出来的“立即停下”信号
                # - UPDATE_ABORT 则是 trainer 在同步前通过 pause_produce() 主动设置的
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
            elif produce_status == ProduceBatchStatus.NORMAL:
                # 只有正常完成一轮生产时，producer 自己维护的 train_step 才前进一步。
                self._produce_progress.producer_future_step += 1

            # 主动让出事件循环，避免 fake strategy / 极快路径在测试里造成忙等空转。
            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        # `get_batch()` 是非共卡路径给 trainer 的消费接口。
        #
        # 设计上它和 `produce_batch()` 明确分工：
        # - `produce_batch()`：colocate，一次调用内完成“生产+收尾+取数”
        # - `get_batch()`：disagg，等待 replay buffer 准备好当前训练步所需 batch 后再取数
        #
        # 因而这里允许返回空 batch 的唯一合法场景仍然只有：
        # - 当 manager 已进入 EXPIRED_BATCH，返回空 batch + 状态信号
        # - trainer 看到后应跳过训练，优先去做权重同步
        progress = self._produce_progress
        progress.next_consumer_step = train_step
        await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])

        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                return ProduceBatchResult(
                    rollout_states=[],
                    status=ProduceBatchStatus.EXPIRED_BATCH,
                )
            # TODO: call self.get_task_batch_sizes before while instead of below two functions
            if await self._is_batch_ready(batch_size=batch_size, train_step=train_step):
                result = await self._get_batch_from_buffer(
                    batch_size=batch_size,
                    train_step=train_step,
                    consume_progress=progress,
                )
                if result.rollout_states:
                    progress.next_consumer_step = train_step + 1
                    await self._refresh_for_all_tasks(train_step + 1, [Status.COMPLETED, Status.ABORTED])
                    return result
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

        return ProduceBatchResult(rollout_states=[])

    def _task_checkpoint_path(self, checkpoint_path: Path | str, task_name: str) -> Path:
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path / self._TASK_CHECKPOINT_DIR / task_name

    def _manager_state_path(self, checkpoint_path: Path | str) -> Path:
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path / self._MANAGER_STATE_PATH

    def _get_pending_task_counts(self) -> dict[str, int]:
        pending_task_counts: dict[str, int] = {}
        for task in self.task_runners:
            pending_tasks = getattr(task.produce_strategy, "_pending_tasks", None)
            if pending_tasks:
                pending_task_counts[task.task_name] = len(pending_tasks)
        return pending_task_counts

    def save(self, checkpoint_path: Path | str, model_step: int) -> None:
        """Save all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = self._get_pending_task_counts()
        if pending_task_counts:
            raise RuntimeError(
                "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                f"{pending_task_counts}. Call pause_produce() first."
            )
        # 保存前显式记录当前 checkpoint 对应的模型步数，resume 时直接恢复这一份状态。
        self._model_step = model_step
        for task in self.task_runners:
            task_checkpoint_path = self._task_checkpoint_path(checkpoint_path, task.task_name)
            task_checkpoint_path.mkdir(parents=True, exist_ok=True)
            task.sampler.save(task_checkpoint_path)
        asyncio_run(self.replay_buffer.save(checkpoint_path))
        manager_state_path = self._manager_state_path(checkpoint_path)
        progress = self._produce_progress
        with manager_state_path.open("w") as f:
            json.dump(
                {
                    "status": self._status.name,
                    "model_step": self._model_step,
                    "next_consumer_step": progress.next_consumer_step,
                    "producer_future_step": progress.producer_future_step,
                    "consumed_samples": progress.consumed_samples,
                    "target_samples": progress.target_samples,
                    "target_upto_future_step": progress.target_upto_future_step,
                },
                f,
            )

    def resume(self, checkpoint_path: Path | str) -> int:
        """Resume all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        for task in self.task_runners:
            task.sampler.resume(self._task_checkpoint_path(checkpoint_path, task.task_name))
        asyncio_run(self.replay_buffer.resume(checkpoint_path))

        manager_state_path = self._manager_state_path(checkpoint_path)
        with manager_state_path.open("r") as f:
            manager_state = json.load(f)
        saved_model_step = manager_state["model_step"]
        progress = self._produce_progress
        progress.next_consumer_step = manager_state["next_consumer_step"]
        progress.producer_future_step = manager_state["producer_future_step"]
        progress.target_upto_future_step = manager_state["target_upto_future_step"]

        # dict 原地更新，避免 strategy 持有旧引用。
        progress.consumed_samples.clear()
        progress.consumed_samples.update(manager_state["consumed_samples"])
        progress.target_samples.clear()
        progress.target_samples.update(manager_state["target_samples"])

        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()
        self._update_event.set()
        self._status = AgentLoopManagerStatus.UPDATE_ABORT
        self._pause_time_s = 0.0
        self._model_step = saved_model_step
        return saved_model_step
