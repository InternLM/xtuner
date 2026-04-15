"""Unified multi-task agent loop manager."""

import asyncio
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation
from xtuner.v1.rl.utils import asyncio_run, create_task
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoopConfig, AgentLoopSpec
from .producer import ProducerTimings, ProduceStrategy, ProduceStrategyConfig, SyncProduceStrategyConfig
from .sampler import Sampler, SamplerConfig


@dataclass
class ProduceBatchResult:
    """一次 produce 调用的统一返回结构。"""

    rollout_states: list[list[RolloutState]]
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    group_gen_times_s: list[float] | None = None
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    task_batch_sizes: dict[str, int] | None = None
    required_task_batch_sizes: dict[str, int] | None = None
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


def _fill_produce_timing_stats(result: ProduceBatchResult, stats: ProducerTimings) -> None:
    if not stats.generate_times_s:
        return
    sorted_times = sorted(stats.generate_times_s)
    n = len(sorted_times)
    mean_s = sum(sorted_times) / n
    p50_s = median(sorted_times)
    p99_s = sorted_times[min(math.ceil(0.99 * n) - 1, n - 1)]
    ratio = p99_s / p50_s if p50_s > 0 else float("inf")
    result.group_gen_count = n
    result.group_gen_mean_s = mean_s
    result.group_gen_p50_s = p50_s
    result.group_gen_p99_s = p99_s
    result.group_gen_p99_p50_ratio = ratio
    result.group_gen_pause_time_s = stats.pause_time_s
    result.group_gen_times_s = list(sorted_times)


async def _get_single_task_completed_batch(
    task_runner: _TaskRunner,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    logger,
    manager_name: str,
    *,
    stats: ProducerTimings | None = None,
) -> ProduceBatchResult:
    start = time.perf_counter()
    logger.info(f"[{manager_name}][{task_runner.task_name}] get_completed_batch start batch={batch_size}")

    result = ProduceBatchResult(rollout_states=[])
    if stats is not None:
        _fill_produce_timing_stats(result, stats)

    batch_rollout_states: list[list[RolloutState]] = await replay_buffer.get(
        batch_size, task_runner.task_name, Status.COMPLETED
    )
    logger.info(
        f"[{manager_name}][{task_runner.task_name}] get_completed_batch done completed_groups={len(batch_rollout_states)} "
        f"elapsed={time.perf_counter() - start:.3f}"
    )
    result.rollout_states = batch_rollout_states
    result.task_batch_sizes = {task_runner.task_name: batch_size}
    completed_sample_count, aborted_sample_count, expired_sample_count = await asyncio.gather(
        replay_buffer.count(task_name=task_runner.task_name, group_status=Status.COMPLETED),
        replay_buffer.count(task_name=task_runner.task_name, group_status=Status.ABORTED),
        replay_buffer.count(task_name=task_runner.task_name, group_status=Status.EXPIRED),
    )
    result.leftover_completed = completed_sample_count
    result.leftover_aborted = aborted_sample_count
    result.leftover_expired = expired_sample_count
    return result


class TaskSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    produce_strategy_config: ProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig


def build_task_runners(
    tasks: list[TaskSpecConfig] | TaskSpecConfig,
    *,
    rollout_controller: RolloutController,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    replay_buffer: ReplayBuffer,
    logger=None,
) -> list[_TaskRunner]:
    tasks = tasks if isinstance(tasks, list) else [tasks]
    if not tasks:
        raise ValueError("At least one task config is required.")

    seen_task_names: set[str] = set()
    task_runners: list[_TaskRunner] = []
    for order, task_cfg in enumerate(tasks):
        if task_cfg.task_name in seen_task_names:
            raise ValueError(f"Duplicate task_name found in task configs: {task_cfg.task_name}")
        seen_task_names.add(task_cfg.task_name)

        agent_loop = task_cfg.agent_loop_config.build(
            rollout_controller=rollout_controller,
            judger=build_judger(task_cfg.judger_config) if task_cfg.judger_config is not None else None,
            logger=logger,
        )
        produce_strategy = task_cfg.produce_strategy_config.build()
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
    return task_runners


class AgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig] | TaskSpecConfig

    def build(
        self,
        rollout_controller: RolloutController,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
    ) -> "AgentLoopManager":
        task_runners = build_task_runners(
            self.tasks,
            rollout_controller=rollout_controller,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
        )
        return AgentLoopManager(task_runners=task_runners, replay_buffer=replay_buffer, logger=logger)


class AgentLoopManager:
    _TASK_CHECKPOINT_DIR = "tasks"

    def __init__(self, task_runners: list[_TaskRunner], replay_buffer: ReplayBuffer, logger=None):
        if not task_runners:
            raise ValueError("Agent loop manager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for the agent loop manager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        self.logger = get_logger() if logger is None else logger

    def get_task_batch_sizes(self, train_batch_size: int, rollout_step: int) -> dict[str, int]:
        if train_batch_size < 0:
            raise ValueError(f"train_batch_size must be non-negative, got {train_batch_size}")

        total_weight = sum(task.weight for task in self.task_runners)
        if total_weight <= 0:
            raise ValueError("Sum of task weights must be positive.")
        if train_batch_size == 0:
            return {task.task_name: 0 for task in self.task_runners}

        raw_allocations = [train_batch_size * task.weight / total_weight for task in self.task_runners]
        floor_allocations = [math.floor(raw) for raw in raw_allocations]
        remaining = train_batch_size - sum(floor_allocations)

        task_batch_sizes = {task.task_name: floor_allocations[idx] for idx, task in enumerate(self.task_runners)}
        if remaining <= 0:
            return task_batch_sizes

        ranked_tasks = sorted(
            enumerate(self.task_runners),
            key=lambda item: (-(raw_allocations[item[0]] - floor_allocations[item[0]]), item[1].order),
        )
        for idx, task in ranked_tasks[:remaining]:
            task_batch_sizes[task.task_name] += 1
        return task_batch_sizes

    def get_step_task_batch_sizes(self, batch_size: int, rollout_step: int) -> dict[str, int]:
        task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        return task_batch_sizes

    def get_interval_task_batch_sizes(self, batch_size: int, start_step: int, end_step: int) -> dict[str, int]:
        interval_task_batch_sizes = {task.task_name: 0 for task in self.task_runners}
        for rollout_step in range(start_step, end_step + 1):
            task_batch_sizes = self.get_step_task_batch_sizes(batch_size, rollout_step)
            for task_name, task_batch_size in task_batch_sizes.items():
                interval_task_batch_sizes[task_name] += task_batch_size
        return interval_task_batch_sizes

    async def start_producer_tasks(
        self,
        task_batch_sizes: dict[str, int],
        rollout_step: int,
    ) -> dict[str, asyncio.Task]:
        self._validate_task_counts(task_batch_sizes)
        active_tasks = [task for task in self.task_runners if task_batch_sizes[task.task_name] > 0]
        producer_tasks: dict[str, asyncio.Task] = {}

        if not active_tasks:
            return producer_tasks

        rollout_ctl = self._get_shared_rollout_ctl(active_tasks)
        await continue_generation(rollout_ctl)
        for task in active_tasks:
            producer_tasks[task.task_name] = create_task(
                task.produce_strategy.produce_batch(
                    task.agent_loop,
                    task.sampler,
                    self.replay_buffer,
                    task_batch_sizes[task.task_name],
                    task.task_name,
                    rollout_step,
                )
            )
        return producer_tasks

    async def consume_completed_batch(
        self,
        task_batch_sizes: dict[str, int],
        *,
        producer_tasks: dict[str, asyncio.Task] | None = None,
        await_producer_tasks: bool = False,
        manager_name: str = "AgentLoopManager",
    ) -> ProduceBatchResult:
        self._validate_task_counts(task_batch_sizes)
        await self._wait_until_completed_counts_ready(task_batch_sizes, producer_tasks=producer_tasks)

        task_stats: dict[str, ProducerTimings] | None = None
        if await_producer_tasks and producer_tasks:
            task_stats = {}
            for task_name, producer_task in producer_tasks.items():
                task_stats[task_name] = await producer_task

        task_results: dict[str, ProduceBatchResult] = {}
        for task in self.task_runners:
            task_batch_size = task_batch_sizes[task.task_name]
            task_results[task.task_name] = await _get_single_task_completed_batch(
                task_runner=task,
                replay_buffer=self.replay_buffer,
                batch_size=task_batch_size,
                logger=self.logger,
                manager_name=manager_name,
                stats=None if task_stats is None else task_stats.get(task.task_name),
            )
        aggregated = self._aggregate_task_results(list(self.task_runners), task_results)
        aggregated.task_batch_sizes = dict(task_batch_sizes)
        aggregated.required_task_batch_sizes = dict(task_batch_sizes)
        return aggregated

    def _validate_task_batch_sizes(self, task_batch_sizes: dict[str, int], train_batch_size: int) -> None:
        self._validate_task_counts(task_batch_sizes)
        total_batch_size = sum(task_batch_sizes.values())
        if total_batch_size != train_batch_size:
            raise ValueError(
                "Task batch sizes must sum to the requested train batch size, "
                f"got total={total_batch_size}, expected={train_batch_size}"
            )

    def _validate_task_counts(self, task_counts: dict[str, int]) -> None:
        expected_task_names = {task.task_name for task in self.task_runners}
        actual_task_names = set(task_counts.keys())
        if actual_task_names != expected_task_names:
            missing_task_names = expected_task_names - actual_task_names
            extra_task_names = actual_task_names - expected_task_names
            raise ValueError(
                f"Invalid task counts: missing={sorted(missing_task_names)}, extra={sorted(extra_task_names)}"
            )

        negative_counts = {task_name: task_count for task_name, task_count in task_counts.items() if task_count < 0}
        if negative_counts:
            raise ValueError(f"Task counts must be non-negative, got {negative_counts}")

    def _get_shared_rollout_ctl(self, active_tasks: list[_TaskRunner]):
        if not active_tasks:
            return None
        rollout_ctl = active_tasks[0].agent_loop.rollout_ctl
        for task in active_tasks[1:]:
            if task.agent_loop.rollout_ctl is not rollout_ctl:
                raise RuntimeError(
                    "The agent loop manager currently requires all active tasks in one produce call to share the same "
                    "rollout_ctl because continue_generation/pause_generation are issued once per produce call. "
                    f"Found a different rollout_ctl on task {task.task_name}."
                )
        return rollout_ctl

    def _raise_if_producer_tasks_failed(self, producer_tasks: dict[str, asyncio.Task] | None = None) -> None:
        if producer_tasks is None:
            return
        for task_name, producer_task in producer_tasks.items():
            if producer_task.done():
                exc = producer_task.exception()
                if exc is not None:
                    raise RuntimeError(f"producer task failed for {task_name}") from exc

    async def _wait_until_completed_counts_ready(
        self,
        task_batch_sizes: dict[str, int],
        producer_tasks: dict[str, asyncio.Task] | None = None,
    ) -> None:
        while True:
            self._raise_if_producer_tasks_failed(producer_tasks)
            counts = await asyncio.gather(
                *[
                    self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED)
                    for task in self.task_runners
                ]
            )
            if all(count >= task_batch_sizes[task.task_name] for task, count in zip(self.task_runners, counts)):
                return
            await asyncio.sleep(0.1)

    @staticmethod
    def _copy_task_result(result: ProduceBatchResult) -> ProduceBatchResult:
        return replace(result, task_results=None)

    @staticmethod
    def _aggregate_task_results(
        ordered_tasks: list[_TaskRunner],
        task_results: dict[str, ProduceBatchResult],
    ) -> ProduceBatchResult:
        rollout_states: list[list[RolloutState]] = []
        group_gen_times_s: list[float] = []
        leftover_completed = 0
        leftover_aborted = 0
        leftover_expired = 0
        total_pause_time_s = 0.0

        for task in ordered_tasks:
            result = task_results[task.task_name]
            rollout_states.extend(result.rollout_states)
            leftover_completed += result.leftover_completed
            leftover_aborted += result.leftover_aborted
            leftover_expired += result.leftover_expired
            if result.group_gen_times_s:
                group_gen_times_s.extend(result.group_gen_times_s)
                total_pause_time_s += result.group_gen_pause_time_s or 0.0

        aggregated = ProduceBatchResult(
            rollout_states=rollout_states,
            leftover_completed=leftover_completed,
            leftover_aborted=leftover_aborted,
            leftover_expired=leftover_expired,
            task_results={task.task_name: task_results[task.task_name] for task in ordered_tasks},
        )
        if group_gen_times_s:
            sorted_times = sorted(group_gen_times_s)
            total_group_count = len(sorted_times)
            p50_s = median(sorted_times)
            p99_s = sorted_times[min(math.ceil(0.99 * total_group_count) - 1, total_group_count - 1)]
            aggregated.group_gen_times_s = sorted_times
            aggregated.group_gen_count = total_group_count
            aggregated.group_gen_mean_s = sum(sorted_times) / total_group_count
            aggregated.group_gen_p50_s = p50_s
            aggregated.group_gen_p99_s = p99_s
            aggregated.group_gen_p99_p50_ratio = p99_s / p50_s if p50_s > 0 else float("inf")
            aggregated.group_gen_pause_time_s = total_pause_time_s
        return aggregated

    def _task_checkpoint_path(self, checkpoint_path: Path | str, task_name: str) -> Path:
        checkpoint_path = Path(checkpoint_path)
        return checkpoint_path / self._TASK_CHECKPOINT_DIR / task_name

    def save(self, checkpoint_path: Path | str) -> None:
        for task in self.task_runners:
            task_checkpoint_path = self._task_checkpoint_path(checkpoint_path, task.task_name)
            task_checkpoint_path.mkdir(parents=True, exist_ok=True)
            task.sampler.save(task_checkpoint_path)
        asyncio_run(self.replay_buffer.save(checkpoint_path))

    def resume(self, checkpoint_path: Path | str) -> None:
        for task in self.task_runners:
            task.sampler.resume(self._task_checkpoint_path(checkpoint_path, task.task_name))
        asyncio_run(self.replay_buffer.resume(checkpoint_path))
