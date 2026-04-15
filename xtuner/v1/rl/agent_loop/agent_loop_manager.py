"""Unified agent loop manager for colocated and fullasync rollout production."""

import asyncio
import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController, continue_generation
from xtuner.v1.rl.utils import create_task

from .manager_base import (
    BaseAgentLoopManager,
    ProduceBatchResult,
    TaskSpecConfig,
    _TaskRunner,
    _get_single_task_completed_batch,
    _produce_single_task_batch,
    build_task_runners,
)
from .producer import ProducerTimings


@dataclass
class _FullAsyncIntervalSession:
    start_step: int
    end_step: int
    batch_size: int
    producer_tasks: dict[str, asyncio.Task]


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
        return AgentLoopManager(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class AgentLoopManager(BaseAgentLoopManager):
    def __init__(self, task_runners: list[_TaskRunner], replay_buffer: ReplayBuffer, logger=None):
        super().__init__(task_runners=task_runners, replay_buffer=replay_buffer, logger=logger)
        self._sync_interval_steps = 1
        self._sync_interval_total_train_steps: int | None = None
        self._fullasync_session: _FullAsyncIntervalSession | None = None

    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        start = time.perf_counter()
        self.logger.info(f"[AgentLoopManager][{self.name}] produce_batch start batch={batch_size}")
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")

        task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        active_tasks = [task for task in self.task_runners if task_batch_sizes[task.task_name] > 0]

        results: list[ProduceBatchResult] = []
        if active_tasks:
            rollout_ctl = self._get_shared_rollout_ctl(active_tasks)
            await continue_generation(rollout_ctl)
            try:
                results = await self._gather_fail_fast(
                    *[
                        _produce_single_task_batch(
                            task_runner=task,
                            replay_buffer=self.replay_buffer,
                            batch_size=task_batch_sizes[task.task_name],
                            rollout_step=rollout_step,
                            logger=self.logger,
                            manager_name="AgentLoopManager",
                        )
                        for task in active_tasks
                    ]
                )
            finally:
                from xtuner.v1.rl.rollout import pause_generation

                await pause_generation(rollout_ctl)

        task_results = {task.task_name: result for task, result in zip(active_tasks, results)}
        for task in self.task_runners:
            task_results.setdefault(task.task_name, ProduceBatchResult(rollout_states=[]))

        ordered_tasks = list(self.task_runners)
        aggregated = self._aggregate_task_results(ordered_tasks, task_results)
        aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
        aggregated.required_task_batch_sizes = dict(aggregated.task_batch_sizes)

        self.logger.info(
            f"[AgentLoopManager][{self.name}] produce_batch done elapsed={time.perf_counter() - start:.3f}, "
            f"completed_groups={len(aggregated.rollout_states)}"
        )
        return aggregated

    def set_sync_interval_context(
        self,
        *,
        trigger_parameter_sync_step: int,
        total_train_steps: int | None = None,
    ) -> None:
        if trigger_parameter_sync_step <= 0:
            raise ValueError(
                "trigger_parameter_sync_step must be positive, "
                f"got {trigger_parameter_sync_step}"
            )
        if total_train_steps is not None and total_train_steps <= 0:
            raise ValueError(f"total_train_steps must be positive, got {total_train_steps}")
        self._sync_interval_steps = trigger_parameter_sync_step
        self._sync_interval_total_train_steps = total_train_steps

    def _get_sync_interval_end_step(self, rollout_step: int) -> int:
        interval_end_step = rollout_step + self._sync_interval_steps - 1
        if self._sync_interval_total_train_steps is not None:
            interval_end_step = min(interval_end_step, self._sync_interval_total_train_steps)
        return interval_end_step

    def _get_step_task_batch_sizes(
        self,
        batch_size: int,
        rollout_step: int,
    ) -> dict[str, int]:
        task_batch_sizes = self.get_task_batch_sizes(batch_size, rollout_step)
        self._validate_task_batch_sizes(task_batch_sizes, batch_size)
        return task_batch_sizes

    def _get_interval_task_batch_sizes(
        self,
        batch_size: int,
        start_step: int,
        end_step: int,
    ) -> dict[str, int]:
        interval_task_batch_sizes = {task.task_name: 0 for task in self.task_runners}
        for rollout_step in range(start_step, end_step + 1):
            task_batch_sizes = self._get_step_task_batch_sizes(batch_size, rollout_step)
            for task_name, task_batch_size in task_batch_sizes.items():
                interval_task_batch_sizes[task_name] += task_batch_size
        return interval_task_batch_sizes

    def _raise_if_fullasync_producers_failed(self) -> None:
        session = self._fullasync_session
        if session is None:
            return
        for task_name, producer_task in session.producer_tasks.items():
            if producer_task.done():
                exc = producer_task.exception()
                if exc is not None:
                    raise RuntimeError(f"fullasync producer task failed for {task_name}") from exc

    async def _wait_until_completed_counts_ready(self, task_batch_sizes: dict[str, int]) -> None:
        from xtuner.v1.data_proto import Status

        while True:
            self._raise_if_fullasync_producers_failed()
            counts = await asyncio.gather(
                *[
                    self.replay_buffer.count(task_name=task.task_name, group_status=Status.COMPLETED)
                    for task in self.task_runners
                ]
            )
            if all(count >= task_batch_sizes[task.task_name] for task, count in zip(self.task_runners, counts)):
                return
            await asyncio.sleep(0.1)

    async def _collect_completed_batch(
        self,
        task_batch_sizes: dict[str, int],
        *,
        manager_name: str,
        task_stats: dict[str, ProducerTimings] | None = None,
    ) -> ProduceBatchResult:
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

    async def fullasync_produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        if self._fullasync_session is not None:
            if self._fullasync_session.start_step == rollout_step:
                return ProduceBatchResult(rollout_states=[])
            raise RuntimeError(
                "A fullasync interval session is already active. Call get_completed_batch() until the interval "
                "finishes before starting the next interval."
            )

        interval_end_step = self._get_sync_interval_end_step(rollout_step)
        interval_task_batch_sizes = self._get_interval_task_batch_sizes(batch_size, rollout_step, interval_end_step)
        active_tasks = [task for task in self.task_runners if interval_task_batch_sizes[task.task_name] > 0]
        producer_tasks: dict[str, asyncio.Task] = {}

        if active_tasks:
            rollout_ctl = self._get_shared_rollout_ctl(active_tasks)
            await continue_generation(rollout_ctl)
            for task in active_tasks:
                producer_tasks[task.task_name] = create_task(
                    task.produce_strategy.produce_batch(
                        task.agent_loop,
                        task.sampler,
                        self.replay_buffer,
                        interval_task_batch_sizes[task.task_name],
                        task.task_name,
                        rollout_step,
                    )
                )

        self._fullasync_session = _FullAsyncIntervalSession(
            start_step=rollout_step,
            end_step=interval_end_step,
            batch_size=batch_size,
            producer_tasks=producer_tasks,
        )
        self.logger.info(
            f"[AgentLoopManager][{self.name}] fullasync interval started start_step={rollout_step}, "
            f"end_step={interval_end_step}, interval_task_batch_sizes={interval_task_batch_sizes}"
        )
        return ProduceBatchResult(rollout_states=[])

    async def get_completed_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        if self._fullasync_session is None:
            raise RuntimeError("No active fullasync interval session. Call fullasync_produce_batch() first.")
        session = self._fullasync_session
        if not (session.start_step <= rollout_step <= session.end_step):
            raise RuntimeError(
                f"rollout_step={rollout_step} does not belong to active fullasync interval "
                f"[{session.start_step}, {session.end_step}]"
            )

        if batch_size != session.batch_size:
            raise RuntimeError(
                f"batch_size={batch_size} does not match active fullasync interval batch_size={session.batch_size}"
            )

        task_batch_sizes = self._get_step_task_batch_sizes(session.batch_size, rollout_step)
        await self._wait_until_completed_counts_ready(task_batch_sizes)

        task_stats: dict[str, ProducerTimings] | None = None
        is_interval_end = rollout_step == session.end_step
        if is_interval_end:
            task_stats = {}
            for task_name, producer_task in session.producer_tasks.items():
                task_stats[task_name] = await producer_task

        result = await self._collect_completed_batch(
            task_batch_sizes,
            manager_name="AgentLoopManager",
            task_stats=task_stats,
        )

        if is_interval_end:
            self.logger.info(
                f"[AgentLoopManager][{self.name}] fullasync interval finished start_step={session.start_step}, "
                f"end_step={session.end_step}"
            )
            self._fullasync_session = None

        return result
