import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status, refresh_seq_staleness, update_expired_status
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import pause_generation
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoop
from .sampler import Sampler


@dataclass
class ProducerTimings:
    """记录一轮 batch 生成过程中每个 group 的生成耗时统计信息。

    Args:
        generate_times_s (list[float]): 每个 group 的生成耗时（秒），长度等于本轮生成 group 的数量。
        pause_time_s (float): 结束时等待所有 pending 任务收尾的总耗时（秒）。
    """

    generate_times_s: list[float] = field(default_factory=list)
    pause_time_s: float = 0.0


logger = get_logger()


async def _timed_generate_group(
    agent_loop: AgentLoop, rollout_state: list[RolloutState], **kwargs
) -> tuple[list[RolloutState], float]:
    start = time.perf_counter()
    result = await agent_loop.generate_group(rollout_state, **kwargs)
    return result, time.perf_counter() - start


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, target_count: int, **kwargs) -> bool:
    return completed_count < target_count


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, target_count: int, **kwargs) -> bool: ...


class ProduceStrategyConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn
    max_finished_groups_multiplier: float = 100.0

    def model_post_init(self, __context) -> None:
        if self.max_finished_groups_multiplier <= 0:
            raise ValueError(
                "max_finished_groups_multiplier must be positive, "
                f"got {self.max_finished_groups_multiplier}"
            )

    @abstractmethod
    def build(self) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    def build(self) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
            max_finished_groups_multiplier=self.max_finished_groups_multiplier,
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    produce_batch_over_sample_threshold: float = 0.0
    produce_batch_enable_partial_rollout: bool = False
    tail_batch_stale_threshold: int = 0
    tail_batch_trigger_size: int = 0

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.produce_batch_over_sample_threshold < 0:
            raise ValueError(
                "produce_batch_over_sample_threshold must be non-negative, "
                f"got {self.produce_batch_over_sample_threshold}"
            )
        if self.tail_batch_stale_threshold < 0:
            raise ValueError(
                f"tail_batch_stale_threshold must be non-negative, got {self.tail_batch_stale_threshold}"
            )
        if self.tail_batch_trigger_size < 0:
            raise ValueError(f"tail_batch_trigger_size must be non-negative, got {self.tail_batch_trigger_size}")

    def build(self) -> "AsyncProduceStrategy":
        return AsyncProduceStrategy(
            produce_batch_over_sample_threshold=self.produce_batch_over_sample_threshold,
            produce_batch_enable_partial_rollout=self.produce_batch_enable_partial_rollout,
            tail_batch_stale_threshold=self.tail_batch_stale_threshold,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
            max_finished_groups_multiplier=self.max_finished_groups_multiplier,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
        max_finished_groups_multiplier: float,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn
        self.max_finished_groups_multiplier = max_finished_groups_multiplier

    def _get_max_finished_groups(self, target_count: int) -> int:
        return max(math.ceil(max(target_count, 1) * self.max_finished_groups_multiplier), target_count)

    def _raise_if_exceeded_finished_group_budget(
        self,
        *,
        task_name: str,
        target_count: int,
        completed_count: int,
        finished_group_count: int,
    ) -> None:
        max_finished_groups = self._get_max_finished_groups(target_count)
        if finished_group_count > max_finished_groups:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Task {task_name} exceeded finished-group budget without reaching "
                f"target_count. completed={completed_count}, target={target_count}, finished={finished_group_count}, "
                f"max_finished={max_finished_groups}. This usually means samples keep failing validation."
            )

    async def _launch_group(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        task_name: str,
        rollout_step: int,
        *,
        source_group_status: Status | None = None,
        enable_partial_rollout: bool = False,
    ) -> asyncio.Task:
        rollout_state = await sampler.sample(task_name=task_name, group_status=source_group_status)
        return create_task(
            _timed_generate_group(
                agent_loop,
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
                rollout_step=rollout_step,
            )
        )

    async def _cleanup_pending_tasks(
        self,
        pending_tasks: set,
        agent_loop: AgentLoop,
        replay_buffer: ReplayBuffer,
        task_name: str,
        tail_batch_stale_threshold: int = 0,
    ) -> tuple[list[float], float]:
        pause_start = time.perf_counter()
        rollout_ctl = agent_loop.rollout_ctl
        generate_times: list[float] = []
        await pause_generation(rollout_ctl)
        while pending_tasks:
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done_tasks:
                paused_items, elapsed = task.result()
                generate_times.append(elapsed)
                paused_items = update_expired_status(
                    paused_items,
                    tail_batch_stale_threshold=tail_batch_stale_threshold,
                )
                for item in paused_items:
                    logger.debug(
                        f"[{self.__class__.__name__}] Task {task_name} | Collecting paused sample "
                        f"(uid: {item.uid}, status: {item.status}, length: {len(item.response_ids or [])})."
                    )
                await replay_buffer.put(paused_items, task_name)
            if pending_tasks and not done_tasks:
                await pause_generation(rollout_ctl)
                await asyncio.sleep(1)
        return generate_times, time.perf_counter() - pause_start

    async def _produce_partial_window(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        required_batch_size: int,
        target_batch_size: int,
        task_name: str,
        rollout_step: int,
        tail_batch_stale_threshold: int,
        source_group_status: Status = Status.ABORTED,
    ) -> "ProducerTimings":
        initial_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        completed_count = initial_completed_count
        target_completed_count = max(required_batch_size, initial_completed_count)
        launch_batch_size = max(target_batch_size - initial_completed_count, 0)
        pending_tasks = set()
        generate_times: list[float] = []
        finished_group_count = 0
        pause_time_s = 0.0

        # A partial window may launch work even after required samples are already available. The launched
        # tasks will be paused into aborted leftovers, keeping the next window warm without delaying training.
        for _ in range(launch_batch_size):
            pending_tasks.add(
                await self._launch_group(
                    agent_loop,
                    sampler,
                    task_name,
                    rollout_step,
                    source_group_status=source_group_status,
                    enable_partial_rollout=True,
                )
            )

        logger.info(
            f"[{self.__class__.__name__}] Task {task_name} | Starting partial window: "
            f"required={required_batch_size}, existing_completed={initial_completed_count}, "
            f"launch={launch_batch_size}, target={target_batch_size}, "
            f"target_completed={target_completed_count}, rollout_step={rollout_step}"
        )

        try:
            while self.should_continue_fn(completed_count, target_completed_count):
                if not pending_tasks:
                    logger.warning(
                        f"[{self.__class__.__name__}] Task {task_name} | No pending tasks before enough samples: "
                        f"completed={completed_count}, target={target_completed_count}."
                    )
                    break
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    items, elapsed = task.result()
                    finished_group_count += 1
                    generate_times.append(elapsed)
                    if self.is_valid_sample_fn(items):
                        completed_count += 1
                    await replay_buffer.put(items, task_name)
                self._raise_if_exceeded_finished_group_budget(
                    task_name=task_name,
                    target_count=target_completed_count,
                    completed_count=completed_count,
                    finished_group_count=finished_group_count,
                )

                while completed_count + len(pending_tasks) < target_completed_count:
                    pending_tasks.add(
                        await self._launch_group(
                            agent_loop,
                            sampler,
                            task_name,
                            rollout_step,
                            source_group_status=source_group_status,
                            enable_partial_rollout=True,
                        )
                    )
        finally:
            if pending_tasks:
                cleanup_generate_times, pause_time_s = await self._cleanup_pending_tasks(
                    pending_tasks,
                    agent_loop,
                    replay_buffer,
                    task_name,
                    tail_batch_stale_threshold=tail_batch_stale_threshold,
                )
                generate_times.extend(cleanup_generate_times)

        return ProducerTimings(generate_times_s=generate_times, pause_time_s=pause_time_s)

    async def _produce_full_window(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        required_batch_size: int,
        target_batch_size: int,
        task_name: str,
        rollout_step: int,
        source_group_status: Status | None,
        tail_batch_stale_threshold: int = 0,
    ) -> "ProducerTimings":
        initial_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        completed_count = initial_completed_count
        target_completed_count = max(required_batch_size, target_batch_size)
        launch_batch_size = max(target_completed_count - initial_completed_count, 0)
        pending_tasks = set()
        generate_times: list[float] = []
        finished_group_count = 0
        pause_time_s = 0.0

        for _ in range(launch_batch_size):
            pending_tasks.add(
                await self._launch_group(
                    agent_loop,
                    sampler,
                    task_name,
                    rollout_step,
                    source_group_status=source_group_status,
                )
            )

        logger.info(
            f"[{self.__class__.__name__}] Task {task_name} | Starting full window: "
            f"existing_completed={initial_completed_count}, launch={launch_batch_size}, "
            f"target_completed={target_completed_count}, rollout_step={rollout_step}"
        )

        try:
            while self.should_continue_fn(completed_count, target_completed_count):
                if not pending_tasks:
                    logger.warning(
                        f"[{self.__class__.__name__}] Task {task_name} | No pending tasks before enough samples: "
                        f"completed={completed_count}, target={target_completed_count}."
                    )
                    break
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    items, elapsed = task.result()
                    finished_group_count += 1
                    generate_times.append(elapsed)
                    if self.is_valid_sample_fn(items):
                        completed_count += 1
                    await replay_buffer.put(items, task_name)
                self._raise_if_exceeded_finished_group_budget(
                    task_name=task_name,
                    target_count=target_completed_count,
                    completed_count=completed_count,
                    finished_group_count=finished_group_count,
                )

                while completed_count + len(pending_tasks) < target_completed_count:
                    pending_tasks.add(
                        await self._launch_group(
                            agent_loop,
                            sampler,
                            task_name,
                            rollout_step,
                            source_group_status=source_group_status,
                        )
                    )
        finally:
            if pending_tasks:
                cleanup_generate_times, pause_time_s = await self._cleanup_pending_tasks(
                    pending_tasks,
                    agent_loop,
                    replay_buffer,
                    task_name,
                    tail_batch_stale_threshold=tail_batch_stale_threshold,
                )
                generate_times.extend(cleanup_generate_times)

        return ProducerTimings(generate_times_s=generate_times, pause_time_s=pause_time_s)

    @abstractmethod
    async def produce_window(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        required_batch_size: int,
        task_name: str,
        target_batch_size: int | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> "ProducerTimings": ...

    @abstractmethod
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
    ) -> "ProducerTimings": ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_window(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        required_batch_size: int,
        task_name: str,
        target_batch_size: int | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> ProducerTimings:
        target_batch_size = required_batch_size if target_batch_size is None else target_batch_size
        unsupported_window = enable_partial_rollout or target_batch_size != required_batch_size
        if unsupported_window:
            raise RuntimeError(
                "SyncProduceStrategy only supports exact full-window production. "
                "Use AsyncProduceStrategyConfig for staleness, over-sampling, tail-batch, or partial-rollout windows."
            )
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        if completed_sample_count > 0:
            raise RuntimeError(
                "SyncProduceStrategy does not consume leftover completed samples in window mode. "
                f"Found {completed_sample_count} completed samples for task {task_name}; use AsyncProduceStrategyConfig "
                "or drain the replay buffer before starting the next sync window."
            )
        return await self._produce_full_window(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=replay_buffer,
            required_batch_size=required_batch_size,
            target_batch_size=target_batch_size,
            task_name=task_name,
            rollout_step=rollout_step,
            source_group_status=None,
        )

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
    ) -> ProducerTimings:
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        if completed_sample_count > 0:
            logger.warning(
                "SyncProduceStrategy found %d completed samples at the start of produce_batch for task %s. "
                "They may be consumed before newly generated samples.",
                completed_sample_count,
                task_name,
            )
        return await self._produce_full_window(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=replay_buffer,
            required_batch_size=batch_size,
            target_batch_size=batch_size,
            task_name=task_name,
            rollout_step=rollout_step,
            source_group_status=None,
        )


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        produce_batch_over_sample_threshold: float,
        produce_batch_enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        tail_batch_stale_threshold: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
        max_finished_groups_multiplier: float,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn, max_finished_groups_multiplier)
        self.produce_batch_over_sample_threshold = produce_batch_over_sample_threshold
        self.produce_batch_enable_partial_rollout = produce_batch_enable_partial_rollout
        self.tail_batch_stale_threshold = tail_batch_stale_threshold
        self.tail_batch_trigger_size = tail_batch_trigger_size

    @staticmethod
    def _refresh_group_staleness(group: list[RolloutState], rollout_step: int) -> None:
        for sample in group:
            refresh_seq_staleness(sample, rollout_step)

    async def _process_leftover_samples(
        self,
        replay_buffer: ReplayBuffer,
        task_name: str,
        rollout_step: int,
        tail_batch_stale_threshold: int,
    ):
        previously_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        if previously_completed_count <= 0:
            return

        pending_requeue_groups = await replay_buffer.get(
            batch_size=previously_completed_count,
            task_name=task_name,
            group_status=Status.COMPLETED,
        )
        processed_group_count = 0
        try:
            for group in pending_requeue_groups:
                self._refresh_group_staleness(group, rollout_step)
                for sample in group:
                    if tail_batch_stale_threshold > 0 and sample.seq_staleness >= tail_batch_stale_threshold:
                        sample.status = Status.EXPIRED
                await replay_buffer.put(group, task_name)
                processed_group_count += 1
        except Exception:
            remaining_groups = pending_requeue_groups[processed_group_count:]
            logger.exception(
                "[%s] Task %s failed while refreshing leftover samples. Attempting best-effort requeue for %d groups.",
                self.__class__.__name__,
                task_name,
                len(remaining_groups),
            )
            for group in remaining_groups:
                try:
                    await replay_buffer.put(group, task_name)
                except Exception:
                    logger.exception(
                        "[%s] Task %s failed to requeue a leftover group during recovery.",
                        self.__class__.__name__,
                        task_name,
                    )
            raise

    async def _produce_with_fixed_concurrency_pool(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        task_name: str,
        rollout_step: int,
        target_count: int,
        initial_completed_count: int,
        data_concurrency: int,
        expired_sample_count: int,
        sample_from_expired_storage: bool,
    ) -> ProducerTimings:
        pending_tasks = set()
        generate_times: list[float] = []
        finished_group_count = 0
        pause_time_s = 0.0

        for _ in range(data_concurrency):
            if sample_from_expired_storage and expired_sample_count > 0:
                source_group_status = Status.EXPIRED
                expired_sample_count -= 1
            else:
                source_group_status = Status.ABORTED
            pending_tasks.add(
                await self._launch_group(
                    agent_loop,
                    sampler,
                    task_name,
                    rollout_step,
                    source_group_status=source_group_status,
                    enable_partial_rollout=self.produce_batch_enable_partial_rollout,
                )
            )

        completed_sample_count = initial_completed_count
        try:
            while self.should_continue_fn(completed_sample_count, target_count):
                if not pending_tasks:
                    logger.warning("All tasks are done but not enough samples collected.")
                    break
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done_tasks:
                    running_items, elapsed = task.result()
                    finished_group_count += 1
                    generate_times.append(elapsed)
                    if self.is_valid_sample_fn(running_items):
                        completed_sample_count += 1
                    running_items = update_expired_status(
                        running_items, tail_batch_stale_threshold=self.tail_batch_stale_threshold
                    )
                    await replay_buffer.put(running_items, task_name)
                    logger.debug(
                        f"[{self.__class__.__name__}] Task {task_name} | Collected {completed_sample_count}/{target_count} valid samples."
                    )
                self._raise_if_exceeded_finished_group_budget(
                    task_name=task_name,
                    target_count=target_count,
                    completed_count=completed_sample_count,
                    finished_group_count=finished_group_count,
                )

                # expired_sample_count is an approximate local budget snapshot. It tracks how many EXPIRED groups
                # we still plan to pull during this pool cycle, but does not try to mirror concurrent replay-buffer
                # mutations caused by newly paused/expired groups.
                while len(pending_tasks) < data_concurrency and self.should_continue_fn(
                    completed_sample_count, target_count
                ):
                    if sample_from_expired_storage and expired_sample_count > 0:
                        source_group_status = Status.EXPIRED
                        expired_sample_count -= 1
                    else:
                        source_group_status = Status.ABORTED
                    pending_tasks.add(
                        await self._launch_group(
                            agent_loop,
                            sampler,
                            task_name,
                            rollout_step,
                            source_group_status=source_group_status,
                            enable_partial_rollout=self.produce_batch_enable_partial_rollout,
                        )
                    )
        finally:
            if pending_tasks:
                cleanup_generate_times, pause_time_s = await self._cleanup_pending_tasks(
                    pending_tasks,
                    agent_loop,
                    replay_buffer,
                    task_name,
                    tail_batch_stale_threshold=self.tail_batch_stale_threshold,
                )
                generate_times.extend(cleanup_generate_times)

        return ProducerTimings(generate_times_s=generate_times, pause_time_s=pause_time_s)

    async def produce_window(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        required_batch_size: int,
        task_name: str,
        target_batch_size: int | None = None,
        rollout_step: int = 0,
        enable_partial_rollout: bool = False,
    ) -> ProducerTimings:
        # Window mode is driven by absolute replay-buffer targets so disaggregated training can request
        # a required train batch plus optional extra pressure for stale/partial rollouts. produce_batch
        # intentionally keeps the fixed-concurrency pool semantics for colocated/eval call sites.
        await self._process_leftover_samples(
            replay_buffer,
            task_name,
            rollout_step,
            self.tail_batch_stale_threshold,
        )
        expired_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        # Window mode samples from replay-buffer leftovers instead of fresh prompts. Normal windows continue
        # aborted groups; tail-batch windows preferentially continue expired groups.
        source_group_status = Status.ABORTED
        target_batch_size = required_batch_size if target_batch_size is None else target_batch_size
        if self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size:
            logger.info(
                f"Tail batch trigger condition met in window mode: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size})."
            )
            source_group_status = Status.EXPIRED
            target_batch_size = required_batch_size

        if enable_partial_rollout:
            return await self._produce_partial_window(
                agent_loop=agent_loop,
                sampler=sampler,
                replay_buffer=replay_buffer,
                required_batch_size=required_batch_size,
                target_batch_size=target_batch_size,
                task_name=task_name,
                rollout_step=rollout_step,
                tail_batch_stale_threshold=self.tail_batch_stale_threshold,
                source_group_status=source_group_status,
            )
        return await self._produce_full_window(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=replay_buffer,
            required_batch_size=required_batch_size,
            target_batch_size=target_batch_size,
            task_name=task_name,
            rollout_step=rollout_step,
            source_group_status=source_group_status,
            tail_batch_stale_threshold=self.tail_batch_stale_threshold,
        )

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
    ) -> ProducerTimings:
        # 1. 处理上一轮遗留的 completed 样本
        await self._process_leftover_samples(
            replay_buffer,
            task_name,
            rollout_step,
            self.tail_batch_stale_threshold,
        )

        # 2. 计算当前并发需求
        previously_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        data_concurrency = max(
            math.ceil((1 + self.produce_batch_over_sample_threshold) * batch_size) - previously_completed_count,
            0,
        )
        expired_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        sample_from_expired_storage = False

        if self.tail_batch_trigger_size > 0 and expired_sample_count >= self.tail_batch_trigger_size:
            logger.info(
                f"Tail batch trigger condition met: {expired_sample_count} expired samples (threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )
            sample_from_expired_storage = True
            data_concurrency = max(batch_size - previously_completed_count, 0)

        logger.info(
            f"[{self.__class__.__name__}] Task {task_name} | Starting produce: data_concurrency: {data_concurrency}, previously_completed: {previously_completed_count}, expired_sample_count: {expired_sample_count}, rollout_step: {rollout_step}"
        )
        return await self._produce_with_fixed_concurrency_pool(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=replay_buffer,
            task_name=task_name,
            rollout_step=rollout_step,
            target_count=batch_size,
            initial_completed_count=previously_completed_count,
            data_concurrency=data_concurrency,
            expired_sample_count=expired_sample_count,
            sample_from_expired_storage=sample_from_expired_storage,
        )
