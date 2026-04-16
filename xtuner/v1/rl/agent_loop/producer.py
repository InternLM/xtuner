import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import ray
from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status, update_expired_status
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import pause_generation
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoopSpec, get_agent_loop_rollout_ctl
from .sampler import Sampler
from .utils import refresh_seq_staleness


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()


async def _timed_generate_group(
    agent_loop: AgentLoopSpec,
    rollout_state: list[RolloutState],
    enable_partial_rollout: bool = False,
    rollout_step: int = 0,
    model_rollout_step: int | None = None,
) -> list[RolloutState]:
    start = time.perf_counter()
    if model_rollout_step is None:
        model_rollout_step = rollout_step
    if isinstance(agent_loop, ray.actor.ActorHandle):
        result = await agent_loop.generate_group.remote(
            rollout_state,
            enable_partial_rollout=enable_partial_rollout,
            rollout_step=rollout_step,
            model_rollout_step=model_rollout_step,
        )
    else:
        result = await agent_loop.generate_group(
            rollout_state,
            enable_partial_rollout=enable_partial_rollout,
            rollout_step=rollout_step,
            model_rollout_step=model_rollout_step,
        )
    elapsed = time.perf_counter() - start
    for item in result:
        extra_fields = getattr(item, "extra_fields", None)
        if extra_fields is None:
            extra_fields = {}
            setattr(item, "extra_fields", extra_fields)
        extra_fields[GROUP_GENERATE_TIME_KEY] = elapsed
    return result


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


class ProduceStrategyConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(self) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    def build(self) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    tail_batch_stale_threshold: int = 0
    tail_batch_trigger_size: int = 0

    def build(self) -> "AsyncProduceStrategy":
        return AsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            tail_batch_stale_threshold=self.tail_batch_stale_threshold,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event: asyncio.Event | None = None,
    ) -> ProduceBatchStatus: ...

    async def cleanup_pending_tasks(
        self, agent_loop: AgentLoopSpec, replay_buffer: ReplayBuffer, task_name: str
    ) -> float:
        return 0.0


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event: asyncio.Event | None = None,
    ) -> ProduceBatchStatus:
        if model_rollout_step is None:
            model_rollout_step = rollout_step
        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(batch_size):
            rollout_state = await sampler.sample(task_name=task_name)
            task = create_task(
                _timed_generate_group(
                    agent_loop,
                    rollout_state,
                    rollout_step=rollout_step,
                    model_rollout_step=model_rollout_step,
                )
            )
            pending_tasks.add(task)

        logger.info(f"Started {len(pending_tasks)} initial tasks for SyncProduceStrategy.")

        while self.should_continue_fn(completed_sample_count, batch_size):
            if not pending_tasks:
                logger.warning("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                items = task.result()
                if self.is_valid_sample_fn(items):
                    completed_sample_count += 1
                    logger.info(f"Collected {completed_sample_count}/{batch_size} valid samples for task {task_name}.")
                await replay_buffer.put(items, task_name)

            while len(pending_tasks) + completed_sample_count < batch_size and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = create_task(
                    _timed_generate_group(
                        agent_loop,
                        rollout_state,
                        rollout_step=rollout_step,
                        model_rollout_step=model_rollout_step,
                    )
                )
                pending_tasks.add(task)

        return ProduceBatchStatus.NORMAL


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        tail_batch_stale_threshold: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.tail_batch_stale_threshold = tail_batch_stale_threshold
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self._pending_tasks: set[asyncio.Task] = set()
        self._current_rollout_step = 0

    def _is_model_expired(self, rollout_step: int, model_rollout_step: int) -> bool:
        if self.tail_batch_stale_threshold <= 0:
            return False
        return rollout_step - model_rollout_step >= self.tail_batch_stale_threshold

    async def _store_generated_group(
        self,
        items: list[RolloutState],
        replay_buffer: ReplayBuffer,
        task_name: str,
    ) -> None:
        items = update_expired_status(items, tail_batch_stale_threshold=self.tail_batch_stale_threshold)
        await replay_buffer.put(items, task_name)

    async def _collect_done_tasks(self, replay_buffer: ReplayBuffer, task_name: str) -> int:
        done_tasks = {task for task in self._pending_tasks if task.done()}
        if not done_tasks:
            return 0

        self._pending_tasks -= done_tasks
        completed_count = 0
        for task in done_tasks:
            items = task.result()
            if self.is_valid_sample_fn(items):
                completed_count += 1
            await self._store_generated_group(items, replay_buffer, task_name)
        return completed_count

    async def _schedule_tasks(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        completed_sample_count: int,
        target_total: int,
        sample_from_expired_storage: bool,
        expired_sample_count: int,
        task_name: str,
        model_rollout_step: int,
    ) -> int:
        while len(self._pending_tasks) + completed_sample_count < target_total:
            if sample_from_expired_storage and expired_sample_count > 0:
                group_status = Status.EXPIRED
                expired_sample_count -= 1
            else:
                group_status = Status.ABORTED
            rollout_state = await sampler.sample(task_name=task_name, group_status=group_status)
            task = create_task(
                _timed_generate_group(
                    agent_loop,
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                    rollout_step=model_rollout_step,
                    model_rollout_step=model_rollout_step,
                )
            )
            self._pending_tasks.add(task)
        return expired_sample_count

    async def _process_leftover_samples(self, replay_buffer: ReplayBuffer, task_name: str, rollout_step: int):
        previously_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        if (not self.enable_partial_rollout or self.tail_batch_stale_threshold > 0) and previously_completed_count > 0:
            previously_completed = await replay_buffer.get(
                batch_size=previously_completed_count,
                task_name=task_name,
                group_status=Status.COMPLETED,
            )
            for group in previously_completed:
                refresh_seq_staleness(group, rollout_step)
                for sample in group:
                    if self.tail_batch_stale_threshold > 0 and sample.seq_staleness >= self.tail_batch_stale_threshold:
                        sample.status = Status.EXPIRED
                    elif not self.enable_partial_rollout:  # TODO: 为什么 COMPLETED 样本要 abort?
                        sample.status = Status.ABORTED
                await replay_buffer.put(group, task_name)

    async def cleanup_pending_tasks(
        self, agent_loop: AgentLoopSpec, replay_buffer: ReplayBuffer, task_name: str
    ) -> float:
        pause_start = time.perf_counter()
        if not self._pending_tasks:
            return 0.0

        rollout_ctl = await get_agent_loop_rollout_ctl(agent_loop)
        await pause_generation(rollout_ctl)
        while len(self._pending_tasks) > 0:
            done_task, pending_tasks = await asyncio.wait(
                self._pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            self._pending_tasks = set(pending_tasks)

            for task in done_task:
                paused_items = task.result()
                refresh_seq_staleness(paused_items, self._current_rollout_step)
                paused_items = update_expired_status(
                    paused_items, tail_batch_stale_threshold=self.tail_batch_stale_threshold
                )
                for item in paused_items:
                    logger.debug(
                        f"[{self.__class__.__name__}] Task {task_name} | Collecting aborted sample (uid: {item.uid}, status: {item.status}, length: {len(item.response_ids or [])}) after pausing generation."
                    )
                await replay_buffer.put(paused_items, task_name)
            if len(self._pending_tasks) > 0:
                await pause_generation(rollout_ctl)
                await asyncio.sleep(1)
        return time.perf_counter() - pause_start

    async def produce_batch(
        self,
        agent_loop: AgentLoopSpec,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event: asyncio.Event | None = None,
    ) -> ProduceBatchStatus:
        # TODO: 兼容共卡使用方式
        self._current_rollout_step = rollout_step
        if model_rollout_step is None:
            model_rollout_step = rollout_step
        if update_event is None:
            update_event = asyncio.Event()

        await self._collect_done_tasks(replay_buffer, task_name)

        # 1. 当前 rollout 权重过旧时立即停机，不再处理 leftovers
        if self._is_model_expired(rollout_step, model_rollout_step):
            return ProduceBatchStatus.EXPIRED_BATCH
        if update_event.is_set():
            return ProduceBatchStatus.UPDATE_ABORT

        # 2. 处理上一轮遗留的 completed 样本
        await self._process_leftover_samples(replay_buffer, task_name, rollout_step)

        # 3. 计算当前并发需求
        previously_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        data_concurrency = max(0, int((1 + self.over_sample_threshold) * batch_size) - previously_completed_count)
        expired_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
        sample_from_expired_storage = False

        if self.tail_batch_trigger_size > 0 and expired_sample_count >= self.tail_batch_trigger_size:
            logger.info(
                f"Tail batch trigger condition met: {expired_sample_count} expired samples (threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )
            sample_from_expired_storage = True
            data_concurrency = max(0, batch_size - previously_completed_count)

        logger.info(
            f"[{self.__class__.__name__}] Task {task_name} | Starting produce: data_concurrency: {data_concurrency}, previously_completed: {previously_completed_count}, expired_sample_count: {expired_sample_count}, rollout_step: {rollout_step}"
        )

        target_total = data_concurrency + previously_completed_count
        completed_sample_count = previously_completed_count
        while self.should_continue_fn(completed_sample_count, batch_size):
            if update_event.is_set():
                return ProduceBatchStatus.UPDATE_ABORT
            if self._is_model_expired(rollout_step, model_rollout_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            expired_sample_count = await self._schedule_tasks(
                agent_loop=agent_loop,
                sampler=sampler,
                completed_sample_count=completed_sample_count,
                target_total=target_total,
                sample_from_expired_storage=sample_from_expired_storage,
                expired_sample_count=expired_sample_count,
                task_name=task_name,
                model_rollout_step=model_rollout_step,
            )
            if completed_sample_count >= batch_size:
                break
            if not self._pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                self._pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )
            self._pending_tasks = set(pending_tasks)

            for task in done_tasks:
                running_items = task.result()
                if self.is_valid_sample_fn(running_items):
                    completed_sample_count += 1
                await self._store_generated_group(running_items, replay_buffer, task_name)
                logger.debug(
                    f"[{self.__class__.__name__}] Task {task_name} | Collected {completed_sample_count}/{batch_size} valid samples."
                )

        return ProduceBatchStatus.NORMAL
