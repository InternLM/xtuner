import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status

from .agent_loop import AgentLoop
from .replay_buffer import ReplayBuffer
from .sampler import Sampler


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


class ProducerConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(self) -> "ProduceStrategy": ...


class SyncProduceStrategyConfig(ProducerConfig):
    def build(self) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class OverProduceStrategyConfig(ProducerConfig):
    staleness_threshold: float = 0.0

    def build(self) -> "OverProduceStrategy":
        return OverProduceStrategy(
            staleness_threshold=self.staleness_threshold,
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
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
    ): ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
    ):
        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(batch_size):
            rollout_state = await sampler.sample(task_name=task_name)
            task = asyncio.create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        while self.should_continue_fn(completed_sample_count, batch_size):
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
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
                await replay_buffer.put(items, task_name)

            while len(pending_tasks) + completed_sample_count < batch_size and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)


class OverProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        staleness_threshold: float,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)
        self.staleness_threshold = staleness_threshold

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
    ):
        data_concurrency = int((1 + self.staleness_threshold) * batch_size)
        pending_tasks = set()
        init_completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)

        for _ in range(data_concurrency):
            rollout_state = await sampler.sample(task_name=task_name)
            task = asyncio.create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        completed_sample_count = init_completed_sample_count
        while self.should_continue_fn(completed_sample_count, batch_size):
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
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
                await replay_buffer.put(items, task_name)

            while len(
                pending_tasks
            ) + completed_sample_count < data_concurrency + init_completed_sample_count and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)

        if len(pending_tasks) > 0:
            await agent_loop.pause()
            while len(pending_tasks) > 0:
                _, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                if len(pending_tasks) > 0:
                    await agent_loop.pause()
                    await asyncio.sleep(1)
        print("All worker tasks have completed after pausing env controller.")
