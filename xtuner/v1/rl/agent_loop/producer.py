import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout.utils import continue_generation, pause_generation
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger

from .agent_loop import AgentLoop
from .sampler import Sampler


logger = get_logger()


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
            task = create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        logger.info(f"Started {len(pending_tasks)} initial tasks for SyncProduceStrategy.")

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
                    logger.info(f"Collected {completed_sample_count}/{batch_size} valid samples for task {task_name}.")
                await replay_buffer.put(items, task_name)

            while len(pending_tasks) + completed_sample_count < batch_size and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)


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

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
    ):
        # 重启 rollout controller
        rollout_ctl = agent_loop.rollout_ctl
        await continue_generation(rollout_ctl)

        pending_tasks = set()
        previously_completed_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        data_concurrency = int((1 + self.over_sample_threshold) * batch_size) - previously_completed_count
        logger.info(
            f"[AsyncProduceStrategy] Task '{task_name}' | Starting produce: data_concurrency: {data_concurrency}, previously_completed: {previously_completed_count}"
        )
        for _ in range(data_concurrency):
            rollout_state = await sampler.sample(task_name=task_name)
            task = create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        completed_sample_count = previously_completed_count
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
                running_items: list[RolloutState] = task.result()
                if self.is_valid_sample_fn(running_items):
                    completed_sample_count += 1
                await replay_buffer.put(running_items, task_name)
                logger.debug(
                    f"[AsyncProduceStrategy] Task '{task_name}' | Collected {completed_sample_count}/{batch_size} valid samples."
                )

            while len(
                pending_tasks
            ) + completed_sample_count < data_concurrency + previously_completed_count and self.should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)

        if len(pending_tasks) > 0:
            await pause_generation(rollout_ctl)
            while len(pending_tasks) > 0:
                done_task, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done_task:
                    paused_items: list[RolloutState] = task.result()
                    for item in paused_items:
                        logger.debug(
                            f"[AsyncProduceStrategy] Task '{task_name}' | Collecting aborted sample (uid: {item.uid}, status: {item.status}, length: {len(item.response_ids or [])}) after pausing generation."
                        )
                    await replay_buffer.put(paused_items, task_name)
                if len(pending_tasks) > 0:
                    await pause_generation(rollout_ctl)
                    await asyncio.sleep(1)

        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        aborted_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.ABORTED)
        logger.info(
            f"[AsyncProduceStrategy] Task '{task_name}' | Finished! Final completed count: {completed_sample_count}, aborted count: {aborted_sample_count} in replay buffer."
        )
        logger.info(
            f"[AsyncProduceStrategy] Task '{task_name}' | All worker tasks completed after pausing env controller."
        )
