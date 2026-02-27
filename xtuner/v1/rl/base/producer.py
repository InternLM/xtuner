import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader

from .agent_loop import AgentLoop
from .replay_buffer import ReplayBuffer


class _DatasetSampler:
    def __init__(self, dataloader: Dataloader, prompt_repeat_k: int):
        self.dataloader = dataloader
        self.dataloader_iter: Optional[Iterator] = None
        self.cur_epoch = 0
        self.prompt_repeat_k = prompt_repeat_k

    def sample_from_dataloader(self) -> list[RolloutState]:
        if self.dataloader_iter is None:
            self.dataloader_iter = iter(self.dataloader)
        assert self.dataloader_iter is not None
        try:
            data = next(self.dataloader_iter)[0]
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        group_data = [data] * self.prompt_repeat_k
        return group_data


class Sampler(_DatasetSampler):
    def __init__(
        self,
        dataloader: Dataloader,
        prompt_repeat_k: int,
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(dataloader, prompt_repeat_k)
        self.replay_buffer = replay_buffer

    async def sample(self, task_name: str) -> list[RolloutState]:
        buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=Status.ABORTED)
        if len(buffer_data) == 0:
            return self.sample_from_dataloader()
        else:
            return buffer_data[0]


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, batch_size: int) -> bool:
    return completed_count < batch_size


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: Optional[Callable[[list[RolloutState]], bool]] = None,
        should_continue_fn: Optional[Callable[[int, int], bool]] = None,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn or default_is_valid_sample_fn
        self.should_continue_fn = should_continue_fn or default_should_continue_fn

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
        staleness_threshold: float = 0.0,
        is_valid_sample_fn: Optional[Callable[[list[RolloutState]], bool]] = None,
        should_continue_fn: Optional[Callable[[int, int], bool]] = None,
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


class SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    dataloader_cfg: DataloaderConfig
    prompt_repeat_k: int = 1

    def build(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str, replay_buffer: ReplayBuffer
    ) -> Sampler:
        if isinstance(tokenizer, str):
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            tokenizer_obj = tokenizer
        dataloader = self.dataloader_cfg.build(
            tokenizer=tokenizer_obj, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        return Sampler(dataloader=dataloader, prompt_repeat_k=self.prompt_repeat_k, replay_buffer=replay_buffer)


class ProducerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    # TODO: 要考虑怎么方便用户使用自定义的 ProduceStrategy, 并且怎么使用用户需要的参数
    type: Type[ProduceStrategy] | Literal["sync", "over"] = "sync"
    staleness_threshold: float = 0.0  # only used for over_produce strategy
    is_valid_sample_fn: Optional[Callable[[list[RolloutState]], bool]] = None
    should_continue_fn: Optional[Callable[[int, int], bool]] = None

    def build(self) -> "ProduceStrategy":
        if self.type == "sync":
            if self.staleness_threshold != 0.0:
                print("Warning: staleness_threshold is ignored in sync produce strategy.")
            return SyncProduceStrategy(
                is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
            )
        elif self.type == "over":
            return OverProduceStrategy(
                staleness_threshold=self.staleness_threshold,
                is_valid_sample_fn=self.is_valid_sample_fn,
                should_continue_fn=self.should_continue_fn,
            )
        else:
            return self.type(is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn)
