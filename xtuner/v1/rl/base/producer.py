import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig

from .agent_loop import AgentLoop
from .replay_buffer import ReplayBuffer


class SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    dataloader_cfg: DataloaderConfig
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]
    prompt_repeat_k: int = 1


class _DatasetSampler:
    def __init__(self, sampler_cfg: SamplerConfig):
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if isinstance(sampler_cfg.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(sampler_cfg.tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = sampler_cfg.tokenizer
        self.dataloader = sampler_cfg.dataloader_cfg.build(
            tokenizer=self.tokenizer, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        self.dataloader_iter: Optional[Iterator] = None
        self.cur_epoch = 0
        self.prompt_repeat_k = sampler_cfg.prompt_repeat_k

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
        sampler_cfg: SamplerConfig,
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(sampler_cfg)
        self.replay_buffer = replay_buffer

    async def sample(self, task_name: str) -> list[RolloutState]:
        buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=Status.ABORTED)
        if len(buffer_data) == 0:
            return self.sample_from_dataloader()
        else:
            return buffer_data[0]


class ProduceStrategy(ABC):
    @abstractmethod
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        is_valid_sample_fn: Optional[
            Callable[[list[RolloutState]], bool]
        ] = None,  # 判断生成的样本是否有效，例如completed有效，filtered / aborted无效等
        should_continue_fn: Optional[
            Callable[[int, int], bool]
        ] = None,  # 判断是否继续生成，例如可以根据已经生成的样本数量来判断是否继续生成
    ): ...


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return all(sample.status == Status.COMPLETED for sample in samples)


def default_should_continue_fn(completed_count: int, batch_size: int) -> bool:
    return completed_count < batch_size


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        is_valid_sample_fn: Optional[Callable[[list[RolloutState]], bool]] = None,
        should_continue_fn: Optional[Callable[[int, int], bool]] = None,
    ):
        if is_valid_sample_fn is None:
            is_valid_sample_fn = default_is_valid_sample_fn
        if should_continue_fn is None:
            should_continue_fn = default_should_continue_fn

        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(batch_size):
            rollout_state = await sampler.sample(task_name=task_name)
            task = asyncio.create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        while should_continue_fn(completed_sample_count, batch_size):
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
                if is_valid_sample_fn(items):
                    completed_sample_count += 1
                await replay_buffer.put(items, task_name)

            while len(pending_tasks) + completed_sample_count < batch_size and should_continue_fn(
                completed_sample_count, batch_size
            ):
                rollout_state = await sampler.sample(task_name=task_name)
                task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)


class OverProduceStrategy(ProduceStrategy):
    def __init__(self, staleness_threshold: float = 0.0):
        self.staleness_threshold = staleness_threshold

    async def produce_batch(
        self,
        agent_loop: AgentLoop,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        task_name: str,
        is_valid_sample_fn: Optional[Callable[[list[RolloutState]], bool]] = None,
        should_continue_fn: Optional[Callable[[int, int], bool]] = None,
    ):
        if is_valid_sample_fn is None:
            is_valid_sample_fn = default_is_valid_sample_fn
        if should_continue_fn is None:
            should_continue_fn = default_should_continue_fn

        data_concurrency = int((1 + self.staleness_threshold) * batch_size)
        pending_tasks = set()
        init_completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)

        for _ in range(data_concurrency):
            rollout_state = await sampler.sample(task_name=task_name)
            task = asyncio.create_task(agent_loop.generate_group(rollout_state))
            pending_tasks.add(task)

        completed_sample_count = init_completed_sample_count
        while should_continue_fn(completed_sample_count, batch_size):
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
                if is_valid_sample_fn(items):
                    completed_sample_count += 1
                await replay_buffer.put(items, task_name)

            while len(
                pending_tasks
            ) + completed_sample_count < data_concurrency + init_completed_sample_count and should_continue_fn(
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
