import asyncio
from abc import ABC, abstractmethod
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig

from ..agent_loop import AgentLoop
from .replay_buffer import ReplayBuffer


# TODO: 用户把自己的数据集转换成rolloutstate的逻辑放在哪里？
class Sampler:
    def __init__(
        self,
        task_name: str,
        dataloader_cfg: DataloaderConfig,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        self.dataloader = dataloader_cfg.build(
            tokenizer=self.tokenizer, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        self.dataloader_iter = iter(self.dataloader)
        self.cur_epoch = 0
        self.task_name = task_name

    async def sample(self) -> RolloutState:
        try:
            data = next(self.dataloader_iter)[0]
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        return data


class SamplerWithReplayBuffer(Sampler):
    def __init__(
        self,
        task_name: str,
        dataloader_cfg: DataloaderConfig,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(task_name, dataloader_cfg, tokenizer)
        self.replay_buffer = replay_buffer

    def _sample_from_dataloader(self) -> RolloutState:
        try:
            data = next(self.dataloader_iter)[0]
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        return data

    async def sample(self) -> list[RolloutState]:
        data = await self.replay_buffer.get(1, task_name=self.task_name, group_status=Status.ABORTED)
        if len(data) == 0:
            data = self._sample_from_dataloader()
        return data


class ProduceStrategy(ABC):
    # NOTE: dataloader不作为ProduceStrategy的成员变量的原因：produce_strategy不绑定dataloader
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    @abstractmethod
    async def produce_batch(self, agent_loop: AgentLoop, sampler: Sampler, batch_size: int, prompt_k: int): ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, agent_loop: AgentLoop, sampler: Sampler, batch_size: int, prompt_k: int):
        data_concurrency = batch_size
        pending_tasks = set()
        for _ in range(data_concurrency):
            rollout_state = await sampler.sample()
            task = asyncio.create_task(agent_loop.generate_group(rollout_state, prompt_k))
            pending_tasks.add(task)

        init_completed_sample_count = await self.replay_buffer.count(
            task_name=agent_loop.task_name, group_status=Status.COMPLETED
        )
        completed_sample_count = init_completed_sample_count
        while completed_sample_count < data_concurrency:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )

            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                try:
                    await self.replay_buffer.put(items=task.result(), task_name=agent_loop.task_name)
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")

            if len(pending_tasks) + completed_sample_count < data_concurrency + init_completed_sample_count:
                rollout_state = await sampler.sample()
                task = asyncio.create_task(agent_loop.generate_group(rollout_state, prompt_k))
                pending_tasks.add(task)

            completed_sample_count = await self.replay_buffer.count(
                task_name=agent_loop.task_name, group_status=Status.COMPLETED
            )


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        staleness_threshold: float = 0.0,
        enable_partial_rollout: bool = False,
        tail_batch_trigger_size: int = 0,
        tail_batch_candidate_step: int = 0,
    ):
        super().__init__(replay_buffer)
        self.staleness_threshold = staleness_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.tail_batch_candidate_step = tail_batch_candidate_step

    async def produce_batch(
        self, agent_loop: AgentLoop, sampler: SamplerWithReplayBuffer, batch_size: int, prompt_k: int
    ):
        data_concurrency = (1 + self.staleness_threshold) * batch_size
        print(
            f"AsyncProduceStrategy: data_concurrency={data_concurrency}, staleness_threshold={self.staleness_threshold}"
        )
        pending_tasks = set()
        for _ in range(data_concurrency):
            rollout_state = await sampler.sample()
            task = asyncio.create_task(agent_loop.generate_group(rollout_state, prompt_k))
            pending_tasks.add(task)

        init_completed_sample_count = await self.replay_buffer.count(
            task_name=agent_loop.task_name, group_status=Status.COMPLETED
        )
        completed_sample_count = init_completed_sample_count
        while completed_sample_count < data_concurrency:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
            )

            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                try:
                    await self.replay_buffer.put(items=task.result(), task_name=agent_loop.task_name)
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")

            print(f"Completed sample count: {completed_sample_count}, Pending task count: {len(pending_tasks)}")
            completed_sample_count = await self.replay_buffer.count(
                task_name=agent_loop.task_name, group_status=Status.COMPLETED
            )
            if len(pending_tasks) + completed_sample_count < data_concurrency + init_completed_sample_count:
                rollout_state = await sampler.sample()
                task = asyncio.create_task(agent_loop.generate_group(rollout_state, prompt_k))
                pending_tasks.add(task)

        if len(pending_tasks) > 0:
            await agent_loop.pause()
            while len(pending_tasks) > 0:
                _, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                if len(pending_tasks) > 0:
                    await agent_loop.pause()
                    await asyncio.sleep(1)
        print("All worker tasks have completed after pausing env controller.")
