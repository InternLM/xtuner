import asyncio
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union

from tqdm.auto import tqdm

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig

from .agent_loop import AgentLoop
from .replay_buffer import ReplayBuffer


# TODO: 用户把自己的数据集转换成rolloutstate的逻辑放在哪里？
class Sampler:
    def __init__(
        self,
        dataloader_cfg: DataloaderConfig,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        prompt_repeat_k: int = 1,
    ):
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        self.dataloader = dataloader_cfg.build(
            tokenizer=self.tokenizer, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        self.dataloader_iter: Optional[Iterator] = None
        self.cur_epoch = 0
        self.prompt_repeat_k = prompt_repeat_k

    async def sample(self, task_name: str) -> list[RolloutState]:
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


class SamplerWithReplayBuffer(Sampler):
    def __init__(
        self,
        dataloader_cfg: DataloaderConfig,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        replay_buffer: ReplayBuffer,
        prompt_repeat_k: int = 1,
    ):
        super().__init__(dataloader_cfg, tokenizer, prompt_repeat_k)
        self.replay_buffer = replay_buffer

    def _sample_from_dataloader(self) -> list[RolloutState]:
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

    async def sample(self, task_name: str) -> list[RolloutState]:
        buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=Status.ABORTED)
        if len(buffer_data) == 0:
            return self._sample_from_dataloader()
        else:
            return buffer_data[0]


class ProduceStrategy(ABC):
    @abstractmethod
    async def produce_batch(
        self, agent_loop: AgentLoop, sampler: Sampler, replay_buffer: ReplayBuffer, batch_size: int, task_name: str
    ): ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(
        self, agent_loop: AgentLoop, sampler: Sampler, replay_buffer: ReplayBuffer, batch_size: int, task_name: str
    ):
        pbar_refrash_step = max(1, int(batch_size * 0.1))
        pending_tasks = set()
        completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."
        with tqdm(total=batch_size, desc=f"Sync Producer [{task_name}]", miniters=pbar_refrash_step) as pbar:
            last_pbar_n = completed_sample_count
            pbar.update(last_pbar_n)
            for _ in range(batch_size):
                rollout_state = await sampler.sample(task_name=task_name)
                task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)

            while completed_sample_count < batch_size:
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
                        await replay_buffer.put(items=task.result(), task_name=task_name)
                    except Exception as e:
                        print(f"Error in generating trajectory: {e}")

                if len(pending_tasks) + completed_sample_count < batch_size:
                    rollout_state = await sampler.sample(task_name=task_name)
                    task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                    pending_tasks.add(task)

                completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
                pbar.update(completed_sample_count - last_pbar_n)
                last_pbar_n = completed_sample_count


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
        self,
        staleness_threshold: float = 0.0,
        enable_partial_rollout: bool = False,
        tail_batch_trigger_size: int = 0,
        tail_batch_candidate_step: int = 0,
    ):
        self.staleness_threshold = staleness_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.tail_batch_candidate_step = tail_batch_candidate_step

    async def produce_batch(
        self, agent_loop: AgentLoop, sampler: Sampler, replay_buffer: ReplayBuffer, batch_size: int, task_name: str
    ):
        data_concurrency = int((1 + self.staleness_threshold) * batch_size)
        pbar_refrash_step = max(1, int(data_concurrency * 0.1))
        pending_tasks = set()
        init_completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        with tqdm(total=batch_size, desc=f"ASync Producer [{task_name}]", miniters=pbar_refrash_step) as pbar:
            last_pbar_n = init_completed_sample_count
            pbar.update(last_pbar_n)
            for _ in range(data_concurrency):
                rollout_state = await sampler.sample(task_name=task_name)
                task = asyncio.create_task(agent_loop.generate_group(rollout_state))
                pending_tasks.add(task)

            completed_sample_count = init_completed_sample_count
            while completed_sample_count < batch_size:
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
                        await replay_buffer.put(items=task.result(), task_name=task_name)
                    except Exception as e:
                        print(f"Error in generating trajectory: {e}")

                print(f"Completed sample count: {completed_sample_count}, Pending task count: {len(pending_tasks)}")
                completed_sample_count = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
                pbar.update(completed_sample_count - last_pbar_n)
                last_pbar_n = completed_sample_count

                if len(pending_tasks) + completed_sample_count < data_concurrency + init_completed_sample_count:
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
