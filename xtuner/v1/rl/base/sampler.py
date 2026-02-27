from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader

from .replay_buffer import ReplayBuffer


class SamplerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    dataloader_cfg: DataloaderConfig
    prompt_repeat_k: int = 1

    def build(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str, replay_buffer: ReplayBuffer
    ) -> "Sampler":
        if isinstance(tokenizer, str):
            tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            tokenizer_obj = tokenizer
        dataloader = self.dataloader_cfg.build(
            tokenizer=tokenizer_obj, dp_mesh=None, global_batch_size=1, micro_batch_size=1, seed=1
        )
        return Sampler(dataloader=dataloader, prompt_repeat_k=self.prompt_repeat_k, replay_buffer=replay_buffer)


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
