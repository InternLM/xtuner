from pathlib import Path
import copy
from typing import Iterator, Optional, cast
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.utils.logger import get_logger


logger = get_logger(__name__)


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
        self._consumed_samples: int = 0

    def __len__(self) -> int:
        return len(self.dataloader)

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

        group_data = []
        for _ in range(self.prompt_repeat_k):
            new_data = copy.deepcopy(data)
            new_data.uid = uuid4().int
            group_data.append(new_data)
        self._consumed_samples += 1
        return cast(list[RolloutState], group_data)


class Sampler(_DatasetSampler):
    _DATALOADER_FILE = "dataloader"

    def __init__(
        self,
        dataloader: Dataloader,
        prompt_repeat_k: int,
        replay_buffer: ReplayBuffer,
    ):
        super().__init__(dataloader, prompt_repeat_k)
        self.replay_buffer = replay_buffer

    async def sample(self, task_name: str, group_status: Status | None = None) -> list[RolloutState]:
        if group_status is not None:
            buffer_data = await self.replay_buffer.get(1, task_name=task_name, group_status=group_status)
            if buffer_data:
                return buffer_data[0]
        return self.sample_from_dataloader()

    def save(self, checkpoint_path: Path | str) -> None:
        """Save the sampler's dataloader state to checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        dataloader_state = self.dataloader.get_state_dict(self._consumed_samples)
        torch.save(dataloader_state, checkpoint_path / self._DATALOADER_FILE)

    def resume(self, checkpoint_path: Path | str) -> None:
        """Resume the sampler's dataloader state from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        dataloader_path = checkpoint_path / self._DATALOADER_FILE
        if not dataloader_path.exists():
            logger.warning(f"Dataloader state {dataloader_path} not found, skipping resume.")
            return
        state = torch.load(dataloader_path, map_location="cpu")
        self.dataloader.load_state_dict(state)
        self.dataloader_iter = iter(self.dataloader)
        self._consumed_samples = state["sampler"]["step"]
        self.cur_epoch = state["sampler"]["epoch"]
