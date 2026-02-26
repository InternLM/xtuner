import abc
import asyncio
from abc import ABC
from typing import Awaitable, Callable

import ray

from xtuner.v1.data_proto import RolloutState, SampleParams
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Callable[[RolloutState], Awaitable[bool]] | ray.actor.ActorHandle | None = None,
    ) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judger = judger

    @abc.abstractmethod
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState: ...

    async def generate_group(self, rollout_state: list[RolloutState]) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = asyncio.create_task(self.generate_sample(state))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples
