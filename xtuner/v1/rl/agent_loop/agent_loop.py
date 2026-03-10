import asyncio
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import NativeJudger, RouterJudger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)  # TODO: extra="forbid"
    hf_checkpoint: str
    sample_params: SampleParams

    @abstractmethod
    def build(self, rollout_controller, judger=None, logger=None) -> "AgentLoop": ...


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    def build(self, rollout_controller, judger=None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            logger=logger,
        )


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Callable | NativeJudger | RouterJudger | None = None,
        logger=None,
    ) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judger = judger
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState: ...

    async def pause(self) -> None:
        """Pause the agent loop if supported by the implementation."""
        # Default implementation is a no-op to keep behavior unchanged.
        return None

    async def generate_group(self, rollout_state: list[RolloutState]) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def judge_sample(self, rollout_state: RolloutState) -> RolloutState:
        if self.judger is None:
            return rollout_state
        if callable(self.judger):
            rollout_state = await self.judger(rollout_state)
        elif isinstance(self.judger, RouterJudger) or isinstance(self.judger, NativeJudger):
            rollout_state = await self.judger.judge(rollout_state)  # type: ignore[operator]
        else:
            raise ValueError(f"Invalid judger type: {type(self.judger)}")
        return rollout_state


class SingleTurnAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        assert rollout_state.sample_params is not None, "sample_params must be set in rollout_state"
        rollout_state.tokens = rollout_state.prompt_ids
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        rollout_state = await self.judge_sample(rollout_state)
        return rollout_state
