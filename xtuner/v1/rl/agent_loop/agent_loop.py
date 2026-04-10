import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams
from xtuner.v1.rl.judger import Judger, JudgerConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import create_task
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.processing_utils import load_processor, load_tokenizer


JudgerCallable: TypeAlias = Callable[[RolloutState], Any]
JudgerLike: TypeAlias = Judger | JudgerCallable
JudgerSpec: TypeAlias = JudgerLike | dict[str, JudgerLike] | None
JudgerConfigLike: TypeAlias = JudgerConfig | JudgerCallable
JudgerConfigSpec: TypeAlias = JudgerConfigLike | dict[str, JudgerConfigLike] | None


class AgentLoopConfig(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    hf_checkpoint: str
    sample_params: SampleParams
    judger_config: JudgerConfigSpec = None

    def build_judger(self) -> JudgerSpec:
        if self.judger_config is None:
            return None

        if isinstance(self.judger_config, dict):
            judger_dict = {}
            for key, config in self.judger_config.items():
                if isinstance(config, JudgerConfig):
                    judger_dict[key] = config.build()
                elif callable(config):
                    judger_dict[key] = config
                else:
                    raise ValueError(f"Invalid judger config type: {type(config)} for key {key}")
            return judger_dict
        elif isinstance(self.judger_config, JudgerConfig):
            return self.judger_config.build()
        elif callable(self.judger_config):
            return self.judger_config
        else:
            raise ValueError(f"Invalid judger config type: {type(self.judger_config)}")

    @abstractmethod
    def build(self, rollout_controller, logger=None) -> "AgentLoop": ...


class AgentLoop(ABC):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: JudgerSpec = None,
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
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState: ...

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        return group_samples

    async def judge_sample(self, rollout_state: RolloutState) -> RolloutState:
        if self.judger is None:
            return rollout_state

        judger = self.judger
        if isinstance(judger, dict):
            if len(judger) > 1:
                raise NotImplementedError("Multiple judgers require a custom AgentLoop.judge_sample implementation.")
            judger = next(iter(judger.values()))

        if isinstance(judger, Judger):
            rollout_state = await judger.judge(rollout_state)
        elif callable(judger):
            rollout_state = judger(rollout_state)
            if inspect.isawaitable(rollout_state):
                rollout_state = await rollout_state
        else:
            raise ValueError(f"Invalid judger type: {type(judger)}")

        if not isinstance(rollout_state, RolloutState):
            raise TypeError(f"Judger must return RolloutState, but got {type(rollout_state)}")
        return rollout_state
