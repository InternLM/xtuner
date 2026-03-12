import asyncio
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams, Status, update_seq_staleness
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
        self.max_tokens = self.sample_params.max_tokens

    @abstractmethod
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState: ...

    # TODO(@duanyanhui): 讨论下是否需要一个PartialRolloutHandler来专门处理partial rollout的逻辑，目前先放在AgentLoop里，并且需要测试下不同的AgentLoop是否兼容
    async def _preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        # for partial rollout
        if not enable_partial_rollout or not rollout_state.response_ids or rollout_state.status == Status.COMPLETED:
            return rollout_state

        # 如果状态是 EXPIRED，重置 tokens, sample_params和responses, 重新生成
        if rollout_state.status == Status.EXPIRED:
            rollout_state.tokens = rollout_state.prompt_ids
            rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": self.max_tokens})
            rollout_state.response_ids = []
            rollout_state.response = ""
            rollout_state.logprobs = []
            rollout_state.response_mask = []
            rollout_state.response_steps = []
            return rollout_state

        # Set up token and length variable
        response_ids = rollout_state.response_ids
        prompt_ids = list(rollout_state.prompt_ids or [])
        response_len = len(response_ids)
        prompt_len = len(prompt_ids)

        rollout_state.tokens = prompt_ids + response_ids  # partial rollout 拼接逻辑
        remaining_tokens = self.max_tokens - response_len  # partial rollout max_tokens 计算逻辑
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        self.logger.debug(
            f"Sample {rollout_state.uid} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Total tokens: {len(rollout_state.tokens)}"
        )
        # TODO: 处理 routed_experts
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": rollout_state.tokens[prompt_len:] if rollout_state.tokens else [],
            "response": rollout_state.response or "",
            "logprobs": rollout_state.logprobs or [],
            "response_mask": rollout_state.response_mask or [],
        }
        return rollout_state

    async def _postprocess(self, rollout_state: RolloutState, rollout_step: int) -> RolloutState:
        history_dict = rollout_state.extra_fields.pop("history_response_dict", None)
        if not history_dict:
            return rollout_state

        # 需要在拼接历史response_ids前更新seq_staleness
        rollout_state = update_seq_staleness(rollout_state, rollout_step)  # 计算 seq_staleness
        rollout_state.response_ids = history_dict.get("response_ids", []) + (rollout_state.response_ids or [])
        rollout_state.response = history_dict.get("response", "") + (rollout_state.response or "")
        rollout_state.logprobs = history_dict.get("logprobs", []) + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_dict.get("response_mask", []) + (rollout_state.response_mask or [])

        return rollout_state

    async def _generate_pipeline(
        self, rollout_state: RolloutState, enable_partial_rollout: bool, rollout_step: int
    ) -> RolloutState:
        rollout_state = await self._preprocess(rollout_state, enable_partial_rollout)  # preprocess for partial rollout
        rollout_state = await self.generate_sample(rollout_state)
        rollout_state = await self._postprocess(rollout_state, rollout_step)  # postprocess for partial rollout
        return rollout_state

    async def generate_group(
        self, rollout_state: list[RolloutState], enable_partial_rollout: bool = False, rollout_step: int = 0
    ) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self._generate_pipeline(state, enable_partial_rollout, rollout_step))
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
