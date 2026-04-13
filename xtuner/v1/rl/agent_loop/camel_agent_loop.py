import copy
import os
from typing import Any, Literal

import ray
from openai import AsyncOpenAI
from pydantic import ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.chat_adapter.collector import reset_current_trace_collector, set_current_trace_collector
from xtuner.v1.rl.rollout.utils import ROLLOUT_RAY_GET_TIMEOUT

from .agent_loop import AgentLoop, AgentLoopConfig

from camel.agents import ChatAgent
from camel.models import OpenAICompatibleModel
from camel.utils import BaseTokenCounter


class XTunerCamelTokenCounter(BaseTokenCounter):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_tokens_from_messages(self, messages: list[dict]) -> int:
        return len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)


class CamelAgentLoopConfig(AgentLoopConfig):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mode: Literal["single_turn"] = "single_turn"
    context_length: int | None = None
    system_message: str | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | None = None

    def build(self, rollout_controller, judger=None, logger=None) -> "CamelAgentLoop":
        return CamelAgentLoop(
            context_length=self.context_length,
            system_message=self.system_message,
            tools=self.tools,
            tool_choice=self.tool_choice,
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
            logger=logger,
        )


class CamelAgentLoop(AgentLoop):
    def __init__(
        self,
        context_length: int | None,
        system_message: str | None,
        tools: list[Any] | None,
        tool_choice: str | dict[str, Any] | None,
        rollout_ctl: RolloutController,
        hf_checkpoint: str,
        sample_params: SampleParams,
        judger=None,
        logger=None,
    ) -> None:
        super().__init__(
            rollout_ctl=rollout_ctl,
            hf_checkpoint=hf_checkpoint,
            sample_params=sample_params,
            judger=judger,
            logger=logger,
        )
        self.context_length = context_length
        self.system_message = system_message
        self.tools = tools
        self.tool_choice = tool_choice
        self._api_server_url: str | None = None
        self._model_name: str | None = None

    def init_agent(self):
        metadata = ray.get(self.rollout_ctl.get_rollout_metadata.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
        self._api_server_url = metadata["api_server_url"]
        self._model_name = getattr(metadata.get("rollout_config"), "model_name", None) or "rollout-controller"

        client = AsyncOpenAI(
            base_url=f"{self._api_server_url.rstrip('/')}/v1",
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=180.0,
            max_retries=3,
        )
        model = OpenAICompatibleModel(
            model_type=self._model_name,
            client=client,
            async_client=client,
            model_config_dict=self._build_camel_model_config(copy.deepcopy(self.sample_params)),
            token_counter=XTunerCamelTokenCounter(self.tokenizer),
        )
        return ChatAgent(
            system_message=self.system_message,
            model=model,
            token_limit=self.context_length,
            step_timeout=180.0,
            tools=self.tools,
        )

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> list[RolloutState]:
        try:
            agent = self.init_agent()
            messages = copy.deepcopy(rollout_state.message)
            turn_rollout_states: list[RolloutState] = []
            collector_token = set_current_trace_collector(turn_rollout_states)
            try:
                await agent.astep(messages[-1]["content"])
            finally:
                reset_current_trace_collector(collector_token)

            if not turn_rollout_states:
                raise RuntimeError("no rollout states captured from gateway")

            normalized_rollout_states: list[RolloutState] = []
            trace_records: list[dict[str, Any]] = []
            for turn_index, turn_state in enumerate(turn_rollout_states):
                if turn_state.prompt_ids is None:
                    raise RuntimeError(f"captured Camel rollout turn {turn_index} is missing prompt_ids")
                if turn_state.response_ids is None:
                    raise RuntimeError(f"captured Camel rollout turn {turn_index} is missing response_ids")
                normalized_turn_state = turn_state.model_copy(deep=True)
                normalized_turn_state.message_uid = rollout_state.message_uid
                normalized_turn_state.message = copy.deepcopy(rollout_state.message)
                normalized_turn_state.data_source = copy.deepcopy(rollout_state.data_source)
                normalized_turn_state.mm_info = copy.deepcopy(rollout_state.mm_info)
                normalized_turn_state.reward_model = copy.deepcopy(rollout_state.reward_model)
                normalized_turn_state.sample_params = copy.deepcopy(rollout_state.sample_params)
                normalized_turn_state.task_name = rollout_state.task_name
                normalized_turn_state.seq_staleness = rollout_state.seq_staleness
                normalized_turn_state.extra_fields = {
                    **copy.deepcopy(rollout_state.extra_fields),
                    **copy.deepcopy(normalized_turn_state.extra_fields),
                    "gateway_rollout_index": turn_index,
                }
                normalized_rollout_states.append(normalized_turn_state)
                trace_records.append(
                    {
                        "request_id": str(normalized_turn_state.uid) if normalized_turn_state.uid is not None else None,
                        "prompt_ids": list(normalized_turn_state.prompt_ids),
                        "response_ids": list(normalized_turn_state.response_ids),
                        "logprobs": None if normalized_turn_state.logprobs is None else list(normalized_turn_state.logprobs),
                        "routed_experts": normalized_turn_state.routed_experts,
                        "finish_reason": normalized_turn_state.finish_reason,
                        "status": normalized_turn_state.status,
                    }
                )
            for normalized_turn_state in normalized_rollout_states:
                normalized_turn_state.extra_fields["gateway_trace_records"] = copy.deepcopy(trace_records)
            return normalized_rollout_states
        except Exception as exc:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = f"Camel agent loop failed: {exc}"
            return [rollout_state]

    def _build_camel_model_config(self, sample_params: SampleParams) -> dict:
        model_config = {
            "temperature": sample_params.temperature,
            "top_p": sample_params.top_p,
            "max_tokens": sample_params.max_tokens,
            "stream": False,
        }
        if self.tool_choice is not None:
            model_config["tool_choice"] = copy.deepcopy(self.tool_choice)
        if sample_params.presence_penalty:
            model_config["presence_penalty"] = sample_params.presence_penalty
        if sample_params.frequency_penalty:
            model_config["frequency_penalty"] = sample_params.frequency_penalty
        if sample_params.stops:
            model_config["stop"] = sample_params.stops
        return model_config

    def _extract_finish_reason(self, response) -> str:
        reasons = response.info.get("termination_reasons", []) if getattr(response, "info", None) else []
        if reasons and reasons[-1] in {"stop", "length", "tool_calls"}:
            return reasons[-1]
        return "stop"
