import base64
from typing import Any, Dict, List, Optional

import ray
from lagent.utils import create_object
from ray import cloudpickle
from ray.actor import ActorClass

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, SampleParams
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.environment.lagent.parsers import (
    Qwen3FunctionCallParser,
    Qwen3TokenReasonParser,
    ResponseParser,
)
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.ray.environment.lagent.tokenize import tokenize
from xtuner.v1.ray.rollout.controller import RolloutController


class ControllerWrapper:
    def __init__(
        self,
        placement_group: Optional[Any] = None,
        rollout_cfg: Optional[RolloutConfig] = None,
        rollout_controller: Optional[ActorClass] = None,
        sample_params: Optional[SampleParams] = None,
        reasoning_parser: Optional[ResponseParser] = None,
        tool_call_parser: Optional[ResponseParser] = None,
        encode_routed_experts_ref: bool = True,
    ):
        assert rollout_controller is not None or (
            placement_group and rollout_cfg
        ), "Either rollout_controller or placement_group and rollout_cfg must be provided."
        if rollout_controller:
            self.rollout_controller = rollout_controller
            self.rollout_cfg = ray.get(rollout_controller.get_rollout_info.remote())["rollout_config"]  # type: ignore[call-overload, attr-defined]
        else:
            self.rollout_controller = RolloutController.remote(rollout_cfg, placement_group)  # type: ignore[attr-defined]
            self.rollout_cfg = rollout_cfg

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.rollout_cfg.tokenizer_path, trust_remote_code=True)
        self.sample_params = sample_params or SampleParams()
        self.encode_routed_experts_ref = encode_routed_experts_ref
        # default parsers
        self.reasoning_parser = (
            reasoning_parser
            and create_object(reasoning_parser)
            or Qwen3TokenReasonParser(self.rollout_cfg.tokenizer_path)
        )
        self.tool_call_parser = tool_call_parser and create_object(tool_call_parser) or Qwen3FunctionCallParser()

    async def chat(self, messages, tools: Optional[List[Dict]] = None, **kwargs):
        sample_params = self.sample_params.model_copy(update=kwargs)
        inputs = tokenize(self.tokenizer, messages, tools)
        if len(inputs["input_ids"]) >= self.rollout_cfg.context_length:
            response = RLRolloutResponseItem(finish_reason="length")
        else:
            response = await self.rollout_controller.rollout.remote(  # type: ignore[no-redef, attr-defined]
                input_ids=inputs["input_ids"],
                sample_params=sample_params,
                extra_info=(
                    {"routed_experts": inputs["routed_experts"]} if inputs["routed_experts"] is not None else {}
                ),
            )
            if (
                response.state == "completed"
                and response.finish_reason != "abort"
                and self.rollout_cfg.enable_return_routed_experts
                and "routed_experts" not in response.extra_info
            ):
                raise ValueError("Routed experts expected in response extra_info but not found.")
            if isinstance(response.extra_info.get('routed_experts'), ray.ObjectRef) and self.encode_routed_experts_ref:
                response.extra_info['routed_experts'] = base64.b64encode(
                    cloudpickle.dumps(response.extra_info['routed_experts'])
                ).decode('utf-8')

        response = AgentMessage.from_model_response(response, "")
        return self.parse_response(response)

    def parse_response(self, response: AgentMessage):
        response = self.reasoning_parser.parse_response(response)
        response = self.tool_call_parser.parse_response(response)
        return response
