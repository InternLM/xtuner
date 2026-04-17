from typing import Any, Dict, List, Optional

import ray
from lagent.utils import create_object
from ray.actor import ActorClass

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, SampleParams
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.environment.lagent.parsers import (
    FunctionCallParser,
    ResponseParser,
    TokenReasonParser,
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
    ):
        assert rollout_controller is not None or (
            placement_group and rollout_cfg
        ), "Either rollout_controller or placement_group and rollout_cfg must be provided."
        if rollout_controller:
            self.rollout_controller = rollout_controller
            self.rollout_cfg = ray.get(rollout_controller.get_rollout_info.remote())['rollout_config']
        else:
            self.rollout_controller = RolloutController.remote(rollout_cfg, placement_group)
            self.rollout_cfg = rollout_cfg

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.rollout_cfg.tokenizer_path, trust_remote_code=True)
        self.sample_params = sample_params or SampleParams()
        # default parsers
        self.reasoning_parser = (
            reasoning_parser and create_object(reasoning_parser) or TokenReasonParser(self.rollout_cfg.tokenizer_path)
        )
        self.tool_call_parser = tool_call_parser and create_object(tool_call_parser) or FunctionCallParser()

    async def chat(self, messages, session_id=None, tools: Optional[List[Dict]] = None, **kwargs):
        sample_params = self.sample_params.model_copy(update=kwargs)
        inputs = tokenize(self.tokenizer, messages, tools)
        if len(inputs['input_ids']) >= self.rollout_cfg.context_length:
            response = RLRolloutResponseItem(finish_reason='length')
        else:
            extra_info = {'action_id': session_id}
            if inputs['routed_experts'] is not None:
                extra_info['routed_experts'] = inputs['routed_experts']
            response: RLRolloutResponseItem = await self.rollout_controller.rollout.remote(
                input_ids=inputs['input_ids'],
                sample_params=sample_params,
                session_id=session_id,
                extra_info=extra_info,
            )
            if (
                response.finish_reason != 'abort'
                and self.rollout_cfg.enable_return_routed_experts
                and 'routed_experts' not in response.extra_info
            ):
                raise ValueError("Routed experts expected in response extra_info but not found.")

        response = AgentMessage.from_model_response(response, '')
        return self.parse_response(response)

    def parse_response(self, response: AgentMessage):
        response = self.reasoning_parser.parse_response(response)
        response = self.tool_call_parser.parse_response(response)
        return response
