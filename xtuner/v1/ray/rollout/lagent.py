import ast
import json
import re
from typing import Any, Dict, List, Optional

import ray
from ray.actor import ActorClass

from xtuner.v1.data_proto.messages.agent import AgentMessage
from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, SampleParams
from xtuner.v1.datasets.multiturn import tokenize
from xtuner.v1.ray.config.worker import RolloutConfig

from .controller import RolloutController


class TokenReasonParser:
    def __init__(self, tokenizer_path: str, resoning_token=dict(start='<think>', end='</think>')):
        self.start = resoning_token.get('start', '<think>')
        self.end = resoning_token.get('end', '</think>')

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def parse_response(self, data: AgentMessage) -> AgentMessage:
        think, content = '', data.content or ''
        thinking_start_idx = thinking_end_idx = -1
        if self.end in data.content:
            think, content = data.content.rsplit(self.end, 1)
            if self.start in think:
                think = think.split(self.start, 1)[-1]
            else:
                thinking_start_idx = 0
        data.thinking = think.strip()
        data.content = content.strip()
        thinking_ids = []
        thinking_logprobs = []
        content_ids = data.content_ids or []
        content_logprobs = data.content_logprobs or []
        start_token_ids = self.tokenizer.encode(self.start, add_special_tokens=False)
        end_token_ids = self.tokenizer.encode(self.end, add_special_tokens=False)
        # find fisrt start_token_ids and last end_token_ids in content_ids
        # thinking ids  should contain start_token_ids and end_token_ids
        for i in range(len(content_ids) - len(start_token_ids) + 1):
            if content_ids[i : i + len(start_token_ids)] == start_token_ids:
                thinking_start_idx = i
                break
        for i in range(len(content_ids) - len(end_token_ids), -1, -1):
            if content_ids[i : i + len(end_token_ids)] == end_token_ids:
                thinking_end_idx = i + len(end_token_ids)
                break
        if thinking_start_idx != -1 and thinking_end_idx != -1 and thinking_end_idx > thinking_start_idx:
            thinking_ids = content_ids[thinking_start_idx:thinking_end_idx]
            thinking_logprobs = content_logprobs[thinking_start_idx:thinking_end_idx]
            data.thinking_ids = thinking_ids
            data.thinking_logprobs = thinking_logprobs
            # remove thinking ids from content ids and logprobs
            data.content_ids = content_ids[:thinking_start_idx] + content_ids[thinking_end_idx:]
            data.content_logprobs = content_logprobs[:thinking_start_idx] + content_logprobs[thinking_end_idx:]
        return data


class FunctionCallParser:
    def parse_response(self, data: AgentMessage) -> AgentMessage:
        matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', data.content, flags=re.DOTALL)
        tool_calls, error_message = [], None
        for m in matches:
            tool_call = None
            try:
                tool_call = json.loads(m)
            except json.JSONDecodeError as json_err:
                try:
                    tool_call = ast.literal_eval(m)
                except (SyntaxError, ValueError) as eval_err:
                    error_message = (
                        f"JSON parsing failed with both json.loads and ast.literal_eval:\n"
                        f"- JSON Decode Error: {json_err}\n"
                        f"- Fallback Syntax/Value Error: {eval_err}\n"
                        f"- Problematic JSON text: {m}"
                    )
                    continue
            if tool_call is not None:
                tool_calls.append(tool_call)

        if tool_calls:
            data.tool_calls = tool_calls
        if error_message:
            if isinstance(data.extra_info, dict):
                data.extra_info.update({'parse_tool_call_error': error_message})
            else:
                data.extra_info = {'parse_tool_call_error': error_message}
        return data


class ControllerWrapper:
    def __init__(
        self,
        placement_group: Optional[Any] = None,
        rollout_cfg: Optional[RolloutConfig] = None,
        rollout_controller: Optional[ActorClass] = None,
        sample_params: Optional[SampleParams] = None,
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
        self.reasoning_parser = TokenReasonParser(self.rollout_cfg.tokenizer_path)
        self.tool_call_parser = FunctionCallParser()

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
