import ast
import json
import re
from typing import Protocol

from xtuner.v1.ray.environment.lagent.schema import AgentMessage


class ResponseParser(Protocol):
    """Protocol for agent response parsers."""

    def parse_response(self, data: AgentMessage) -> AgentMessage: ...


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
            data.extra_info['parse_tool_call_error'] = error_message
        return data


class XMLFunctionCallParser:
    def parse_response(self, data: AgentMessage) -> AgentMessage:
        tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', data.content, flags=re.DOTALL)
        tool_calls, error_message = [], None

        for block in tool_call_blocks:
            func_match = re.search(r'<function=([^>]+)>(.*?)</function>', block, flags=re.DOTALL)
            if not func_match:
                error_message = "Could not find a valid <function=name>...</function> block inside <tool_call>."
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            param_matches = re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', func_body, flags=re.DOTALL)
            parameters = {}
            for p_match in param_matches:
                p_name = p_match.group(1).strip()
                p_value = p_match.group(2).strip()
                try:
                    parsed_value = ast.literal_eval(p_value)
                except (ValueError, SyntaxError):
                    parsed_value = p_value
                parameters[p_name] = parsed_value

            tool_calls.append({"name": func_name, "arguments": parameters})

        if tool_calls:
            data.tool_calls = tool_calls
        if error_message:
            data.extra_info['parse_tool_call_error'] = error_message
        return data
