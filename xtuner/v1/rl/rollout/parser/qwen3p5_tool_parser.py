from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from xtuner.v1.data_proto.rl_data import RolloutToolCall

from .tool_parser import ParsedToolCallResult, ToolCallParser, build_rollout_tool_call, coerce_parameter_value


class Qwen3p5ToolCallParser(ToolCallParser):
    _tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
    _parameter_pattern = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)

    def parse_text(self, text: str) -> ParsedToolCallResult:
        if not text:
            return ParsedToolCallResult()
        cleaned_text, tool_calls = self._extract_tool_call_tags(text)
        return ParsedToolCallResult(remaining_text=cleaned_text.strip(), tool_calls=tool_calls)

    def should_reject_unparsed_markup(
        self,
        *,
        has_tools: bool,
        text: str | None,
        parsed_tool_calls: list[Any] | None,
    ) -> bool:
        if not has_tools:
            return False
        if parsed_tool_calls:
            return False
        if not text:
            return False
        return any(marker in text for marker in ("<tool_call>", "</tool_call>", "<function=", "<parameter="))

    def _extract_tool_call_tags(self, text: str) -> tuple[str, list[RolloutToolCall]]:
        tool_calls: list[RolloutToolCall] = []
        text_parts: list[str] = []
        last_end = 0

        for match in self._tool_call_pattern.finditer(text):
            if match.start() > last_end:
                text_parts.append(text[last_end : match.start()])
            parsed_tool_call = self._parse_single_tool_call(match.group(1).strip())
            if parsed_tool_call is None:
                text_parts.append(match.group(0))
            else:
                tool_calls.append(parsed_tool_call)
            last_end = match.end()

        if last_end < len(text):
            text_parts.append(text[last_end:])
        return "".join(text_parts), tool_calls

    def _parse_single_tool_call(self, raw_payload: str) -> RolloutToolCall | None:
        function_name = self._extract_function_name(raw_payload)
        if not function_name:
            return None

        arguments: dict[str, Any] = {}
        for parameter_match in self._parameter_pattern.finditer(raw_payload):
            parameter_name = parameter_match.group(1).strip()
            parameter_value = parameter_match.group(2).strip()
            arguments[parameter_name] = coerce_parameter_value(parameter_value)

        return build_rollout_tool_call(
            name=function_name,
            arguments=arguments,
            call_id=f"call_{uuid4().hex}",
        )

    def _extract_function_name(self, raw_payload: str) -> str | None:
        function_start = raw_payload.find("<function=")
        if function_start == -1:
            return None

        name_start = function_start + len("<function=")
        terminators = [
            index for index in (raw_payload.find(">", name_start), raw_payload.find("\n", name_start)) if index != -1
        ]
        if not terminators:
            return None

        function_name = raw_payload[name_start : min(terminators)].strip()
        return function_name or None
