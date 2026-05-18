from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from xtuner.v1.data_proto.rl_data import RolloutToolCall

from .tool_parser import (
    ParsedToolCallResult,
    ToolCallParser,
    build_rollout_tool_call,
    coerce_parameter_value,
    parse_json_or_python_mapping,
)


class Qwen3ToolCallParser(ToolCallParser):
    _tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
    _qwen_function_pattern = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
    _qwen_parameter_pattern = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
    _xml_tag_pattern = re.compile(r"<([a-zA-Z_][^>\n/]*)>(.*?)</\1>", re.DOTALL)

    def parse_text(self, text: str) -> ParsedToolCallResult:
        if not text:
            return ParsedToolCallResult()
        cleaned_text, tool_calls = self._extract_tool_call_tags(text)
        cleaned_text, qwen_tool_calls = self._extract_qwen_function_calls(cleaned_text)
        tool_calls.extend(qwen_tool_calls)
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
            parsed_tool_call = self._parse_single_textual_tool_call(match.group(1).strip())
            if parsed_tool_call is None:
                text_parts.append(match.group(0))
            else:
                tool_calls.append(parsed_tool_call)
            last_end = match.end()
        if last_end < len(text):
            text_parts.append(text[last_end:])
        return "".join(text_parts), tool_calls

    def _extract_qwen_function_calls(self, text: str) -> tuple[str, list[RolloutToolCall]]:
        tool_calls: list[RolloutToolCall] = []
        text_parts: list[str] = []
        last_end = 0
        for match in self._qwen_function_pattern.finditer(text):
            if match.start() > last_end:
                text_parts.append(text[last_end : match.start()])
            parsed_tool_call = self._parse_qwen_function_call(match.group(1).strip(), match.group(2))
            if parsed_tool_call is None:
                text_parts.append(match.group(0))
            else:
                tool_calls.append(parsed_tool_call)
            last_end = match.end()
        if last_end < len(text):
            text_parts.append(text[last_end:])
        return "".join(text_parts), tool_calls

    def _parse_single_textual_tool_call(self, raw_payload: str) -> RolloutToolCall | None:
        payload = parse_json_or_python_mapping(raw_payload)
        if isinstance(payload, dict) and payload.get("name"):
            arguments = payload.get("arguments", payload.get("parameters", {}))
            return build_rollout_tool_call(
                name=str(payload["name"]),
                arguments=arguments,
                call_id=str(payload.get("id") or f"call_{uuid4().hex}"),
            )
        function_match = self._qwen_function_pattern.search(raw_payload)
        if function_match is None:
            return None
        return self._parse_qwen_function_call(function_match.group(1).strip(), function_match.group(2))

    def _parse_qwen_function_call(self, function_name: str, function_body: str) -> RolloutToolCall | None:
        arguments: dict[str, Any] = {}
        for parameter_match in self._qwen_parameter_pattern.finditer(function_body):
            param_name = parameter_match.group(1).strip()
            param_value = parameter_match.group(2).strip()
            arguments[param_name] = coerce_parameter_value(param_value)
        if not arguments:
            for tag_match in self._xml_tag_pattern.finditer(function_body):
                tag_name = tag_match.group(1).strip()
                if tag_name.startswith("function="):
                    continue
                tag_value = tag_match.group(2).strip()
                if tag_name in {"path", "file_path"}:
                    arguments[tag_name] = tag_value
                else:
                    arguments[tag_name] = coerce_parameter_value(tag_value)
        return build_rollout_tool_call(
            name=function_name,
            arguments=arguments,
            call_id=f"call_{uuid4().hex}",
        )
