from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError, JSONDecoder
from typing import Any
from uuid import uuid4

from xtuner.v1.data_proto import ParsedToolCallResult, RolloutFunctionCall, RolloutState, RolloutToolCall


class ToolCallParser(ABC):
    @abstractmethod
    def parse(self, rollout_state: RolloutState) -> ParsedToolCallResult:
        raise NotImplementedError


class TextToolCallParser(ToolCallParser):
    def parse(self, rollout_state: RolloutState) -> ParsedToolCallResult:
        return self.parse_text(rollout_state.response or "")

    @abstractmethod
    def parse_text(self, text: str) -> ParsedToolCallResult:
        raise NotImplementedError


class JsonToolCallParser(TextToolCallParser):
    _decoder = JSONDecoder()

    def parse_text(self, text: str) -> ParsedToolCallResult:
        if not text:
            return ParsedToolCallResult()
        tool_calls: list[RolloutToolCall] = []
        text_parts: list[str] = []
        last_end = 0
        index = 0
        while index < len(text):
            if text[index] != "{":
                index += 1
                continue
            try:
                payload, end = self._decoder.raw_decode(text[index:])
            except JSONDecodeError:
                index += 1
                continue
            absolute_end = index + end
            tool_call = self._payload_to_tool_call(payload)
            if tool_call is None:
                index += 1
                continue
            if index > last_end:
                text_parts.append(text[last_end:index])
            tool_calls.append(tool_call)
            last_end = absolute_end
            index = absolute_end
        if last_end < len(text):
            text_parts.append(text[last_end:])
        return ParsedToolCallResult(remaining_text="".join(text_parts).strip(), tool_calls=tool_calls)

    def _payload_to_tool_call(self, payload: Any) -> RolloutToolCall | None:
        if not isinstance(payload, dict) or payload.get("type") != "tool_call":
            return None
        function_payload = payload.get("function")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if isinstance(function_payload, dict):
            name = name or function_payload.get("name")
            arguments = arguments if arguments is not None else function_payload.get("arguments")
        if not name:
            return None
        return _build_rollout_tool_call(
            name=str(name),
            arguments=arguments,
            call_id=str(payload.get("id") or f"call_{uuid4().hex}"),
            parsing_mode="json_envelope",
        )


class RegexToolCallParser(TextToolCallParser):
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
        payload = _parse_json_or_python_mapping(raw_payload)
        if isinstance(payload, dict) and payload.get("name"):
            arguments = payload.get("arguments", payload.get("parameters", {}))
            return _build_rollout_tool_call(
                name=str(payload["name"]),
                arguments=arguments,
                call_id=str(payload.get("id") or f"call_{uuid4().hex}"),
                parsing_mode="legacy_regex",
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
            arguments[param_name] = _coerce_parameter_value(param_value)
        if not arguments:
            for tag_match in self._xml_tag_pattern.finditer(function_body):
                tag_name = tag_match.group(1).strip()
                if tag_name.startswith("function="):
                    continue
                tag_value = tag_match.group(2).strip()
                if tag_name in {"path", "file_path"}:
                    arguments[tag_name] = tag_value
                else:
                    arguments[tag_name] = _coerce_parameter_value(tag_value)
        return _build_rollout_tool_call(
            name=function_name,
            arguments=arguments,
            call_id=f"call_{uuid4().hex}",
            parsing_mode="legacy_regex",
        )


class CompositeToolCallParser(ToolCallParser):
    def __init__(self, parsers: list[TextToolCallParser]):
        self._parsers = parsers

    def parse(self, rollout_state: RolloutState) -> ParsedToolCallResult:
        current_text = rollout_state.response or ""
        tool_calls: list[RolloutToolCall] = []
        for parser in self._parsers:
            parsed = parser.parse_text(current_text)
            current_text = parsed.remaining_text
            tool_calls.extend(parsed.tool_calls)
        return ParsedToolCallResult(remaining_text=current_text, tool_calls=tool_calls)


def _parse_json_or_python_mapping(raw_payload: str) -> Any:
    try:
        return json.loads(raw_payload)
    except Exception:
        try:
            return ast.literal_eval(raw_payload)
        except Exception:
            return None


def _coerce_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except Exception:
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return stripped


def _build_rollout_tool_call(
    *,
    name: str,
    arguments: Any,
    call_id: str,
    parsing_mode: str,
) -> RolloutToolCall:
    raw_arguments_text = arguments if isinstance(arguments, str) else None
    parsed_arguments = arguments
    if isinstance(arguments, str):
        decoded = _parse_json_or_python_mapping(arguments)
        parsed_arguments = decoded if decoded is not None else {"raw": arguments}
    return RolloutToolCall(
        id=call_id,
        function=RolloutFunctionCall(
            name=name,
            arguments=parsed_arguments,
            raw_arguments_text=raw_arguments_text,
        ),
        parsing_mode=parsing_mode,
    )
