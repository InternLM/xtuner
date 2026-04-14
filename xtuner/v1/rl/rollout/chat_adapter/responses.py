from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status

from .base import BaseChatAPIAdapter
from .openai import OpenAIChatAdapterError
from .trace import normalize_trace_payload


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_uid: int | None = None
    model: str | None = None
    instructions: str | None = None
    input: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    store: bool = False
    parallel_tool_calls: bool | None = None
    include: list[Any] | None = None
    reasoning: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class ResponsesUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed"] = "completed"
    model: str
    output: list[dict[str, Any]]
    output_text: str = ""
    parallel_tool_calls: bool = False
    store: bool = False
    text: dict[str, Any] = {"format": {"type": "text"}}
    usage: ResponsesUsage


class OpenAIResponsesAdapter(BaseChatAPIAdapter[ResponsesRequest, ResponsesResponse]):
    _tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
    _qwen_function_pattern = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
    _qwen_parameter_pattern = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
    _xml_tag_pattern = re.compile(r"<([a-zA-Z_][^>\n/]*)>(.*?)</\1>", re.DOTALL)
    _disabled_tool_names = {
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "request_user_input",
    }
    _tool_name_aliases = {
        "read_file_dd": "exec_command",
        "read_file": "exec_command",
        "readfile": "exec_command",
    }

    def __init__(
        self,
        generate_handler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str | None,
        default_model_name: str | None = None,
        context_length: int | None = None,
        capture_path: str | None = None,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(generate_handler, tokenizer=tokenizer, capture_path=capture_path)
        self._default_model_name = default_model_name
        self._context_length = context_length

    async def responses(self, request: ResponsesRequest) -> ResponsesResponse:
        return await self.handle_request(request)

    def validate_request(self, request: ResponsesRequest) -> None:
        return None

    def request_to_rollout_state(self, request: ResponsesRequest) -> RolloutState:
        internal_messages = self._build_internal_messages(request)
        tokenizer_tools = self._normalize_tools_for_backend(request.tools)
        normalized_tool_choice = self._normalize_tool_choice_for_backend(request.tool_choice)

        prompt_ids = None
        if self._tokenizer is not None:
            raw_prompt_ids = self._tokenizer.apply_chat_template(
                internal_messages,
                tools=tokenizer_tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = raw_prompt_ids.get("input_ids") if hasattr(raw_prompt_ids, "get") else list(raw_prompt_ids)

        max_tokens = self._fit_max_tokens_to_context(
            prompt_ids=prompt_ids, requested_max_tokens=request.max_output_tokens
        )
        return RolloutState(
            uid=uuid4().int,
            message=internal_messages,
            prompt_ids=prompt_ids,
            tokens=prompt_ids,
            session_uid=request.session_uid,
            tools=tokenizer_tools,
            tool_choice=normalized_tool_choice,
            sample_params=self._build_sample_params(request, max_tokens=max_tokens),
        )

    def raise_for_failed_response(self, response: RolloutState, request_id: str) -> None:
        if response.status == Status.FAILED:
            raise OpenAIChatAdapterError(
                response.error_msg or "Rollout generation failed",
                "server_error",
                "rollout_failed",
                request_id,
            )

    def normalize_request(self, request: ResponsesRequest) -> dict[str, Any]:
        return normalize_trace_payload(request.model_dump(mode="python", exclude_none=True))

    def normalize_response(self, response: ResponsesResponse) -> dict[str, Any]:
        return normalize_trace_payload(response.model_dump(mode="python", exclude_none=True))

    def rollout_state_to_response(
        self,
        rollout_state: RolloutState,
        request: ResponsesRequest,
    ) -> ResponsesResponse:
        request_id = str(rollout_state.uid)
        model_name = request.model or self._default_model_name or "rollout-controller"
        prompt_tokens = self._count_prompt_tokens(rollout_state)
        completion_tokens = self._count_completion_tokens(rollout_state)
        output_items, output_text = self._build_response_output_items(rollout_state)
        return ResponsesResponse(
            id=f"resp_{request_id}",
            created_at=int(time.time()),
            model=model_name,
            output=output_items,
            output_text=output_text,
            parallel_tool_calls=bool(request.parallel_tool_calls),
            store=bool(request.store),
            usage=ResponsesUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def build_output_message_list(
        self,
        rollout_state: RolloutState,
        request: ResponsesRequest,
    ) -> list[dict[str, Any]]:
        output_items, output_text = self._build_response_output_items(rollout_state)
        content: list[dict[str, Any]] = []
        if output_text:
            content.append({"type": "text", "text": output_text})
        for item in output_items:
            if item.get("type") == "function_call":
                try:
                    input_payload = json.loads(item.get("arguments", "{}"))
                except Exception:
                    input_payload = {"raw": item.get("arguments", "")}
                content.append({"type": "tool_use", "name": item.get("name", ""), "input": input_payload})
        return [{"role": "assistant", "content": content}]

    def _build_internal_messages(self, request: ResponsesRequest) -> list[dict[str, Any]]:
        system_chunks: list[str] = []
        messages: list[dict[str, Any]] = []
        tool_name_by_call_id: dict[str, str] = {}
        if request.instructions:
            system_chunks.append(request.instructions)

        if request.input is None:
            return self._prepend_system_message(messages, system_chunks)
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
            return self._prepend_system_message(messages, system_chunks)

        for item in request.input:
            item_type = item.get("type", "message")
            if item_type == "message":
                role = self._normalize_input_role(item.get("role"))
                if role == "system":
                    system_text = self._extract_message_item_text(item.get("content"))
                    if system_text:
                        system_chunks.append(system_text)
                else:
                    messages.extend(self._convert_message_item_to_backend_messages(role, item.get("content")))
            elif item_type == "function_call":
                call_id = item.get("call_id") or f"call_{uuid4().hex}"
                tool_name_by_call_id[call_id] = str(item.get("name", ""))
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": str(item.get("name", "")),
                                    "arguments": self._parse_json_string_or_mapping(item.get("arguments")),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                call_id = item.get("call_id")
                messages.append(
                    {
                        "role": "tool",
                        "content": self._serialize_tool_output(
                            item.get("output"),
                            tool_name=tool_name_by_call_id.get(str(call_id or "")),
                        ),
                        "tool_call_id": call_id,
                    }
                )
            elif item_type == "reasoning":
                continue
        return self._prepend_system_message(messages, system_chunks)

    def _prepend_system_message(
        self,
        messages: list[dict[str, Any]],
        system_chunks: list[str],
    ) -> list[dict[str, Any]]:
        joined_system = "\n\n".join(chunk.strip() for chunk in system_chunks if chunk and chunk.strip())
        if not joined_system:
            return messages
        return [{"role": "system", "content": joined_system}, *messages]

    def _normalize_input_role(self, role: Any) -> str:
        if role in {"developer", "system"}:
            return "system"
        if role in {"assistant", "tool"}:
            return str(role)
        return "user"

    def _convert_message_item_to_backend_messages(self, role: str, content: Any) -> list[dict[str, Any]]:
        text = self._extract_message_item_text(content)
        if not text:
            return []
        if role == "assistant":
            text = self._sanitize_assistant_text(text)
        return [{"role": role, "content": text}]

    def _extract_message_item_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        text_chunks: list[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type in {"input_text", "output_text", "text", "summary_text", "reasoning_text"}:
                text_chunks.append(str(part.get("text", "")))
        return "\n".join(chunk for chunk in text_chunks if chunk)

    def _serialize_tool_output(self, output: Any, tool_name: str | None = None) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return self._sanitize_tool_output_text(output, tool_name=tool_name)
        if isinstance(output, list):
            text_chunks = [str(part.get("text", "")) for part in output if isinstance(part, dict) and "text" in part]
            if text_chunks:
                return self._sanitize_tool_output_text("\n".join(text_chunks), tool_name=tool_name)
            return json.dumps(output, ensure_ascii=False)
        if isinstance(output, dict):
            return json.dumps(output, ensure_ascii=False)
        return str(output)

    def _sanitize_tool_output_text(self, text: str, tool_name: str | None = None) -> str:
        if tool_name not in {"exec_command", "write_stdin"}:
            return text

        marker = "\nOutput:\n"
        if marker in text:
            prefix, body = text.split(marker, 1)
            exit_code = self._extract_exec_exit_code(prefix)
            body = body.strip()
            if exit_code is None:
                return body
            if body:
                return f"[exit_code={exit_code}]\n{body}"
            return f"[exit_code={exit_code}]"
        return text

    def _extract_exec_exit_code(self, text: str) -> int | None:
        match = re.search(r"Process exited with code (\d+)", text)
        if match is not None:
            return int(match.group(1))
        return None

    def _normalize_tools_for_backend(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        normalized_tools = []
        for tool in tools:
            tool_name = tool.get("name")
            if tool_name in self._disabled_tool_names:
                continue
            if tool.get("type") != "function":
                continue
            normalized_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
            )
        return normalize_trace_payload(normalized_tools) or None

    def _normalize_tool_choice_for_backend(self, tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        if tool_choice.get("type") == "function":
            return {"type": "function", "function": {"name": tool_choice.get("name")}}
        return normalize_trace_payload(tool_choice)

    def _build_response_output_items(self, rollout_state: RolloutState) -> tuple[list[dict[str, Any]], str]:
        tool_calls = rollout_state.extra_fields.get("tool_calls")
        response_text = rollout_state.response or ""
        if not tool_calls:
            text_blocks, parsed_tool_calls = self._parse_textual_tool_calls(response_text)
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls
                rollout_state.extra_fields["tool_calls"] = parsed_tool_calls
                response_text = "".join(block["text"] for block in text_blocks if block["type"] == "text")

        response_text = self._sanitize_assistant_text(response_text)

        output_items: list[dict[str, Any]] = []
        if response_text:
            output_items.append(
                {
                    "id": f"msg_{uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response_text, "annotations": []}],
                }
            )

        for tool_call in tool_calls or []:
            call_id = tool_call.get("id") or f"call_{uuid4().hex}"
            output_items.append(
                {
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": tool_call["function"]["name"],
                    "arguments": self._stringify_tool_arguments(tool_call["function"].get("arguments")),
                }
            )
        return output_items, response_text

    def _parse_textual_tool_calls(self, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not text:
            return [], []
        content_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        last_end = 0
        for match in self._tool_call_pattern.finditer(text):
            if match.start() > last_end:
                content_blocks.append({"type": "text", "text": text[last_end : match.start()]})
            raw_payload = match.group(1).strip()
            parsed_tool_call = self._parse_single_textual_tool_call(raw_payload)
            if parsed_tool_call is not None:
                tool_calls.append(parsed_tool_call)
            else:
                content_blocks.append({"type": "text", "text": match.group(0)})
            last_end = match.end()
        if last_end < len(text):
            content_blocks.append({"type": "text", "text": text[last_end:]})
        return content_blocks, tool_calls

    def _parse_single_textual_tool_call(self, raw_payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw_payload)
            return {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False),
                },
            }
        except Exception:
            pass

        function_match = self._qwen_function_pattern.search(raw_payload)
        if function_match is None:
            return None
        function_name = function_match.group(1).strip()
        function_body = function_match.group(2)
        arguments: dict[str, Any] = {}
        for parameter_match in self._qwen_parameter_pattern.finditer(function_body):
            param_name = parameter_match.group(1).strip()
            param_value = parameter_match.group(2).strip()
            arguments[param_name] = self._coerce_parameter_value(param_value)
        if not arguments:
            for tag_match in self._xml_tag_pattern.finditer(function_body):
                tag_name = tag_match.group(1).strip()
                if tag_name.startswith("function="):
                    continue
                tag_value = tag_match.group(2).strip()
                if tag_name in {"path", "file_path"}:
                    arguments[tag_name] = tag_value
                else:
                    arguments[tag_name] = self._coerce_parameter_value(tag_value)

        function_name, arguments = self._normalize_tool_call(function_name, arguments)
        return {
            "id": f"call_{uuid4().hex}",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }

    def _stringify_tool_arguments(self, arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments or {}, ensure_ascii=False)

    def _parse_json_string_or_mapping(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {"raw": value}
        return value or {}

    def _coerce_parameter_value(self, value: str) -> Any:
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except Exception:
            return stripped

    def _normalize_tool_call(self, function_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        normalized_name = self._tool_name_aliases.get(function_name, function_name)
        normalized_arguments = dict(arguments)
        if normalized_name == "exec_command" and function_name in self._tool_name_aliases:
            path = normalized_arguments.pop("path", None) or normalized_arguments.pop("file_path", None)
            if path:
                quoted_path = shlex.quote(str(path))
                normalized_arguments = {"cmd": f"cat {quoted_path}"}
        return normalized_name, normalized_arguments

    def _sanitize_assistant_text(self, text: str) -> str:
        cleaned = text.replace("<|im_end|>", "")
        cleaned = cleaned.replace("<think>", "")
        cleaned = cleaned.replace("</think>", "")
        return cleaned.strip()

    def _build_sample_params(self, request: ResponsesRequest, max_tokens: int | None = None) -> SampleParams:
        kwargs = {
            "return_token_ids": True,
            "return_logprob": False,
            "stream": request.stream,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else request.max_output_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        return SampleParams(**kwargs)

    def _fit_max_tokens_to_context(self, prompt_ids: list[int] | None, requested_max_tokens: int | None) -> int | None:
        if self._context_length is None or prompt_ids is None or requested_max_tokens is None:
            return requested_max_tokens
        prompt_tokens = len(prompt_ids)
        available_completion_tokens = self._context_length - prompt_tokens
        if available_completion_tokens <= 0:
            raise OpenAIChatAdapterError(
                (
                    f"Input is too long for this model deployment: prompt_tokens={prompt_tokens}, "
                    f"context_length={self._context_length}."
                ),
                "invalid_request_error",
                "context_length_exceeded",
            )
        return min(requested_max_tokens, available_completion_tokens)

    def _count_prompt_tokens(self, rollout_state: RolloutState) -> int:
        if rollout_state.tokens is not None:
            return len(rollout_state.tokens)
        if rollout_state.prompt_ids is not None:
            return len(rollout_state.prompt_ids)
        return 0

    def _count_completion_tokens(self, rollout_state: RolloutState) -> int:
        if rollout_state.response_ids is not None:
            return len(rollout_state.response_ids)
        if self._tokenizer is not None and rollout_state.response:
            return len(self._tokenizer(rollout_state.response, add_special_tokens=False)["input_ids"])
        return 0


def bind_openai_responses_interface(
    rollout_controller: Any,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None,
    default_model_name: str | None = None,
) -> Any:
    if getattr(rollout_controller, "openai_responses_adapter", None) is None:
        rollout_controller.openai_responses_adapter = OpenAIResponsesAdapter(
            rollout_controller.generate,
            tokenizer=tokenizer,
            default_model_name=default_model_name,
            context_length=getattr(rollout_controller.config, "context_length", None),
            capture_path=str(getattr(rollout_controller.config, "worker_log_dir", ".")) + "/gateway_capture.jsonl",
        )
    rollout_controller.responses = rollout_controller.openai_responses_adapter.responses
    return rollout_controller
