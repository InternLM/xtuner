from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..core.models import (
    CanonicalGenerateRequest,
    CanonicalGenerateResponse,
    CanonicalMessage,
    CanonicalReasoning,
    CanonicalReasoningBlock,
    CanonicalReasoningStep,
    CanonicalTextBlock,
    CanonicalToolCall,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolDefinition,
    CanonicalToolResult,
    CanonicalToolResultBlock,
)
from .base import BaseChatAPIAdapter, stringify_tool_arguments
from .openai import OpenAIChatAdapterError
from .streaming import build_sse_response, encode_sse_event
from .trace import ChatTraceStore, normalize_trace_payload


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
    _disabled_tool_names = {
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
        "request_user_input",
    }

    def __init__(
        self,
        generate_handler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str | None,
        default_model_name: str | None = None,
        context_length: int | None = None,
        capture_folder: str | None = None,
        trace_store: ChatTraceStore | None = None,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(generate_handler, tokenizer=tokenizer, capture_folder=capture_folder, trace_store=trace_store)
        self._default_model_name = default_model_name
        self._context_length = context_length

    async def responses(
        self,
        request: ResponsesRequest,
        *,
        api_key: str | None = None,
    ) -> ResponsesResponse | StreamingResponse:
        if request.stream:
            response = await self.handle_request(request, api_key=api_key)
            return build_sse_response(self.iter_stream_events(response))
        return await self.handle_request(request, api_key=api_key)

    def validate_request(self, request: ResponsesRequest) -> None:
        return None

    def request_to_canonical_request(self, request: ResponsesRequest) -> CanonicalGenerateRequest:
        return CanonicalGenerateRequest(
            request_id=f"responses_req_{uuid4().hex}",
            model=request.model or self._default_model_name or "rollout-controller",
            messages=self._responses_input_to_canonical_messages(request),
            tools=self._responses_tools_to_canonical(request.tools),
            tool_choice=self._responses_tool_choice_to_canonical(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_output_tokens,
            stream=False,
            metadata={
                key: value
                for key, value in {
                    "source_protocol": "openai_responses",
                    "client_stream": bool(request.stream),
                    "session_uid": request.session_uid,
                    "store": request.store,
                    "include": request.include,
                    "reasoning": request.reasoning,
                }.items()
                if value is not None
            },
        )

    def normalize_request(self, request: ResponsesRequest) -> dict[str, Any]:
        return normalize_trace_payload(request.model_dump(mode="python", exclude_none=True))

    def normalize_response(self, response: ResponsesResponse) -> dict[str, Any]:
        return normalize_trace_payload(response.model_dump(mode="python", exclude_none=True))

    async def iter_stream_events(
        self,
        response: ResponsesResponse,
    ) -> AsyncIterator[str]:
        created_response = response.model_dump(mode="json", exclude_none=True)
        created_response["status"] = "in_progress"

        yield encode_sse_event(
            {
                "type": "response.created",
                "response": created_response,
            },
            event="response.created",
        )
        yield encode_sse_event(
            {
                "type": "response.in_progress",
                "response": created_response,
            },
            event="response.in_progress",
        )

        for output_index, item in enumerate(response.output):
            yield encode_sse_event(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                },
                event="response.output_item.added",
            )

            if item.get("type") == "message":
                for content_index, part in enumerate(item.get("content", [])):
                    yield encode_sse_event(
                        {
                            "type": "response.content_part.added",
                            "output_index": output_index,
                            "content_index": content_index,
                            "item_id": item.get("id"),
                            "part": part,
                        },
                        event="response.content_part.added",
                    )
                    if part.get("type") == "output_text":
                        yield encode_sse_event(
                            {
                                "type": "response.output_text.delta",
                                "output_index": output_index,
                                "content_index": content_index,
                                "item_id": item.get("id"),
                                "delta": part.get("text", ""),
                            },
                            event="response.output_text.delta",
                        )
                        yield encode_sse_event(
                            {
                                "type": "response.output_text.done",
                                "output_index": output_index,
                                "content_index": content_index,
                                "item_id": item.get("id"),
                                "text": part.get("text", ""),
                            },
                            event="response.output_text.done",
                        )
                    yield encode_sse_event(
                        {
                            "type": "response.content_part.done",
                            "output_index": output_index,
                            "content_index": content_index,
                            "item_id": item.get("id"),
                            "part": part,
                        },
                        event="response.content_part.done",
                    )

            if item.get("type") == "function_call":
                yield encode_sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": item.get("id"),
                        "delta": item.get("arguments", ""),
                    },
                    event="response.function_call_arguments.delta",
                )
                yield encode_sse_event(
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": output_index,
                        "item_id": item.get("id"),
                        "arguments": item.get("arguments", ""),
                    },
                    event="response.function_call_arguments.done",
                )

            yield encode_sse_event(
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                },
                event="response.output_item.done",
            )

        yield encode_sse_event(
            {
                "type": "response.completed",
                "response": response.model_dump(mode="json", exclude_none=True),
            },
            event="response.completed",
        )

    def canonical_response_to_protocol_response(
        self,
        canonical_response: CanonicalGenerateResponse,
        request: ResponsesRequest,
    ) -> ResponsesResponse:
        output_items = self._canonical_response_to_responses_output_items(canonical_response)
        output_text = "".join(
            block.text for block in canonical_response.output.content if isinstance(block, CanonicalTextBlock)
        ).strip()
        return ResponsesResponse(
            id=f"resp_{canonical_response.request_id}",
            created_at=int(time.time()),
            model=canonical_response.model or self._default_model_name or "rollout-controller",
            output=output_items,
            output_text=output_text,
            parallel_tool_calls=bool(
                request.parallel_tool_calls
                if request is not None
                else canonical_response.metadata.get("parallel_tool_calls")
            ),
            store=bool(request.store) if request is not None else False,
            usage=ResponsesUsage(
                input_tokens=canonical_response.usage.prompt_tokens,
                output_tokens=canonical_response.usage.completion_tokens,
                total_tokens=canonical_response.usage.total_tokens,
            ),
        )

    def _normalize_input_role(self, role: Any) -> str:
        if role in {"developer", "system"}:
            return "system"
        if role in {"assistant", "tool"}:
            return str(role)
        return "user"

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

    def _responses_input_to_canonical_messages(self, request: ResponsesRequest) -> list[CanonicalMessage]:
        messages: list[CanonicalMessage] = []
        if request.instructions:
            messages.append(
                CanonicalMessage(
                    role="system",
                    content=[CanonicalTextBlock(text=request.instructions)],
                    metadata={"source_protocol": "openai_responses"},
                )
            )
        if request.input is None:
            return messages
        if isinstance(request.input, str):
            messages.append(
                CanonicalMessage(
                    role="user",
                    content=[CanonicalTextBlock(text=request.input)],
                    metadata={"source_protocol": "openai_responses"},
                )
            )
            return messages

        tool_name_by_call_id: dict[str, str] = {}
        for item in request.input:
            item_type = item.get("type", "message")
            if item_type == "message":
                role = self._normalize_input_role(item.get("role"))
                content_blocks = self._responses_message_content_to_canonical(item.get("content"))
                messages.append(
                    CanonicalMessage(
                        role=role if role in {"system", "user", "assistant", "tool"} else "user",
                        content=content_blocks,
                        metadata={"source_protocol": "openai_responses"},
                    )
                )
            elif item_type == "function_call":
                call_id = str(item.get("call_id") or f"call_{uuid4().hex}")
                tool_name = str(item.get("name", ""))
                tool_name_by_call_id[call_id] = tool_name
                messages.append(
                    CanonicalMessage(
                        role="assistant",
                        content=[
                            CanonicalToolCallBlock(
                                tool_call=CanonicalToolCall(
                                    id=call_id,
                                    name=tool_name,
                                    arguments=self._parse_json_string_or_mapping(item.get("arguments")),
                                    raw_arguments_text=item.get("arguments")
                                    if isinstance(item.get("arguments"), str)
                                    else None,
                                    metadata={"source_protocol": "openai_responses"},
                                )
                            )
                        ],
                        metadata={"source_protocol": "openai_responses"},
                    )
                )
            elif item_type == "function_call_output":
                call_id = str(item.get("call_id") or "")
                output = item.get("output")
                messages.append(
                    CanonicalMessage(
                        role="tool",
                        content=[
                            CanonicalToolResultBlock(
                                tool_result=CanonicalToolResult(
                                    tool_call_id=call_id,
                                    name=tool_name_by_call_id.get(call_id),
                                    output=output,
                                    output_text=self._serialize_tool_output(
                                        output, tool_name=tool_name_by_call_id.get(call_id)
                                    ),
                                    metadata={"source_protocol": "openai_responses"},
                                )
                            )
                        ],
                        metadata={"source_protocol": "openai_responses"},
                    )
                )
            elif item_type == "reasoning":
                reasoning_text = self._responses_reasoning_item_to_text(item)
                messages.append(
                    CanonicalMessage(
                        role="assistant",
                        content=[
                            CanonicalReasoningBlock(
                                reasoning=CanonicalReasoning(
                                    steps=[CanonicalReasoningStep(text=reasoning_text)] if reasoning_text else [],
                                    metadata={"source_protocol": "openai_responses"},
                                )
                            )
                        ],
                        metadata={"source_protocol": "openai_responses"},
                    )
                )
        return messages

    def _responses_message_content_to_canonical(self, content: Any) -> list[Any]:
        if isinstance(content, str):
            return [CanonicalTextBlock(text=content)] if content else []
        if not isinstance(content, list):
            return [CanonicalTextBlock(text=str(content))]

        blocks: list[Any] = []
        unsupported_types: list[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type in {"input_text", "output_text", "text"}:
                text = str(part.get("text", ""))
                if text:
                    blocks.append(CanonicalTextBlock(text=text))
            elif part_type in {"summary_text", "reasoning_text"}:
                reasoning_text = str(part.get("text", ""))
                if reasoning_text:
                    blocks.append(
                        CanonicalReasoningBlock(
                            reasoning=CanonicalReasoning(
                                steps=[CanonicalReasoningStep(text=reasoning_text)],
                                metadata={"source_protocol": "openai_responses"},
                            )
                        )
                    )
            else:
                unsupported_types.append(str(part_type))
        if unsupported_types:
            unsupported_str = ", ".join(sorted(set(unsupported_types)))
            raise OpenAIChatAdapterError(
                f"Unsupported Responses content block type(s): {unsupported_str}",
                "invalid_request_error",
                "unsupported_content_block",
            )
        return blocks

    def _responses_reasoning_item_to_text(self, item: dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"reasoning_text", "summary_text", "text"}:
                    chunks.append(str(part.get("text", "")))
            if chunks:
                return "\n".join(chunk for chunk in chunks if chunk)
        summary = item.get("summary")
        if isinstance(summary, list):
            chunks = [str(part.get("text", "")) for part in summary if isinstance(part, dict)]
            if chunks:
                return "\n".join(chunk for chunk in chunks if chunk)
        return str(item.get("text", ""))

    def _responses_tools_to_canonical(self, tools: list[dict[str, Any]] | None) -> list[CanonicalToolDefinition]:
        if not tools:
            return []
        canonical_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            tool_name = str(tool.get("name", ""))
            if tool_name in self._disabled_tool_names:
                continue
            canonical_tools.append(
                CanonicalToolDefinition(
                    name=tool_name,
                    description=tool.get("description"),
                    parameters_json_schema=tool.get("parameters", {}),
                    metadata={"source_protocol": "openai_responses"},
                )
            )
        return canonical_tools

    def _responses_tool_choice_to_canonical(
        self, tool_choice: str | dict[str, Any] | None
    ) -> CanonicalToolChoice | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return CanonicalToolChoice(type=tool_choice)
        if tool_choice.get("type") == "function":
            return CanonicalToolChoice(
                type="specific",
                tool_name=tool_choice.get("name"),
                metadata={"source_protocol": "openai_responses"},
            )
        return CanonicalToolChoice(
            type=str(tool_choice.get("type", "auto")),
            metadata={"source_protocol": "openai_responses"},
        )

    def _canonical_response_to_responses_output_items(
        self,
        response: CanonicalGenerateResponse,
    ) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        for block in response.output.content:
            if isinstance(block, CanonicalTextBlock):
                output_items.append(
                    {
                        "id": f"msg_{uuid4().hex}",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": block.text, "annotations": []}],
                    }
                )
            elif isinstance(block, CanonicalToolCallBlock):
                output_items.append(
                    {
                        "id": f"fc_{uuid4().hex}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": block.tool_call.id,
                        "name": block.tool_call.name,
                        "arguments": stringify_tool_arguments(block.tool_call),
                    }
                )
            elif isinstance(block, CanonicalToolResultBlock):
                output_items.append(
                    {
                        "id": f"fco_{uuid4().hex}",
                        "type": "function_call_output",
                        "call_id": block.tool_result.tool_call_id,
                        "output": block.tool_result.output
                        if block.tool_result.output is not None
                        else block.tool_result.output_text,
                    }
                )
            elif isinstance(block, CanonicalReasoningBlock):
                reasoning_text = "\n".join(step.text for step in block.reasoning.steps if step.text).strip()
                if reasoning_text:
                    output_items.append(
                        {
                            "id": f"rs_{uuid4().hex}",
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": reasoning_text}],
                        }
                    )
        return output_items

    def _parse_json_string_or_mapping(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {"raw": value}
        return value or {}
