import json
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
from .base import BaseChatAPIAdapter
from .streaming import build_sse_response, encode_sse_event
from .trace import ChatTraceStore, normalize_trace_payload


class AnthropicTextContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str


AnthropicContentBlock = dict[str, Any]


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_uid: int | None = None
    model: str | None = None
    system: str | list[dict[str, Any]] | None = None
    messages: list[AnthropicMessage]
    max_tokens: int
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class AnthropicCountTokensRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    system: str | list[dict[str, Any]] | None = None
    messages: list[AnthropicMessage]
    tools: list[dict[str, Any]] | None = None


class AnthropicCountTokensResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int


class AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicChatAdapterError(RuntimeError):
    def __init__(self, message: str, error_type: str, request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.request_id = request_id


class AnthropicChatAdapter(BaseChatAPIAdapter[AnthropicMessagesRequest, AnthropicMessagesResponse]):
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

    async def messages(
        self,
        request: AnthropicMessagesRequest,
        *,
        api_key: str | None = None,
    ) -> AnthropicMessagesResponse | StreamingResponse:
        if request.stream:
            response = await self.handle_request(request, api_key=api_key)
            return build_sse_response(self.iter_stream_events(response))
        return await self.handle_request(request, api_key=api_key)

    async def count_tokens(self, request: AnthropicCountTokensRequest) -> AnthropicCountTokensResponse:
        internal_messages = self._build_internal_messages(request)
        tokenizer_tools = self._normalize_tools_for_backend(request.tools)
        if self._tokenizer is None:
            return AnthropicCountTokensResponse(input_tokens=0)
        raw_prompt_ids = self._tokenizer.apply_chat_template(
            internal_messages,
            tools=tokenizer_tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = raw_prompt_ids.get("input_ids") if hasattr(raw_prompt_ids, "get") else list(raw_prompt_ids)
        return AnthropicCountTokensResponse(input_tokens=len(prompt_ids))

    def validate_request(self, request: AnthropicMessagesRequest) -> None:
        return None

    def request_to_canonical_request(self, request: AnthropicMessagesRequest) -> CanonicalGenerateRequest:
        messages: list[CanonicalMessage] = []
        if request.system:
            messages.append(self._anthropic_system_to_canonical_message(request.system))
        messages.extend(self._anthropic_messages_to_canonical_messages(request.messages))
        return CanonicalGenerateRequest(
            request_id=f"anthropic_req_{uuid4().hex}",
            model=request.model or self._default_model_name or "rollout-controller",
            messages=messages,
            tools=self._anthropic_tools_to_canonical(request.tools),
            tool_choice=self._anthropic_tool_choice_to_canonical(request.tool_choice),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=list(request.stop_sequences or []),
            stream=False,
            metadata={
                key: value
                for key, value in {
                    "source_protocol": "anthropic_messages",
                    "client_stream": bool(request.stream),
                    "session_uid": request.session_uid,
                }.items()
                if value is not None
            },
        )

    def normalize_request(self, request: AnthropicMessagesRequest) -> dict[str, Any]:
        return normalize_trace_payload(request.model_dump(mode="python", exclude_none=True))

    def normalize_response(self, response: AnthropicMessagesResponse) -> dict[str, Any]:
        return normalize_trace_payload(response.model_dump(mode="python", exclude_none=True))

    async def iter_stream_events(
        self,
        response: AnthropicMessagesResponse,
    ) -> AsyncIterator[str]:
        yield encode_sse_event(
            {
                "type": "message_start",
                "message": {
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                    "content": [],
                    "model": response.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": 0,
                    },
                },
            },
            event="message_start",
        )

        for index, block in enumerate(response.content):
            block_type = block.get("type")
            start_block: dict[str, Any]
            delta: dict[str, Any]
            if block_type == "reasoning":
                start_block = {"type": "thinking", "thinking": ""}
                delta = {"type": "thinking_delta", "thinking": str(block.get("text", ""))}
            elif block_type == "tool_use":
                start_block = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False),
                }
            else:
                start_block = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": str(block.get("text", ""))}

            yield encode_sse_event(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": start_block,
                },
                event="content_block_start",
            )
            yield encode_sse_event(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": delta,
                },
                event="content_block_delta",
            )
            yield encode_sse_event(
                {
                    "type": "content_block_stop",
                    "index": index,
                },
                event="content_block_stop",
            )

        yield encode_sse_event(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stream_stop_reason(response.stop_reason),
                    "stop_sequence": response.stop_sequence,
                },
                "usage": {
                    "output_tokens": response.usage.output_tokens,
                },
            },
            event="message_delta",
        )
        yield encode_sse_event({"type": "message_stop"}, event="message_stop")

    def canonical_response_to_protocol_response(
        self,
        canonical_response: CanonicalGenerateResponse,
        request: AnthropicMessagesRequest,
    ) -> AnthropicMessagesResponse:
        content = self._canonical_response_to_anthropic_blocks(
            canonical_response,
            tools=self._anthropic_tools_to_canonical(request.tools),
        )
        stop_reason = canonical_response.finish_reason or "stop"
        if any(block.get("type") == "tool_use" for block in content):
            stop_reason = "tool_use"
        return AnthropicMessagesResponse(
            id=f"msg_{canonical_response.request_id}",
            content=content,
            model=canonical_response.model or self._default_model_name or "rollout-controller",
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=canonical_response.usage.prompt_tokens,
                output_tokens=canonical_response.usage.completion_tokens,
            ),
        )

    def _build_internal_messages(self, request: AnthropicCountTokensRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                system_text = self._join_text_blocks(request.system, context="system")
            messages.append({"role": "system", "content": system_text})

        for message in request.messages:
            if isinstance(message.content, str):
                messages.append({"role": message.role, "content": message.content})
            else:
                messages.extend(self._convert_content_blocks_to_backend_messages(message.role, message.content))
        return messages

    def _join_text_blocks(self, blocks: list[dict[str, Any]], context: str) -> str:
        unsupported_types = [str(block.get("type")) for block in blocks if block.get("type") != "text"]
        if unsupported_types:
            unsupported_str = ", ".join(sorted(set(unsupported_types)))
            raise AnthropicChatAdapterError(
                f"Unsupported Anthropic content block type(s) in {context}: {unsupported_str}",
                "invalid_request_error",
            )
        return "\n".join(str(block.get("text", "")) for block in blocks)

    def _convert_content_blocks_to_backend_messages(
        self,
        role: str,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        backend_messages: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        def flush_text_chunks() -> None:
            if text_chunks:
                backend_messages.append({"role": role, "content": "\n".join(text_chunks)})
                text_chunks.clear()

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_value = str(block.get("text", ""))
                if role == "assistant":
                    text_value = self._sanitize_assistant_text(text_value)
                text_chunks.append(text_value)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"toolu_{uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            "arguments": normalize_trace_payload(block.get("input", {})),
                        },
                    }
                )
            elif block_type == "tool_result":
                flush_text_chunks()
                backend_messages.append(
                    {
                        "role": "tool",
                        "content": self._serialize_tool_result_content(block.get("content")),
                        "tool_call_id": block.get("tool_use_id"),
                    }
                )
            else:
                raise AnthropicChatAdapterError(
                    f"Unsupported Anthropic content block type in messages[{role}]: {block_type}",
                    "invalid_request_error",
                )

        if tool_calls:
            backend_messages.append(
                {
                    "role": role,
                    "content": "\n".join(text_chunks),
                    "tool_calls": tool_calls,
                }
            )
            text_chunks.clear()
        flush_text_chunks()
        return backend_messages

    def _serialize_tool_result_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if all(isinstance(item, dict) and item.get("type") == "text" for item in content):
                return "\n".join(str(item.get("text", "")) for item in content)
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _normalize_tools_for_backend(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        normalized_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                normalized_tools.append(normalize_trace_payload(tool))
            else:
                normalized_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool["input_schema"],
                        },
                    }
                )
        return normalize_trace_payload(normalized_tools)

    def _sanitize_assistant_text(self, text: str) -> str:
        cleaned = text.replace("<|im_end|>", "")
        cleaned = cleaned.replace("<think>", "")
        cleaned = cleaned.replace("</think>", "")
        return cleaned.strip()

    def _anthropic_system_to_canonical_message(
        self,
        system: str | list[dict[str, Any]],
    ) -> CanonicalMessage:
        if isinstance(system, str):
            content = [CanonicalTextBlock(text=system)] if system else []
        else:
            content = []
            for block in system:
                if block.get("type") != "text":
                    raise AnthropicChatAdapterError(
                        f"Unsupported Anthropic content block type(s) in system: {block.get('type')}",
                        "invalid_request_error",
                    )
                text = str(block.get("text", ""))
                if text:
                    content.append(CanonicalTextBlock(text=text))
        return CanonicalMessage(
            role="system",
            content=content,
            metadata={"source_protocol": "anthropic_messages"},
        )

    def _anthropic_messages_to_canonical_messages(
        self,
        messages: list[AnthropicMessage],
    ) -> list[CanonicalMessage]:
        canonical_messages = []
        for message in messages:
            if isinstance(message.content, str):
                content_blocks = [CanonicalTextBlock(text=message.content)] if message.content else []
            else:
                content_blocks = self._anthropic_content_blocks_to_canonical(message.content)
            canonical_messages.append(
                CanonicalMessage(
                    role=message.role,
                    content=content_blocks,
                    metadata={"source_protocol": "anthropic_messages"},
                )
            )
        return canonical_messages

    def _anthropic_content_blocks_to_canonical(
        self,
        blocks: list[dict[str, Any]],
    ) -> list[Any]:
        canonical_blocks: list[Any] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                canonical_blocks.append(CanonicalTextBlock(text=str(block.get("text", ""))))
            elif block_type == "tool_use":
                canonical_blocks.append(
                    CanonicalToolCallBlock(
                        tool_call=CanonicalToolCall(
                            id=str(block.get("id") or f"toolu_{uuid4().hex}"),
                            name=str(block.get("name", "")),
                            arguments=normalize_trace_payload(block.get("input", {})),
                            metadata={"source_protocol": "anthropic_messages"},
                        )
                    )
                )
            elif block_type == "tool_result":
                content = block.get("content")
                canonical_blocks.append(
                    CanonicalToolResultBlock(
                        tool_result=CanonicalToolResult(
                            tool_call_id=str(block.get("tool_use_id") or ""),
                            output=content,
                            output_text=self._serialize_tool_result_content(content),
                            is_error=bool(block.get("is_error", False)),
                            metadata={"source_protocol": "anthropic_messages"},
                        )
                    )
                )
            elif block_type in {"reasoning", "thinking"}:
                reasoning_text = str(block.get("text", ""))
                canonical_blocks.append(
                    CanonicalReasoningBlock(
                        reasoning=CanonicalReasoning(
                            steps=[CanonicalReasoningStep(text=reasoning_text)] if reasoning_text else [],
                            metadata={"source_protocol": "anthropic_messages"},
                        )
                    )
                )
            else:
                raise AnthropicChatAdapterError(
                    f"Unsupported Anthropic content block type in canonical mapping: {block_type}",
                    "invalid_request_error",
                )
        return canonical_blocks

    def _anthropic_tools_to_canonical(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[CanonicalToolDefinition]:
        if not tools:
            return []
        canonical_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function_spec = tool.get("function", {})
                name = function_spec.get("name")
                description = function_spec.get("description")
                parameters = function_spec.get("parameters", {})
            else:
                name = tool.get("name")
                description = tool.get("description")
                parameters = tool.get("input_schema", {})
            canonical_tools.append(
                CanonicalToolDefinition(
                    name=str(name or ""),
                    description=description,
                    parameters_json_schema=parameters,
                    metadata={"source_protocol": "anthropic_messages"},
                )
            )
        return canonical_tools

    def _anthropic_tool_choice_to_canonical(
        self,
        tool_choice: str | dict[str, Any] | None,
    ) -> CanonicalToolChoice | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            mapped_type = "required" if tool_choice == "any" else tool_choice
            return CanonicalToolChoice(type=mapped_type)
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            return CanonicalToolChoice(
                type="specific",
                tool_name=tool_choice.get("name"),
                metadata={"source_protocol": "anthropic_messages"},
            )
        mapped_type = "required" if choice_type == "any" else str(choice_type or "auto")
        return CanonicalToolChoice(
            type=mapped_type,
            metadata={"source_protocol": "anthropic_messages"},
        )

    def _canonical_response_to_anthropic_blocks(
        self,
        response: CanonicalGenerateResponse,
        tools: list[CanonicalToolDefinition] | None = None,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in response.output.content:
            if isinstance(block, CanonicalTextBlock):
                if block.text:
                    blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, CanonicalToolCallBlock):
                tool_call = self._sanitize_tool_call_for_request(block.tool_call, tools=tools or [])
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments if tool_call.arguments is not None else {},
                    }
                )
            elif isinstance(block, CanonicalToolResultBlock):
                tool_result_content: Any = block.tool_result.output
                if tool_result_content is None:
                    tool_result_content = block.tool_result.output_text or ""
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_result.tool_call_id,
                        "content": tool_result_content,
                        "is_error": block.tool_result.is_error,
                    }
                )
            elif isinstance(block, CanonicalReasoningBlock):
                reasoning_text = self._reasoning_to_text(block.reasoning)
                if reasoning_text:
                    blocks.append({"type": "thinking", "thinking": reasoning_text})
        return blocks or [{"type": "text", "text": ""}]

    def _sanitize_tool_call_for_request(
        self,
        tool_call: CanonicalToolCall,
        *,
        tools: list[CanonicalToolDefinition],
    ) -> CanonicalToolCall:
        tool_definition = next((tool for tool in tools if tool.name == tool_call.name), None)
        if tool_definition is None:
            return tool_call

        properties = tool_definition.parameters_json_schema.get("properties")
        if not isinstance(properties, dict):
            return tool_call

        arguments = tool_call.arguments
        normalized_arguments = False
        if not isinstance(arguments, dict):
            normalized_arguments = True
            if tool_call.raw_arguments_text is not None:
                try:
                    decoded = json.loads(tool_call.raw_arguments_text)
                except Exception:
                    decoded = {"raw": tool_call.raw_arguments_text}
                arguments = decoded if isinstance(decoded, dict) else {"value": decoded}
            elif arguments is None:
                arguments = {}
            elif isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                except Exception:
                    decoded = {"raw": arguments}
                arguments = decoded if isinstance(decoded, dict) else {"value": decoded}
            else:
                arguments = {"value": arguments}

        allowed_keys = set(properties)
        cleaned_arguments = {key: value for key, value in arguments.items() if key in allowed_keys}
        if cleaned_arguments == arguments and not normalized_arguments:
            return tool_call

        dropped_keys = sorted(set(arguments) - set(cleaned_arguments))
        metadata = dict(tool_call.metadata)
        if dropped_keys:
            metadata["dropped_arguments"] = dropped_keys
        return CanonicalToolCall(
            id=tool_call.id,
            name=tool_call.name,
            arguments=cleaned_arguments,
            raw_arguments_text=None,
            metadata=metadata,
        )

    def _reasoning_to_text(self, reasoning: CanonicalReasoning) -> str:
        return "\n".join(step.text for step in reasoning.steps if step.text).strip()

    def _stream_stop_reason(self, stop_reason: str | None) -> str | None:
        if stop_reason == "stop":
            return "end_turn"
        if stop_reason == "length":
            return "max_tokens"
        return stop_reason
