import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..core.models import (
    CanonicalGenerateRequest,
    CanonicalGenerateResponse,
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalTextBlock,
    CanonicalToolCall,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolDefinition,
    CanonicalToolResult,
    CanonicalToolResultBlock,
)
from .base import BaseChatAPIAdapter, coerce_content_to_text, stringify_tool_arguments
from .streaming import build_sse_response, encode_sse_event
from .trace import ChatTraceStore, normalize_trace_payload


class ChatCompletionStreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False
    continuous_usage_stats: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_uid: int | str | None = None
    session_id: int | str | None = None
    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stream: bool = False
    stream_options: ChatCompletionStreamOptions | None = None
    n: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    min_tokens: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    skip_special_tokens: bool | None = None
    no_stop_trim: bool | None = None
    seed: int | None = None
    user: str | None = None
    return_routed_experts: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class DeltaMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionResponseChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionResponseStreamChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None


class OpenAIChatAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        error_type: str,
        code: str,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.code = code
        self.request_id = request_id


class OpenAIChatAdapter(BaseChatAPIAdapter[ChatCompletionRequest, ChatCompletionResponse]):
    def __init__(
        self,
        generate_handler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str,
        default_model_name: str | None = None,
        context_length: int | None = None,
        capture_folder: str | None = None,
        trace_store: ChatTraceStore | None = None,
        trace_store_max_entries: int = 10000,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(
            generate_handler,
            tokenizer=tokenizer,
            capture_folder=capture_folder,
            trace_store=trace_store,
            trace_store_max_entries=trace_store_max_entries,
        )
        self._default_model_name = default_model_name
        self._context_length = context_length

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        api_key: str | None = None,
    ) -> ChatCompletionResponse | StreamingResponse:
        if request.stream:
            response = await self.handle_request(request, api_key=api_key)
            return build_sse_response(self.iter_stream_events(response, request))
        return await self.handle_request(request, api_key=api_key)

    def validate_request(self, request: ChatCompletionRequest) -> None:
        if request.n not in (None, 1):
            raise OpenAIChatAdapterError(
                "n>1 is not supported yet",
                "invalid_request_error",
                "n_not_supported",
            )

    def request_to_canonical_request(self, request: ChatCompletionRequest) -> CanonicalGenerateRequest:
        normalized_messages = normalize_trace_payload(request.messages)
        normalized_tools = normalize_trace_payload(request.tools)
        normalized_tool_choice = normalize_trace_payload(request.tool_choice)
        stop = [] if request.stop is None else [request.stop] if isinstance(request.stop, str) else list(request.stop)
        chat_template_kwargs = request.chat_template_kwargs or {}
        return CanonicalGenerateRequest(
            request_id=f"chatcmpl_req_{uuid4().hex}",
            model=request.model or self._default_model_name or "rollout-controller",
            messages=[self._openai_message_to_canonical_message(message) for message in normalized_messages],
            tools=self._openai_tools_to_canonical(normalized_tools),
            tool_choice=self._openai_tool_choice_to_canonical(normalized_tool_choice),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens,
            stop=stop,
            stream=False,
            metadata={
                key: value
                for key, value in {
                    "source_protocol": "openai_chat_completions",
                    "client_stream": bool(request.stream),
                    "session_uid": getattr(request, "session_uid", getattr(request, "session_id", None)),
                    "n": request.n,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "min_tokens": request.min_tokens,
                    "stop_token_ids": request.stop_token_ids,
                    "skip_special_tokens": request.skip_special_tokens,
                    "no_stop_trim": request.no_stop_trim,
                    "spaces_between_special_tokens": chat_template_kwargs.get("spaces_between_special_tokens"),
                    "sampling_seed": request.seed,
                    "user": request.user,
                    "return_routed_experts": request.return_routed_experts,
                }.items()
                if value is not None
            },
        )

    def canonical_response_to_chat_completion_response(
        self,
        response: CanonicalGenerateResponse,
    ) -> ChatCompletionResponse:
        message_content = self._render_openai_response_text(response)
        reasoning_content = self._render_openai_reasoning_text(response)
        tool_calls = self._canonical_tool_calls_to_openai(response)
        finish_reason = response.finish_reason or ("tool_calls" if tool_calls else "stop")
        return ChatCompletionResponse(
            id=response.request_id,
            created=int(time.time()),
            model=response.model or self._default_model_name or "rollout-controller",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=None if tool_calls and not message_content else message_content,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def canonical_response_to_protocol_response(
        self,
        canonical_response: CanonicalGenerateResponse,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        return self.canonical_response_to_chat_completion_response(canonical_response)

    def normalize_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        return normalize_trace_payload(
            {
                "messages": request.messages,
                "tools": request.tools,
                "tool_choice": request.tool_choice,
            }
        )

    def normalize_response(self, response: ChatCompletionResponse) -> dict[str, Any]:
        normalized_choices = []
        for choice in response.choices:
            normalized_choices.append(
                {
                    "message": getattr(choice.message, "model_dump", lambda **_: choice.message)(
                        mode="python",
                        exclude_none=True,
                    )
                    if choice.message is not None
                    else None,
                    "finish_reason": choice.finish_reason,
                }
            )
        return normalize_trace_payload({"choices": normalized_choices})

    async def iter_stream_events(
        self,
        response: ChatCompletionResponse,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        choice = response.choices[0]
        include_usage = bool(getattr(request.stream_options, "include_usage", False))

        initial_chunk = ChatCompletionStreamResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                )
            ],
        )
        yield encode_sse_event(initial_chunk.model_dump(mode="json", exclude_none=True))

        if choice.message.reasoning_content:
            reasoning_chunk = ChatCompletionStreamResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(reasoning_content=choice.message.reasoning_content),
                    )
                ],
            )
            yield encode_sse_event(reasoning_chunk.model_dump(mode="json", exclude_none=True))

        if choice.message.content:
            content_chunk = ChatCompletionStreamResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=choice.message.content),
                    )
                ],
            )
            yield encode_sse_event(content_chunk.model_dump(mode="json", exclude_none=True))

        for index, tool_call in enumerate(choice.message.tool_calls or []):
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            tool_call_type = (
                tool_call.get("type", "function")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "type", "function")
            )
            function_payload = (
                tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
            )
            if isinstance(function_payload, BaseModel):
                function_payload = function_payload.model_dump(mode="json", exclude_none=True)
            tool_call_chunk = ChatCompletionStreamResponse(
                id=response.id,
                created=response.created,
                model=response.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                {
                                    "index": index,
                                    "id": tool_call_id,
                                    "type": tool_call_type,
                                    "function": function_payload,
                                }
                            ]
                        ),
                    )
                ],
            )
            yield encode_sse_event(tool_call_chunk.model_dump(mode="json", exclude_none=True))

        final_chunk = ChatCompletionStreamResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=choice.finish_reason,
                )
            ],
            usage=response.usage if include_usage else None,
        )
        yield encode_sse_event(final_chunk.model_dump(mode="json", exclude_none=True))
        yield encode_sse_event("[DONE]")

    def _openai_message_to_canonical_message(self, message: dict[str, Any]) -> CanonicalMessage:
        role = str(message.get("role", "user"))
        content_blocks: list[Any] = []
        if role == "tool":
            content_blocks.append(
                CanonicalToolResultBlock(
                    tool_result=CanonicalToolResult(
                        tool_call_id=str(message.get("tool_call_id") or message.get("name") or ""),
                        name=message.get("name"),
                        output=message.get("content"),
                        output_text=coerce_content_to_text(message.get("content")),
                        metadata={"source_protocol": "openai_chat_completions"},
                    )
                )
            )
        else:
            content_text = coerce_content_to_text(message.get("content"))
            if content_text:
                content_blocks.append(CanonicalTextBlock(text=content_text))
            for tool_call in message.get("tool_calls") or []:
                content_blocks.append(CanonicalToolCallBlock(tool_call=self._openai_tool_call_to_canonical(tool_call)))
        return CanonicalMessage(
            role=role if role in {"system", "user", "assistant", "tool"} else "user",
            content=content_blocks,
            name=message.get("name"),
            metadata={
                key: value
                for key, value in {
                    "source_protocol": "openai_chat_completions",
                    "tool_call_id": message.get("tool_call_id"),
                }.items()
                if value is not None
            },
        )

    def _openai_tools_to_canonical(self, tools: list[dict[str, Any]] | None) -> list[CanonicalToolDefinition]:
        if not tools:
            return []
        canonical_tools = []
        for tool in tools:
            function_spec = tool.get("function", tool)
            canonical_tools.append(
                CanonicalToolDefinition(
                    name=str(function_spec.get("name", "")),
                    description=function_spec.get("description"),
                    parameters_json_schema=function_spec.get("parameters", {}),
                    metadata={"source_protocol": "openai_chat_completions"},
                )
            )
        return canonical_tools

    def _openai_tool_choice_to_canonical(self, tool_choice: Any) -> CanonicalToolChoice | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return CanonicalToolChoice(type=tool_choice)
        function_spec = tool_choice.get("function") or {}
        return CanonicalToolChoice(
            type="specific",
            tool_name=function_spec.get("name"),
            metadata={"source_protocol": "openai_chat_completions"},
        )

    def _openai_tool_call_to_canonical(self, tool_call: dict[str, Any]) -> CanonicalToolCall:
        function_spec = tool_call.get("function") or {}
        raw_arguments = function_spec.get("arguments")
        parsed_arguments = self._parse_tool_arguments(raw_arguments)
        metadata: dict[str, Any] = {"source_protocol": "openai_chat_completions"}
        if isinstance(parsed_arguments, dict) and parsed_arguments.pop("__parse_error__", False):
            metadata["arguments_parse_error"] = True
        return CanonicalToolCall(
            id=str(tool_call.get("id") or f"call_{uuid4().hex}"),
            name=str(function_spec.get("name", "")),
            arguments=parsed_arguments,
            raw_arguments_text=raw_arguments if isinstance(raw_arguments, str) else None,
            metadata=metadata,
        )

    def _canonical_tool_calls_to_openai(self, response: CanonicalGenerateResponse) -> list[dict[str, Any]]:
        tool_calls = []
        for block in response.output.content:
            if isinstance(block, CanonicalToolCallBlock):
                tool_calls.append(
                    {
                        "id": block.tool_call.id,
                        "type": "function",
                        "function": {
                            "name": block.tool_call.name,
                            "arguments": stringify_tool_arguments(block.tool_call),
                        },
                    }
                )
        return tool_calls

    def _render_openai_response_text(self, response: CanonicalGenerateResponse) -> str | None:
        text_chunks = []
        for block in response.output.content:
            if isinstance(block, CanonicalTextBlock):
                text_chunks.append(block.text)
        joined = "".join(text_chunks).strip()
        return joined or None

    def _render_openai_reasoning_text(self, response: CanonicalGenerateResponse) -> str | None:
        reasoning_chunks: list[str] = []
        for block in response.output.content:
            if isinstance(block, CanonicalReasoningBlock):
                reasoning_chunks.extend(step.text for step in block.reasoning.steps if step.text)
        joined = "\n".join(chunk for chunk in reasoning_chunks if chunk).strip()
        return joined or None

    def _parse_tool_arguments(self, raw_arguments: Any) -> Any:
        if not isinstance(raw_arguments, str):
            return raw_arguments
        try:
            return json.loads(raw_arguments)
        except Exception:
            return {"__parse_error__": True, "raw": raw_arguments}
