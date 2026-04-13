import json
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status

from .base import BaseChatAPIAdapter
from .trace import normalize_trace_payload


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
    _tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
    _qwen_function_pattern = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
    _qwen_parameter_pattern = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)

    def __init__(
        self,
        generate_handler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str | None,
        default_model_name: str | None = None,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(generate_handler, tokenizer=tokenizer)
        self._default_model_name = default_model_name

    async def messages(self, request: AnthropicMessagesRequest) -> AnthropicMessagesResponse:
        return await self.handle_request(request)

    async def count_tokens(self, request: AnthropicCountTokensRequest) -> AnthropicCountTokensResponse:
        internal_messages = self._build_internal_messages(request)
        rollout_state = RolloutState(message=internal_messages)
        tokenizer_tools = self._normalize_tools_for_backend(request.tools)
        if self._tokenizer is not None:
            raw_prompt_ids = self._tokenizer.apply_chat_template(
                internal_messages,
                tools=tokenizer_tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            rollout_state.prompt_ids = raw_prompt_ids.get("input_ids") if hasattr(raw_prompt_ids, "get") else list(raw_prompt_ids)
            rollout_state.tokens = rollout_state.prompt_ids
        return AnthropicCountTokensResponse(input_tokens=self._count_prompt_tokens(rollout_state))

    def validate_request(self, request: AnthropicMessagesRequest) -> None:
        if request.stream:
            raise AnthropicChatAdapterError(
                "stream=true is not supported yet",
                "invalid_request_error",
            )

    def request_to_rollout_state(self, request: AnthropicMessagesRequest) -> RolloutState:
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
        return RolloutState(
            uid=uuid4().int,
            message=internal_messages,
            prompt_ids=prompt_ids,
            tokens=prompt_ids,
            session_uid=request.session_uid,
            tools=tokenizer_tools,
            tool_choice=normalized_tool_choice,
            sample_params=self._build_sample_params(request),
        )

    def raise_for_failed_response(self, response: RolloutState, request_id: str) -> None:
        if response.status == Status.FAILED:
            raise AnthropicChatAdapterError(
                response.error_msg or "Rollout generation failed",
                "api_error",
                request_id,
            )

    def normalize_request(self, request: AnthropicMessagesRequest) -> dict[str, Any]:
        return normalize_trace_payload(request.model_dump(mode="python", exclude_none=True))

    def normalize_response(self, response: AnthropicMessagesResponse) -> dict[str, Any]:
        return normalize_trace_payload(response.model_dump(mode="python", exclude_none=True))

    def rollout_state_to_response(
        self,
        rollout_state: RolloutState,
        request: AnthropicMessagesRequest,
    ) -> AnthropicMessagesResponse:
        assert rollout_state.uid is not None, "uid should not be None when generating response"
        request_id = str(rollout_state.uid)
        model_name = request.model or self._default_model_name or "rollout-controller"
        prompt_tokens = self._count_prompt_tokens(rollout_state)
        completion_tokens = self._count_completion_tokens(rollout_state)
        content_blocks = self._build_response_content_blocks(rollout_state)
        stop_reason = "tool_use" if any(block.get("type") == "tool_use" for block in content_blocks) else rollout_state.finish_reason

        return AnthropicMessagesResponse(
            id=f"msg_{request_id}",
            content=content_blocks,
            model=model_name,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )

    def _build_internal_messages(self, request: AnthropicMessagesRequest) -> list[dict[str, str]]:
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
        unsupported_types = [block.get("type") for block in blocks if block.get("type") != "text"]
        if unsupported_types:
            unsupported_str = ", ".join(sorted(set(unsupported_types)))
            raise AnthropicChatAdapterError(
                f"Unsupported Anthropic content block type(s) in {context}: {unsupported_str}",
                "invalid_request_error",
            )
        return "\n".join(str(block.get("text", "")) for block in blocks)

    def _convert_content_blocks_to_backend_messages(self, role: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        backend_messages: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        def flush_text_chunks():
            if text_chunks:
                backend_messages.append({"role": role, "content": "\n".join(text_chunks)})
                text_chunks.clear()

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_chunks.append(str(block.get("text", "")))
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

    def _normalize_tool_choice_for_backend(self, tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "none":
            return "none"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {
                "type": "function",
                "function": {
                    "name": tool_choice.get("name"),
                },
            }
        return normalize_trace_payload(tool_choice)

    def _build_response_content_blocks(self, rollout_state: RolloutState) -> list[dict[str, Any]]:
        tool_calls = rollout_state.extra_fields.get("tool_calls")
        if not tool_calls:
            text_blocks, parsed_tool_calls = self._parse_textual_tool_calls(rollout_state.response or "")
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls
                rollout_state.extra_fields["tool_calls"] = parsed_tool_calls
                if text_blocks:
                    rollout_state.response = "".join(block["text"] for block in text_blocks if block["type"] == "text")
                else:
                    rollout_state.response = ""

        if not tool_calls:
            return [{"type": "text", "text": rollout_state.response or ""}]

        content_blocks: list[dict[str, Any]] = []
        if rollout_state.response:
            content_blocks.append({"type": "text", "text": rollout_state.response})
        for tool_call in tool_calls:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{uuid4().hex}",
                    "name": tool_call["function"]["name"],
                    "input": self._parse_tool_arguments(tool_call["function"].get("arguments")),
                }
            )
        return content_blocks

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
            arguments[param_name] = param_value
        return {
            "id": f"call_{uuid4().hex}",
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }

    def _parse_tool_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except Exception:
                return {"raw": arguments}
        return arguments

    def _build_sample_params(self, request: AnthropicMessagesRequest) -> SampleParams:
        kwargs = {
            "return_token_ids": True,
            "return_logprob": False,
            "stream": request.stream,
            "max_tokens": request.max_tokens,
            "stops": request.stop_sequences or [],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        return SampleParams(**kwargs)

    def _count_prompt_tokens(self, rollout_state: RolloutState) -> int:
        if rollout_state.tokens is not None:
            return len(rollout_state.tokens)
        if rollout_state.prompt_ids is not None:
            return len(rollout_state.prompt_ids)
        if self._tokenizer is not None and rollout_state.message:
            text_prompt = self._tokenizer.apply_chat_template(
                rollout_state.message,
                tokenize=False,
                add_generation_prompt=True,
            )
            return len(self._tokenizer(text_prompt, add_special_tokens=False)["input_ids"])
        return 0

    def _count_completion_tokens(self, rollout_state: RolloutState) -> int:
        if rollout_state.response_ids is not None:
            return len(rollout_state.response_ids)
        if self._tokenizer is not None and rollout_state.response:
            return len(self._tokenizer(rollout_state.response, add_special_tokens=False)["input_ids"])
        return 0

    def build_output_message_list(
        self,
        rollout_state: RolloutState,
        request: AnthropicMessagesRequest,
    ) -> list[dict[str, Any]]:
        return [{"role": "assistant", "content": self._build_response_content_blocks(rollout_state)}]


def bind_anthropic_chat_interface(
    rollout_controller: Any,
    default_model_name: str | None = None,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str | None = None,
) -> Any:
    if getattr(rollout_controller, "anthropic_chat_adapter", None) is None:
        rollout_controller.anthropic_chat_adapter = AnthropicChatAdapter(
            rollout_controller.generate,
            default_model_name=default_model_name,
            tokenizer=tokenizer,
        )
    rollout_controller.anthropic_messages = rollout_controller.anthropic_chat_adapter.messages
    rollout_controller.anthropic_count_tokens = rollout_controller.anthropic_chat_adapter.count_tokens
    return rollout_controller
