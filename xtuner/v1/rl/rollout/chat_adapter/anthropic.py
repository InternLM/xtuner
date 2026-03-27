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


AnthropicContentBlock = AnthropicTextContent


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_uid: int | None = None
    model: str | None = None
    system: str | list[AnthropicTextContent] | None = None
    messages: list[AnthropicMessage]
    max_tokens: int
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


class AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicTextContent]
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
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(generate_handler, tokenizer=tokenizer)
        self._default_model_name = default_model_name

    async def messages(self, request: AnthropicMessagesRequest) -> AnthropicMessagesResponse:
        return await self.handle_request(request)

    def validate_request(self, request: AnthropicMessagesRequest) -> None:
        if request.stream:
            raise AnthropicChatAdapterError(
                "stream=true is not supported yet",
                "invalid_request_error",
            )

    def request_to_rollout_state(self, request: AnthropicMessagesRequest) -> RolloutState:
        return RolloutState(
            uid=uuid4().int,
            message=self._build_internal_messages(request),
            session_uid=request.session_uid,
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

        return AnthropicMessagesResponse(
            id=f"msg_{request_id}",
            content=[AnthropicTextContent(text=rollout_state.response or "")],
            model=model_name,
            stop_reason=rollout_state.finish_reason,
            usage=AnthropicUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )
    
    def _build_internal_messages(self, request: AnthropicMessagesRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                system_text = self._join_text_blocks(request.system, context="system")
            messages.append({"role": "system", "content": system_text})

        for message in request.messages:
            if isinstance(message.content, str):
                content = message.content
            else:
                content = self._join_text_blocks(message.content, context=f"messages[{message.role}]")
            messages.append({"role": message.role, "content": content})

        return messages

    def _join_text_blocks(self, blocks: list[AnthropicContentBlock], context: str) -> str:
        unsupported_types = [block.type for block in blocks if block.type != "text"]
        if unsupported_types:
            unsupported_str = ", ".join(sorted(set(unsupported_types)))
            raise AnthropicChatAdapterError(
                f"Unsupported Anthropic content block type(s) in {context}: {unsupported_str}",
                "invalid_request_error",
            )
        return "\n".join(block.text for block in blocks)

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
        return [
            {
                "role": "assistant",
                "content": rollout_state.response or "",
            }
        ]


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
    return rollout_controller
