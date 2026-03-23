from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status

from .openai_chat import ensure_rollout_request_id


GenerateHandler = Callable[[RolloutState], Awaitable[RolloutState]]


class ClaudeTextContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "text"
    text: str


ClaudeContentBlock = ClaudeTextContent


class ClaudeMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: str | list[ClaudeContentBlock]


class ClaudeMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    system: str | list[ClaudeTextContent] | None = None
    messages: list[ClaudeMessage]
    max_tokens: int
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


class ClaudeUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class ClaudeMessagesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ClaudeTextContent]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


class ClaudeChatAdapterError(RuntimeError):
    def __init__(self, message: str, error_type: str, request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.request_id = request_id


class ClaudeChatAdapter:
    def __init__(
        self,
        generate_handler: GenerateHandler,
        default_model_name: str | None = None,
        tokenizer: Any | None = None,
    ):
        self._generate_handler = generate_handler
        self._default_model_name = default_model_name
        self._tokenizer = tokenizer

    async def messages(self, request: ClaudeMessagesRequest) -> ClaudeMessagesResponse:
        if request.stream:
            raise ClaudeChatAdapterError(
                "stream=true is not supported yet",
                "invalid_request_error",
            )

        rollout_state = self._build_rollout_state(request)
        request_id = self._ensure_request_id(rollout_state)
        response = await self._generate_handler(rollout_state)

        if not response.extra_fields.get("request_id"):
            response.extra_fields["request_id"] = request_id

        if response.status == Status.FAILED:
            raise ClaudeChatAdapterError(
                response.error_msg or "Rollout generation failed",
                "api_error",
                request_id,
            )

        return self._build_messages_response(response, request)

    def _build_rollout_state(self, request: ClaudeMessagesRequest) -> RolloutState:
        messages = self._build_internal_messages(request)
        rollout_state = RolloutState(
            message=messages,
            sample_params=self._build_sample_params(request),
        )
        self._ensure_request_id(rollout_state)
        return rollout_state

    def _build_internal_messages(self, request: ClaudeMessagesRequest) -> list[dict[str, str]]:
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

    def _join_text_blocks(self, blocks: list[ClaudeContentBlock], context: str) -> str:
        unsupported_types = [block.type for block in blocks if block.type != "text"]
        if unsupported_types:
            unsupported_str = ", ".join(sorted(set(unsupported_types)))
            raise ClaudeChatAdapterError(
                f"Unsupported Claude content block type(s) in {context}: {unsupported_str}",
                "invalid_request_error",
            )
        return "\n".join(block.text for block in blocks)

    def _build_sample_params(self, request: ClaudeMessagesRequest) -> SampleParams:
        kwargs = {
            "return_token_ids": False,
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

    def _build_messages_response(
        self,
        rollout_state: RolloutState,
        request: ClaudeMessagesRequest,
    ) -> ClaudeMessagesResponse:
        request_id = self._ensure_request_id(rollout_state)
        model_name = request.model or self._default_model_name or "rollout-controller"
        prompt_tokens = self._count_prompt_tokens(rollout_state)
        completion_tokens = self._count_completion_tokens(rollout_state)

        return ClaudeMessagesResponse(
            id=f"msg_{request_id}",
            content=[ClaudeTextContent(text=rollout_state.response or "")],
            model=model_name,
            stop_reason=rollout_state.finish_reason,
            usage=ClaudeUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )

    @staticmethod
    def _ensure_request_id(rollout_state: RolloutState) -> str:
        return ensure_rollout_request_id(rollout_state)

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


def bind_claude_chat_interface(
    rollout_controller: Any,
    default_model_name: str | None = None,
) -> Any:
    if getattr(rollout_controller, "claude_chat_adapter", None) is None:
        rollout_controller.claude_chat_adapter = ClaudeChatAdapter(
            rollout_controller.generate,
            default_model_name=default_model_name,
            tokenizer=getattr(rollout_controller, "tokenizer", None),
        )
    rollout_controller.claude_messages = rollout_controller.claude_chat_adapter.messages
    return rollout_controller
