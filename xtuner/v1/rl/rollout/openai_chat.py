import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.utils import get_logger

from .utils import ensure_rollout_request_id


logger = get_logger(__name__)
GenerateHandler = Callable[[RolloutState], Awaitable[RolloutState]]


class ChatCompletionRequest(BaseModel):
    messages: list[dict[str, Any]]
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


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


class OpenAIChatAdapter:
    def __init__(
        self,
        generate_handler: GenerateHandler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str,
        default_model_name: str | None = None,
    ):
        self._generate_handler = generate_handler
        self._default_model_name = default_model_name
        if isinstance(tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self._tokenizer = tokenizer

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise OpenAIChatAdapterError(
                "stream=true is not supported yet",
                "invalid_request_error",
                "stream_not_supported",
            )
        rollout_state = self._build_rollout_state(request)
        request_id = ensure_rollout_request_id(rollout_state)
        response = await self._generate_handler(rollout_state)
        response.extra_fields.setdefault("request_id", request_id)

        if response.status == Status.FAILED:
            raise OpenAIChatAdapterError(
                response.error_msg or "Rollout generation failed",
                "server_error",
                "rollout_failed",
                request_id,
            )

        return self._build_chat_completion_response(response, request)

    def _build_rollout_state(self, request: ChatCompletionRequest) -> RolloutState:
        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            raise OpenAIChatAdapterError(
                "tool_choice object form is not supported yet",
                "invalid_request_error",
                "unsupported_tool_choice",
            )
        rollout_state = RolloutState(
            message=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            sample_params=self._build_sample_params(request),
        )
        return rollout_state

    def _build_sample_params(self, request: ChatCompletionRequest) -> SampleParams:
        stops: list[str]
        if request.stop is None:
            stops = []
        elif isinstance(request.stop, str):
            stops = [request.stop]
        else:
            stops = request.stop

        kwargs = {
            "return_token_ids": False,
            "return_logprob": False,
            "stream": request.stream,
            "stops": stops,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.n is not None:
            kwargs["n"] = request.n
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        return SampleParams(**kwargs)

    def _build_chat_completion_response(
        self,
        rollout_state: RolloutState,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        request_id = ensure_rollout_request_id(rollout_state)
        response_id = f"chatcmpl-{request_id}"
        model_name = request.model or self._default_model_name or "rollout-controller"
        prompt_tokens = self._count_prompt_tokens(rollout_state)
        completion_tokens = self._count_completion_tokens(rollout_state)
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        choice = ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(content=rollout_state.response),
            finish_reason=rollout_state.finish_reason,
        )
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=model_name,
            choices=[choice],
            usage=usage,
        )

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


def bind_openai_chat_interface(
    rollout_controller: Any,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast,
    default_model_name: str | None = None,
) -> Any:
    rollout_controller.openai_chat_adapter = OpenAIChatAdapter(
        rollout_controller.generate,
        tokenizer=tokenizer,
        default_model_name=default_model_name,
    )
    rollout_controller.chat = rollout_controller.openai_chat_adapter.chat
    return rollout_controller
