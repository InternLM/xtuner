import time
from typing import Any
from uuid import uuid4

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

from .base import BaseChatAPIAdapter
from .trace import normalize_trace_payload


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
        trace_store_max_entries: int = 10000,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(
            generate_handler,
            tokenizer=tokenizer,
            trace_store_max_entries=trace_store_max_entries,
        )
        self._default_model_name = default_model_name

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self.handle_request(request)

    def validate_request(self, request: ChatCompletionRequest) -> None:
        if request.stream:
            raise OpenAIChatAdapterError(
                "stream=true is not supported yet",
                "invalid_request_error",
                "stream_not_supported",
            )

    def request_to_rollout_state(self, request: ChatCompletionRequest) -> RolloutState:
        prompt_ids = None
        if self._tokenizer:
            raw_prompt_ids = self._tokenizer.apply_chat_template(
                request.messages,
                tools=request.tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = raw_prompt_ids["input_ids"] if isinstance(raw_prompt_ids, dict) else list(raw_prompt_ids)

        return RolloutState(
            uid=uuid4().int,
            message=request.messages,
            prompt_ids=prompt_ids,
            tokens=prompt_ids,
            session_uid=getattr(request, "session_uid", getattr(request, "session_id", None)),
            tools=request.tools,
            tool_choice=request.tool_choice,
            sample_params=self._build_sample_params(request),
        )

    def raise_for_failed_response(self, response: RolloutState, request_id: str) -> None:
        if response.status == Status.FAILED:
            raise OpenAIChatAdapterError(
                response.error_msg or "Rollout generation failed",
                "server_error",
                "rollout_failed",
                request_id,
            )

    def rollout_state_to_response(
        self,
        rollout_state: RolloutState,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        request_id = str(rollout_state.uid)
        model_name = request.model or self._default_model_name or "rollout-controller"
        assert rollout_state.response_ids is not None, "response_ids should not be None when generating response"
        assert rollout_state.tokens is not None, "tokens should not be None when generating response"
        prompt_tokens = len(rollout_state.tokens)
        completion_tokens = len(rollout_state.response_ids)
        tool_calls = rollout_state.extra_fields.get("tool_calls")
        response_message = ChatMessage(
            role="assistant",
            content=None if tool_calls else rollout_state.response,
            tool_calls=tool_calls,
        )
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=response_message,
                    finish_reason=rollout_state.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def build_output_message_list(
        self,
        rollout_state: RolloutState,
        request: ChatCompletionRequest,
    ) -> list[dict[str, Any]]:
        return [{"role": "assistant", "content": rollout_state.response or ""}]

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

    def _build_sample_params(self, request: ChatCompletionRequest) -> SampleParams:
        stops = [] if request.stop is None else [request.stop] if isinstance(request.stop, str) else request.stop
        kwargs = {
            "stops": stops,
            **{
                key: value
                for key, value in {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                    "max_tokens": request.max_tokens,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                }.items()
                if value is not None
            },
        }
        return SampleParams(**kwargs)


def bind_openai_chat_interface(
    rollout_controller: Any,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None,
    default_model_name: str | None = None,
) -> Any:
    if getattr(rollout_controller, "openai_chat_adapter", None) is None:
        rollout_controller.openai_chat_adapter = OpenAIChatAdapter(
            rollout_controller.generate,
            tokenizer=tokenizer,
            default_model_name=default_model_name,
        )
    rollout_controller.chat = rollout_controller.openai_chat_adapter.chat
    return rollout_controller
