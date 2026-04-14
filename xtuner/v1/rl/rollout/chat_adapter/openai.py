import time
from typing import Any
from uuid import uuid4

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status

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
        context_length: int | None = None,
        capture_path: str | None = None,
        trace_store_max_entries: int = 10000,
    ):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        super().__init__(
            generate_handler,
            tokenizer=tokenizer,
            capture_path=capture_path,
            trace_store_max_entries=trace_store_max_entries,
        )
        self._default_model_name = default_model_name
        self._context_length = context_length

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
        normalized_messages = normalize_trace_payload(request.messages)
        tokenizer_tools = self._normalize_tools_for_tokenizer(request.tools)
        normalized_tool_choice = normalize_trace_payload(request.tool_choice)
        prompt_ids = None
        if self._tokenizer:
            raw_prompt_ids = self._tokenizer.apply_chat_template(
                normalized_messages,
                tools=tokenizer_tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(raw_prompt_ids, "get"):
                prompt_ids = raw_prompt_ids.get("input_ids")
            else:
                prompt_ids = list(raw_prompt_ids)
        max_tokens = self._fit_max_tokens_to_context(prompt_ids=prompt_ids, requested_max_tokens=request.max_tokens)

        return RolloutState(
            uid=uuid4().int,
            message=normalized_messages,
            prompt_ids=prompt_ids,
            tokens=prompt_ids,
            session_uid=getattr(request, "session_uid", getattr(request, "session_id", None)),
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

    def _normalize_tools_for_tokenizer(self, tools: Any) -> Any:
        if tools is None:
            return None
        return normalize_trace_payload(tools)

    def _build_sample_params(self, request: ChatCompletionRequest, max_tokens: int | None = None) -> SampleParams:
        stops = [] if request.stop is None else [request.stop] if isinstance(request.stop, str) else request.stop
        kwargs = {
            "stops": stops,
            **{
                key: value
                for key, value in {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                    "max_tokens": max_tokens if max_tokens is not None else request.max_tokens,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                }.items()
                if value is not None
            },
        }
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
            context_length=getattr(rollout_controller.config, "context_length", None),
            capture_path=str(getattr(rollout_controller.config, "worker_log_dir", ".")) + "/gateway_capture.jsonl",
        )
    rollout_controller.chat = rollout_controller.openai_chat_adapter.chat
    return rollout_controller
