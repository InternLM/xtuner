from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import ray
from ray.actor import ActorHandle

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, RolloutToolCall, SampleParams, Status
from xtuner.v1.rl.rollout.parser.factory import build_tool_call_parser
from xtuner.v1.rl.rollout.worker import RolloutConfig

from ..adapters.base import coerce_content_to_text
from ..adapters.trace import normalize_trace_payload
from ..core.exceptions import ContextLengthExceededError, ToolCallParseError
from ..core.models import (
    BackendHealth,
    CanonicalAssistantTurn,
    CanonicalGenerateRequest,
    CanonicalGenerateResponse,
    CanonicalReasoning,
    CanonicalReasoningBlock,
    CanonicalReasoningStep,
    CanonicalTextBlock,
    CanonicalToolCall,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolDefinition,
    CanonicalToolResultBlock,
    CanonicalUsage,
    ModelCapabilities,
    ModelCard,
)


class LocalRolloutBackend:
    def __init__(
        self,
        controller: ActorHandle,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | str | None = None,
        rollout_config: RolloutConfig | None = None,
    ):
        self._controller = controller
        self._config = rollout_config or self._resolve_rollout_config(controller)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        resolved_tokenizer = tokenizer
        if resolved_tokenizer is None:
            resolved_tokenizer = AutoTokenizer.from_pretrained(
                self._config.tokenizer_path,
                trust_remote_code=True,
            )
        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = resolved_tokenizer
        self._tool_call_parser = build_tool_call_parser(self._config.tool_call_parser)

    async def generate(self, request: CanonicalGenerateRequest) -> CanonicalGenerateResponse:
        rollout_state = self._canonical_request_to_rollout_state(request)
        rollout_state = await self._controller.generate.remote(rollout_state)
        self._raise_for_failed_rollout(rollout_state, request_id=str(rollout_state.uid))
        return self._rollout_state_to_canonical_response(rollout_state, request)

    async def health(self) -> BackendHealth:
        ready, details = await self._controller.get_ready_status.remote()
        return BackendHealth(
            ready=ready,
            status="ready" if ready else "unavailable",
            details=details,
        )

    async def list_models(self) -> list[ModelCard]:
        return [
            ModelCard(
                id=self._model_name,
                backend=self._config.rollout_backend,
                context_length=self._config.context_length,
            )
        ]

    async def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self._model_name,
            backend=self._config.rollout_backend,
            context_length=self._config.context_length,
            supports_stream=True,
            supports_tools=True,
            supports_cancel=False,
            supports_parallel_tool_calls=True,
            supports_reasoning=True,
        )

    async def cancel(self, request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "cancelled": False,
            "status": "not_supported",
        }

    @property
    def _model_name(self) -> str:
        return self._config.model_name or "rollout-controller"

    def _resolve_rollout_config(self, controller: ActorHandle) -> RolloutConfig:
        rollout_metadata = ray.get(controller.get_rollout_metadata.remote())
        return rollout_metadata["rollout_config"]

    def _canonical_request_to_rollout_state(self, canonical_request: CanonicalGenerateRequest) -> RolloutState:
        internal_messages = self._canonical_messages_to_backend_messages(canonical_request.messages)
        rollout_tools = self._canonical_tools_to_backend(canonical_request.tools)
        rollout_tool_choice = self._canonical_tool_choice_to_backend(canonical_request.tool_choice)
        prompt_ids = self._render_prompt_ids(internal_messages, rollout_tools)
        max_tokens = self._fit_max_tokens_to_context(prompt_ids, canonical_request.max_tokens)
        return RolloutState(
            uid=uuid4().int,
            message=internal_messages,
            prompt_ids=prompt_ids,
            tokens=prompt_ids,
            session_uid=canonical_request.metadata.get("session_uid"),
            tools=rollout_tools,
            tool_choice=rollout_tool_choice,
            sample_params=self._build_sample_params(canonical_request, max_tokens=max_tokens),
        )

    def _raise_for_failed_rollout(self, rollout_state: RolloutState, request_id: str) -> None:
        if rollout_state.status == Status.FAILED:
            raise RuntimeError(rollout_state.error_msg or f"Rollout generation failed for request {request_id}")

    def _rollout_state_to_canonical_response(
        self,
        rollout_state: RolloutState,
        canonical_request: CanonicalGenerateRequest,
    ) -> CanonicalGenerateResponse:
        request_id = str(rollout_state.uid)
        normal_text = rollout_state.response
        tool_calls = [
            self._rollout_tool_call_to_canonical(tool_call) for tool_call in (rollout_state.tool_calls or [])
        ]
        self._raise_for_unparsed_tool_call_markup(
            canonical_request=canonical_request,
            normal_text=normal_text,
            tool_calls=tool_calls,
        )
        reasoning_text = None
        if isinstance(rollout_state.extra_fields.get("reasoning_text"), str):
            reasoning_text = rollout_state.extra_fields.get("reasoning_text")
        content_blocks: list[Any] = []
        if reasoning_text:
            content_blocks.append(
                CanonicalReasoningBlock(
                    reasoning=CanonicalReasoning(
                        steps=[CanonicalReasoningStep(text=reasoning_text)],
                        metadata={"source_backend": "local_rollout"},
                    )
                )
            )
        if normal_text:
            content_blocks.append(CanonicalTextBlock(text=normal_text))
        for tool_call in tool_calls:
            content_blocks.append(CanonicalToolCallBlock(tool_call=tool_call))

        finish_reason = rollout_state.finish_reason or "stop"
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        prompt_tokens = len(rollout_state.prompt_ids or [])
        completion_tokens = self._count_completion_tokens(rollout_state)
        metadata = {
            "rollout_trace": self._build_rollout_trace_snapshot(rollout_state),
            "parallel_tool_calls": canonical_request.parallel_tool_calls,
            "source_backend": "local_rollout",
        }
        return CanonicalGenerateResponse(
            request_id=request_id,
            model=canonical_request.model or self._model_name,
            output=CanonicalAssistantTurn(content=content_blocks),
            finish_reason=finish_reason,
            usage=CanonicalUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metadata=metadata,
        )

    def _raise_for_unparsed_tool_call_markup(
        self,
        *,
        canonical_request: CanonicalGenerateRequest,
        normal_text: str | None,
        tool_calls: list[CanonicalToolCall],
    ) -> None:
        if self._tool_call_parser is None:
            return
        if self._tool_call_parser.should_reject_unparsed_markup(
            has_tools=bool(canonical_request.tools),
            text=normal_text,
            parsed_tool_calls=tool_calls,
        ):
            raise ToolCallParseError(
                "Tool-enabled generation returned tool-call markup that could not be parsed into structured "
                "tool calls."
            )

    def _canonical_messages_to_backend_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        backend_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                for block in message.content:
                    if isinstance(block, CanonicalToolResultBlock):
                        backend_messages.append(
                            {
                                "role": "tool",
                                "content": block.tool_result.output_text
                                if block.tool_result.output_text is not None
                                else coerce_content_to_text(block.tool_result.output),
                                "tool_call_id": block.tool_result.tool_call_id,
                            }
                        )
                continue

            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, CanonicalTextBlock):
                    if block.text:
                        text_chunks.append(block.text)
                elif isinstance(block, CanonicalReasoningBlock):
                    reasoning_text = "\n".join(step.text for step in block.reasoning.steps if step.text).strip()
                    if reasoning_text:
                        text_chunks.append(reasoning_text)
                elif isinstance(block, CanonicalToolCallBlock):
                    tool_calls.append(
                        {
                            "id": block.tool_call.id,
                            "type": "function",
                            "function": {
                                "name": block.tool_call.name,
                                "arguments": self._render_tool_arguments_for_template(block.tool_call),
                            },
                        }
                    )
            payload: dict[str, Any] = {"role": message.role, "content": "\n".join(text_chunks)}
            if message.name:
                payload["name"] = message.name
            if tool_calls:
                payload["tool_calls"] = tool_calls
            backend_messages.append(self._normalize_backend_message(payload))
        return backend_messages

    def _canonical_tools_to_backend(self, tools: list[CanonicalToolDefinition]) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return normalize_trace_payload(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters_json_schema,
                    },
                }
                for tool in tools
            ]
        )

    def _canonical_tool_choice_to_backend(self, tool_choice: CanonicalToolChoice | None) -> Any:
        if tool_choice is None:
            return None
        if tool_choice.type == "specific":
            return {
                "type": "function",
                "function": {"name": tool_choice.tool_name},
            }
        return tool_choice.type

    def _render_prompt_ids(
        self,
        internal_messages: list[dict[str, Any]],
        rollout_tools: list[dict[str, Any]] | None,
    ) -> list[int] | None:
        raw_prompt_ids = self._tokenizer.apply_chat_template(
            internal_messages,
            tools=rollout_tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(raw_prompt_ids, "get"):
            return raw_prompt_ids.get("input_ids")
        return list(raw_prompt_ids)

    def _build_sample_params(
        self,
        canonical_request: CanonicalGenerateRequest,
        *,
        max_tokens: int | None,
    ) -> SampleParams:
        kwargs = {
            "return_token_ids": True,
            "return_logprob": True,
            "stream": canonical_request.stream,
            "stops": canonical_request.stop,
            **{
                key: value
                for key, value in {
                    "n": canonical_request.metadata.get("n"),
                    "max_tokens": max_tokens if max_tokens is not None else canonical_request.max_tokens,
                    "temperature": canonical_request.temperature,
                    "top_p": canonical_request.top_p,
                    "top_k": canonical_request.metadata.get("top_k"),
                    "repetition_penalty": canonical_request.metadata.get("repetition_penalty"),
                    "presence_penalty": canonical_request.metadata.get("presence_penalty"),
                    "frequency_penalty": canonical_request.metadata.get("frequency_penalty"),
                    "min_tokens": canonical_request.metadata.get("min_tokens"),
                    "stop_token_ids": canonical_request.metadata.get("stop_token_ids"),
                    "skip_special_tokens": canonical_request.metadata.get("skip_special_tokens"),
                    "no_stop_trim": canonical_request.metadata.get("no_stop_trim"),
                    "spaces_between_special_tokens": canonical_request.metadata.get("spaces_between_special_tokens"),
                    "sampling_seed": canonical_request.metadata.get("sampling_seed"),
                    "return_routed_experts": canonical_request.metadata.get("return_routed_experts"),
                }.items()
                if value is not None
            },
        }
        return SampleParams(**kwargs)

    def _fit_max_tokens_to_context(
        self,
        prompt_ids: list[int] | None,
        requested_max_tokens: int | None,
    ) -> int | None:
        context_length = self._config.context_length
        if context_length is None or prompt_ids is None or requested_max_tokens is None:
            return requested_max_tokens
        prompt_tokens = len(prompt_ids)
        available_completion_tokens = context_length - prompt_tokens
        if available_completion_tokens <= 0:
            raise ContextLengthExceededError(prompt_tokens=prompt_tokens, context_length=context_length)
        return min(requested_max_tokens, available_completion_tokens)

    def _count_completion_tokens(self, rollout_state: RolloutState) -> int:
        if rollout_state.response_ids is not None:
            return len(rollout_state.response_ids)
        if rollout_state.response:
            return len(self._tokenizer(rollout_state.response, add_special_tokens=False)["input_ids"])
        return 0

    def _rollout_tool_call_to_canonical(self, tool_call: RolloutToolCall) -> CanonicalToolCall:
        return CanonicalToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
            raw_arguments_text=tool_call.function.raw_arguments_text,
        )

    def _build_rollout_trace_snapshot(self, rollout_state: RolloutState) -> dict[str, Any]:
        return {
            "session_uid": rollout_state.session_uid,
            "status": rollout_state.status.value,
            "rollout_finish_reason": rollout_state.finish_reason,
            "prompt_ids": list(rollout_state.prompt_ids or []),
            "response_ids": list(rollout_state.response_ids or []),
            "logprobs": None if rollout_state.logprobs is None else list(rollout_state.logprobs),
            "routed_experts": normalize_trace_payload(rollout_state.routed_experts),
            "internal_messages": normalize_trace_payload(rollout_state.message),
            "rollout_tools": normalize_trace_payload(rollout_state.tools),
            "rollout_tool_choice": normalize_trace_payload(rollout_state.tool_choice),
            "rollout_sample_params": normalize_trace_payload(
                rollout_state.sample_params.model_dump(mode="python", exclude_none=True)
            ),
            "input_text": self._decode_prompt_ids(rollout_state),
            "output_text": self._render_rollout_output_text(rollout_state),
        }

    def _render_rollout_output_text(self, rollout_state: RolloutState) -> str:
        parts = []
        if rollout_state.response:
            parts.append(rollout_state.response)
        for rollout_tool_call in rollout_state.tool_calls or []:
            tool_call = self._rollout_tool_call_to_canonical(rollout_tool_call)
            arguments = self._stringify_tool_arguments(tool_call)
            parts.append(f"<tool_use name={tool_call.name}>{arguments}</tool_use>")
        return "\n".join(parts)

    def _decode_prompt_ids(self, rollout_state: RolloutState) -> str:
        """Decode prompt token IDs to text without re-running the chat
        template."""
        try:
            return self._tokenizer.decode(rollout_state.prompt_ids or [], skip_special_tokens=False)
        except Exception:
            return ""

    def _stringify_tool_arguments(self, tool_call: CanonicalToolCall) -> str:
        if tool_call.raw_arguments_text is not None:
            return tool_call.raw_arguments_text
        if isinstance(tool_call.arguments, str):
            return tool_call.arguments
        return json.dumps(tool_call.arguments if tool_call.arguments is not None else {}, ensure_ascii=False)

    def _render_tool_arguments_for_template(self, tool_call: CanonicalToolCall) -> dict[str, Any]:
        arguments = tool_call.arguments
        if isinstance(arguments, dict):
            return arguments
        if tool_call.raw_arguments_text is not None:
            try:
                decoded = json.loads(tool_call.raw_arguments_text)
            except Exception:
                return {"raw": tool_call.raw_arguments_text}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except Exception:
                return {"raw": arguments}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        return {"value": arguments}

    def _normalize_backend_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize a backend message dict: remove None values and sort keys."""
        return {
            str(key): val for key, val in sorted(payload.items(), key=lambda item: str(item[0])) if val is not None
        }
