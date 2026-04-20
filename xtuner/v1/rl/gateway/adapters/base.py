from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status

from ..core.models import (
    CanonicalGenerateRequest,
    CanonicalGenerateResponse,
    CanonicalReasoningBlock,
    CanonicalTextBlock,
    CanonicalToolCall,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from .capture import append_gateway_capture_record, render_blocks_as_text
from .collector import reset_current_trace_collector, set_current_trace_collector
from .trace import (
    ChatTraceRecord,
    ChatTraceStore,
    build_api_key_trace_key,
    normalize_trace_payload,
    snapshot_routed_experts,
)


GenerateHandler = Callable[[CanonicalGenerateRequest], Awaitable[CanonicalGenerateResponse]]
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
logger = logging.getLogger(__name__)


def coerce_content_to_text(content: Any) -> str | None:
    """Coerce arbitrary content (str, list of blocks, dict) to a plain
    string."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                text_chunks.append(str(item.get("text", "")))
        joined = "\n".join(chunk for chunk in text_chunks if chunk)
        return joined or None
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def stringify_tool_arguments(tool_call: CanonicalToolCall) -> str:
    if tool_call.raw_arguments_text is not None:
        return tool_call.raw_arguments_text
    if isinstance(tool_call.arguments, str):
        return tool_call.arguments
    return json.dumps(tool_call.arguments if tool_call.arguments is not None else {}, ensure_ascii=False)


class BaseChatAPIAdapter(ABC, Generic[RequestT, ResponseT]):
    def __init__(
        self,
        generate_handler: GenerateHandler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None,
        *,
        capture_folder: str | None = None,
        trace_store_max_entries: int = 10000,
    ):
        self._generate_handler = generate_handler
        self._tokenizer = tokenizer
        self._capture_folder = capture_folder
        self._trace_store = ChatTraceStore(max_entries=trace_store_max_entries)

    async def handle_request(self, request: RequestT, *, api_key: str | None = None) -> ResponseT:
        self.validate_request(request)
        canonical_request = self.request_to_canonical_request(request)
        rollout_states: list[RolloutState] = []
        token = set_current_trace_collector(rollout_states)
        try:
            canonical_response = await self._generate_handler(canonical_request)
        finally:
            reset_current_trace_collector(token)
        response = self.canonical_response_to_protocol_response(canonical_response, request)
        record_trace_key = build_api_key_trace_key(api_key)
        self._trace_store.append(
            self._build_trace_record(
                record_trace_key,
                request,
                response,
                canonical_response,
                rollout_states=rollout_states,
            )
        )
        self._write_capture_record(
            request=request,
            response=response,
            canonical_response=canonical_response,
            rollout_states=rollout_states,
            api_key=api_key,
        )
        return response

    def get_trace_records(self, trace_key: str) -> list[ChatTraceRecord]:
        return self._trace_store.get(trace_key)

    def pop_trace_records(self, trace_key: str) -> list[ChatTraceRecord]:
        return self._trace_store.pop(trace_key)

    def clear_trace_records(self, trace_key: str) -> None:
        self._trace_store.clear(trace_key)

    def _build_trace_record(
        self,
        trace_key: str,
        request: RequestT,
        response: ResponseT,
        canonical_response: CanonicalGenerateResponse,
        rollout_states: list[RolloutState] | None = None,
    ) -> ChatTraceRecord:
        request_snapshot = self.normalize_request(request)
        response_snapshot = self.normalize_response(response)
        rollout_trace = self._get_rollout_trace(canonical_response)
        status = rollout_trace.get("status", Status.COMPLETED.value)
        output_text = rollout_trace.get("output_text") or render_blocks_as_text(
            self._build_output_message_list(canonical_response)
        )
        return ChatTraceRecord(
            trace_key=trace_key,
            request_snapshot=request_snapshot,
            response_snapshot=response_snapshot,
            prompt_ids=list(rollout_trace.get("prompt_ids") or []),
            response_ids=list(rollout_trace.get("response_ids") or []),
            input_text=rollout_trace.get("input_text", ""),
            output_text=output_text,
            logprobs=rollout_trace.get("logprobs"),
            routed_experts=snapshot_routed_experts(rollout_trace.get("routed_experts")),
            finish_reason=rollout_trace.get("rollout_finish_reason") or canonical_response.finish_reason,
            status=Status(status) if isinstance(status, str) else status,
            request_id=canonical_response.request_id,
            rollout_states=list(rollout_states) if rollout_states else [],
        )

    def _write_capture_record(
        self,
        request: RequestT,
        response: ResponseT,
        canonical_response: CanonicalGenerateResponse,
        rollout_states: list[RolloutState] | None = None,
        api_key: str | None = None,
    ) -> None:
        if self._capture_folder is None:
            return
        rollout_trace = self._get_rollout_trace(canonical_response)
        try:
            response_snapshot = self.normalize_response(response)
            response_finish_reason = (
                response_snapshot.get("stop_reason")
                or response_snapshot.get("finish_reason")
                or canonical_response.finish_reason
            )
            output_messages = self._build_output_message_list(canonical_response)
            append_gateway_capture_record(
                self._capture_folder,
                {
                    "protocol": self.__class__.__name__,
                    "request_id": canonical_response.request_id,
                    "session_uid": rollout_trace.get("session_uid"),
                    "status": rollout_trace.get("status", Status.COMPLETED.value),
                    "finish_reason": response_finish_reason,
                    "rollout_finish_reason": rollout_trace.get("rollout_finish_reason"),
                    "prompt_tokens": canonical_response.usage.prompt_tokens,
                    "completion_tokens": canonical_response.usage.completion_tokens,
                    "request": self.normalize_request(request),
                    "response": response_snapshot,
                    "internal_messages": rollout_trace.get("internal_messages"),
                    "rollout_tools": rollout_trace.get("rollout_tools"),
                    "rollout_tool_choice": rollout_trace.get("rollout_tool_choice"),
                    "rollout_sample_params": rollout_trace.get("rollout_sample_params"),
                    "output_messages": output_messages,
                    "input_text": rollout_trace.get("input_text", ""),
                    "output_text": render_blocks_as_text(output_messages),
                },
                api_key=api_key,
            )
        except Exception:
            logger.warning(f"Failed to write gateway capture record to {self._capture_folder}", exc_info=True)
            return

    def _get_rollout_trace(self, canonical_response: CanonicalGenerateResponse) -> dict[str, Any]:
        trace_payload = canonical_response.metadata.get("rollout_trace", {})
        if not isinstance(trace_payload, dict):
            return {}
        return trace_payload

    def _build_output_message_list(
        self,
        canonical_response: CanonicalGenerateResponse,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for block in canonical_response.output.content:
            if isinstance(block, CanonicalTextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, CanonicalReasoningBlock):
                reasoning_text = "\n".join(step.text for step in block.reasoning.steps if step.text).strip()
                if reasoning_text:
                    content.append({"type": "reasoning", "text": reasoning_text})
            elif isinstance(block, CanonicalToolCallBlock):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.tool_call.id,
                        "name": block.tool_call.name,
                        "input": normalize_trace_payload(block.tool_call.arguments),
                    }
                )
            elif isinstance(block, CanonicalToolResultBlock):
                tool_result_content = block.tool_result.output
                if tool_result_content is None:
                    tool_result_content = block.tool_result.output_text or ""
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_result.tool_call_id,
                        "content": normalize_trace_payload(tool_result_content),
                    }
                )
        return [{"role": "assistant", "content": content or ""}]

    @abstractmethod
    def validate_request(self, request: RequestT) -> None:
        raise NotImplementedError

    @abstractmethod
    def request_to_canonical_request(self, request: RequestT) -> CanonicalGenerateRequest:
        raise NotImplementedError

    @abstractmethod
    def normalize_request(self, request: RequestT) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def normalize_response(self, response: ResponseT) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def canonical_response_to_protocol_response(
        self,
        canonical_response: CanonicalGenerateResponse,
        request: RequestT,
    ) -> ResponseT:
        raise NotImplementedError
