from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState

from .collector import append_current_trace_rollout_state
from .trace import ChatTraceRecord, ChatTraceStore, snapshot_routed_experts


GenerateHandler = Callable[[RolloutState], Awaitable[RolloutState]]
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseChatAPIAdapter(ABC, Generic[RequestT, ResponseT]):
    def __init__(
        self,
        generate_handler: GenerateHandler,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None,
        *,
        trace_store_max_entries: int = 10000,
    ):
        self._generate_handler = generate_handler
        self._tokenizer = tokenizer
        self._trace_store = ChatTraceStore(max_entries=trace_store_max_entries)

    async def handle_request(self, request: RequestT) -> ResponseT:
        self.validate_request(request)
        rollout_state = self.request_to_rollout_state(request)
        if rollout_state.uid is None:
            raise ValueError("request_to_rollout_state must assign rollout_state.uid before generate is called.")
        request_id = str(rollout_state.uid)
        rollout_state = await self._generate_handler(rollout_state)
        append_current_trace_rollout_state(rollout_state)

        self.raise_for_failed_response(rollout_state, request_id)
        response = self.rollout_state_to_response(rollout_state, request)
        self._trace_store.put(self._build_trace_record(request, response, rollout_state, request_id))
        return response

    def get_trace_by_request_response(self, request: RequestT, response: ResponseT) -> ChatTraceRecord | None:
        response_hash = self._trace_store.build_hash(
            request_snapshot=self.normalize_request(request),
            response_snapshot=self.normalize_response(response),
        )
        return self._trace_store.get(response_hash)

    def get_trace_by_response_hash(self, response_hash: str) -> ChatTraceRecord | None:
        return self._trace_store.get(response_hash)

    def _build_trace_record(
        self,
        request: RequestT,
        response: ResponseT,
        rollout_state: RolloutState,
        request_id: str,
    ) -> ChatTraceRecord:
        request_snapshot = self.normalize_request(request)
        response_snapshot = self.normalize_response(response)
        response_hash = self._trace_store.build_hash(
            request_snapshot=request_snapshot,
            response_snapshot=response_snapshot,
        )
        return ChatTraceRecord(
            response_hash=response_hash,
            request_snapshot=request_snapshot,
            response_snapshot=response_snapshot,
            prompt_ids=list(rollout_state.prompt_ids or []),
            response_ids=list(rollout_state.response_ids or []),
            logprobs=None if rollout_state.logprobs is None else list(rollout_state.logprobs),
            routed_experts=snapshot_routed_experts(rollout_state.routed_experts),
            finish_reason=rollout_state.finish_reason,
            status=rollout_state.status,
            request_id=request_id,
        )

    @abstractmethod
    def validate_request(self, request: RequestT) -> None:
        return None

    @abstractmethod
    def request_to_rollout_state(self, request: RequestT) -> RolloutState:
        raise NotImplementedError

    @abstractmethod
    def raise_for_failed_response(self, response: RolloutState, request_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize_request(self, request: RequestT) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def normalize_response(self, response: ResponseT) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def rollout_state_to_response(
        self,
        rollout_state: RolloutState,
        request: RequestT,
    ) -> ResponseT:
        raise NotImplementedError

    @abstractmethod
    def build_output_message_list(
        self,
        rollout_state: RolloutState,
        request: RequestT,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError
