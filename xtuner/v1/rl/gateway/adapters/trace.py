from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from xtuner.v1.data_proto.rl_data import Status


DEFAULT_CHAT_TRACE_KEY = "__default__"


def build_api_key_trace_key(api_key: str | None) -> str:
    if not api_key:
        return DEFAULT_CHAT_TRACE_KEY
    api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"api_key_{api_key_hash}"


def normalize_trace_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return normalize_trace_payload(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, dict):
        return {
            str(key): normalize_trace_payload(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if val is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_trace_payload(item) for item in value]
    return value


def snapshot_routed_experts(routed_experts: Any) -> Any:
    if routed_experts is None:
        return None
    try:
        import ray

        if isinstance(routed_experts, ray.ObjectRef):
            return routed_experts
    except Exception:
        pass
    return deepcopy(routed_experts)


@dataclass
class ChatTraceRecord:
    trace_key: str
    request_snapshot: dict[str, Any]
    response_snapshot: dict[str, Any]
    prompt_ids: list[int]
    response_ids: list[int]
    input_text: str
    output_text: str
    logprobs: list[float] | None
    routed_experts: Any
    finish_reason: str | None
    status: Status
    sequence: int = -1
    created_at: float = 0.0
    request_id: str | None = None


class ChatTraceStore:
    def __init__(self, max_entries: int = 10000):
        self._max_entries = max_entries
        self._records: OrderedDict[str, OrderedDict[int, ChatTraceRecord]] = OrderedDict()
        self._record_order: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._next_sequence: dict[str, int] = {}
        self._lock = threading.RLock()

    def append(self, record: ChatTraceRecord) -> ChatTraceRecord:
        with self._lock:
            sequence = self._next_sequence.get(record.trace_key, 0)
            self._next_sequence[record.trace_key] = sequence + 1
            record.sequence = sequence
            record.created_at = time.time()
            records = self._records.setdefault(record.trace_key, OrderedDict())
            records[sequence] = record
            self._record_order[(record.trace_key, sequence)] = None
            self._evict_if_needed()
            return record

    def get(self, trace_key: str) -> list[ChatTraceRecord]:
        with self._lock:
            records = self._records.get(trace_key)
            if records is None:
                return []
            return list(records.values())

    def pop(self, trace_key: str) -> list[ChatTraceRecord]:
        with self._lock:
            records = self._records.pop(trace_key, None)
            self._next_sequence.pop(trace_key, None)
            if records is None:
                return []
            for sequence in records:
                self._record_order.pop((trace_key, sequence), None)
            return list(records.values())

    def clear(self, trace_key: str) -> None:
        with self._lock:
            records = self._records.pop(trace_key, None)
            self._next_sequence.pop(trace_key, None)
            if records is None:
                return
            for sequence in records:
                self._record_order.pop((trace_key, sequence), None)

    def _evict_if_needed(self) -> None:
        while len(self._record_order) > self._max_entries:
            (trace_key, sequence), _ = self._record_order.popitem(last=False)
            records = self._records.get(trace_key)
            if records is None:
                continue
            records.pop(sequence, None)
            if not records:
                self._records.pop(trace_key, None)
                self._next_sequence.pop(trace_key, None)
