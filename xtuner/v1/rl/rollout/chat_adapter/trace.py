from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Any

from pydantic import BaseModel

from xtuner.v1.data_proto.rl_data import Status


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


def build_trace_hash(request_snapshot: Any, response_snapshot: Any) -> str:
    payload = {
        "request": normalize_trace_payload(request_snapshot),
        "response": normalize_trace_payload(response_snapshot),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    response_hash: str
    request_snapshot: dict[str, Any]
    response_snapshot: dict[str, Any]
    prompt_ids: list[int]
    response_ids: list[int]
    logprobs: list[float] | None
    routed_experts: Any
    finish_reason: str | None
    status: Status
    request_id: str | None = None


class ChatTraceStore:
    def __init__(self, max_entries: int = 10000):
        self._max_entries = max_entries
        self._records: OrderedDict[str, ChatTraceRecord] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, record: ChatTraceRecord) -> None:
        with self._lock:
            self._records[record.response_hash] = record
            self._records.move_to_end(record.response_hash)
            while len(self._records) > self._max_entries:
                self._records.popitem(last=False)

    def get(self, response_hash: str) -> ChatTraceRecord | None:
        with self._lock:
            record = self._records.get(response_hash)
            if record is None:
                return None
            self._records.move_to_end(response_hash)
            return record

    def build_hash(self, request_snapshot: Any, response_snapshot: Any) -> str:
        return build_trace_hash(request_snapshot=request_snapshot, response_snapshot=response_snapshot)
