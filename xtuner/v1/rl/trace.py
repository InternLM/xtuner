from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import json
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Literal, Sequence, TextIO, cast

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils import get_logger


logger = get_logger()

TRACE_JSONL_FLUSH_INTERVAL_S = 1.0
TRACE_JSONL_FLUSH_EVENTS = 1024
TRACE_JSONL_FLUSH_BYTES = 1 * 1024 * 1024
TRACE_JSONL_SHARD_BYTES = 256 * 1024 * 1024
TRACE_JSONL_QUEUE_MAX_EVENTS = 100_000
TRACE_JSONL_FLUSH_TIMEOUT_S = 0.2
TRACE_JSONL_CLOSE_TIMEOUT_S = 2.0
TRACE_JSONL_BASENAME = "producer_trace"
TRACE_ENV_ENABLED = "XTUNER_TRACE_ENABLED"
TRACE_ENV_OUTPUT_DIR = "XTUNER_TRACE_OUTPUT_DIR"
TRACE_ENV_MAX_EVENTS = "XTUNER_TRACE_MAX_EVENTS"
TRACE_ENV_MAX_EVENTS_PER_TRACE = "XTUNER_TRACE_MAX_EVENTS_PER_TRACE"
TraceViewerScope = Literal["latest-produce-batch", "all"]
TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH: TraceViewerScope = "latest-produce-batch"
TRACE_VIEWER_SCOPE_ALL: TraceViewerScope = "all"
TRACE_EXTRA_TRAIN_STEP = "_trace_train_step"
TRACE_EXTRA_MODEL_STEP = "_trace_model_step"
TRACE_EXTRA_PRODUCER_FUTURE_STEP = "_trace_producer_future_step"
TRACE_EXTRA_PRODUCE_BATCH_ID = "_trace_produce_batch_id"


# Config and event schema.
class TraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Whether producer task tracing is enabled.
    enabled: bool = False
    # Directory that receives producer_trace_*.jsonl shards. If None, only memory tracing is used.
    output_dir: str | Path | None = None
    # Max events retained in memory across all traces.
    max_events: int = 100_000
    # Max events retained in memory for one trace_id.
    max_events_per_trace: int = 256
    # Whether rank0 should start the live producer trace viewer when tracing is enabled.
    viewer_enabled: bool = True
    # Host for the live producer trace viewer. Use 0.0.0.0 explicitly when remote access is needed.
    viewer_host: str = "127.0.0.1"
    # Port for the live producer trace viewer. 0 asks the OS to pick an available port.
    viewer_port: int = 0
    # Browser polling interval for the live producer trace viewer.
    viewer_refresh_interval_s: float = 1.0
    # Default viewer data scope. The latest produce batch keeps the UI focused during training.
    viewer_scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    stage: str
    timestamp_s: float
    status: str | None = None
    task_name: str | None = None
    uid: int | str | None = None
    session_uid: int | str | None = None
    train_step: int | None = None
    model_step: int | None = None
    producer_future_step: int | None = None
    produce_batch_id: str | None = None
    worker_rank: int | None = None
    elapsed_s: float | None = None
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(
            trace_id=str(data["trace_id"]),
            stage=str(data["stage"]),
            timestamp_s=float(data["timestamp_s"]),
            status=data.get("status"),
            task_name=data.get("task_name"),
            uid=data.get("uid"),
            session_uid=data.get("session_uid"),
            train_step=data.get("train_step"),
            model_step=data.get("model_step"),
            producer_future_step=data.get("producer_future_step"),
            produce_batch_id=data.get("produce_batch_id"),
            worker_rank=data.get("worker_rank"),
            elapsed_s=data.get("elapsed_s"),
            error_msg=data.get("error_msg"),
        )


class TraceEventBuilder:
    @classmethod
    def trace_id(cls, task_name: str | None, uid: int | str | None) -> str | None:
        if uid is None:
            return None
        return f"{task_name or 'unknown'}:{uid}"

    @classmethod
    def produce_batch_id(
        cls,
        train_step: int | None,
        model_step: int | None,
        producer_future_step: int | None,
    ) -> str | None:
        if train_step is None and model_step is None and producer_future_step is None:
            return None
        return (
            f"train_step={cls._format_batch_value(train_step)}/"
            f"model_step={cls._format_batch_value(model_step)}/"
            f"producer_future_step={cls._format_batch_value(producer_future_step)}"
        )

    @classmethod
    def build(
        cls,
        target: RolloutState | None,
        stage: str,
        *,
        task_name: str | None = None,
        uid: int | str | None = None,
        session_uid: int | str | None = None,
        status: Any | None = None,
        train_step: int | None = None,
        model_step: int | None = None,
        producer_future_step: int | None = None,
        produce_batch_id: str | None = None,
        worker_rank: int | None = None,
        elapsed_s: float | None = None,
        error_msg: str | None = None,
        timestamp_s: float | None = None,
    ) -> TraceEvent | None:
        resolved_task_name = task_name if task_name is not None else getattr(target, "task_name", None)
        resolved_uid = uid if uid is not None else getattr(target, "uid", None)
        trace_id = cls.trace_id(resolved_task_name, resolved_uid)
        if trace_id is None:
            return None

        resolved_status = status if status is not None else getattr(target, "status", None)
        resolved_session_uid = session_uid if session_uid is not None else getattr(target, "session_uid", None)
        trace_extra_fields = cls._extra_fields(target)
        if train_step is None:
            train_step = trace_extra_fields.get(TRACE_EXTRA_TRAIN_STEP)
        if model_step is None:
            model_step = trace_extra_fields.get(TRACE_EXTRA_MODEL_STEP, getattr(target, "model_step", None))
        if producer_future_step is None:
            producer_future_step = trace_extra_fields.get(TRACE_EXTRA_PRODUCER_FUTURE_STEP)
        if produce_batch_id is None:
            produce_batch_id = trace_extra_fields.get(TRACE_EXTRA_PRODUCE_BATCH_ID)
        if produce_batch_id is None:
            produce_batch_id = cls.produce_batch_id(train_step, model_step, producer_future_step)

        return TraceEvent(
            trace_id=trace_id,
            stage=stage,
            timestamp_s=time.time() if timestamp_s is None else timestamp_s,
            status=cls._stringify_status(resolved_status),
            task_name=resolved_task_name,
            uid=resolved_uid,
            session_uid=resolved_session_uid,
            train_step=train_step,
            model_step=model_step,
            producer_future_step=producer_future_step,
            produce_batch_id=produce_batch_id,
            worker_rank=worker_rank,
            elapsed_s=elapsed_s,
            error_msg=error_msg,
        )

    @staticmethod
    def short_error(exc: BaseException, max_len: int = 500) -> str:
        message = f"{type(exc).__name__}: {exc}"
        if len(message) <= max_len:
            return message
        return message[: max_len - 3] + "..."

    @staticmethod
    def _extra_fields(target: RolloutState | None) -> dict[str, Any]:
        extra_fields = getattr(target, "extra_fields", None)
        if isinstance(extra_fields, dict):
            return cast(dict[str, Any], extra_fields)
        return {}

    @staticmethod
    def _format_batch_value(value: int | None) -> str:
        return "none" if value is None else str(value)

    @staticmethod
    def _stringify_status(status: Any) -> str | None:
        if status is None:
            return None
        value = getattr(status, "value", None)
        if value is not None:
            return str(value)
        return str(status)


@dataclass(frozen=True)
class _StoredEvent:
    seq: int
    event: TraceEvent


@dataclass
class _FlushRequest:
    done: threading.Event


@dataclass
class _CloseRequest:
    done: threading.Event


# Durable JSONL writer. All writes are best-effort and must not block training.
class BufferedTraceJsonlWriter:
    _writer_seq: ClassVar[int] = 0
    _writer_seq_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer_id = self._next_writer_id()
        self._queue: queue.Queue[str | _FlushRequest | _CloseRequest] = queue.Queue(
            maxsize=TRACE_JSONL_QUEUE_MAX_EVENTS
        )
        self._thread = threading.Thread(target=self._run, name="TraceJsonlWriter", daemon=True)
        self._closed = False
        self._failed = False
        self._dropped_events = 0
        self._closed_lock = threading.Lock()
        self._file: TextIO | None = None
        self._shard_idx = 0
        self._bytes_written = 0
        self._thread.start()

    def append(self, event: TraceEvent) -> bool:
        with self._closed_lock:
            if self._closed or self._failed:
                return False
            line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                self._record_drop("queue full")
                return False
            except Exception:
                self._record_drop("queue append failed")
                return False
            return True

    def flush(self, timeout_s: float = TRACE_JSONL_FLUSH_TIMEOUT_S) -> bool:
        with self._closed_lock:
            if self._closed or self._failed:
                return False
            request = _FlushRequest(threading.Event())
            try:
                self._queue.put(request, timeout=max(0.0, min(timeout_s, TRACE_JSONL_FLUSH_TIMEOUT_S)))
            except Exception:
                return False
        return request.done.wait(timeout=max(0.0, timeout_s))

    def close(self, timeout_s: float = TRACE_JSONL_CLOSE_TIMEOUT_S) -> bool:
        with self._closed_lock:
            if self._closed:
                return True
            self._closed = True
            request = _CloseRequest(threading.Event())
            try:
                self._queue.put(request, timeout=max(0.0, min(timeout_s, TRACE_JSONL_FLUSH_TIMEOUT_S)))
            except Exception:
                return False
        closed = request.done.wait(timeout=max(0.0, timeout_s))
        self._thread.join(timeout=max(0.0, timeout_s))
        return closed

    def _run(self) -> None:
        batch: list[str] = []
        batch_bytes = 0
        last_flush = time.monotonic()
        while True:
            timeout = TRACE_JSONL_FLUSH_INTERVAL_S if batch else None
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._safe_write_batch(batch)
                batch = []
                batch_bytes = 0
                last_flush = time.monotonic()
                continue

            if isinstance(item, str):
                batch.append(item)
                batch_bytes += len(item.encode("utf-8"))
                self._queue.task_done()
                should_flush = (
                    len(batch) >= TRACE_JSONL_FLUSH_EVENTS
                    or batch_bytes >= TRACE_JSONL_FLUSH_BYTES
                    or time.monotonic() - last_flush >= TRACE_JSONL_FLUSH_INTERVAL_S
                )
                if should_flush:
                    self._safe_write_batch(batch)
                    batch = []
                    batch_bytes = 0
                    last_flush = time.monotonic()
                continue

            if isinstance(item, _FlushRequest):
                self._safe_write_batch(batch)
                batch = []
                batch_bytes = 0
                self._safe_flush_file()
                last_flush = time.monotonic()
                item.done.set()
                self._queue.task_done()
                continue

            if isinstance(item, _CloseRequest):
                self._safe_write_batch(batch)
                self._safe_flush_file()
                self._safe_close_file()
                item.done.set()
                self._queue.task_done()
                return

    def _safe_write_batch(self, batch: list[str]) -> bool:
        try:
            self._write_batch(batch)
        except Exception:
            self._mark_failed("Failed to write producer trace JSONL batch")
            return False
        return True

    def _write_batch(self, batch: list[str]) -> None:
        if not batch:
            return
        for line in batch:
            line_bytes = len(line.encode("utf-8"))
            self._ensure_file(line_bytes)
            assert self._file is not None
            self._file.write(line)
            self._bytes_written += line_bytes
        self._flush_file()

    def _ensure_file(self, next_bytes: int) -> None:
        if self._file is None:
            self._open_file()
            return
        if self._bytes_written > 0 and self._bytes_written + next_bytes > TRACE_JSONL_SHARD_BYTES:
            self._close_file()
            self._shard_idx += 1
            self._open_file()

    def _open_file(self) -> None:
        path = self.output_dir / f"{TRACE_JSONL_BASENAME}_{self._writer_id}_{self._shard_idx:06d}.jsonl"
        self._file = path.open("a", encoding="utf-8")
        self._bytes_written = path.stat().st_size if path.exists() else 0

    def _flush_file(self) -> None:
        if self._file is not None:
            self._file.flush()

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _safe_flush_file(self) -> bool:
        try:
            self._flush_file()
        except Exception:
            self._mark_failed("Failed to flush producer trace JSONL file")
            return False
        return True

    def _safe_close_file(self) -> bool:
        try:
            self._close_file()
        except Exception:
            self._mark_failed("Failed to close producer trace JSONL file")
            return False
        return True

    def _mark_failed(self, message: str) -> None:
        with self._closed_lock:
            if self._failed:
                return
            self._failed = True
        logger.exception(message)

    def _record_drop(self, reason: str) -> None:
        self._dropped_events += 1
        if self._dropped_events == 1 or self._dropped_events % 1000 == 0:
            logger.warning(
                "Dropped producer trace events because %s. dropped_events=%s",
                reason,
                self._dropped_events,
            )

    @classmethod
    def _next_writer_id(cls) -> str:
        with cls._writer_seq_lock:
            cls._writer_seq += 1
            seq = cls._writer_seq
        return f"{os.getpid()}_{seq:04d}"


# In-process state store used by the local tracer and tests.
class InMemoryTraceStore:
    def __init__(self, config: TraceConfig | None = None) -> None:
        self.config = config or TraceConfig()
        self._timelines: dict[str, deque[_StoredEvent]] = {}
        self._global_events: OrderedDict[int, _StoredEvent] = OrderedDict()
        self._latest: dict[str, TraceEvent] = {}
        self._seq = 0
        self._lock = threading.RLock()
        self._jsonl_writer = None
        if self.config.enabled and self.config.output_dir is not None:
            self._jsonl_writer = BufferedTraceJsonlWriter(self.config.output_dir)

    def append(self, event: TraceEvent) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            self._seq += 1
            stored = _StoredEvent(seq=self._seq, event=event)
            timeline = self._timelines.setdefault(event.trace_id, deque())
            timeline.append(stored)
            self._global_events[stored.seq] = stored
            self._latest[event.trace_id] = event
            self._enforce_per_trace_limit(event.trace_id)
            self._enforce_global_limit()
        if self._jsonl_writer is not None:
            try:
                self._jsonl_writer.append(event)
            except Exception:
                logger.exception("Failed to enqueue producer trace JSONL event")

    def get_timeline(self, trace_id: str) -> list[TraceEvent]:
        with self._lock:
            return [stored.event for stored in self._timelines.get(trace_id, ())]

    def get_all_timelines(self) -> dict[str, list[TraceEvent]]:
        with self._lock:
            return {trace_id: [stored.event for stored in timeline] for trace_id, timeline in self._timelines.items()}

    def get_latest(self, trace_id: str) -> TraceEvent | None:
        with self._lock:
            return self._latest.get(trace_id)

    def query_latest(self) -> dict[str, TraceEvent]:
        with self._lock:
            return dict(self._latest)

    def has_stage(self, trace_id: str, stage: str) -> bool:
        with self._lock:
            return any(stored.event.stage == stage for stored in self._timelines.get(trace_id, ()))

    def flush_jsonl(self) -> bool:
        if self._jsonl_writer is not None:
            try:
                return self._jsonl_writer.flush()
            except Exception:
                logger.exception("Failed to flush producer trace JSONL writer")
                return False
        return True

    def close(self) -> bool:
        if self._jsonl_writer is not None:
            try:
                closed = self._jsonl_writer.close()
            except Exception:
                logger.exception("Failed to close producer trace JSONL writer")
                closed = False
            self._jsonl_writer = None
            return closed
        return True

    def _enforce_per_trace_limit(self, trace_id: str) -> None:
        limit = max(1, self.config.max_events_per_trace)
        timeline = self._timelines.get(trace_id)
        if timeline is None:
            return
        while len(timeline) > limit:
            removed = timeline.popleft()
            self._global_events.pop(removed.seq, None)
        if not timeline:
            self._timelines.pop(trace_id, None)
            self._latest.pop(trace_id, None)
        else:
            self._latest[trace_id] = timeline[-1].event

    def _enforce_global_limit(self) -> None:
        limit = max(1, self.config.max_events)
        while len(self._global_events) > limit:
            _, removed = self._global_events.popitem(last=False)
            timeline = self._timelines.get(removed.event.trace_id)
            if timeline is not None:
                try:
                    timeline.remove(removed)
                except ValueError:
                    pass
                if not timeline:
                    self._timelines.pop(removed.event.trace_id, None)
                    self._latest.pop(removed.event.trace_id, None)
                else:
                    self._latest[removed.event.trace_id] = timeline[-1].event


# Recorder API used by trace_event, trace_span, and trace_function.
class TraceRecorder:
    def __init__(self, store: InMemoryTraceStore) -> None:
        self.store = store

    def record(
        self,
        target: RolloutState | None,
        stage: str,
        *,
        task_name: str | None = None,
        uid: int | str | None = None,
        session_uid: int | str | None = None,
        status: Any | None = None,
        train_step: int | None = None,
        model_step: int | None = None,
        producer_future_step: int | None = None,
        produce_batch_id: str | None = None,
        worker_rank: int | None = None,
        elapsed_s: float | None = None,
        error_msg: str | None = None,
        timestamp_s: float | None = None,
    ) -> TraceEvent | None:
        event = build_trace_event(
            target,
            stage,
            task_name=task_name,
            uid=uid,
            session_uid=session_uid,
            status=status,
            train_step=train_step,
            model_step=model_step,
            producer_future_step=producer_future_step,
            produce_batch_id=produce_batch_id,
            worker_rank=worker_rank,
            elapsed_s=elapsed_s,
            error_msg=error_msg,
            timestamp_s=timestamp_s,
        )
        if event is None:
            return None
        try:
            self.store.append(event)
        except Exception:
            logger.exception("Failed to append trace event stage=%s trace_id=%s", stage, event.trace_id)
            return None
        return event

    def record_many(self, targets: Iterable[RolloutState], stage: str, **kwargs: Any) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        for target in targets:
            event = self.record(target, stage, **kwargs)
            if event is not None:
                events.append(event)
        return events

    async def mark(self, target: RolloutState | None, stage: str, **kwargs: Any) -> TraceEvent | None:
        return self.record(target, stage, **kwargs)

    async def mark_many(self, targets: Iterable[RolloutState], stage: str, **kwargs: Any) -> list[TraceEvent]:
        return self.record_many(targets, stage, **kwargs)


class NoopTraceRecorder:
    def record(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_many(self, *args: Any, **kwargs: Any) -> list[TraceEvent]:
        return []

    async def mark(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def mark_many(self, *args: Any, **kwargs: Any) -> list[TraceEvent]:
        return []


class TraceTargetResolver:
    @classmethod
    def as_rollout_state_list(cls, value: Any) -> list[RolloutState]:
        if value is None:
            return []
        if isinstance(value, RolloutState):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            states: list[RolloutState] = []
            for item in value:
                if isinstance(item, RolloutState):
                    states.append(item)
            return states
        return []

    @classmethod
    def resolve(
        cls,
        bound_arguments: dict[str, Any],
        *,
        target: str | RolloutState | Sequence[RolloutState] | Callable[..., Any] | None,
        target_getter: Callable[..., Any] | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if target_getter is not None:
            return target_getter(*args, **kwargs)
        if isinstance(target, str):
            return bound_arguments.get(target)
        if callable(target):
            return target(*args, **kwargs)
        if target is not None:
            return target
        rollout_state = bound_arguments.get("rollout_state")
        rollout_states = cls.as_rollout_state_list(rollout_state)
        if rollout_states:
            return rollout_states if len(rollout_states) > 1 else rollout_states[0]
        for value in bound_arguments.values():
            states = cls.as_rollout_state_list(value)
            if states:
                return states if len(states) > 1 else states[0]
        return None

    @classmethod
    def record_event(cls, target: Any, name: str, **kwargs: Any) -> TraceEvent | list[TraceEvent] | None:
        targets = cls.as_rollout_state_list(target)
        recorder = current_trace_recorder()
        if targets:
            if len(targets) == 1:
                return recorder.record(targets[0], name, **kwargs)
            return recorder.record_many(targets, name, **kwargs)
        return recorder.record(None, name, **kwargs)

    @classmethod
    async def mark_event(cls, target: Any, name: str, **kwargs: Any) -> TraceEvent | list[TraceEvent] | None:
        targets = cls.as_rollout_state_list(target)
        recorder = current_trace_recorder()
        if targets:
            if len(targets) == 1:
                return await recorder.mark(targets[0], name, **kwargs)
            return await recorder.mark_many(targets, name, **kwargs)
        return await recorder.mark(None, name, **kwargs)


_NOOP_TRACE_RECORDER = NoopTraceRecorder()
_CURRENT_TRACE_RECORDER: contextvars.ContextVar[TraceRecorder | NoopTraceRecorder | None] = contextvars.ContextVar(
    "xtuner_current_trace_recorder",
    default=None,
)


# Global trace runtime. Each process owns one local runtime, propagated through Ray env vars.
def current_trace_recorder() -> TraceRecorder | NoopTraceRecorder:
    recorder = _CURRENT_TRACE_RECORDER.get()
    if recorder is not None:
        return recorder
    return get_tracer()


@contextlib.contextmanager
def use_trace_recorder(recorder: TraceRecorder | NoopTraceRecorder):
    token = _CURRENT_TRACE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _CURRENT_TRACE_RECORDER.reset(token)


def configure_trace(
    config: TraceConfig | None, *, output_dir: str | Path | None = None
) -> TraceRecorder | NoopTraceRecorder:
    return _TRACE_RUNTIME_MANAGER.configure(config, output_dir=output_dir)


def get_tracer() -> TraceRecorder | NoopTraceRecorder:
    return _TRACE_RUNTIME_MANAGER.get_tracer()


def flush_trace() -> bool:
    return _TRACE_RUNTIME_MANAGER.flush()


def close_trace() -> None:
    _TRACE_RUNTIME_MANAGER.close()


def reset_trace_for_test() -> None:
    close_trace()
    _CURRENT_TRACE_RECORDER.set(None)


def get_trace_env_vars() -> dict[str, str]:
    return _TRACE_RUNTIME_MANAGER.env_vars()


def merge_trace_runtime_env(actor_options: dict[str, Any]) -> dict[str, Any]:
    return _TRACE_RUNTIME_MANAGER.merge_runtime_env(actor_options)


def build_trace_id(task_name: str | None, uid: int | str | None) -> str | None:
    return TraceEventBuilder.trace_id(task_name, uid)


def build_produce_batch_id(
    train_step: int | None,
    model_step: int | None,
    producer_future_step: int | None,
) -> str | None:
    return TraceEventBuilder.produce_batch_id(train_step, model_step, producer_future_step)


def build_trace_event(
    target: RolloutState | None,
    stage: str,
    *,
    task_name: str | None = None,
    uid: int | str | None = None,
    session_uid: int | str | None = None,
    status: Any | None = None,
    train_step: int | None = None,
    model_step: int | None = None,
    producer_future_step: int | None = None,
    produce_batch_id: str | None = None,
    worker_rank: int | None = None,
    elapsed_s: float | None = None,
    error_msg: str | None = None,
    timestamp_s: float | None = None,
) -> TraceEvent | None:
    return TraceEventBuilder.build(
        target,
        stage,
        task_name=task_name,
        uid=uid,
        session_uid=session_uid,
        status=status,
        train_step=train_step,
        model_step=model_step,
        producer_future_step=producer_future_step,
        produce_batch_id=produce_batch_id,
        worker_rank=worker_rank,
        elapsed_s=elapsed_s,
        error_msg=error_msg,
        timestamp_s=timestamp_s,
    )


async def trace_event(target: Any, name: str, **kwargs: Any) -> TraceEvent | list[TraceEvent] | None:
    return await TraceTargetResolver.mark_event(target, name, **kwargs)


@contextlib.asynccontextmanager
async def trace_span(target: Any, name: str, **kwargs: Any):
    start_time = time.monotonic()
    await trace_event(target, f"{name}.start", **kwargs)
    try:
        yield
    except Exception as exc:
        await trace_event(
            target,
            f"{name}.error",
            elapsed_s=time.monotonic() - start_time,
            error_msg=TraceEventBuilder.short_error(exc),
            **kwargs,
        )
        raise
    else:
        await trace_event(target, f"{name}.end", elapsed_s=time.monotonic() - start_time, **kwargs)


class TraceFunctionDecorator:
    def __init__(
        self,
        name: str | Callable[..., str],
        *,
        target: str | RolloutState | Sequence[RolloutState] | Callable[..., Any] | None = None,
        target_getter: Callable[..., Any] | None = None,
        trace_kwargs_getter: Callable[..., dict[str, Any] | None] | None = None,
        trace_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.target = target
        self.target_getter = target_getter
        self.trace_kwargs_getter = trace_kwargs_getter
        self.trace_kwargs = trace_kwargs or {}

    def decorate(self, func: Callable[..., Any]):
        signature = inspect.signature(func)
        if inspect.iscoroutinefunction(func):
            return self._decorate_async(func, signature)
        return self._decorate_sync(func, signature)

    def _start_target(self, signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return TraceTargetResolver.resolve(
            bound.arguments,
            target=self.target,
            target_getter=self.target_getter,
            args=args,
            kwargs=kwargs,
        )

    def _end_target(self, start_target: Any, return_value: Any) -> Any:
        if TraceTargetResolver.as_rollout_state_list(return_value):
            return return_value
        return start_target

    def _name(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        if callable(self.name):
            return self.name(*args, **kwargs)
        return self.name

    def _kwargs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(self.trace_kwargs)
        if self.trace_kwargs_getter is None:
            return resolved
        dynamic_kwargs = self.trace_kwargs_getter(*args, **kwargs)
        if dynamic_kwargs:
            resolved.update(dynamic_kwargs)
        return resolved

    def _decorate_async(self, func: Callable[..., Any], signature: inspect.Signature):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            start_target = self._start_target(signature, args, kwargs)
            trace_name = self._name(args, kwargs)
            trace_kwargs = self._kwargs(args, kwargs)
            start_time = time.monotonic()
            await trace_event(start_target, f"{trace_name}.start", **trace_kwargs)
            try:
                return_value = await func(*args, **kwargs)
            except Exception as exc:
                await trace_event(
                    start_target,
                    f"{trace_name}.error",
                    elapsed_s=time.monotonic() - start_time,
                    error_msg=TraceEventBuilder.short_error(exc),
                    **trace_kwargs,
                )
                raise
            end_target = self._end_target(start_target, return_value)
            await trace_event(
                end_target,
                f"{trace_name}.end",
                elapsed_s=time.monotonic() - start_time,
                **trace_kwargs,
            )
            return return_value

        return async_wrapper

    def _decorate_sync(self, func: Callable[..., Any], signature: inspect.Signature):
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            start_target = self._start_target(signature, args, kwargs)
            trace_name = self._name(args, kwargs)
            trace_kwargs = self._kwargs(args, kwargs)
            start_time = time.monotonic()
            TraceTargetResolver.record_event(start_target, f"{trace_name}.start", **trace_kwargs)
            try:
                return_value = func(*args, **kwargs)
            except Exception as exc:
                TraceTargetResolver.record_event(
                    start_target,
                    f"{trace_name}.error",
                    elapsed_s=time.monotonic() - start_time,
                    error_msg=TraceEventBuilder.short_error(exc),
                    **trace_kwargs,
                )
                raise
            end_target = self._end_target(start_target, return_value)
            TraceTargetResolver.record_event(
                end_target,
                f"{trace_name}.end",
                elapsed_s=time.monotonic() - start_time,
                **trace_kwargs,
            )
            return return_value

        return sync_wrapper


def trace_function(
    name: str | Callable[..., str],
    *,
    target: str | RolloutState | Sequence[RolloutState] | Callable[..., Any] | None = None,
    target_getter: Callable[..., Any] | None = None,
    trace_kwargs_getter: Callable[..., dict[str, Any] | None] | None = None,
    **trace_kwargs: Any,
):
    """Trace a whole sync/async function as one task-level span.

    Target resolution for the `.start` event:
    - If `target_getter` is provided, use its return value.
    - Else if `target` is provided, resolve that explicit target.
    - Else prefer the argument named `rollout_state` when it is a `RolloutState`
      or `list[RolloutState]`.
    - Else fall back to the first `RolloutState` / `list[RolloutState]` found
      in the bound arguments.

    Target resolution for the `.end` event:
    - If the function returns a `RolloutState` or `list[RolloutState]`, use the
      return value so the end event reflects the latest task state.
    - Otherwise reuse the start target.

    In practice this means standard XTuner functions whose task parameter is
    named `rollout_state` usually do not need to pass `target=...`. Functions
    with non-standard parameter names such as `group` should still pass an
    explicit `target`.
    """
    return TraceFunctionDecorator(
        name,
        target=target,
        target_getter=target_getter,
        trace_kwargs_getter=trace_kwargs_getter,
        trace_kwargs=trace_kwargs,
    ).decorate


# Runtime wrapper that owns store lifecycle.
@dataclass
class TraceRuntime:
    config: TraceConfig
    recorder: TraceRecorder | NoopTraceRecorder
    store: InMemoryTraceStore | None = None

    def flush(self) -> bool:
        if self.store is not None:
            return self.store.flush_jsonl()
        return True

    def close(self) -> bool:
        if self.store is not None:
            return self.store.close()
        return True


def build_trace_runtime(config: TraceConfig | None, *, output_dir: str | Path | None = None) -> TraceRuntime:
    if config is None:
        config = TraceConfig()
    if output_dir is not None and config.output_dir is None:
        config = config.model_copy(update={"output_dir": output_dir})
    if not config.enabled:
        return TraceRuntime(config=config, recorder=_NOOP_TRACE_RECORDER, store=None)
    store = InMemoryTraceStore(config)
    return TraceRuntime(config=config, recorder=TraceRecorder(store), store=store)


class TraceRuntimeManager:
    def __init__(self) -> None:
        self._runtime: TraceRuntime | None = None
        self._identity: tuple[Any, ...] | None = None
        self._lock = threading.RLock()
        self._atexit_registered = False

    def configure(
        self, config: TraceConfig | None, *, output_dir: str | Path | None = None
    ) -> TraceRecorder | NoopTraceRecorder:
        config = self._normalize_config(config or TraceConfig(), output_dir=output_dir)
        self._export_env(config)
        return self._replace_runtime(config).recorder

    def get_tracer(self) -> TraceRecorder | NoopTraceRecorder:
        config = self._load_config_from_env()
        identity = self._config_identity(config)
        with self._lock:
            if self._runtime is not None and self._identity == identity:
                return self._runtime.recorder
        return self._replace_runtime(config).recorder

    def flush(self) -> bool:
        with self._lock:
            runtime = self._runtime
        if runtime is not None:
            return runtime.flush()
        return True

    def close(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._identity = None
            self._clear_env()
        if runtime is not None:
            runtime.close()

    def env_vars(self) -> dict[str, str]:
        if os.environ.get(TRACE_ENV_ENABLED) != "1":
            return {}
        env_vars: dict[str, str] = {}
        for name in (
            TRACE_ENV_ENABLED,
            TRACE_ENV_OUTPUT_DIR,
            TRACE_ENV_MAX_EVENTS,
            TRACE_ENV_MAX_EVENTS_PER_TRACE,
        ):
            value = os.environ.get(name)
            if value is not None:
                env_vars[name] = value
        return env_vars

    def merge_runtime_env(self, actor_options: dict[str, Any]) -> dict[str, Any]:
        trace_env_vars = self.env_vars()
        if not trace_env_vars:
            return actor_options
        runtime_env = dict(actor_options.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.update(trace_env_vars)
        runtime_env["env_vars"] = env_vars
        actor_options["runtime_env"] = runtime_env
        return actor_options

    def _replace_runtime(self, config: TraceConfig) -> TraceRuntime:
        config = self._normalize_config(config)
        identity = self._config_identity(config)
        with self._lock:
            if self._runtime is not None and self._identity == identity:
                return self._runtime
            old_runtime = self._runtime
            self._runtime = build_trace_runtime(config)
            self._identity = identity
            self._register_atexit()
            runtime = self._runtime
        if old_runtime is not None:
            old_runtime.close()
        return runtime

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(close_trace)
        self._atexit_registered = True

    @staticmethod
    def _normalize_config(config: TraceConfig, *, output_dir: str | Path | None = None) -> TraceConfig:
        updates: dict[str, Any] = {}
        resolved_output_dir = output_dir if output_dir is not None else config.output_dir
        if resolved_output_dir is not None:
            updates["output_dir"] = str(Path(resolved_output_dir).absolute())
        if updates:
            config = config.model_copy(update=updates)
        return config

    @staticmethod
    def _export_env(config: TraceConfig) -> None:
        if not config.enabled:
            TraceRuntimeManager._clear_env()
            os.environ[TRACE_ENV_ENABLED] = "0"
            return

        os.environ[TRACE_ENV_ENABLED] = "1"
        if config.output_dir is not None:
            os.environ[TRACE_ENV_OUTPUT_DIR] = str(Path(config.output_dir).absolute())
        else:
            os.environ.pop(TRACE_ENV_OUTPUT_DIR, None)
        os.environ[TRACE_ENV_MAX_EVENTS] = str(config.max_events)
        os.environ[TRACE_ENV_MAX_EVENTS_PER_TRACE] = str(config.max_events_per_trace)

    @staticmethod
    def _clear_env() -> None:
        for name in (
            TRACE_ENV_ENABLED,
            TRACE_ENV_OUTPUT_DIR,
            TRACE_ENV_MAX_EVENTS,
            TRACE_ENV_MAX_EVENTS_PER_TRACE,
        ):
            os.environ.pop(name, None)

    @staticmethod
    def _load_config_from_env() -> TraceConfig:
        if os.environ.get(TRACE_ENV_ENABLED) != "1":
            return TraceConfig()

        output_dir = os.environ.get(TRACE_ENV_OUTPUT_DIR)
        max_events = int(os.environ.get(TRACE_ENV_MAX_EVENTS, TraceConfig.model_fields["max_events"].default))
        max_events_per_trace = int(
            os.environ.get(TRACE_ENV_MAX_EVENTS_PER_TRACE, TraceConfig.model_fields["max_events_per_trace"].default)
        )
        return TraceConfig(
            enabled=True,
            output_dir=output_dir,
            max_events=max_events,
            max_events_per_trace=max_events_per_trace,
        )

    @staticmethod
    def _config_identity(config: TraceConfig) -> tuple[Any, ...]:
        output_dir = str(Path(config.output_dir).absolute()) if config.output_dir is not None else None
        return (
            config.enabled,
            output_dir,
            config.max_events,
            config.max_events_per_trace,
        )


_TRACE_RUNTIME_MANAGER = TraceRuntimeManager()
