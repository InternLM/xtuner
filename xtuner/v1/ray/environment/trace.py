"""Per-sample trace emission for the InstallAgentEnvironment pipeline.

Three jsonl channels written under ``$WORK_DIR/trace/``:

* ``fates.{actor_id}.{pid}.jsonl`` — one terminal line per sample, capturing
  whether it ended up COMPLETED or SKIPPED plus the stage/reason.
* ``spans.{actor_id}.{pid}.jsonl`` — one line per stage enter/exit, with
  duration and ok/err. Sampled inside the async pipeline via a context
  manager whose yield region may ``await``.
* ``llm_calls.{pid}.jsonl`` — one line per ``/v1/chat/completions`` request
  served by :class:`RolloutController`, with total / tokenize / rollout /
  post durations and token counts.  Independent of the per-sample fate/span
  writer because the controller is its own actor with no owning sample uid.

Each Ray actor writes its own two files so concurrent writes never contend
a single file descriptor across processes. Line-buffered + per-line flush
keeps ``tail -f`` and crash-recovery both simple.

If the ``WORK_DIR`` environment variable is unset the module degrades to a
zero-cost no-op — tests and local one-shot scripts never have to care.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from xtuner.v1.utils import get_logger


_writer: _TraceWriter | None = None


def init_writer(actor_id: str | None = None) -> None:
    """Open the per-actor fates/spans files. Safe to call twice (subsequent
    calls are no-ops).

    Args:
        actor_id (str | None): Stable identifier for this actor process,
            folded into the output filename to keep per-actor files apart.
            When ``None`` the pid alone is used.
    """
    global _writer
    if _writer is not None:
        return
    work_dir = os.environ.get("WORK_DIR")
    if not work_dir:
        get_logger().info("[trace] WORK_DIR not set; trace emission disabled")
        return
    try:
        trace_dir = Path(work_dir) / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        _writer = _TraceWriter(trace_dir, actor_id)
        get_logger().info(f"[trace] writing to {trace_dir} (actor={actor_id or 'none'}, pid={os.getpid()})")
    except Exception as exc:
        get_logger().warning(f"[trace] init failed ({exc}); trace emission disabled")
        _writer = None


def emit_fate(
    uid: str,
    task_id: str | None,
    group_id: str | None,
    final: str,
    failed_stage: str | None = None,
    reason: str | None = None,
    **extra: Any,
) -> None:
    """Record a sample's terminal outcome.

    Args:
        uid (str): Per-sample observation id.
        task_id (str | None): Dataset task id; best-effort identifier.
        group_id (str | None): Shared id across prompt_repeat_k siblings.
        final (str): ``"COMPLETED"`` or ``"SKIPPED"``.
        failed_stage (str | None): Stage name where failure originated.
        reason (str | None): Human-readable error string.
        **extra (Any): Additional keys merged into the record.
    """
    if _writer is None:
        return
    record: dict[str, Any] = {
        "ts": time.time(),
        "uid": uid,
        "task_id": task_id,
        "group_id": group_id,
        "final": final,
        "failed_stage": failed_stage,
        "reason": reason,
    }
    if extra:
        record.update(extra)
    _writer.write_fate(record)


class SpanHandle:
    """Mutable handle yielded by :func:`span` so the caller can flag logical
    failures that do not raise (e.g. a subprocess returning non-zero without
    throwing).

    Fields default to "success"; see ``mark_error``.
    """

    def __init__(self) -> None:
        self.ok: bool = True
        self.err: str | None = None

    def mark_error(self, err: str) -> None:
        self.ok = False
        self.err = err


@contextmanager
def span(uid: str, stage: str, **extra: Any) -> Iterator[SpanHandle]:
    """Time a stage and emit a span record on exit.

    The context manager is synchronous but safe to wrap ``await``-bearing
    code because the yield region runs on the caller's event loop.

    Args:
        uid (str): Per-sample observation id.
        stage (str): Short stage name (e.g. ``"acquire"``, ``"infer"``).
        **extra (Any): Additional keys merged into the record.

    Returns:
        SpanHandle: Yielded handle; call ``handle.mark_error(...)`` inside
        the block to flag a non-raising logical failure.
    """
    handle = SpanHandle()
    t_start = time.monotonic()
    try:
        yield handle
    except BaseException as exc:
        handle.ok = False
        handle.err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        if _writer is not None:
            record: dict[str, Any] = {
                "ts": time.time(),
                "uid": uid,
                "stage": stage,
                "duration_ms": duration_ms,
                "ok": handle.ok,
                "err": handle.err,
            }
            if extra:
                record.update(extra)
            _writer.write_span(record)


def _reset_for_testing() -> None:
    """Close the writer and clear module state.

    For unit tests only.
    """
    global _writer, _llm_writer, _llm_writer_ready
    if _writer is not None:
        _writer.close()
    _writer = None
    if _llm_writer is not None:
        try:
            _llm_writer.close()
        except Exception:
            pass
    _llm_writer = None
    _llm_writer_ready = False


# ─────────────────────────────────────────────────────────────────
# LLM call stream (separate writer, opened lazily per process)
# ─────────────────────────────────────────────────────────────────
#
# ``RolloutController`` emits one record per ``/v1/chat/completions`` via
# :func:`emit_llm_call`.  The writer is lazy so install_agent_env actors
# (which never call emit_llm_call) don't create empty files.

_llm_writer: TextIO | None = None
_llm_writer_ready: bool = False


def _ensure_llm_writer() -> None:
    global _llm_writer, _llm_writer_ready
    if _llm_writer_ready:
        return
    _llm_writer_ready = True
    work_dir = os.environ.get("WORK_DIR")
    if not work_dir:
        return
    try:
        trace_dir = Path(work_dir) / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"llm_calls.{os.getpid()}.jsonl"
        _llm_writer = open(path, "a", buffering=1, encoding="utf-8")
        get_logger().info(f"[trace] LLM call stream → {path}")
    except Exception as exc:
        get_logger().warning(f"[trace] LLM call stream init failed ({exc}); disabled")
        _llm_writer = None


def emit_llm_call(
    total_ms: int,
    tokenize_ms: int,
    rollout_ms: int,
    post_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    **extra: Any,
) -> None:
    """Record timing and token counts for one LLM request.

    Called from ``RolloutController`` on every ``/v1/chat/completions`` —
    unlike the slow-request warning log this captures *all* requests so
    ``view.py --llm-stats`` can compute real p50 / p95 / p99.

    Args:
        total_ms (int): End-to-end request duration.
        tokenize_ms (int): Tokenize phase duration.
        rollout_ms (int): Rollout worker call duration.
        post_ms (int): Post-processing (detokenize + response packing).
        prompt_tokens (int): Input token count.
        completion_tokens (int): Output token count.
        **extra (Any): Additional fields merged into the record.
    """
    _ensure_llm_writer()
    if _llm_writer is None:
        return
    record: dict[str, Any] = {
        "ts": time.time(),
        "total_ms": total_ms,
        "tokenize_ms": tokenize_ms,
        "rollout_ms": rollout_ms,
        "post_ms": post_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if extra:
        record.update(extra)
    try:
        _llm_writer.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        get_logger().warning(f"[trace] LLM call write failed: {exc}")


class _TraceWriter:
    def __init__(self, trace_dir: Path, actor_id: str | None) -> None:
        suffix = f"{actor_id or 'noactor'}.{os.getpid()}"
        self._fates_path = trace_dir / f"fates.{suffix}.jsonl"
        self._spans_path = trace_dir / f"spans.{suffix}.jsonl"
        self._fates: TextIO = open(self._fates_path, "a", buffering=1, encoding="utf-8")
        self._spans: TextIO = open(self._spans_path, "a", buffering=1, encoding="utf-8")

    def write_fate(self, record: dict[str, Any]) -> None:
        self._write(self._fates, record)

    def write_span(self, record: dict[str, Any]) -> None:
        self._write(self._spans, record)

    def close(self) -> None:
        for fp in (self._fates, self._spans):
            try:
                fp.close()
            except Exception:
                pass

    @staticmethod
    def _write(fp: TextIO, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            line = json.dumps(
                {"_encode_error": traceback.format_exc(limit=1), "_repr": repr(record)},
                ensure_ascii=False,
            )
        try:
            fp.write(line + "\n")
        except Exception as exc:
            get_logger().warning(f"[trace] write failed: {exc}")
