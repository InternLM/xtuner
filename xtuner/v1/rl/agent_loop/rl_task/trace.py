"""Per-sample trace emission for the InstallAgentEnvironment pipeline.

Four outputs under ``$WORK_DIR/trace/``:

* ``fates.{actor_id}.{pid}.jsonl`` — one terminal line per sample, capturing
  whether it ended up COMPLETED or SKIPPED plus the stage/reason.
* ``spans.{actor_id}.{pid}.jsonl`` — one line per stage enter/exit, with
  duration and ok/err. Sampled inside the async pipeline via a context
  manager whose yield region may ``await``.
* ``llm_calls.{pid}.jsonl`` — one line per ``/v1/chat/completions`` request
  served by :class:`RolloutController`, with total / tokenize / rollout /
  post durations and token counts.  Independent of the per-sample fate/span
  writer because the controller is its own actor with no owning sample uid.
* ``diagnostics/{ts}_{task_id}_{uid}.log`` (+ optional ``.daemon.log``) —
  unstructured per-failure bundle with the pulled daemon log tail.  Written
  whenever a sample fails with a non-zero entry rc or runner-level
  exception; the header file is always produced so a missing daemon log
  still leaves a breadcrumb (``daemon_log_file=(unavailable)``).

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
    throwing) or attach runtime-discovered fields (e.g. a sandbox URL that only
    becomes known after ``acquire`` succeeds).

    Fields default to "success"; see ``mark_error`` and ``annotate``.
    """

    def __init__(self) -> None:
        self.ok: bool = True
        self.err: str | None = None
        # Runtime-discovered fields merged into the span record at emit time.
        # Separate from the constructor ``extra`` kwargs because those are
        # fixed at span-entry, whereas these are set mid-block.
        self.annotations: dict[str, Any] = {}

    def mark_error(self, err: str) -> None:
        self.ok = False
        self.err = err

    def annotate(self, **fields: Any) -> None:
        """Attach runtime-discovered fields to this span record (merged at emit
        time).

        Useful for values that are only known after the guarded
        code ran — e.g. the sandbox URL returned by ``SandboxPool.get``.
        """
        self.annotations.update(fields)


@contextmanager
def span(uid: str, stage: str, **extra: Any) -> Iterator[SpanHandle]:
    """Time a stage and emit an enter + exit span pair.

    The context manager is synchronous but safe to wrap ``await``-bearing
    code because the yield region runs on the caller's event loop.

    Two records are written per span so consumers tailing the file in real
    time can see "stage entered" before the stage finishes:

      * ``{"event": "enter", "ts", "uid", "stage", **extra}``
      * ``{"event": "exit",  "ts", "uid", "stage", "duration_ms", "ok", "err",
          **extra, **annotations}``

    Args:
        uid (str): Per-sample observation id.
        stage (str): Short stage name (e.g. ``"acquire"``, ``"infer"``).
        **extra (Any): Additional keys merged into both records.

    Returns:
        SpanHandle: Yielded handle; call ``handle.mark_error(...)`` or
        ``handle.annotate(...)`` inside the block to customize the exit record.
    """
    handle = SpanHandle()
    t_start = time.monotonic()
    if _writer is not None:
        enter: dict[str, Any] = {
            "ts": time.time(),
            "event": "enter",
            "uid": uid,
            "stage": stage,
        }
        if extra:
            enter.update(extra)
        _writer.write_span(enter)
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
                "event": "exit",
                "uid": uid,
                "stage": stage,
                "duration_ms": duration_ms,
                "ok": handle.ok,
                "err": handle.err,
            }
            if extra:
                record.update(extra)
            if handle.annotations:
                record.update(handle.annotations)
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


# ─────────────────────────────────────────────────────────────────
# Failure-path diagnostics (unstructured dump)
# ─────────────────────────────────────────────────────────────────
#
# ``emit_diagnostic`` writes a per-failure bundle under
# ``$WORK_DIR/trace/diagnostics/`` whenever a sample dies with a non-zero
# entry rc or runner-level exception.  The header ``.log`` is always
# written so a missing ``.daemon.log`` (sandbox unreachable, TTL expired,
# etc.) still leaves a breadcrumb with the ``download_err`` explanation.
#
# Caller (``runner._dump_skipped_diagnostic``) owns the sandbox client and
# is responsible for downloading the daemon log bytes; this function just
# writes files.  Keeping HTTP out of trace.py means the module has no
# runtime deps on lagent/sandbox-client internals.

_DIAGNOSTIC_TAIL_PREVIEW_LINES = 50


def emit_diagnostic(
    task_id: str | None,
    uid: str | None,
    data_source: str | None,
    exception_type: str,
    exception_msg: str,
    daemon_log: bytes | None = None,
    download_err: str | None = None,
) -> None:
    """Persist a per-failure diagnostic bundle under
    ``$WORK_DIR/trace/diagnostics/``.

    Args:
        task_id (str | None): Dataset task id; used in filename + header.
        uid (str | None): Per-sample observation id; truncated to 12 chars
            in the filename so multiple sibling failures don't collide.
        data_source (str | None): Data source label (e.g. ``"tb2-rl"``).
        exception_type (str): Class name of the exception that triggered
            the dump.
        exception_msg (str): Short error message.
        daemon_log (bytes | None): Full ``/tmp/agent_daemon.log`` bytes if
            download succeeded; ``None`` when the caller's download failed
            (sandbox unreachable, etc.).
        download_err (str | None): Short description of the download
            failure when ``daemon_log is None``.
    """
    work_dir = os.environ.get("WORK_DIR")
    if not work_dir:
        return
    diag_dir = Path(work_dir) / "trace" / "diagnostics"
    try:
        diag_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        get_logger().debug(f"[trace] diagnostic dir mkdir failed: {exc}")
        return

    ts = time.strftime("%H%M%S")
    uid_short = (uid or "nouid")[:12]
    base = diag_dir / f"{ts}_{task_id or 'notask'}_{uid_short}"

    full_size = 0
    tail_preview = "(no daemon log)"
    if daemon_log is not None:
        full_size = len(daemon_log)
        text = daemon_log.decode(errors="replace")
        tail_preview = "\n".join(text.splitlines()[-_DIAGNOSTIC_TAIL_PREVIEW_LINES:])
        try:
            base.with_suffix(".daemon.log").write_bytes(daemon_log)
        except Exception as exc:
            get_logger().debug(f"[trace] daemon log write failed: {exc}")
    elif download_err is not None:
        tail_preview = f"(could not pull daemon log: {download_err})"

    try:
        base.with_suffix(".log").write_text(
            f"task_id={task_id}\n"
            f"uid={uid}\n"
            f"data_source={data_source}\n"
            f"timestamp={time.time()}\n"
            f"exception_type={exception_type}\n"
            f"exception={exception_msg}\n"
            f"daemon_log_bytes={full_size}\n"
            f"daemon_log_file={base.with_suffix('.daemon.log').name if daemon_log else '(unavailable)'}\n"
            f"---daemon_log_tail_preview (last {_DIAGNOSTIC_TAIL_PREVIEW_LINES} lines)---\n"
            f"{tail_preview}\n"
        )
    except Exception as exc:
        get_logger().debug(f"[trace] diagnostic header write failed at {base}: {exc}")
