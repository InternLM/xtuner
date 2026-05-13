"""Unit tests for xtuner.v1.ray.environment.trace."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from xtuner.v1.ray.environment import trace as trace_mod


@pytest.fixture
def trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("WORK_DIR", str(work))
    trace_mod._reset_for_testing()
    trace_mod.init_writer(actor_id="test")
    yield work / "trace"
    trace_mod._reset_for_testing()


@pytest.fixture
def no_work_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORK_DIR", raising=False)
    trace_mod._reset_for_testing()
    yield
    trace_mod._reset_for_testing()


def _read_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"expected {path} to exist"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fates(trace_dir: Path) -> list[dict]:
    files = list(trace_dir.glob("fates.*.jsonl"))
    assert len(files) == 1, f"expected 1 fates file, got {files}"
    return _read_jsonl(files[0])


def _spans(trace_dir: Path) -> list[dict]:
    files = list(trace_dir.glob("spans.*.jsonl"))
    assert len(files) == 1, f"expected 1 spans file, got {files}"
    return _read_jsonl(files[0])


class TestNoop:
    def test_emit_fate_without_work_dir_is_noop(self, no_work_dir: None) -> None:
        trace_mod.init_writer(actor_id="x")
        trace_mod.emit_fate(uid="u", task_id="t", group_id="g", final="COMPLETED")

    def test_span_without_work_dir_is_noop(self, no_work_dir: None) -> None:
        trace_mod.init_writer(actor_id="x")
        with trace_mod.span(uid="u", stage="s"):
            pass

    def test_span_without_work_dir_still_propagates_exception(self, no_work_dir: None) -> None:
        trace_mod.init_writer(actor_id="x")
        with pytest.raises(ValueError):
            with trace_mod.span(uid="u", stage="s"):
                raise ValueError("boom")


class TestFate:
    def test_emit_fate_writes_line(self, trace_dir: Path) -> None:
        trace_mod.emit_fate(
            uid="uid1",
            task_id="task_a",
            group_id="grp_a",
            final="SKIPPED",
            failed_stage="infer",
            reason="heartbeat stale",
            extra_field=123,
        )
        rows = _fates(trace_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row["uid"] == "uid1"
        assert row["task_id"] == "task_a"
        assert row["group_id"] == "grp_a"
        assert row["final"] == "SKIPPED"
        assert row["failed_stage"] == "infer"
        assert row["reason"] == "heartbeat stale"
        assert row["extra_field"] == 123
        assert isinstance(row["ts"], float)

    def test_emit_fate_completed(self, trace_dir: Path) -> None:
        trace_mod.emit_fate(uid="u", task_id="t", group_id="g", final="COMPLETED")
        rows = _fates(trace_dir)
        assert rows[0]["final"] == "COMPLETED"
        assert rows[0]["failed_stage"] is None
        assert rows[0]["reason"] is None


class TestSpan:
    def test_span_normal_exit(self, trace_dir: Path) -> None:
        with trace_mod.span(uid="u", stage="acquire", task_id="t"):
            pass
        rows = _spans(trace_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row["uid"] == "u"
        assert row["stage"] == "acquire"
        assert row["task_id"] == "t"
        assert row["ok"] is True
        assert row["err"] is None
        assert row["duration_ms"] >= 0

    def test_span_propagates_and_marks_error(self, trace_dir: Path) -> None:
        with pytest.raises(RuntimeError):
            with trace_mod.span(uid="u", stage="infer"):
                raise RuntimeError("daemon stuck")
        rows = _spans(trace_dir)
        assert len(rows) == 1
        row = rows[0]
        assert row["ok"] is False
        assert "RuntimeError" in row["err"]
        assert "daemon stuck" in row["err"]

    def test_span_usable_around_await(self, trace_dir: Path) -> None:
        async def job() -> None:
            with trace_mod.span(uid="u", stage="infer"):
                await asyncio.sleep(0.01)

        asyncio.run(job())
        rows = _spans(trace_dir)
        assert rows[0]["ok"] is True
        assert rows[0]["duration_ms"] >= 10

    def test_span_handle_mark_error(self, trace_dir: Path) -> None:
        with trace_mod.span(uid="u", stage="infer") as handle:
            handle.mark_error("rc=5 non-zero exit")
        rows = _spans(trace_dir)
        assert rows[0]["ok"] is False
        assert rows[0]["err"] == "rc=5 non-zero exit"


class TestConcurrency:
    def test_concurrent_async_writes_produce_well_formed_lines(self, trace_dir: Path) -> None:
        async def worker(i: int) -> None:
            with trace_mod.span(uid=f"u{i}", stage="infer"):
                await asyncio.sleep(0)
            trace_mod.emit_fate(
                uid=f"u{i}", task_id=f"t{i}", group_id="g", final="COMPLETED"
            )

        async def main() -> None:
            await asyncio.gather(*(worker(i) for i in range(100)))

        asyncio.run(main())

        span_rows = _spans(trace_dir)
        fate_rows = _fates(trace_dir)
        assert len(span_rows) == 100
        assert len(fate_rows) == 100
        assert {row["uid"] for row in span_rows} == {f"u{i}" for i in range(100)}
        assert all(row["ok"] for row in span_rows)


class TestIdempotent:
    def test_init_writer_twice_keeps_first_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.setenv("WORK_DIR", str(work))
        trace_mod._reset_for_testing()
        trace_mod.init_writer(actor_id="a")
        first = trace_mod._writer
        trace_mod.init_writer(actor_id="b")
        assert trace_mod._writer is first
        trace_mod._reset_for_testing()

    def test_reset_closes_and_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.setenv("WORK_DIR", str(work))
        trace_mod._reset_for_testing()
        trace_mod.init_writer(actor_id="a")
        trace_mod.emit_fate(uid="u", task_id="t", group_id="g", final="COMPLETED")
        trace_mod._reset_for_testing()
        # After reset, further calls are no-ops.
        trace_mod.emit_fate(uid="u2", task_id="t", group_id="g", final="COMPLETED")
        files = list((work / "trace").glob("fates.*.jsonl"))
        # Only the first emit survived.
        rows = _read_jsonl(files[0])
        assert len(rows) == 1
        assert rows[0]["uid"] == "u"
