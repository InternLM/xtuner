"""Unit tests for the daemon-silence watchdog in ``runner.py``.

No Ray, no real sandbox — we stub ``client.download_file`` and
``client.execute`` to simulate the daemon log growing / going silent /
sandbox becoming unreachable / daemon ping responses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, List

import pytest

from xtuner.v1.ray.environment.rl_task.runner import (
    DaemonStuckError,
    _daemon_silence_watchdog,
    _ping_daemon,
    _run_infer_with_watchdog,
)


class FakeClient:
    """Minimal stub exposing ``download_file`` and ``execute``.

    ``log_sequence`` feeds ``download_file``.  ``ping_sequence`` feeds
    ``execute`` (only the daemon-ping commands; anything else errors).
    Each element is either a bytes/dict return value or an Exception
    to raise.  Once a list is exhausted, the last value is repeated.
    """

    def __init__(
        self,
        log_sequence: List | None = None,
        ping_sequence: List | None = None,
    ):
        self._logs = log_sequence or []
        self._pings = ping_sequence or []
        self.downloads = 0
        self.executes = 0

    async def download_file(self, remote_path: str) -> bytes:
        idx = min(self.downloads, len(self._logs) - 1) if self._logs else -1
        self.downloads += 1
        if idx < 0:
            return b""
        val = self._logs[idx]
        if isinstance(val, Exception):
            raise val
        return val

    async def execute(self, command: str, cwd: str = "/root", timeout_sec: int = 60) -> dict:
        # Marker proves we're being invoked by the ping path, not some
        # stray /exec from tests.
        assert "PING_EOF" in command, f"unexpected execute: {command}"
        idx = min(self.executes, len(self._pings) - 1) if self._pings else -1
        self.executes += 1
        if idx < 0:
            return {"return_code": 0, "stdout": json.dumps({"status": "ok", "type": "agent"}), "stderr": ""}
        val = self._pings[idx]
        if isinstance(val, Exception):
            raise val
        return val


def ok_ping_reply() -> dict:
    return {"return_code": 0, "stdout": json.dumps({"status": "ok", "type": "agent"}), "stderr": ""}


def bad_rc_ping_reply() -> dict:
    return {"return_code": 1, "stdout": "", "stderr": "connect: no such file or directory"}


class FakeInferStage:
    """Stub SandboxStage with a configurable run duration/result."""

    def __init__(self, run_coro_factory: Callable):
        self._factory = run_coro_factory

    async def run(self, client, ctx):
        return await self._factory()


# ─────────────────────────────────────────────────────────────
# Tests that still exercise the log-size signal
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_raises_on_unchanged_log():
    """ping OK + log flat -> DaemonStuckError('agent deadlocked')."""
    # Both signals configured: ping always OK, log never grows.
    client = FakeClient(
        log_sequence=[b"line1\n" for _ in range(200)],
        ping_sequence=[ok_ping_reply() for _ in range(200)],
    )
    with pytest.raises(DaemonStuckError, match="agent deadlocked"):
        await _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=1, poll_interval_sec=0.05
        )
    assert client.downloads >= 2
    assert client.executes >= 2


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_when_log_grows():
    """ping OK + log grows -> watchdog keeps waiting."""
    seq = [b"l" * (i + 1) for i in range(100)]
    client = FakeClient(
        log_sequence=seq,
        ping_sequence=[ok_ping_reply() for _ in range(100)],
    )

    task = asyncio.create_task(
        _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=5, poll_interval_sec=0.02
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.downloads >= 5
    assert client.executes >= 5


@pytest.mark.asyncio
async def test_watchdog_download_failures_raise_log_unavailable():
    """ping OK + download always fails -> 'daemon log unavailable'."""
    import xtuner.v1.ray.environment.rl_task.runner as r

    class Boom(Exception):
        pass

    client = FakeClient(
        log_sequence=[Boom("boom") for _ in range(20)],
        ping_sequence=[ok_ping_reply() for _ in range(20)],
    )
    with pytest.raises(DaemonStuckError, match="log unavailable"):
        await _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=999, poll_interval_sec=0.01
        )
    assert client.downloads >= r._WATCHDOG_MAX_CONSECUTIVE_POLL_FAILURES


# ─────────────────────────────────────────────────────────────
# Tests for the new ping signal
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_ping_dead_raises():
    """ping fails repeatedly -> DaemonStuckError('daemon dead/unreachable')."""
    import xtuner.v1.ray.environment.rl_task.runner as r

    client = FakeClient(
        log_sequence=[b"whatever" for _ in range(20)],
        ping_sequence=[bad_rc_ping_reply() for _ in range(20)],
    )
    with pytest.raises(DaemonStuckError, match="daemon dead/unreachable"):
        await _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=999, poll_interval_sec=0.01
        )
    assert client.executes >= r._WATCHDOG_MAX_CONSECUTIVE_PING_FAILURES


@pytest.mark.asyncio
async def test_watchdog_ping_exception_raises():
    """ping raises (/exec itself fails) -> counts as failure, raises."""
    class NetBoom(Exception):
        pass

    client = FakeClient(
        log_sequence=[b"whatever" for _ in range(20)],
        ping_sequence=[NetBoom("network down") for _ in range(20)],
    )
    with pytest.raises(DaemonStuckError, match="daemon dead/unreachable"):
        await _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=999, poll_interval_sec=0.01
        )


@pytest.mark.asyncio
async def test_watchdog_ping_recovers_resets_counter():
    """one ping fail then ok -> counter resets, watchdog keeps running."""
    client = FakeClient(
        log_sequence=[b"line" * (i + 1) for i in range(100)],  # log grows so it won't trip
        ping_sequence=[
            bad_rc_ping_reply(),
            bad_rc_ping_reply(),
            ok_ping_reply(),       # recovery
            ok_ping_reply(),
        ] + [ok_ping_reply() for _ in range(100)],
    )
    task = asyncio.create_task(
        _daemon_silence_watchdog(
            client, "task-x", stale_threshold_sec=999, poll_interval_sec=0.01
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # If the counter hadn't reset after the 3rd (OK) ping, we'd have
    # raised DaemonStuckError on the 3rd consecutive bad ping. It didn't.
    assert client.executes >= 4


# ─────────────────────────────────────────────────────────────
# _ping_daemon smoke tests
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ping_daemon_ok():
    client = FakeClient(ping_sequence=[ok_ping_reply()])
    alive, detail = await _ping_daemon(client, timeout=5)
    assert alive is True
    assert '"status": "ok"' in detail


@pytest.mark.asyncio
async def test_ping_daemon_bad_rc():
    client = FakeClient(ping_sequence=[bad_rc_ping_reply()])
    alive, detail = await _ping_daemon(client, timeout=5)
    assert alive is False
    assert "rc=1" in detail


@pytest.mark.asyncio
async def test_ping_daemon_non_json():
    client = FakeClient(ping_sequence=[{"return_code": 0, "stdout": "garbage", "stderr": ""}])
    alive, detail = await _ping_daemon(client, timeout=5)
    assert alive is False
    assert "non-JSON" in detail


@pytest.mark.asyncio
async def test_ping_daemon_exec_exception():
    class Boom(Exception):
        pass

    client = FakeClient(ping_sequence=[Boom("boom")])
    alive, detail = await _ping_daemon(client, timeout=5)
    assert alive is False
    assert "Boom" in detail


# ─────────────────────────────────────────────────────────────
# _run_infer_with_watchdog glue tests (same as before)
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_infer_finishes_before_watchdog():
    async def quick_infer():
        await asyncio.sleep(0.05)
        return "infer_ok"

    infer = FakeInferStage(quick_infer)
    client = FakeClient(
        log_sequence=[b"flat" for _ in range(100)],
        ping_sequence=[ok_ping_reply() for _ in range(100)],
    )

    import xtuner.v1.ray.environment.rl_task.runner as r
    orig_stale = r._WATCHDOG_STALE_SEC
    orig_poll = r._WATCHDOG_POLL_SEC
    r._WATCHDOG_STALE_SEC = 999
    r._WATCHDOG_POLL_SEC = 0.01
    try:
        result = await _run_infer_with_watchdog(infer, client, {}, "tid-ok")
    finally:
        r._WATCHDOG_STALE_SEC = orig_stale
        r._WATCHDOG_POLL_SEC = orig_poll
    assert result == "infer_ok"


@pytest.mark.asyncio
async def test_run_infer_cancelled_when_watchdog_fires():
    infer_cancel_seen = asyncio.Event()

    async def slow_infer():
        try:
            await asyncio.sleep(10)
            return "should_not_get_here"
        except asyncio.CancelledError:
            infer_cancel_seen.set()
            raise

    infer = FakeInferStage(slow_infer)
    # ping OK, log flat -> triggers "agent deadlocked" branch
    client = FakeClient(
        log_sequence=[b"flat" for _ in range(100)],
        ping_sequence=[ok_ping_reply() for _ in range(100)],
    )

    import xtuner.v1.ray.environment.rl_task.runner as r
    orig_stale = r._WATCHDOG_STALE_SEC
    orig_poll = r._WATCHDOG_POLL_SEC
    r._WATCHDOG_STALE_SEC = 0
    r._WATCHDOG_POLL_SEC = 0.02
    try:
        with pytest.raises(DaemonStuckError, match="agent deadlocked"):
            await _run_infer_with_watchdog(infer, client, {}, "tid-stuck")
    finally:
        r._WATCHDOG_STALE_SEC = orig_stale
        r._WATCHDOG_POLL_SEC = orig_poll
    assert infer_cancel_seen.is_set()


@pytest.mark.asyncio
async def test_run_infer_propagates_infer_exception():
    class InferBoom(RuntimeError):
        pass

    async def failing_infer():
        await asyncio.sleep(0.05)
        raise InferBoom("infer blew up")

    infer = FakeInferStage(failing_infer)
    client = FakeClient(
        log_sequence=[b"flat" for _ in range(100)],
        ping_sequence=[ok_ping_reply() for _ in range(100)],
    )

    import xtuner.v1.ray.environment.rl_task.runner as r
    orig_stale = r._WATCHDOG_STALE_SEC
    orig_poll = r._WATCHDOG_POLL_SEC
    r._WATCHDOG_STALE_SEC = 999
    r._WATCHDOG_POLL_SEC = 0.01
    try:
        with pytest.raises(InferBoom, match="blew up"):
            await _run_infer_with_watchdog(infer, client, {}, "tid-infer-boom")
    finally:
        r._WATCHDOG_STALE_SEC = orig_stale
        r._WATCHDOG_POLL_SEC = orig_poll
