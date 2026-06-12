"""Fast tests for asyncio diagnostics.

These tests cover the control-plane contract only:
- diagnostics are disabled unless XTUNER_ASYNCIO_DIAGNOSTICS is set;
- signal installation success is tracked by (pid, loop_id), so a new loop can
  reinstall handlers while repeated calls on the same loop are skipped;
- if both signal handlers fail, installation returns False and the watchdog is
  not started by asyncio_run.

The tests use fake event loops and monkeypatched signal functions so they do
not install process-wide signal handlers or depend on Ray/GPU backends.
"""

from __future__ import annotations

import asyncio

from xtuner.v1.rl.utils import async_utils, asyncio_diagnostics


class _FakeLoop:
    def __init__(self, *, fail_sigusr1: bool = False):
        self.fail_sigusr1 = fail_sigusr1
        self.signal_handlers: list[tuple[object, object]] = []

    def add_signal_handler(self, sig, callback):
        if self.fail_sigusr1:
            raise RuntimeError("not in main thread")
        self.signal_handlers.append((sig, callback))


def test_install_asyncio_diagnostics_is_disabled_without_env(monkeypatch):
    monkeypatch.delenv("XTUNER_ASYNCIO_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(asyncio_diagnostics, "_INSTALLED_FOR", None)

    assert asyncio_diagnostics.install_asyncio_diagnostics(_FakeLoop()) is False
    assert asyncio_diagnostics._INSTALLED_FOR is None


def test_install_asyncio_diagnostics_tracks_pid_and_loop(monkeypatch):
    monkeypatch.setenv("XTUNER_ASYNCIO_DIAGNOSTICS", "1")
    monkeypatch.setattr(asyncio_diagnostics, "_INSTALLED_FOR", None)
    monkeypatch.setattr(asyncio_diagnostics.faulthandler, "enable", lambda **kwargs: None)

    installed_sigusr2: list[object] = []

    def fake_signal(sig, handler):
        installed_sigusr2.append(sig)

    monkeypatch.setattr(asyncio_diagnostics.signal, "signal", fake_signal)

    loop1 = _FakeLoop()
    assert asyncio_diagnostics.install_asyncio_diagnostics(loop1) is True
    assert len(loop1.signal_handlers) == 1
    assert len(installed_sigusr2) == 1

    assert asyncio_diagnostics.install_asyncio_diagnostics(loop1) is True
    assert len(loop1.signal_handlers) == 1
    assert len(installed_sigusr2) == 1

    loop2 = _FakeLoop()
    assert asyncio_diagnostics.install_asyncio_diagnostics(loop2) is True
    assert len(loop2.signal_handlers) == 1
    assert len(installed_sigusr2) == 2


def test_failed_install_does_not_start_watchdog(monkeypatch):
    monkeypatch.setenv("XTUNER_ASYNCIO_DIAGNOSTICS", "1")
    monkeypatch.setattr(asyncio_diagnostics, "_INSTALLED_FOR", None)
    monkeypatch.setattr(asyncio_diagnostics.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(
        asyncio_diagnostics.signal,
        "signal",
        lambda sig, handler: (_ for _ in ()).throw(ValueError("not in main thread")),
    )

    assert asyncio_diagnostics.install_asyncio_diagnostics(_FakeLoop(fail_sigusr1=True)) is False
    assert asyncio_diagnostics._INSTALLED_FOR is None

    start_calls = 0

    def fake_start_asyncio_run_watchdog(*, loop):
        nonlocal start_calls
        start_calls += 1
        return None

    monkeypatch.setattr(asyncio_diagnostics, "install_asyncio_diagnostics", lambda loop: False)
    monkeypatch.setattr(asyncio_diagnostics, "start_asyncio_run_watchdog", fake_start_asyncio_run_watchdog)

    loop = asyncio.new_event_loop()
    try:
        assert async_utils.asyncio_run(asyncio.sleep(0, result="done"), loop=loop) == "done"
    finally:
        loop.close()
    assert start_calls == 0


def test_dump_asyncio_tasks_writes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XTUNER_ASYNCIO_DIAG_DIR", str(tmp_path))

    async def run_dump():
        asyncio_diagnostics.dump_asyncio_tasks(reason="unit-test")

    asyncio.run(run_dump())

    dump_files = list(tmp_path.glob("asyncio_tasks_*.txt"))
    assert len(dump_files) == 1

    content = dump_files[0].read_text(encoding="utf-8")
    assert "reason=unit-test" in content
    assert "loop_id=current" in content
    assert "task_count=" in content

