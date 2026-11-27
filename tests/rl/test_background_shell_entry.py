"""Focused command-ID contract checks; no SDK, sandbox, Ray or GPU required."""

import asyncio
import functools
from types import SimpleNamespace

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.background_entry import BackgroundShellEntry
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import StageRecord, StageStatus


def run_async(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


class Client:
    def __init__(self, rc=0, *, running=False):
        self.rc, self.running = rc, running
        self.starts = 0
        self.interrupted = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.delay_start = False

    async def start_command(self, command, cwd, timeout):
        self.starts += 1
        self.command, self.cwd = command, cwd
        self.started.set()
        if self.delay_start:
            await self.release.wait()
        return "command-1"

    async def get_command_status(self, command_id):
        assert command_id == "command-1"
        return {"running": self.running, "return_code": self.rc}

    async def get_command_output(self, command_id):
        assert command_id == "command-1"
        return {"stdout": "ok", "stderr": ""}

    async def interrupt_command(self, command_id):
        assert command_id == "command-1"
        self.interrupted = True
        self.running = False


def entry(**kwargs):
    return BackgroundShellEntry("echo done", poll_interval_sec=0.001, cleanup_timeout_sec=0.1, **kwargs)


@run_async
@pytest.mark.parametrize("rc,reason", [(0, "entry_completed"), (7, "entry_failed"), (None, "exit_code_missing")])
async def test_terminal_command_needs_no_pid_or_rc_file(rc, reason):
    client, record = Client(rc), StageRecord(workspace="/workspace")
    outcome = await entry().run(client, SimpleNamespace(), record)
    assert (outcome.reason, outcome.result.return_code) == (reason, rc)
    assert outcome.ok == (rc == 0)
    assert client.starts == 1
    assert client.cwd == "/workspace"
    assert record.entries[0].outcome == outcome
    assert record.entries[0].rc_file is None


@run_async
@pytest.mark.parametrize("during_launch", [False, True])
async def test_cancel_waits_for_same_command_interrupt(during_launch):
    client, record = Client(running=True), StageRecord()
    client.delay_start = during_launch
    task = asyncio.create_task(entry().run(client, SimpleNamespace(), record))
    await client.started.wait()
    task.cancel()
    client.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.starts == 1 and client.interrupted
    assert record.entries[0].status == StageStatus.FAILED
    assert record.entries[0].outcome.details["command_terminal_after_cleanup"]


@run_async
async def test_timeout_interrupts_instead_of_relaunching():
    client = Client(running=True)
    outcome = await entry(timeout=0.01, host_grace_sec=0).run(client, SimpleNamespace(), StageRecord())
    assert outcome.reason == "timeout"
    assert client.interrupted and client.starts == 1
