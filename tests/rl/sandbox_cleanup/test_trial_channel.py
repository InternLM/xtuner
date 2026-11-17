"""Real socket and process checks for actor-owned sandbox creation permits."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "xtuner/v1/rl/agent_loop/sandbox_agent_loop"
sys.path.insert(0, str(SOURCE))
import trial_channel  # noqa: E402


@pytest.fixture
def shared_limiter(monkeypatch):
    # Load the real optional dependency without importing its unrelated extras.
    lagent = SOURCE.parents[5] / "lagent-sandbox-cleanup/lagent/utils/rate_limiter.py"
    if not lagent.is_file():
        pytest.skip("Lagent stack source is required for the shared limiter integration")
    spec = importlib.util.spec_from_file_location("trial_rate_limiter", lagent)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(trial_channel, "_local_limiter", module.get_shared_async_token_bucket)
    return module.get_shared_async_token_bucket


def test_no_child_context_keeps_existing_registry(shared_limiter):
    key = uuid4().hex
    assert trial_channel.get_creation_limiter(key, 2, 1) is shared_limiter(key, 2, 1)


def test_independent_children_share_one_parent_bucket(shared_limiter):
    async def run():
        key = uuid4().hex
        # Consume initial burst before launching children, so child startup
        # cannot hide an accidentally replenished per-child bucket.
        limiter = shared_limiter(key, 8, 1)
        children, servers = [], []
        script = """
import asyncio,json,sys,time
sys.path.insert(0,sys.argv[1])
from trial_channel import child_channel,get_creation_limiter
async def main():
    async with child_channel(int(sys.argv[2])):
        print('ready',flush=True)
        await asyncio.get_running_loop().run_in_executor(None,sys.stdin.readline)
        await get_creation_limiter(sys.argv[3],8,1).acquire()
        print(json.dumps(time.monotonic()),flush=True)
asyncio.run(main())
"""
        try:
            for _ in range(4):
                parent, child = socket.socketpair()
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    script,
                    str(SOURCE),
                    str(child.fileno()),
                    key,
                    pass_fds=(child.fileno(),),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                child.close()
                children.append(process)
                servers.append(asyncio.create_task(trial_channel.serve_channel(parent)))
            for process in children:
                assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
            await limiter.acquire()
            started = time.monotonic()
            for process in children:
                process.stdin.write(b"go\n")
                await process.stdin.drain()
            times = sorted([json.loads(await asyncio.wait_for(p.stdout.readline(), 5)) for p in children])
            assert times[-1] - started >= 0.40
            assert all(right - left >= 0.08 for left, right in zip(times, times[1:]))
            for process in children:
                assert await asyncio.wait_for(process.wait(), 5) == 0
            await asyncio.wait_for(asyncio.gather(*servers), 2)
        finally:
            for process in children:
                if process.returncode is None:
                    process.kill()
                await asyncio.wait_for(process.communicate(), 5)
            for server in servers:
                server.cancel()
            await asyncio.gather(*servers, return_exceptions=True)

    asyncio.run(run())


def test_canceled_waiter_cannot_steal_later_response(monkeypatch):
    async def run():
        gates, entered, canceled = {}, {}, []

        class Limiter:
            def __init__(self, key):
                self.key = key

            async def acquire(self):
                entered[self.key].set()
                try:
                    await gates[self.key].wait()
                except asyncio.CancelledError:
                    canceled.append(self.key)
                    raise

        monkeypatch.setattr(trial_channel, "_local_limiter", lambda key, *_: Limiter(key))
        for key in ("cancelled", "survivor", "next"):
            gates[key], entered[key] = asyncio.Event(), asyncio.Event()
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        async with trial_channel.child_channel(child.detach()):
            first = asyncio.create_task(trial_channel.get_creation_limiter("cancelled", 1, 1).acquire())
            second = asyncio.create_task(trial_channel.get_creation_limiter("survivor", 1, 1).acquire())
            await entered["cancelled"].wait()
            await entered["survivor"].wait()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            gates["survivor"].set()
            await asyncio.wait_for(second, 1)
            third = asyncio.create_task(trial_channel.get_creation_limiter("next", 1, 1).acquire())
            await entered["next"].wait()
            assert not third.done()
            gates["next"].set()
            await asyncio.wait_for(third, 1)
            assert canceled == ["cancelled"]
        await asyncio.wait_for(server, 1)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_disconnect_cancels_and_settles_parent_waiters(monkeypatch):
    async def run():
        entered, stopped = asyncio.Event(), asyncio.Event()

        class Limiter:
            async def acquire(self):
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    stopped.set()

        monkeypatch.setattr(trial_channel, "_local_limiter", lambda *_: Limiter())
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        child.sendall(b'{"op":"acquire","id":1,"key":"pending","rate":1,"capacity":1}\n')
        await entered.wait()
        child.close()
        await asyncio.wait_for(server, 1)
        assert stopped.is_set()

    asyncio.run(asyncio.wait_for(run(), 5))


def test_parent_disconnect_fails_child_waiters():
    async def run():
        parent, child = socket.socketpair()
        parent.setblocking(False)
        async with trial_channel.child_channel(child.detach()):
            waiter = asyncio.create_task(trial_channel.get_creation_limiter("pending", 1, 1).acquire())
            await asyncio.get_running_loop().sock_recv(parent, 4096)
            parent.close()
            with pytest.raises(ConnectionError):
                await asyncio.wait_for(waiter, 1)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_oversized_message_is_rejected():
    async def run():
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        child.sendall(b"x" * 4097)
        with pytest.raises(ValueError, match="too large"):
            await asyncio.wait_for(server, 1)
        child.close()

    asyncio.run(asyncio.wait_for(run(), 5))


@pytest.mark.parametrize("failure_side", ["parent", "child"])
def test_write_failure_disconnects_waiters_without_task_errors(monkeypatch, failure_side):
    async def run():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _, context: errors.append(context))
        original = trial_channel._send

        async def fail_reply(sock, lock, message):
            if (failure_side == "parent" and message.get("ok")) or (
                failure_side == "child" and message.get("op") == "acquire"
            ):
                raise OSError("injected reply failure")
            await original(sock, lock, message)

        class Limiter:
            async def acquire(self):
                return None

        monkeypatch.setattr(trial_channel, "_send", fail_reply)
        monkeypatch.setattr(trial_channel, "_local_limiter", lambda *_: Limiter())
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        async with trial_channel.child_channel(child.detach()):
            with pytest.raises((ConnectionError, OSError)):
                await trial_channel.get_creation_limiter("write-error", 1, 1).acquire()
        await server
        await asyncio.sleep(0)
        assert not errors

    asyncio.run(asyncio.wait_for(run(), 5))


def test_canceled_server_settles_active_permits_and_fails_child_waiters(monkeypatch):
    async def run():
        entered, stopped = [], []

        class Limiter:
            async def acquire(self):
                entered.append(True)
                try:
                    await asyncio.Future()
                finally:
                    stopped.append(True)

        monkeypatch.setattr(trial_channel, "_local_limiter", lambda *_: Limiter())
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        async with trial_channel.child_channel(child.detach()):
            waiters = [
                asyncio.create_task(trial_channel.get_creation_limiter("pending", 1, 1).acquire()) for _ in range(2)
            ]
            while len(entered) < 2:
                await asyncio.sleep(0)
            server.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server
            results = await asyncio.gather(*waiters, return_exceptions=True)
            assert all(isinstance(result, ConnectionError) for result in results)
            assert len(stopped) == 2

    asyncio.run(asyncio.wait_for(run(), 5))


def test_cancellation_during_socket_write_preserves_later_frames(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        writing, resume = asyncio.Event(), asyncio.Event()
        original = loop.sock_sendall

        async def stalled_write(sock, data):
            if not writing.is_set() and b'"op":"acquire"' in data:
                writing.set()
                await resume.wait()
            await original(sock, data)

        class Limiter:
            async def acquire(self):
                return None

        monkeypatch.setattr(loop, "sock_sendall", stalled_write)
        monkeypatch.setattr(trial_channel, "_local_limiter", lambda *_: Limiter())
        parent, child = socket.socketpair()
        server = asyncio.create_task(trial_channel.serve_channel(parent))
        async with trial_channel.child_channel(child.detach()):
            first = asyncio.create_task(trial_channel.get_creation_limiter("first", 1, 1).acquire())
            await writing.wait()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            survivor = asyncio.create_task(trial_channel.get_creation_limiter("survivor", 1, 1).acquire())
            resume.set()
            await survivor
        await server

    asyncio.run(asyncio.wait_for(run(), 5))
