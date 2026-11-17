"""CPU lifecycle tests: python tests/rl/sandbox_cleanup_cpu.py.

Loads the actual Pool, Runner, schemas, finalizer and recovery modules. Only
training package initializers and optional import-only helpers are bypassed.
Management calls use real loopback TCP HTTP, without GPU or cloud resources.
Requires pydantic v2; no pytest or training dependency installation is needed.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "xtuner.v1.rl.agent_loop.sandbox_agent_loop"
SOURCE = ROOT / PACKAGE.replace(".", "/")


def load_runtime():
    for name in ("xtuner", "xtuner.v1", "xtuner.v1.rl", "xtuner.v1.rl.agent_loop", PACKAGE):
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / name.replace(".", "/"))]
        sys.modules[name] = module
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules[parent], child, module)
    utils = types.ModuleType("xtuner.v1.utils")
    utils.get_logger = lambda: logging.getLogger("cleanup-test")
    sys.modules[utils.__name__] = utils
    # Pool tests never execute commands through httpx; management HTTP below is real.
    sys.modules.setdefault("httpx", types.ModuleType("httpx"))
    sys.modules.setdefault("lagent", types.ModuleType("lagent"))
    lagent = types.ModuleType("lagent.utils")

    def create_object(value):
        if not isinstance(value, dict):
            return value
        config = dict(value)
        return config.pop("type")(**config)

    lagent.create_object = create_object
    lagent.ctx_session_id = contextvars.ContextVar("session_id", default=None)
    sys.modules[lagent.__name__] = lagent
    limiter = types.ModuleType("lagent.utils.rate_limiter")
    limiter.get_shared_async_token_bucket = lambda *args: None
    sys.modules[limiter.__name__] = limiter
    if override := os.environ.get("XTUNER_TEST_POOL_SOURCE"):
        spec = importlib.util.spec_from_file_location(PACKAGE + ".sandbox", override)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(PACKAGE + ".sandbox")


sandbox = load_runtime()
cleanup = importlib.import_module(PACKAGE + ".cleanup")
schemas = importlib.import_module(PACKAGE + ".schemas")
Runner = importlib.import_module(PACKAGE + ".runner").Runner
Judger = importlib.import_module(PACKAGE + ".judger").Judger
CleanupError = importlib.import_module(PACKAGE + ".finalization").CleanupError


class HTTPError(RuntimeError):
    def __init__(self, status):
        self.status_code = status


class Client:
    def __init__(self, healthy=True):
        self.healthy = healthy
        self.closed = False

    async def health_check(self):
        return {"ok": self.healthy}

    async def aclose(self):
        self.closed = True


class Provider:
    supports_deletion_confirmation = True

    def __init__(self, port, *, healthy=True):
        self.port, self.healthy = port, healthy
        self.owned_env_ids = set()
        self.closed = False

    async def request(self, method, path):
        if self.closed:
            raise RuntimeError("closed transport")
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        try:
            writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
            headers = await reader.readuntil(b"\r\n\r\n")
            status = int(headers.split(b" ")[1])
            body = await reader.read()
            if status >= 400:
                raise HTTPError(status)
            return json.loads(body)
        finally:
            writer.close()
            await writer.wait_closed()

    async def create(self, image_tag, **kwargs):
        result = await self.request("POST", "/" + image_tag)
        self.owned_env_ids.add(result["id"])
        return Client(self.healthy), result["id"]

    async def delete(self, env_id):
        await self.request("DELETE", "/" + env_id)
        # Models providers dropping ownership on an accepted asynchronous DELETE.
        self.owned_env_ids.discard(env_id)

    async def get(self, env_id):
        return await self.request("GET", "/" + env_id)

    async def aclose(self):
        self.closed = True


class TestSandboxCleanup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Match production loop settings. Unittest enables asyncio debug by
        # default; stack collection from a network-mounted Python environment
        # can consume the entire subsecond fault-injection budget by itself.
        asyncio.get_running_loop().set_debug(False)
        env_patch = patch.dict(os.environ, {"SANDBOX_PROVIDER_USER": "cleanup-test-user"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.live = set()
        self.calls = []
        self.fail_delete = False
        self.accept_only = False
        self.get_error = False
        self.slow_delete = None
        self.fail_create = None
        self.next_id = 0
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.providers = []
        self.pools = []

    async def asyncTearDown(self):
        for pool in self.pools:
            if pool._lifecycle._journal and not pool._lifecycle._journal.lock.closed:
                pool._lifecycle._journal.lock.close()
        await cleanup.drain_cleanup_tasks(2)
        self.server.close()
        await self.server.wait_closed()
        self.temp.cleanup()

    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            method, path, _ = head.split(b"\r\n")[0].decode().split()
            env_id = path[1:]
            self.calls.append((method, env_id))
            status, body = 200, {}
            if method == "POST":
                if env_id == self.fail_create:
                    status = 503
                else:
                    self.next_id += 1
                    env_id += "-" + str(self.next_id)
                    self.live.add(env_id)
                    body = {"id": env_id}
            elif method == "DELETE":
                if env_id == self.slow_delete:
                    await asyncio.sleep(0.6)
                if self.fail_delete:
                    status = 503
                elif env_id not in self.live:
                    status = 404
                else:
                    status = 202
                    if not self.accept_only:
                        self.live.remove(env_id)
            elif self.get_error:
                status = 503
            elif env_id in self.live:
                body = {"env": {"status": "running"}}
            else:
                status = 404
            raw = json.dumps(body).encode()
            writer.write(
                f"HTTP/1.1 {status} response\r\nContent-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def provider(self, key="gateway-test"):
        self.assertEqual(key, "gateway-test")
        provider = Provider(self.port)
        self.providers.append(provider)
        return provider

    def pool(self, *, persistent=True, recovery=True, **kwargs):
        provider = self.provider()
        pool = sandbox.SandboxPool(
            provider=provider,
            close_provider=True,
            specs={
                "main": schemas.SandboxSpec(image="main", dependencies={"dep": schemas.SandboxSpec(image="dep")}),
                "verifier": schemas.SandboxSpec(image="verifier"),
            },
            creates_per_sec=None,
            max_attempts=1,
            cleanup_timeout_sec=0.4,
            cleanup_retry_timeout_sec=0.5,
            cleanup_provider={"type": lambda: self.provider()} if recovery else None,
            cleanup_dir=str(self.directory) if persistent else None,
            cleanup_provider_key="gateway-test",
            **kwargs,
        )
        self.pools.append(pool)
        return pool

    async def _abandoned_registration(self, *, cancel):
        pool = self.pool()
        pool._lifecycle._cleanup_timeout_sec = 0.08
        entered, release = threading.Event(), threading.Event()
        flush = cleanup.CleanupJournal.flush

        def stalled_flush(journal):
            if not entered.is_set():
                entered.set()
                release.wait(2)
            flush(journal)

        with patch.object(cleanup.CleanupJournal, "flush", stalled_flush):
            started = time.monotonic()
            task = asyncio.create_task(pool.get("main"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                await asyncio.sleep(0.01)  # Other tasks must run while registration is blocked.
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(self.calls, [])
                if cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError if cancel else CleanupError):
                    await asyncio.wait_for(task, 1)
                self.assertIsNone(pool._lifecycle._journal)
                self.assertEqual(self.calls, [], "Abandoned registration must not issue POST")
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                paths = list(self.directory.glob("*.json"))
                if paths and json.loads(paths[0].read_text())["state"] == "complete":
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(paths), 1)
            data = json.loads(paths[0].read_text())
            self.assertEqual(data["state"], "complete")
            self.assertEqual(data["resources"], {})
            # The initializer thread must release its late owner even though
            # the caller never received that journal.
            with cleanup.locking.open_lock(paths[0].with_suffix(".lock")) as lock:
                cleanup.locking.acquire(lock)
            self.assertEqual(self.calls, [])
        await pool.aclose()

    async def test_cancelled_initial_registration_detaches_late_journal_without_post(self):
        await self._abandoned_registration(cancel=True)

    async def test_initial_registration_timeout_detaches_late_journal_without_post(self):
        await self._abandoned_registration(cancel=False)

    async def test_pool_observation_never_allocates_or_reopens_a_phase(self):
        pool = self.pool()
        self.assertIsNone(pool.peek("main"))
        self.assertEqual(self.calls, [])
        client = await pool.get("main")
        self.assertIs(pool.peek("main"), client)
        owned = pool.pending_resource_ids
        self.assertIsInstance(owned, frozenset)
        self.assertEqual(owned, frozenset(self.live))
        await pool.release_all()
        calls = list(self.calls)
        self.assertIsNone(pool.peek("main"))
        self.assertFalse(pool.pending_resource_ids)
        self.assertTrue(owned, "An earlier observation must remain an immutable snapshot")
        self.assertEqual(self.calls, calls)
        await pool.aclose()

    async def test_task_scope_preserves_primary_error_and_finalizes(self):
        for error in (ValueError("business failure"), asyncio.CancelledError("cancelled")):
            with self.subTest(error=type(error).__name__):
                pool = self.pool()
                with self.assertRaises(type(error)) as observed:
                    async with pool.task_scope("public-contract"):
                        await pool.get("main")
                        raise error
                self.assertIs(observed.exception, error)
                self.assertFalse(self.live)
                self.assertTrue(pool.cleanup["provider_closed"])
                data = json.loads(Path(pool.cleanup["journal"]).read_text())
                self.assertEqual(data["state"], "complete")
                self.assertEqual(data["trial_id"], "public-contract")

    async def test_task_scope_honors_specialized_public_final_close(self):
        closed = []

        class SpecializedPool(sandbox.SandboxPool):
            async def aclose(self):
                closed.append("close")
                await super().aclose()

        provider = self.provider()
        pool = SpecializedPool(
            provider=provider,
            close_provider=True,
            specs={"main": schemas.SandboxSpec(image="main")},
            creates_per_sec=None,
        )
        self.pools.append(pool)
        async with pool.task_scope("specialized"):
            await pool.get("main")
        self.assertEqual(closed, ["close"])
        self.assertTrue(provider.closed)
        self.assertFalse(self.live)

    async def test_task_scope_preserves_primary_when_cleanup_also_fails(self):
        for error in (ValueError("business failure"), asyncio.CancelledError("cancelled")):
            with self.subTest(error=type(error).__name__):
                pool = self.pool()
                self.fail_delete = True
                with self.assertRaises(type(error)) as observed:
                    async with pool.task_scope("failed-cleanup"):
                        await pool.get("main")
                        raise error
                self.assertIs(observed.exception, error)
                self.assertFalse(pool.cleanup["ok"])
                self.fail_delete = False
                await pool._lifecycle._deferred_cleanup
                self.assertFalse(self.live)

    async def test_public_release_reports_failure_inside_handled_exception(self):
        pool = self.pool()
        await pool.get("main")
        self.fail_delete = True
        try:
            raise ValueError("unrelated caller fallback")
        except ValueError:
            with self.assertRaises(CleanupError):
                await pool.release_all()
        self.assertFalse(pool.cleanup["ok"])
        self.assertEqual(pool.pending_resource_ids, frozenset(self.live))
        self.fail_delete = False
        await pool.aclose()
        self.assertFalse(self.live)

    async def test_public_final_close_reports_failure_inside_handled_exception(self):
        pool = self.pool()
        await pool.get("main")
        self.fail_delete = True
        try:
            raise ValueError("unrelated caller fallback")
        except ValueError:
            with self.assertRaises(CleanupError):
                await pool.aclose()
        self.assertFalse(pool.cleanup["ok"])
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)

    async def test_concurrent_registration_uses_one_journal(self):
        pool = self.pool()
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = cleanup.CleanupJournal.open

        async def delayed(*args, **kwargs):
            entered.set()
            await proceed.wait()
            return await original(*args, **kwargs)

        with patch.object(cleanup.CleanupJournal, "open", side_effect=delayed) as opened:
            first = asyncio.create_task(pool._lifecycle._ensure_journal())
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(pool._lifecycle._ensure_journal())
            await asyncio.sleep(0)
            proceed.set()
            await asyncio.wait_for(asyncio.gather(first, second), 2)
            self.assertEqual(opened.call_count, 1)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)
        await pool.aclose()
        self.assertTrue(pool._lifecycle._journal.lock.closed)

    async def test_close_during_registration_prevents_allocation_and_detaches(self):
        pool = self.pool()
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = cleanup.CleanupJournal.open

        async def delayed(*args, **kwargs):
            entered.set()
            await proceed.wait()
            return await original(*args, **kwargs)

        with patch.object(cleanup.CleanupJournal, "open", side_effect=delayed):
            creation = asyncio.create_task(pool.get("main"))
            await asyncio.wait_for(entered.wait(), 1)
            closing = asyncio.create_task(pool.aclose())
            await asyncio.sleep(0)
            self.assertTrue(pool._lifecycle._closing)
            proceed.set()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await asyncio.wait_for(creation, 2)
            await asyncio.wait_for(closing, 2)
        self.assertEqual(self.calls, [])
        self.assertTrue(pool._lifecycle._journal.lock.closed)
        self.assertEqual(json.loads(pool._lifecycle._journal.path.read_text())["state"], "complete")

    async def test_get_method_alone_does_not_enable_deletion_confirmation(self):
        class UnnormalizedProvider(Provider):
            supports_deletion_confirmation = False

            async def get(self, env_id):
                raise AssertionError("An arbitrary status interface is not a confirmation contract")

        pool = sandbox.SandboxPool(
            provider={"type": UnnormalizedProvider, "port": self.port},
            specs={"main": schemas.SandboxSpec(image="main")},
            creates_per_sec=None,
        )
        self.pools.append(pool)
        await pool.get("main")
        await pool.aclose()
        self.assertTrue(pool.cleanup["ok"])
        self.assertFalse(self.live)
        self.assertFalse(any(method == "GET" for method, _ in self.calls))

    async def test_persistent_cleanup_requires_explicit_confirmation_contract(self):
        class LegacyProvider(Provider):
            supports_deletion_confirmation = False

        for verify in (None, False):
            with self.subTest(verify=verify), self.assertRaisesRegex(ValueError, "persistent cleanup"):
                sandbox.SandboxPool(
                    provider={"type": LegacyProvider, "port": self.port},
                    specs={},
                    cleanup_dir=str(self.directory),
                    cleanup_provider_key="gateway-test",
                    verify_deletion=verify,
                )
        self.assertEqual(self.calls, [])
        self.assertFalse(list(self.directory.glob("*.json")))

        # Explicit opt-in is allowed for a provider whose get interface already
        # returns the documented normalized result but has no capability flag.
        pool = sandbox.SandboxPool(
            provider={"type": LegacyProvider, "port": self.port},
            specs={"main": schemas.SandboxSpec(image="main")},
            creates_per_sec=None,
            cleanup_dir=str(self.directory),
            cleanup_provider_key="gateway-test",
            verify_deletion=True,
        )
        self.pools.append(pool)
        await pool.get("main")
        await pool.aclose()
        self.assertEqual(json.loads(pool._lifecycle._journal.path.read_text())["state"], "complete")
        self.assertTrue(any(method == "GET" for method, _ in self.calls))

    async def test_memory_recovery_preserves_foreground_delete_acceptance_policy(self):
        class NoQueryProvider(Provider):
            supports_deletion_confirmation = False
            get = None

        class QueryDisabledProvider(Provider):
            async def get(self, env_id):
                raise AssertionError("Explicit verify_deletion=False also applies to recovery")

        for provider_type, verify in ((NoQueryProvider, None), (QueryDisabledProvider, False)):
            with self.subTest(provider=provider_type.__name__):
                pool = sandbox.SandboxPool(
                    provider={"type": provider_type, "port": self.port},
                    specs={"main": schemas.SandboxSpec(image="main")},
                    creates_per_sec=None,
                    cleanup_timeout_sec=0.2,
                    cleanup_retry_timeout_sec=0.3,
                    verify_deletion=verify,
                )
                self.pools.append(pool)
                await pool.get("main")
                self.fail_delete = True
                with self.assertRaises(CleanupError):
                    await pool.aclose()
                self.fail_delete = False
                result = await pool._lifecycle._deferred_cleanup
                self.assertTrue(result["ok"])
                self.assertFalse(result["deletion_confirmed"])
                self.assertFalse(self.live)
        self.assertFalse(any(method == "GET" for method, _ in self.calls))

    async def test_delete_retry_preserves_reverse_acquisition_order(self):
        pool = self.pool()
        await pool.get("main")
        delete = pool._lifecycle._provider.delete
        attempts = []

        async def fail_first(env_id):
            attempts.append(env_id)
            if attempts.count(env_id) == 1:
                raise HTTPError(503)
            await delete(env_id)

        pool._lifecycle._provider.delete = fail_first
        await pool.aclose()
        self.assertEqual(attempts, ["main-2", "dep-1", "main-2", "dep-1"])
        self.assertFalse(self.live)

    async def test_cleanup_snapshot_cannot_change_release_or_final_close(self):
        pool = self.pool(persistent=False, recovery=False)
        await pool.get("main")
        early = pool.cleanup
        early.update(ok=True, provider_closed=True)
        await pool.release_all()
        self.assertFalse(self.live)
        snapshot = pool.cleanup
        snapshot["delete_attempts"][0]["env_id"] = "foreign-id"
        snapshot["closed_client_count"] = -1
        self.assertNotEqual(pool.cleanup["delete_attempts"][0]["env_id"], "foreign-id")
        self.assertGreater(pool.cleanup["closed_client_count"], 0)
        self.assertNotIn("delete_attempts", early)
        deletes = [call for call in self.calls if call[0] == "DELETE"]
        await pool.release_all()
        self.assertEqual([call for call in self.calls if call[0] == "DELETE"], deletes)
        close = pool._lifecycle._provider.aclose
        closes = []

        async def counted_close():
            closes.append(True)
            await close()

        pool._lifecycle._provider.aclose = counted_close
        await pool.aclose()
        pool.cleanup.update(ok=False, provider_closed=False)
        await pool.aclose()
        self.assertEqual(closes, [True])

    async def test_phase_release_then_verifier_then_final_close(self):
        pool = self.pool()
        await pool.get("main")
        await pool.release_all()
        self.assertFalse(pool._lifecycle._provider.closed)
        await pool.get("verifier")
        await pool.aclose()
        self.assertTrue(pool._lifecycle._provider.closed)
        self.assertFalse(self.live)
        deletes = [key for method, key in self.calls if method == "DELETE"]
        self.assertEqual(deletes, ["main-2", "dep-1", "verifier-3"])
        self.assertEqual(json.loads(pool._lifecycle._journal.path.read_text())["state"], "complete")
        with self.assertRaises(RuntimeError):
            await pool.get("main")

    async def test_accepted_delete_is_pending_until_confirmed(self):
        self.accept_only = True
        pool = self.pool(recovery=False)
        await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.release_all()
        self.assertFalse(pool.cleanup["ok"])
        self.assertEqual(len(pool._lifecycle._pending_delete_ids), 2)
        self.accept_only = False
        await pool.release_all()
        self.assertTrue(pool.cleanup["ok"])
        await pool.aclose()

    async def test_fresh_provider_recovers_after_foreground_transport_closed(self):
        self.fail_delete = True
        pool = self.pool()
        await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.aclose()
        self.assertTrue(pool._lifecycle._provider.closed)
        self.assertTrue(pool.cleanup["deferred"])
        self.fail_delete = False
        result = await pool._lifecycle._deferred_cleanup
        self.assertTrue(result["ok"])
        self.assertFalse(self.live)
        self.assertGreater(len(self.providers), 1)
        self.assertFalse(pool.cleanup["ok"], "background success must not revive a failed sample")

    async def test_long_outage_retains_ids_for_later_reconciliation(self):
        self.fail_delete = True
        pool = self.pool()
        await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.aclose()
        result = await pool._lifecycle._deferred_cleanup
        self.assertFalse(result["ok"])
        data = json.loads(pool._lifecycle._journal.path.read_text())
        self.assertEqual(len(data["pending_ids"]), 2)
        self.assertEqual(data["state"], "pending")
        self.fail_delete = False
        result = await cleanup.reconcile(self.directory, self.provider, 0.5)
        self.assertTrue(result["ok"])
        self.assertFalse(self.live)

    async def test_missing_factory_retains_transport_for_in_memory_retry(self):
        self.fail_delete = True
        pool = self.pool(persistent=False, recovery=False)
        await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.aclose()
        self.assertFalse(pool._lifecycle._provider.closed)
        self.fail_delete = False
        await pool.release_all()
        self.assertFalse(self.live)
        await pool.aclose()
        self.assertTrue(pool._lifecycle._provider.closed)

    async def test_query_error_does_not_forget_accepted_id(self):
        pool = self.pool(recovery=False)
        await pool.get("main")
        self.accept_only = self.get_error = True
        with self.assertRaises(CleanupError):
            await pool.release_all()
        self.assertEqual(len(pool._lifecycle._pending_verification_ids), 2)
        self.accept_only = self.get_error = False
        await pool.aclose()
        self.assertFalse(self.live)

    async def test_slow_primary_does_not_starve_dependency(self):
        pool = self.pool()
        await pool.get("main")
        self.slow_delete = "main-2"
        started = time.monotonic()
        with self.assertRaises(CleanupError):
            await pool.aclose()
        self.assertLess(time.monotonic() - started, 1)
        self.assertNotIn("dep-1", self.live)
        self.slow_delete = None
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)

    async def test_partial_create_failure_records_unknown_and_releases_dependency(self):
        pool = self.pool()
        self.fail_create = "main"
        with self.assertRaises(HTTPError):
            await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.aclose()
        self.assertFalse(self.live)
        data = json.loads(pool._lifecycle._journal.path.read_text())
        self.assertTrue(data["resources"]["dep-1"]["confirmed"])
        self.assertEqual(data["ambiguous_creates"], 1)
        self.assertEqual(sum(method == "POST" for method, _ in self.calls), 2)

    async def test_shared_provider_keeps_foreign_resource_and_transport(self):
        pool = self.pool()
        pool._lifecycle._close_provider = False
        pool._lifecycle._provider.owned_env_ids.add("foreign")
        self.live.add("foreign")
        await pool.get("main")
        await pool.aclose()
        self.assertEqual(self.live, {"foreign"})
        self.assertFalse(pool._lifecycle._provider.closed)

    async def test_cancelled_journal_write_finishes_before_unlock(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        entered, release = threading.Event(), threading.Event()
        original = journal.created

        def slow(env_id):
            entered.set()
            release.wait(1)
            original(env_id)

        journal.created = slow
        task = asyncio.create_task(journal.persist(journal.created, "owned"))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        result = await cleanup.reconcile(self.directory, self.provider, 0.2)
        self.assertEqual(result["results"][0]["state"], "active")
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await journal.persist(journal.detach)
        self.assertIn("owned", json.loads(journal.path.read_text())["resources"])

    async def test_independent_recovery_cli_and_exit_fallback(self):
        fixture = self.directory / "recovery_fixture.py"
        fixture.write_text(
            "import os\nfrom sandbox_cleanup_cpu import Provider\n"
            "def resolve(key):\n"
            "    assert key == 'gateway-test'\n"
            "    return Provider(int(os.environ['CLEANUP_TEST_PORT']))\n"
        )
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("cli-owned")
        journal.detach()
        self.live.add("cli-owned")
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(self.directory), str(Path(__file__).parent)]),
            "CLEANUP_TEST_PORT": str(self.port),
        }
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            str(SOURCE / "cleanup.py"),
            "--directory",
            str(self.directory),
            "--factory",
            "recovery_fixture:resolve",
            "--timeout",
            "0.3",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(child.communicate(), 5)
        self.assertEqual(child.returncode, 0, stderr.decode())
        self.assertTrue(json.loads(stdout)["ok"])
        self.assertFalse(self.live)
        exit_journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        exit_journal.created("exit-owned")
        exit_journal.detach()
        self.live.add("exit-owned")
        with patch.dict(
            os.environ,
            {
                **environment,
                "XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory),
                "XTUNER_SANDBOX_CLEANUP_FACTORY": "recovery_fixture:resolve",
                "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "5",
            },
        ):
            self.assertTrue(await asyncio.to_thread(cleanup.cleanup_from_environment))
        self.assertFalse(self.live)

    async def test_reconciliation_skips_active_pool_and_other_run(self):
        pool = self.pool()
        await pool.get("main")
        foreign = cleanup.CleanupJournal(self.directory / "other-run", "gateway-test")
        foreign.created("foreign")
        foreign.detach()
        result = await cleanup.reconcile(self.directory, self.provider, 0.2)
        self.assertFalse(result["ok"])
        self.assertEqual(result["results"][0]["state"], "active")
        self.assertFalse(any(method == "DELETE" for method, _ in self.calls))
        await pool.aclose()

    async def test_concurrent_release_is_idempotent(self):
        pool = self.pool()
        await pool.get("main")
        await asyncio.gather(pool.release_all(), pool.release_all())
        self.assertEqual(sum(method == "DELETE" for method, _ in self.calls), 2)
        await asyncio.gather(pool.aclose(), pool.aclose())
        self.assertTrue(pool.cleanup["ok"])

    async def test_repeated_cancel_still_hands_off_and_preserves_cancel(self):
        self.fail_delete = True
        pool = self.pool()
        await pool.get("main")
        task = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(pool._lifecycle._journal.lock.closed)
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)

    async def test_invalid_journal_does_not_block_other_ids(self):
        (self.directory / ("f" * 32 + ".json")).write_text('{"partial":')
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("owned")
        journal.detach()
        self.live.add("owned")
        result = await cleanup.reconcile(self.directory, self.provider, 0.3)
        self.assertFalse(result["ok"])
        self.assertFalse(self.live)
        self.assertEqual(sum(row["ok"] for row in result["results"]), 1)

    async def test_process_crash_unlocks_durable_journal(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
from cleanup import CleanupJournal
j = CleanupJournal(sys.argv[2], 'gateway-test')
j.created('crash-owned')
print('ready', flush=True)
sys.stdin.read()
"""
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(SOURCE),
            str(self.directory),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            self.assertEqual(await asyncio.wait_for(child.stdout.readline(), 3), b"ready\n")
            self.live.add("crash-owned")
            result = await cleanup.reconcile(self.directory, self.provider, 0.3)
            self.assertFalse(result["ok"])
            self.assertIn("crash-owned", self.live)
            child.kill()
            await child.wait()
            result = await cleanup.reconcile(self.directory, self.provider, 0.3)
            self.assertTrue(result["ok"])
            self.assertFalse(self.live)
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()

    async def test_cancel_during_create_limit_does_not_leave_unknown_allocation(self):
        pool = self.pool()
        entered = asyncio.Event()

        class Limiter:
            async def acquire(self):
                entered.set()
                await asyncio.Event().wait()

        pool._create_limiter = Limiter()
        task = asyncio.create_task(pool.get("main"))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await pool.aclose()
        self.assertFalse(self.live)
        self.assertEqual(self.calls, [])
        self.assertEqual(pool._lifecycle._journal.data["ambiguous_creates"], 0)
        self.assertFalse(pool._lifecycle._journal.data["creating"])
        self.assertTrue((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])

    async def test_stop_during_create_limit_prevents_post_without_unknown_allocation(self):
        pool = self.pool()
        entered, resume = asyncio.Event(), asyncio.Event()

        class Limiter:
            async def acquire(self):
                entered.set()
                await resume.wait()

        pool._create_limiter = Limiter()
        task = asyncio.create_task(pool.get("main"))
        await asyncio.wait_for(entered.wait(), 2)
        cleanup.run_control.request_stop(self.directory, "test")
        resume.set()
        with self.assertRaises(cleanup.run_control.RunStoppingError):
            await task
        await pool.aclose()
        self.assertEqual(self.calls, [])
        self.assertEqual(pool._lifecycle._journal.data["ambiguous_creates"], 0)
        self.assertTrue((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])

    async def test_invalid_create_parameters_do_not_open_intent(self):
        pool = self.pool()
        with patch.dict(os.environ, {"SANDBOX_PROVIDER_USER": ""}):
            with self.assertRaises(ValueError) as error:
                await pool.get("main")
        self.assertEqual(error.exception.allocation_state, "not_allocated")
        await pool.aclose()
        self.assertEqual(self.calls, [])
        self.assertEqual(pool._lifecycle._journal.data["ambiguous_creates"], 0)

    async def test_allocation_callback_failure_retains_id_for_final_delete(self):
        pool = self.pool()
        original = pool._lifecycle._provider.create
        pool._lifecycle._provider.supports_allocation_callback = True

        async def create(*, on_allocated, **kwargs):
            client, env_id = await original(**kwargs)
            await on_allocated(env_id)
            raise AssertionError("readiness must not run after callback failure")

        pool._lifecycle._provider.create = create
        with patch.object(cleanup.CleanupJournal, "allocated", side_effect=OSError("disk unavailable")):
            with self.assertRaises(CleanupError):
                await pool.get("main")
        self.assertEqual(len(self.live), 1)
        await pool.aclose()
        self.assertFalse(self.live)
        self.assertEqual(pool._lifecycle._journal.data["ambiguous_creates"], 0)
        self.assertTrue((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])

    async def test_crash_during_provider_readiness_recovers_callback_id(self):
        code = """
import asyncio, sys
sys.path.insert(0, sys.argv[1])
from sandbox_cleanup_cpu import Provider, sandbox, schemas
class PendingProvider(Provider):
    supports_allocation_callback = True
    async def create(self, *, on_allocated, **kwargs):
        client, env_id = await super().create(**kwargs)
        await on_allocated(env_id)
        print(env_id, flush=True)
        await asyncio.Event().wait()
async def main():
    pool = sandbox.SandboxPool(provider=PendingProvider(int(sys.argv[3])), close_provider=True,
        specs={'main': schemas.SandboxSpec(image='main', user='test')}, creates_per_sec=None,
        cleanup_dir=sys.argv[2], cleanup_provider_key='gateway-test')
    await pool.get('main')
asyncio.run(main())
"""
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(Path(__file__).parent),
            str(self.directory),
            str(self.port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            env_id = (await asyncio.wait_for(child.stdout.readline(), 5)).decode().strip()
            self.assertIn(env_id, self.live)
            path = next(p for p in self.directory.glob("*.json") if len(p.stem) == 32)
            data = json.loads(path.read_text())
            self.assertIn(env_id, data["resources"])
            self.assertFalse(data["creating"])
            self.assertFalse((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])
            self.assertIn(env_id, self.live)
            child.kill()
            await child.wait()
            self.assertTrue((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])
            self.assertFalse(self.live)
        finally:
            if child.returncode is None:
                child.kill()
            await child.wait()

    async def test_exit_handoff_lock_wait_obeys_total_budget_and_retains_journal(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("retained")
        journal.detach()
        self.live.add("retained")
        env = self._recovery_environment()
        env.update(
            XTUNER_SANDBOX_CLEANUP_FACTORY="missing_test_factory:resolve", XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT="0.6"
        )
        with cleanup.recycler.queue_gate(self.directory), patch.dict(os.environ, env):
            started = time.monotonic()
            confirmed = await asyncio.to_thread(cleanup.cleanup_from_environment, finalize=True)
            elapsed = time.monotonic() - started
        self.assertFalse(confirmed)
        self.assertLess(elapsed, 1.6)
        self.assertFalse(json.loads(journal.path.read_text())["resources"]["retained"]["confirmed"])
        self.assertTrue((await cleanup.reconcile(self.directory, self.provider, 0.2))["ok"])
        self.assertFalse(self.live)

    async def test_handoff_bounds_stalled_fsync_and_can_resume(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("retained")
        journal.detach()
        self.live.add("retained")
        site = self.directory / "sitecustomize.py"
        marker = self.directory / "fsync-entered"
        site.write_text(
            "import os, time\nfrom pathlib import Path\n"
            "def blocked(fd):\n"
            f"    Path({str(marker)!r}).touch()\n"
            "    time.sleep(60)\n"
            "os.fsync = blocked\n"
        )
        env = self._recovery_environment()
        with patch.dict(os.environ, env):
            started = time.monotonic()
            confirmed = await asyncio.to_thread(
                cleanup.recycler.handoff_recovery,
                self.directory,
                "recovery_fixture:resolve",
                journal.path,
                timeout=0.7,
            )
        self.assertTrue(marker.exists())
        self.assertFalse(confirmed)
        self.assertLess(time.monotonic() - started, 1.7)
        self.assertTrue(journal.path.exists())
        site.unlink()
        with patch.dict(os.environ, env):
            self.assertTrue(
                await asyncio.to_thread(
                    cleanup.recycler.handoff_recovery,
                    self.directory,
                    "recovery_fixture:resolve",
                    journal.path,
                    timeout=3,
                )
            )
        await self._wait_deleted("retained")
        await self._wait_recycler_idle()

    async def test_handoff_launch_failure_preserves_queued_request(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("retained")
        journal.detach()
        self.live.add("retained")
        site = self.directory / "sitecustomize.py"
        site.write_text(
            "import subprocess\noriginal = subprocess.Popen\n"
            "def launch(args, *a, **kw):\n"
            "    if '--lease-fd' in args: raise OSError('injected launch failure')\n"
            "    return original(args, *a, **kw)\n"
            "subprocess.Popen = launch\n"
        )
        env = self._recovery_environment()
        with patch.dict(os.environ, env):
            self.assertFalse(
                await asyncio.to_thread(
                    cleanup.recycler.handoff_recovery,
                    self.directory,
                    "recovery_fixture:resolve",
                    journal.path,
                    timeout=3,
                )
            )
        self.assertEqual(len(cleanup.recycler.requests(self.directory)), 1)
        self.assertTrue(journal.path.exists())
        site.unlink()
        with patch.dict(os.environ, env):
            self.assertTrue(
                await asyncio.to_thread(
                    cleanup.recycler.handoff_recovery,
                    self.directory,
                    "recovery_fixture:resolve",
                    journal.path,
                    timeout=3,
                )
            )
        await self._wait_deleted("retained")
        await self._wait_recycler_idle()
        self.assertEqual(cleanup.recycler.requests(self.directory), [])

    async def test_persistent_concurrent_create_keeps_unknown_intent_after_other_success(self):
        pool = self.pool(recovery=False)
        entered, fail = asyncio.Event(), asyncio.Event()
        original = pool._lifecycle._provider.create

        async def create(*, image_tag, **kwargs):
            if image_tag == "dep":
                entered.set()
                await fail.wait()
                raise OSError("POST result unknown")
            return await original(image_tag=image_tag, **kwargs)

        pool._lifecycle._provider.create = create
        unknown = asyncio.create_task(pool.get("main"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await pool.get("verifier")
            journal = pool._lifecycle._journal
            self.assertTrue(json.loads(journal.path.read_text())["creating"])
            self.assertEqual(journal.data["ambiguous_creates"], 0)
        finally:
            fail.set()
            with self.assertRaisesRegex(OSError, "POST result unknown"):
                await asyncio.wait_for(unknown, 1)
            with self.assertRaises(CleanupError):
                await pool.aclose()
        self.assertFalse(self.live)
        self.assertFalse(json.loads(journal.path.read_text())["creating"])
        result = await cleanup.reconcile(self.directory, self.provider, 0.3)
        self.assertFalse(result["ok"])
        self.assertEqual(result["results"][0]["ambiguous_creates"], 1)
        self.assertEqual(json.loads(journal.path.read_text())["state"], "pending")

    async def test_persistent_early_allocation_resolves_only_its_own_intent(self):
        pool = self.pool(recovery=False)
        provider = pool._lifecycle._provider
        provider.supports_allocation_callback = True
        entered, report_allocation = asyncio.Event(), asyncio.Event()
        unknown_entered, fail_unknown = asyncio.Event(), asyncio.Event()
        original = provider.create

        async def create(*, image_tag, on_allocated, **kwargs):
            if image_tag == "verifier":
                unknown_entered.set()
                await fail_unknown.wait()
                raise OSError("other POST result unknown")
            client, env_id = await original(image_tag=image_tag, **kwargs)
            if image_tag == "dep":
                entered.set()
                await report_allocation.wait()
                await on_allocated(env_id)
                self.assertTrue(json.loads(pool._lifecycle._journal.path.read_text())["creating"])
            return client, env_id

        provider.create = create
        known = asyncio.create_task(pool.get("main"))
        unknown = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            unknown = asyncio.create_task(pool.get("verifier"))
            await asyncio.wait_for(unknown_entered.wait(), 1)
            report_allocation.set()
            await asyncio.wait_for(known, 1)
            journal = pool._lifecycle._journal
            self.assertTrue(json.loads(journal.path.read_text())["creating"])
            self.assertEqual(journal.data["ambiguous_creates"], 0)
        finally:
            report_allocation.set()
            fail_unknown.set()
            await asyncio.gather(known, return_exceptions=True)
            if unknown is not None:
                with self.assertRaisesRegex(OSError, "other POST result unknown"):
                    await asyncio.wait_for(unknown, 1)
            with self.assertRaises(CleanupError):
                await pool.aclose()
        result = await cleanup.reconcile(self.directory, self.provider, 0.3)
        self.assertFalse(result["ok"])
        self.assertEqual(result["results"][0]["ambiguous_creates"], 1)
        self.assertFalse(self.live)

    async def test_journal_legacy_mutations_cannot_clear_named_create_intents(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.begin_create(create_id="first")
        journal.begin_create(create_id="second")
        journal.allocated("known-first", create_id="first")
        self.assertTrue(json.loads(journal.path.read_text())["creating"])
        journal.end_create(create_id="first", ambiguous=True)
        self.assertEqual(journal.data["ambiguous_creates"], 0)
        journal.begin_create()
        journal.end_create(ambiguous=True)
        self.assertTrue(json.loads(journal.path.read_text())["creating"])
        self.assertEqual(journal.data["ambiguous_creates"], 1)
        journal.end_create(ambiguous=True)
        self.assertEqual(journal.data["ambiguous_creates"], 1)
        journal.allocated("legacy-fixture")
        self.assertTrue(json.loads(journal.path.read_text())["creating"])
        journal.end_create(create_id="second", ambiguous=True)
        self.assertFalse(json.loads(journal.path.read_text())["creating"])
        self.assertEqual(journal.data["ambiguous_creates"], 2)
        journal.end_create(create_id="second", ambiguous=True)
        self.assertEqual(journal.data["ambiguous_creates"], 2)
        journal.detach()
        self.assertEqual(json.loads(journal.path.read_text())["state"], "pending")

    async def test_unknown_create_never_reports_cleanup_success(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.begin_create()
        journal.detach()
        result = await cleanup.reconcile(self.directory, self.provider, 0.3)
        self.assertFalse(result["ok"])
        self.assertEqual(result["results"][0]["ambiguous_creates"], 1)
        self.assertEqual(self.calls, [])

    def test_driver_shutdown_defers_repeated_signal(self):
        # Exercise the actual shutdown closure without importing/starting Ray.
        source = ROOT / "xtuner/v1/rl/utils/ray_utils.py"
        node = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "register_cleanup"
        )
        handlers, exits, events = {}, [], []

        def close_ray():
            events.append("ray-close-start")
            handlers[15](15, None)
            events.append("ray-close-finish")

        namespace = {
            "atexit": types.SimpleNamespace(register=exits.append),
            "signal": types.SimpleNamespace(
                SIGINT=2, SIGTERM=15, signal=lambda key, value: handlers.update({key: value})
            ),
            "logger": logging.getLogger("cleanup-test"),
            "close_ray": close_ray,
            "os": types.SimpleNamespace(environ={}, getpid=lambda: 1),
            "_cleanup_registration": None,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        namespace["register_cleanup"]()
        # Model ray.init() replacing the handler after the CLI's early hook.
        handlers[15] = lambda *args: (_ for _ in ()).throw(SystemExit(15))
        namespace["register_cleanup"]()
        self.assertEqual(len(exits), 1, "Reinstallation must not duplicate cleanup")
        with self.assertRaises(SystemExit) as captured:
            handlers[15](15, None)
        self.assertEqual(captured.exception.code, 143)
        exits[0]()
        self.assertEqual(events, ["ray-close-start", "ray-close-finish"])

        # Even callers omitting reinstallation must protect the atexit path
        # from a second signal while Ray shutdown is already in progress.
        handlers.clear()
        exits.clear()
        events.clear()
        namespace["_cleanup_registration"] = None
        namespace["register_cleanup"]()
        handlers[15] = lambda *args: (_ for _ in ()).throw(SystemExit(15))
        exits[0]()
        self.assertEqual(events, ["ray-close-start", "ray-close-finish"])

        # A configured signal path closes admission before Ray, then records
        # interruption in final recovery. The atexit callback remains idempotent.
        handlers.clear()
        exits.clear()
        events.clear()
        namespace["_cleanup_registration"] = None
        namespace["os"].environ = {"XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory)}

        def stop():
            events.append("stop")
            return True

        def recover(**kwargs):
            self.assertEqual(kwargs, {"finalize": True, "training_outcome": "interrupted"})
            events.append("recover")
            return True

        with (
            patch.object(cleanup, "request_stop_from_environment", stop),
            patch.object(cleanup, "cleanup_from_environment", recover),
        ):
            namespace["register_cleanup"]()
            with self.assertRaises(SystemExit) as captured:
                handlers[15](15, None)
            self.assertEqual(captured.exception.code, 143)
            exits[0]()
        self.assertEqual(events, ["stop", "ray-close-start", "ray-close-finish", "recover"])

    async def test_rollback_must_confirm_release_before_retry(self):
        for failure in ("delete", "confirmation"):
            with self.subTest(failure=failure):
                pool = self.pool()
                pool._max_attempts = 2
                checks = 0
                before = self.next_id

                async def first_unhealthy(client):
                    nonlocal checks
                    checks += 1
                    return checks > 1

                pool._wait_healthy = first_unhealthy
                self.fail_delete = failure == "delete"
                self.accept_only = failure == "confirmation"
                with self.assertRaisesRegex(RuntimeError, "unhealthy"):
                    await pool.get("main")
                self.assertEqual(self.next_id - before, 1)
                self.assertEqual(pool._lifecycle._pending_delete_ids, {f"dep-{before + 1}"})
                self.assertIn(f"dep-{before + 1}", self.live)
                self.fail_delete = False
                self.accept_only = False
                await pool.aclose()
                self.assertFalse(self.live)

    async def test_unclaimed_create_failure_never_retries_transaction(self):
        pool = self.pool()
        pool._max_attempts = 2
        persist = pool._lifecycle._journal
        self.assertIsNone(persist)
        await pool._lifecycle._ensure_journal()
        journal = pool._lifecycle._journal
        original = journal.persist
        failed_create_id = None

        async def fail_end_create(action, *args, **kwargs):
            nonlocal failed_create_id
            if action == journal.end_create:
                failed_create_id = kwargs["create_id"]
                raise OSError("end-create write failed")
            return await original(action, *args, **kwargs)

        with patch.object(journal, "persist", side_effect=fail_end_create):
            with self.assertRaisesRegex(OSError, "end-create write failed"):
                await pool.get("main")
        self.assertEqual([key for method, key in self.calls if method == "POST"], ["dep"])
        self.assertEqual(set(pool._lifecycle._unclaimed_creates), {"dep-1"})
        self.assertFalse(pool._lifecycle._active_acquisitions)
        # Allocation already resolved this request's intent. Replay the failed
        # end-create write with its exact token before closing the known resource.
        await journal.persist(journal.end_create, ambiguous=False, create_id=failed_create_id)
        await pool.aclose()
        self.assertFalse(self.live)

    async def test_failed_rollback_client_close_prevents_retry(self):
        pool = self.pool()
        pool._max_attempts = 2
        failed_client = Client()
        close_failed = True

        async def close():
            if close_failed:
                raise RuntimeError("socket close failed")
            failed_client.closed = True

        failed_client.aclose = close
        original = pool._lifecycle._provider.create

        async def create(**kwargs):
            _, env_id = await original(**kwargs)
            return failed_client, env_id

        async def unhealthy(_):
            return False

        pool._lifecycle._provider.create = create
        pool._wait_healthy = unhealthy
        with self.assertRaisesRegex(RuntimeError, "unhealthy"):
            await pool.get("main")
        self.assertEqual([key for method, key in self.calls if method == "POST"], ["dep"])
        self.assertIn(failed_client, pool._lifecycle._cleanup_clients)
        close_failed = False
        await pool.aclose()
        self.assertTrue(failed_client.closed)
        self.assertFalse(self.live)

    async def test_named_groups_remain_concurrent(self):
        pool = self.pool(persistent=False)
        entered, verifier_entered = asyncio.Event(), asyncio.Event()
        original = pool._lifecycle._provider.create

        async def create(*, image_tag, **kwargs):
            if image_tag == "dep":
                entered.set()
                await verifier_entered.wait()
            elif image_tag == "verifier":
                verifier_entered.set()
            return await original(image_tag=image_tag, **kwargs)

        pool._lifecycle._provider.create = create
        first = asyncio.create_task(pool.get("main"))
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(pool.get("verifier"))
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        self.assertIsNotNone(pool.peek("main"))
        self.assertIsNotNone(pool.peek("verifier"))
        self.assertFalse(pool._lifecycle._active_acquisitions)
        await pool.aclose()
        self.assertFalse(self.live)

    async def test_failed_create_does_not_claim_concurrent_successful_group(self):
        for callback in (False, True):
            for committed in (False, True):
                for failed_allocated in (False, True):
                    with self.subTest(callback=callback, committed=committed, failed_allocated=failed_allocated):
                        pool = self.pool(persistent=False)
                        pool._max_attempts = 2
                        provider = pool._lifecycle._provider
                        provider.supports_allocation_callback = callback
                        original = provider.create
                        entered, fail = asyncio.Event(), asyncio.Event()
                        allocated, resume = asyncio.Event(), asyncio.Event()
                        failure_attempts = 0
                        successful_ids = []

                        async def create(*, image_tag, on_allocated=None, **kwargs):
                            nonlocal failure_attempts
                            if image_tag == "dep":
                                failure_attempts += 1
                                if failed_allocated:
                                    _, env_id = await original(image_tag=image_tag, **kwargs)
                                    if on_allocated is not None:
                                        await on_allocated(env_id)
                                entered.set()
                                await fail.wait()
                                raise OSError("one concurrent create failed")
                            client, env_id = await original(image_tag=image_tag, **kwargs)
                            successful_ids.append(env_id)
                            if on_allocated is not None:
                                await on_allocated(env_id)
                            allocated.set()
                            if not committed:
                                await resume.wait()
                            return client, env_id

                        provider.create = create
                        failing = asyncio.create_task(pool.get("main"))
                        await asyncio.wait_for(entered.wait(), 1)
                        successful = asyncio.create_task(pool.get("verifier"))
                        try:
                            await asyncio.wait_for(allocated.wait(), 1)
                            if committed:
                                await asyncio.wait_for(successful, 1)
                            fail.set()
                            with self.assertRaisesRegex(OSError, "one concurrent create failed") as error:
                                await asyncio.wait_for(failing, 1)
                            expected = "allocated" if failed_allocated and (callback or committed) else "unknown"
                            self.assertEqual(error.exception.allocation_state, expected)
                            self.assertEqual(failure_attempts, 1)
                            self.assertNotIn(successful_ids[0], pool._lifecycle._pending_delete_ids)
                            resume.set()
                            client = await asyncio.wait_for(successful, 1)
                            self.assertIs(pool.peek("verifier"), client)
                            self.assertFalse(client.closed)
                            self.assertIn(successful_ids[0], self.live)
                            self.assertNotIn(("DELETE", successful_ids[0]), self.calls)
                        finally:
                            fail.set()
                            resume.set()
                            await asyncio.gather(failing, successful, return_exceptions=True)
                            await pool.aclose()
                        self.assertFalse(self.live)
                        self.assertFalse(pool._lifecycle._active_acquisitions)

    async def test_final_close_drains_provider_await_before_sealing(self):
        pool = self.pool()
        entered = asyncio.Event()
        original = pool._lifecycle._provider.create

        async def create(**kwargs):
            await original(**kwargs)
            entered.set()
            await asyncio.Event().wait()

        pool._lifecycle._provider.create = create
        acquiring = asyncio.create_task(pool.get("main"))
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(pool.aclose(), 2)
        with self.assertRaises(asyncio.CancelledError):
            await acquiring
        self.assertFalse(pool._lifecycle._active_acquisitions)
        self.assertFalse(self.live)
        self.assertTrue(pool._lifecycle._journal.lock.closed)
        self.assertEqual(pool._lifecycle._journal.data["state"], "complete")
        self.assertTrue(pool._lifecycle._provider.closed)

    async def test_release_drains_health_await_before_reopening_admission(self):
        for final in (False, True):
            with self.subTest(final=final):
                pool = self.pool()
                entered = asyncio.Event()
                clients = []
                original_health = pool._wait_healthy

                async def health(client):
                    clients.append(client)
                    entered.set()
                    await asyncio.Event().wait()

                pool._wait_healthy = health
                acquiring = asyncio.create_task(pool.get("main"))
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(pool.aclose() if final else pool.release_all(), 2)
                with self.assertRaises(asyncio.CancelledError):
                    await acquiring
                self.assertFalse(pool._lifecycle._active_acquisitions)
                self.assertTrue(all(client.closed for client in clients))
                self.assertFalse(self.live)
                if not final:
                    self.assertFalse(pool._lifecycle._provider.closed)
                    pool._wait_healthy = original_health
                    await pool.get("verifier")
                    await pool.aclose()
                self.assertTrue(pool._lifecycle._journal.lock.closed)

    async def test_close_waits_for_acquisition_not_its_callers_later_work(self):
        pool = self.pool()
        entered, cancelled, unrelated_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def health(_):
            entered.set()
            await asyncio.Event().wait()

        async def caller():
            try:
                await pool.get("main")
            except asyncio.CancelledError:
                cancelled.set()
            await unrelated_done.wait()

        pool._wait_healthy = health
        acquiring = asyncio.create_task(caller())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(pool.aclose(), 2)
            self.assertTrue(cancelled.is_set())
            self.assertFalse(acquiring.done())
            self.assertFalse(pool._lifecycle._active_acquisitions)
            self.assertFalse(self.live)
            self.assertTrue(pool._lifecycle._journal.lock.closed)
        finally:
            unrelated_done.set()
            await asyncio.wait_for(acquiring, 2)

    async def test_close_timeout_retains_active_create_ownership_for_retry(self):
        pool = self.pool()
        entered, resume = asyncio.Event(), asyncio.Event()
        original = pool._lifecycle._provider.create
        clients = []

        async def create(**kwargs):
            client, env_id = await original(**kwargs)
            clients.append(client)
            entered.set()
            while not resume.is_set():
                try:
                    await resume.wait()
                except asyncio.CancelledError:
                    pass
            return client, env_id

        pool._lifecycle._provider.create = create
        acquiring = asyncio.create_task(pool.get("main"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            with self.assertRaises(CleanupError) as error:
                await asyncio.wait_for(pool.aclose(), 2)
            self.assertEqual(error.exception.kind, "cleanup_acquisition_pending")
            self.assertFalse(pool._lifecycle._closed)
            self.assertFalse(pool._lifecycle._provider.closed)
            self.assertFalse(pool._lifecycle._journal.lock.closed)
            self.assertTrue(pool._lifecycle._journal.data["creating"])
            self.assertNotEqual(pool._lifecycle._journal.data["state"], "complete")
            self.assertTrue(pool._lifecycle._active_acquisitions)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await pool.get("verifier")
        finally:
            resume.set()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await asyncio.wait_for(acquiring, 2)
            await pool.aclose()
        self.assertFalse(pool._lifecycle._active_acquisitions)
        self.assertFalse(self.live)
        self.assertTrue(all(client.closed for client in clients))
        self.assertTrue(pool._lifecycle._journal.lock.closed)
        self.assertEqual(pool._lifecycle._journal.data["state"], "complete")
        self.assertEqual([key for method, key in self.calls if method == "POST"], ["dep"])

    async def test_confirmed_rollback_still_allows_retry(self):
        pool = self.pool()
        pool._max_attempts = 2
        checks = 0

        async def first_unhealthy(client):
            nonlocal checks
            checks += 1
            return checks > 1

        pool._wait_healthy = first_unhealthy
        await pool.get("main")
        posts = [key for method, key in self.calls if method == "POST"]
        self.assertEqual(posts, ["dep", "dep", "main"])
        self.assertNotIn("dep-1", self.live)
        self.assertFalse(pool._lifecycle._pending_delete_ids)
        await pool.aclose()
        self.assertFalse(self.live)

    async def test_later_phase_rollback_has_its_own_delete_budget(self):
        pool = self.pool()
        pool._max_attempts = 2
        lifecycle = importlib.import_module(PACKAGE + ".lifecycle")
        offset = 0.0
        clock = types.SimpleNamespace(monotonic=lambda: time.monotonic() + offset)
        delete = pool._lifecycle._provider.delete
        attempts = []
        checks = 0

        async def fail_first_delete(env_id):
            attempts.append(env_id)
            if env_id.startswith("verifier-") and attempts.count(env_id) == 1:
                raise HTTPError(503)
            await delete(env_id)

        async def first_unhealthy(_):
            nonlocal checks
            checks += 1
            return checks > 1

        with patch.object(lifecycle, "time", clock):
            await pool.get("main")
            await pool.release_all()
            # Move beyond the previous phase's deadline without a timing race.
            offset += pool._lifecycle._cleanup_timeout_sec * 2
            pool._wait_healthy = first_unhealthy
            pool._lifecycle._provider.delete = fail_first_delete
            client = await pool.get("verifier")
            self.assertIs(pool.peek("verifier"), client)
            self.assertEqual([key for method, key in self.calls if method == "POST"].count("verifier"), 2)
            self.assertEqual(attempts, ["verifier-3", "verifier-3"])
            self.assertFalse(pool._lifecycle._pending_delete_ids)
            self.assertNotIn("verifier-3", self.live)
            await pool.aclose()
        self.assertTrue(client.closed)
        self.assertFalse(self.live)
        self.assertTrue(pool._lifecycle._provider.closed)

    async def test_phase_drain_rollback_and_groups_share_the_original_delete_budget(self):
        pool = self.pool()
        lifecycle = importlib.import_module(PACKAGE + ".lifecycle")
        offset = 0.0
        clock = types.SimpleNamespace(monotonic=lambda: time.monotonic() + offset)
        entered = asyncio.Event()
        clients = []

        async def health_until_cancel(client):
            nonlocal offset
            clients.append(client)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Drain used the first-pass slice. Neither its rollback nor
                # another group may start a fresh first-pass budget now.
                offset += pool._lifecycle._cleanup_timeout_sec * 0.45
                raise

        with patch.object(lifecycle, "time", clock):
            await pool.get("main")
            pool._wait_healthy = health_until_cancel
            acquiring = asyncio.create_task(pool.get("verifier"))
            await asyncio.wait_for(entered.wait(), 1)
            await pool.release_all()
            with self.assertRaises(asyncio.CancelledError):
                await acquiring
            for env_id in ("verifier-3", "main-2", "dep-1"):
                rows = [row for row in pool.cleanup["delete_attempts"] if row["env_id"] == env_id]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["error_kind"], "delete_budget_exhausted")
                self.assertTrue(rows[1]["ok"])
            self.assertEqual(
                [key for method, key in self.calls if method == "DELETE"], ["verifier-3", "main-2", "dep-1"]
            )
            self.assertFalse(pool._lifecycle._active_acquisitions)
            self.assertTrue(all(client.closed for client in clients))
            self.assertFalse(self.live)
            self.assertFalse(pool._lifecycle._provider.closed)
            await pool.aclose()

    async def test_corrupt_requests_do_not_block_or_get_acknowledged(self):
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        for i, content in enumerate(("{broken", "null", "[]", '{"kind":"journal","journal":"../bad"}')):
            (queue / f"bad-{i}.json").write_text(content)
        # Test both journal-only recovery and an exit sweep. Neither may
        # acknowledge corrupt requests just because valid resources are gone.
        for kind in ("journal", "exit"):
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            journal.allocated(kind)
            journal.detach()
            self.live.add(kind)
            valid = queue / "valid.json"
            valid.write_text(json.dumps({"kind": kind, "journal": journal.path.name if kind == "journal" else None}))
            result = await cleanup.recover_queued(self.directory, self.provider, 0.4, 2, 8)
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["invalid_requests"]), 4)
            self.assertEqual(result["acknowledged"], 1)
            self.assertFalse(valid.exists())
            self.assertFalse(self.live)
            self.assertEqual(len(list(queue.glob("*.json"))), 4)
        result = await cleanup.recover_queued(self.directory, self.provider, 0.4, 2, 8)
        self.assertFalse(result["ok"])
        self.assertEqual(result["acknowledged"], 0)

    async def test_witness_retries_io_errors_without_false_driver_death(self):
        environment = self._recovery_environment()
        # Test the actual supervisor process, after READY. The second probe
        # checks that the first error never initiated a premature exit sweep.
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import supervisor
calls = 0
def probe(directory):
    global calls
    calls += 1
    if calls <= 2:
        assert not (directory / 'stop.requested.json').exists()
        raise OSError('transient lease I/O')
    if calls == 3:
        assert not (directory / 'stop.requested.json').exists()
        return True
    return False
supervisor.driver_alive = probe
raise SystemExit(supervisor.main())
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(SOURCE),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 10)
        self.assertEqual(proc.returncode, 0, stderr.decode())
        self.assertTrue(stdout.startswith(b"READY\n"))
        self.assertTrue((self.directory / "stop.requested.json").exists())
        self.assertEqual(stderr.count(b"Driver lease probe failed"), 2)
        self.assertFalse(cleanup.recycler.requests(self.directory))

    async def test_forked_signal_and_atexit_callbacks_do_not_stop_parent(self):
        code = """
import ast, atexit, logging, os, signal, sys, time
from types import FrameType
from pathlib import Path
import sandbox_cleanup_cpu as runtime
source = runtime.ROOT / 'xtuner/v1/rl/utils/ray_utils.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'register_cleanup')
callbacks = []
import types
scope = dict(os=os, signal=signal, FrameType=FrameType, atexit=types.SimpleNamespace(register=callbacks.append),
             logger=logging.getLogger('test'), close_ray=lambda: None, _cleanup_registration=None)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
lease = runtime.cleanup.run_control.claim_driver(Path(sys.argv[1]))
scope['register_cleanup']()
for mode in ('signal', 'atexit'):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        if mode == 'atexit':
            callbacks[0]()
            os._exit(0)
        os.write(write_fd, b'R')
        os.close(write_fd)
        while True: time.sleep(.1)
    os.close(write_fd)
    if mode == 'signal':
        assert os.read(read_fd, 1) == b'R'
        os.kill(pid, signal.SIGTERM)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == (-15 if mode == 'signal' else 0)
    assert runtime.cleanup.run_control.driver_alive(Path(sys.argv[1]))
    assert not Path(sys.argv[1], 'stop.requested.json').exists()
    assert not Path(sys.argv[1], 'run-cleanup.json').exists()
print('fork callbacks isolated', flush=True)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(self.directory),
            env=self._recovery_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 10)
        self.assertEqual(proc.returncode, 0, stderr.decode())
        self.assertIn(b"fork callbacks isolated", stdout)
        self.assertFalse(self.calls)

    async def test_ray_shutdown_guard_disarms_after_normal_drain(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.allocated("normal-drain")
        journal.detach()
        self.live.add("normal-drain")
        environment = self._recovery_environment()
        environment["XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT"] = ".3"
        with patch.dict(os.environ, environment):
            with cleanup.guard_ray_shutdown(training_outcome="interrupted"):
                pass
        await asyncio.to_thread(cleanup._SHUTDOWN_GUARD.wait, timeout=10)
        self.assertIn("normal-drain", self.live)
        self.assertFalse(self.calls)
        self.assertFalse(cleanup.recycler.requests(self.directory))
        self.assertFalse((self.directory / "stop.requested.json").exists())
        self.assertEqual(cleanup._SHUTDOWN_GUARD.poll(), 0)
        await cleanup.reconcile(self.directory, self.provider, 0.4)

    async def test_blocked_ray_shutdown_recovers_with_driver_alive_and_protects_live_pool(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.allocated("abandoned-before-ray-drain")
        journal.detach()
        self.live.add("abandoned-before-ray-drain")
        active = cleanup.CleanupJournal(self.directory, "gateway-test")
        active.allocated("still-owned")
        self.live.add("still-owned")
        environment = self._recovery_environment()
        environment["XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT"] = ".2"
        code = """
import ast, atexit, logging, os, signal, sys, time
from types import FrameType
from pathlib import Path
import sandbox_cleanup_cpu as runtime
source = runtime.ROOT / 'xtuner/v1/rl/utils/ray_utils.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'register_cleanup')
def close_ray():
    Path(sys.argv[1], 'ray-entered').touch()
    time.sleep(60)
scope = dict(os=os, signal=signal, FrameType=FrameType, atexit=atexit, logger=logging.getLogger('test'),
             close_ray=close_ray, _cleanup_registration=None)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
lease = runtime.cleanup.run_control.claim_driver(Path(sys.argv[1]))
scope['register_cleanup']()
print('READY', flush=True)
while True: time.sleep(.1)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(self.directory),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            import signal

            self.assertEqual(await asyncio.wait_for(proc.stdout.readline(), 10), b"READY\n")
            proc.send_signal(signal.SIGTERM)
            async with asyncio.timeout(5):
                while not (self.directory / "ray-entered").exists():
                    await asyncio.sleep(0.02)
            proc.send_signal(signal.SIGTERM)
            await self._wait_deleted("abandoned-before-ray-drain")
            self.assertIsNone(proc.returncode, "Blocked driver must still be alive at recovery")
            self.assertTrue(cleanup.run_control.driver_alive(self.directory))
            self.assertIn("still-owned", self.live)
            self.assertNotIn(("DELETE", "still-owned"), self.calls)
            # The failed exit sweep retains the active ledger in the durable
            # queue; later release allows the recycler to finish it.
            async with asyncio.timeout(5):
                while not cleanup.recycler.requests(self.directory):
                    await asyncio.sleep(0.02)
            active.detach()
            await self._wait_deleted("still-owned")
            await self._wait_recycler_idle()
            result = json.loads((self.directory / "run-cleanup.json").read_text())
            if not result["cleanup_confirmed"]:
                details = {
                    "result": result,
                    "calls": self.calls,
                    "live": sorted(self.live),
                    "files": {},
                    "file_mtimes_ns": {},
                }
                for path in sorted(self.directory.rglob("*")):
                    if path.is_file() and (path.suffix in {".json", ".tmp"} or path.name == "recovery.log"):
                        try:
                            details["files"][str(path.relative_to(self.directory))] = path.read_text()
                            details["file_mtimes_ns"][str(path.relative_to(self.directory))] = path.stat().st_mtime_ns
                        except OSError as error:
                            details["files"][str(path.relative_to(self.directory))] = type(error).__name__
                diagnostic = json.dumps(details, indent=2)
                if failure_directory := os.environ.get("XTUNER_TEST_CLEANUP_FAILURE_DIR"):
                    destination = Path(failure_directory)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / f"blocked-ray-shutdown-{time.time_ns()}.json").write_text(diagnostic)
                self.fail(diagnostic)
            self.assertEqual(result["training_outcome"], "interrupted")
        finally:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.communicate(), 5)
            active.lock.close()

    async def test_shutdown_guard_recovers_on_pipe_eof_before_deadline(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.allocated("driver-died-in-drain")
        journal.detach()
        self.live.add("driver-died-in-drain")
        environment = self._recovery_environment()
        environment["XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT"] = "60"
        code = """
from sandbox_cleanup_cpu import cleanup
with cleanup.guard_ray_shutdown(training_outcome='interrupted'):
    print('READY', flush=True)
    import time
    time.sleep(60)
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            self.assertEqual(await asyncio.wait_for(proc.stdout.readline(), 10), b"READY\n")
            proc.kill()
            await self._wait_deleted("driver-died-in-drain")
            # The guard inherited stdout. Drain EOF as well as reaping the
            # driver, so its transport cannot outlive this test's event loop.
            await asyncio.wait_for(proc.communicate(), 5)
            self.assertEqual(proc.returncode, -9)
            self.assertFalse(cleanup.recycler.requests(self.directory))
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    async def test_shutdown_guard_launch_failure_runs_recovery_before_ray(self):
        events = []
        environment = self._recovery_environment()
        with (
            patch.dict(os.environ, environment),
            patch.object(cleanup.subprocess, "Popen", side_effect=OSError("cannot spawn guard")),
            patch.object(cleanup, "cleanup_from_environment", side_effect=lambda **kw: events.append("recover")),
        ):
            with cleanup.guard_ray_shutdown(training_outcome="interrupted"):
                events.append("ray")
        self.assertEqual(events, ["recover", "ray"])

    async def test_completed_journal_is_not_rewritten_by_periodic_scan(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("done")
        journal.confirmed("done")
        journal.detach()
        with patch.object(cleanup, "_atomic_json", side_effect=AssertionError("unexpected rewrite")):
            result = await cleanup.reconcile_file(journal.path, self.provider, 0.3)
        self.assertTrue(result["ok"])
        self.assertEqual(self.calls, [])

    async def test_raw_create_timeout_is_not_replayed_without_journal(self):
        pool = self.pool()
        pool._lifecycle._journal_dir = None
        pool._max_attempts = 3
        calls = 0

        async def timeout(**kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("response lost")

        pool._lifecycle._provider.create = timeout
        with self.assertRaises(TimeoutError) as error:
            await pool.get("main")
        self.assertEqual(calls, 1)
        self.assertEqual(error.exception.allocation_state, "unknown")
        await pool.aclose()

    async def test_unclassified_create_failure_does_not_duplicate_allocations(self):
        pool = self.pool(persistent=False)
        pool._max_attempts = 3
        calls = 0

        async def lost_response(**kwargs):
            nonlocal calls
            calls += 1
            raise ConnectionError("connection closed after allocation")

        pool._lifecycle._provider.create = lost_response
        with self.assertRaises(ConnectionError):
            await pool.get("main")
        self.assertEqual(calls, 1)
        await pool.aclose()

    def _recovery_environment(self):
        (self.directory / "recovery_fixture.py").write_text(
            "import os\nfrom sandbox_cleanup_cpu import Provider\n"
            "def resolve(key):\n"
            "    assert key == 'gateway-test'\n"
            "    return Provider(int(os.environ['CLEANUP_TEST_PORT']))\n"
        )
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(self.directory), str(Path(__file__).parent)]),
            "CLEANUP_TEST_PORT": str(self.port),
            "XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory),
            "XTUNER_SANDBOX_CLEANUP_FACTORY": "recovery_fixture:resolve",
            "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "3",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "2",
            "XTUNER_SANDBOX_RECOVERY_BUDGET": "8",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "3",
            "XTUNER_SANDBOX_RECOVERY_BACKOFF": ".1",
            "XTUNER_SANDBOX_RECOVERY_MAX_BACKOFF": ".2",
        }
        return environment

    async def _supervised_driver(self):
        environment = self._recovery_environment()
        self.supervisor_environment = environment
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import cleanup
cleanup.start_cleanup_supervisor()
cleanup.start_cleanup_supervisor()
j = cleanup.CleanupJournal(sys.argv[2], 'gateway-test')
j.created('active-driver')
from pathlib import Path
Path(sys.argv[2], 'driver-witness.pid').write_text(str(cleanup._SUPERVISOR[4].pid))
print('ready', flush=True)
sys.stdin.read()
"""
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(SOURCE),
            str(self.directory),
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.assertEqual(await asyncio.wait_for(driver.stdout.readline(), 15), b"ready\n")
        self.live.add("active-driver")
        return driver

    async def _wait_deleted(self, env_id):
        async with asyncio.timeout(10):
            while env_id in self.live:
                await asyncio.sleep(0.05)

    async def _wait_recycler_idle(self):
        async with asyncio.timeout(25):
            while True:
                with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as lease:
                    try:
                        cleanup.locking.acquire(lease)
                        return
                    except BlockingIOError:
                        pass
                await asyncio.sleep(0.05)

    async def test_exit_sweeps_serialize_and_limit_delete_concurrency(self):
        for index in range(6):
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            env_id = f"orphan-{index}"
            journal.created(env_id)
            journal.detach()
            self.live.add(env_id)
        running, peak = 0, 0
        original_provider = self.provider

        def resolve(key):
            provider = original_provider(key)
            original_delete = provider.delete

            async def delete(env_id):
                nonlocal running, peak
                running += 1
                peak = max(peak, running)
                try:
                    await asyncio.sleep(0.05)
                    await original_delete(env_id)
                finally:
                    running -= 1

            provider.delete = delete
            return provider

        reports = await asyncio.gather(
            cleanup.finalize_run(self.directory, resolve, 2, 5, concurrency=2),
            cleanup.finalize_run(self.directory, resolve, 2, 5, concurrency=2),
        )
        self.assertTrue(all(report["cleanup_confirmed"] for report in reports))
        self.assertEqual(peak, 2)
        self.assertFalse(self.live)
        self.assertEqual(len([call for call in self.calls if call[0] == "DELETE"]), 6)

    async def test_recycler_singleton_only_handles_explicit_failures(self):
        environment = self._recovery_environment()
        journals = []
        for index in range(7):
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            env_id = f"queued-{index}"
            journal.created(env_id)
            journal.detach()
            self.live.add(env_id)
            journals.append(journal)
        # Parent enqueue calls are synchronous; worker lease remains inherited
        # while its HTTP requests wait for this event loop to run again.
        launched = []
        popen = cleanup.recycler.subprocess.Popen

        def launch(*args, **kwargs):
            process = popen(*args, **kwargs)
            launched.append(process)
            return process

        with patch.dict(os.environ, environment), patch.object(cleanup.recycler.subprocess, "Popen", launch):
            for journal in journals[:6]:
                cleanup.recycler.request_recovery(self.directory, "recovery_fixture:resolve", journal.path)
        self.assertEqual(len(launched), 1)
        await self._wait_recycler_idle()
        await asyncio.to_thread(launched[0].wait, 5)
        self.assertEqual(launched[0].returncode, 0)
        self.assertEqual(self.live, {"queued-6"})
        self.assertFalse(cleanup.recycler.requests(self.directory))
        self.assertEqual(len([call for call in self.calls if call[0] == "DELETE"]), 6)
        self.assertFalse((self.directory / "stop.requested.json").exists())

    async def test_recycler_final_attempt_transfers_only_fresh_generations(self):
        recycler = cleanup.recycler
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        old = queue / ("a" * 32 + ".json")
        cleanup._atomic_json(old, {"kind": "journal", "journal": "1" * 32 + ".json"})
        manifest = batches / "first.json"
        cleanup._atomic_json(manifest, {"requests": [old.name], "seen": [old.name]})
        lease = cleanup.locking.open_lock(self.directory / "recovery-worker.lock")
        self.addCleanup(lease.close)
        cleanup.locking.acquire(lease)
        environment = {
            "XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory),
            "XTUNER_SANDBOX_CLEANUP_FACTORY": "fixture:resolve",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "1",
        }

        def final_sweep(command, **kwargs):
            self.assertEqual(command[command.index("--request-batch") + 1], str(manifest))
            recycler.request_recovery(self.directory, "fixture:resolve", self.directory / ("2" * 32 + ".json"))
            return types.SimpleNamespace(returncode=1)

        # The guardian owns another FD for the same lease. The worker must not
        # unlock it when its last attempt leaves a newly enqueued request.
        arguments = [
            "recycler.py",
            "--lease-fd",
            str(os.dup(lease.fileno())),
            "--run-worker",
            "--batch",
            str(manifest),
        ]
        with (
            patch.dict(os.environ, environment),
            patch.object(sys, "argv", arguments),
            patch.object(recycler.subprocess, "run", final_sweep),
            patch.object(recycler.subprocess, "Popen") as launch,
        ):
            self.assertEqual(recycler.main(), 1)
            launch.assert_not_called()
        with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as probe:
            with self.assertRaises(BlockingIOError):
                cleanup.locking.acquire(probe)

        launched = []

        def capture(command, **kwargs):
            batch = Path(command[command.index("--batch") + 1])
            launched.append((batch, json.loads(batch.read_text())))
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["pass_fds"], (lease.fileno(),))
            return types.SimpleNamespace(wait=lambda: 0)

        with patch.object(recycler.subprocess, "Popen", capture):
            recycler.retire_batch(self.directory, "fixture:resolve", lease, manifest)
            second_manifest, second = launched[-1]
            self.assertEqual(len(second["requests"]), 1)
            self.assertNotIn(old.name, second["requests"])
            self.assertIn(old.name, second["seen"])
            recycler.request_recovery(self.directory, "fixture:resolve", self.directory / ("3" * 32 + ".json"))
            recycler.retire_batch(self.directory, "fixture:resolve", lease, second_manifest)
            third_manifest, third = launched[-1]
            self.assertEqual(len(third["requests"]), 1)
            self.assertTrue(set(third["requests"]).isdisjoint(second["seen"]))
            self.assertEqual(len(third["seen"]), 3)
            recycler.retire_batch(self.directory, "fixture:resolve", lease, third_manifest)
        self.assertEqual(len(launched), 2)
        self.assertEqual(len(recycler.requests(self.directory)), 3)  # Exhausted work is retained.
        with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as probe:
            cleanup.locking.acquire(probe)

    async def test_recycler_batch_does_not_consume_later_requests(self):
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        names = []
        for prefix in ("a", "b"):
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            journal.allocated(prefix)
            journal.detach()
            self.live.add(prefix)
            request = queue / (prefix * 32 + ".json")
            cleanup._atomic_json(request, {"kind": "journal", "journal": journal.path.name})
            names.append(request.name)
        manifest = batches / "batch.json"
        cleanup._atomic_json(manifest, {"requests": names[:1], "seen": names[:1]})
        report = await cleanup.recover_queued(self.directory, self.provider, 1, 5, 2, manifest)
        self.assertTrue(report["ok"])
        self.assertEqual(report["requests"], 1)
        self.assertEqual(self.live, {"b"})
        self.assertEqual([p.name for p in cleanup.recycler.requests(self.directory)], names[1:])

    async def test_successor_exit_does_not_renew_exhausted_journal_budget(self):
        recycler = cleanup.recycler
        old = cleanup.CleanupJournal(self.directory, "gateway-test")
        old.allocated("old-exhausted")
        old.detach()
        fresh = cleanup.CleanupJournal(self.directory, "gateway-test")
        fresh.allocated("fresh-resource")
        fresh.detach()
        self.live.update({"old-exhausted", "fresh-resource"})
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        old_request = queue / ("a" * 32 + ".json")
        exit_request = queue / ("b" * 32 + ".json")
        cleanup._atomic_json(old_request, {"kind": "journal", "journal": old.path.name})
        manifests = []

        def capture(command, **kwargs):
            manifests.append(Path(command[command.index("--batch") + 1]))
            return types.SimpleNamespace(wait=lambda: 0)

        with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as lease:
            cleanup.locking.acquire(lease)
            with patch.object(recycler.subprocess, "Popen", capture):
                recycler._launch_batch(self.directory, "fixture:resolve", lease, [old_request], seen=set())
                cleanup._atomic_json(exit_request, {"kind": "exit", "journal": None})
                recycler.retire_batch(self.directory, "fixture:resolve", lease, manifests[0])
        successor = manifests[1]
        with self.assertRaisesRegex(ValueError, "stop barrier"):
            recycler.batch_journals(self.directory, successor)
        result = await cleanup.recover_queued(self.directory, self.provider, 0.3, 1, 2, successor)
        scope, owned = recycler.batch_journals(self.directory, successor)
        self.assertEqual(scope, [fresh.path])
        self.assertEqual(owned, {old.path.name, fresh.path.name})
        self.assertFalse(result["ok"])
        self.assertEqual(result["requests"], 1)
        self.assertEqual(self.live, {"old-exhausted"})
        self.assertFalse(any(call == ("DELETE", "old-exhausted") for call in self.calls))
        self.assertTrue(old_request.exists())
        self.assertTrue(exit_request.exists())
        state = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertFalse(state["cleanup_confirmed"])

    async def test_selected_missing_journal_prevents_run_completion(self):
        missing = self.directory / ("1" * 32 + ".json")
        result = await cleanup.finalize_run(self.directory, self.provider, 0.1, 0.2, journals=[missing])
        self.assertFalse(result["cleanup_confirmed"])
        self.assertFalse(result["success"])
        self.assertEqual(self.calls, [])
        state = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertFalse(state["cleanup_confirmed"])

    async def test_successor_exit_preserves_missing_ancestor_journal_evidence(self):
        missing = self.directory / ("1" * 32 + ".json")
        fresh = cleanup.CleanupJournal(self.directory, "gateway-test")
        fresh.allocated("fresh-resource")
        fresh.detach()
        self.live.add("fresh-resource")
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        old_request = queue / ("a" * 32 + ".json")
        exit_request = queue / ("b" * 32 + ".json")
        cleanup._atomic_json(old_request, {"kind": "journal", "journal": missing.name})
        cleanup._atomic_json(exit_request, {"kind": "exit", "journal": None})
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        manifest = batches / "successor.json"
        cleanup.run_control.request_stop(self.directory, "test-sealed-scope")
        cleanup._atomic_json(
            manifest,
            {
                "requests": [exit_request.name],
                "exit_scope": "sealed",
                "seen": [old_request.name, exit_request.name],
                "journals": [fresh.path.name],
                "seen_journals": [missing.name, fresh.path.name],
            },
        )
        result = await cleanup.recover_queued(self.directory, self.provider, 0.3, 1, 2, manifest)
        self.assertFalse(result["ok"])
        self.assertEqual(result["acknowledged"], 0)
        self.assertFalse(self.live)
        self.assertEqual([call for call in self.calls if call[0] == "DELETE"], [("DELETE", "fresh-resource")])
        self.assertTrue(old_request.exists())
        self.assertTrue(exit_request.exists())
        state = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertFalse(state["cleanup_confirmed"])

    async def test_completion_observation_preserves_unknown_or_live_ownership(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        self.assertFalse(cleanup.observe_journals([journal.path])["ok"])
        journal.detach()
        self.assertTrue(cleanup.observe_journals([journal.path])["ok"])
        original = json.loads(journal.path.read_text())
        for changes in (
            {"creating": True},
            {"ambiguous_creates": 1},
            {"resources": {"unknown": {"confirmed": False}}},
            {"resources": {"invalid": True}},
            {"provider_key": ""},
        ):
            with self.subTest(changes=changes):
                cleanup._atomic_json(journal.path, {**original, **changes})
                self.assertFalse(cleanup.observe_journals([journal.path])["ok"])
        journal.path.write_text("invalid json")
        self.assertFalse(cleanup.observe_journals([journal.path])["ok"])
        journal.path.unlink()
        self.assertFalse(cleanup.observe_journals([journal.path])["ok"])
        self.assertEqual(self.calls, [])

    def test_legacy_batch_does_not_guess_missing_ancestor_work(self):
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        fresh = queue / ("b" * 32 + ".json")
        cleanup._atomic_json(fresh, {"kind": "journal", "journal": "2" * 32 + ".json"})
        manifest = batches / "legacy.json"
        cleanup._atomic_json(manifest, {"requests": [fresh.name], "seen": ["a" * 32 + ".json", fresh.name]})
        with self.assertRaises(FileNotFoundError):
            cleanup.recycler.batch_journals(self.directory, manifest)
        self.assertTrue(fresh.exists())

    async def test_recycler_retirement_release_cannot_unlock_a_later_guardian(self):
        recycler = cleanup.recycler
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        manifest = batches / "empty.json"
        cleanup._atomic_json(manifest, {"requests": [], "seen": []})
        old = cleanup.locking.open_lock(self.directory / "recovery-worker.lock")
        self.addCleanup(old.close)
        cleanup.locking.acquire(old)
        recycler.retire_batch(self.directory, "fixture:resolve", old, manifest)
        inherited = []

        def inherit(command, **kwargs):
            inherited.append(os.fdopen(os.dup(kwargs["pass_fds"][0]), "a"))
            self.addCleanup(inherited[-1].close)
            return types.SimpleNamespace(wait=lambda: 0)

        with patch.object(recycler.subprocess, "Popen", inherit):
            recycler.request_recovery(self.directory, "fixture:resolve", self.directory / ("a" * 32 + ".json"))
        old.close()  # A different open-file description must keep its new lock.
        with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as probe:
            with self.assertRaises(BlockingIOError):
                cleanup.locking.acquire(probe)

    async def test_recycler_successor_launch_failure_retains_new_requests(self):
        recycler = cleanup.recycler
        batches = self.directory / "recovery-batches"
        batches.mkdir()
        manifest = batches / "empty.json"
        cleanup._atomic_json(manifest, {"requests": [], "seen": []})
        queue = self.directory / "recovery-requests"
        queue.mkdir()
        request = queue / ("a" * 32 + ".json")
        cleanup._atomic_json(request, {"kind": "exit", "journal": None})
        with cleanup.locking.open_lock(self.directory / "recovery-worker.lock") as lease:
            cleanup.locking.acquire(lease)
            with patch.object(recycler.subprocess, "Popen", side_effect=OSError("launch failed")):
                with self.assertRaises(OSError):
                    recycler.retire_batch(self.directory, "fixture:resolve", lease, manifest)
        self.assertEqual(recycler.requests(self.directory), [request])

    async def test_failed_exit_retries_with_backoff_and_retains_exhausted_work(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("exit-failure")
        journal.detach()
        self.live.add("exit-failure")
        self.fail_delete = True
        environment = self._recovery_environment()
        environment.update(
            {
                "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "6",
                "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "4",
                "XTUNER_SANDBOX_RECOVERY_BUDGET": "20",
                "XTUNER_SANDBOX_RECOVERY_BACKOFF": ".2",
                "XTUNER_SANDBOX_RECOVERY_MAX_BACKOFF": ".4",
            }
        )
        with patch.dict(os.environ, environment):
            confirmed = await asyncio.to_thread(
                cleanup.cleanup_from_environment, finalize=True, training_outcome="succeeded"
            )
            self.assertFalse(confirmed)
            await self._wait_recycler_idle()
            status = json.loads((self.directory / "recovery-status.json").read_text())
            self.assertEqual(status["status"], "exhausted")
            history = status["history"]
            self.assertEqual(len(history), 3)
            for index, (before, after) in enumerate(zip(history, history[1:])):
                self.assertGreaterEqual(after["started_at"] - before["finished_at"], 0.16 * 2**index)
                self.assertLessEqual(before["next_delay_seconds"], 0.4)
            self.assertTrue(cleanup.recycler.requests(self.directory))
            self.assertIn("exit-failure", self.live)
            self.assertFalse(json.loads(journal.path.read_text())["resources"]["exit-failure"]["confirmed"])
            # A later explicit handoff restarts recovery with a fresh connection.
            self.fail_delete = False
            cleanup.recycler.request_recovery(self.directory, "recovery_fixture:resolve")
            await self._wait_recycler_idle()
        self.assertFalse(self.live)
        self.assertFalse(cleanup.recycler.requests(self.directory))
        report = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertTrue(report["cleanup_confirmed"])
        self.assertTrue(report["finalization_failed"])
        self.assertFalse(report["success"])

    async def test_pool_failure_hands_off_to_durable_recycler(self):
        environment = self._recovery_environment()
        self.fail_delete = True
        with patch.dict(os.environ, environment):
            pool = self.pool(recovery=False)
            await pool.get("main")
            with self.assertRaises(CleanupError):
                await pool.aclose()
            await cleanup.drain_cleanup_tasks(3)
            self.assertTrue(cleanup.recycler.requests(self.directory))
            self.assertTrue(pool._lifecycle._journal.lock.closed)
            self.assertTrue(pool._lifecycle._provider.closed)
            self.fail_delete = False
            await self._wait_recycler_idle()
        self.assertFalse(self.live)
        self.assertFalse(cleanup.recycler.requests(self.directory))
        self.assertFalse((self.directory / "stop.requested.json").exists())

    async def test_supervisor_recovers_driver_sigkill_automatically(self):
        driver = await self._supervised_driver()
        try:
            await asyncio.sleep(0.5)
            self.assertIn("active-driver", self.live)
            driver.kill()
            await driver.wait()
            await self._wait_deleted("active-driver")
            self.assertEqual(driver.returncode, -9)
            await asyncio.sleep(0.3)
            self.assertFalse(
                (self.directory / "recovery-worker.lock").exists(),
                "Successful exit sweep must not enqueue background retries",
            )
        finally:
            if driver.returncode is None:
                driver.kill()
                await driver.wait()
            await asyncio.sleep(0.2)

    async def test_exit_witness_bounds_admission_lock_and_hands_off(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.allocated("blocked-admission")
        journal.detach()
        self.live.add("blocked-admission")
        # An unlocked registered lease is exactly the state after driver death.
        (self.directory / "driver.started.json").write_text("{}")
        environment = self._recovery_environment()
        # Leave enough of the 20% handoff reserve to start an interpreter from
        # shared storage; the injected admission lock, not startup latency, is
        # the operation whose bounded termination this case verifies.
        environment["XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT"] = "4"
        with cleanup.locking.open_lock(self.directory / "admission.lock") as lock:
            cleanup.locking.acquire(lock, shared=True)
            witness = await asyncio.create_subprocess_exec(
                sys.executable,
                str(SOURCE / "supervisor.py"),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                self.assertEqual(await asyncio.wait_for(witness.stdout.readline(), 10), b"READY\n")
                started = time.monotonic()
                await asyncio.wait_for(witness.wait(), 6)
                self.assertLess(time.monotonic() - started, 5)
                self.assertEqual(witness.returncode, 1)
                self.assertFalse((self.directory / "stop.requested.json").exists())
                self.assertTrue(cleanup.recycler.requests(self.directory))
                self.assertIn("blocked-admission", self.live)
            finally:
                if witness.returncode is None:
                    witness.kill()
                    await witness.wait()
        await self._wait_deleted("blocked-admission")
        await self._wait_recycler_idle()
        self.assertTrue(json.loads((self.directory / "run-cleanup.json").read_text())["cleanup_confirmed"])

    async def test_recycler_total_budget_covers_status_and_cleanup_child_fsync(self):
        for target in ("status", "journal"):
            with self.subTest(blocked_write=target):
                first = cleanup.CleanupJournal(self.directory, "gateway-test")
                first.allocated("first-" + target)
                first.detach()
                self.live.add("first-" + target)
                site = self.directory / "sitecustomize.py"
                marker = self.directory / (target + "-fsync-entered")
                role = "--run-worker" if target == "status" else "--recover-queued"
                prefix = "recovery-status.json." if target == "status" else first.path.name + "."
                site.write_text(
                    "import os, sys, time\nfrom pathlib import Path\noriginal = os.fsync\n"
                    "def blocked(fd):\n"
                    f"    if {role!r} in sys.argv and Path(os.readlink('/proc/self/fd/' + str(fd))).name.startswith({prefix!r}):\n"
                    f"        Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                    "        time.sleep(60)\n"
                    "    return original(fd)\n"
                    "os.fsync = blocked\n"
                )
                environment = self._recovery_environment()
                environment.update(XTUNER_SANDBOX_RECOVERY_BUDGET="4", XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT="10")
                with patch.dict(os.environ, environment):
                    self.assertTrue(
                        await asyncio.to_thread(
                            cleanup.recycler.handoff_recovery,
                            self.directory,
                            "recovery_fixture:resolve",
                            first.path,
                            timeout=3,
                        )
                    )
                    async with asyncio.timeout(5):
                        while not marker.exists():
                            await asyncio.sleep(0.02)
                    # A second handoff arrives after the immutable batch was
                    # claimed. The guardian must transfer it after killing the
                    # stalled worker, without extending the old batch's budget.
                    second = cleanup.CleanupJournal(self.directory, "gateway-test")
                    second.allocated("second-" + target)
                    second.detach()
                    self.live.add("second-" + target)
                    self.assertTrue(
                        await asyncio.to_thread(
                            cleanup.recycler.handoff_recovery,
                            self.directory,
                            "recovery_fixture:resolve",
                            second.path,
                            timeout=3,
                        )
                    )
                    started = time.monotonic()
                    await asyncio.wait_for(self._wait_recycler_idle(), 10)
                    self.assertLess(time.monotonic() - started, 9)
                self.assertNotIn("second-" + target, self.live)
                # With a status-only stall, the original child already deleted
                # and acknowledged its batch. A journal fsync stall must retain
                # the first request until a later explicit retry confirms it.
                self.assertEqual(bool(cleanup.recycler.requests(self.directory)), target == "journal")
                # The cleanup child must also be dead; otherwise it could
                # overwrite a later reconciler's durable confirmation.
                stat = Path("/proc") / marker.read_text() / "stat"
                if stat.exists():
                    self.assertEqual(stat.read_text().split(") ", 1)[1][0], "Z")
                site.unlink()
                with patch.dict(os.environ, self._recovery_environment()):
                    self.assertTrue(
                        await asyncio.to_thread(
                            cleanup.recycler.handoff_recovery,
                            self.directory,
                            "recovery_fixture:resolve",
                            second.path,
                            timeout=3,
                        )
                    )
                await self._wait_deleted("second-" + target)
                await self._wait_recycler_idle()
                self.assertFalse(self.live)
                self.assertFalse(cleanup.recycler.requests(self.directory))

    async def test_exit_witness_does_not_scan_until_failure_is_handed_off(self):
        driver = await self._supervised_driver()
        try:
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            journal.created("dead-worker")
            journal.detach()
            self.live.add("dead-worker")
            await asyncio.sleep(0.4)
            self.assertIn("dead-worker", self.live)
            self.assertFalse(self.calls, "An exit witness must not scan during normal training")
            with patch.dict(os.environ, self.supervisor_environment):
                cleanup.recycler.request_recovery(self.directory, "recovery_fixture:resolve", journal.path)
            await self._wait_deleted("dead-worker")
            self.assertIsNone(driver.returncode)
            self.assertIn("active-driver", self.live)
            driver.stdin.close()
            await driver.wait()
            await self._wait_deleted("active-driver")
        finally:
            if driver.returncode is None:
                driver.kill()
                await driver.wait()
            await asyncio.sleep(0.2)

    async def test_forked_helper_does_not_keep_dead_driver_or_pool_lease(self):
        code = """
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cleanup
lease = cleanup.run_control.claim_driver(Path(sys.argv[2]))
journal = cleanup.CleanupJournal(sys.argv[2], 'gateway-test')
journal.created('fork-owned')
pid = os.fork()
if pid == 0:
    time.sleep(20)
    os._exit(0)
print(pid, flush=True)
sys.stdin.read()
"""
        driver = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(SOURCE),
            str(self.directory),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        helper_pid = None
        try:
            helper_pid = int(await asyncio.wait_for(driver.stdout.readline(), 5))
            self.live.add("fork-owned")
            self.assertTrue(cleanup.run_control.driver_alive(self.directory))
            driver.kill()
            # The forked helper still owns stdout, so communicate()/wait() on
            # asyncio's pipe transport cannot complete until it exits as well.
            async with asyncio.timeout(5):
                while driver.returncode is None:
                    await asyncio.sleep(0.02)
            self.assertFalse(cleanup.run_control.driver_alive(self.directory))
            os.kill(helper_pid, 0)
            report = await cleanup.reconcile(self.directory, self.provider, 0.2)
            self.assertTrue(report["ok"])
            self.assertFalse(self.live)
        finally:
            if helper_pid is not None:
                try:
                    os.kill(helper_pid, 9)
                except ProcessLookupError:
                    pass
            if driver.returncode is None:
                driver.kill()
            await driver.wait()

    async def test_successful_finish_publishes_confirmed_run_status(self):
        (self.directory / "empty_recovery.py").write_text(
            "def resolve(key):\n    raise AssertionError('no resources')\n"
        )
        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": str(self.directory),
                "XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory),
                "XTUNER_SANDBOX_CLEANUP_FACTORY": "empty_recovery:resolve",
                "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "3",
                "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "2",
                "XTUNER_SANDBOX_RECOVERY_BUDGET": "8",
                "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "3",
                "XTUNER_SANDBOX_RECOVERY_BACKOFF": ".1",
                "XTUNER_SANDBOX_RECOVERY_MAX_BACKOFF": ".2",
            },
        ):
            await asyncio.to_thread(cleanup.finish_training, "succeeded")
        report = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertTrue(report["cleanup_confirmed"])
        self.assertTrue(report["fit_cleanup_confirmed"])
        self.assertTrue(report["success"])
        self.assertFalse((self.directory / "recovery-requests").exists())
        self.assertFalse((self.directory / "recovery-worker.lock").exists())
        with self.assertRaises(cleanup.run_control.RunStoppingError):
            cleanup.CleanupJournal(self.directory, "gateway-test")

    async def test_cleanup_child_confirms_resources_without_driver_fit_ack(self):
        result = await cleanup.finalize_run(self.directory, self.provider, 0.1, 0.2, training_outcome="succeeded")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertFalse(result["fit_cleanup_confirmed"])
        self.assertFalse(result["success"])

    def test_post_publication_cleanup_failure_never_publishes_fit_success(self):
        for failure in ("timeout", "nonzero"):
            with self.subTest(failure=failure):
                directory = self.directory / failure

                def run(command, **kwargs):
                    if "--record-fit" in command:
                        # Even failure-report storage being unavailable cannot
                        # introduce a driver ACK which was never issued.
                        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                    cleanup.run_control.request_stop(directory, "training_exit")
                    cleanup.run_control.record_result(directory, cleanup_confirmed=True, training_outcome="succeeded")
                    if failure == "timeout":
                        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                    return types.SimpleNamespace(returncode=1)

                environment = {
                    "XTUNER_SANDBOX_CLEANUP_DIR": str(directory),
                    "XTUNER_SANDBOX_CLEANUP_FACTORY": "fixture:resolve",
                    "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "2",
                }
                with (
                    patch.dict(os.environ, environment),
                    patch.object(cleanup.subprocess, "run", run),
                    patch.object(cleanup.recycler, "handoff_recovery", return_value=True),
                ):
                    with self.assertRaises(CleanupError):
                        cleanup.finish_training("succeeded")
                result = cleanup.run_control.record_result(directory, cleanup_confirmed=True)
                self.assertTrue(result["cleanup_confirmed"])
                self.assertFalse(result["fit_cleanup_confirmed"])
                self.assertFalse(result["success"])

    def test_fit_ack_timeout_reports_observed_success_without_new_fit_failure(self):
        for after_write in (False, True):
            with self.subTest(after_write=after_write):
                directory = self.directory / str(after_write)

                def run(command, **kwargs):
                    cleanup.run_control.request_stop(directory, "training_exit")
                    if "--record-fit" in command:
                        self.assertEqual(command[command.index("--record-fit") + 1], "confirmed")
                        if after_write:
                            cleanup.run_control.record_result(
                                directory,
                                cleanup_confirmed=True,
                                training_outcome="succeeded",
                                fit_acknowledged=True,
                            )
                        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                    cleanup.run_control.record_result(directory, cleanup_confirmed=True, training_outcome="succeeded")
                    return types.SimpleNamespace(returncode=0)

                environment = {
                    "XTUNER_SANDBOX_CLEANUP_DIR": str(directory),
                    "XTUNER_SANDBOX_CLEANUP_FACTORY": "fixture:resolve",
                    "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "2",
                }
                with patch.dict(os.environ, environment), patch.object(cleanup.subprocess, "run", run):
                    cleanup.finish_training("succeeded")
                result = json.loads((directory / "run-cleanup.json").read_text())
                self.assertTrue(result["cleanup_confirmed"])
                self.assertEqual(result["success"], after_write)

    def test_driver_failure_report_is_sticky_after_resource_recovery(self):
        arguments = [
            "cleanup.py",
            "--directory",
            str(self.directory),
            "--record-fit",
            "unconfirmed",
            "--training-outcome",
            "succeeded",
        ]
        with patch.object(sys, "argv", arguments):
            self.assertEqual(cleanup.main(), 0)
        result = cleanup.run_control.record_result(
            self.directory, cleanup_confirmed=True, training_outcome="succeeded", fit_acknowledged=True
        )
        self.assertTrue(result["cleanup_confirmed"])
        self.assertTrue(result["finalization_failed"])
        self.assertFalse(result["success"])

    def test_final_success_requires_stop_barrier(self):
        with self.assertRaises(RuntimeError):
            cleanup.run_control.record_result(self.directory, cleanup_confirmed=True, training_outcome="succeeded")
        self.assertFalse((self.directory / "run-cleanup.json").exists())

    def test_pool_cannot_escape_driver_cleanup_directory(self):
        with patch.dict(os.environ, {"XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory / "driver")}):
            with self.assertRaisesRegex(ValueError, "match the driver"):
                self.pool()
        self.assertEqual(self.calls, [])

    def test_colocate_stop_barrier_precedes_manager_drain(self):
        source = ROOT / "xtuner/v1/train/rl_trainer.py"
        cls = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "RLColocateTrainer"
        )
        method = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_fit_with_cleanup"
        )
        events = []
        namespace = {"sys": sys, "asyncio_run": lambda value: value, "close_trace": lambda: events.append("trace")}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        trainer = types.SimpleNamespace(
            rl_health_manager=types.SimpleNamespace(start=lambda: None, stop=lambda: None),
            _fit=lambda: events.append("fit"),
            _turn_keepalive=True,
            _debug_train=False,
            agent_loop_manager=types.SimpleNamespace(close=lambda: events.append("manager-close")),
            _exp_tracker=types.SimpleNamespace(close=lambda: None),
            logger=logging.getLogger("test"),
        )
        with patch.object(cleanup, "request_stop_from_environment", lambda: events.append("stop") or True):
            namespace["_fit_with_cleanup"](trainer)
        self.assertEqual(events, ["fit", "stop", "manager-close", "trace"])

    async def test_stop_barrier_rejects_new_pool_and_phase_creates(self):
        pool = self.pool()
        await pool.get("main")
        await pool.release_all()
        before = list(self.calls)
        cleanup.run_control.request_stop(self.directory, "test_stop")
        with self.assertRaises(cleanup.run_control.RunStoppingError):
            await pool.get("verifier")
        with self.assertRaises(cleanup.run_control.RunStoppingError):
            cleanup.CleanupJournal(self.directory, "gateway-test")
        self.assertEqual(self.calls, before)
        await pool.aclose()
        report = await cleanup.finalize_run(self.directory, self.provider, 0.1, 0.2, training_outcome="succeeded")
        self.assertTrue(report["cleanup_confirmed"])
        self.assertFalse(report["success"])

    async def test_inflight_create_is_accounted_for_before_final_confirmation(self):
        pool = self.pool()
        entered, resume = asyncio.Event(), asyncio.Event()
        original = pool._lifecycle._provider.create

        async def delayed_create(**kwargs):
            entered.set()
            await resume.wait()
            return await original(**kwargs)

        pool._lifecycle._provider.create = delayed_create
        acquiring = asyncio.create_task(pool.get("main"))
        await entered.wait()
        report = await cleanup.finalize_run(self.directory, self.provider, 0.05, 0.1)
        self.assertFalse(report["cleanup_confirmed"])
        resume.set()
        with self.assertRaises(cleanup.run_control.RunStoppingError) as error:
            await acquiring
        self.assertEqual(error.exception.allocation_state, "allocated")
        await pool.aclose()
        self.assertFalse(self.live)
        report = await cleanup.finalize_run(self.directory, self.provider, 0.1, 0.2, training_outcome="succeeded")
        self.assertTrue(report["cleanup_confirmed"])
        self.assertFalse(report["success"])
        self.assertEqual(pool._lifecycle._journal.data["ambiguous_creates"], 0)

    async def test_stop_and_journal_registration_are_serialized(self):
        for index in range(10):
            directory = self.directory / str(index)
            barrier = threading.Barrier(2)

            def allocate():
                barrier.wait()
                try:
                    return cleanup.CleanupJournal(directory, "gateway-test")
                except cleanup.run_control.RunStoppingError:
                    return None

            def stop():
                barrier.wait()
                cleanup.run_control.request_stop(directory, "race")

            journal, _ = await asyncio.gather(asyncio.to_thread(allocate), asyncio.to_thread(stop))
            self.assertTrue(cleanup.run_control.is_stopping(directory))
            report = await cleanup.reconcile(directory, self.provider, 0.1)
            self.assertEqual(report["ok"], journal is None)
            if journal is not None:
                journal.detach()
            with self.assertRaises(cleanup.run_control.RunStoppingError):
                cleanup.CleanupJournal(directory, "gateway-test")

    async def test_unconfirmed_fit_cannot_be_relabelled_success_by_later_recovery(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        journal.created("still-live")
        self.live.add("still-live")
        try:
            result = await cleanup.finalize_run(self.directory, self.provider, 0.05, 0.1, training_outcome="succeeded")
            self.assertFalse(result["success"])
            self.assertTrue(result["finalization_failed"])
        finally:
            journal.detach()
        result = await cleanup.finalize_run(self.directory, self.provider, 0.1, 0.2)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertFalse(result["success"])
        self.assertFalse(self.live)

    def test_recovery_without_fit_ack_does_not_invent_training_success(self):
        control = cleanup.run_control
        control.request_stop(self.directory, "driver_lost")
        control.record_result(self.directory, cleanup_confirmed=False, training_outcome="succeeded")
        result = control.record_result(self.directory, cleanup_confirmed=True)
        self.assertFalse(result["success"])
        self.assertFalse(result["fit_cleanup_confirmed"])
        result = control.record_result(self.directory, cleanup_confirmed=True, training_outcome="failed")
        self.assertFalse(result["success"])
        result = control.record_result(self.directory, cleanup_confirmed=False)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(result["training_outcome"], "failed")

    async def test_surviving_peer_performs_exit_sweep_after_head_witness_dies(self):
        driver = await self._supervised_driver()
        peer = None
        try:
            code = """
import sys
sys.path.insert(0, sys.argv[1])
import cleanup
cleanup.start_cleanup_supervisor(peer=True)
print('ready', flush=True)
sys.stdin.read()
"""
            peer = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                str(SOURCE),
                env=self.supervisor_environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.assertEqual(await asyncio.wait_for(peer.stdout.readline(), 15), b"ready\n")
            owner_pid = int((self.directory / "driver-witness.pid").read_text())
            os.kill(owner_pid, 9)
            # Standby outlives its own actor and must not interpret actor death
            # as permission to stop the still-live driver.
            peer.kill()
            await peer.wait()
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            journal.created("orphan-after-owner-kill")
            journal.detach()
            self.live.add("orphan-after-owner-kill")
            await asyncio.sleep(0.4)
            self.assertIn("orphan-after-owner-kill", self.live)
            self.assertFalse(cleanup.run_control.is_stopping(self.directory))
            self.assertIn("active-driver", self.live)
            driver.kill()
            await driver.wait()
            await self._wait_deleted("active-driver")
            await self._wait_deleted("orphan-after-owner-kill")
        finally:
            for process in (driver, peer):
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
            await asyncio.sleep(0.3)

    async def test_factory_import_failure_still_seals_run_and_records_failure(self):
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            str(SOURCE / "cleanup.py"),
            "--directory",
            str(self.directory),
            "--factory",
            "missing_cleanup_factory:resolve",
            "--finalize",
            "--training-outcome",
            "succeeded",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(child.communicate(), 5)
        self.assertNotEqual(child.returncode, 0)
        self.assertTrue(cleanup.run_control.is_stopping(self.directory))
        report = json.loads((self.directory / "run-cleanup.json").read_text())
        self.assertFalse(report["success"])
        self.assertTrue(report["finalization_failed"])

    def test_supervisor_rejects_incomplete_configuration(self):
        with patch.dict(os.environ, {"XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory)}):
            os.environ.pop("XTUNER_SANDBOX_CLEANUP_FACTORY", None)
            with self.assertRaises(ValueError):
                cleanup.start_cleanup_supervisor()

    def test_shutdown_guard_rejects_invalid_timeout_before_training(self):
        environment = self._recovery_environment()
        for value in ("0", "-1", "nan", "inf", "bad"):
            with (
                self.subTest(timeout=value),
                patch.dict(os.environ, {**environment, "XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT": value}),
            ):
                with self.assertRaises(ValueError):
                    cleanup.start_cleanup_supervisor()
                self.assertFalse((self.directory / "driver.started.json").exists())

    def test_both_trainer_fit_paths_recover_and_preserve_original_outcome(self):
        source = ROOT / "xtuner/v1/train/rl_trainer.py"
        classes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef)]
        for cls in classes:
            if cls.name not in {"RLColocateTrainer", "RLDisaggregatedTrainer"}:
                continue
            fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
            namespace = {"sys": sys}
            exec(compile(ast.Module(body=[fit], type_ignores=[]), str(source), "exec"), namespace)
            for failure in (None, ValueError("training failed"), KeyboardInterrupt()):
                events = []

                def run():
                    events.append("fit")
                    if failure is not None:
                        raise failure
                    return "trained"

                def recover(**kwargs):
                    events.append("recover")
                    raise RuntimeError("cleanup unavailable")

                trainer = types.SimpleNamespace(_fit_with_cleanup=run, logger=logging.getLogger("test"))
                with patch.object(cleanup, "cleanup_from_environment", recover):
                    if failure is None:
                        with self.assertRaises(CleanupError):
                            namespace["fit"](trainer)
                    else:
                        with self.assertRaises(type(failure)) as caught:
                            namespace["fit"](trainer)
                        self.assertIs(caught.exception, failure)
                self.assertEqual(events, ["fit", "recover"])

    def test_agent_actor_paths_forward_cleanup_environment(self):
        source = ROOT / "xtuner/v1/rl/agent_loop/agent_loop.py"
        cls = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "AgentLoopConfig"
        )
        options = []
        namespace = {
            "os": os,
            "ray": types.SimpleNamespace(remote=lambda **kwargs: options.append(kwargs) or (lambda cls: cls)),
            "AgentLoopActor": object,
            "AGENT_LOOP_CONCURRENCY_GROUP_GENERATE": "generate",
            "CPUActorLauncher": types.SimpleNamespace(
                build_actor=lambda *a, **k: None, build_actors=lambda *a, **k: []
            ),
            "cast": lambda annotation, value: value,
            "RayAgentLoopProxy": object,
        }
        resources = types.SimpleNamespace(num_cpus_per_worker=1, cpu_memory_per_worker=1024, num_workers=2)
        for name in ("_build_ray_actor", "_build_ray_actors"):
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            tree = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                type_ignores=[],
            )
            ast.fix_missing_locations(tree)
            exec(compile(tree, str(source), "exec"), namespace)
            with patch.dict(
                os.environ,
                {
                    "XTUNER_SANDBOX_CLEANUP_DIR": "/shared/run-1",
                    "XTUNER_SANDBOX_PROVIDER_KEY": "backend-a",
                    "XTUNER_SANDBOX_CLEANUP_FACTORY": "private:factory",
                    "XTUNER_SANDBOX_CLEANUP_PEERS": "0",
                },
            ):
                namespace[name](object(), object(), resources, 256)
                os.environ["XTUNER_SANDBOX_CLEANUP_PEERS"] = "1"
                namespace[name](object(), object(), resources, 256)
        self.assertEqual(len(options), 4)
        for option in options[1::2]:
            self.assertEqual(option["runtime_env"]["env_vars"]["XTUNER_SANDBOX_CLEANUP_FACTORY"], "private:factory")
            self.assertEqual(option["runtime_env"]["env_vars"]["XTUNER_SANDBOX_CLEANUP_PEERS"], "1")
        for option in options[::2]:
            self.assertEqual(
                option["runtime_env"]["env_vars"],
                {
                    "XTUNER_SANDBOX_CLEANUP_DIR": "/shared/run-1",
                    "XTUNER_SANDBOX_PROVIDER_KEY": "backend-a",
                    "XTUNER_SANDBOX_CLEANUP_FACTORY": "private:factory",
                },
            )
            self.assertEqual(option["concurrency_groups"], {"generate": 256})

    def test_default_ray_shutdown_does_not_stop_shared_cluster(self):
        source = ROOT / "xtuner/v1/rl/utils/ray_utils.py"
        node = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "close_ray"
        )
        events = []
        namespace = {
            "ray": types.SimpleNamespace(is_initialized=lambda: True, shutdown=lambda: events.append("disconnect")),
            "subprocess": types.SimpleNamespace(run=lambda *a, **k: events.append("cluster-stop")),
            "logger": logging.getLogger("test"),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        namespace["close_ray"]()
        self.assertEqual(events, ["disconnect"])

    async def test_create_keeps_user_env_and_extra_params(self):
        pool = self.pool()
        calls = []
        create = pool._lifecycle._provider.create

        async def capture(**kwargs):
            calls.append(kwargs)
            return await create(**kwargs)

        pool._lifecycle._provider.create = capture
        spec = pool._specs["main"]
        spec.user = "${CLEANUP_TEST_USER}"
        spec.env = "preview-template"
        spec.extra_params = {"project": "${CLEANUP_TEST_PROJECT}"}
        spec.env_vars = {"TASK_MODE": "preview"}
        with patch.dict(os.environ, {"CLEANUP_TEST_USER": "explicit-user", "CLEANUP_TEST_PROJECT": "test-project"}):
            await pool.get("main")
        self.assertEqual(calls[0]["user"], "cleanup-test-user")
        self.assertEqual(calls[1]["user"], "explicit-user")
        self.assertEqual(calls[1]["env"], "preview-template")
        self.assertEqual(calls[1]["extra_params"], {"project": "test-project"})
        self.assertEqual(calls[1]["env_vars"], {"TASK_MODE": "preview"})
        await pool.aclose()
        self.assertFalse(self.live)

    async def _stopped_runner(self, phase, *, driver_loss=False):
        driver = await self._supervised_driver() if driver_loss else None
        pool = self.pool()
        entered = asyncio.Event()

        async def block(client, item, record):
            entered.set()
            await asyncio.Event().wait()

        runner = Runner(
            pool=pool,
            infer=sandbox.SandboxStage(post=[block] if phase == "infer" else []),
            validate=Judger("score", sandbox.SandboxStage(post=[block] if phase == "validate" else [])),
        )
        item = schemas.AgentRolloutItem(
            id="stop", uid=4, data_source="test", instruction="test", task_root=self.directory
        )
        task = asyncio.create_task(runner.run(item))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if driver is not None:
                driver.kill()
                await driver.wait()
            else:
                cleanup.run_control.request_stop(self.directory, "test")
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            self.assertEqual(item.status, schemas.RolloutStatus.FAILED)
            self.assertTrue(pool._lifecycle._journal.lock.closed)
            if driver is not None:
                await self._wait_deleted("active-driver")
            self.assertFalse(self.live)
        finally:
            if driver is not None and driver.returncode is None:
                driver.kill()
                await driver.wait()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_finished_runner_stop_observer_is_removed(self):
        control = importlib.import_module(PACKAGE + ".task_control")
        async with control.watch_task(str(self.directory)) as stopped:
            pass
        cleanup.run_control.request_stop(self.directory, "test")
        await asyncio.sleep(0.6)
        self.assertFalse(stopped.is_set(), "Later caller work must not receive a stale stop cancellation")

    async def test_stop_marker_cancels_live_infer_and_releases_pool(self):
        await self._stopped_runner("infer")

    async def test_stop_marker_cancels_live_verifier_and_releases_pool(self):
        await self._stopped_runner("validate")

    async def test_driver_sigkill_supervisor_stops_surviving_runner(self):
        await self._stopped_runner("infer", driver_loss=True)

    async def test_runner_repeated_cancellation_finishes_cleanup(self):
        pool = self.pool()
        entered, deleting, allow_delete = asyncio.Event(), asyncio.Event(), asyncio.Event()
        delete = pool._lifecycle._provider.delete

        async def blocked_delete(env_id):
            deleting.set()
            await allow_delete.wait()
            await delete(env_id)

        async def block(client, item, record):
            entered.set()
            await asyncio.Event().wait()

        pool._lifecycle._provider.delete = blocked_delete
        runner = Runner(
            pool=pool, infer=sandbox.SandboxStage(post=[block]), validate=Judger("score", sandbox.SandboxStage())
        )
        item = schemas.AgentRolloutItem(
            id="cancel", uid=2, data_source="test", instruction="test", task_root=self.directory
        )
        task = asyncio.create_task(runner.run(item))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.wait_for(deleting.wait(), 2)
        task.cancel()
        allow_delete.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(item.status, schemas.RolloutStatus.FAILED)
        self.assertEqual(item.error.category, "rollout_cancelled")
        self.assertTrue(item.artifacts["sandbox_cleanup"]["ok"])
        self.assertFalse(self.live)

    async def test_runner_preserves_business_error_when_cleanup_fails(self):
        pool = self.pool()
        self.fail_delete = True

        async def fail(client, item, record):
            raise ValueError("business failure")

        runner = Runner(
            pool=pool, infer=sandbox.SandboxStage(post=[fail]), validate=Judger("score", sandbox.SandboxStage())
        )
        item = schemas.AgentRolloutItem(
            id="failed", uid=3, data_source="test", instruction="test", task_root=self.directory
        )
        result = await runner.run(item)
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)
        self.assertEqual(result.error.type, "ValueError")
        self.assertIn("business failure", result.error.message)
        self.assertFalse(result.artifacts["sandbox_cleanup"]["ok"])
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)

    async def test_runner_keeps_cleanup_failure_out_of_training(self):
        pool = self.pool()
        self.fail_delete = True

        async def score(client, item, record):
            record.score = 1.0

        runner = Runner(
            pool=pool, infer=sandbox.SandboxStage(), validate=Judger("score", sandbox.SandboxStage(post=[score]))
        )
        item = schemas.AgentRolloutItem(
            id="test", uid=1, data_source="test", instruction="test", task_root=self.directory
        )
        result = await runner.run(item)
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)
        self.assertIsNotNone(result.error)
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)

    async def test_runner_cleanup_failure_ignores_handled_caller_exception(self):
        pool = self.pool()
        self.fail_delete = True

        async def score(client, item, record):
            record.score = 1.0

        runner = Runner(
            pool=pool, infer=sandbox.SandboxStage(), validate=Judger("score", sandbox.SandboxStage(post=[score]))
        )
        item = schemas.AgentRolloutItem(
            id="fallback", uid=5, data_source="test", instruction="test", task_root=self.directory
        )
        try:
            raise ValueError("unrelated caller fallback")
        except ValueError:
            result = await runner.run(item)
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)
        self.assertEqual(result.error.stage, "cleanup")
        self.assertEqual(result.error.category, "cleanup_failed")
        self.assertFalse(result.artifacts["sandbox_cleanup"]["ok"])
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)

    async def test_runner_cancelled_task_preserves_cancellation_when_cleanup_fails(self):
        pool = self.pool()
        self.fail_delete = True
        entered = asyncio.Event()

        async def block(client, item, record):
            entered.set()
            await asyncio.Event().wait()

        runner = Runner(
            pool=pool, infer=sandbox.SandboxStage(pre=[block]), validate=Judger("unused", sandbox.SandboxStage())
        )
        item = schemas.AgentRolloutItem(
            id="cancel-with-failed-cleanup", uid=6, data_source="test", instruction="test", task_root=self.directory
        )
        task = asyncio.create_task(runner.run(item))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        self.assertTrue(task.cancelled())
        self.assertEqual(item.error.category, "rollout_cancelled")
        self.assertFalse(item.artifacts["sandbox_cleanup"]["ok"])
        self.fail_delete = False
        await pool._lifecycle._deferred_cleanup
        self.assertFalse(self.live)


if __name__ == "__main__":
    unittest.main(verbosity=2)
