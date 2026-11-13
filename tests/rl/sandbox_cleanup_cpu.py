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
            if pool._journal and not pool._journal.lock.closed:
                pool._journal.lock.close()
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

    async def test_phase_release_then_verifier_then_final_close(self):
        pool = self.pool()
        await pool.get("main")
        await pool.release_all()
        self.assertFalse(pool._provider.closed)
        await pool.get("verifier")
        await pool.aclose()
        self.assertTrue(pool._provider.closed)
        self.assertFalse(self.live)
        deletes = [key for method, key in self.calls if method == "DELETE"]
        self.assertEqual(deletes, ["main-2", "dep-1", "verifier-3"])
        self.assertEqual(json.loads(pool._journal.path.read_text())["state"], "complete")
        with self.assertRaises(RuntimeError):
            await pool.get("main")

    async def test_accepted_delete_is_pending_until_confirmed(self):
        self.accept_only = True
        pool = self.pool(recovery=False)
        await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.release_all()
        self.assertFalse(pool.cleanup["ok"])
        self.assertEqual(len(pool._pending_delete_ids), 2)
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
        self.assertTrue(pool._provider.closed)
        self.assertTrue(pool.cleanup["deferred"])
        self.fail_delete = False
        result = await pool._deferred_cleanup
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
        result = await pool._deferred_cleanup
        self.assertFalse(result["ok"])
        data = json.loads(pool._journal.path.read_text())
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
        self.assertFalse(pool._provider.closed)
        self.fail_delete = False
        await pool.release_all()
        self.assertFalse(self.live)
        await pool.aclose()
        self.assertTrue(pool._provider.closed)

    async def test_query_error_does_not_forget_accepted_id(self):
        pool = self.pool(recovery=False)
        await pool.get("main")
        self.accept_only = self.get_error = True
        with self.assertRaises(CleanupError):
            await pool.release_all()
        self.assertEqual(len(pool._pending_verification_ids), 2)
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
        await pool._deferred_cleanup
        self.assertFalse(self.live)

    async def test_partial_create_failure_records_unknown_and_releases_dependency(self):
        pool = self.pool()
        self.fail_create = "main"
        with self.assertRaises(HTTPError):
            await pool.get("main")
        with self.assertRaises(CleanupError):
            await pool.aclose()
        self.assertFalse(self.live)
        data = json.loads(pool._journal.path.read_text())
        self.assertTrue(data["resources"]["dep-1"]["confirmed"])
        self.assertEqual(data["ambiguous_creates"], 1)
        self.assertEqual(sum(method == "POST" for method, _ in self.calls), 2)

    async def test_shared_provider_keeps_foreign_resource_and_transport(self):
        pool = self.pool()
        pool._close_provider = False
        pool._provider.owned_env_ids.add("foreign")
        self.live.add("foreign")
        await pool.get("main")
        await pool.aclose()
        self.assertEqual(self.live, {"foreign"})
        self.assertFalse(pool._provider.closed)

    async def test_cancelled_journal_write_finishes_before_unlock(self):
        journal = cleanup.CleanupJournal(self.directory, "gateway-test")
        entered, release = threading.Event(), threading.Event()
        original = journal.created

        def slow(env_id):
            entered.set()
            release.wait(1)
            original(env_id)

        journal.created = slow
        task = asyncio.create_task(journal.write("created", "owned"))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        result = await cleanup.reconcile(self.directory, self.provider, 0.2)
        self.assertEqual(result["results"][0]["state"], "active")
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await journal.write("detach")
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
        self.assertTrue(pool._journal.lock.closed)
        self.fail_delete = False
        await pool._deferred_cleanup
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
        pool._journal_dir = None
        pool._max_attempts = 3
        calls = 0

        async def timeout(**kwargs):
            nonlocal calls
            calls += 1
            raise TimeoutError("response lost")

        pool._provider.create = timeout
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

        pool._provider.create = lost_response
        with self.assertRaises(ConnectionError):
            await pool.get("main")
        self.assertEqual(calls, 1)
        await pool.aclose()

    async def _supervised_driver(self):
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
            "XTUNER_SANDBOX_CLEANUP_INTERVAL": "0.1",
            "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "3",
        }
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import cleanup
cleanup.start_cleanup_supervisor()
cleanup.start_cleanup_supervisor()
j = cleanup.CleanupJournal(sys.argv[2], 'gateway-test')
j.created('active-driver')
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

    async def test_supervisor_recovers_driver_sigkill_automatically(self):
        driver = await self._supervised_driver()
        try:
            await asyncio.sleep(0.5)
            self.assertIn("active-driver", self.live)
            driver.kill()
            await driver.wait()
            await self._wait_deleted("active-driver")
            self.assertEqual(driver.returncode, -9)
        finally:
            if driver.returncode is None:
                driver.kill()
                await driver.wait()
            await asyncio.sleep(0.2)

    async def test_supervisor_recovers_orphan_while_driver_and_trial_live(self):
        driver = await self._supervised_driver()
        try:
            journal = cleanup.CleanupJournal(self.directory, "gateway-test")
            journal.created("dead-worker")
            journal.detach()
            self.live.add("dead-worker")
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

    def test_supervisor_rejects_incomplete_configuration(self):
        with patch.dict(os.environ, {"XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory)}):
            os.environ.pop("XTUNER_SANDBOX_CLEANUP_FACTORY", None)
            with self.assertRaises(ValueError):
                cleanup.start_cleanup_supervisor()

    def test_both_trainer_fit_paths_recover_and_preserve_original_outcome(self):
        source = ROOT / "xtuner/v1/train/rl_trainer.py"
        classes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef)]
        for cls in classes:
            if cls.name not in {"RLColocateTrainer", "RLDisaggregatedTrainer"}:
                continue
            fit = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
            namespace = {}
            exec(compile(ast.Module(body=[fit], type_ignores=[]), str(source), "exec"), namespace)
            for failure in (None, ValueError("training failed"), KeyboardInterrupt()):
                events = []

                def run():
                    events.append("fit")
                    if failure is not None:
                        raise failure
                    return "trained"

                def recover():
                    events.append("recover")
                    raise RuntimeError("cleanup unavailable")

                trainer = types.SimpleNamespace(_fit_with_cleanup=run, logger=logging.getLogger("test"))
                with patch.object(cleanup, "cleanup_from_environment", recover):
                    if failure is None:
                        self.assertEqual(namespace["fit"](trainer), "trained")
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
                },
            ):
                namespace[name](object(), object(), resources, 256)
        self.assertEqual(len(options), 2)
        for option in options:
            self.assertEqual(
                option["runtime_env"]["env_vars"],
                {
                    "XTUNER_SANDBOX_CLEANUP_DIR": "/shared/run-1",
                    "XTUNER_SANDBOX_PROVIDER_KEY": "backend-a",
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
        create = pool._provider.create

        async def capture(**kwargs):
            calls.append(kwargs)
            return await create(**kwargs)

        pool._provider.create = capture
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

    async def test_runner_repeated_cancellation_finishes_cleanup(self):
        pool = self.pool()
        entered, deleting, allow_delete = asyncio.Event(), asyncio.Event(), asyncio.Event()
        delete = pool._provider.delete

        async def blocked_delete(env_id):
            deleting.set()
            await allow_delete.wait()
            await delete(env_id)

        async def block(client, item, record):
            entered.set()
            await asyncio.Event().wait()

        pool._provider.delete = blocked_delete
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
        await pool._deferred_cleanup
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
        await pool._deferred_cleanup
        self.assertEqual(result.status, schemas.RolloutStatus.FAILED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
