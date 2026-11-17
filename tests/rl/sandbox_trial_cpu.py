"""Real exec/TERM/KILL tests; optional training imports alone are bypassed."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import signal
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sandbox_cleanup_cpu as runtime
from sandbox_trial_fixture import Pipeline


process = importlib.import_module(runtime.PACKAGE + ".trial_process")


class TestTrialProcess(runtime.TestSandboxCleanup):
    # Do not inherit/rerun the entire existing suite in this isolated entry.
    __unittest_skip__ = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bootstrap = self.directory / "bootstrap"
        self.bootstrap.mkdir()
        (self.bootstrap / "sitecustomize.py").write_text("import sandbox_cleanup_cpu\n")
        sys.path.insert(0, str(self.bootstrap))
        self.addCleanup(sys.path.remove, str(self.bootstrap))
        environment = {
            "XTUNER_SANDBOX_CLEANUP_DIR": str(self.directory),
            "XTUNER_SANDBOX_PROVIDER_KEY": "trial-test",
            "XTUNER_SANDBOX_CLEANUP_FACTORY": "sandbox_trial_fixture:provider",
            "TRIAL_TEST_PORT": str(self.port),
        }
        self.environment = patch.dict(os.environ, environment)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.limits = process.TrialProcessLimits(execution=10, stop=0.3, recovery=5)

    def item(self, uid=1, **metadata):
        return runtime.schemas.AgentRolloutItem(
            id="trial-" + str(uid),
            uid=uid,
            data_source="test",
            instruction="instruction.md",
            task_root=self.directory,
            pipeline={"type": Pipeline},
            metadata={"port": self.port, "marker": str(self.directory / (str(uid) + ".pid")), **metadata},
        )

    async def wait_started(self, item):
        marker = Path(item.metadata["marker"])
        async with asyncio.timeout(8):
            while not marker.exists():
                await asyncio.sleep(0.02)
        return int(marker.read_text())

    async def test_process_infer_verify_result_and_delete_confirmation(self):
        item = self.item()
        result = await process.run_trial_process(item, limits=self.limits)
        self.assertEqual(result.status, runtime.schemas.RolloutStatus.COMPLETED)
        self.assertEqual(result.reward, 1)
        self.assertNotEqual(result.artifacts["infer_pid"], os.getpid())
        self.assertEqual(result.artifacts["infer_pid"], result.artifacts["verify_pid"])
        self.assertEqual(result.artifacts["bytes"], b"unchanged\x00artifact")
        self.assertEqual(result.uid, item.uid)
        self.assertEqual(item.status, runtime.schemas.RolloutStatus.PENDING)
        self.assertFalse(self.live)
        self.assertEqual(len([call for call in self.calls if call[0] == "DELETE"]), 2)

    async def test_process_crash_recovers_journal_after_owner_exit(self):
        with self.assertRaisesRegex(RuntimeError, "71"):
            await process.run_trial_process(self.item(crash=True), limits=self.limits)
        self.assertFalse(self.live)
        self.assertEqual(len([call for call in self.calls if call[0] == "DELETE"]), 1)

    async def test_process_real_cancel_kills_blocked_trial_and_recovers(self):
        item = self.item(block=True)
        task = asyncio.create_task(process.run_trial_process(item, limits=self.limits))
        pid = await self.wait_started(item)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse(self.live)

    async def test_process_cancel_resistant_trial_cannot_hold_actor(self):
        item = self.item(cancel_resistant=True)
        task = asyncio.create_task(process.run_trial_process(item, limits=self.limits))
        await self.wait_started(item)
        started = time.monotonic()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=8)
        self.assertLess(time.monotonic() - started, 7)
        self.assertFalse(self.live)

    async def test_process_cancel_does_not_delete_concurrent_trial(self):
        blocked, healthy = self.item(uid=1, block=True), self.item(uid=2, slow=True)
        failed = asyncio.create_task(process.run_trial_process(blocked, limits=self.limits))
        sibling = asyncio.create_task(process.run_trial_process(healthy, limits=self.limits))
        await self.wait_started(blocked)
        await self.wait_started(healthy)
        failed.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await failed
        result = await sibling
        self.assertEqual(result.status, runtime.schemas.RolloutStatus.COMPLETED)
        self.assertFalse(self.live)

    async def test_process_timeout_kills_and_recovers(self):
        limits = process.TrialProcessLimits(execution=1.5, stop=0.2, recovery=5)
        with self.assertRaises(TimeoutError):
            await process.run_trial_process(self.item(block=True), limits=limits)
        self.assertFalse(self.live)

    async def test_process_missing_durable_cleanup_rejects_before_allocation(self):
        with patch.dict(os.environ, {"XTUNER_SANDBOX_CLEANUP_FACTORY": ""}):
            with self.assertRaises(RuntimeError):
                await process.run_trial_process(self.item(), limits=self.limits)
        self.assertFalse(self.calls)

    async def test_process_business_error_survives_failed_recovery(self):
        self.fail_delete = True
        limits = process.TrialProcessLimits(execution=10, stop=0.3, recovery=3)
        settings = {
            "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "1",
            "XTUNER_SANDBOX_RECOVERY_BUDGET": "2",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "1",
        }
        try:
            with patch.dict(os.environ, settings):
                result = await process.run_trial_process(self.item(business_error=True), limits=limits)
            self.assertEqual(result.status, runtime.schemas.RolloutStatus.FAILED)
            self.assertEqual(result.error.category, "business_error")
            self.assertFalse(result.artifacts["trial_process"]["cleanup"]["ok"])
            self.assertTrue(list((self.directory / "recovery-requests").glob("*.json")))
        finally:
            self.fail_delete = False
            for journal in self.directory.glob("*.json"):
                if len(journal.stem) == 32:
                    await runtime.cleanup.reconcile_file(journal, lambda _: runtime.Provider(self.port), 2)

    async def test_process_live_runner_is_rejected_without_allocation(self):
        item = self.item()
        item.pipeline = Pipeline()
        with self.assertRaisesRegex(TypeError, "never a live Runner"):
            await process.run_trial_process(item, limits=self.limits)
        self.assertFalse(self.calls)

    async def test_process_live_provider_in_config_or_overrides_is_rejected(self):
        for field in ("pipeline", "pipeline_overrides"):
            item = self.item()
            configuration = getattr(item, field)
            configuration["pool"] = {"provider": runtime.Provider(self.port)}
            with self.assertRaisesRegex(TypeError, "live sandbox provider"):
                await process.run_trial_process(item, limits=self.limits)
        self.assertFalse(self.calls)

    async def test_process_cancel_during_spawn_reaps_late_child(self):
        original = asyncio.create_subprocess_exec
        spawned, release = asyncio.Event(), asyncio.Event()
        children = []

        async def delayed(*args, **kwargs):
            child = await original(*args, **kwargs)
            children.append(child)
            spawned.set()
            await release.wait()
            return child

        with patch.object(asyncio, "create_subprocess_exec", delayed):
            task = asyncio.create_task(process.run_trial_process(self.item(block=True), limits=self.limits))
            await asyncio.wait_for(spawned.wait(), timeout=5)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(children)
        self.assertTrue(all(child.returncode is not None for child in children))
        self.assertFalse(self.live)

    async def test_process_spawn_later_than_stop_budget_still_recovers(self):
        from xtuner.v1.rl.agent_loop.sandbox_agent_loop import finalization

        original_spawn = asyncio.create_subprocess_exec
        original_finalizer = finalization.finalize_bounded
        release = asyncio.Event()
        children = []

        async def delayed(*args, **kwargs):
            child = await original_spawn(*args, **kwargs)
            children.append(child)
            await release.wait()
            return child

        async def short_finalizer(operation, *, timeout_seconds, diagnostic):
            return await original_finalizer(operation, timeout_seconds=0.1, diagnostic=diagnostic)

        item = self.item(block=True)
        with (
            patch.object(asyncio, "create_subprocess_exec", delayed),
            patch.object(finalization, "finalize_bounded", short_finalizer),
        ):
            task = asyncio.create_task(process.run_trial_process(item, limits=self.limits))
            await self.wait_started(item)
            self.assertTrue(self.live)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            async with asyncio.timeout(8):
                while self.live or children[0].returncode is None:
                    await asyncio.sleep(0.02)
        self.assertFalse(self.live)

    async def test_process_parent_death_stops_child_and_releases_journal(self):
        import cloudpickle

        item = self.item(block=True)
        request = self.directory / "parent-request.pkl"
        request.write_bytes(cloudpickle.dumps(item))
        script = self.directory / "owner.py"
        script.write_text(
            "import asyncio, cloudpickle, sys\n"
            "from pathlib import Path\n"
            "from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trial_process import run_trial_process\n"
            "item=cloudpickle.loads(Path(sys.argv[1]).read_bytes())\n"
            "asyncio.run(run_trial_process(item))\n"
        )
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        owner = await asyncio.create_subprocess_exec(sys.executable, str(script), str(request), env=environment)
        try:
            pid = await self.wait_started(item)
            owner.kill()
            await asyncio.wait_for(owner.wait(), timeout=5)
            async with asyncio.timeout(5):
                while True:
                    stat = Path(f"/proc/{pid}/stat")
                    if not stat.exists() or stat.read_text().split()[2] == "Z":
                        break
                    await asyncio.sleep(0.02)
            for journal in self.directory.glob("*.json"):
                if len(journal.stem) == 32:
                    # The orphan leader can already be Z while sibling threads
                    # are still releasing the shared fd table. Recovery must
                    # keep skipping its live lease, then succeed after release.
                    async with asyncio.timeout(5):
                        while True:
                            result = await runtime.cleanup.reconcile_file(
                                journal, lambda _: runtime.Provider(self.port), 2
                            )
                            if result["ok"]:
                                break
                            self.assertEqual(
                                result.get("state"),
                                "active",
                                json.dumps({"result": result, "journal": journal.read_text()}),
                            )
                            await asyncio.sleep(0.02)
            self.assertFalse(self.live)
            staging = Path(Path(item.metadata["marker"]).with_suffix(".staging").read_text())
            shutil.rmtree(staging)
        finally:
            if owner.returncode is None:
                owner.send_signal(signal.SIGKILL)
                await owner.wait()

    async def test_process_staging_cleanup_cannot_replace_cancellation(self):
        item = self.item(block=True)
        task = asyncio.create_task(process.run_trial_process(item, limits=self.limits))
        await self.wait_started(item)
        original, leftovers = shutil.rmtree, []

        def fail(path):
            leftovers.append(path)
            raise OSError("local staging error")

        with patch.object(process.shutil, "rmtree", side_effect=fail):
            task.cancel()
            with self.assertLogs(process.__name__, level="WARNING"):
                with self.assertRaises(asyncio.CancelledError):
                    await task
        for path in leftovers:
            original(path)
        self.assertFalse(self.live)

    async def test_process_setup_failure_removes_private_request(self):
        original, created = process.tempfile.mkdtemp, []

        def tracked(**kwargs):
            directory = original(**kwargs)
            created.append(Path(directory))
            return directory

        with (
            patch.object(process.tempfile, "mkdtemp", tracked),
            patch.object(process.socket, "socketpair", side_effect=OSError("socket setup failed")),
        ):
            with self.assertRaisesRegex(OSError, "socket setup failed"):
                await process.run_trial_process(self.item(), limits=self.limits)
        self.assertTrue(created)
        self.assertTrue(all(not path.exists() for path in created))
        self.assertFalse(self.calls)

    async def test_process_recovery_parent_death_releases_blocked_provider(self):
        journal = runtime.cleanup.CleanupJournal(self.directory, "trial-test")
        journal.allocated("owned-recovery")
        journal.detach()
        self.live.add("owned-recovery")
        directory = self.directory / "recovery-stage"
        (directory / "journals").mkdir(parents=True)
        (directory / "journals" / journal.path.name).write_text(json.dumps(str(journal.path)))
        started = self.directory / "recovery-started"
        child_pid = self.directory / "recovery-pid"
        script = self.directory / "recovery-owner.py"
        script.write_text(
            "import os,sys,subprocess,time\n"
            "from pathlib import Path\n"
            "child=subprocess.Popen([sys.executable,sys.argv[1],'--recover',sys.argv[2],"
            "'--parent-pid',str(os.getpid()),'--recovery-timeout','60'],start_new_session=True)\n"
            "Path(sys.argv[3]).write_text(str(child.pid))\n"
            "time.sleep(60)\n"
        )
        environment = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(sys.path),
            XTUNER_SANDBOX_CLEANUP_FACTORY="sandbox_trial_fixture:hanging_provider",
            TRIAL_TEST_RECOVERY_STARTED=str(started),
        )
        lock = runtime.cleanup.locking.open_lock(self.directory / "recovery-worker.lock")
        runtime.cleanup.locking.acquire(lock)
        owner = await asyncio.create_subprocess_exec(
            sys.executable, str(script), str(process.__file__), str(directory), str(child_pid), env=environment
        )
        try:
            async with asyncio.timeout(8):
                while not started.exists():
                    await asyncio.sleep(0.02)
            owner.kill()
            await owner.wait()
            async with asyncio.timeout(5):
                while True:
                    result = await runtime.cleanup.reconcile_file(
                        journal.path, lambda _: runtime.Provider(self.port), 2
                    )
                    if result["ok"]:
                        break
                    self.assertEqual(result.get("state"), "active")
                    await asyncio.sleep(0.02)
            self.assertFalse(self.live)
        finally:
            lock.close()
            if owner.returncode is None:
                owner.kill()
                await owner.wait()
            if child_pid.exists():
                try:
                    os.kill(int(child_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already confirmed gone by the parent-death guard.

    def test_process_invalid_limits_and_manifest_scope(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                process.TrialProcessLimits(execution=value)
        with patch.dict(os.environ, {"XTUNER_SANDBOX_TRIAL_MANIFEST": str(self.directory / "manifest")}):
            with self.assertRaises(ValueError):
                process.register_trial_journal(Path("/tmp/other-run/journal.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
