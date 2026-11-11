import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks, retain_pending_tasks
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt


def remote(fn):
    return SimpleNamespace(remote=fn)


class PendingTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_task_identity_survives_two_pauses_claimed_once(self):
        pending = _PendingTasks()
        finish = asyncio.Event()
        task = None
        claimed = []

        async def episode():
            await finish.wait()
            return "original-agent-result"

        async def spawn():
            nonlocal task
            task = asyncio.create_task(episode())
            return task

        await pending.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn)
        original = task

        async def put(done):
            claimed.append(done.result())

        for _ in range(2):
            await retain_pending_tasks(pending_tasks=pending, ctx=None, put_claimed_task=put)
            self.assertEqual(pending.count(), 1)
            self.assertIs(task, original)
            self.assertFalse(task.cancelled())
        finish.set()
        await task
        await asyncio.gather(
            *[retain_pending_tasks(pending_tasks=pending, ctx=None, put_claimed_task=put) for _ in range(2)]
        )
        self.assertEqual(claimed, ["original-agent-result"])
        self.assertEqual(pending.count(), 0)

    async def test_pause_failure_never_aborts_or_resumes_backend(self):
        events = []

        async def fail(**kwargs):
            raise RuntimeError("unsettled")

        async def abort():
            events.append("abort")

        ctl = SimpleNamespace(prepare_pause=remote(fail), pause_generation=remote(abort))
        policy = AgentTurnPausePolicy(freeze_timeout=1, drain_timeout=1)
        with self.assertRaises(RuntimeError):
            await policy.pause(ctl)
        self.assertEqual(events, [])

    async def test_publication_owns_backend_resume_and_keeps_barrier_on_failure(self):
        events = []

        async def publish(barrier, step, **kwargs):
            events.append((barrier, step, kwargs))
            raise TimeoutError("publication")

        async def resume():
            events.append("unexpected separate backend resume")

        ctl = SimpleNamespace(resume_requests=remote(publish), continue_generation=remote(resume))
        policy = AgentTurnPausePolicy(freeze_timeout=2, drain_timeout=3)
        policy.barrier = {"epoch": 1}
        with self.assertRaises(TimeoutError):
            await policy.resume(ctl, 4)
        self.assertEqual(events, [({"epoch": 1}, 4, {"timeout": 2})])
        self.assertEqual(policy.barrier, {"epoch": 1})

    async def test_delayed_install_keeps_real_gates_closed_and_backend_paused(self):
        loop = asyncio.get_running_loop()
        gates = [RequestGate(wait_timeout=2, initially_open=False) for _ in range(2)]
        for gate in gates:
            await gate.seal(1)
        installing = asyncio.Event()
        release = asyncio.Event()
        resumed = []

        def async_remote(fn):
            return remote(lambda *args: asyncio.run_coroutine_threadsafe(fn(*args), loop))

        async def delayed_install(epoch, step):
            installing.set()
            await release.wait()
            await gates[1].install_version(epoch, step)

        async def backend_resume(index):
            # Every entrypoint must have the new version before any backend resumes.
            self.assertEqual([g.snapshot()["model_step"] for g in gates], [4, 4])
            resumed.append(index)

        async def admit(gate):
            async with gate.request("retained-agent") as lease:
                self.assertEqual(sorted(resumed), [0, 1])
                self.assertEqual(lease.model_step, 4)
                lease.settlement = SettlementReceipt(True, True, "completed")

        servers = [
            SimpleNamespace(
                install_request_version=async_remote(gates[0].install_version),
                reopen_requests=async_remote(gates[0].reopen),
            ),
            SimpleNamespace(
                install_request_version=async_remote(delayed_install),
                reopen_requests=async_remote(gates[1].reopen),
            ),
        ]
        entries = [
            SimpleNamespace(
                session_url=f"url{i}",
                actor=SimpleNamespace(
                    continue_generation=remote(lambda i=i: asyncio.run_coroutine_threadsafe(backend_resume(i), loop))
                ),
            )
            for i in range(2)
        ]
        barrier = dict(epoch=1, endpoints=("url0", "url1"))
        ctl = RolloutController.__new__(RolloutController)
        ctl._request_barrier = (barrier, servers)
        ctl.registry = SimpleNamespace(
            active_entrypoints=lambda: entries, get_target_state_workers=lambda state: entries
        )
        ctl.health_manager = SimpleNamespace(resume=lambda: None)

        def get(refs, *, timeout):
            return [ref.result(timeout=timeout) for ref in refs]

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=get):
            publication = asyncio.create_task(asyncio.to_thread(ctl.resume_requests, barrier, 4, timeout=2))
            waiters = [asyncio.create_task(admit(gate)) for gate in gates]
            try:
                await asyncio.wait_for(installing.wait(), timeout=2)
                self.assertEqual(resumed, [])
                self.assertTrue(all(not gate.snapshot()["open"] for gate in gates))
                self.assertTrue(all(not waiter.done() for waiter in waiters))
            finally:
                release.set()
                await publication
                await asyncio.gather(*waiters)
        self.assertIsNone(ctl._request_barrier)


class ControllerTests(unittest.TestCase):
    def make_controller(self, *, fail_seal=False, fail_install=False, fail_resume=False):
        events = []
        controller = RolloutController.__new__(RolloutController)
        controller._request_barrier = None
        controller._request_epoch = 0
        controller.health_manager = SimpleNamespace(
            pause=lambda: events.append("freeze"),
            wait_recovery_done=lambda timeout: True,
            resume=lambda: events.append("health_resume"),
        )

        def call(name, index):
            def invoke(*args, **kwargs):
                events.append((name, index, args))
                if name == "seal" and index == 1 and fail_seal:
                    raise RuntimeError("seal_failed")
                if name == "install" and index == 1 and fail_install:
                    raise RuntimeError("install_failed")
                if name == "resume" and index == 1 and fail_resume:
                    return False

            return remote(invoke)

        servers = [
            SimpleNamespace(
                seal_requests=call("seal", i),
                drain_requests=call("drain", i),
                install_request_version=call("install", i),
                reopen_requests=call("open", i),
            )
            for i in range(2)
        ]
        entries = [
            SimpleNamespace(
                session_url=f"url{i}",
                actor=SimpleNamespace(
                    get_session_server=remote(lambda i=i: servers[i]), continue_generation=call("resume", i)
                ),
            )
            for i in range(2)
        ]
        controller.registry = SimpleNamespace(
            active_entrypoints=lambda: entries, get_target_state_workers=lambda state: entries
        )
        return controller, events, entries

    def test_every_entrypoint_sealed_before_any_drain_all_versions_before_open(self):
        ctl, events, _ = self.make_controller()
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
            b = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
            ctl.resume_requests(b, 3)
        self.assertEqual(events[0], "freeze")
        self.assertEqual(
            [e[0] for e in events[1:-1]],
            ["seal", "seal", "drain", "drain", "install", "install", "resume", "resume", "open", "open"],
        )
        self.assertEqual(events[-1], "health_resume")

    def test_failed_seal_or_version_install_keeps_entries_closed(self):
        for failure in ("seal", "install", "resume"):
            ctl, events, _ = self.make_controller(
                fail_seal=failure == "seal", fail_install=failure == "install", fail_resume=failure == "resume"
            )
            with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
                with self.assertRaises(RuntimeError):
                    b = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
                    ctl.resume_requests(b, 3)
            self.assertFalse(any(isinstance(e, tuple) and e[0] == "open" for e in events))
            self.assertNotIn("health_resume", events)

    def test_one_publication_deadline_prevents_reopen_after_budget_exhaustion(self):
        ctl, events, _ = self.make_controller()
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
            barrier = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
        waits = []

        def get(refs, *, timeout):
            waits.append(timeout)
            return refs

        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=get),
            patch("xtuner.v1.rl.rollout.controller.monotonic", side_effect=[0, 1, 3, 6]),
        ):
            with self.assertRaisesRegex(TimeoutError, "before reopen_requests"):
                ctl.resume_requests(barrier, 3, timeout=5)
        self.assertEqual(waits, [4, 2])
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "open" for e in events))
        self.assertIsNotNone(ctl._request_barrier)
        self.assertNotIn("health_resume", events)

    def test_a_poisoned_gate_stops_consumption_before_waiting_for_a_batch(self):
        ctl = RolloutController.__new__(RolloutController)
        server = SimpleNamespace(request_status=remote(lambda: {"failures": {"request": "backend_unsettled"}}))
        ctl.registry = SimpleNamespace(
            active_entrypoints=lambda: [
                SimpleNamespace(actor=SimpleNamespace(get_session_server=remote(lambda: server)))
            ]
        )
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
            with self.assertRaisesRegex(RuntimeError, "Unsettled Agent requests"):
                ctl.assert_request_gates_healthy()

    def test_changed_topology_prevents_publication(self):
        ctl, events, entries = self.make_controller()
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
            b = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
            entries.pop()
            with self.assertRaises(RuntimeError):
                ctl.resume_requests(b, 3)
        self.assertFalse(any(isinstance(e, tuple) and e[0] == "install" for e in events))
