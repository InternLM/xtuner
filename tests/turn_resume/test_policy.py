import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks, retain_pending_tasks
from xtuner.v1.rl.rollout.controller import RolloutController


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


class ControllerTests(unittest.TestCase):
    def make_controller(self, *, fail_seal=False, fail_install=False):
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
                session_url=f"url{i}", actor=SimpleNamespace(get_session_server=remote(lambda i=i: servers[i]))
            )
            for i in range(2)
        ]
        controller.registry = SimpleNamespace(active_entrypoints=lambda: entries)
        return controller, events, entries

    def test_every_entrypoint_sealed_before_any_drain_all_versions_before_open(self):
        ctl, events, _ = self.make_controller()
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
            b = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
            ctl.resume_requests(b, 3)
        self.assertEqual(events[0], "freeze")
        self.assertEqual(
            [e[0] for e in events[1:-1]], ["seal", "seal", "drain", "drain", "install", "install", "open", "open"]
        )
        self.assertEqual(events[-1], "health_resume")

    def test_failed_seal_or_version_install_keeps_entries_closed(self):
        for failure in ("seal", "install"):
            ctl, events, _ = self.make_controller(fail_seal=failure == "seal", fail_install=failure == "install")
            with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda x, **kwargs: x):
                with self.assertRaises(RuntimeError):
                    b = ctl.prepare_pause(freeze_timeout=1, drain_timeout=2)
                    ctl.resume_requests(b, 3)
            self.assertFalse(any(isinstance(e, tuple) and e[0] == "open" for e in events))
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
