"""RolloutController 的暂停/恢复屏障公开契约测试。

Good Tests:
- 通过 pause_generation/continue_generation/assert_request_gates_healthy 的 public 入口验证控制平面顺序和失败语义。
- 使用真实 registry、health manager、配置与控制方法，仅替代外部 GPU actor / Ray 传输。

Bad Tests:
- 不断言私有 barrier/epoch 字段。
- 不测试 SessionServer 内部的 RequestGate 状态机（见 test_request_gate.py）。
"""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.rl.rollout.controller import RolloutController, RolloutRequestState
from xtuner.v1.rl.rollout.cross_weight_update_rollout import CrossWeightUpdateRolloutConfig
from xtuner.v1.rl.rollout.health_manager import RolloutHealthManager
from xtuner.v1.rl.rollout.rollout_topology import RolloutEngine, RolloutServerProcess, RolloutTopology
from xtuner.v1.rl.rollout.worker import RolloutWorkerInitResult
from xtuner.v1.rl.rollout.worker_registry import RolloutWorkerRegistry


def _remote(value=None, calls=None, label=None):
    class _Remote:
        def remote(self, *args, **kwargs):
            if calls is not None:
                calls.append((label, args))
            return value

    return _Remote()


class _FakeSessionServer:
    def __init__(self, url, calls, *, status=None, outcome=True):
        self.url = url
        self.status = status or {"failures": {}}
        self.seal_requests = _remote(None, calls, f"seal:{url}")
        self.drain_requests = _remote(None, calls, f"drain:{url}")
        self.install_request_version = _remote(None, calls, f"install:{url}")
        self.reopen_requests = _remote(None, calls, f"reopen:{url}")
        self.request_status = _remote(self.status, calls, f"status:{url}")


class _FakeWorker:
    def __init__(self, url, calls, *, server=None, pause_result=True, continue_result=True):
        self.session_url = url
        self.actor = SimpleNamespace(
            get_session_server=_remote(server, calls, f"get_server:{url}"),
            pause_generation=_remote(pause_result, calls, f"pause_backend:{url}"),
            continue_generation=_remote(continue_result, calls, f"continue_backend:{url}"),
        )


class TestRolloutControllerRequestBarrier(unittest.TestCase):
    def _build(self, worker_count=2, *, pause_result=True, continue_result=True):
        calls: list[tuple[str, tuple]] = []
        servers = [_FakeSessionServer(f"http://s{i}", calls) for i in range(worker_count)]
        workers = [
            _FakeWorker(
                f"http://s{i}",
                calls,
                server=servers[i],
                pause_result=pause_result,
                continue_result=continue_result,
            )
            for i in range(worker_count)
        ]
        settings = CrossWeightUpdateRolloutConfig(wait_timeout=10, freeze_timeout=5, drain_timeout=10)
        topology = RolloutTopology(
            engines=tuple(
                RolloutEngine(
                    engine_ranks=(rank,),
                    dist_init_addr=f"addr{rank}",
                    server_processes=(RolloutServerProcess(rank, (rank,), (rank,)),),
                )
                for rank in range(worker_count)
            )
        )
        registry = RolloutWorkerRegistry(rollout_topology=topology)
        registry.register_started_servers(
            init_results=tuple(
                RolloutWorkerInitResult(rank, worker.session_url, worker.session_url)
                for rank, worker in enumerate(workers)
            ),
            workers_by_rank=tuple(worker.actor for worker in workers),
        )
        config = SimpleNamespace(
            cross_weight_update_rollout=settings,
            health_check_interval_seconds=1,
            health_check_timeout_seconds=1,
            health_check_failure_threshold=0,
        )
        controller = RolloutController.__new__(RolloutController)
        controller.config = config
        controller.registry = registry
        controller.health_manager = RolloutHealthManager(config, registry)
        controller.logger = logging.getLogger(__name__)
        controller._request_state = RolloutRequestState()
        return controller, calls, workers

    def test_pause_seals_every_server_before_pausing_backends(self):
        controller, calls, _ = self._build(worker_count=2)

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
            controller.pause_generation()

        labels = [label for label, _ in calls]
        self.assertEqual(labels[0], "get_server:http://s0")
        self.assertEqual(
            [label for label in labels if label.startswith(("seal:", "drain:", "pause_backend:"))],
            [
                "seal:http://s0",
                "seal:http://s1",
                "drain:http://s0",
                "drain:http://s1",
                "pause_backend:http://s0",
                "pause_backend:http://s1",
            ],
        )

    def test_pause_rejects_entrypoint_without_session_server(self):
        controller, calls, workers = self._build(worker_count=1)
        # The single entrypoint lost its SessionServer.
        workers[0].actor.get_session_server = _remote(None, calls, "get_server:missing")

        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "must own a SessionServer"),
        ):
            controller.pause_generation()

    def test_continue_installs_all_versions_before_reopening(self):
        controller, calls, _ = self._build(worker_count=2)
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
            controller.pause_generation()
            calls.clear()
            controller.continue_generation(7)
            controller.continue_generation(7)  # Same open version: no extra barrier is required.

        labels = [label for label, _ in calls]
        self.assertEqual(
            labels,
            [
                "install:http://s0",
                "install:http://s1",
                "continue_backend:http://s0",
                "continue_backend:http://s1",
                "reopen:http://s0",
                "reopen:http://s1",
            ],
        )
        self.assertIn(("install:http://s0", (1, 7)), calls)

    def test_continue_without_barrier_rejects_a_different_version(self):
        controller, _, _ = self._build(worker_count=1)
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
            controller.pause_generation()
            controller.continue_generation(3)
            with self.assertRaisesRegex(RuntimeError, "requires a pause barrier"):
                controller.continue_generation(4)

    def test_failed_backend_resume_does_not_reopen_or_clear_barrier(self):
        controller, calls, _ = self._build(worker_count=2, continue_result=False)
        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Backend resume failed"),
        ):
            controller.pause_generation()
            calls.clear()
            controller.continue_generation(1)

        labels = [label for label, _ in calls]
        self.assertNotIn("reopen:http://s0", labels)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            controller.pause_generation()

    def test_pause_fails_when_a_backend_refuses_to_pause(self):
        controller, _, _ = self._build(worker_count=1, pause_result=False)
        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Backend pause failed"),
        ):
            controller.pause_generation()

    def test_drain_timeout_or_unsafe_settlement_stops_before_backend_pause(self):
        for error in (TimeoutError("drain timed out"), RuntimeError("Unsettled requests")):
            with self.subTest(error=type(error).__name__):
                controller, calls, workers = self._build(worker_count=1)
                server = workers[0].actor.get_session_server.remote()
                server.drain_requests = _remote(error, calls, "drain:http://s0")
                calls.clear()

                def resolve(refs, timeout=None):
                    for ref in refs:
                        if isinstance(ref, Exception):
                            raise ref
                    return refs

                with (
                    patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=resolve),
                    self.assertRaises(type(error)),
                ):
                    controller.pause_generation()
                self.assertFalse(any(label.startswith("pause_backend:") for label, _ in calls))
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    controller.pause_generation()

    def test_topology_change_under_barrier_is_rejected(self):
        controller, calls, workers = self._build(worker_count=2)
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
            controller.pause_generation()
            controller.registry.register_started_servers(
                init_results=(RolloutWorkerInitResult(0, "http://s0", "http://changed"),),
                workers_by_rank=(workers[0].actor,),
            )
            with self.assertRaisesRegex(RuntimeError, "topology changed"):
                controller.continue_generation(1)

    def test_assert_healthy_reports_unsettled_requests(self):
        controller, _, workers = self._build(worker_count=2)
        workers[0].actor.get_session_server = _remote(
            _FakeSessionServer("http://s0", [], status={"failures": {"req": "unsettled"}}), [], None
        )
        workers[1].actor.get_session_server = _remote(
            _FakeSessionServer("http://s1", [], status={"failures": {}}), [], None
        )
        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Unsettled requests"),
        ):
            controller.assert_request_gates_healthy()

    def test_legacy_pause_and_continue_do_not_use_request_gates(self):
        controller, calls, _ = self._build(worker_count=1, pause_result=False, continue_result=False)
        controller.config.cross_weight_update_rollout = None
        with (
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertLogs(controller.logger, level="WARNING") as logs,
        ):
            controller.pause_generation()
            controller.continue_generation(model_step=0)
        self.assertIn("Abort request failed", logs.output[0])
        self.assertEqual([label for label, _ in calls], ["pause_backend:http://s0", "continue_backend:http://s0"])
        with self.assertRaisesRegex(RuntimeError, "not enabled"):
            controller.assert_request_gates_healthy()

    def test_continue_requires_model_step_and_accepts_initial_version_zero(self):
        controller, calls, _ = self._build(worker_count=1)
        with self.assertRaises(TypeError):
            controller.continue_generation()
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
            controller.pause_generation()
            controller.continue_generation(model_step=0)
        self.assertIn(("install:http://s0", (1, 0)), calls)

    def test_resume_phases_share_the_control_deadline(self):
        controller, _, _ = self._build(worker_count=1)
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs) as ray_get:
            controller.pause_generation()
            ray_get.reset_mock()
            with patch("xtuner.v1.rl.rollout.controller.monotonic", side_effect=[0, 1, 2, 3]):
                controller.continue_generation(model_step=1)
        self.assertEqual([call.kwargs["timeout"] for call in ray_get.call_args_list], [4, 3, 2])

    def test_expired_control_deadline_does_not_dispatch_the_next_phase(self):
        # Expiration before either backend resume or gate reopen must stop dispatch, not just shorten ray.get.
        for clock, blocked_prefix in [([0, 1, 6], "continue_backend:"), ([0, 1, 2, 6], "reopen:")]:
            with self.subTest(blocked_phase=blocked_prefix):
                controller, calls, _ = self._build(worker_count=1)
                with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=lambda refs, timeout=None: refs):
                    controller.pause_generation()
                    calls.clear()
                    with (
                        patch("xtuner.v1.rl.rollout.controller.monotonic", side_effect=clock),
                        self.assertRaises(TimeoutError),
                    ):
                        controller.continue_generation(model_step=1)
                    self.assertFalse(any(label.startswith(blocked_prefix) for label, _ in calls))
                    with self.assertRaisesRegex(RuntimeError, "already exists"):
                        controller.pause_generation()


if __name__ == "__main__":
    unittest.main()
