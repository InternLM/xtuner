"""RequestPublication 的暂停/发布顺序测试。

Good Tests:
- 通过 pause/publish/assert_healthy 的 public 入口验证控制平面顺序和失败语义。
- 用真实 TurnResumeConfig 和轻量 actor stub 替代 GPU worker / Ray 传输。

Bad Tests:
- 不断言私有 barrier/epoch 字段。
- 不测试 SessionServer 内部的 RequestGate 状态机（见 test_request_gate.py）。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xtuner.v1.rl.rollout.request_publication import RequestPublication, TurnResumeConfig


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


class _FakeRegistry:
    def __init__(self, workers):
        self._workers = list(workers)

    def active_entrypoints(self):
        return tuple(self._workers)

    def active_workers(self):
        return tuple(self._workers)

    def get_target_state_workers(self, _state):
        return tuple(self._workers)


class _FakeHealthManager:
    def __init__(self, calls):
        self.calls = calls
        self.pause = MagicMock(side_effect=lambda: calls.append(("health_pause", ())))
        self.resume = MagicMock(side_effect=lambda: calls.append(("health_resume", ())))
        self.wait_recovery_done = MagicMock(return_value=True)


class TestRequestPublication(unittest.TestCase):
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
        settings = TurnResumeConfig(wait_timeout=10, freeze_timeout=5, drain_timeout=10, quiesce_timeout=10)
        publication = RequestPublication(_FakeRegistry(workers), _FakeHealthManager(calls), settings)
        return publication, calls, workers

    def test_pause_seals_every_server_before_pausing_backends(self):
        publication, calls, _ = self._build(worker_count=2)

        with patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs):
            publication.pause()

        labels = [label for label, _ in calls]
        self.assertEqual(labels[0], "health_pause")
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
        publication, calls, _ = self._build(worker_count=1)
        # The single entrypoint lost its SessionServer.
        workers = publication._registry._workers
        workers[0].actor.get_session_server = _remote(None, calls, "get_server:missing")

        with (
            patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "must own a SessionServer"),
        ):
            publication.pause()

    def test_publish_installs_all_versions_before_reopening(self):
        publication, calls, _ = self._build(worker_count=2)
        with patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs):
            publication.pause()
            calls.clear()
            publication.publish(7)
            publication.publish(7)  # Same open version: no extra barrier is required.

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
                "health_resume",
            ],
        )
        self.assertIn(("install:http://s0", (1, 7)), calls)

    def test_publish_without_barrier_rejects_a_different_version(self):
        publication, _, _ = self._build(worker_count=1)
        with patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs):
            publication.pause()
            publication.publish(3)
            with self.assertRaisesRegex(RuntimeError, "requires a pause barrier"):
                publication.publish(4)

    def test_failed_backend_resume_does_not_reopen_or_clear_barrier(self):
        publication, calls, _ = self._build(worker_count=2, continue_result=False)
        with (
            patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Backend resume failed"),
        ):
            publication.pause()
            calls.clear()
            publication.publish(1)

        labels = [label for label, _ in calls]
        self.assertNotIn("reopen:http://s0", labels)
        self.assertNotIn("health_resume", labels)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            publication.pause()

    def test_pause_fails_when_a_backend_refuses_to_pause(self):
        publication, _, _ = self._build(worker_count=1, pause_result=False)
        with (
            patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Backend pause failed"),
        ):
            publication.pause()

    def test_topology_change_under_barrier_is_rejected(self):
        publication, calls, workers = self._build(worker_count=2)
        with patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs):
            publication.pause()
            workers[0].session_url = "http://changed"
            with self.assertRaisesRegex(RuntimeError, "topology changed"):
                publication.publish(1)

    def test_assert_healthy_reports_unsettled_requests(self):
        publication, _, workers = self._build(worker_count=2)
        workers[0].actor.get_session_server = _remote(
            _FakeSessionServer("http://s0", [], status={"failures": {"req": "unsettled"}}), [], None
        )
        workers[1].actor.get_session_server = _remote(
            _FakeSessionServer("http://s1", [], status={"failures": {}}), [], None
        )
        with (
            patch("xtuner.v1.rl.rollout.request_publication.ray.get", side_effect=lambda refs, timeout=None: refs),
            self.assertRaisesRegex(RuntimeError, "Unsettled requests"),
        ):
            publication.assert_healthy()


if __name__ == "__main__":
    unittest.main()
