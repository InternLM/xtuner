import threading
import time
from types import SimpleNamespace

import pytest

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.rollout.controller import RolloutController, WorkerInfo
from xtuner.v1.rl.rollout.utils import RolloutHealthChecker, SessionRouter


class FakeWorkerActor:
    def __init__(self, url: str = "http://worker", healthy: bool = True):
        self.url = url
        self.healthy = healthy
        self.offload_calls = 0
        self.shutdown_calls = 0
        self.init_dist_port_calls = 0
        self.init_calls = 0
        self.generate_calls = 0

        self.offload = SimpleNamespace(remote=self._offload)
        self.shutdown = SimpleNamespace(remote=self._shutdown)
        self.init_dist_port = SimpleNamespace(remote=self._init_dist_port)
        self.init = SimpleNamespace(remote=self._init)
        self.check_health = SimpleNamespace(remote=self._check_health)
        self.generate = SimpleNamespace(remote=self._generate)

    def _offload(self):
        self.offload_calls += 1
        return True

    def _shutdown(self):
        self.shutdown_calls += 1
        return True

    def _init_dist_port(self):
        self.init_dist_port_calls += 1
        return "127.0.0.1:12345"

    def _init(self, dist_init_addr):
        self.init_calls += 1
        return (0, self.url)

    async def _check_health(self):
        return self.healthy

    async def _generate(self, rollout_state):
        self.generate_calls += 1
        rollout_state.status = Status.SUCCESS
        return rollout_state


@pytest.mark.asyncio
async def test_session_router_sticky_session_routed_to_same_worker():
    w0 = FakeWorkerActor(url="http://w0")
    w1 = FakeWorkerActor(url="http://w1")
    infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=True), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

    router = SessionRouter(infos)

    worker_a = await router.get_worker(100)
    worker_b = await router.get_worker(100)

    assert worker_a is worker_b
    assert worker_a is w0


@pytest.mark.asyncio
async def test_session_router_skips_inactive_worker_and_routes_next_active():
    w0 = FakeWorkerActor(url="http://w0")
    w1 = FakeWorkerActor(url="http://w1")
    infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=False), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

    router = SessionRouter(infos)
    worker = await router.get_worker(200)

    assert worker is w1


@pytest.mark.asyncio
async def test_session_router_reroutes_existing_session_if_previous_worker_inactive():
    w0 = FakeWorkerActor(url="http://w0")
    w1 = FakeWorkerActor(url="http://w1")
    infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=True), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

    router = SessionRouter(infos)
    worker_first = await router.get_worker(300)
    infos[0].is_active = False
    worker_second = await router.get_worker(300)

    assert worker_first is w0
    assert worker_second is w1


def test_rollout_health_checker_periodically_runs_health_check(monkeypatch):
    w0 = FakeWorkerActor(url="http://w0")
    infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=True)}
    config = SimpleNamespace(
        health_check_interval_seconds=0.05,
        health_check_first_wait_seconds=0.0,
        health_check_failure_threshold=1,
    )

    call_counter = {"count": 0}

    async def _fake_check_worker_health(*args, **kwargs):
        call_counter["count"] += 1
        return True

    monkeypatch.setattr("xtuner.v1.rl.rollout.utils.check_worker_health", _fake_check_worker_health)

    checker = RolloutHealthChecker(config=config, workers_info=infos)
    checker.start()
    checker.resume()
    time.sleep(0.18)
    checker.stop()

    assert call_counter["count"] >= 2


def test_rollout_health_checker_offloads_and_shutdowns_unhealthy_worker(monkeypatch):
    w0 = FakeWorkerActor(url="http://w0")
    w1 = FakeWorkerActor(url="http://w1")
    infos = {
        0: WorkerInfo(actor=w0, url=w0.url, is_active=True),
        1: WorkerInfo(actor=w1, url=w1.url, is_active=True),
    }
    config = SimpleNamespace(
        health_check_interval_seconds=1,
        health_check_first_wait_seconds=0,
        health_check_failure_threshold=1,
    )

    async def _fake_check_worker_health(actor, rank, url, is_active, failure_threshold=1):
        return rank != 0

    monkeypatch.setattr("xtuner.v1.rl.rollout.utils.check_worker_health", _fake_check_worker_health)
    monkeypatch.setattr("xtuner.v1.rl.rollout.utils.ray.get", lambda x, timeout=None: x)

    checker = RolloutHealthChecker(config=config, workers_info=infos)
    checker.run_once()

    assert infos[0].is_active is False
    assert w0.offload_calls == 1
    assert w0.shutdown_calls == 1
    assert infos[1].is_active is True


@pytest.mark.asyncio
async def test_controller_recover_keeps_url_and_worker_can_generate(monkeypatch):
    controller = RolloutController.__new__(RolloutController)
    controller.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    controller.worker_info_lock = threading.RLock()
    controller.config = SimpleNamespace(rollout_timeout=1)
    controller.timeout_multiplier = 1.0

    actor = FakeWorkerActor(url="http://same-url")
    controller.rank2info = {0: WorkerInfo(actor=actor, url="http://same-url", is_active=False)}
    controller.router = SessionRouter(controller.rank2info, worker_infos_lock=controller.worker_info_lock)

    monkeypatch.setattr("xtuner.v1.rl.rollout.controller.ray.get", lambda x, timeout=None: x)

    controller.recover()
    assert controller.rank2info[0].is_active is True

    rollout_state = SimpleNamespace(session_uid=123, status=None, error_msg="")
    out = await controller.generate(rollout_state)

    assert out.status == Status.SUCCESS
    assert actor.generate_calls == 1
