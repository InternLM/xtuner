import threading
import time
import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout.controller import RolloutController, WorkerInfo
from xtuner.v1.rl.rollout.utils import RolloutHealthChecker, SessionRouter


MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH", "")
RESOURCE_MAP = {"npu": "NPU", "cuda": "GPU"}


class FakeWorkerActor:
    def __init__(self, url: str = "http://worker", healthy: bool = True):
        self.url = url
        self.healthy = healthy
        self.offload_calls = 0
        self.shutdown_calls = 0
        self.init_dist_port_calls = 0
        self.init_calls = 0
        self.generate_calls = 0
        self.memory_used = 1024

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
        self.healthy = False
        self.memory_used = 0
        return True

    def _init_dist_port(self):
        self.init_dist_port_calls += 1
        return "127.0.0.1:12345"

    def _init(self, dist_init_addr):
        self.init_calls += 1
        self.healthy = True
        if self.memory_used == 0:
            self.memory_used = 1024
        return (0, self.url)

    def get_gpu_memory_used(self):
        return self.memory_used

    async def _check_health(self):
        return self.healthy

    async def _generate(self, rollout_state):
        self.generate_calls += 1
        rollout_state.status = Status.SUCCESS
        return rollout_state


class TestSessionRouter(unittest.IsolatedAsyncioTestCase):
    async def test_sticky_session_routed_to_same_worker(self):
        w0 = FakeWorkerActor(url="http://w0")
        w1 = FakeWorkerActor(url="http://w1")
        infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=True), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

        router = SessionRouter(infos)

        worker_a = await router.get_worker(100)
        worker_b = await router.get_worker(100)

        self.assertIs(worker_a, worker_b)
        self.assertIs(worker_a, w0)

    async def test_skips_inactive_worker_and_routes_next_active(self):
        w0 = FakeWorkerActor(url="http://w0")
        w1 = FakeWorkerActor(url="http://w1")
        infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=False), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

        router = SessionRouter(infos)
        worker = await router.get_worker(200)

        self.assertIs(worker, w1)

    async def test_reroutes_existing_session_if_previous_worker_inactive(self):
        w0 = FakeWorkerActor(url="http://w0")
        w1 = FakeWorkerActor(url="http://w1")
        infos = {0: WorkerInfo(actor=w0, url=w0.url, is_active=True), 1: WorkerInfo(actor=w1, url=w1.url, is_active=True)}

        router = SessionRouter(infos)
        worker_first = await router.get_worker(300)
        infos[0].is_active = False
        worker_second = await router.get_worker(300)

        self.assertIs(worker_first, w0)
        self.assertIs(worker_second, w1)


class TestRolloutHealthChecker(unittest.TestCase):
    def test_periodically_runs_health_check(self):
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

        with patch("xtuner.v1.rl.rollout.utils.check_worker_health", _fake_check_worker_health):
            checker = RolloutHealthChecker(config=config, workers_info=infos)
            checker.start()
            checker.resume()
            time.sleep(0.18)
            checker.stop()

        self.assertGreaterEqual(call_counter["count"], 2)

    def test_offloads_and_shutdowns_unhealthy_worker(self):
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

        with patch("xtuner.v1.rl.rollout.utils.check_worker_health", _fake_check_worker_health), patch(
            "xtuner.v1.rl.rollout.utils.ray.get", lambda x, timeout=None: x
        ):
            checker = RolloutHealthChecker(config=config, workers_info=infos)
            checker.run_once()

        self.assertFalse(infos[0].is_active)
        self.assertEqual(w0.offload_calls, 1)
        self.assertEqual(w0.shutdown_calls, 1)
        self.assertTrue(infos[1].is_active)


class TestRolloutControllerRecover(unittest.IsolatedAsyncioTestCase):
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    @unittest.skipIf(not MODEL_PATH, "ROLLOUT_MODEL_PATH is required")
    async def test_deactivate_then_restart_worker_with_two_workers(self):
        import ray
        import torch

        from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers

        ray.init(ignore_reinit_error=True)
        temp_dir = tempfile.TemporaryDirectory()
        try:
            resource_cfg = AcceleratorResourcesConfig(
                accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
                num_workers=2,
                num_cpus_per_worker=4,
                cpu_memory_per_worker=8 * 1024**3,
            )
            pg = AutoAcceleratorWorkers.build_placement_group(resource_cfg, name="recover_test_pg")
            rollout_cfg = RolloutConfig(
                env="test_rollout_utils",
                model_path=MODEL_PATH,
                model_name=os.path.basename(MODEL_PATH).lower(),
                tokenizer_path=MODEL_PATH,
                tensor_parallel_size=2,
                expert_parallel_size=1,
                worker_log_dir=temp_dir.name,
                health_check_first_wait_seconds=0,
                health_check_interval_seconds=600,
                health_check_failure_threshold=1,
            )
            controller = RolloutController(rollout_cfg, pg)

            rank0 = min(controller.rank2info.keys())
            rank1 = max(controller.rank2info.keys())
            actor0 = controller.rank2info[rank0].actor

            # Simulate worker-0 deactivation and lifecycle by forcing a failed health result for rank-0.
            async def _fake_check_worker_health(actor, rank, url, is_active, failure_threshold=1):
                return rank != rank0

            with patch("xtuner.v1.rl.rollout.utils.check_worker_health", _fake_check_worker_health):
                controller.health_checker.run_once()

            self.assertFalse(controller.rank2info[rank0].is_active)
            self.assertTrue(controller.rank2info[rank1].is_active)

            health_before_recover = await actor0.check_health.remote()
            self.assertFalse(health_before_recover)
            url_before = controller.rank2info[rank0].url

            controller.recover()

            self.assertTrue(controller.rank2info[rank0].is_active)
            self.assertEqual(url_before, controller.rank2info[rank0].url)
            health_after_recover = await actor0.check_health.remote()
            self.assertTrue(health_after_recover)
        finally:
            try:
                controller.shutdown()
            except Exception:
                pass
            ray.shutdown()
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
