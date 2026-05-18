import ray
import torch
import threading
import time
import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.data_proto.rl_data import Status, RolloutState, SampleParams
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout.controller import RolloutController, WorkerInfo
from xtuner.v1.rl.rollout.utils import (
    PartialRolloutHandler,
    RolloutHealthChecker,
    SessionRouter,
)
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run

MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH", "")
RESOURCE_MAP = {"npu": "NPU", "cuda": "GPU"}
TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]


class _FakeRemoteMethod:
    def __init__(self, name, call_log):
        self.name = name
        self.call_log = call_log

    def remote(self):
        self.call_log.append((self.name, "remote"))
        return self.name


class _FakeWorker:
    def __init__(self):
        self.call_log = []
        self.offload = _FakeRemoteMethod("offload", self.call_log)
        self.shutdown = _FakeRemoteMethod("shutdown", self.call_log)


class TestRolloutHealthChecker(unittest.TestCase):
    def _build_checker(self, workers_info):
        config = SimpleNamespace(health_check_interval_seconds=10, health_check_failure_threshold=1)
        return RolloutHealthChecker(config, workers_info)

    def test_shutdown_runs_when_offload_fails(self):
        worker = _FakeWorker()
        workers_info = {0: SimpleNamespace(actor=worker, url="http://worker-0", is_active=True)}
        checker = self._build_checker(workers_info)

        async def unhealthy_worker(*args, **kwargs):
            return False

        def ray_get(ref, timeout=None):
            worker.call_log.append((ref, "get"))
            if ref == "offload":
                raise RuntimeError("offload failed")
            return None

        with (
            patch("xtuner.v1.rl.rollout.utils.check_worker_health", side_effect=unhealthy_worker),
            patch("xtuner.v1.rl.rollout.utils.ray.get", side_effect=ray_get),
        ):
            checker.run_once()

        self.assertFalse(workers_info[0].is_active)
        self.assertEqual(
            worker.call_log,
            [
                ("offload", "remote"),
                ("offload", "get"),
                ("shutdown", "remote"),
                ("shutdown", "get"),
            ],
        )

    def test_inactive_worker_is_not_cleaned_up_again(self):
        worker = _FakeWorker()
        workers_info = {0: SimpleNamespace(actor=worker, url="http://worker-0", is_active=False)}
        checker = self._build_checker(workers_info)

        with (
            patch("xtuner.v1.rl.rollout.utils.check_worker_health") as check_worker_health_mock,
            patch("xtuner.v1.rl.rollout.utils.ray.get") as ray_get_mock,
        ):
            checker.run_once()

        check_worker_health_mock.assert_not_called()
        ray_get_mock.assert_not_called()
        self.assertEqual(worker.call_log, [])


class TestPartialRolloutHandler(unittest.IsolatedAsyncioTestCase):
    async def test_postprocess_frees_old_routed_expert_refs_after_concat(self):
        class FakeObjectRef:
            pass

        history_ref = FakeObjectRef()
        cur_ref = FakeObjectRef()
        concat_ref = FakeObjectRef()
        rollout_state = RolloutState(
            message=[],
            response="old",
            response_ids=[1, 2],
            logprobs=[0.1, 0.2],
            routed_experts=history_ref,
            status=Status.ABORTED,
        )

        async def resolve_routed_experts(value):
            if value is history_ref:
                return [[1], [2]]
            if value is cur_ref:
                return [[1], [2], [3]]
            raise AssertionError(f"unexpected routed_experts value: {value}")

        with (
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", FakeObjectRef),
            patch("xtuner.v1.rl.rollout.utils._resolve_routed_experts", side_effect=resolve_routed_experts),
            patch("xtuner.v1.rl.rollout.utils.ray.put", return_value=concat_ref) as ray_put,
            patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free_object_refs,
        ):
            out = await PartialRolloutHandler().postprocess(
                rollout_state,
                response="new",
                response_ids=[3],
                logprobs=[0.3],
                routed_experts=cur_ref,
                finish_reason="abort",
                status=Status.ABORTED,
                enable_partial_rollout=True,
            )

        self.assertIs(out.routed_experts, concat_ref)
        ray_put.assert_called_once_with([[1], [2], [3]])
        free_object_refs.assert_called_once_with([history_ref, cur_ref])


class TestRolloutControllerRecover(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"
        
    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]

    def setUp(self):
        ray.init(num_cpus=80, address="local", ignore_reinit_error=True)
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    def init_rollout_controller(self):
        resource_cfg = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=1,
            num_cpus_per_worker=4,
            cpu_memory_per_worker=8 * 1024**3,
        )
        pg = AutoAcceleratorWorkers.build_placement_group(resource_cfg, name="recover_test_pg")
        rollout_cfg = RolloutConfig(
            env="test_rollout_utils",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            worker_log_dir=self.temp_dir.name,
            context_length=8192,
            health_check_interval_seconds=10,
            health_check_failure_threshold=1,
        )
        controller = RolloutController(rollout_cfg, pg)
        return controller

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_healthcheck_deactivate_and_recover(self):
        controller = self.init_rollout_controller()
        ranks = list(controller.rank2info.keys())
        rank0 = ranks[0]
        actor0 = controller.rank2info[rank0].actor
        ray.get(actor0.shutdown.remote())
        time.sleep(3)  # wait for the actor to be fully killed
        health_before_recover = ray.get(actor0.check_health.remote())
        url = controller.rank2info[rank0].url
        self.assertFalse(health_before_recover)

        controller.health_checker.run_once()

        self.assertFalse(controller.rank2info[rank0].is_active)
        rollout_state = RolloutState(
            message=TEST_TEXT_MESSAGES,
            sample_params=SampleParams(return_token_ids=True),
        )
        out = asyncio_run(controller.generate(rollout_state))
        self.assertEqual(out.status, Status.FAILED)

        controller.recover_failed_workers()

        self.assertTrue(controller.rank2info[rank0].is_active)
        self.assertEqual(url, controller.rank2info[rank0].url)
        health_after_recover = ray.get(actor0.check_health.remote())
        self.assertTrue(health_after_recover)
        out = asyncio_run(controller.generate(rollout_state))
        self.assertNotEqual(out.status, Status.FAILED)


if __name__ == "__main__":
    unittest.main()
