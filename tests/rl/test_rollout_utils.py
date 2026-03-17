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
from xtuner.v1.rl.rollout.utils import RolloutHealthChecker, SessionRouter
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers, asyncio_run

MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH", "")
RESOURCE_MAP = {"npu": "NPU", "cuda": "GPU"}
TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]

def _kill_lmdeploy_server_wrapper(self):
    """Force kill lmdeploy wrapper process to simulate hard engine crash."""
    result = subprocess.run(
        ["ps", "-ef"],
        check=True,
        capture_output=True,
        text=True,
    )
    target_pids = []
    for line in result.stdout.splitlines():
        if "run_lmdeploy_server_wrapper" not in line:
            continue
        if "grep" in line:
            continue
        cols = line.split()
        if len(cols) > 1 and cols[1].isdigit():
            target_pids.append(int(cols[1]))

    for pid in target_pids:
        subprocess.run(["kill", "-9", str(pid)], check=False)


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
