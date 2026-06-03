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


class _FakeControllerRemoteMethod:
    def __init__(self, name, call_log, return_value=None):
        self.name = name
        self.call_log = call_log
        self.return_value = return_value if return_value is not None else name

    def remote(self, *args, **kwargs):
        self.call_log.append((self.name, args, kwargs))
        if isinstance(self.return_value, list):
            if len(self.return_value) > 1:
                value = self.return_value.pop(0)
            else:
                value = self.return_value[0]
            if isinstance(value, Exception):
                raise value
            return value
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value


class _FakeControllerHealthChecker:
    def __init__(self):
        self.call_log = []

    def pause(self):
        self.call_log.append("pause")

    def run_once(self):
        self.call_log.append("run_once")

    def resume(self):
        self.call_log.append("resume")


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


class TestRolloutControllerTrainingBarrier(unittest.TestCase):
    def _build_controller(self, workers_info):
        controller = RolloutController.__new__(RolloutController)
        controller.rank2info = workers_info
        controller.worker_info_lock = threading.RLock()
        controller.health_checker = _FakeControllerHealthChecker()
        controller.logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        )
        return controller

    def _ray_get(self, ref, timeout=None):
        if isinstance(ref, list):
            return ref
        return ref

    def _remote(self, func):
        return SimpleNamespace(remote=func)

    def test_ensure_workers_healthy_before_training_recovers_with_original_url(self):
        # recovery 必须在原 URL 上重启成功，才能允许共卡训练继续。
        call_log = []
        health_results = [False, True]

        def check_health():
            result = health_results.pop(0)
            call_log.append(("check_health", workers_info[0].is_active, result))
            return result

        def shutdown():
            call_log.append(("shutdown", workers_info[0].is_active))

        def init(*args, **kwargs):
            call_log.append(("init", args, kwargs))
            return 0, "http://worker-0"

        def init_dist_port():
            call_log.append(("init_dist_port",))
            return "127.0.0.1:12345"

        worker = SimpleNamespace(
            offload=_FakeControllerRemoteMethod("offload", call_log),
            shutdown=self._remote(shutdown),
            init_dist_port=self._remote(init_dist_port),
            init=self._remote(init),
            check_health=self._remote(check_health),
        )
        workers_info = {0: WorkerInfo(actor=worker, url="http://worker-0", is_active=True)}
        controller = self._build_controller(workers_info)

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=self._ray_get):
            controller.ensure_workers_healthy_before_training()

        self.assertTrue(workers_info[0].is_active)
        self.assertEqual(workers_info[0].url, "http://worker-0")
        self.assertEqual(
            call_log,
            [
                ("check_health", True, False),
                ("shutdown", False),
                ("init", (), {}),
                ("check_health", False, True),
            ],
        )

    def test_ensure_workers_healthy_before_training_fails_if_restarted_url_changes(self):
        # recovery 后 URL 改变时，不能允许共卡训练继续。
        call_log = []

        def check_health():
            call_log.append(("check_health", workers_info[0].is_active, False))
            return False

        def shutdown():
            call_log.append(("shutdown", workers_info[0].is_active))

        def init(*args, **kwargs):
            call_log.append(("init", args, kwargs))
            return 0, "http://worker-0-new"

        def init_dist_port():
            call_log.append(("init_dist_port",))
            return "127.0.0.1:12345"

        worker = SimpleNamespace(
            offload=_FakeControllerRemoteMethod("offload", call_log),
            shutdown=self._remote(shutdown),
            init_dist_port=self._remote(init_dist_port),
            init=self._remote(init),
            check_health=self._remote(check_health),
        )
        workers_info = {0: WorkerInfo(actor=worker, url="http://worker-0", is_active=True)}
        controller = self._build_controller(workers_info)

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=self._ray_get):
            with self.assertRaisesRegex(RuntimeError, "inactive rollout workers before training"):
                controller.ensure_workers_healthy_before_training()

        self.assertFalse(workers_info[0].is_active)
        self.assertEqual(workers_info[0].url, "http://worker-0")
        self.assertEqual(
            call_log,
            [
                ("check_health", True, False),
                ("shutdown", False),
                ("init", (), {}),
            ],
        )

    def test_ensure_workers_healthy_before_training_fails_if_worker_cannot_recover(self):
        # recover 失败时不能进入训练，也不能执行最终 offload 伪装成安全。
        call_log = []
        worker = SimpleNamespace(
            offload=_FakeControllerRemoteMethod("offload", call_log),
            shutdown=_FakeControllerRemoteMethod("shutdown", call_log),
            init_dist_port=_FakeControllerRemoteMethod("init_dist_port", call_log, "127.0.0.1:12345"),
            init=_FakeControllerRemoteMethod("init", call_log, (0, "http://worker-0")),
            check_health=_FakeControllerRemoteMethod("check_health", call_log, False),
        )
        workers_info = {0: WorkerInfo(actor=worker, url="http://worker-0", is_active=False)}
        controller = self._build_controller(workers_info)

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=self._ray_get):
            with self.assertRaisesRegex(RuntimeError, "inactive rollout workers before training"):
                controller.ensure_workers_healthy_before_training()

        self.assertFalse(workers_info[0].is_active)
        self.assertNotEqual(call_log[-1], ("offload", (), {}))

    def test_ensure_workers_healthy_before_training_fails_if_shutdown_fails(self):
        # 旧 rollout server 不能确认释放时，不能继续进入共卡训练。
        call_log = []
        worker = SimpleNamespace(
            offload=_FakeControllerRemoteMethod("offload", call_log),
            shutdown=_FakeControllerRemoteMethod("shutdown", call_log, RuntimeError("shutdown failed")),
            init_dist_port=_FakeControllerRemoteMethod("init_dist_port", call_log, "127.0.0.1:12345"),
            init=_FakeControllerRemoteMethod("init", call_log, (0, "http://worker-0")),
            check_health=_FakeControllerRemoteMethod("check_health", call_log, False),
        )
        workers_info = {0: WorkerInfo(actor=worker, url="http://worker-0", is_active=True)}
        controller = self._build_controller(workers_info)

        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=self._ray_get):
            with self.assertRaisesRegex(RuntimeError, "inactive rollout workers before training"):
                controller.ensure_workers_healthy_before_training()

        self.assertFalse(workers_info[0].is_active)
        self.assertIn(("shutdown", (), {}), call_log)
        self.assertNotIn(("init", (), {}), call_log)


class TestPartialRolloutHandler(unittest.IsolatedAsyncioTestCase):
    async def test_postprocess_frees_old_routed_expert_refs_after_concat(self):
        class FakeObjectRef:
            def __init__(self, value):
                self.value = value

            def __await__(self):
                async def _resolve():
                    return self.value

                return _resolve().__await__()

        history_ref = FakeObjectRef([[1], [2]])
        cur_ref = FakeObjectRef([[1], [2], [3]])
        concat_ref = FakeObjectRef(None)
        rollout_state = RolloutState(
            message=[],
            response="old",
            response_ids=[1, 2],
            logprobs=[0.1, 0.2],
            routed_experts=history_ref,
            status=Status.ABORTED,
        )

        with (
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", FakeObjectRef),
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
                prompt_tokens=3,
                completion_tokens=1,
            )

        self.assertIs(out.routed_experts, concat_ref)
        self.assertEqual(ray_put.call_args.args[0].tolist(), [[1], [2], [3]])
        free_object_refs.assert_any_call([history_ref])
        free_object_refs.assert_any_call([cur_ref])
        self.assertEqual(free_object_refs.call_count, 2)


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

        controller.ensure_workers_healthy_before_training()

        self.assertTrue(controller.rank2info[rank0].is_active)
        self.assertEqual(url, controller.rank2info[rank0].url)
        health_after_recover = ray.get(actor0.check_health.remote())
        self.assertTrue(health_after_recover)
        out = asyncio_run(controller.generate(rollout_state))
        self.assertNotEqual(out.status, Status.FAILED)


if __name__ == "__main__":
    unittest.main()
