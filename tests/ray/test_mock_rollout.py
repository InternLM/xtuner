import os
import asyncio
import unittest
import ray
from transformers import AutoTokenizer
import torch
import tempfile
import httpx
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.rollout.lmdeploy import LMDeployWorker
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"] 
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
resource_map = {"npu": "NPU", "cuda": "GPU"}

class MockTimeoutRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            raise httpx.TimeoutException("Mocked timeout error")
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked timeout exception: {e.__class__.__name__}")
            return result

    def _launch_server(self):
        pass  # Override


class MockRequestErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            req = httpx.Request("POST", url)
            raise httpx.RequestError("Mocked httpx request error", request=req)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked request error exception: {e.__class__.__name__}")
            return result

    def _launch_server(self):
        pass  # Override


class MockClientErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            req = httpx.Request("POST", url)
            res = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("Mocked client error", request=req, response=res)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked client exception: {e.__class__.__name__}")
            return result

    def _launch_server(self):
        pass  # Override


class MockServerErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            req = httpx.Request("POST", url)
            res = httpx.Response(500, request=req)
            raise httpx.HTTPStatusError("Mocked server error", request=req, response=res)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked server exception: {e.__class__.__name__}")
            return result

    def _launch_server(self):
        pass  # Override

class MockInvalidResponseRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        mock_rollout_state = RolloutState(message=TEST_TEXT_MESSAGES, status=Status.FAILED)
        result = HttpRequestResult(response=mock_rollout_state)
        return result
    
    async def _safe_handle_response(self, rollout_state, http_response) -> RolloutState:
        mock_rollout_state = RolloutState(message=TEST_TEXT_MESSAGES, status=Status.FAILED)
        return mock_rollout_state
    
    def _launch_server(self):
        pass  # Override

@ray.remote
class MockTimeoutRolloutController(RolloutController):
    def _get_worker_cls(self): return ray.remote(MockTimeoutRolloutWorker)

@ray.remote
class MockRequestErrorRolloutController(RolloutController):
    def _get_worker_cls(self): return ray.remote(MockRequestErrorRolloutWorker)

@ray.remote    
class MockClientErrorRolloutController(RolloutController):
    def _get_worker_cls(self): return ray.remote(MockClientErrorRolloutWorker)

@ray.remote
class MockServerErrorRolloutController(RolloutController):
    def _get_worker_cls(self): return ray.remote(MockServerErrorRolloutWorker)

@ray.remote
class MockInvalidResponseRolloutController(RolloutController):
    def _get_worker_cls(self): return ray.remote(MockInvalidResponseRolloutWorker)
    
class TestMockRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["XTUNER_USE_FA3"] = "1"

    @classmethod
    def tearDownClass(cls):
        del os.environ["XTUNER_USE_FA3"]

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.global_batch_size = 3
        self.max_prompt_length = 4096
        self.max_response_length = 128
        self.max_concurrent = 3
        self.max_retry_times = 3
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.rollout_cfg = RolloutConfig(
            env="test_mock_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=1,
            context_length=self.max_prompt_length + self.max_response_length,
            max_retry_per_worker=2,
            max_retry_per_sample=3,
            worker_log_dir=self.worker_log_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    async def _run_mock_test(self, mock_controller_cls, error_name, pg):
        rollout_controller = mock_controller_cls.remote(self.rollout_cfg, pg)
        input_state = RolloutState(message=TEST_TEXT_MESSAGES)
        result_state = await rollout_controller.generate.remote(rollout_state=input_state)
        self.assertEqual(result_state.status, Status.FAILED, f"Expected rollout to fail due to {error_name}, but it succeeded.")
        self.assertIsNotNone(result_state.error_msg, f"Expected an error message for {error_name} case, but got None.")
        if error_name == "server_error":
            self.assertIn("Server error", result_state.error_msg, f"Expected error message to indicate a server error for {error_name} case, but got: {result_state.error_msg}")
        elif error_name == "client_error":
            self.assertIn("Client error", result_state.error_msg, f"Expected error message to indicate a client error for {error_name} case, but got: {result_state.error_msg}")
        elif error_name in ["request_error", "timeout"]:
            self.assertIn("Request failed", result_state.error_msg, f"Expected error message to indicate a request error for {error_name} case, but got: {result_state.error_msg}")
            self.assertIn(str(self.rollout_cfg.max_retry_per_sample), result_state.error_msg, f"Expected error message to include max retry times for {error_name} case, but got: {result_state.error_msg}")
        elif error_name == "invalid_response":
            self.assertIn("Invalid rollout response", result_state.error_msg, f"Expected error message to indicate an invalid response for {error_name} case, but got: {result_state.error_msg}")
            self.assertIn(str(self.rollout_cfg.max_retry_per_sample), result_state.error_msg, f"Expected error message to include max retry times for {error_name} case, but got: {result_state.error_msg}")

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_parallel_mock_rollout(self):
        async def run_parallel():
            res_cfg_small = AcceleratorResourcesConfig(
                accelerator=resource_map[torch.accelerator.current_accelerator().type],
                num_workers=1,
                num_cpus_per_worker=2,
            )
            
            pgs = [AutoAcceleratorWorkers.build_placement_group(res_cfg_small, name=f"pg_{i}") for i in range(5)]
            await asyncio.gather(*[pg.ready() for pg in pgs])

            tasks = [
                self._run_mock_test(MockTimeoutRolloutController, "timeout", pgs[0]),
                self._run_mock_test(MockRequestErrorRolloutController, "request_error", pgs[1]),
                self._run_mock_test(MockClientErrorRolloutController, "client_error", pgs[2]),
                self._run_mock_test(MockServerErrorRolloutController, "server_error", pgs[3]),
                self._run_mock_test(MockInvalidResponseRolloutController, "invalid_response", pgs[4]),
            ]
            await asyncio.gather(*tasks)

        asyncio.run(run_parallel())

if __name__ == "__main__":
    unittest.main()