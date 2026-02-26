import os
import subprocess
from functools import wraps
import unittest
import tempfile
import ray
import torch
from pathlib import Path
from transformers import AutoTokenizer
import tempfile
from xtuner.v1.ray.config.worker import RolloutConfig
# from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
# from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.data_proto.rl_data import Status, SampleParams, RolloutState
# from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout import RolloutController
# from xtuner.v1.ray.judger import JudgerController
# from xtuner.v1.datasets import RLTokenizeFnConfig, build_datasets, build_dataloader
# from xtuner.v1.datasets.config import (
#     DataloaderConfig,
#     DatasetConfig,
# )
import asyncio

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
MOE_MODEL_PATH = os.environ["QWEN3_MOE_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}
class TestRollout(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"
        
    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]
        
    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=8,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.max_prompt_length = 512
        self.max_response_length = 1024
        self.context_length = self.max_prompt_length + self.max_response_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()

    def tearDown(self):
        ray.shutdown()
        # When lmdeploy enable ep>1, it uses deep_ep. Buffer implicit destroy would cause some ray actor stucked.
        # Use pkill cleen up ray::WorkerWrapper process after close ray cluster connection as workaround.
        # TODO(chenchiyu): add excplicit deep_ep destroy in lmdeploy.
        self._cleanup_lmdeploy_ray_worker_wrapper()
        self.temp_dir.cleanup()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_parallel_rollout(self):
        resource_config = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=4,
            cpu_memory_per_worker=8 * 1024**3,  # 8 GB
        )
        pg1 = AutoAcceleratorWorkers.build_placement_group(resource_config, name="tp_pg")
        pg2 = AutoAcceleratorWorkers.build_placement_group(resource_config, name="ep_pg")
        dense_model_path = MODEL_PATH
        moe_model_path = MOE_MODEL_PATH
        dist_port_base = 38000
        async def run_both():
            return await asyncio.gather(
                self._run_rollout(model_path=dense_model_path, tp_size=4, ep_size=1, pg=pg1, dist_port_base=dist_port_base),
                self._run_rollout(model_path=moe_model_path, tp_size=1, ep_size=4, pg=pg2, dist_port_base=dist_port_base + 1024 * 4),
                return_exceptions=False
            )
        
        asyncio.run(run_both())

    def _cleanup_lmdeploy_ray_worker_wrapper(self):
        try:
            result = subprocess.run(["pkill", "-f", "ray::RayWorkerWrapper*"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"pkill command failed with return code {result.returncode}: {result.stderr}."
                      " Maybe no lmdeploy ray::RayWorkerWrapper processes found.")
        except Exception as e:
            print(f"Error stopping ray::RayWorkerWrapper cluster: {e}")

    async def _run_rollout(self, model_path, tp_size, ep_size, pg, dist_port_base):
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=model_path,
            model_name=os.path.basename(model_path).lower(),
            tokenizer_path=model_path,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
            dist_port_base=dist_port_base,
            enable_return_routed_experts=ep_size > 1, # ep_size > 1 默认打开r3
        )
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        result_refs = []

        # Test Case 1: 文本输入 + 文本输出
        # TODO(@duanyanhui): test prompt in and prompt out with v1/chat/completion api
        # sample_params1 = SampleParams(return_token_ids=False)
        # input1 = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params1)
        # result1_ref = rollout_controller.generate.remote(rollout_state=input1)
        # result_refs.append(result1_ref)

        # Test Case 2: 文本输入 + Token 输出
        sample_params2 = SampleParams(return_token_ids=True)
        input2 = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params2)
        result2_ref = rollout_controller.generate.remote(rollout_state=input2)
        result_refs.append(result2_ref)

        # Test Case 3: Token 输入 + Token 输出
        text_prompt = self.tokenizer.apply_chat_template(TEST_TEXT_MESSAGES, tokenize=False, add_generation_prompt=True)
        input_tokens = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        sample_params3 = SampleParams(return_token_ids=True)
        input3 = RolloutState(message=TEST_TEXT_MESSAGES, tokens=input_tokens, sample_params=sample_params3)
        result3_ref = rollout_controller.generate.remote(rollout_state=input3)
        result_refs.append(result3_ref)

        try:
            results = await asyncio.wait_for(asyncio.gather(*result_refs), timeout=300)
            for i, result in enumerate(results):
                case_id = f"Case {i+1}"
                self.assertEqual(result.status, Status.COMPLETED, 
                                 msg=f"{case_id} failed: Expected status COMPLETED but got {result.status}")
                self.assertEqual(result.finish_reason, 'stop', 
                                 msg=f"{case_id} failed: Expected finish_reason 'stop' but got {result.finish_reason}")
            
                if result.sample_params.return_token_ids:
                    self.assertGreater(len(result.response_ids), 0, 
                                       msg=f"{case_id} failed: response_ids should not be empty when return_token_ids is True")
                
                if result.sample_params.return_logprob:
                    self.assertEqual(len(result.logprobs), len(result.response_ids),
                                     msg=f"{case_id} failed: logprobs length ({len(result.logprobs)}) "
                                         f"does not match response_ids length ({len(result.response_ids)})")
                    
        except asyncio.TimeoutError:
            if tp_size > 1 and ep_size == 1:
                self.fail("TP and Dense Rollout timed out!")
            if ep_size > 1 and tp_size == 1:
                self.fail("EP and MoE Rollout timed out!") 
        finally:
            await asyncio.wait_for(rollout_controller.shutdown.remote(), timeout=300)

if __name__ == "__main__":
    unittest.main()
