import os
import subprocess
import unittest
import tempfile
import ray
import torch
from transformers import AutoTokenizer
import tempfile
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
import asyncio
from xtuner.v1.rl.rollout import RolloutController


MODEL_PATH=os.getenv("QWEN3_VL_DENSE_PATH")
MOE_MODEL_PATH=os.getenv("QWEN3_VL_MOE_PATH")
MEDIA_ROOT=os.getenv("GEO3K_MEDIA_ROOT")

resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}
class TestVLMRollout(unittest.IsolatedAsyncioTestCase):

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
        self.max_prompt_length = 1024
        self.max_response_length = 2048
        self.context_length = self.max_prompt_length + self.max_response_length

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenize_fn = RLQwen3VLTokenizeFnConfig(processor_path=self.model_path, max_length=self.max_prompt_length)
        self.tokenize_fn = tokenize_fn.build(tokenizer)
    
    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
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

    def _cleanup_lmdeploy_ray_worker_wrapper(self):
        try:
            result = subprocess.run(["pkill", "-f", "ray::RayWorkerWrapper*"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"pkill command failed with return code {result.returncode}: {result.stderr}."
                      " Maybe no lmdeploy ray::RayWorkerWrapper processes found.")
        except Exception as e:
            print(f"Error stopping ray::RayWorkerWrapper cluster: {e}")
    

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
                # self._run_rollout(model_path=moe_model_path, tp_size=1, ep_size=4, pg=pg2, dist_port_base=dist_port_base + 1024 * 4), # TODO: lmdeploy 修复后启动
                return_exceptions=False
            )
        
        asyncio.run(run_both())
    
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

        # Test Case 1: 纯文本
        rollout_state = self.tokenize_fn({'prompt':[{"role": "user", "content": "Hello!"}]})
        result1_ref = rollout_controller.generate.remote(rollout_state=rollout_state)
        result_refs.append(result1_ref)
        
        # Test Case 2: 图片
        input_data = {"prompt": [{"content": [{"image_url": {"image_wh": [297, 265], "url": "images/test_0.jpg"}, "type": "image_url"}, {"text": "<IMG_CONTEXT>Chords $\\overline{A C}$ and $\\overline{D F}$ are equidistant from the center. If the radius of $\\odot G$ is 26 find $A C$ You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", "type": "text"}], "role": "user"}], "data_source": "hiyouga/geometry3k", "ability": "math", "reward_model": {"ground_truth": "48", "style": "rule"}}
        rollout_state = self.tokenize_fn(input_data, media_root=MEDIA_ROOT)
        rollout_state.tokens = rollout_state.prompt_ids
        result2_ref = rollout_controller.generate.remote(rollout_state=rollout_state)
        result_refs.append(result2_ref)

        try:
            results = await asyncio.wait_for(asyncio.gather(*result_refs), timeout=300)
            for i, result in enumerate(results):
                case_id = f"Case {i+1}"
                self.assertEqual(result.status, Status.COMPLETED, 
                                 msg=f"{case_id} failed: Expected status COMPLETED but got {result.status} and error_msg {result.error_msg}")
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

    # @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    # def test_vl_resume_with_partial_rollout(self):
    #     # TODO: 后续实现
    #     pass


if __name__ == "__main__":
    unittest.main()
