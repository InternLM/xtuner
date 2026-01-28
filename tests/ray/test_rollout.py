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
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.ray.judger import JudgerController
from xtuner.v1.datasets import RLTokenizeFnConfig, build_datasets, build_dataloader
from xtuner.v1.datasets.config import (
    DataloaderConfig,
    DatasetConfig,
)

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
MOE_MODEL_PATH = os.environ["QWEN3_MOE_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}
class TestRollout(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]

    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=8,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.max_prompt_length = 512
        self.max_response_length = 1024
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            rollout_cross_node_comm=False,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
            launch_server_method="ray",
            context_length=self.max_prompt_length + self.max_response_length,
            worker_log_dir=self.worker_log_dir,
        )
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        self.judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.worker_log_dir,
        )
        self.dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=0,
            max_retry_times=1,
            worker_log_dir=self.worker_log_dir,
        )
        self.train_dataset_cfg = [
            {
            "dataset": DatasetConfig(name="gsm8k",
                                    anno_path=TRAIN_DATA_PATH,
                                    sample_ratio=1.0),
            "tokenize_fn": RLTokenizeFnConfig(max_length=self.max_prompt_length),
            },
        ]
        self.dataloader_cfg = DataloaderConfig(
            collator='fake_collator',
            pack_level='none',
            group_by_length=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=self.train_dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            tokenizer=self.tokenizer,
            worker_log_dir=self.worker_log_dir,
        )

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)

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
    def test_lmdeploy_generate(self):
        rollout_cfg = self.rollout_cfg.model_copy(
            deep=True,
            update=dict(tensor_parallel_size=2),
        )
        rollout_cfg.model_post_init(None)

        sample_params = SampleParams(temperature=0.0)
        rollout_controller = ray.remote(RolloutController).remote(rollout_cfg, self.pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
       
        self.assertEqual(res1.finish_reason, "stop") 
        print("Response from LMDeploy infer:", res1)
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow(self):
        rollout_cfg = self.rollout_cfg.model_copy(
            deep=True,
            update=dict(
                expert_parallel_size=2,
                model_path=self.model_path, 
                model_name=os.path.basename(MOE_MODEL_PATH).lower(),
                tokenizer_path=MOE_MODEL_PATH,
            ),
        )
        rollout_cfg.model_post_init(None)

        self.dataflow_cfg.enable_partial_rollout = 0
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=rollout_cfg,
        )
        self.test_flow = DataFlow.remote("test_env",
                                         self.dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         self.test_env
                                        )
        responses = ray.get(self.test_flow.run.remote(), timeout=300)
        finished_samples_count = sum(1 for data in responses[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote(), timeout=300)

    def _get_sorted_input_ids(self, responses):
        """Helper to extract and sort input_ids from responses."""
        all_ids = []
        for data_items in responses[0]:
            for data_item in data_items:
                all_ids.extend(data_item.data.input_ids)
        all_ids.sort()
        return all_ids
    
    def _run_dataflow_save_resume_test(self, rollout_cfg, dataflow_cfg):
        """
        Generic driver for dataflow save/resume tests.
        """
        # 1. Initialize Environment and DataFlow
        is_partial_rollout = dataflow_cfg.enable_partial_rollout == 1
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=rollout_cfg,
        )
        self.test_flow = DataFlow.remote(
            "test_env",
            dataflow_cfg,
            self.replay_buffer_cfg,
            self.test_env
        )

        # 2. Initial Run
        ray.get(self.test_flow.run.remote(), timeout=300)
        
        # Capture status before saving (critical for partial rollout consistency check)
        rl_status_before_save = ray.get(self.test_flow.get_replaybuffer_status.remote())

        # 3. Save
        save_dir = Path(self.temp_dir.name) / 'checkpoints' / f'ckpt-step-2'
        save_dir.mkdir(parents=True, exist_ok=True)
        ray.get(self.test_flow.save.remote(save_dir))

        # Define run logic based on mode
        def run_continuation(status_ref):
            if is_partial_rollout:
                remain = status_ref["rollout_paused_count"] + status_ref["rollout_finished_count"]
                # Finish the remaining paused samples
                return ray.get(self.test_flow.run.remote(num=remain, staleness_threshold=0), timeout=300)
            else:
                # Normal run
                return ray.get(self.test_flow.run.remote(), timeout=300)

        # continue running after save
        responses_old = run_continuation(rl_status_before_save)
        rb_status_old = ray.get(self.test_flow.get_replaybuffer_status.remote())


        # resume from saved checkpoint
        ray.get(self.test_flow.resume.remote(save_dir))
        rl_status_resume = ray.get(self.test_flow.get_replaybuffer_status.remote())
        responses_new = run_continuation(rl_status_resume)
        rb_status_new = ray.get(self.test_flow.get_replaybuffer_status.remote())

        # 6. Cleanup
        ray.get(self.test_env.shutdown.remote(), timeout=300)

        # 7. Assertions
        # Compare Data
        ids_old = self._get_sorted_input_ids(responses_old)
        ids_new = self._get_sorted_input_ids(responses_new)
        self.assertEqual(ids_old, ids_new)

        # Compare ReplayBuffer Status (Old run vs New run)
        for key in rb_status_old:
            self.assertEqual(rb_status_old[key], rb_status_new[key])

        # For partial rollout, verify the resumed state matches the saved state
        if is_partial_rollout:
            for key in rl_status_before_save:
                self.assertEqual(rl_status_before_save[key], rl_status_resume[key])
        
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow_save_resume(self):
        rollout_cfg = self.rollout_cfg
        dataflow_cfg = self.dataflow_cfg
        dataflow_cfg.enable_partial_rollout = 0
        self._run_dataflow_save_resume_test(rollout_cfg, dataflow_cfg)

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow_save_resume_with_partial_rollout(self):
        rollout_cfg = self.rollout_cfg
        dataflow_cfg = self.dataflow_cfg
        dataflow_cfg.staleness_threshold = 1
        dataflow_cfg.enable_partial_rollout = 1
        self._run_dataflow_save_resume_test(rollout_cfg, dataflow_cfg)

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow_save_resume_with_partial_rollout_r3(self):
        model_path = MOE_MODEL_PATH
        rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=model_path,
            model_name=os.path.basename(model_path).lower(),
            tokenizer_path=model_path,
            rollout_cross_node_comm=False,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpus_per_node=8,
            dtype="bfloat16",
            launch_server_method="ray",
            context_length=self.max_prompt_length + self.max_response_length,
            worker_log_dir=self.worker_log_dir,
            enable_return_routed_experts=True,
        )
        dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=1,
            staleness_threshold=1,
            worker_log_dir=self.worker_log_dir,
        )
        self._run_dataflow_save_resume_test(rollout_cfg, dataflow_cfg)

    @unittest.skip("skip lmdeploy turbomind generate test due to ci environment issue")
    def test_lmdeploy_turbomind_generate(self):
        from xtuner.v1.ray.rollout import LMDeployWorker
        self.rollout_cfg.extra_rollout_config["lmdeploy_backend"] = "turbomind"
        sample_params = SampleParams(temperature=0.0)
        rollout_controller = ray.remote(RolloutController).remote(self.rollout_cfg, self.pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        res2 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        self.assertEqual(res1, res2, f"res1 != res2, res1={res1}, res2={res2}")
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "lmdeploy backend is not enabled")
    def test_sglang_generate(self):
        from xtuner.v1.ray.rollout import SGLangWorker
        self.rollout_cfg.launch_server_method="multiprocessing"
        sample_params = SampleParams(temperature=0.0)
        rollout_controller = ray.remote(RolloutController).remote(self.rollout_cfg, self.pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        self.assertEqual(res1.finish_reason, "stop")
        print("Response from SGLang infer:", res1)
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "lmdeploy backend is not enabled")
    def test_sglang_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 0
        self.rollout_cfg.launch_server_method="multiprocessing"
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=self.rollout_cfg,
        )
        self.test_flow = DataFlow.remote("test_env",
                                         self.dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         self.test_env
                                        )
        responses = ray.get(self.test_flow.run.remote(), timeout=300)
        finished_samples_count = sum(1 for data in responses[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote(), timeout=300)
        print("responses: ", responses)

if __name__ == "__main__":
    unittest.main()
