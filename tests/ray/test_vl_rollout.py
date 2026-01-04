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

MODEL_PATH=os.getenv("QWEN3_VL_DENSE_PATH")
TRAIN_DATA_PATH=os.getenv("GEO3K_TRAIN_DATA_PATH")
MEDIA_ROOT=os.getenv("GEO3K_MEDIA_ROOT")

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
        self.max_prompt_length = 2048
        self.max_response_length = 2048
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            rollout_cross_node_comm=False,
            tensor_parallel_size=2,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
            launch_server_method="ray",
            context_length=self.max_prompt_length + self.max_response_length,
            worker_log_dir=self.worker_log_dir,
        )
        from xtuner.v1.ray.judger.geo3k import GEO3KJudgerConfig
        geo3k_judger_config = GEO3KJudgerConfig()
        self.judger_cfg = JudgerConfig(reward_judger_configs=[geo3k_judger_config])

        self.dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=0,
            max_retry_times=1,
            worker_log_dir=self.worker_log_dir,
        )
        self.training_sample_params = SampleParams(
            max_tokens=self.max_response_length,
        )
        self.evaluation_sample_params = SampleParams(
            max_tokens=self.max_response_length,
            top_p=1.0,
            temperature=0.0,
            top_k=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
        tokenize_fn_cfg = Qwen3VLTokenizeFnConfig(processor_path=self.model_path)
        train_dataset_cfg = [
            {
                "dataset": DatasetConfig(name="geo3k",
                                        anno_path=self.data_path,
                                        class_name='VLMJsonlDataset',
                                        media_root=self.media_root,
                                        sample_ratio=1.0),
                "tokenize_fn": RLTokenizeFnConfig(max_length=self.max_prompt_length,
                                                tokenize_fn_cfg=tokenize_fn_cfg),
            }
        ]
        dataloader_config = DataloaderConfig(num_workers=8,
                                            collator="fake_collator",
                                            pack_level="none")
        
        self.replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=train_dataset_cfg, 
            dataloader_cfg=dataloader_config, 
            tokenizer=self.tokenizer,
        )

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        self.model_path = MODEL_PATH
        self.media_root = MEDIA_ROOT
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
    def test_vl_resume_with_partial_rollout(self):
        rollout_cfg = self.rollout_cfg
        # rollout_cfg.enable_return_routed_experts = True
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=rollout_cfg,
        )
        dataflow_cfg = self.dataflow_cfg
        dataflow_cfg.max_concurrent = 4
        dataflow_cfg.enable_partial_rollout = 0
        self.test_flow = DataFlow.remote("test_env",
                                         dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         self.test_env
                                         )
        ray.get(self.test_flow.run.remote(), timeout=300)
        rl_status_save = ray.get(self.test_flow.get_replaybuffer_status.remote())
        save_dir = Path(self.temp_dir.name) / 'checkpoints' / 'ckpt-step-2'
        save_dir.mkdir(parents=True, exist_ok=True)

        ray.get(self.test_flow.save.remote(save_dir))
        remain_paused_samples_old = rl_status_save["rollout_paused_count"]
        responses_old = ray.get(self.test_flow.run.remote(num=remain_paused_samples_old, enable_partial_rollout=0), timeout=300)
        rb_status_old = ray.get(self.test_flow.get_replaybuffer_status.remote())
        
        mm_info_old = []
        for multimodal_train_infos in responses_old[1]:
            image_grid_thw = multimodal_train_infos["image_grid_thw"].numpy().flatten()
            mm_info_old.extend(image_grid_thw)

        ray.get(self.test_flow.resume.remote(save_dir))
        rl_status_resume = ray.get(self.test_flow.get_replaybuffer_status.remote())
        remain_paused_samples_new = rl_status_resume["rollout_paused_count"]
        responses_new = ray.get(self.test_flow.run.remote(num=remain_paused_samples_new, enable_partial_rollout=0), timeout=300)
        rb_status_new = ray.get(self.test_flow.get_replaybuffer_status.remote())

        mm_info_new = []
        for multimodal_train_infos in responses_new[1]:
            image_grid_thw = multimodal_train_infos["image_grid_thw"].numpy().flatten()
            mm_info_new.extend(image_grid_thw)
            
        all_train_prompt_ids_old = []
        for data_items in responses_old[0]:
            for data_item in data_items:
                all_train_prompt_ids_old.extend(data_item.data.input_ids)

        all_train_prompt_ids_new = []
        for data_items in responses_new[0]:
            for data_item in data_items:
                all_train_prompt_ids_new.extend(data_item.data.input_ids)

        all_train_prompt_ids_old.sort()
        all_train_prompt_ids_new.sort()
        mm_info_old.sort()
        mm_info_new.sort()
        self.assertEqual(all_train_prompt_ids_old, all_train_prompt_ids_new)
        self.assertEqual(mm_info_old, mm_info_new)
        for key in rb_status_old:
            self.assertEqual(rb_status_old[key], rb_status_new[key])
        for key in rl_status_save:
            self.assertEqual(rl_status_save[key], rl_status_resume[key])
        ray.get(self.test_env.shutdown.remote(), timeout=300)

if __name__ == "__main__":
    unittest.main()