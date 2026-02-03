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
class TestRollout(unittest.TestCase):

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
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        self.judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.worker_log_dir,
        )
        self.dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=1,
            global_batch_size=1,
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

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_parallel_model_save_and_resume(self):
        resource_config = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=4,
            cpu_memory_per_worker=8 * 1024**3,  # 8 GB
        )
        pg1 = AutoAcceleratorWorkers.build_placement_group(resource_config, name="dense_pg")
        pg2 = AutoAcceleratorWorkers.build_placement_group(resource_config, name="moe_pg")

        async def run_both():
            return await asyncio.wait_for(
                asyncio.gather(
                    self._run_dense_save_resume_sync_async(pg1), 
                    self._run_moe_save_resume_with_r3(pg2), 
                    return_exceptions=False
                ),
                timeout=300
            )
        try:
            asyncio.run(run_both())
        except asyncio.TimeoutError:
            self.fail("test_parallel_model_save_and_resume timed out after 300s")

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

        )
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        try:
            result = await asyncio.wait_for(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES), timeout=300)
            self.assertEqual(result.finish_reason, "stop") 
        except asyncio.TimeoutError:
            self.fail("TP Rollout timed out!") 
        finally:
            await asyncio.wait_for(rollout_controller.shutdown.remote(), timeout=300)

    async def _run_dataflow_save_resume_test(self, test_env, dataflow_cfg: DataFlowConfig, replay_buffer_cfg: ReplayBufferConfig):
        """
        Generic driver for dataflow save/resume tests.
        """
        # 1. Initialize Environment and DataFlow
        is_partial_rollout = dataflow_cfg.enable_partial_rollout == 1
        test_flow = DataFlow.remote("test_env", dataflow_cfg, replay_buffer_cfg, test_env)

        # 2. Initial Run
        await test_flow.run.remote()
        
        # Capture status before saving (critical for partial rollout consistency check)
        rl_status_before_save = await test_flow.get_replaybuffer_status.remote()

        # 3. Save
        save_dir = Path(self.temp_dir.name) / 'checkpoints' / f'ckpt-step-2'
        save_dir.mkdir(parents=True, exist_ok=True)
        await test_flow.save.remote(save_dir)

        # Define run logic based on mode
        async def run_continuation(status_ref):
            if is_partial_rollout:
                remain = status_ref["remain_aborted_samples_count"] + status_ref["remain_completed_samples_count"]
                # Finish the remaining paused samples
                result = await test_flow.run.remote(num=remain, enable_partial_rollout=0)                 
                return result["data_groups"]
            else:
                # Normal run
                result = await test_flow.run.remote()
                return result["data_groups"]

        # continue running after save
        responses_old = await run_continuation(rl_status_before_save)
        rb_status_old = await test_flow.get_replaybuffer_status.remote()


        # resume from saved checkpoint
        await test_flow.resume.remote(save_dir)
        rl_status_resume = await test_flow.get_replaybuffer_status.remote()
        responses_new = await run_continuation(rl_status_resume)
        rb_status_new = await test_flow.get_replaybuffer_status.remote()

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

    async def _run_dense_save_resume_sync_async(self, pg):
        model_path = MODEL_PATH
        worker_log_dir = os.path.join(self.worker_log_dir, "test_dense")
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=model_path,
            model_name=os.path.basename(model_path).lower(),
            tokenizer_path=model_path,
            context_length=self.context_length,
            worker_log_dir=worker_log_dir,
            dist_port_base=37000,
        )
        test_env = SingleTurnEnvironment.remote(
            "test_env",
            pg,
            rollout_cfg=rollout_config,
        )
        sync_dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=0,
            max_concurrent=2,
            max_retry_times=1,
            worker_log_dir=worker_log_dir,
        )
        async_dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=1,
            staleness_threshold=1,
            max_retry_times=1,
            worker_log_dir=self.worker_log_dir,
        )
        replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=self.train_dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            tokenizer=self.tokenizer,
            worker_log_dir=worker_log_dir,
        )
        self._run_dataflow_save_resume_test(test_env, sync_dataflow_cfg, replay_buffer_cfg)
        self._run_dataflow_save_resume_test(test_env, async_dataflow_cfg, replay_buffer_cfg)

    async def _run_moe_save_resume_with_r3(self, pg):
        model_path = MOE_MODEL_PATH
        worker_log_dir = os.path.join(self.worker_log_dir, "test_moe_r3")
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=model_path,
            model_name=os.path.basename(model_path).lower(),
            tokenizer_path=model_path,
            expert_parallel_size=2,
            context_length=self.context_length,
            worker_log_dir=worker_log_dir,
            dist_port_base=36000,
        )
        test_env = SingleTurnEnvironment.remote(
            "test_env",
            pg,
            rollout_cfg=rollout_config,
        )
        async_dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=1,
            max_concurrent=4,
            max_retry_times=1,
            worker_log_dir=worker_log_dir,
        )
        replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=self.train_dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            tokenizer=self.tokenizer,
            worker_log_dir=worker_log_dir,
        )
        self._run_dataflow_save_resume_test(test_env, async_dataflow_cfg, replay_buffer_cfg)

    def _get_sorted_input_ids(self, responses):
        """Helper to extract and sort input_ids from responses."""
        all_ids = []
        for data_items in responses[0]:
            for data_item in data_items:
                all_ids.extend(data_item.data.input_ids)
        all_ids.sort()
        return all_ids

    @unittest.skip("skip lmdeploy turbomind generate test due to ci environment issue")
    def test_lmdeploy_turbomind_generate(self):
        from xtuner.v1.ray.rollout import LMDeployWorker
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
            extra_rollout_config={"lmdeploy_backend": "turbomind"},
        )
        sample_params = SampleParams(temperature=0.0)
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        res2 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        self.assertEqual(res1, res2, f"res1 != res2, res1={res1}, res2={res2}")
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "lmdeploy backend is not enabled")
    def test_sglang_generate(self):
        from xtuner.v1.ray.rollout import SGLangWorker
        self.rollout_cfg.launch_server_method="multiprocessing"
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
            launch_server_method="multiprocessing"
        )
        sample_params = SampleParams(temperature=0.0)
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
        self.assertEqual(res1.finish_reason, "stop")
        print("Response from SGLang infer:", res1)
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "lmdeploy backend is not enabled")
    def test_sglang_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 0
        rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
            launch_server_method="multiprocessing"
        )
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        test_env = SingleTurnEnvironment.remote(
            "test_env",
            pg,
            rollout_cfg=rollout_config,
        )
        test_flow = DataFlow.remote("test_env",
                                         self.dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         test_env
                                        )
        responses = ray.get(test_flow.run.remote(), timeout=300)["data_groups"]
        finished_samples_count = sum(1 for data in responses for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        ray.get(test_env.shutdown.remote(), timeout=300)
        print("responses: ", responses)

if __name__ == "__main__":
    unittest.main()
