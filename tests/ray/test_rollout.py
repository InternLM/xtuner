import os
from functools import wraps
import unittest
import tempfile
import ray
import torch
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
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}
class TestRollout(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # RL默认使用FA3
        os.environ["XTUNER_USE_FA3"] = "1"
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.worker_log_dir = os.path.join(cls.temp_dir.name, "work_dirs")

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        cls.temp_dir.cleanup()

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
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
            launch_server_method="ray",
            context_length=self.max_prompt_length + self.max_response_length,
            worker_log_dir=self.__class__.worker_log_dir,
        )
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        self.judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.__class__.worker_log_dir,
        )
        self.dataflow_cfg = DataFlowConfig(
            env="test",
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=0,
            max_retry_times=1,
            worker_log_dir=self.__class__.worker_log_dir,
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
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=self.train_dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            tokenizer=self.tokenizer,
            worker_log_dir=self.__class__.worker_log_dir,
        )

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        self.model_path = MODEL_PATH
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_generate(self):
        sample_params = SampleParams(temperature=0.0)
        rollout_controller = ray.remote(RolloutController).remote(self.rollout_cfg, self.pg)  # type: ignore[attr-defined]
        res1 = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))
       
        self.assertEqual(res1.finish_reason, "stop") 
        print("Response from LMDeploy infer:", res1)
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 0
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
    
    @unittest.skip("skip lmdeploy async dataflow after lmdeploy support abort_request")
    def test_lmdeploy_async_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 1
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
        extra_params = {"stream": False, "return_token_ids": True, "return_logprobs": True}
        dump_path = os.path.join(self.temp_dir.name, "unfinished_buffer.pickle")
        responses = ray.get(self.test_flow.run.remote(extra_params=extra_params, dump=True, dump_path=dump_path))
        finished_samples_count = sum(1 for data in responses[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        status = ray.get(self.test_flow.get_replaybuffer_status.remote())
        finished_count = status["rollout_finished_count"] # 已经去掉了data_flow返回的数量
        paused_count = status["rollout_paused_count"]
        sample_count = status["action_count"]
        self.assertEqual(len(responses) + finished_count + paused_count, sample_count)
        self.assertEqual(len(responses), self.dataflow_cfg.global_batch_size)

        ray.get(self.test_flow.clear_replaybuffer.remote())
        response_resume = ray.get(self.test_flow.run.remote(extra_params=extra_params, resume=True, resume_path=dump_path))
        finished_resume_samples_count = sum(1 for data in response_resume[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_resume_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        status = ray.get(self.test_flow.get_replaybuffer_status.remote())
        finished_count = status["rollout_finished_count"] 
        paused_count = status["rollout_paused_count"]
        sample_count = status["action_count"]
        self.assertEqual(len(response_resume) + finished_count + paused_count, sample_count)
        self.assertEqual(len(response_resume), self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote())

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
    
    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "lmdeploy backend is not enabled")
    def test_sglang_async_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 1
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
        extra_params = {"stream": True, "return_token_ids": True, "return_logprobs": True}
        dump_path = os.path.join(self.temp_dir.name, "unfinished_buffer.pickle")
        responses = ray.get(self.test_flow.run.remote(extra_params=extra_params, dump=False, dump_path=dump_path))
        finished_samples_count = sum(1 for data in responses[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        status = ray.get(self.test_flow.get_replaybuffer_status.remote())
        finished_count = status["rollout_finished_count"] # 已经去掉了data_flow返回的数量
        paused_count = status["rollout_paused_count"]
        sample_count = status["action_count"]
        self.assertEqual(len(responses) + finished_count + paused_count, sample_count)
        self.assertEqual(len(responses), self.dataflow_cfg.global_batch_size)

        ray.get(self.test_flow.clear_replaybuffer.remote())
        response_resume = ray.get(self.test_flow.run.remote(extra_params=extra_params, resume=True, resume_path=dump_path))
        finished_resume_samples_count = sum(1 for data in response_resume[0] for item in data if item.env.rollout.finish_reason == "stop" or item.env.rollout.finish_reason == "length")
        self.assertEqual(finished_resume_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        status = ray.get(self.test_flow.get_replaybuffer_status.remote())
        finished_count = status["rollout_finished_count"] 
        paused_count = status["rollout_paused_count"]
        sample_count = status["action_count"]
        self.assertEqual(len(response_resume) + finished_count + paused_count, sample_count)
        self.assertEqual(len(response_resume), self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote())

if __name__ == "__main__":
    unittest.main()
