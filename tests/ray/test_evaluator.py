import os
import unittest
import ray
import tempfile
from transformers import AutoTokenizer

from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets import RLTokenizeFnConfig, DatasetConfig, OpenaiTokenizeFunctionConfig


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]


class TestEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]

    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
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
            tensor_parallel_size=8,
            context_length=self.max_prompt_length + self.max_response_length,
            worker_log_dir=self.worker_log_dir
        )
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        self.judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.worker_log_dir
        )
        self.eval_dataset_cfg = [
            {
            "dataset": DatasetConfig(name="gsm8k",
                                    anno_path=TEST_DATA_PATH,
                                    sample_ratio=1.0),
            "tokenize_fn": RLTokenizeFnConfig(max_length=self.max_prompt_length)
            },
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            self.rollout_cfg,
            None,
            self.judger_cfg
        )
        self.sample_params = SampleParams(
            top_p=1.0, 
            temperature=0.0,
            max_tokens=self.max_response_length, 
            top_k=1
        )

    def setUp(self):
        ray.init(num_cpus=80)
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_evaluator(self):
        def custom_compute_metric(samples):
            return {"custom_accuracy": sum(s.env.judger.reward["score"] > 0 for s in samples) / len(samples)}

        evaluator_cfg = EvaluatorConfig(
            dataset_cfg=self.eval_dataset_cfg,
            tokenizer=self.tokenizer,
            max_concurrent=16,
            eval_sample_ratio=0.004,  # generate 5 samples
            compute_metric_func=custom_compute_metric,
            sample_params=self.sample_params,
            worker_log_dir=self.worker_log_dir
        )
        evaluator = Evaluator.remote(evaluator_cfg, self.test_env)
        try:
            ray.get(evaluator.run.remote())
        except Exception as e:
            self.fail(f"evaluator.run.remote() raised an exception: {e}")
        
if __name__ == '__main__':
    unittest.main()
