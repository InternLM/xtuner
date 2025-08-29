import os
import unittest
import ray
from transformers import AutoTokenizer


from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.environment import EnvController, SampleParams
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.datasets import RLTextTokenizeFnConfig
from xtuner.v1.config import DatasetConfig


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]


class TestEvaluator(unittest.TestCase):
    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.max_prompt_length = 512
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=8,
        )
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig()
        self.judger_cfg = JudgerConfig(
            reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
        )
        
        self.eval_dataset_cfg = [
            {
            "dataset": DatasetConfig(name="gsm8k",
                                    anno_path=TEST_DATA_PATH,
                                    sample_ratio=1.0),
            "tokenize_fn": RLTextTokenizeFnConfig(max_length=self.max_prompt_length),
            },
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        self.test_env = EnvController.remote(
            "test_env",
            self.pg,
            self.rollout_cfg,
            self.judger_cfg
        )
        self.sample_params = SampleParams(
            top_p=1.0, 
            temperature=0.0, 
            do_sample=False, 
            max_tokens=1024, 
            top_k=1
        )

    def setUp(self):
        ray.init(num_cpus=80)
        self.model_path = MODEL_PATH
        self.init_config()
        
    def tearDown(self):
        ray.shutdown()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_evaluator(self):
        def custom_compute_metric(samples):
            return {"custom_accuracy": sum(s["reward"] > 0 for s in samples) / len(samples)}

        evaluator_cfg = EvaluatorConfig(
            dataset_cfg=self.eval_dataset_cfg,
            tokenizer=self.tokenizer,
            max_concurrent=1,
            eval_sample_ratio=0.004,  # generate 5 samples
            compute_metric_func=None
        )
        evaluator = Evaluator.remote(evaluator_cfg, self.test_env)
        correctness = ray.get(evaluator.run.remote(sample_params=self.sample_params))

        custom_evaluator_cfg = EvaluatorConfig(
            dataset_cfg=self.eval_dataset_cfg,
            tokenizer=self.tokenizer,
            max_concurrent=1,
            eval_sample_ratio=0.004,  # generate 5 samples
            compute_metric_func=custom_compute_metric
        )
        custom_evaluator = Evaluator.remote(custom_evaluator_cfg, self.test_env)
        custom_correctness = ray.get(custom_evaluator.run.remote(sample_params=self.sample_params))
        self.assertEqual(correctness['accuracy'], custom_correctness['custom_accuracy'])


if __name__ == '__main__':
    unittest.main()