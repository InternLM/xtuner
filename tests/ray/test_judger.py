import os
import argparse
from pathlib import Path
import copy
import json
import ray
import unittest
import numpy as np

from xtuner.v1.ray.environment import EnvController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.datasets.data_item import RLTextDataItem


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
VERL_ROLLOUT_DATA_PATH = os.environ["VERL_ROLLOUT_DATA_PATH"]

FAKE_ROLLOUT_INPUT_STR = '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n'
FAKE_INPUT_DATA_ITEM = {
    'prompt_str': '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n',
    'num_tokens': 62,
    'reward_model': {'ground_truth': '72', 'style': 'rule'},
    'ability': 'math',
    'data_source': {'openai/gsm8k': 1.0},
    'extra_info': {'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', 'index': 0, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'split': 'train', 'raw_prompt': '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n'},
    'env': 'test_env',
    'group_id': 255971142656329732139546771377476227093,
    'prompt_id': 22175756018538642401581407443664245296,
    'retry_times': 0}

FAKE_JUDGER_INPUT_ITEM = copy.deepcopy(FAKE_INPUT_DATA_ITEM)
FAKE_JUDGER_INPUT_ITEM["response_str"] = "<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many. So first, I need to figure out how many she sold in May. Half of 48 is 24, right? Because 48 divided by 2 is 24. So in May, she sold 24 clips.\n\nNow, to find the total number of clips sold in both months, I need to add the number from April and May together. That would be 48 (April) plus 24 (May). Let me do the addition: 48 + 24. Hmm, 40 + 20 is 60, and 8 + 4 is 12. So 60 + 12 is 72. So altogether, she sold 72 clips.\n\nWait, let me check that again. 48 plus 24. Yes, 48 + 20 is 68, then plus 4 more is 72. Yep, that seems right. So the total is 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is 48 ÷ 2 = 24 clips. Adding both months together: 48 + 24 = 72.  \n\n#### 72"
FAKE_JUDGER_INPUT_ITEM_MULTI_DATA = [FAKE_JUDGER_INPUT_ITEM] * 2
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE = copy.deepcopy(FAKE_JUDGER_INPUT_ITEM)
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE['data_source'] = {'openai/gsm8k-1': 0.5, 'openai/gsm8k-2': 0.5}

def construct_judger_data(data_path):
    dataitem = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除行尾的空白字符并解析JSON
            data = json.loads(line.strip())
            data_item = RLTextDataItem(
                prompt_str=data['input'],
                reward_model={"ground_truth": data["gts"]},
                response_str=data["output"],
                data_source={"openai/gsm8k": 1.0}
            )
            dataitem.append(data_item)
    return dataitem

class TestJudgerController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize Ray and placement group once for all tests."""
        assert MODEL_PATH, "Environment variable ROLLOUT_MODEL_PATH is not set."
        ray.init(num_cpus=80, ignore_reinit_error=True)
        resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        cls.pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after all tests are done."""
        ray.shutdown()

    def test_gsm8k_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig()
        judger_cfg = JudgerConfig(
            reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
        )
        test_judger_env = EnvController.remote(
            "test_judger",
            self.pg,
            rollout_cfg = None,
            judger_cfg = judger_cfg
        )
        res1 = ray.get(test_judger_env.run.remote(FAKE_JUDGER_INPUT_ITEM))
        self.assertEqual(res1["reward"], 1.0)
        res2 = ray.get(test_judger_env.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_DATA))
        self.assertEqual(res2[0]["reward"], 1.0)
        self.assertEqual(res2[1]["reward"], 1.0)

    def test_gsm8k_multi_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config_1 = GSM8KJudgerConfig()
        gsm8k_judger_config_2 = GSM8KJudgerConfig()
        judger_cfg = JudgerConfig(
            reward_judger_configs={
                "openai/gsm8k-1": gsm8k_judger_config_1,
                "openai/gsm8k-2": gsm8k_judger_config_2,}
        )
        test_multi_judger_env = EnvController.remote(
            "test_multi_judger",
            self.pg,
            rollout_cfg = None,
            judger_cfg = judger_cfg
        )
        res3 = ray.get(test_multi_judger_env.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE))
        self.assertEqual(res3["reward"], 1.0)
        
    def test_gsm8k_judger_score(self):
        """Test the judger functionality with single and multiple data sources."""
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig()
        judger_cfg = JudgerConfig(
            reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
        )
        test_judger_env = EnvController.remote(
            "test_judger", self.pg, rollout_cfg=None, judger_cfg=judger_cfg
        )
        judger_test_data_path="/cpfs01/shared/llm_razor/duanyanhui/workspace/verl/outputs/0.jsonl"
        judger_data = construct_judger_data(VERL_ROLLOUT_DATA_PATH)
        group_data = ray.get(test_judger_env.run.remote(judger_data))
        reward = []
        for data in group_data:
            reward.append(data["reward"])
        avg_score = np.mean(reward)
        verl_score = 0.2418
        self.assertLessEqual(float(np.abs(avg_score - verl_score)), 0.001)

    def tearDown(self):
        ray.shutdown()
    

if __name__ == "__main__":
    unittest.main()