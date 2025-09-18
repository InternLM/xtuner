import os
import argparse
from pathlib import Path
import copy
import json
import ray
import unittest
import numpy as np
from uuid import uuid4
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.judger.controller import JudgerController, JudgerConfig
from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLDatasetItem, RLEnvDataItem, RLRolloutResponseItem, RLUIDItem


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
VERL_ROLLOUT_DATA_PATH = os.environ["VERL_ROLLOUT_DATA_PATH"]

FAKE_JUDGER_INPUT_ITEM = RLDataFlowItem(
    uid = RLUIDItem(action_id=uuid4().int,
                    observation_id=uuid4().int),
    data = RLDatasetItem(
        messages=[{ 
            'role': 'user', 'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####"'
        }],
        num_tokens=62,
        reward_model={'ground_truth': '72', 'style': 'rule'},
        ability='math',
        data_source={'openai/gsm8k': 1.0}
    ),
    env = RLEnvDataItem(
        rollout=RLRolloutResponseItem(
            response="<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many. So first, I need to figure out how many she sold in May. Half of 48 is 24, right? Because 48 divided by 2 is 24. So in May, she sold 24 clips.\n\nNow, to find the total number of clips sold in both months, I need to add the number from April and May together. That would be 48 (April) plus 24 (May). Let me do the addition: 48 + 24. Hmm, 40 + 20 is 60, and 8 + 4 is 12. So 60 + 12 is 72. So altogether, she sold 72 clips.\n\nWait, let me check that again. 48 plus 24. Yes, 48 + 20 is 68, then plus 4 more is 72. Yep, that seems right. So the total is 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is 48 ÷ 2 = 24 clips. Adding both months together: 48 + 24 = 72.  \n\n#### 72<|im_end|>",
        )
    )
)
FAKE_JUDGER_INPUT_ITEM_1 = copy.deepcopy(FAKE_JUDGER_INPUT_ITEM)
FAKE_JUDGER_INPUT_ITEM_1.uid.observation_id = uuid4().int
FAKE_JUDGER_INPUT_ITEM_MULTI_DATA = [FAKE_JUDGER_INPUT_ITEM, FAKE_JUDGER_INPUT_ITEM_1] # 用action_id来标识是不同的输入数据    
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE = copy.deepcopy(FAKE_JUDGER_INPUT_ITEM)
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE.data.data_source = {'openai/gsm8k-1': 0.5, 'openai/gsm8k-2': 0.5}

def construct_judger_data(data_path):
    dataitem = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())
            data_item = RLDataFlowItem(
                uid = RLUIDItem(
                    action_id=uuid4().int,
                    observation_id=uuid4().int
                    ),
                data = RLDatasetItem(
                    messages=[{
                        'role': 'user', 
                        'content': data["input"][5:-11]
                    }],
                    reward_model={"ground_truth": data["gts"]},
                    data_source={"openai/gsm8k": 1.0}
                ),
                env = RLEnvDataItem(
                    rollout=RLRolloutResponseItem(response=data['output'])
                )
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
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config]
        )
        judger_controller = JudgerController.remote(judger_cfg) 
        # 返回的形式为：RLJudgerResponseItem(uid=112750990920317762694895938380669501546, reward={'openai/gsm8k': 1}, extra_info={})
        res1 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM)) 
        self.assertEqual(res1.reward["openai/gsm8k"], 1.0)
        res2 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_DATA))
        self.assertEqual(res2[0].reward["openai/gsm8k"], 1.0)
        self.assertEqual(res2[1].reward["openai/gsm8k"], 1.0)

    def test_gsm8k_multi_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        # 支持一个GSM8KJudgerConfig创建多个实例
        gsm8k_judger_config_1 = GSM8KJudgerConfig(judger_name="openai/gsm8k-1")
        gsm8k_judger_config_2 = GSM8KJudgerConfig(judger_name="openai/gsm8k-2")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[
                gsm8k_judger_config_1,
                gsm8k_judger_config_2
            ]
        )
        judger_controller = JudgerController.remote(judger_cfg)
        res3 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE))
        self.assertEqual(res3.reward["weighted_reward"], 1.0) # weighted_reward为固定字段，表示加权后的reward
        
    def test_gsm8k_judger_score(self):
        """Test the judger functionality with single and multiple data sources."""
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config]
        )
        judger_controller = JudgerController.remote(judger_cfg)
        judger_data = construct_judger_data(VERL_ROLLOUT_DATA_PATH)
        group_data = ray.get(judger_controller.run.remote(judger_data))
        reward = [data.reward["weighted_reward"] for data in group_data]
        avg_score = np.mean(reward)
        verl_score = 0.2418
        self.assertLessEqual(float(np.abs(avg_score - verl_score)), 0.001)

    def test_gsm8k_remote_judger(self):
        from xtuner.v1.utils.rl_test_utils import JudgerServer, GSM8KRemoteJudgerConfig

        server = JudgerServer(port=8018) 
        server.start()

        remote_judger_config = GSM8KRemoteJudgerConfig(judger_name="openai/gsm8k", remote_url=server.url)
        judger_cfg = JudgerConfig(
            reward_judger_configs=[remote_judger_config]
        )
        judger_controller = JudgerController.remote(judger_cfg)
        judger_data = construct_judger_data(VERL_ROLLOUT_DATA_PATH)
        group_data = ray.get(judger_controller.run.remote(judger_data))
        reward = [data.reward["reward"] for data in group_data]
        avg_score = np.mean(reward)
        verl_score = 0.2418
        self.assertLessEqual(float(np.abs(avg_score - verl_score)), 0.001)
        server.stop()
        
    def tearDown(self):
        ray.shutdown()
    

if __name__ == "__main__":
    unittest.main()