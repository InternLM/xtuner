import os
import copy
import json
import ray
import unittest
import tempfile
import numpy as np
from uuid import uuid4
from xtuner.v1.ray.judger.controller import JudgerController, JudgerConfig
from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLDatasetItem, RLEnvDataItem, RLRolloutResponseItem, RLUIDItem
from xtuner.v1.ray.base import AutoCPUWorkers, CPUResourcesConfig
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
GEO_ROLLOUT_DATA_PATH = os.environ["GEO_ROLLOUT_DATA_PATH"]
VERL_ROLLOUT_DATA_PATH = os.environ["VERL_ROLLOUT_DATA_PATH"]
DAPO_DATA_PATH = os.environ.get("ROLLOUT_DAPO_DATA_PATH")

FAKE_JUDGER_INPUT_ITEM = RLDataFlowItem(
    uid=RLUIDItem(action_id=uuid4().int,
                  observation_id=uuid4().int),
    data=RLDatasetItem(
        messages=[{
            'role': 'user',
            'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####"'
        }],
        num_tokens=62,
        reward_model={'ground_truth': '72', 'style': 'rule'},
        ability='math',
        data_source={'openai/gsm8k': 1.0}
    ),
    env=RLEnvDataItem(
        rollout=RLRolloutResponseItem(
            response="<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many. So first, I need to figure out how many she sold in May. Half of 48 is 24, right? Because 48 divided by 2 is 24. So in May, she sold 24 clips.\n\nNow, to find the total number of clips sold in both months, I need to add the number from April and May together. That would be 48 (April) plus 24 (May). Let me do the addition: 48 + 24. Hmm, 40 + 20 is 60, and 8 + 4 is 12. So 60 + 12 is 72. So altogether, she sold 72 clips.\n\nWait, let me check that again. 48 plus 24. Yes, 48 + 20 is 68, then plus 4 more is 72. Yep, that seems right. So the total is 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is 48 ÷ 2 = 24 clips. Adding both months together: 48 + 24 = 72.  \n\n#### 72<|im_end|>",
        )
    )
)
FAKE_JUDGER_INPUT_ITEM_1 = copy.deepcopy(FAKE_JUDGER_INPUT_ITEM)
FAKE_JUDGER_INPUT_ITEM_1.uid.observation_id = uuid4().int
FAKE_JUDGER_INPUT_ITEM_MULTI_DATA = [FAKE_JUDGER_INPUT_ITEM, FAKE_JUDGER_INPUT_ITEM_1]  # 用action_id来标识是不同的输入数据
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE = copy.deepcopy(FAKE_JUDGER_INPUT_ITEM)
FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE.data.data_source = {'openai/gsm8k-1': 0.5, 'openai/gsm8k-2': 0.5}


def construct_judger_data(data_path):
    dataitem = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line.strip())
            data_item = RLDataFlowItem(
                uid=RLUIDItem(
                    action_id=uuid4().int,
                    observation_id=uuid4().int
                ),
                data=RLDatasetItem(
                    messages=[{
                        'role': 'user',
                        'content': data["input"][5:-11]
                    }],
                    reward_model={"ground_truth": data["gts"]},
                    data_source={"openai/gsm8k": 1.0}
                ),
                env=RLEnvDataItem(
                    rollout=RLRolloutResponseItem(response=data['output'])
                )
            )
            dataitem.append(data_item)
    return dataitem


def construct_new_judger_data(data_path, judger_name='dapo_math'):
    data_item_list = []
    save_reward = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 7):
            group = ''.join(lines[i:i + 7]).strip()
            if group:
                try:
                    item = json.loads(group)
                    data_item = RLDataFlowItem(
                        uid=RLUIDItem(
                            action_id=uuid4().int,
                            observation_id=uuid4().int
                        ),
                        data=RLDatasetItem(
                            messages=[{
                                'role': 'user',
                                'content': ""
                            }],
                            reward_model={"ground_truth": item["label"]},
                            data_source={judger_name: 1.0}
                        ),
                        env=RLEnvDataItem(
                            rollout=RLRolloutResponseItem(response=item['response'])
                        )
                    )
                    data_item_list.append(data_item)
                    save_reward.append(item["reward"])
                except Exception as e:
                    print(f"Error parsing group starting at line {i + 12}: {e}")
    return data_item_list, save_reward


class TestJudgerController(unittest.TestCase):

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")

    def tearDown(self): 
        ray.shutdown()
        self.temp_dir.cleanup()

    def test_gsm8k_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.worker_log_dir
        )
        judger_controller = JudgerController.remote(judger_cfg)
        # 返回的形式为：RLJudgerResponseItem(uid=112750990920317762694895938380669501546, reward={'openai/gsm8k': 1}, extra_info={})
        res1 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM))
        self.assertEqual(res1.reward["score"], 1.0)
        res2 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_DATA))
        self.assertEqual(res2[0].reward["score"], 1.0)
        self.assertEqual(res2[1].reward["score"], 1.0)

    def test_dapo_judger(self):
        from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
        from xtuner.v1.utils.rl_test_utils import get_eos_token
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        eos_token = get_eos_token(MODEL_PATH)
        eos_token_str = tokenizer.convert_ids_to_tokens(eos_token)

        dapo_judger_config = DapoMathJudgerConfig(
            judger_name="dapo_math",
            eos_token=eos_token_str,
            enable_overlong_buffer=True,
            max_response_len=32768,
            overlong_buffer_len=4096,
            overlong_penalty_factor=1.0,
            tokenizer=tokenizer

        )
        judger_cfg = JudgerConfig(
            reward_judger_configs=[dapo_judger_config],
            worker_log_dir=self.worker_log_dir
        )
        judger_controller = JudgerController.remote(judger_cfg)
        judger_data, save_reward = construct_new_judger_data(DAPO_DATA_PATH)
        group_data = ray.get(judger_controller.run.remote(judger_data))
        reward = [data.reward["score"] for data in group_data]
        self.assertEqual(np.mean(reward), np.mean(save_reward))

    def test_geo_judger(self):
        from xtuner.v1.ray.judger.geo3k import GEO3KJudgerConfig
        geo_judger_config = GEO3KJudgerConfig()
        judger_cfg = JudgerConfig(
            reward_judger_configs=[geo_judger_config],
            worker_log_dir=self.worker_log_dir
        )
        judger_controller = JudgerController.remote(judger_cfg)
        judger_data, save_reward = construct_new_judger_data(GEO_ROLLOUT_DATA_PATH, judger_name="hiyouga/geometry3k")
        group_data = ray.get(judger_controller.run.remote(judger_data))
        reward = [data.reward["score"] for data in group_data]
        self.assertEqual(np.mean(reward), np.mean(save_reward))

    def test_gsm8k_multi_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        # 支持一个GSM8KJudgerConfig创建多个实例
        gsm8k_judger_config_1 = GSM8KJudgerConfig(judger_name="openai/gsm8k-1")
        gsm8k_judger_config_2 = GSM8KJudgerConfig(judger_name="openai/gsm8k-2")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[
                gsm8k_judger_config_1,
                gsm8k_judger_config_2
            ],
            enable_weighted_judgers=True,
            worker_log_dir=self.worker_log_dir,
        )
        cpu_resources_config = CPUResourcesConfig.from_total(
            total_cpus=2,
            total_memory=2 * 1024**3,
            num_workers=2
        )
        pg = AutoCPUWorkers.build_placement_group(cpu_resources_config)
        judger_controller = JudgerController.remote(judger_cfg, pg)
        res3 = ray.get(judger_controller.run.remote(FAKE_JUDGER_INPUT_ITEM_MULTI_SOURCE))
        self.assertEqual(res3.reward["weighted_score"], 1.0)  # weighted_score为固定字段，表示加权后的reward

    def test_gsm8k_judger_score(self):
        """Test the judger functionality with single and multiple data sources."""
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config],
            worker_log_dir=self.worker_log_dir
        )
        judger_controller = JudgerController.remote(judger_cfg)
        judger_data = construct_judger_data(VERL_ROLLOUT_DATA_PATH)
        group_data = ray.get(judger_controller.run.remote(judger_data))
        reward = [data.reward["score"] for data in group_data]
        verl_score = 0.2418
        self.assertEqual(round(np.mean(reward), 4), verl_score)

    def test_gsm8k_remote_judger(self):
        from xtuner.v1.utils.rl_test_utils import JudgerServer, GSM8KRemoteJudgerConfig

        server = JudgerServer(port=8018)
        server.start()
        try:
            remote_judger_config = GSM8KRemoteJudgerConfig(judger_name="openai/gsm8k", remote_url=server.url)
            judger_cfg = JudgerConfig(
                reward_judger_configs=[remote_judger_config],
                worker_log_dir=self.worker_log_dir
            )
            judger_controller = JudgerController.remote(judger_cfg)
            judger_data = construct_judger_data(VERL_ROLLOUT_DATA_PATH)
            group_data = ray.get(judger_controller.run.remote(judger_data))
            reward = [data.reward["score"] for data in group_data]
            verl_score = 0.2418
            self.assertEqual(round(np.mean(reward), 4), verl_score)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()