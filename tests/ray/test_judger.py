import os
import json
import ray
import unittest
import tempfile
import numpy as np
import asyncio
from xtuner.v1.ray.base import AutoCPUWorkers, CPUResourcesConfig
from xtuner.v1.data_proto.rl_data import RolloutState

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
GEO_ROLLOUT_DATA_PATH = os.environ["GEO_ROLLOUT_DATA_PATH"]
VERL_ROLLOUT_DATA_PATH = os.environ["VERL_ROLLOUT_DATA_PATH"]
DAPO_DATA_PATH = os.environ.get("ROLLOUT_DAPO_DATA_PATH")
FAKE_JUDGER_INPUT_ITEM = RolloutState(
    message=[{
        'role': 'user',
        'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####"'
    }],
    reward_model={'ground_truth': '72', 'style': 'rule'},
    response="<think>\nOkay, let's see. Natalia sold clips to 48 friends in April. Then in May, she sold half as many. So first, I need to figure out how many she sold in May. Half of 48 is 24, right? Because 48 divided by 2 is 24. So in May, she sold 24 clips.\n\nNow, to find the total number of clips sold in both months, I need to add the number from April and May together. That would be 48 (April) plus 24 (May). Let me do the addition: 48 + 24. Hmm, 40 + 20 is 60, and 8 + 4 is 12. So 60 + 12 is 72. So altogether, she sold 72 clips.\n\nWait, let me check that again. 48 plus 24. Yes, 48 + 20 is 68, then plus 4 more is 72. Yep, that seems right. So the total is 72.\n</think>\n\nNatalia sold 48 clips in April. In May, she sold half as many, which is 48 ÷ 2 = 24 clips. Adding both months together: 48 + 24 = 72.  \n\n#### 72<|im_end|>"
)

def construct_gsm8k_judger_data(data_path) -> tuple[list[RolloutState], list[float]]:
    states = []
    history_reward = []
    if not data_path or not os.path.exists(data_path):
        return states
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            prompt = item["input"][5:-11]
            response = item["output"]
            gt = item["gts"]
            states.append(
                RolloutState(
                    message=[{"role": "user", "content": prompt}],
                    response=response,
                    reward_model={"ground_truth": str(gt)}
                )
            )
            history_reward.append(item["reward"])
    return states, history_reward

def construct_geo3k_dapo_judger_data(data_path) -> tuple[list[RolloutState], list[float]]:
    states = []
    history_reward = []
    if not data_path or not os.path.exists(data_path):
        return states
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 7):
            group = ''.join(lines[i:i + 7]).strip()
            if not group: continue
            item = json.loads(group)
            states.append(
                RolloutState(
                    message=[{"role": "user", "content": ""}],
                    response=item['response'],
                    reward_model={"ground_truth": str(item["label"])}
                )
            )
            history_reward.append(item["reward"])
    return states, history_reward   

class TestJudgerController(unittest.TestCase):

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")

    def tearDown(self): 
        ray.shutdown()
        self.temp_dir.cleanup()

    async def _judger_batch(self, judger_router, states):
        return await asyncio.gather(*(judger_router.judge(s) for s in states))
    
    def test_gsm8k_judger(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        # Test Case 1: NativeJudger
        native_judger = gsm8k_judger_config.build()
        res1 = asyncio.run(native_judger.judge(FAKE_JUDGER_INPUT_ITEM))
        self.assertEqual(res1.reward["score"], 1.0)

        # Test Case 2: NativeJudger with cpu resource + 从外面传入pg
        cpu_cfg = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1)
        pg = AutoCPUWorkers.build_placement_group(cpu_cfg)
        ray.get(pg.ready())
        native_judger_actors = gsm8k_judger_config.build_router(pg, 0)
        res2 = asyncio.run(native_judger_actors.judge(FAKE_JUDGER_INPUT_ITEM))
        self.assertEqual(res2.reward["score"], 1.0)
        del native_judger_actors

        # Test Case 3: NativeJudgerRouter + 一批数据的分数是否正确
        judger_router = gsm8k_judger_config.build_router(pg)
        states, history_reward = construct_gsm8k_judger_data(VERL_ROLLOUT_DATA_PATH)
        rollout_states = asyncio.run(self._judger_batch(judger_router, states))
        rewards = [s.reward["score"] for s in rollout_states]
        expected_avg_score = np.mean(history_reward)
        self.assertEqual(round(np.mean(rewards), 4), round(expected_avg_score, 4))
        
    def test_dapo_batch_judge_score(self):
        # 测试dapo judger + 1个实例 + NativeJudgerRouter的评判分数是否正确
        from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
        from xtuner.v1.utils.rl_test_utils import get_eos_token
        from transformers import AutoTokenizer
        # 构建数据
        states, history_reward = construct_geo3k_dapo_judger_data(DAPO_DATA_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        eos_token = get_eos_token(MODEL_PATH)
        eos_token_str = tokenizer.convert_ids_to_tokens(eos_token)
        # 定义 Judger Config
        config = DapoMathJudgerConfig(
            judger_name="dapo_math",
            eos_token=eos_token_str,
            enable_overlong_buffer=True,
            max_response_len=32768,
            overlong_buffer_len=4096,
            overlong_penalty_factor=1.0,
            tokenizer=tokenizer
        )
        router = config.build_router()
        rollout_states = asyncio.run(self._judger_batch(router, states))
        rewards = [s.reward["score"] for s in rollout_states]
        expected_avg_score = np.mean(history_reward)
        self.assertEqual(round(np.mean(rewards), 4), round(expected_avg_score, 4))

    def test_geo_batch_judge_score(self):
        # 测试geo judger + 4个实例 + NativeJudgerRouter的评判分数是否正确
        from xtuner.v1.ray.judger.geo3k import GEO3KJudgerConfig
        config = GEO3KJudgerConfig(judger_name="geo3k", num_ray_actors=4)
        states, history_reward = construct_geo3k_dapo_judger_data(GEO_ROLLOUT_DATA_PATH)
        router = config.build_router()
        rollout_states = asyncio.run(self._judger_batch(router, states))
        rewards = [s.reward["score"] for s in rollout_states]
        expected_avg_score = np.mean(history_reward)
        self.assertEqual(round(np.mean(rewards), 4), round(expected_avg_score, 4))
        # 验证Router中确实有4个Worker实例在运行
        self.assertEqual(len(router.get_worker_status()), 4)

    def test_multi_judger_router(self):
        import time
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig

        gsm8k_config_1 = GSM8KJudgerConfig(judger_name="openai/gsm8k_1", num_ray_actors=2, num_cpus_per_actor=1)
        gsm8k_config_2 = GSM8KJudgerConfig(judger_name="openai/gsm8k_2", num_ray_actors=8, num_cpus_per_actor=2)

        gsm8k_router_1 = gsm8k_config_1.build_router()
        gsm8k_router_2 = gsm8k_config_2.build_router() 

        states, history_reward = construct_gsm8k_judger_data(VERL_ROLLOUT_DATA_PATH)
        gsm8k_results_1 = asyncio.run(self._judger_batch(gsm8k_router_1, states))
        gsm8k_results_2 = asyncio.run(self._judger_batch(gsm8k_router_2, states)) 

        gsm8k_rewards_1 = [s.reward["score"] for s in gsm8k_results_1]
        gsm8k_rewards_2 = [s.reward["score"] for s in gsm8k_results_2]

        expected_avg_score = np.mean(history_reward)
        self.assertEqual(round(np.mean(gsm8k_rewards_1), 4), round(expected_avg_score, 4))
        self.assertEqual(round(np.mean(gsm8k_rewards_2), 4), round(expected_avg_score, 4))
        self.assertEqual(len(gsm8k_router_1.get_worker_status()), 2)
        self.assertEqual(len(gsm8k_router_2.get_worker_status()), 8)

    def test_gsm8k_remote_judger(self):
        # 测试输入remote_url时 + 1个实例 + 裸的NativeJudger的评判分数是否正确
        from xtuner.v1.utils.rl_test_utils import JudgerServer, GSM8KRemoteJudgerConfig

        server = JudgerServer(port=8018)
        server.start()
        try:
            remote_judger_config = GSM8KRemoteJudgerConfig(judger_name="openai/gsm8k", reward_handler=server.url)
            native_remote_judger = remote_judger_config.build()
            res = asyncio.run(native_remote_judger.judge(FAKE_JUDGER_INPUT_ITEM))
            self.assertEqual(res.reward["score"], 1.0)
        finally:
            server.stop()

if __name__ == "__main__":
    unittest.main()