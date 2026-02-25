import os
import unittest
import asyncio
import ray
import tempfile
import torch
from transformers import AutoTokenizer
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.base.agent_loop_manager import AgentLoopManager
from xtuner.v1.data_proto import RolloutState, Status, SampleParams 
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
from xtuner.v1.rl.base.producer import SyncProduceStrategy, Sampler
from xtuner.v1.rl.base.replay_buffer import ReplayBuffer
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
MOE_MODEL_PATH = os.environ["QWEN3_MOE_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
FAKE_INPUT_ITEM = RolloutState(
    message=[{
        'role': 'user',
        'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####"'
    }],
    reward_model={'ground_truth': '72', 'style': 'rule'},
)
resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}

class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.rollout_config = RolloutConfig(
            env="test_agent_loop",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
        )

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    # async def test_gsm8k_agent_loop(self):
    #     # 1. 创建 rollout_controller, judger
    #     pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
    #     rollout_controller = ray.remote(RolloutController).remote(self.rollout_config, pg)
    #     gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
    #     gsm8k_judger = gsm8k_judger_config.build_router()
    #     # 2. 创建 AgentLoop
    #     sample_params = SampleParams(max_tokens=self.max_response_length, temperature=0.0)
    #     agent_loop = SingleTurnAgentLoop(rollout_ctl=rollout_controller, sample_params=sample_params, hf_checkpoint=self.model_path, judger=gsm8k_judger)
    #     # 3. 构造输入数据
    #     rollout_state = FAKE_INPUT_ITEM
    #     # 4. 执行 generate_group && generate_sample
    #     group_rollout_state = await agent_loop.generate_group(rollout_state, prompt_repeat_k=4)
    #     single_rollout_state = await agent_loop.generate_sample(rollout_state)
    #     # 5. 验证结果
    #     self.assertEqual(len(group_rollout_state), 4)
    #     for state in group_rollout_state:
    #         self.assertEqual(state.status, Status.COMPLETED)
    #         self.assertGreater(len(state.response_ids), 0)
    #         self.assertEqual(single_rollout_state.reward["score"], 1)  
    #     self.assertEqual(single_rollout_state.status, Status.COMPLETED)
    #     self.assertGreater(len(single_rollout_state.response_ids), 0)
    #     self.assertEqual(single_rollout_state.reward["score"], 1)  

    async def test_gsm8k_agent_loop_manager(self):
        # 1. 创建 rollout_controller, judger
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        rollout_controller = ray.remote(RolloutController).remote(self.rollout_config, pg)
        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        gsm8k_judger = gsm8k_judger_config.build_router()
        # 2. 创建 AgentLoop
        sample_params = SampleParams(max_tokens=self.max_response_length, temperature=0.0)
        agent_loop = SingleTurnAgentLoop(rollout_ctl=rollout_controller, sample_params=sample_params, hf_checkpoint=self.model_path, judger=gsm8k_judger)
        # 3. 创建 AgentLoopManager
        replay_buffer = ReplayBuffer()
        stragegy = SyncProduceStrategy(replay_buffer=replay_buffer)
        dataloader_cfg = DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(name="gsm8k",
                                            anno_path=TRAIN_DATA_PATH,
                                            sample_ratio=1.0),
                    "tokenize_fn": RLTextTokenizeFnConfig(max_length=self.max_prompt_length),
                },
            ],
            collator='fake_collator',
            pack_level='none',
            group_by_length=False,
        )
        sampler = Sampler(task_name="test_gsm8k", dataloader_cfg=dataloader_cfg, tokenizer=self.tokenizer)
        agent_loop_manager = AgentLoopManager(agent_loop=agent_loop, replay_buffer=replay_buffer, sampler=sampler, produce_strategy=stragegy)
        # 4. 执行 produce_batch
        batch_rollout_states = await agent_loop_manager.produce_batch(task_name="test_gsm8k", batch_size=4, prompt_k=2)
        # 5. 验证结果
        self.assertEqual(len(batch_rollout_states), 4)
        for group_state in batch_rollout_states:
            self.assertEqual(len(group_state), 2)
            group_message = group_state[0].message  
            for state in group_state:
                self.assertEqual(state.status, Status.COMPLETED)
                self.assertGreater(len(state.response_ids), 0)
                self.assertEqual(state.message, group_message)

if __name__ == "__main__":
    unittest.main()