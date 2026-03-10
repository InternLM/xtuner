import os
import sys
import json
import socket
import asyncio
import tempfile
import unittest

import ray
import torch
import fastapi
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from recipe.verl_agent.common.agent_loop_verl_tool import VerlToolAgentLoopConfig
from xtuner.v1.rl.agent_loop import AgentLoopManagerConfig, SyncProduceStrategyConfig, SamplerConfig
from xtuner.v1.data_proto import RolloutState, Status, SampleParams
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.judger.gsm8k import GSM8KRouterJudgerConfig
from xtuner.v1.rl.utils import create_task
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
VERL_TRAIN_DATA_PATH = "/fake/path/to/train.parquet"
VERL_TEST_DATA_PATH = "/fake/path/to/test.parquet"

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


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code for tool-calling agent tests."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = self._get_free_port()
        create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]

        _, temp_file = tempfile.mkstemp(
            suffix=".py", prefix="temp_code", dir=None, text=True
        )
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            response = {
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode,
                },
            }
            return JSONResponse(content=response)
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    async def _start_fastapi_server(self):
        app = fastapi.FastAPI()
        app.router.add_api_route(
            "/run_code", self.code_execution, methods=["POST"]
        )
        config = uvicorn.Config(
            app, host=["::", "0.0.0.0"], port=self.port, log_level="warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        return f"{self.address}:{self.port}"


def _build_verl_config(
    model_path: str,
    train_file: str,
    test_file: str,
    tool_config_path: str,
    max_prompt_length: int,
    max_response_length: int,
    rollout_name: str = "sglang",
    tool_call_parser_name: str = "hermes",
):
    from hydra import compose, initialize_config_dir
    import verl

    verl_config_dir = os.path.join(
        os.path.dirname(verl.__file__), "trainer/config"
    )
    with initialize_config_dir(config_dir=verl_config_dir):
        verl_config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.adv_estimator=grpo",
                "data.train_files=" + train_file,
                "data.val_files=" + test_file,
                "data.return_raw_chat=True",
                "data.train_batch_size=32",
                "data.max_prompt_length=" + str(max_prompt_length),
                "data.max_response_length=" + str(max_response_length),
                "+data.apply_chat_template_kwargs.enable_thinking=False",
                "actor_rollout_ref.model.path=" + model_path,
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "actor_rollout_ref.rollout.name=" + rollout_name,
                "actor_rollout_ref.rollout.mode=async",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.n=8",
                "actor_rollout_ref.rollout.response_length=" + str(max_response_length),
                "actor_rollout_ref.rollout.skip_tokenizer_init=False",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
                "+actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen25",
                "actor_rollout_ref.rollout.multi_turn.format=" + tool_call_parser_name,
                "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool_config_path,
                "+actor_rollout_ref.rollout.multi_turn.multi_turn.max_tool_response_length=" + str(max_response_length),
                "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
                "trainer.val_before_train=True",
                "trainer.log_val_generations=10",
                "trainer.n_gpus_per_node=8",
                "trainer.test_freq=-1",
                "trainer.total_training_steps=5",
                "trainer.logger=['console','tensorboard']",
                "trainer.project_name=verl",
                "trainer.experiment_name=test_verl_tool_agent_loop",
            ],
        )
    return verl_config


class TestVerlToolAgentLoop(unittest.IsolatedAsyncioTestCase):
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
            num_workers=1,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.max_prompt_length = 512
        self.max_response_length = 4096
        self.context_length = self.max_prompt_length + self.max_response_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.model_path = MODEL_PATH
        self.data_path = TRAIN_DATA_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    def _setup_sandbox_and_verl_config(self):
        """Create sandbox actor and verl config, return (verl_config, tool_config_path)."""
        sandbox = Sandbox.remote()
        self._sandbox = sandbox
        # TODO: replace with a real sandbox server address
        sandbox_address = ray.get(sandbox.get_server_address.remote())

        tool_config = {
            "tools": [
                {
                    "class_name": "recipe.verl_agent.sandbox_example.sandbox.SandboxTool",
                    "config": {
                        "type": "native",
                        "sandbox_fusion_url": f"http://{sandbox_address}/run_code",
                    },
                },
            ],
        }
        tool_config_path = os.path.join(self.temp_dir.name, "tool_config.json")
        with open(tool_config_path, "w") as f:
            json.dump(tool_config, f)

        verl_config = _build_verl_config(
            model_path=self.model_path,
            train_file=VERL_TRAIN_DATA_PATH,
            test_file=VERL_TEST_DATA_PATH,
            tool_config_path=tool_config_path,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )
        return verl_config

    async def test_verl_tool_agent_loop(self):
        # 1. 初始化 config
        self.init_config()
        verl_config = self._setup_sandbox_and_verl_config()

        rollout_config = RolloutConfig(
            env="test_verl_tool_agent_loop",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
        )
        judger_config = GSM8KRouterJudgerConfig(judger_name="openai/gsm8k")

        training_sample_params = SampleParams(
            max_tokens=self.max_response_length,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            min_tokens=0,
            return_token_ids=True,
            return_logprob=True,
        )
        agent_loop_cfg = VerlToolAgentLoopConfig(
            hf_checkpoint=self.model_path,
            sample_params=training_sample_params,
            config=verl_config,
        )

        # 2. 创建 rollout_controller, judger
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        rollout_controller = ray.remote(RolloutController).remote(
            rollout_config, pg
        )
        gsm8k_judger = judger_config.build()

        # 3. 创建 VerlToolAgentLoop
        agent_loop = agent_loop_cfg.build(
            rollout_controller=rollout_controller, judger=gsm8k_judger
        )

        # 4. 构造输入数据
        prompt_repeat_k = 4
        rollout_state = FAKE_INPUT_ITEM.model_copy(deep=True)
        group_in_rollout_state = [
            FAKE_INPUT_ITEM.model_copy(deep=True) for _ in range(prompt_repeat_k)
        ]

        # 5. 执行 generate_group && generate_sample
        group_rollout_state = await agent_loop.generate_group(group_in_rollout_state)
        single_rollout_state = await agent_loop.generate_sample(rollout_state)
        
        print(f"prompt: {single_rollout_state.extra_fields['raw_prompt']}")
        print(f"response: {single_rollout_state.response}")

        # 6. 验证结果
        self.assertEqual(len(group_rollout_state), prompt_repeat_k)
        for state in group_rollout_state:
            self.assertEqual(state.status, Status.COMPLETED)
            self.assertIsNotNone(state.response_ids)
            self.assertGreater(len(state.response_ids), 0)
            self.assertIsNotNone(state.prompt_ids)
            self.assertIsNotNone(state.logprobs)
            self.assertIsNotNone(state.loss_mask)

        self.assertEqual(single_rollout_state.status, Status.COMPLETED)
        self.assertIsNotNone(single_rollout_state.response_ids)
        self.assertGreater(len(single_rollout_state.response_ids), 0)
        self.assertIsNotNone(single_rollout_state.prompt_ids)
        self.assertIsNotNone(single_rollout_state.logprobs)
        self.assertIsNotNone(single_rollout_state.loss_mask)

    async def test_verl_tool_agent_loop_manager(self):
        # 1. 初始化 config
        self.init_config()
        verl_config = self._setup_sandbox_and_verl_config()

        rollout_config = RolloutConfig(
            env="test_verl_tool_agent_loop_manager",
            model_path=self.model_path,
            model_name=os.path.basename(self.model_path).lower(),
            tokenizer_path=self.model_path,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
        )
        judger_config = GSM8KRouterJudgerConfig(judger_name="openai/gsm8k")

        training_sample_params = SampleParams(
            max_tokens=self.max_response_length,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            min_tokens=0,
        )
        agent_loop_cfg = VerlToolAgentLoopConfig(
            hf_checkpoint=self.model_path,
            sample_params=training_sample_params,
            config=verl_config,
        )

        prompt_repeat_k = 2
        sampler_config = SamplerConfig(
            dataloader_cfg=DataloaderConfig(
                dataset_config_list=[
                    {
                        "dataset": DatasetConfig(
                            name="gsm8k",
                            anno_path=TRAIN_DATA_PATH,
                            sample_ratio=1.0,
                        ),
                        "tokenize_fn": RLTextTokenizeFnConfig(
                            max_length=self.max_prompt_length
                        ),
                    },
                ],
                collator="fake_collator",
                pack_level="none",
                group_by_length=False,
            ),
            prompt_repeat_k=prompt_repeat_k,
        )
        agent_loop_manager_cfg = AgentLoopManagerConfig(
            task_name="test_verl_tool",
            agent_loop_config=agent_loop_cfg,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=sampler_config,
        )

        # 2. 创建 rollout_controller, judger
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        rollout_controller = ray.remote(RolloutController).remote(
            rollout_config, pg
        )
        gsm8k_judger = judger_config.build()

        # 3. 创建 AgentLoopManager
        replay_buffer_cfg = SyncReplayBufferConfig()
        replay_buffer = replay_buffer_cfg.build()
        agent_loop_manager = agent_loop_manager_cfg.build(
            rollout_controller=rollout_controller,
            judger=gsm8k_judger,
            tokenizer=self.tokenizer,
            replay_buffer=replay_buffer,
        )

        # 4. 执行 produce_batch
        batch_rollout_states = await agent_loop_manager.produce_batch(batch_size=4)

        # 5. 验证结果
        self.assertEqual(len(batch_rollout_states), 4)
        for group_state in batch_rollout_states:
            self.assertEqual(len(group_state), prompt_repeat_k)
            group_message = group_state[0].message
            for state in group_state:
                self.assertEqual(state.status, Status.COMPLETED)
                self.assertIsNotNone(state.response_ids)
                self.assertGreater(len(state.response_ids), 0)
                self.assertEqual(state.message, group_message)
                self.assertIsNotNone(state.prompt_ids)
                self.assertIsNotNone(state.logprobs)
                self.assertIsNotNone(state.loss_mask)


if __name__ == "__main__":
    unittest.main()
