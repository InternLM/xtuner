"""RL Colocate Trainer 示例配置（GRPO + GSM8K）。

用法：通过环境变量传入路径后，由 CLI 加载本配置并 trainer_cfg.build().fit()。
需设置: WORK_DIR, MODEL_PATH, DATA_PATH, EVAL_DATA_PATH
可选: WORLD_SIZE, ENABLE_RETURN_ROUTED_EXPERTS, LOSS_TYPE, LOSS_MODE, SP_SIZE
"""
import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.gsm8k import GSM8KRouterJudgerConfig
from xtuner.v1.ray.utils import create_task
from xtuner.v1.rl.base.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.base.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.base.agent_loop_manager import AgentLoopManagerConfig
from xtuner.v1.rl.base.producer import SyncProduceStrategyConfig
from xtuner.v1.rl.base.sampler import SamplerConfig
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_colocate_trainer import RLColocateTrainerConfig
from recipe.verl_agent.common.agent_loop_verl_tool import VerlToolAgentLoopConfig
# env
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
data_path = os.environ["DATA_PATH"]
eval_data_path = os.environ["EVAL_DATA_PATH"]
enable_return_routed_experts = os.environ.get("ENABLE_RETURN_ROUTED_EXPERTS", "0")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

# basic settings
experimental_name = "grpo_gsm8k_verl_tool"
rollout_steps = 45
evaluate_step = 45
train_optimizer_steps = 1
global_batch_size = 64 * train_optimizer_steps
prompt_repeat_k = 5
rollout_tp_size = 1
rollout_ep_size = 1
max_prompt_length = 512
max_response_length = 2048
pack_max_length = 32 * 1024

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8 * WORLD_SIZE,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.8,
    context_length=max_response_length + max_prompt_length,
    enable_return_routed_experts=(enable_return_routed_experts == "1"),
)

# 3. judger
judger_config = GSM8KRouterJudgerConfig(judger_name="openai/gsm8k")

# 4. train worker
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
model_cfg = get_model_config_from_hf(Path(model_path))
if hasattr(model_cfg, "balancing_loss_cfg"):
    model_cfg.balancing_loss_cfg = None
if hasattr(model_cfg, "z_loss_cfg"):
    model_cfg.z_loss_cfg = None
optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.28,
        cliprange_low=0.2,
        loss_type=os.environ.get("LOSS_TYPE", "vanilla"),
        clip_ratio_c=10.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=512,
)
train_worker_cfg = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# code sand box just for toy example
import ray
import asyncio
import socket
import tempfile
import sys
import fastapi
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import json

@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code."""

    def __init__(self):
        self.address = ray._private.services.get_node_ip_address()
        self.port = self._get_free_port()
        create_task(self._start_fastapi_server())

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        # print(f"execute code:\n{code}")

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
        app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])

        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        return f"{self.address}:{self.port}"

sandbox = Sandbox.remote()
sandbox_address = ray.get(sandbox.get_server_address.remote())
print(f"Sandbox server address: {sandbox_address}")
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

tool_config_path = "tool_config.json"
with open(tool_config_path, "w") as f:
    json.dump(tool_config, f)

# 5.0 verl config
rollout_name = "sglang"
tool_call_parser_name = "hermes"
train_file = "/fake/path/to/train.parquet"
test_file = "/fake/path/to/test.parquet"

from hydra import compose, initialize_config_dir
import verl

verl_config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer/config")
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
            # actor related
            "actor_rollout_ref.model.path=" + model_path,
            "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            # rollout related
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
            # hydra.errors.ConfigCompositionException: Could not override 'actor_rollout_ref.rollout.multi_turn.multi_turn.max_tool_response_length'.
            # To append to your config use +actor_rollout_ref.rollout.multi_turn.multi_turn.max_tool_response_length=10384
            "+actor_rollout_ref.rollout.multi_turn.multi_turn.max_tool_response_length=" + str(max_response_length),
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            # trainer related
            "trainer.val_before_train=True",
            "trainer.log_val_generations=10",
            "trainer.n_gpus_per_node=8",
            "trainer.test_freq=-1",
            "trainer.total_training_steps=5",
            "trainer.logger=['console','tensorboard']",
            "trainer.project_name=verl",
            "trainer.experiment_name=" + experimental_name,
        ],
    )

# 5.1 train agent loop
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    min_tokens=0,
)
verl_tool_agent_loop_config = VerlToolAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
    config=verl_config,
)

# 5.2 train agent loop manager
train_dataset = DatasetConfig(name=experimental_name, anno_path=data_path)
tokenizer_config = RLTextTokenizeFnConfig(max_length=max_prompt_length)
train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=train_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
sampler_config = SamplerConfig(
    dataloader_cfg=dataloader_cfg,
    prompt_repeat_k=prompt_repeat_k,
)
produce_strategy_config = SyncProduceStrategyConfig()
agent_loop_manager_cfg = AgentLoopManagerConfig(
    task_name="train_task",
    agent_loop_config=verl_tool_agent_loop_config,
    produce_strategy_config=produce_strategy_config,
    sampler_config=sampler_config,
)

# 6.1 eval agent loop
evaluation_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=1,
    top_p=1.0,
    temperature=0.0,
    min_tokens=0,
)
eval_verl_tool_agent_loop_config = VerlToolAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=evaluation_sample_params,
    config=verl_config,
)

# 6.2 eval agent loop manager
eval_dataset = DatasetConfig(
    name=experimental_name, anno_path=eval_data_path, sample_ratio=1.0
)
eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}]
eval_dataloader_cfg = DataloaderConfig(
    dataset_config_list=eval_dataset_cfg,
    pack_max_length=pack_max_length,
    collator="fake_collator",
    pack_level="none",
)
eval_sampler_config = SamplerConfig(
    dataloader_cfg=eval_dataloader_cfg,
    prompt_repeat_k=1,
)
eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
    task_name="eval_task",
    agent_loop_config=eval_verl_tool_agent_loop_config,
    sampler_config=eval_sampler_config,
)

# 7. evaluator
evaluator_config = EvaluatorConfig(compute_metric_func=None)

# 8. RL Colocate Trainer Config（CLI 通过 config["trainer"].build() 得到 Trainer）
trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,  # TODO: uniform naming of cfg and config
    rollout_config=rollout_config,
    judger_config=judger_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    rollout_steps=rollout_steps,
    global_batch_size=global_batch_size,
    enable_evaluate=True,
    enable_initial_evaluate=False,
    evaluate_step=evaluate_step,
    work_dir=work_dir,
    seed=123,
    debug_rollout=False,
)
