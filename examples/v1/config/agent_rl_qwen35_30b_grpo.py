import json
import os
from copy import deepcopy

import ray
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    SampleParams,
)
from xtuner.v1.datasets import DatasetConfig
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.ray.base import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers
)
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment.install_agent_env import InstallAgentEnvironment
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.base.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.agent_rl_trainer import AgentRLTrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig

from projects.claw_bench.claw_tokenize_fn import RLClawTokenizeFnConfig

# export RL_LLM_MODEL='xtuner-qwen35-30b'
# bash examples/v1/scripts/run_rl.sh examples/v1/config/agent_rl_qwen35_30b_grpo.py "lmdeploy" $QWEN3P5_VL_MODEL_PATH $TRAIN_DATA_PATH
experimental_name = 'agent_rl_qwen3.5_30b_grpo'
work_dir = os.environ["WORK_DIR"]
model_path = os.environ["MODEL_PATH"]
model_name = os.environ["RL_LLM_MODEL"]

# basic settings
global_batch_size = 8
prompt_repeat_k = 2
train_optimizer_steps = 8  # mini batch steps
max_concurrent_groups = 512

max_prompt_length = 4096
pack_max_length = 68 * 1024
max_response_length = 64 * 1024

train_ep_size = 1
rollout_tp_size = 2
rollout_ep_size = 1
enable_float8_rollout = False
rollout_max_batch_size = 1024
max_prefill_token_num = 1024

enable_return_routed_experts = False

enable_partial_rollout = False
auto_resume = False
skip_load_weights = False
lr = 1e-6
hf_interval = 5
total_epochs = 10
sp_size = 1
# evaluation settings
enable_evaluate = False
enable_initial_evaluate = False
evaluate_step = 5

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=2,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    model_name=model_name,
    device=resources.accelerator,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.7,
    enable_float8=enable_float8_rollout,
    skip_load_weights=skip_load_weights,
    context_length=max_response_length,
    rollout_max_batch_size_per_instance=rollout_max_batch_size,
    allow_over_concurrency_ratio=2,
    rollout_timeout=36000,
    enable_return_routed_experts=enable_return_routed_experts,
    max_prefill_token_num=max_prefill_token_num,
    extra_rollout_config=dict(lmdeploy_log_level="INFO", lmdeploy_uvicorn_log_level="INFO"),
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length, top_k=0, top_p=0.999, temperature=1.0, min_tokens=0
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8

# 2. dataset
def parse_xpuyu_json_cfg(path, max_prompt_length):
    with open(path, "r") as f:
        json_cfg = json.load(f)
    converted_cfg = []
    for ds_name, ds_cfg in json_cfg.items():
        annotation = ds_cfg["annotation"]
        if isinstance(annotation, str):
            annotation = [annotation]
        for ann in annotation:
            converted_cfg.append(
                {
                    "dataset": DatasetConfig(
                        name=ds_name,
                        anno_path=ann,
                        sample_ratio=ds_cfg["sample_ratio"],
                        class_name='JsonlDataset',
                    ),
                    "tokenize_fn": RLClawTokenizeFnConfig(
                        root_path=ds_cfg.get("root_path", None),
                        max_length=max_prompt_length
                    ),
                }
            )
    return converted_cfg


train_dataset_cfg = parse_xpuyu_json_cfg(os.environ['TRAIN_DATA_PATH'], max_prompt_length)
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")


def prepare_agent_inputs(env, group_data_item: RLDataFlowItem):
    return group_data_item


def convert_rollout_tractory_to_train(env, group_data_items):
    return group_data_items


pg = AutoAcceleratorWorkers.build_placement_group(resources)
rollout_controller = ray.remote(max_concurrency=1000)(RolloutController).remote(rollout_config, pg)
load_checkpoint_cfg = LoadCheckpointConfig(load_optimizer_states=False, load_optimizer_args=False)


environment_config = dict(
    type=InstallAgentEnvironment,
    environment='claw-bench',
    rollout_controller=rollout_controller,
    preprocess_func=prepare_agent_inputs,
    postprocess_func=convert_rollout_tractory_to_train,
)


# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    max_concurrent=max_concurrent_groups,
    enable_partial_rollout=enable_partial_rollout,
    max_retry_times=3,
    prompt_repeat_k=prompt_repeat_k,
    global_batch_size=global_batch_size,
    sample_params=training_sample_params,
)

evaluator_cfg = None

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=model_path,
    # postprocessor_func=group_sample_filter_func,
)

# # 5. Train worker
model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
model_cfg.compile_cfg = False
model_cfg.text_config.freeze_routers = True
model_cfg.text_config.balancing_loss_cfg = None

optim_cfg = AdamWConfig(
    lr=lr,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=0.9,
    eps=1e-15,
)
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type="vanilla",
    ),
    ignore_idx=-100,
    use_kl_loss=False,
    kl_loss_coef=0.0,
    kl_loss_type="low_var_kl",
    mode="chunk",
    chunk_size=512,
    rollout_is=RolloutImportanceSampling(
        rollout_is_level="token",
        rollout_is_mode="both",
        rollout_is_threshold=(5, 0),
        rollout_is_mask_threshold=(5, 0.5),
        rollout_is_veto_threshold=(20, 0),
    ),
)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=lr)
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size)
train_worker_cfg: WorkerConfig = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=sp_size,
    optimizer_steps=train_optimizer_steps,
    pack_max_length=pack_max_length,
)

# 6. RL Trainer
trainer = AgentRLTrainerConfig(
    load_from=model_path,
    pg=pg,
    environment_config=environment_config,
    dataflow_config=dataflow_config,
    replay_buffer_config=replay_buffer_cfg,
    train_worker_cfg=train_worker_cfg,
    evaluator_config=evaluator_cfg,
    tokenizer_path=model_path,
    work_dir=work_dir,
    total_epochs=total_epochs,
    hf_interval=hf_interval,
    skip_load_weights=skip_load_weights,
    auto_resume=auto_resume,
    checkpoint_interval=2,
    checkpoint_maxkeep=1,
    load_checkpoint_cfg=load_checkpoint_cfg,
    checkpoint_no_save_optimizer=True,
    skip_checkpoint_validation=True,
)
