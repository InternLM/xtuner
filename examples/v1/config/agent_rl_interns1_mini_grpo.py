import os

os.environ["XTUNER_USE_LMDEPLOY"] = "1"

# os.environ["HF_HOME"] = "/mnt/shared-storage-user/liukuikun/.cache/huggingface"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
from copy import deepcopy
from functools import partial

import ray
from lagent.actions.mcp_client import AsyncMCPClient
from lagent.agents.fc_agent import FunctionCallAgent, get_tool_prompt
from ray.util.placement_group import placement_group

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import (
    RLAgentDataItem,
    RLDataFlowItem,
    RLJudgerResponseItem,
    RLRolloutResponseItem,
    RolloutState,
    SampleParams,
    update_dataflow_item,
)
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.model import Qwen3VLMoE30BA3Config
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.ray.base import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourcesConfig,
)
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment.agent_env import AgentEnvironment
from xtuner.v1.ray.environment.composed_env import ComposedEnvironment
from xtuner.v1.ray.environment.lagent.agents import (
    AsyncTokenInOutAgent,
    EnvAgent,
    JudgerWrapper,
    finish_condition_func,
)
from xtuner.v1.ray.environment.lagent.llms.controller_wrapper import ControllerWrapper
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.compass_verifier_v2 import CompassVerifierV2Config
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.base.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.agent_rl_trainer import AgentRLTrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils.compute_metric import compute_metric

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_DEBUG_POST_MORTEM": "0"}})

experimental_name = os.path.basename(__file__).split(".py")[0]
base_work_dir = os.environ["BASE_WORK_DIR"]
work_dir = os.path.join(base_work_dir, experimental_name)

model_path = os.environ["MODEL_PATH"]
stop_word = "<|im_end|>"

# basic settings
global_batch_size = 8
prompt_repeat_k = 8
max_concurrent_groups = 512

max_prompt_length = 4096
pack_max_length = 68 * 1024
max_response_length = 64 * 1024

train_ep_size = 1
rollout_tp_size = 4
rollout_ep_size = 1
enable_float8_rollout = False
rollout_max_batch_size = 1024
max_prefill_token_num = 1024
enable_return_routed_experts = True
enable_partial_rollout = False
auto_resume = False
skip_load_weights = False
lr = 1e-6
train_optimizer_steps = 8  # mini batch steps
hf_interval = 5
total_epochs = 10
sp_size = 2
# evaluation settings
enable_evaluate = True
enable_initial_evaluate = False
evaluate_step = 5

# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)
judger_cpu_resources = CPUResourcesConfig.from_total(total_cpus=16, num_workers=16, total_memory=64 * 1024**3)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
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
    router_n_groups=8,
    max_prefill_token_num=max_prefill_token_num,
    extra_rollout_config=dict(lmdeploy_log_level="INFO", lmdeploy_uvicorn_log_level="INFO"),
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length,
    top_k=0,
    top_p=0.999,
    temperature=1.0,
    min_tokens=0,
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8
# evaluation_sample_params.max_tokens = max_response_length

data_judger_mapping = {
    'GAIA_sft_1229': {"compass_verifier_v2": 1.0},
    'gaia-level1': {"compass_verifier_v2": 1.0},
    'gaia-level2': {"compass_verifier_v2": 1.0},
    'gaia-level3': {"compass_verifier_v2": 1.0},
    'BrowseComp-ZH': {"compass_verifier_v2": 1.0},
    'HLE': {"compass_verifier_v2": 1.0},
}
tokenize_fn_cfg = Qwen3VLTokenizeFnConfig(
    processor_path=model_path,
    min_pixels=None,
    # max_pixels=None,
    # max_pixels=2097152,
    video_min_total_pixels=None,
    video_max_total_pixels=None,
    video_min_frames=None,
    video_max_frames=None,
    fps=None,
    rand_video_max_frames=24,
    add_vision_id=True,
    system_message=None,
    hash=None,
    enable_3d_rope=False,
    oss_loader_cfg=None,
    debug=True,
    oss_time_log_thr=10,
)

# 2. dataset
from xtuner.v1.datasets.rl_tokenize_fn.xpuyu_dataset_vl import parse_xpuyu_json_cfg

train_dataset_cfg = parse_xpuyu_json_cfg(
    os.environ['TRAIN_DATA_PATH'], tokenize_fn_cfg, max_prompt_length, data_judger_mapping
)
eval_dataset_cfg = parse_xpuyu_json_cfg(
    os.environ['EVAL_DATA_PATH'], tokenize_fn_cfg, max_prompt_length, data_judger_mapping, ignore_multimodal_info=True
)
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")

# 3. judger
judger_cfg = JudgerConfig(reward_judger_configs=[CompassVerifierV2Config(hosts=[])])


def prepare_agent_inputs(env, group_data_item: RLDataFlowItem, is_training=False):
    env_agent, session_id = env.agent.env_agent, group_data_item.uid.observation_id
    user_prompt = group_data_item.data.messages[-1]['content']
    if is_training:
        group_data_item.data.reward_model['ground_truth'] = group_data_item.data.reward_model['ground_truth']['target']
    env_message = AgentMessage(role="env", content=user_prompt)
    if session_id not in env_agent.memory.memory_map or not env_agent.memory.get_memory(session_id):
        set_env_message = AgentMessage(role="env", content=group_data_item)
        env_agent.update_memory(set_env_message, session_id=session_id)
    return (env_message,)


def convert_rollout_tractory_to_train(env, group_data_items):
    agent_data_items, rollout_response_items, judger_response_items = [], [], []
    for i in range(len(group_data_items)):
        session_id = group_data_items[i].uid.observation_id
        history = env.agent.select_agent.state_dict(session_id=session_id)['memory']
        messages = env.agent.get_messages(session_id, keypath='select_agent')
        agent_data_items.append(RLAgentDataItem(extra_info=dict(messages=messages, state={"history": history})))
        rollout_response_items.append(
            RLRolloutResponseItem(
                response=history[-1]['raw_content'],
                response_ids=history[-1]['raw_content_ids'],
                logprobs=history[-1]['raw_content_logprobs'],
                state=RolloutState.COMPLETED,
            )
        )
        env_history = env.agent.env_agent.state_dict(session_id=session_id)['memory']
        judger_response_items.append(RLJudgerResponseItem(reward=dict(score=env_history[-1]['reward'])))
        # reset agent memory
        env.agent.reset(session_id=session_id, recursive=True)
    group_data_items = update_dataflow_item(group_data_items, "env.agent", agent_data_items)
    group_data_items = update_dataflow_item(group_data_items, "env.rollout", rollout_response_items)
    group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_response_items)
    return group_data_items


pg = AutoAcceleratorWorkers.build_placement_group(resources)
rollout_controller = ray.remote(max_concurrency=1000)(RolloutController).remote(rollout_config, pg)
load_checkpoint_cfg = LoadCheckpointConfig(load_optimizer_states=False, load_optimizer_args=False)

actions = [
    dict(
        type=AsyncMCPClient,
        name='SerperSearch',
        server_type='http',
        rate_limit=100.0,
        max_concurrency=30,
        url=[],
    ),
    dict(
        type=AsyncMCPClient,
        name='JinaBrowse',
        server_type='http',
        rate_limit=100.0,
        max_concurrency=40,
        url=[],
    ),
]
tool_prompt = get_tool_prompt(actions)

train_agent = dict(
    type=FunctionCallAgent,
    select_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
        ),
        template=tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=actions,
        judger=dict(
            type=JudgerWrapper,
            judger_cfg=judger_cfg,
            placement_group=ray.get(
                placement_group(bundles=[{"CPU": 1, "memory": 1024**3}], strategy="PACK").ready(), timeout=30
            ),
        ),
        max_turn=25,
        enable_no_thinking_penalty=False,
        max_tool_response_length=4096,
    ),
    finish_condition=finish_condition_func,
)

eval_agent = dict(
    type=FunctionCallAgent,
    select_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
        ),
        template=tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=actions,
        judger=dict(
            type=JudgerWrapper,
            judger_cfg=judger_cfg,
            placement_group=ray.get(
                placement_group(bundles=[{"CPU": 1, "memory": 1024**3}], strategy="PACK").ready(), timeout=30
            ),
        ),
        max_turn=25,
        enable_no_thinking_penalty=False,
        max_tool_response_length=4096,
    ),
    finish_condition=finish_condition_func,
)


def rollout_env_router_fn(item: RLDataFlowItem):
    if item.data.extra_info['origin_data_source'].startswith('gaia') or item.data.extra_info['origin_data_source'] in [
        'BrowseComp-ZH',
        'HLE',
    ]:
        return 'eval'
    return 'train_agent'


environment_config = dict(
    type=ComposedEnvironment,
    environment=experimental_name,
    rollout_controller=rollout_controller,
    environments={
        'train_agent': dict(
            type=AgentEnvironment,
            environment='websailor',
            agent_cfg=train_agent,
            rollout_controller=rollout_controller,
            preprocess_func=partial(prepare_agent_inputs, is_training=True),
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'eval': dict(
            type=AgentEnvironment,
            environment='eval',
            agent_cfg=eval_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
    },
    router=rollout_env_router_fn,
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

evaluator_cfg = (
    EvaluatorConfig(
        enable_evaluate=enable_evaluate,
        enable_initial_evaluate=enable_initial_evaluate,
        dataset_cfg=eval_dataset_cfg,
        tokenizer=model_path,
        eval_sample_ratio=1.0,
        evaluate_step=evaluate_step,
        compute_metric_func=partial(
            compute_metric,
            source_normalizer={
                'miroRL': 'websearch',
                'musique': 'websearch',
                'websailor': 'websearch',
                'webdancer': 'websearch',
                'gaia-level1': ('gaia-level1', 'gaia'),
                'gaia-level2': ('gaia-level2', 'gaia'),
                'gaia-level3': ('gaia-level3', 'gaia'),
            },
        ),
        sample_params=evaluation_sample_params,
        max_concurrent=8192,
    )
    if enable_evaluate
    else None
)

replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=model_path,
    # postprocessor_func=group_sample_filter_func,
)

# # 5. Train worker
model_cfg = Qwen3VLMoE30BA3Config()
model_cfg.compile_cfg = False
model_cfg.freeze_vision = True
model_cfg.freeze_projector = True
model_cfg.vision_config.depth = 24
model_cfg.vision_config.hidden_size = 1024
model_cfg.vision_config.intermediate_size = 4096
model_cfg.vision_config.deepstack_visual_indexes = []

model_cfg.projector_config.vision_hidden_size = 1024
model_cfg.projector_config.deepstack_visual_indexes = []

model_cfg.text_config.max_position_embeddings = 32768
model_cfg.text_config.rope_theta = 1000000
model_cfg.text_config.rope_scaling_cfg = RopeScalingConfig(
    fope_init_factor=0.1,
    fope_sep_head=True,
    num_inv_freq=None,
)
model_cfg.text_config.vocab_size = 155008
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
        loss_type="intern_s1_delivery.modules.pg_loss.pg_loss_fn",
        clip_ratio_c=5.0,
        log_prob_diff_min=-20.0,
        log_prob_diff_max=20.0,
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


import torch.distributed as dist

from xtuner.v1.train.agent_rl_trainer import AgentRLTrainer

trainer = AgentRLTrainer.from_config(trainer)
trainer.fit()

if dist.is_initialized():
    dist.destroy_process_group()
