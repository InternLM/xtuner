import os

os.environ["XTUNER_USE_LMDEPLOY"] = "1"

import hashlib
import json

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
from copy import deepcopy
from functools import partial

import ray
from intern_s1_delivery.advantage.rloo_entropy_badword import (
    OverlongRLOOGroupEntropyBadwordAdvantageConfig,
)
from lagent.actions.mcp_client import AsyncMCPClient
from lagent.actions.web_visitor import WebVisitor
from lagent.agents.fc_agent import FunctionCallAgent, get_tool_prompt
from ray.util.placement_group import placement_group

from claw_bench.claw_tokenize_fn import RLClawTokenizeFnConfig
from tb2_eval.tb2_eval_tokenize_fn import RLTB2EvalTokenizeFnConfig
from tb2_rl.tb2_rl_tokenize_fn import RLTB2RLTokenizeFnConfig
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
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.ray.base import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourcesConfig,
)
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment.agent_env import AgentEnvironment
from xtuner.v1.ray.environment.composed_env import ComposedEnvironment
from xtuner.v1.ray.environment.install_agent_env import InstallAgentEnvironment
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
from xtuner.v1.ray.judger.frontierscience_judger import FrontierScienceJudgerConfig
from xtuner.v1.ray.judger.review import ReviewJudgerConfig
from xtuner.v1.ray.judger.sgi_judger import SGIJudgerConfig
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
base_work_dir = "/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/interns2_preview_0430rc9"
work_dir = os.path.join(base_work_dir, experimental_name)
model_name = os.environ["RL_LLM_MODEL"]
model_path = '/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/interns2-35ba3-base05-20260424a-rl-data260428rc0-56k-badword-mtp4-resume800/20260430074140/hf-40'
stop_word = "<|im_end|>"

# basic settings
global_batch_size = 256
prompt_repeat_k = 16
max_concurrent_groups = 512

max_prompt_length = 16 * 1024
pack_max_length =  130 * 1024
max_response_length = 128 * 1024

train_ep_size = 1
train_sp_size = 2
rollout_tp_size = 4
rollout_ep_size = 1
fp32_lm_head = True
enable_float8_rollout = False
rollout_max_batch_size = 128
max_prefill_token_num = 1024
enable_return_routed_experts = True
enable_partial_rollout = False
staleness_threshold = 0.0
tail_batch_candidate_steps = 0
auto_resume = True
skip_load_weights = True
lr = 1e-6
train_optimizer_steps = 8  # mini batch steps
hf_interval = 5
total_epochs = 10
# evaluation settings
enable_evaluate = True
enable_initial_evaluate = True
evaluate_step = 5

# agent setting
max_turn = 50  # 最大对话轮次
lower_tool_turn_bound = 12  # 是否惩罚使用工具的轮次小于该值的样本 , None表示不启用该惩罚
lower_tool_turn_bound_science = 8
enable_repeated_tool_call_penalty = True  # 是否惩罚重复调用工具的样本
enable_no_thinking_penalty = False
max_tool_response_length = 8192


# 1. resources
resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=64,
    num_cpus_per_worker=12,
    cpu_memory_per_worker=16 * 1024**3,  # 16 GB
)
judger_cpu_resources = CPUResourcesConfig.from_total(total_cpus=64, num_workers=64, total_memory=512 * 1024**3)

# 2. rollout
rollout_config = RolloutConfig(
    env=experimental_name,
    device=resources.accelerator,
    model_name=model_name,
    model_path=model_path,
    dtype="bfloat16",
    tensor_parallel_size=rollout_tp_size,
    expert_parallel_size=rollout_ep_size,
    gpu_memory_utilization=0.7,
    enable_float8=enable_float8_rollout,
    skip_load_weights=skip_load_weights,
    context_length=max_response_length,
    rollout_max_batch_size_per_instance=rollout_max_batch_size,
    # chunked_prefill_size=4096,
    allow_over_concurrency_ratio=1.2,
    rollout_timeout=1800,
    enable_return_routed_experts=enable_return_routed_experts,
    # max_prefill_token_num=max_prefill_token_num,
    extra_rollout_config=dict(
        lmdeploy_log_level="ERROR",
        lmdeploy_uvicorn_log_level="ERROR",
        lmdeploy_speculative_algorithm='qwen3_5_mtp',
        lmdeploy_speculative_num_draft_tokens=4,
    ),
    fp32_lm_head=fp32_lm_head,
)

# sampling params
training_sample_params = SampleParams(
    max_tokens=max_response_length, top_k=0, top_p=0.999, temperature=1.0, min_tokens=0
)
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.temperature = 0.8
# evaluation_sample_params.max_tokens = max_response_length

data_judger_mapping = {
    "agent": {"compass_verifier_v2": 1.0},
    "agent_science": {"compass_verifier_v2": 1.0},
    'GAIA_sft_1229': {"compass_verifier_v2": 1.0},
    'gaia-level1': {"compass_verifier_v2": 1.0},
    'gaia-level2': {"compass_verifier_v2": 1.0},
    'gaia-level3': {"compass_verifier_v2": 1.0},
    'BrowseComp-ZH': {"compass_verifier_v2": 1.0},
    'HLE': {"compass_verifier_v2": 1.0},
    'browsecomp': {"compass_verifier_v2": 1.0},
    'math': {"compass_verifier_v2": 1.0},
    'AIME2024': {"compass_verifier_v2": 1.0},
    'AIME2025': {"compass_verifier_v2": 1.0},
    'aime2026': {"compass_verifier_v2": 1.0},
    'hmmt26': {"compass_verifier_v2": 1.0},
    'IMO-Bench-AnswerBench': {"compass_verifier_v2": 1.0},
    'UGD_hard': {"compass_verifier_v2": 1.0},
    'openreview': {"openreview": 1.0},
    'openreview_test': {"openreview": 1.0},
    'sgi-deep-research': {"sgi_judger": 1.0},
    'frontierscience': {"frontierscience_judger": 1.0},
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
    chat_template='qwen3-vl-rl',
)

from intern_s1_delivery.dataset.xpuyu_dataset_vl import (
    RLTokenizeFnConfig,
)
from intern_s1_delivery.dataset.xpuyu_dataset_vl import (
    parse_xpuyu_json_cfg as parse_xpuyu_json_cfg_vl,
)

from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig

# 2. dataset
# from lagent_rl.datasets.parse_cfg import parse_xpuyu_json_cfg


def parse_xpuyu_json_cfg(path, max_prompt_length):
    with open(path, "r") as f:
        json_cfg = json.load(f)
    converted_cfg = []
    for ds_name, ds_cfg in json_cfg.items():
        annotation = ds_cfg["annotation"]
        tokenize_fn = ds_cfg["tokenize_fn"]

        if tokenize_fn == "RLClawTokenizeFnConfig":
            tokenize_fn_cfg = RLClawTokenizeFnConfig
        elif tokenize_fn == "RLTB2RLTokenizeFnConfig":
            tokenize_fn_cfg = RLTB2RLTokenizeFnConfig
        elif tokenize_fn == "RLTB2EvalTokenizeFnConfig":
            tokenize_fn_cfg = RLTB2EvalTokenizeFnConfig

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
                    "tokenize_fn": tokenize_fn_cfg(
                        root_path=ds_cfg.get("root_path", None), max_length=max_prompt_length
                    ),
                }
            )
    return converted_cfg


TRAIN_DATA_PATH_SCIENCE_SEARCH = "/mnt/shared-storage-user/llmit/user/liujiangning/projects/crg_rl_projects/src/lagent_rl/scripts/s2_preview_35b_agentrl/interns2_35ba3_b03_0413a_reasoningRL_scienceSearch0421a/scienceSearch0421a.json"
TRAIN_DATA_PATH_INTERNLM_SCIENCE = "/mnt/shared-storage-user/llmit/user/liujiangning/projects/crg_rl_projects/src/lagent_rl/scripts/s2_preview_35b_agentrl/interns2_35ba3_b03_0413a_reasoningRL_scienceSearch0423a/scienceSearch0423a.json"

TRAIN_DATA_PATH_SEARCH = (
    '/mnt/shared-storage-user/llmit1/user/liujiangning/data/s1_1_rl_delivery_agent/exp_rl/rl_data_260126.json'
)
TRAIN_DATA_PATH_MATH = '/mnt/shared-storage-user/llmit1/user/liujiangning/data/s1_1_rl_delivery_agent/exp_rl/train_interns1-1_260124rc0_pure_math.json'
TEST_DATA_PATH_MATH = '/mnt/shared-storage-user/llmit1/user/liujiangning/data/s1_1_rl_delivery_agent/math_benchmark/val_python_toolcall.json'


train_dataset_cfg_science_search = parse_xpuyu_json_cfg_vl(
    TRAIN_DATA_PATH_SCIENCE_SEARCH, tokenize_fn_cfg, max_prompt_length, data_judger_mapping
)
train_dataset_cfg_internlm_science = parse_xpuyu_json_cfg_vl(
    TRAIN_DATA_PATH_INTERNLM_SCIENCE, tokenize_fn_cfg, max_prompt_length, data_judger_mapping
)

train_dataset_cfg_search = parse_xpuyu_json_cfg_vl(
    TRAIN_DATA_PATH_SEARCH, tokenize_fn_cfg, max_prompt_length, data_judger_mapping
)
train_dataset_cfg_math = parse_xpuyu_json_cfg_vl(
    TRAIN_DATA_PATH_MATH, tokenize_fn_cfg, max_prompt_length, data_judger_mapping
)
train_dataset_cfg_review = [
    {
        "dataset": DatasetConfig(
            name='openreview',
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/demo/xtuner/examples/demo_data/agent/openreview/train.jsonl",
            sample_ratio=0.1,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=False,
        ),
    }
]
train_dataset_cfg_tb2rl = parse_xpuyu_json_cfg(
    '/mnt/shared-storage-user/llmit/user/wangziyi/projs/xtuner_agent_dev/examples/demo_data/agent_dev/tb2_rl/meta.json',
    max_prompt_length,
)
train_dataset_cfg = (
    train_dataset_cfg_science_search
    + train_dataset_cfg_internlm_science
    + train_dataset_cfg_search
    + train_dataset_cfg_math
    + train_dataset_cfg_review
    + train_dataset_cfg_tb2rl
)
eval_dataset_cfg_search = [
    {
        "dataset": DatasetConfig(
            name="gaia",
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/crg_rl_projects/data/gaia_text_103.jsonl",
            sample_ratio=4.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="browsecomp-zh",
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/crg_rl_projects/data/browsecomp-zh.jsonl",
            sample_ratio=0.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="browsecomp",
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/crg_rl_projects/data/browsecomp.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="hle",
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/crg_rl_projects/data/hle.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="sgi-deep-research",
            anno_path="/mnt/shared-storage-user/llmit1/user/liujiangning/data/eval_benchmark_testset/sgi_deep_research_gaia_format.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
    {
        "dataset": DatasetConfig(
            name="frontierscience",
            anno_path="/mnt/shared-storage-user/llmit1/user/liujiangning/data/eval_benchmark_testset/frontierscience_gaia_format.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
]
eval_dataset_cfg_math = parse_xpuyu_json_cfg_vl(
    TEST_DATA_PATH_MATH, tokenize_fn_cfg, max_prompt_length, data_judger_mapping, ignore_multimodal_info=True
)
eval_dataset_cfg_review = [
    {
        "dataset": DatasetConfig(
            name="openreview",
            anno_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/demo/xtuner/examples/demo_data/agent/openreview/test.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='VLMJsonlDataset',
        ),
        "tokenize_fn": RLTokenizeFnConfig(
            tokenize_fn_cfg=tokenize_fn_cfg,
            system_prompt=None,
            max_length=max_prompt_length,
            data_judger_mapping=data_judger_mapping,
            ignore_multimodal_info=True,
        ),
    },
]
eval_data_cfg_tb2eval = [
    {
        "dataset": DatasetConfig(
            name="tb2-eval",
            anno_path="/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/tb2_eval_tasks.jsonl",
            sample_ratio=1.0,
            media_root=None,
            class_name='JsonlDataset',
        ),
        "tokenize_fn": RLTB2EvalTokenizeFnConfig(
            root_path="/mnt/shared-storage-user/llmit/user/wangziyi/projs/terminalbench2-harbor-p-cluster/terminal-bench-2",
            max_length=max_prompt_length,
        ),
    },
]
# eval_dataset_cfg = (
#     (eval_dataset_cfg_search + eval_dataset_cfg_math + eval_dataset_cfg_review + eval_data_cfg_tb2eval)
#     if enable_evaluate
#     else []
# )
eval_dataset_cfg = eval_data_cfg_tb2eval
dataloader_config = DataloaderConfig(pack_max_length=pack_max_length, collator="fake_collator", pack_level="none")


# 3. judger
compass_judger_cfg = JudgerConfig(
    enable_weighted_judgers=True,
    reward_judger_configs=[
        CompassVerifierV2Config(
            hosts=[
                "10.102.251.61:23333",
                "10.102.251.61:23334",
                "10.102.251.61:23335",
                "10.102.251.61:23336",
                "10.102.251.61:23337",
                "10.102.251.61:23338",
                "10.102.251.61:23339",
                "10.102.251.61:23340",
                "10.102.216.52:23333",
                "10.102.216.52:23334",
                "10.102.216.52:23335",
                "10.102.216.52:23336",
                "10.102.216.52:23337",
                "10.102.216.52:23338",
                "10.102.216.52:23339",
                "10.102.216.52:23340",
                "10.102.238.19:23333",
                "10.102.238.19:23334",
                "10.102.238.19:23335",
                "10.102.238.19:23336",
                "10.102.238.19:23337",
                "10.102.238.19:23338",
                "10.102.238.19:23339",
                "10.102.238.19:23340",
                "10.102.239.68:23333",
                "10.102.239.68:23334",
                "10.102.239.68:23335",
                "10.102.239.68:23336",
                "10.102.239.68:23337",
                "10.102.239.68:23338",
                "10.102.239.68:23339",
                "10.102.239.68:23340",
            ]
        ),
        SGIJudgerConfig(
            hosts=[
                "10.102.213.32:30030",
                "10.102.213.32:30031",
                "10.102.213.32:30032",
                "10.102.213.32:30033",
                "10.102.213.32:30034",
                "10.102.213.32:30035",
                "10.102.213.32:30036",
                "10.102.213.32:30037",
            ],
            model_name="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307",
            num_ray_actors=1,
            request_timeout=60,
        ),
        FrontierScienceJudgerConfig(
            hosts=[
                "10.102.213.32:30030",
                "10.102.213.32:30031",
                "10.102.213.32:30032",
                "10.102.213.32:30033",
                "10.102.213.32:30034",
                "10.102.213.32:30035",
                "10.102.213.32:30036",
                "10.102.213.32:30037",
            ],
            model_name="/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307",
            num_ray_actors=1,
            request_timeout=60,
        ),
    ],
)
review_judger_cfg = JudgerConfig(reward_judger_configs=[ReviewJudgerConfig(judger_name="openreview")])

from xtuner.v1.ray.judger.controller import JudgerController

compass_judger_controller = JudgerController.remote(
    compass_judger_cfg,
    ray.get(
        placement_group(
            bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            strategy="PACK",
        ).ready(),
        timeout=30,
    )
)
review_judger_controller = JudgerController.remote(
    review_judger_cfg,
    ray.get(
        placement_group(
            bundles=[{"CPU": 1, "memory": 1024**3}] * len(review_judger_cfg.reward_judger_configs),
            strategy="PACK",
        ).ready(),
        timeout=30,
    )
)


from xtuner.v1.ray.environment.lagent.parsers import Qwen3_5FunctionCallParser


def prepare_agent_inputs(env, group_data_item: RLDataFlowItem):
    env_agent = group_data_item.env.agent.extra_info.pop('agent').env_agent
    user_prompt = group_data_item.data.messages[-1]['content']
    env_message = AgentMessage(
        sender="env", content=user_prompt, uid=hashlib.md5(user_prompt.encode('utf-8')).hexdigest()
    )
    if not env_agent.memory.get_memory():
        set_env_message = AgentMessage(sender="env", content=group_data_item)
        env_agent.memory and env_agent.memory.add(set_env_message)  # type: ignore[union-attr]
    return (env_message,)


def convert_rollout_tractory_to_train(env, group_data_items):
    agent_data_items, rollout_response_items, judger_response_items = [], [], []
    for i in range(len(group_data_items)):
        history = group_data_items[i].env.rollout.extra_info['agent_state_dict']['policy_agent.memory']
        env_history = group_data_items[i].env.rollout.extra_info['agent_state_dict']['env_agent.memory']
        messages = group_data_items[i].env.rollout.extra_info['agent_message_dict']['policy_agent.messages']
        agent_data_items.append(RLAgentDataItem(extra_info=dict(messages=messages, state={"history": history})))
        rollout_response_items.append(
            RLRolloutResponseItem(
                response=history[-1]['raw_content'],
                response_ids=history[-1]['raw_content_ids'],
                logprobs=history[-1]['raw_content_logprobs'],
                state=RolloutState.COMPLETED,
            )
        )
        reward_payload = env_history[-1]['reward']
        if isinstance(reward_payload, dict):
            if 'score' in reward_payload and group_data_items[i].data.extra_info.get('origin_data_source') in [
                'openreview',
            ]:  # scale down the reward for review data
                reward_payload['score'] = reward_payload['score'] / 10
            judger_response_items.append(RLJudgerResponseItem(reward=reward_payload))
        else:
            judger_response_items.append(RLJudgerResponseItem(reward=dict(score=reward_payload)))
    group_data_items = update_dataflow_item(group_data_items, "env.agent", agent_data_items)
    group_data_items = update_dataflow_item(group_data_items, "env.rollout", rollout_response_items)
    group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_response_items)
    return group_data_items


def prepare_agent_inputs_for_tb2rl(env, group_data_item: RLDataFlowItem):
    return group_data_item


def convert_rollout_tractory_to_train_for_tb2rl(env, group_data_items):
    agent_data_items, rollout_response_items, judger_response_items = [], [], []
    for i in range(len(group_data_items)):
        messages = group_data_items[i].env.agent.extra_info['message_dict']['policy_agent.messages']
        tools = group_data_items[i].env.agent.extra_info['message_dict'].get('policy_agent.tools')
        agent_data_items.append(RLAgentDataItem(extra_info=dict(messages=messages, tools=tools)))
        # breakpoint()
        rollout_response_items.append(
            RLRolloutResponseItem(
                response=messages[-1]['raw_content'],
                response_ids=messages[-1]['raw_content_ids'],
                logprobs=messages[-1]['raw_content_logprobs'],
                state=RolloutState.COMPLETED,
            )
        )
        reward_payload = group_data_items[i].env.judger.extra_info['total']
        judger_response_items.append(RLJudgerResponseItem(reward=dict(score=reward_payload)))
    group_data_items = update_dataflow_item(group_data_items, "env.agent", agent_data_items)
    group_data_items = update_dataflow_item(group_data_items, "env.rollout", rollout_response_items)
    group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_response_items)
    # breakpoint()
    return group_data_items


pg = AutoAcceleratorWorkers.build_placement_group(resources)
rollout_controller = ray.remote(max_concurrency=1000)(RolloutController).remote(rollout_config, pg)
load_checkpoint_cfg = LoadCheckpointConfig(load_optimizer_states=False, load_optimizer_args=False)

search_tool = AsyncMCPClient(
    # type=AsyncMCPClient,
    name='SerperSearch',
    server_type='http',
    rate_limit=500.0,
    # max_concurrency=128,
    url=[
        'http://10.102.103.157:8091/mcp',
        'http://10.102.103.155:8096/mcp',
        'http://10.102.103.155:8092/mcp',
        'http://10.102.103.155:8095/mcp',
        'http://10.102.103.155:8097/mcp',
        'http://10.102.103.155:8098/mcp',
        'http://10.102.103.155:8094/mcp',
        'http://10.102.103.155:8093/mcp',
    ],
)
browse_tool = AsyncMCPClient(
    # type=AsyncMCPClient,
    name='JinaBrowse',
    server_type='http',
    rate_limit=100.0,
    max_concurrency=40,
    url=[
        'http://10.102.103.155:8104/mcp',
        'http://10.102.103.155:8100/mcp',
        'http://10.102.103.148:8101/mcp',
        'http://10.102.103.157:8105/mcp',
        'http://10.102.103.155:8099/mcp',
        'http://10.102.103.155:8102/mcp',
        'http://10.102.103.155:8103/mcp',
        'http://10.102.103.155:8106/mcp',
    ],
)
visit_tool = WebVisitor(
    # type=WebVisitor,
    browse_tool=dict(
        type=AsyncMCPClient,
        name='JinaBrowse',
        server_type='http',
        rate_limit=100.0,
        max_concurrency=40,
        url=[
            'http://10.102.103.155:8104/mcp',
            'http://10.102.103.155:8100/mcp',
            'http://10.102.103.148:8101/mcp',
            'http://10.102.103.157:8105/mcp',
            'http://10.102.103.155:8099/mcp',
            'http://10.102.103.155:8102/mcp',
            'http://10.102.103.155:8103/mcp',
            'http://10.102.103.155:8106/mcp',
        ],
    ),
    llm=dict(
        type=ControllerWrapper,
        rollout_controller=rollout_controller,
        sample_params=SampleParams(max_tokens=max_response_length),
        tool_call_parser=Qwen3_5FunctionCallParser(),
    ),
    truncate_browse_response_length=60000,
    tokenizer_path=model_path,
)
arxiv_tool = AsyncMCPClient(
    # type=AsyncMCPClient,
    name='arxiv_search',
    server_type='http',
    rate_limit=50.0,
    max_concurrency=20,
    url=[
        'http://10.102.252.176:2364/mcp',
    ],
)


tool_template = """# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

review_sys_prompt = r"""You are an expert reviewer in the field of pre-training, 3D human pose and shape estimation, and self-supervised representation learning for the ICLR 2023 conference. Your responsibilitie is conducting an initial review. You must follow a strict reasoning process.

### INTERACTION PROTOCOL
You must perform a **Thought-Action** loop for every step. Do not rush to the final review.
1. **THINK (`<think>`)**   - **Before Tool Use**: Analyze the paper step-by-step. Identify gaps in your knowledge. Formulate search queries. Explicitly check if the `end_date` constraint is met.
   - **After Tool Response**: Analyze the search results. synthesis the information. Decide if more searches are needed or if you have sufficient context to write the review.
   - The reasoning content must be enclosed with `<think>` and `</think>` tags.

2. **ACTION (`<tool_call>`)**:
   - If you need external information, output a tool call enclosed in `<tool_call>` and `</tool_call>` tags.
   - The content inside the tags must be valid JSON format.
   - **CRITICAL**: When calling `arxiv_search`, you MUST set the `end_date` argument to '20230603'.

3. **FINALIZE (`<reviewer>`)**:
   - Only when you have completed all necessary research and reasoning, output the final review inside `<reviewer>` and `</reviewer>` tags.
   - The review must include: Summary, Strengths, Weaknesses, Questions, and References.

### CITATION & REFERENCE STANDARDS (CRITICAL)
You must adhere to the following strict formatting rules for the final review:
1. **Sequential Numbering**: Citations in the text must be numbered sequentially starting from [1] based on the order they first appear (e.g., [1], [2], [3]). **Do NOT use the ID returned by the search tool (e.g., do not use [81], [28]).**
2. **Inline Citation**: Every external claim must have an inline citation. Example: 'Recent studies [1] have shown that...'
3. **Reference List**: The 'References' section at the end must strictly match the inline citations. Each entry must contain:
   - Format: `[ID] Authors. **Title**. Venue, Year. URL`
   - Example: `[1] J. Smith et al. **Deep Learning**. NeurIPS, 2023. https://arxiv.org/...`
   - Ensure the Title and URL are complete. The URL field in references must be the EXACT URL returned by the search tool. Do not use placeholders like '...'.

### OUTPUT FORMAT (`<reviewer>`)
<reviewer>
## Summary
...
## Strengths
...
## Weaknesses
...
## Questions
...
## References
[1] ...
[2] ...
</reviewer>

### CONSTRAINTS
- Never output `<tool_call>` and `<reviewer>` in the same turn.
- Strictly adhere to the submission deadline: do not use knowledge published after 20230603.- Do NOT hallucinate citations. You can ONLY cite papers that explicitly appear in the search results from the <tool_call> output. If you cannot find a relevant paper, do not cite a fake one.
"""


from lagent_rl.environment.lagent_ext.actions.python_executor import PythonExecutor

python_action = PythonExecutor(
    # type=PythonExecutor,
    rate_limit_qps=500.0,
    burst=20,
    retries=5,
    connect_timeout=5.0,
    read_timeout=30.0
)

# tool prompts with python (for science search)
search_browse_python_tool_prompt = get_tool_prompt([search_tool, browse_tool, python_action], template=tool_template)
search_visit_python_tool_prompt = get_tool_prompt([search_tool, visit_tool, python_action], template=tool_template)
# tool prompts without python (for pure search)
search_browse_tool_prompt = get_tool_prompt([search_tool, browse_tool], template=tool_template)
search_visit_tool_prompt = get_tool_prompt([search_tool, visit_tool], template=tool_template)
# other tool prompts
review_tool_prompt = get_tool_prompt([arxiv_tool], template=tool_template)
python_tool_prompt = get_tool_prompt([python_action], template=tool_template)


# ============================================================
# Science search agents (with python_action) - for agent_science data
# ============================================================
train_science_search_browse_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=search_browse_python_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[search_tool, browse_tool, python_action],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=max_turn,
        lower_tool_turn_bound=lower_tool_turn_bound_science,
        enable_repeated_tool_call_penalty=enable_repeated_tool_call_penalty,
        enable_no_thinking_penalty=enable_no_thinking_penalty,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)
train_science_search_visit_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=search_visit_python_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[search_tool, visit_tool, python_action],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=max_turn,
        lower_tool_turn_bound=lower_tool_turn_bound_science,
        enable_repeated_tool_call_penalty=enable_repeated_tool_call_penalty,
        enable_no_thinking_penalty=enable_no_thinking_penalty,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)


train_agent_with_search_browse = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=search_browse_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[search_tool, browse_tool],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=max_turn,
        lower_tool_turn_bound=lower_tool_turn_bound,
        enable_repeated_tool_call_penalty=enable_repeated_tool_call_penalty,
        enable_no_thinking_penalty=enable_no_thinking_penalty,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)
train_agent_with_search_visit = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=search_visit_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[search_tool, visit_tool],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=max_turn,
        lower_tool_turn_bound=lower_tool_turn_bound,
        enable_repeated_tool_call_penalty=enable_repeated_tool_call_penalty,
        enable_no_thinking_penalty=enable_no_thinking_penalty,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)
eval_search_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=search_visit_python_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[search_tool, visit_tool, python_action],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=max_turn,
        lower_tool_turn_bound=None,
        enable_repeated_tool_call_penalty=False,
        enable_no_thinking_penalty=False,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)

train_math_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=python_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[python_action],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=25,
        lower_tool_turn_bound=5,
        enable_repeated_tool_call_penalty=False,
        enable_no_thinking_penalty=False,
        max_tool_response_length=4096,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)
eval_math_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(),
        ),
        template=python_tool_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[python_action],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=compass_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(
            #         bundles=[{"CPU": 1, "memory": 1024**3}] * len(compass_judger_cfg.reward_judger_configs),
            #         strategy="PACK",
            #     ).ready(),
            #     timeout=30,
            # ),
            judger_controller=compass_judger_controller,
        ),
        max_turn=25,
        lower_tool_turn_bound=None,
        enable_repeated_tool_call_penalty=False,
        enable_no_thinking_penalty=False,
        max_tool_response_length=4096,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)

train_review_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(argument_type={'end_date': str}),
        ),
        template=review_tool_prompt + "\n\n" + review_sys_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[arxiv_tool],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=review_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(bundles=[{"CPU": 1, "memory": 1024**3}], strategy="PACK").ready(), timeout=30
            # ),
            judger_controller=review_judger_controller,
            reward_key=None,
        ),
        max_turn=25,
        lower_tool_turn_bound=None,
        enable_repeated_tool_call_penalty=False,
        enable_no_thinking_penalty=False,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)
eval_review_agent = dict(
    type=FunctionCallAgent,
    policy_agent=dict(
        type=AsyncTokenInOutAgent,
        llm=dict(
            type=ControllerWrapper,
            rollout_controller=rollout_controller,
            sample_params=SampleParams(max_tokens=max_response_length),
            tool_call_parser=Qwen3_5FunctionCallParser(argument_type={'end_date': str}),
        ),
        template=review_tool_prompt + "\n\n" + review_sys_prompt,
    ),
    env_agent=dict(
        type=EnvAgent,
        actions=[arxiv_tool],
        judger=JudgerWrapper(
            # type=JudgerWrapper,
            # judger_cfg=review_judger_cfg,
            # placement_group=ray.get(
            #     placement_group(bundles=[{"CPU": 1, "memory": 1024**3}], strategy="PACK").ready(), timeout=30
            # ),
            judger_controller=review_judger_controller,
            reward_key=None,
        ),
        max_turn=25,
        lower_tool_turn_bound=None,
        enable_repeated_tool_call_penalty=False,
        enable_no_thinking_penalty=False,
        max_tool_response_length=max_tool_response_length,
    ),
    finish_condition=finish_condition_func,
    initialize_input=False,
)


def bucket(key: str, n: int) -> int:
    hashkey = hashlib.md5(key.encode('utf-8')).hexdigest()
    return sum(ord(c) for c in hashkey) % n


def rollout_env_router_fn(item: RLDataFlowItem):
    source = item.data.extra_info.get('origin_data_source', '')

    # 1. Routing for Evaluation Environments
    if source in ['openreview_test']:
        return 'eval_review_agent'
    if source.startswith('gaia') or source in [
        'BrowseComp-ZH',
        'HLE',
        'browsecomp',
        'GAIA_sft_1229',
        'sgi-deep-research',
        'frontierscience',
    ]:
        return 'eval_search_agent'
    if source in ['AIME2024', 'AIME2025', 'aime2026', 'hmmt26', 'IMO-Bench-AnswerBench', 'UGD_hard']:
        return 'eval_math_agent'
    if source == 'tb2-eval':
        return 'eval_tb2eval'

    # 2. Routing for Train Environments
    if source == 'openreview':
        return 'train_review_agent'
    elif source == 'math':
        return 'train_math_agent'
    elif source == 'claw-bench':
        return 'train_clawbench'
    elif source == 'tb2-rl':
        return 'train_tb2rl'
    elif source == 'agent_science':
        # Science search data (with python_action)
        match bucket(item.data.messages[-1]['content'], 2):
            case 0:
                return 'train_science_search_browse_agent'
            case 1:
                return 'train_science_search_visit_agent'
    else:
        # Default fallback to Search Train Environments
        match bucket(item.data.messages[-1]['content'], 2):
            case 0:
                return 'train_agent_with_search_browse'
            case 1:
                return 'train_agent_with_search_visit'


environment_config = dict(
    type=ComposedEnvironment,
    environment=experimental_name,
    rollout_controller=rollout_controller,
    environments={
        'train_science_search_browse_agent': dict(
            type=AgentEnvironment,
            environment='train_science_search_browse_agent',
            agent_cfg=train_science_search_browse_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_science_search_visit_agent': dict(
            type=AgentEnvironment,
            environment='train_science_search_visit_agent',
            agent_cfg=train_science_search_visit_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_agent_with_search_browse': dict(
            type=AgentEnvironment,
            environment='train_agent_with_search_browse',
            agent_cfg=train_agent_with_search_browse,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_agent_with_search_visit': dict(
            type=AgentEnvironment,
            environment='train_agent_with_search_visit',
            agent_cfg=train_agent_with_search_visit,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'eval_search_agent': dict(
            type=AgentEnvironment,
            environment='eval_search_agent',
            agent_cfg=eval_search_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_math_agent': dict(
            type=AgentEnvironment,
            environment='train_math_agent',
            agent_cfg=train_math_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'eval_math_agent': dict(
            type=AgentEnvironment,
            environment='eval_math_agent',
            agent_cfg=eval_math_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_review_agent': dict(
            type=AgentEnvironment,
            environment='train_review_agent',
            agent_cfg=train_review_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'eval_review_agent': dict(
            type=AgentEnvironment,
            environment='eval_review_agent',
            agent_cfg=eval_review_agent,
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs,
            postprocess_func=convert_rollout_tractory_to_train,
        ),
        'train_tb2rl': dict(
            type=InstallAgentEnvironment,
            environment='train_tb2rl',
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs_for_tb2rl,
            postprocess_func=convert_rollout_tractory_to_train_for_tb2rl,
        ),
        'eval_tb2eval': dict(
            type=InstallAgentEnvironment,
            environment='eval_tb2eval',
            rollout_controller=rollout_controller,
            preprocess_func=prepare_agent_inputs_for_tb2rl,
            postprocess_func=convert_rollout_tractory_to_train_for_tb2rl,
        ),
    },
    router=rollout_env_router_fn,
)

# 4. dataflow and evaluator
dataflow_config = DataFlowConfig(
    env=experimental_name,
    max_concurrent=max_concurrent_groups,
    enable_partial_rollout=enable_partial_rollout,
    tail_batch_candidate_steps=tail_batch_candidate_steps,
    staleness_threshold=staleness_threshold,
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


def group_sample_filter_func(group_samples):
    # filter all correct or all wrong sample
    group_samples = [s for s in group_samples if s.env.rollout.response_ids is not None]

    # filter all same reward sample
    rewards = [d.env.judger.reward["score"] for d in group_samples]
    if len(set(rewards)) == 1:
        print(f"filter all same reward sample: {rewards}")
        return []
    return group_samples


replay_buffer_cfg = ReplayBufferConfig(
    dataset_cfg=train_dataset_cfg,
    dataloader_cfg=dataloader_config,
    tokenizer=model_path,
    # postprocessor_func=group_sample_filter_func,
)

# # 5. Train worker
model_cfg = Qwen3_5_VLMoE35BA3Config(
    freeze_vision=True,
    freeze_projector=True,
)
model_cfg.float8_cfg = None
model_cfg.text_config.ep_size = 1
model_cfg.text_config.z_loss_cfg = None
model_cfg.text_config.balancing_loss_cfg = None
model_cfg.text_config.freeze_routers = True
model_cfg.text_config.mtp_config = MTPConfig(
    num_layers=4,
    loss_scaling_factor=1.0,
    detach_mtp_lm_head_weight=True,
    detach_mtp_inputs=True,
    share_weights=True,
)
model_cfg.text_config.vocab_size = 251392
# model_cfg.text_config.embed_grad_max_token_id = 251173

optim_cfg = AdamWConfig(
    lr=lr,
    betas=(0.9, 0.95),
    max_grad_norm=1.0,
    weight_decay=0.1,
    foreach=False,
    skip_grad_norm_threshold=5,
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
fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=train_ep_size, fp32_lm_head=fp32_lm_head)
train_worker_cfg: WorkerConfig = WorkerConfig(
    model_cfg=model_cfg,
    load_from=model_path,
    optim_cfg=optim_cfg,
    loss_cfg=loss_cfg,
    lr_cfg=lr_cfg,
    fsdp_cfg=fsdp_cfg,
    sp_size=train_sp_size,
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
    advantage_estimator_config=OverlongRLOOGroupEntropyBadwordAdvantageConfig(
        entropy_upper_bound=0.65,
        entropy_lower_bound=0.25,
        tau_upper=0.0,
        tau_lower=0.0,
        coeff_min_upper=0.2,
        coeff_min_lower=0.5,
        overlong_filer=True,
        badword_ratio_cost_factor=1.0,
        tokenizer_path=model_path,
    ),
)


import torch.distributed as dist

from xtuner.v1.train.agent_rl_trainer import AgentRLTrainer

trainer = AgentRLTrainer.from_config(trainer)
trainer.fit()

if dist.is_initialized():
    dist.destroy_process_group()
