import os
from copy import deepcopy

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.model.dense.qwen2 import Qwen2Dense7BConfig
from xtuner.v1.train.rl_trainer import RLTrainerConfig

from xtuner.v1.utils.rl_common_config import (
    get_gsm8k_judger_config,
    get_dataflow_config,
    get_evaluator_config,
    get_replay_buffer_config,
    get_resources_config,
    get_rollout_config,
    get_train_worker_config,
)

# 实验超参
params = {
    "work_dir": os.environ.get("WORK_DIR", ""),
    "model_path": os.environ.get("MODEL_PATH", ""),
    "data_path": os.environ.get("DATA_PATH", ""),
    "eval_data_path": os.environ.get("EVAL_DATA_PATH", ""),
    # training settings
    "total_epochs": 15,
    "train_optimizer_steps": 4,
    "hf_interval": 15,
    # model settings
    "max_prompt_length": 512,
    "max_response_length": 1024,
    # rollout settings
    "tensor_parallel_size": 2,
    "expert_parallel_size": 1,
    "rollout_max_batch_size_per_instance": 1024,
    # dataflow settings
    "global_batch_size": 1024,
    "prompt_repeat_k": 5,
    "enable_partial_rollout": 0,
    "partial_rollout_step": 0,
    "max_concurrent": 512,
    # evaluate settings
    "enable_evaluate": os.environ.get("EVAL_DATA_PATH") != "",
    "enable_initial_evaluate": True,
    "evaluate_step": 10,
    # resource setting
    "num_workers": 8,
}
params["context_length"] = params["pack_max_length"]
if params["pack_max_length"] < params["max_prompt_length"] + params["max_response_length"]:
    params["pack_max_length"] = params["max_prompt_length"] + params["max_response_length"]

# 创建核心组件
tokenizer = AutoTokenizer.from_pretrained(params["model_path"], trust_remote_code=True)
training_sample_params = SampleParams(max_tokens=params["max_response_length"])
evaluation_sample_params = deepcopy(training_sample_params)
evaluation_sample_params.top_p = 1.0
evaluation_sample_params.temperature = 0.0
evaluation_sample_params.top_k = 1.0

resource_cfg = get_resources_config(**params)
rollout_cfg = get_rollout_config(**params)
gsm8k_judger_cfg = get_gsm8k_judger_config()
dataflow_cfg = get_dataflow_config(sample_params=training_sample_params, **params)
replay_buffer_cfg = get_replay_buffer_config(tokenizer=tokenizer, **params)
evaluator_cfg = get_evaluator_config(tokenizer=tokenizer,sample_params=evaluation_sample_params,**params)
train_worker_cfg = get_train_worker_config(**params)

# 组装RL trainer
trainer = RLTrainerConfig(
    work_dir=params["work_dir"],
    load_from=params["model_path"],
    tokenizer_path=params["model_path"],
    total_epochs=params["total_epochs"],
    hf_interval=params["hf_interval"],
    resources=resource_cfg,
    rollout_config=rollout_cfg,
    dataflow_config=dataflow_cfg,
    judger_config=gsm8k_judger_cfg,
    replay_buffer_config=replay_buffer_cfg,
    evaluator_config=evaluator_cfg,
    train_worker_config=train_worker_cfg,
)