from pathlib import Path
from typing import Any, Dict, Optional

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import RLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.utils.rl_test_utils import get_eos_token


def _filter_pydantic_kwargs(target_class: Any, kwargs: Dict) -> Dict:
    accepted_keys = set(target_class.model_fields.keys())
    return {k: v for k, v in kwargs.items() if k in accepted_keys}


def _build_config(config_class, **kwargs):
    filtered_params = _filter_pydantic_kwargs(config_class, kwargs)
    return config_class(**filtered_params)


def get_resources_config(**kwargs) -> AcceleratorResourcesConfig:
    return _build_config(AcceleratorResourcesConfig, **kwargs)


def get_rollout_config(**kwargs) -> RolloutConfig:
    return _build_config(RolloutConfig, **kwargs)


def get_dataflow_config(**kwargs) -> DataFlowConfig:
    return _build_config(DataFlowConfig, **kwargs)


def get_replay_buffer_config(tokenizer: Any, **kwargs) -> ReplayBufferConfig:
    tokenizer_config = RLTokenizeFnConfig(max_length=kwargs["max_prompt_length"])
    train_dataset = DatasetConfig(anno_path=kwargs["data_path"])
    train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
    dataloader_config = DataloaderConfig(collator="fake_collator", pack_level="none")
    return ReplayBufferConfig(
        dataset_cfg=train_dataset_cfg,
        dataloader_cfg=dataloader_config,
        tokenizer=tokenizer,
        postprocessor_func=kwargs.get("filter_func"),
    )


def get_dapo_judger_config(tokenizer: Any, **kwargs):
    dapo_defaults_args = {
        "enable_overlong_buffer": True,
        "overlong_buffer_len": 4096,
        "overlong_penalty_factor": 1.0,
    }
    dapo_config_params = {**dapo_defaults_args, **kwargs}
    eos_token_id = get_eos_token(kwargs["model_path"])
    eos_token_str = tokenizer.convert_ids_to_tokens(eos_token_id)
    from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig

    filtered_params = _filter_pydantic_kwargs(DapoMathJudgerConfig, dapo_config_params)
    dapomath_judger_config = DapoMathJudgerConfig(
        judger_name="dapo_math",
        eos_token=eos_token_str,
        max_response_len=kwargs["max_response_length"],
        tokenizer=tokenizer,
        **filtered_params,
    )
    return JudgerConfig(reward_judger_configs=[dapomath_judger_config])


def get_gsm8k_judger_config():
    from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig

    gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
    judger_cfg = JudgerConfig(reward_judger_configs=[gsm8k_judger_config])
    return judger_cfg


def get_evaluator_config(tokenizer: Any, **kwargs) -> Optional[EvaluatorConfig]:
    if not kwargs["enable_evaluate"]:
        return None

    eval_dataset = DatasetConfig(anno_path=kwargs["eval_data_path"])
    tokenizer_config = RLTokenizeFnConfig(max_length=kwargs["max_prompt_length"])
    eval_dataset_cfg = [{"dataset": eval_dataset, "tokenize_fn": tokenizer_config}]

    filtered_params = _filter_pydantic_kwargs(EvaluatorConfig, kwargs)

    return EvaluatorConfig(
        dataset_cfg=eval_dataset_cfg,
        tokenizer=tokenizer,
        **filtered_params,
    )


def get_train_worker_config(**kwargs) -> WorkerConfig:
    from xtuner.v1.model import get_model_config_from_hf

    model_cfg = get_model_config_from_hf(Path(kwargs["model_path"]))
    defaults = {
        "optim_cfg": AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False),
        "loss_cfg": GRPOLossConfig(
            policy_loss_cfg={
                "cliprange_high": 0.28,
                "cliprange_low": 0.2,
                "loss_type": "vanilla",
                "clip_ratio_c": 10.0,
                "log_prob_diff_min": -20.0,
                "log_prob_diff_max": 20.0,
            },
            ignore_idx=-100,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
        ),
        "lr_cfg": LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
        "fsdp_cfg": FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1),
        "sp_size": 1,
        "optimizer_steps": 16,
        "pack_max_length": 4096,
    }
    config_params = {**defaults, **kwargs}
    filtered_params = _filter_pydantic_kwargs(WorkerConfig, config_params)
    return WorkerConfig(load_from=config_params["model_path"], model_cfg=model_cfg, **filtered_params)
