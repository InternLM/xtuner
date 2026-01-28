from pathlib import Path

from transformers import AutoConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .base import DEFAULT_FLOAT8_CFG, BaseModel, TorchCompileOption, TransformerConfig, XTunerBaseModelConfig
from .compose.intern_s1 import InternS1BaseConfig, InternS1Config, InternS1MiniConfig
from .compose.internvl import (
    InternVL3P5Dense1BConfig,
    InternVL3P5Dense8BConfig,
    InternVL3P5MoE30BA3Config,
    InternVLBaseConfig,
)
from .compose.qwen3_vl import (
    Qwen3VLBaseConfig,
    Qwen3VLDense4BConfig,
    Qwen3VLDense8BConfig,
    Qwen3VLMoE30BA3Config,
    Qwen3VLMoE235BA22Config,
)
from .dense.dense import Dense
from .dense.qwen2 import Qwen2DenseConfig
from .dense.qwen3 import Qwen3Dense0P6BConfig, Qwen3Dense4BConfig, Qwen3Dense8BConfig, Qwen3DenseConfig
from .moe.deepseek_v3 import DeepSeekV3Config
from .moe.gpt_oss import GptOss21BA3P6Config, GptOss117BA5P8Config, GptOssConfig
from .moe.moe import BalancingLossConfig, MoE, MoEModelOutputs, ZLossConfig
from .moe.qwen3 import Qwen3MoE30BA3Config, Qwen3MoEConfig, Qwen3MoEFoPEConfig


model_mapping = {
    "qwen3-moe-30BA3": Qwen3MoE30BA3Config(),
    "qwen3-8B": Qwen3Dense8BConfig(),
    "qwen3-4B": Qwen3Dense4BConfig(),
    "intern-s1": InternS1Config(),
    "intern-s1-mini": InternS1MiniConfig(),
    "gpt-oss-20b": GptOss21BA3P6Config(),
    "gpt-oss-120b": GptOss117BA5P8Config(),
    "internvl-3.5-8b-hf": InternVL3P5Dense8BConfig(),
    "internvl-3.5-1b-hf": InternVL3P5Dense1BConfig(),
    "internvl-3.5-30b-a3b-hf": InternVL3P5MoE30BA3Config(),
}


def get_model_config(model_alias: str):
    lower_key_mapping = {key.lower().replace("-", "_"): value for key, value in model_mapping.items()}
    return lower_key_mapping.get(model_alias.lower().replace("-", "_"))


def get_model_config_from_hf(model_path: Path):
    """Convert HuggingFace config to XTuner."""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if cfg.model_type == "qwen3_moe":
        return Qwen3MoEConfig.from_hf(model_path)
    elif cfg.model_type == "qwen3_moe_fope":
        return Qwen3MoEFoPEConfig.from_hf(model_path)
    elif cfg.model_type == "qwen2":
        return Qwen2DenseConfig.from_hf(model_path)
    elif cfg.model_type == "qwen3":
        return Qwen3DenseConfig.from_hf(model_path)
    elif cfg.model_type == "gpt_oss":
        return GptOssConfig.from_hf(model_path)
    elif cfg.model_type == "deepseek_v3":
        return DeepSeekV3Config.from_hf(model_path)
    elif cfg.model_type == "qwen3_vl_moe" or cfg.model_type == "qwen3_vl":
        return Qwen3VLBaseConfig.from_hf(model_path)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")


__all__ = [
    "BaseModel",
    "TransformerConfig",
    "Qwen3DenseConfig",
    "Qwen3Dense0P6BConfig",
    "Qwen3Dense8BConfig",
    "Qwen3MoEConfig",
    "Qwen3MoE30BA3Config",
    "InternS1Config",
    "InternS1MiniConfig",
    "InternS1BaseConfig",
    "GptOssConfig",
    "GptOss21BA3P6Config",
    "GptOss117BA5P8Config",
    "InternVLBaseConfig",
    "InternVL3P5Dense1BConfig",
    "InternVL3P5Dense8BConfig",
    "InternVL3P5MoE30BA3Config",
    "get_model_config",
    "get_model_config_from_hf",
    "MoE",
    "MoEModelOutputs",
    "BalancingLossConfig",
    "ZLossConfig",
    "GreedyRouterConfig",
    "Dense",
    "Qwen3VLMoE30BA3Config",
    "Qwen3VLDense4BConfig",
    "Qwen3VLDense8BConfig",
    "Qwen3VLMoE235BA22Config",
    "TorchCompileOption",
    "DEFAULT_FLOAT8_CFG",
    "XTunerBaseModelConfig",
]
