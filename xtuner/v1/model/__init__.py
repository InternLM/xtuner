from pathlib import Path
from typing import cast

from transformers import AutoConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig
from xtuner.v1.config.loss import BalancingLossConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .base import BaseModel
from .dense.qwen3 import Qwen3_8BConfig, Qwen3DenseConfig
from .moe.qwen3 import Qwen3MoE30BA3Config, Qwen3MoEConfig


model_mapping = {
    "qwen3-moe-30BA3": Qwen3MoE30BA3Config(),
    "qwen3-8B": Qwen3_8BConfig(),
}


def get_model_config(model_alias: str):
    lower_key_mapping = {key.lower().replace("-", "_"): value for key, value in model_mapping.items()}
    return lower_key_mapping.get(model_alias.lower().replace("-", "_"))


def get_model_config_from_hf(model_path: Path):
    """Convert HuggingFace config to XTuner Qwen3MoEConfig."""
    cfg = cast(Qwen3MoeConfig, AutoConfig.from_pretrained(model_path))

    if cfg.model_type == "qwen3_moe":
        return Qwen3MoEConfig(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=0,
            num_hidden_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                qk_norm=True,
            ),
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.num_experts,
            n_shared_experts=0,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=cfg.norm_topk_prob,
                router_scaling_factor=1.0,
            ),
            balancing_loss_cfg=BalancingLossConfig(),
        )
    elif cfg.model_type == "qwen3":
        return Qwen3DenseConfig(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                qk_norm=True,
            ),
            tie_word_embeddings=cfg.tie_word_embeddings,
        )

    raise ValueError(f"Unsupported model type: {cfg.model_type}")
