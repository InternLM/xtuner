from pathlib import Path

from transformers import AutoConfig
from xtuner.v1.module.attention import MHAConfig, MLAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .base import BaseModel, TransformerConfig
from .compose.intern_s1 import InternS1BaseConfig, InternS1Config, InternS1MiniConfig
from .compose.internvl import InternVL3P5Dense8BConfig, InternVL3P5MoE30BA3Config, InternVLBaseConfig
from .dense.dense import Dense
from .dense.qwen3 import Qwen3Dense8BConfig, Qwen3DenseConfig
from .moe.gpt_oss import GptOss21BA3P6Config, GptOss117BA5P8Config, GptOssConfig
from .moe.deepseek_v3 import DeepSeekV3Config
from .moe.moe import BalancingLossConfig, MoE, MoEModelOutputs, ZLossConfig
from .moe.qwen3 import Qwen3MoE30BA3Config, Qwen3MoEConfig


model_mapping = {
    "qwen3-moe-30BA3": Qwen3MoE30BA3Config(),
    "qwen3-8B": Qwen3Dense8BConfig(),
    "intern-s1": InternS1Config(),
    "intern-s1-mini": InternS1MiniConfig(),
    "gpt-oss-20b": GptOss21BA3P6Config(),
    "gpt-oss-120b": GptOss117BA5P8Config(),
    "internvl-3.5-8b-hf": InternVL3P5Dense8BConfig(),
    "internvl-3.5-30b-a3b-hf": InternVL3P5MoE30BA3Config(),
}


def get_model_config(model_alias: str):
    lower_key_mapping = {key.lower().replace("-", "_"): value for key, value in model_mapping.items()}
    return lower_key_mapping.get(model_alias.lower().replace("-", "_"))


def get_model_config_from_hf(model_path: Path):
    """Convert HuggingFace config to XTuner."""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if cfg.model_type == "qwen3_moe":
        return Qwen3MoEConfig(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            max_window_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                sliding_window=cfg.sliding_window,
                qk_norm=True,
            ),
            use_sliding_window=cfg.use_sliding_window,
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
            max_window_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                sliding_window=cfg.sliding_window,
                qk_norm=True,
            ),
            use_sliding_window=cfg.use_sliding_window,
            tie_word_embeddings=cfg.tie_word_embeddings,
        )
    elif cfg.model_type == "gpt_oss":
        return GptOssConfig(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=cfg.pad_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            moe_intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            hidden_act=cfg.hidden_act,
            attention=MHAConfig(
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                rms_norm_eps=cfg.rms_norm_eps,
                sliding_window=cfg.sliding_window,
                with_sink=True,
                qkv_bias=True,
                o_bias=True,
            ),
            n_routed_experts=cfg.num_local_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            tie_word_embeddings=cfg.tie_word_embeddings,
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=True,
                router_scaling_factor=1.0,
            ))
    elif cfg.model_type == "deepseek_v3":
        return DeepSeekV3Config(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            first_k_dense_replace=cfg.first_k_dense_replace,
            max_window_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            rope_scaling=dict(
                type=cfg.rope_scaling.get("type", "yarn"),
                beta_fast=cfg.rope_scaling.get("beta_fast", 32),
                beta_slow=cfg.rope_scaling.get("beta_slow", 1),
                factor=cfg.rope_scaling.get("factor", 40.0),
                mscale=cfg.rope_scaling.get("mscale", 1.0),
                mscale_all_dim=cfg.rope_scaling.get("mscale_all_dim", 1.0),
                original_max_position_embeddings=cfg.rope_scaling.get("original_max_position_embeddings", 4096),
            )
            if cfg.rope_scaling is not None
            else None,
            hidden_act=cfg.hidden_act,
            attention=MLAConfig(
                kv_lora_rank=cfg.kv_lora_rank,
                q_lora_rank=cfg.q_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                head_dim=cfg.qk_rope_head_dim,
                num_attention_heads=cfg.num_attention_heads,
                qkv_bias=cfg.attention_bias,
                o_bias=cfg.attention_bias,
            ),
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            hidden_factor=1.0,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router=NoAuxRouterConfig(
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
                scoring_func=cfg.scoring_func,
                norm_topk_prob=cfg.norm_topk_prob,
                router_scaling_factor=cfg.routed_scaling_factor,
            ),
            balancing_loss_cfg=BalancingLossConfig(),
        )

    raise ValueError(f"Unsupported model type: {cfg.model_type}")


__all__ = [
    "BaseModel",
    "TransformerConfig",
    "Qwen3DenseConfig",
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
]
