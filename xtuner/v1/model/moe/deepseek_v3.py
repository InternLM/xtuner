import re
from pathlib import Path

import torch
from typing_extensions import Self
from transformers import PretrainedConfig
from transformers.models.deepseek_v3 import DeepseekV3Config as HFDeepseekV3Config
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import MLAConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils import get_logger

from .moe import MoE


logger = get_logger()


class DeepSeekV3(MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)

        n_routed_experts = self.config.n_routed_experts

        if "fused_w1w3.weight" in key:
            w1w3_keys: list[str] = []

            for i in range(n_routed_experts):
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))

            return w1w3_keys

        elif "fused_w2.weight" in key:
            w2_keys: list[str] = []
            for i in range(n_routed_experts):
                w2_keys.append(key.replace("fused_w2.weight", f"{i}.down_proj.weight"))
            return w2_keys

        elif key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        elif "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        else:
            return [key]


class DeepSeekV3Config(MoEConfig):
    vocab_size: int = 129280
    max_position_embeddings: int = 163840
    pad_token_id: int | None = None
    eos_token_id: int = 1
    num_hidden_layers: int = 61
    first_k_dense_replace: int = 3
    max_window_layers: int = 61
    hidden_size: int = 7168
    intermediate_size: int = 18432
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling_cfg: RopeScalingConfig = RopeScalingConfig(
        type="yarn",
        beta_fast=32,
        beta_slow=1,
        factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        original_max_position_embeddings=4096,
    )
    hidden_act: str = "silu"
    attention: MLAConfig = MLAConfig(
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        head_dim=64,
        num_attention_heads=128,
        qkv_bias=False,
        o_bias=False,
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 2048
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=2.5,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None

    def build(self) -> DeepSeekV3:
        return DeepSeekV3(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path | None = None, hf_config: PretrainedConfig | None = None) -> Self:
        if hf_path is not None:
            cfg = HFDeepseekV3Config.from_pretrained(hf_path)
            assert isinstance(cfg, HFDeepseekV3Config)
        else:
            cfg = hf_config
        assert cfg is not None and isinstance(cfg, PretrainedConfig)

        config = cls(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=getattr(cfg, "pad_token_id"),
            eos_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            first_k_dense_replace=cfg.first_k_dense_replace,
            max_window_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            rope_scaling_cfg=RopeScalingConfig(
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

        return config

    @property
    def hf_config(self):
        """HuggingFace configuration."""
        assert isinstance(self.router, NoAuxRouterConfig), "Only support saving NoAuxRouter to HF DeepSeekV3 format."
        return HFDeepseekV3Config(
            architectures=["DeepseekV3ForCausalLM"],
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.pad_token_id,
            num_hidden_layers=self.num_hidden_layers,
            first_k_dense_replace=self.first_k_dense_replace,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling_cfg.model_dump() if self.rope_scaling_cfg is not None else None,
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            kv_lora_rank=self.attention.kv_lora_rank,
            q_lora_rank=self.attention.q_lora_rank,
            qk_nope_head_dim=self.attention.qk_nope_head_dim,
            qk_rope_head_dim=self.attention.qk_rope_head_dim,
            v_head_dim=self.attention.v_head_dim,
            attention_bias=self.attention.qkv_bias or self.attention.o_bias,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.router.n_group,
            topk_group=self.router.topk_group,
            scoring_func=self.router.scoring_func,
            norm_topk_prob=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            tie_word_embeddings=self.tie_word_embeddings,
            dtype=torch.bfloat16,
        )
