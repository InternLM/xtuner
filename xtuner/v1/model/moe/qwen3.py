import re
from pathlib import Path

import torch
from typing_extensions import Self

from transformers.models.qwen3_moe import Qwen3MoeConfig as HFQwen3MoeConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .moe import MoE


class Qwen3MoE(MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

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
        else:
            return [key]


class Qwen3MoEConfig(MoEConfig):
    bos_token_id: int

    def build(self) -> Qwen3MoE:
        return Qwen3MoE(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        hf_config = HFQwen3MoeConfig.from_pretrained(hf_path)

        assert isinstance(hf_config, HFQwen3MoeConfig)

        config = cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            pad_token_id=getattr(hf_config, "pad_token_id"),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            num_hidden_layers=hf_config.num_hidden_layers,
            max_window_layers=getattr(hf_config, "max_window_layers"),
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            hidden_act=hf_config.hidden_act,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                head_dim=hf_config.head_dim,
                sliding_window=hf_config.sliding_window,
                qk_norm=True,
            ),
            use_sliding_window=hf_config.use_sliding_window,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            n_routed_experts=hf_config.num_experts,
            n_shared_experts=0,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            router=GreedyRouterConfig(
                scoring_func="softmax",
                norm_topk_prob=hf_config.norm_topk_prob,
                router_scaling_factor=1.0,
            ),
            balancing_loss_cfg=BalancingLossConfig(),
        )

        return config

    @property
    def hf_config(self) -> HFQwen3MoeConfig:
        """HuggingFace configuration."""
        assert isinstance(self.router, GreedyRouterConfig), "Only support saving GreedyRouter to HF Qwen3MoE format."
        return HFQwen3MoeConfig(
            architectures=["Qwen3MoeForCausalLM"],
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_hidden_layers=self.num_hidden_layers,
            max_window_layers=self.max_window_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_key_value_heads,
            head_dim=self.attention.head_dim,
            sliding_window=self.attention.sliding_window,
            use_sliding_window=self.use_sliding_window,
            tie_word_embeddings=self.tie_word_embeddings,
            num_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            norm_topk_prob=self.router.norm_topk_prob,
            dtype=torch.bfloat16,
        )


class Qwen3MoE30BA3Config(Qwen3MoEConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    # Qwen3 Model(dense and moe)'s pad_token_id is not set, so we need to set it to None.
    # If this pad_token_id is not set, the embedding module will not act specially for pad token.
    # Note: Qwen3 Model's pad_token_id may be different from Qwen tokenizer's pad_token_id.
    pad_token_id: int | None = None
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    num_hidden_layers: int = 48
    max_window_layers: int = 48
    hidden_size: int = 2048
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=32, num_key_value_heads=4, head_dim=128, qk_norm=True, sliding_window=1024
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 128
    n_shared_experts: int = 0
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 768
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None


class Qwen3MoE235BA22Config(Qwen3MoEConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    pad_token_id: int | None = None
    eos_token_id: int = 151645
    bos_token_id: int = 151643
    num_hidden_layers: int = 94
    max_window_layers: int = 94
    hidden_size: int = 4096
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=64, num_key_value_heads=4, head_dim=128, qk_norm=True, sliding_window=1024
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 128
    n_shared_experts: int = 0
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 1536
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None
