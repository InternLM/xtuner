
from pydantic import computed_field
from typing import Literal
import re
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import MHAConfig, GateDeltaNetConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.module.rope import RopeScalingConfig

from xtuner.v1.model.moe.moe import MoEConfig
from .qwen3vl_text import Qwen3VLTextMoE


class Qwen3_5_VLTextMoEConfig(MoEConfig):
    with_shared_expert_gate: bool = True
    rms_norm_type: Literal["defalut", "zero_centered"] = "zero_centered"

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        return ["full_attention" if bool((i + 1) % 4) else "linear_attention" for i in range(self.num_hidden_layers)]

    def build(self) -> Qwen3VLTextMoE:
        return Qwen3VLTextMoE(self)


class Qwen3_5_VLTextMoE35BA3BConfig(Qwen3_5_VLTextMoEConfig):
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    # Qwen3 Model(dense and moe)'s pad_token_id is not set, so we need to set it to None.
    # If this pad_token_id is not set, the embedding module will not act specially for pad token.
    # Note: Qwen3 Model's pad_token_id may be different from Qwen tokenizer's pad_token_id.
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    num_hidden_layers: int = 40
    max_window_layers: int = 40
    hidden_size: int = 2048
    intermediate_size: int = 0 # not used
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        with_gate=True,
        num_attention_heads=16, 
        num_key_value_heads=2, 
        head_dim=256, 
        qk_norm=True, 
        rms_norm_eps=1e-6,
        rms_norm_type="zero_centered",
        sliding_window=1024
    )
    linear_attention: GateDeltaNetConfig = GateDeltaNetConfig(
        num_value_heads=32,
        num_key_heads=16,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        hidden_act='silu',
        rms_norm_eps=1e-6,
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 512
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    rope_scaling_cfg = RopeScalingConfig(type="qwen3_vl", mrope_section=[11, 11, 10], partial_rotary_factor=0.25)
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None
