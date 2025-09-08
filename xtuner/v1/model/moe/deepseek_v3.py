import re

from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import MLAConfig
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
    pad_token_id: int = 1  # eos_id
    num_hidden_layers: int = 61
    first_k_dense_replace: int = 3
    max_window_layers: int = 61
    hidden_size: int = 7168
    intermediate_size: int = 18432
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict = dict(
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
        # rope_scaling=dict(
        #     type="yarn",beta_fast=32, beta_slow=1, factor=40, mscale=1.0, mscale_all_dim=1.0, original_max_position_embeddings=4096
        # ),
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
