import re

from xtuner.v1.config import BaseRouterConfig, MoEConfig
from xtuner.v1.config.loss import BalancingLossConfig, ZLossConfig
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


class Qwen3MoE30BA3Config(MoEConfig):
    vocab_size: int = 151936
    max_position_embeddings: int = 4096
    padding_idx: int = 0
    num_hidden_layers: int = 48
    hidden_size: int = 2048
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(num_attention_heads=32, num_key_value_heads=4, head_dim=128, qk_norm=True)
    tie_word_embeddings: bool = False
    n_routed_experts: int = 128
    n_shared_experts: int = 0
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 768
    router: BaseRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None

    def build(self) -> Qwen3MoE:
        return Qwen3MoE(self)
