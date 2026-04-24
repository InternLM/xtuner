import re
from typing import Literal

from pydantic import computed_field

from xtuner.v1.model.base import (
    HFSaveCfg,
)
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .moe import MoE


# Qwen3.5 series's hf model save experts with fused format. XTuner defines a split style model and model config
# to speedup the saving and loading progress.
# You can sue `xtuner/tools/model_converters/split_qwen3_5_moe_fused_experts.py` to convert the official weight
# to split format.
class Qwen3_5_VLTextMoESplit(MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        # Handle MTP parameters
        if key.startswith("mtp_block."):
            # Remove "mtp_block." prefix
            key = key.replace("mtp_block.", "", 1)

            # Handle MTP layer-specific parameters
            # xtuner: mtp_block.layers.{idx}.decoder_layer.{param}
            # HF: mtp.layers.{idx}.{param}
            key = re.sub(r"layers\.(\d+)\.decoder_layer\.", r"layers.\1.", key)

            # Handle MTP normalization layers
            # xtuner: mtp_block.layers.{idx}.enorm -> HF: mtp.pre_fc_norm_embedding
            # xtuner: mtp_block.layers.{idx}.hnorm -> HF: mtp.pre_fc_norm_hidden
            # xtuner: mtp_block.layers.{idx}.final_layernorm -> HF: mtp.norm
            # Note: Currently assuming single MTP layer (idx=0), may need adjustment for multiple layers
            if ".enorm." in key:
                key = re.sub(r"layers\.\d+\.enorm\.", "pre_fc_norm_embedding.", key)
            elif ".hnorm." in key:
                key = re.sub(r"layers\.\d+\.hnorm\.", "pre_fc_norm_hidden.", key)
            elif ".final_layernorm." in key:
                key = re.sub(r"layers\.\d+\.final_layernorm\.", "norm.", key)

            # Handle MTP projection layer
            # xtuner: mtp_block.layers.{idx}.eh_proj -> HF: mtp.fc
            if ".eh_proj." in key:
                key = re.sub(r"layers\.\d+\.eh_proj\.", "fc.", key)

            # Handle MoE-specific transformations within MTP layers
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts|shared_expert_gate)", r"layers.\1.mlp.\2", key)
            key = key.replace("shared_experts", "shared_expert")

            # Handle fused weights
            n_routed_experts = self.config.n_routed_experts
            if "fused_w1w3.weight" in key:
                w1w3_keys: list[str] = []

                for i in range(n_routed_experts):
                    w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                    w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))

                return [f"mtp.{key}" for key in w1w3_keys]

            elif "fused_w2.weight" in key:
                w2_keys: list[str] = []
                for i in range(n_routed_experts):
                    w2_keys.append(key.replace("fused_w2.weight", f"{i}.down_proj.weight"))
                return [f"mtp.{key}" for key in w2_keys]
            else:
                return ["mtp." + key]

        # Handle main model parameters
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts|shared_expert_gate)", r"layers.\1.mlp.\2", key)
            key = key.replace("shared_experts", "shared_expert")

            layer_idx = int(re.findall(r"layers\.(\d+)\.", key)[0])
            if self.config.layers_type[layer_idx] == "linear_attention":
                key = key.replace("self_attn", "linear_attn")

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

        # MoE bias tensors are [num_experts, out_features], while BaseModel's FUSED save path
        # partitions only dim0 evenly across HF keys. Keep the fused bias naming unless that save path
        # learns a separate split rule for 2D grouped-linear bias tensors.
        if "fused_w1w3.bias" in key:
            key = key.replace("fused_w1w3.bias", "gate_up_proj_bias")
        elif "fused_w2.bias" in key:
            key = key.replace("fused_w2.bias", "down_proj_bias")

        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        elif key.startswith("rotary_emb."):
            # FoPE has model.rotary_emb.sin_coef and model.rotary_emb.cos_coef in the safetensors
            return [key.replace("rotary_emb.", "model.rotary_emb.")]
        else:
            return [key]


class Qwen3_5_VLTextMoESplitConfig(MoEConfig):
    with_shared_expert_gate: bool = True
    rms_norm_type: Literal["default", "zero_centered"] = "zero_centered"
    hf_save_cfg: HFSaveCfg = HFSaveCfg(
        fp32_keys_pattern=[
            r"model(?:\.language_model)?\.layers\.\d+\.linear_attn\.norm\.weight",
            r"model(?:\.language_model)?\.layers\.\d+\.linear_attn\.A_log",
        ],
    )

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        return ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(self.num_hidden_layers)]

    def build(self) -> Qwen3_5_VLTextMoESplit:
        return Qwen3_5_VLTextMoESplit(self)


class Qwen3_5_VLTextMoE35BA3BSplitConfig(Qwen3_5_VLTextMoESplitConfig):
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    num_hidden_layers: int = 40
    max_window_layers: int = 40
    hidden_size: int = 2048
    intermediate_size: int = 0  # not used
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
        sliding_window=1024,
    )
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=32,
        num_key_heads=16,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        hidden_act="silu",
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
    rope_scaling_cfg: RopeScalingConfig = RopeScalingConfig(
        type="qwen3_vl", mrope_section=[11, 11, 10], partial_rotary_factor=0.25
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None


class Qwen3_5_VLTextMoE397BA17BSplitConfig(Qwen3_5_VLTextMoESplitConfig):
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    # Qwen3 Model(dense and moe)'s pad_token_id is not set, so we need to set it to None.
    # If this pad_token_id is not set, the embedding module will not act specially for pad token.
    # Note: Qwen3 Model's pad_token_id may be different from Qwen tokenizer's pad_token_id.
    pad_token_id: int | None = None
    eos_token_id: int = 248044
    num_hidden_layers: int = 60
    max_window_layers: int = 60
    hidden_size: int = 4096
    intermediate_size: int = 0  # not used
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        with_gate=True,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=256,
        qk_norm=True,
        rms_norm_eps=1e-6,
        rms_norm_type="zero_centered",
        sliding_window=1024,
    )
    linear_attention: GatedDeltaNetConfig = GatedDeltaNetConfig(
        num_value_heads=64,
        num_key_heads=16,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 10
    first_k_dense_replace: int = 0
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 1024
    router: GreedyRouterConfig = GreedyRouterConfig(
        scoring_func="softmax",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    rope_scaling_cfg: RopeScalingConfig = RopeScalingConfig(
        type="qwen3_vl", mrope_section=[11, 11, 10], partial_rotary_factor=0.25
    )
    balancing_loss_cfg: BalancingLossConfig | None = BalancingLossConfig()
    z_loss_cfg: ZLossConfig | None = None
