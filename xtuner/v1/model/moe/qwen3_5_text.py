import re
from typing import Literal

import torch
from pydantic import computed_field
from typing_extensions import override

from xtuner.v1.model.base import (
    DEFAULT_FLOAT8_CFG,
    HFSaveCfg,
    TorchCompileOption,
)
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig

from .qwen3vl_text import Qwen3VLTextMoE


MOE_NON_EP_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward": TorchCompileOption(
        fullgraph=False
    ),
    "xtuner.v1.module.attention.mha.MultiHeadAttention.forward": TorchCompileOption(fullgraph=True),
    # TODO: GatedDeltaNet does not currently support torch.compile(full_graph=True); support will be added in the future.
    "xtuner.v1.module.attention.gated_deltanet.GatedDeltaNet.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._post_moe_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}

MOE_EP_COMPILE_CFG = MOE_NON_EP_COMPILE_CFG.copy()
MOE_EP_COMPILE_CFG.pop("xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward")


class Qwen3_5_VLTextMoE(Qwen3VLTextMoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        # Handle MTP parameters
        if key.startswith("mtp_block."):
            # Extract MTP name from mtp_block.{mtp_name}.{rest}
            # Only "normal" and "sci" are supported
            match = re.match(r"mtp_block\.(normal|sci)\.(.*)", key)
            if not match:
                raise ValueError(
                    f"Invalid mtp_block key format: {key}. "
                    f"Expected 'mtp_block.normal.*' or 'mtp_block.sci.*'"
                )

            mtp_name = match.group(1)
            key = match.group(2)  # Get everything after mtp_block.{mtp_name}.

            # Handle MTP layer-specific parameters
            # xtuner: mtp_block.{mtp_name}.layers.{idx}.decoder_layer.{param}
            # HF normal: mtp.layers.{idx}.{param}
            # HF sci: mtp.sci.layers.{idx}.{param}
            key = re.sub(r"layers\.(\d+)\.decoder_layer\.", r"layers.\1.", key)

            # Handle MTP normalization layers
            # xtuner: mtp_block.{mtp_name}.layers.{idx}.enorm -> HF: mtp[.sci].pre_fc_norm_embedding
            # xtuner: mtp_block.{mtp_name}.layers.{idx}.hnorm -> HF: mtp[.sci].pre_fc_norm_hidden
            # xtuner: mtp_block.{mtp_name}.layers.{idx}.final_layernorm -> HF: mtp[.sci].norm
            if ".enorm." in key:
                key = re.sub(r"layers\.\d+\.enorm\.", "pre_fc_norm_embedding.", key)
            elif ".hnorm." in key:
                key = re.sub(r"layers\.\d+\.hnorm\.", "pre_fc_norm_hidden.", key)
            elif ".final_layernorm." in key:
                key = re.sub(r"layers\.\d+\.final_layernorm\.", "norm.", key)

            # Handle MTP projection layer
            # xtuner: mtp_block.{mtp_name}.layers.{idx}.eh_proj -> HF: mtp[.sci].fc
            if ".eh_proj." in key:
                key = re.sub(r"layers\.\d+\.eh_proj\.", "fc.", key)

            # Handle MoE-specific transformations within MTP layers
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts|shared_expert_gate)", r"layers.\1.mlp.\2", key)
            key = key.replace("shared_experts", "shared_expert")

            # Determine HF prefix based on mtp_name
            # Normal MTP (mtp_block.normal.*): mtp.{key}
            # Science MTP (mtp_block.sci.*): mtp.sci.{key}
            hf_prefix = "mtp." if mtp_name == "normal" else "mtp.sci."

            # Handle fused weights
            n_routed_experts = self.config.n_routed_experts
            if "fused_w1w3.weight" in key:
                w1w3_keys: list[str] = []

                for i in range(n_routed_experts):
                    w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                    w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))

                return [f"{hf_prefix}{key}" for key in w1w3_keys]

            elif "fused_w2.weight" in key:
                w2_keys: list[str] = []
                for i in range(n_routed_experts):
                    w2_keys.append(key.replace("fused_w2.weight", f"{i}.down_proj.weight"))
                return [f"{hf_prefix}{key}" for key in w2_keys]
            else:
                return [hf_prefix + key]

        # Handle main model parameters
        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts|shared_expert_gate)", r"layers.\1.mlp.\2", key)
            key = key.replace("shared_experts", "shared_expert")

            layer_idx = int(re.findall(r"layers\.(\d+)\.", key)[0])
            if self.config.layers_type[layer_idx] == "linear_attention":
                key = key.replace("self_attn", "linear_attn")

        if "fused_w1w3.weight" in key:
            key = key.replace("fused_w1w3.weight", "gate_up_proj")
        elif "fused_w2.weight" in key:
            key = key.replace("fused_w2.weight", "down_proj")
        if "fused_w1w3.bias" in key:
            key = key.replace("fused_w1w3.bias", "gate_up_proj_bias")
        elif "fused_w2.bias" in key:
            key = key.replace("fused_w2.bias", "down_proj_bias")

        if key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]
        elif key.startswith("rotary_emb."):
            # FoPE has model.rotary_emb.sin_coef and model.rotary_emb.cos_coef in the safetensors
            return [key.replace("rotary_emb.", "model.language_model.rotary_emb.")]
        else:
            return [key]

    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ):
        if len(safetensors) > 1:
            assert dim is not None, "Internal Error dim must not be None when len(safetensors) > 1"
            loaded_tensor = torch.cat(safetensors, dim=dim)
        else:
            loaded_tensor = safetensors[0]

        if "fused_w1w3.weight" in param_name and "mtp" not in param_name:
            # hf: num_experts, 2 * expert_dim, hidden_size
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            # num_experts * 2 * expert_dim, hidden_size
            loaded_tensor = loaded_tensor.flatten(0, 1)

        elif "fused_w2.weight" in param_name and "mtp" not in param_name:
            # hf: num_experts, hidden_size, expert_dim
            # xtuner: num_experts * hidden_size, expert_dim
            loaded_tensor = loaded_tensor.flatten(0, 1)

        if start is not None and end is not None:
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = end - start
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)

            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        if "mtp" in hf_param_name:
            return safetensor

        assert isinstance(hf_param_name, str)
        if "gate_up_proj" in hf_param_name:
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            # hf: num_experts, 2 * expert_dim, hidden_size
            num_experts = self.config.n_routed_experts
            hidden_size = safetensor.size(1)
            safetensor = safetensor.reshape(
                num_experts, -1, hidden_size
            ).contiguous()  # num_experts, 2 * expert_dim, hidden_size
        elif "down_proj" in hf_param_name and "shared_expert" not in hf_param_name:
            # xtuner: num_experts * hidden_size, expert_dim
            # hf: num_experts, hidden_size, expert_dim
            num_experts = self.config.n_routed_experts
            expert_dim = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, expert_dim).contiguous()
        return safetensor

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.ep_size > 1:
            return MOE_EP_COMPILE_CFG
        else:
            return MOE_NON_EP_COMPILE_CFG


class Qwen3_5_VLTextMoEConfig(MoEConfig):
    with_shared_expert_gate: bool = True
    rms_norm_type: Literal["default", "zero_centered"] = "zero_centered"
    hf_save_cfg: HFSaveCfg = HFSaveCfg(
        fp32_keys_pattern=[
            r"model\.language_model\.layers\.\d+\.linear_attn\.norm\.weight",
            r"model\.language_model\.layers\.\d+\.linear_attn\.A_log",
        ],
    )

    @computed_field
    def layers_type(self) -> list[Literal["full_attention", "linear_attention"]]:
        return ["linear_attention" if bool((i + 1) % 4) else "full_attention" for i in range(self.num_hidden_layers)]

    def build(self) -> Qwen3_5_VLTextMoE:
        return Qwen3_5_VLTextMoE(self)


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
