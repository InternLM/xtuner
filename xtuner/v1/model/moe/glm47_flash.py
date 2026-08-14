# Copyright (c) OpenMMLab. All rights reserved.
import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, computed_field
from typing_extensions import Self


try:
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig as HFGlm4MoeLiteConfig
except ImportError:
    HFGlm4MoeLiteConfig = None  # type: ignore[misc, assignment]
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import MLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .moe import MoE


class Glm47Flash(MoE):
    """XTuner training model for Hugging Face ``glm4_moe_lite`` checkpoints."""

    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if key.startswith("mtp_block."):
            match = re.match(r"mtp_block\.layers\.(\d+)\.(.+)", key)
            assert match is not None, f"Unexpected GLM-4.7 Flash MTP key: {key}"
            mtp_layer_idx = self.config.num_hidden_layers + int(match.group(1))
            key = f"layers.{mtp_layer_idx}.{match.group(2)}"
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            return [key.replace("fused_w1w3.weight", "gate_up_proj")]
        if "fused_w2.weight" in key:
            return [key.replace("fused_w2.weight", "down_proj")]
        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]

    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ) -> None:
        loaded_tensor = torch.cat(safetensors, dim=dim) if len(safetensors) > 1 and dim is not None else safetensors[0]
        if ("fused_w1w3.weight" in param_name or "fused_w2.weight" in param_name) and loaded_tensor.ndim == 3:
            loaded_tensor = loaded_tensor.flatten(0, 1)

        if start is not None and end is not None:
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            index = torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            loaded_tensor_slice = loaded_tensor.index_select(dim=self.FSDP_SHARD_DIM, index=index)
            non_pad_len = end - start
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)
            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].zero_()
        else:
            local_tensor.copy_(loaded_tensor)

    def param_to_safetensor(self, safetensor: torch.Tensor, hf_param_name: str) -> torch.Tensor:
        if hf_param_name.endswith("experts.gate_up_proj"):
            return safetensor.reshape(self.config.n_routed_experts, -1, self.config.hidden_size).contiguous()
        if hf_param_name.endswith("experts.down_proj"):
            return safetensor.reshape(self.config.n_routed_experts, self.config.hidden_size, -1).contiguous()
        return safetensor


class Glm47FlashConfig(MoEConfig):
    """Configuration for training the 30B-A3B GLM-4.7-Flash checkpoint."""

    model_type: str = "glm4_moe_lite"
    vocab_size: int = 154880
    max_position_embeddings: int = 202752
    pad_token_id: int | None = 154820
    eos_token_id: int = 154820
    hf_eos_token_id: int | list[int] = Field(default_factory=lambda: [154820, 154827, 154829])
    num_hidden_layers: int = 47
    first_k_dense_replace: int = 1
    hidden_size: int = 2048
    intermediate_size: int = 10240
    rms_norm_eps: float = 1e-5
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=1000000.0)
    )
    rope_interleave: bool = True
    hidden_act: str = "silu"
    attention: MLAConfig = MLAConfig(
        kv_lora_rank=512,
        q_lora_rank=768,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        head_dim=64,
        num_attention_heads=20,
        qkv_bias=False,
        o_bias=False,
        rope_interleave=True,
    )
    hf_head_dim: int = 64
    qk_head_dim: int = 256
    tie_word_embeddings: bool = False
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 1536
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=1.8,
    )
    balancing_loss_cfg: BalancingLossConfig | None = None
    z_loss_cfg: ZLossConfig | None = None
    mlp_layer_types: list[Literal["dense", "sparse"]] | None = None
    num_nextn_predict_layers: int = 1
    mtp_config: MTPConfig | None = MTPConfig(num_layers=1, share_weights=True)

    @computed_field
    def num_key_value_heads(self) -> int:
        return self.attention.num_attention_heads

    def build(self) -> Glm47Flash:
        return Glm47Flash(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        if HFGlm4MoeLiteConfig is None:
            raise ImportError("GLM-4.7 Flash requires a Transformers version with glm4_moe_lite support.")
        cfg = HFGlm4MoeLiteConfig.from_pretrained(hf_path)
        rope_parameters_cfg = RopeParametersConfig.from_hf_config(cfg)
        mlp_layer_types = list(cfg.mlp_layer_types)
        expected_mlp_types = ["dense"] * cfg.first_k_dense_replace + ["sparse"] * (
            cfg.num_hidden_layers - cfg.first_k_dense_replace
        )
        if mlp_layer_types != expected_mlp_types:
            raise ValueError("XTuner currently requires GLM-4.7 Flash dense MLP layers to form a prefix.")

        num_nextn_predict_layers = int(getattr(cfg, "num_nextn_predict_layers", 0))
        return cls(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=getattr(cfg, "pad_token_id", None),
            eos_token_id=cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id,
            hf_eos_token_id=cfg.eos_token_id,
            num_hidden_layers=cfg.num_hidden_layers,
            first_k_dense_replace=cfg.first_k_dense_replace,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_parameters_cfg=rope_parameters_cfg,
            rope_interleave=cfg.rope_interleave,
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
                dropout=cfg.attention_dropout,
                rope_interleave=cfg.rope_interleave,
            ),
            hf_head_dim=cfg.qk_rope_head_dim,
            qk_head_dim=cfg.qk_head_dim,
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            moe_intermediate_size=cfg.moe_intermediate_size,
            router=NoAuxRouterConfig(
                n_group=cfg.n_group,
                topk_group=cfg.topk_group,
                scoring_func="sigmoid",
                norm_topk_prob=cfg.norm_topk_prob,
                router_scaling_factor=cfg.routed_scaling_factor,
            ),
            balancing_loss_cfg=None,
            mlp_layer_types=mlp_layer_types,
            num_nextn_predict_layers=num_nextn_predict_layers,
            mtp_config=MTPConfig(num_layers=num_nextn_predict_layers, share_weights=True)
            if num_nextn_predict_layers
            else None,
        )

    @property
    def hf_config(self):
        if HFGlm4MoeLiteConfig is None:
            return None
        return HFGlm4MoeLiteConfig(
            architectures=["Glm4MoeLiteForCausalLM"],
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.hf_eos_token_id,
            num_hidden_layers=self.num_hidden_layers,
            first_k_dense_replace=self.first_k_dense_replace,
            mlp_layer_types=self.mlp_layer_types,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_parameters=self.rope_parameters,
            rope_interleave=self.rope_interleave,
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_attention_heads,
            kv_lora_rank=self.attention.kv_lora_rank,
            q_lora_rank=self.attention.q_lora_rank,
            qk_nope_head_dim=self.attention.qk_nope_head_dim,
            qk_rope_head_dim=self.attention.qk_rope_head_dim,
            v_head_dim=self.attention.v_head_dim,
            attention_bias=self.attention.qkv_bias or self.attention.o_bias,
            attention_dropout=self.attention.dropout,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.router.n_group,
            topk_group=self.router.topk_group,
            norm_topk_prob=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            tie_word_embeddings=self.tie_word_embeddings,
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            dtype=torch.bfloat16,
        )
