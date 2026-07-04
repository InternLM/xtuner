import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, computed_field
from typing_extensions import Self

from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as HFGlmMoeDsaConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import DSAMLAConfig, MLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .moe import MoE


class Glm52MoE(MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate|shared_experts)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            w1w3_keys: list[str] = []
            for i in range(self.config.n_routed_experts):
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))
            return w1w3_keys
        elif "fused_w2.weight" in key:
            return [
                key.replace("fused_w2.weight", f"{i}.down_proj.weight") for i in range(self.config.n_routed_experts)
            ]
        elif key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        elif "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
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

        if (
            "fused_w1w3.weight" in param_name or "fused_w2.weight" in param_name
        ) and loaded_tensor.ndim == local_tensor.ndim + 1:
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
                local_tensor[non_pad_len:].zero_()
        else:
            local_tensor.copy_(loaded_tensor)

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        assert isinstance(hf_param_name, str)
        if hf_param_name.endswith("experts.gate_up_proj"):
            safetensor = safetensor.reshape(self.config.n_routed_experts, -1, self.config.hidden_size)
        elif hf_param_name.endswith("experts.down_proj"):
            safetensor = safetensor.reshape(self.config.n_routed_experts, self.config.hidden_size, -1)
        return safetensor


class Glm52MoEConfig(MoEConfig):
    model_type: str = "glm_moe_dsa"
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    pad_token_id: int | None = 154820
    eos_token_id: int = 154820
    hf_eos_token_id: int | list[int] = Field(default_factory=lambda: [154820, 154827, 154829])
    num_hidden_layers: int = 78
    first_k_dense_replace: int = 3
    hidden_size: int = 6144
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-5
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=8000000.0)
    )
    hidden_act: str = "silu"
    attention: MLAConfig = DSAMLAConfig(
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        head_dim=64,
        num_attention_heads=64,
        qkv_bias=False,
        o_bias=False,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_rope_interleave=True,
    )
    hf_head_dim: int = 192
    qk_head_dim: int = 256
    tie_word_embeddings: bool = False
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    hidden_factor: float = 1.0
    moe_intermediate_size: int = 2048
    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=2.5,
    )
    balancing_loss_cfg: BalancingLossConfig | None = None
    z_loss_cfg: ZLossConfig | None = None
    mlp_layer_types: list[Literal["dense", "sparse"]] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    index_share_for_mtp_iteration: bool = True
    indexer_rope_interleave: bool = True
    indexer_types: list[str] | None = None
    num_nextn_predict_layers: int | None = 1
    mtp_config: MTPConfig | None = None

    @computed_field
    def num_key_value_heads(self) -> int:
        return self.attention.num_attention_heads

    def build(self) -> Glm52MoE:
        return Glm52MoE(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        cfg = HFGlmMoeDsaConfig.from_pretrained(hf_path)

        assert isinstance(cfg, HFGlmMoeDsaConfig)

        rope_parameters_cfg = RopeParametersConfig.from_hf_config(cfg)
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
            model_type=cfg.model_type,
            rope_parameters_cfg=rope_parameters_cfg,
            hidden_act=cfg.hidden_act,
            attention=DSAMLAConfig(
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
                index_topk=cfg.index_topk,
                index_head_dim=cfg.index_head_dim,
                index_n_heads=cfg.index_n_heads,
                index_topk_freq=cfg.index_topk_freq,
                index_skip_topk_offset=cfg.index_skip_topk_offset,
                indexer_rope_interleave=cfg.indexer_rope_interleave,
                indexer_types=cfg.indexer_types,
            ),
            hf_head_dim=cfg.head_dim,
            qk_head_dim=cfg.qk_head_dim,
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
            balancing_loss_cfg=None,
            mlp_layer_types=cfg.mlp_layer_types,
            index_topk=cfg.index_topk,
            index_head_dim=cfg.index_head_dim,
            index_n_heads=cfg.index_n_heads,
            index_topk_freq=cfg.index_topk_freq,
            index_skip_topk_offset=cfg.index_skip_topk_offset,
            index_share_for_mtp_iteration=cfg.index_share_for_mtp_iteration,
            indexer_rope_interleave=cfg.indexer_rope_interleave,
            indexer_types=cfg.indexer_types,
            num_nextn_predict_layers=getattr(cfg, "num_nextn_predict_layers", None),
            mtp_config=MTPConfig(num_layers=cfg.num_nextn_predict_layers)
            if getattr(cfg, "num_nextn_predict_layers", 0)
            else None,
        )

    @property
    def hf_config(self) -> HFGlmMoeDsaConfig:
        """HuggingFace configuration."""
        assert isinstance(self.router, NoAuxRouterConfig), "Only support saving NoAuxRouter to HF GLM-5.2 format."
        return HFGlmMoeDsaConfig(
            architectures=["GlmMoeDsaForCausalLM"],
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
            hidden_act=self.hidden_act,
            num_attention_heads=self.attention.num_attention_heads,
            num_key_value_heads=self.attention.num_attention_heads,
            head_dim=self.hf_head_dim,
            kv_lora_rank=self.attention.kv_lora_rank,
            q_lora_rank=self.attention.q_lora_rank,
            qk_nope_head_dim=self.attention.qk_nope_head_dim,
            qk_rope_head_dim=self.attention.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.attention.v_head_dim,
            attention_bias=self.attention.qkv_bias or self.attention.o_bias,
            attention_dropout=self.attention.dropout,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.router.n_group,
            topk_group=self.router.topk_group,
            scoring_func=self.router.scoring_func,
            norm_topk_prob=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            tie_word_embeddings=self.tie_word_embeddings,
            index_topk=self.index_topk,
            index_head_dim=self.index_head_dim,
            index_n_heads=self.index_n_heads,
            index_topk_freq=self.index_topk_freq,
            index_skip_topk_offset=self.index_skip_topk_offset,
            index_share_for_mtp_iteration=self.index_share_for_mtp_iteration,
            indexer_rope_interleave=self.indexer_rope_interleave,
            indexer_types=self.indexer_types,
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            dtype=torch.bfloat16,
        )
