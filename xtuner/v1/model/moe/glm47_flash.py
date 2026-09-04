# Copyright (c) OpenMMLab. All rights reserved.
import re
from pathlib import Path
from typing import Literal, cast

import torch
from pydantic import Field, computed_field
from torch.distributed.fsdp import CPUOffloadPolicy, register_fsdp_forward_method
from typing_extensions import Self, override


try:
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig as HFGlm4MoeLiteConfig
except ImportError:
    HFGlm4MoeLiteConfig = None  # type: ignore[misc, assignment]
from xtuner.v1.model.moe.moe import BalancingLossConfig, MoEConfig, ZLossConfig
from xtuner.v1.module.attention import DSAMLAConfig, DSAMultiLatentAttention, MLAConfig
from xtuner.v1.module.attention.dsa_topk_sharing import (
    build_dsa_topk_release_plan,
    configure_dsa_topk_decoder_lifecycle,
)
from xtuner.v1.module.mtp import MTPConfig, MTPLayer
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils import default_init_weights

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
            expert_prefix = key.removesuffix(".fused_w1w3.weight")
            return [
                f"{expert_prefix}.{expert_idx}.{projection}_proj.weight"
                for expert_idx in range(self.config.n_routed_experts)
                for projection in ("gate", "up")
            ]
        if "fused_w2.weight" in key:
            expert_prefix = key.removesuffix(".fused_w2.weight")
            return [
                f"{expert_prefix}.{expert_idx}.down_proj.weight"
                for expert_idx in range(self.config.n_routed_experts)
            ]
        if key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        if "router.e_score_correction_bias" in key:
            return [key.replace("router.e_score_correction_bias", "e_score_correction_bias")]
        return [key]

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


class Glm47FlashDSA(Glm47Flash):
    """GLM-4.7 Flash with trainable main-stack and physical-MTP indexers."""

    def _dsa_layers(self) -> list[tuple[torch.nn.Module, DSAMultiLatentAttention]]:
        layers: list[tuple[torch.nn.Module, DSAMultiLatentAttention]] = []
        for decoder_layer in self.layers.values():
            self_attn = decoder_layer.self_attn  # type: ignore[attr-defined]
            if not isinstance(self_attn, DSAMultiLatentAttention):
                raise TypeError(f"GLM-4.7 DSA requires DSAMultiLatentAttention, got {type(self_attn).__name__}.")
            layers.append((decoder_layer, self_attn))

        if self.mtp_block is not None and self.config.mtp_config is not None:
            num_physical = 1 if self.config.mtp_config.share_weights else self.config.mtp_config.num_layers
            for mtp_idx in range(num_physical):
                mtp_layer = self.mtp_block.layers[mtp_idx]
                if not isinstance(mtp_layer, MTPLayer):
                    raise TypeError(f"Expected MTPLayer, got {type(mtp_layer).__name__}.")
                decoder_layer = mtp_layer.decoder_layer
                self_attn = decoder_layer.self_attn  # type: ignore[attr-defined]
                if not isinstance(self_attn, DSAMultiLatentAttention):
                    raise TypeError(
                        f"GLM-4.7 DSA MTP requires DSAMultiLatentAttention, got {type(self_attn).__name__}."
                    )
                if self_attn.indexer_training is not None and not self_attn.indexer_training.train_mtp_indexer:
                    self_attn.disable_indexer_training()
                layers.append((decoder_layer, self_attn))
        return layers

    @override
    def _configure_model_specific_layer_lifecycle(self) -> None:
        dsa_layers = self._dsa_layers()
        sample_attention = dsa_layers[0][1]
        if sample_attention.indexer_training is not None and sample_attention.indexer_training.indexer_only:
            self.requires_grad_(False)
            for _, self_attn in dsa_layers:
                if self_attn.indexer_training is not None and self_attn.source_layer_idx == self_attn.layer_idx:
                    self_attn.indexer.requires_grad_(True)

        num_physical_mtp = len(dsa_layers) - self.config.num_hidden_layers
        release_plan = build_dsa_topk_release_plan(
            num_main_layers=self.config.num_hidden_layers,
            num_mtp_layers=num_physical_mtp,
            indexer_types=sample_attention.indexer_types,
            index_skip_topk_offset=sample_attention.index_skip_topk_offset,
            index_topk_freq=sample_attention.index_topk_freq,
        )
        for decoder_layer, self_attn in dsa_layers:
            configure_dsa_topk_decoder_lifecycle(
                decoder_layer=decoder_layer,
                attention=self_attn,
                release_plan=release_plan,
            )

    @override
    def _fully_shard_model_specific_submodules(self) -> None:
        """Make side-channel indexer loss visible to composable FSDP."""

        assert self.fsdp_config is not None
        assert self.fsdp_mesh is not None
        mesh = self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh
        offload_policy = CPUOffloadPolicy() if self.fsdp_config.cpu_offload else None
        for _, self_attn in self._dsa_layers():
            if self_attn.source_layer_idx != self_attn.layer_idx or not hasattr(self_attn, "indexer"):
                continue
            indexer = self_attn.indexer
            if not any(parameter.requires_grad for parameter in indexer.parameters()):
                continue
            self._fully_shard(
                mesh=mesh,
                mp_policy=self.mp_policy,
                reshard_after_forward=self.fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=indexer,
            )
            register_fsdp_forward_method(indexer, "project_features")

    @override
    @torch.no_grad()
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        """Load official dense weights and initialize only absent indexers."""

        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict=False)
        indexers: list[tuple[str, DSAMultiLatentAttention]] = []
        indexer_names: set[str] = set()
        for layer_idx, decoder_layer in self.layers.items():
            indexers.append((f"layers.{layer_idx}.self_attn.indexer", decoder_layer.self_attn))  # type: ignore[attr-defined]
        if self.mtp_block is not None and self.config.mtp_config is not None:
            num_physical = 1 if self.config.mtp_config.share_weights else self.config.mtp_config.num_layers
            for mtp_idx in range(num_physical):
                attention = self.mtp_block.layers[mtp_idx].decoder_layer.self_attn
                indexers.append((f"mtp_block.layers.{mtp_idx}.decoder_layer.self_attn.indexer", attention))

        for prefix, attention in indexers:
            self_attn = cast(DSAMultiLatentAttention, attention)
            if self_attn.source_layer_idx != self_attn.layer_idx:
                continue
            current_indexer_names = {
                self._clean_param_name(name) for name, _ in self_attn.indexer.named_parameters(prefix=prefix)
            }
            indexer_names.update(current_indexer_names)
            loaded_indexer_names = loaded_keys & current_indexer_names
            if loaded_indexer_names and loaded_indexer_names != current_indexer_names:
                raise RuntimeError(
                    f"Partially loaded DSA indexer {prefix}: {sorted(current_indexer_names - loaded_indexer_names)}"
                )
            if not loaded_indexer_names:
                default_init_weights(self_attn.indexer)

        if unloaded_base_keys := unloaded_keys - indexer_names:
            raise RuntimeError(f"Failed to load GLM-4.7 base weights: {sorted(unloaded_base_keys)}")
        return loaded_keys, unloaded_keys, missing_keys


class Glm47FlashDSAConfig(Glm47FlashConfig):
    """Two-stage DSA conversion config for official GLM-4.7 Flash."""

    model_type: str = "glm4_moe_lite_dsa"
    attention: DSAMLAConfig = DSAMLAConfig(
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
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_topk_freq=4,
        index_skip_topk_offset=1,
        indexer_rope_interleave=True,
    )

    def _normalize_indexer_types(self) -> None:
        num_physical_mtp = 0
        if self.mtp_config is not None:
            num_physical_mtp = 1 if self.mtp_config.share_weights else self.mtp_config.num_layers
        expected = [
            "full" if layer_idx % self.attention.index_topk_freq == 0 else "shared"
            for layer_idx in range(self.num_hidden_layers)
        ]
        expected.extend(["full"] * num_physical_mtp)
        if self.attention.indexer_types is None:
            self.attention.indexer_types = expected
        elif self.attention.indexer_types != expected:
            raise ValueError(f"GLM-4.7 DSA expects indexer_types={expected}, got {self.attention.indexer_types}.")

    def build(self) -> Glm47FlashDSA:
        self._normalize_indexer_types()
        return Glm47FlashDSA(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        dense_config = Glm47FlashConfig.from_hf(hf_path)
        dense_attention = dense_config.attention
        config_values = {
            name: getattr(dense_config, name)
            for name in Glm47FlashConfig.model_fields
            if name not in ("attention", "model_type")
        }
        config = cls(
            **config_values,
            attention=DSAMLAConfig(
                **dense_attention.model_dump(),
                index_topk=2048,
                index_head_dim=128,
                index_n_heads=32,
                index_topk_freq=4,
                index_skip_topk_offset=1,
                indexer_rope_interleave=dense_attention.rope_interleave,
            ),
        )
        # The official checkpoint has one physical MTP layer and one training
        # prediction depth. No training-time weight-sharing loop is needed.
        if config.mtp_config is not None:
            config.mtp_config = config.mtp_config.model_copy(update={"share_weights": False})
        config._normalize_indexer_types()
        return config
