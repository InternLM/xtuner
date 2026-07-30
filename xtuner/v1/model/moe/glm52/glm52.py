import os
import re
from pathlib import Path
from typing import Callable, Literal, cast

import torch
from pydantic import Field, computed_field
from typing_extensions import Self, override

from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as HFGlmMoeDsaConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.base import DEFAULT_FLOAT8_CFG, TorchCompileOption
from xtuner.v1.model.moe.moe import (
    MOE_RECOMPUTE_CFG,
    BalancingLossConfig,
    MoE,
    MoEConfig,
    ZLossConfig,
)
from xtuner.v1.module.decoder_layer.dense_decoder_layer import (
    DenseDecoderLayerMicroBatchOutput,
    DenseDecoderLayerOutput,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils import MarkerInterval, RecomputeIntervalMap, RecomputeUnit

from .decoder_layer import (
    GLM52DenseDecoderLayer,
    GLM52DenseDecoderLayerMicroBatchOutput,
    GLM52DenseDecoderLayerOutput,
    GLM52MoEDecoderLayer,
    GLM52MoEDecoderLayerMicroBatchOutput,
    GLM52MoEDecoderLayerOutput,
)
from .dsa_mla import DSAMLAConfig, DSAMultiLatentAttention
from .dsa_topk_sharing import dsa_topk_source_layer, dsa_topk_source_layers
from .mtp import GLM52MTPBlock, GLM52MTPLayer


# Keep the existing graph boundaries while explicit DSA top-k tensor inputs and
# results are validated. Each boundary can be tightened independently later.
MOE_NON_EP_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.model.moe.glm52.decoder_layer.GLM52MoEDecoderLayer.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward": TorchCompileOption(
        fullgraph=False
    ),
    "xtuner.v1.model.moe.glm52.dsa_mla.DSAMultiLatentAttention.forward": TorchCompileOption(fullgraph=False),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._post_moe_forward": TorchCompileOption(
        fullgraph=True
    ),
    "xtuner.v1.model.moe.glm52.decoder_layer.GLM52DenseDecoderLayer.forward": TorchCompileOption(fullgraph=False),
    **DEFAULT_FLOAT8_CFG,
}

MOE_EP_COMPILE_CFG = MOE_NON_EP_COMPILE_CFG.copy()
MOE_EP_COMPILE_CFG.pop("xtuner.v1.model.moe.glm52.decoder_layer.GLM52MoEDecoderLayer.forward")

# GLM's native RMSNorm mutates an intermediate inside the generic attention
# interval, so retain the other MoE units but replace that interval with the
# no-grad, mutation-safe DSA top-k selection kernel.
GLM52_RECOMPUTE_CFG: RecomputeIntervalMap = {
    unit: intervals for unit, intervals in MOE_RECOMPUTE_CFG.items() if unit is not RecomputeUnit.SAVE_ATTN
}
DSA_INDEXER_INTERVAL: MarkerInterval = ("dsa.indexer.begin", "dsa.indexer.end")
GLM52_RECOMPUTE_CFG[RecomputeUnit.SAVE_DSA_INDEXER] = [DSA_INDEXER_INTERVAL]


class Glm52MoE(MoE):
    dense_decoder_layer_cls = GLM52DenseDecoderLayer
    moe_decoder_layer_cls = GLM52MoEDecoderLayer
    mtp_layer_cls = GLM52MTPLayer
    mtp_block_cls = GLM52MTPBlock

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.ep_size > 1:
            return MOE_EP_COMPILE_CFG
        return MOE_NON_EP_COMPILE_CFG

    @property
    @override
    def default_recompute_cfg(self) -> RecomputeIntervalMap:
        return GLM52_RECOMPUTE_CFG

    @property
    @override
    def mtp_recompute_intervals(self) -> list[MarkerInterval]:
        return [DSA_INDEXER_INTERVAL] if DSA_INDEXER_INTERVAL in self.recompute_intervals else []

    @override
    def _compiles_whole_decoder_layer(self) -> bool:
        # GLM keeps decoder forward non-fullgraph so selected eager islands,
        # including the DSA top-k kernel, remain addressable to SAC.
        return False

    @override
    def _configure_model_specific_layers(self) -> None:
        dsa_layers: list[DSAMultiLatentAttention] = []
        for decoder_layer in self.layers.values():
            self_attn = decoder_layer.self_attn  # type: ignore[attr-defined]
            assert isinstance(self_attn, DSAMultiLatentAttention), (
                f"GLM-5.2 requires DSAMultiLatentAttention, got {type(self_attn).__name__}."
            )
            dsa_layers.append(self_attn)

        if self.mtp_block is not None and self.config.mtp_config is not None:
            num_physical_mtp_layers = 1 if self.config.mtp_config.share_weights else self.config.mtp_config.num_layers
            for mtp_idx in range(num_physical_mtp_layers):
                mtp_layer = self.mtp_block.layers[mtp_idx]
                assert isinstance(mtp_layer, GLM52MTPLayer)
                self_attn = mtp_layer.decoder_layer.self_attn  # type: ignore[attr-defined]
                assert isinstance(self_attn, DSAMultiLatentAttention), (
                    f"GLM-5.2 MTP requires DSAMultiLatentAttention, got {type(self_attn).__name__}."
                )
                dsa_layers.append(self_attn)

        keep_topk = DSA_INDEXER_INTERVAL in self.recompute_intervals
        for self_attn in dsa_layers:
            if hasattr(self_attn, "indexer"):
                self_attn.indexer.selective_checkpoint_topk = keep_topk

        sample_attn = dsa_layers[0]
        self._dsa_topk_source_layers = dsa_topk_source_layers(
            num_layers=self.config.num_hidden_layers,
            indexer_types=sample_attn.indexer_types,
            index_skip_topk_offset=sample_attn.index_skip_topk_offset,
            index_topk_freq=sample_attn.index_topk_freq,
        )
        self._dsa_topk_last_consumers = frozenset(
            layer_idx
            for layer_idx, source_layer_idx in enumerate(self._dsa_topk_source_layers)
            if layer_idx == self.config.num_hidden_layers - 1
            or self._dsa_topk_source_layers[layer_idx + 1] != source_layer_idx
        )

    @override
    def _call_decoder_layer(
        self,
        *,
        decoder_layer: torch.nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        previous_layer_results: (
            DenseDecoderLayerOutput
            | DenseDecoderLayerMicroBatchOutput
            | MoEDecoderLayerOutput
            | MoEDecoderLayerMicroBatchOutput
            | None
        ),
    ) -> (
        DenseDecoderLayerOutput
        | DenseDecoderLayerMicroBatchOutput
        | MoEDecoderLayerOutput
        | MoEDecoderLayerMicroBatchOutput
    ):
        """Arrange GLM-5.2 DSA dataflow and offload around one decoder
        layer."""
        is_micro_batch = isinstance(hidden_states, list)
        is_source_layer = self._dsa_topk_source_layers[layer_idx] == layer_idx
        if is_source_layer:
            dsa_topk_ids: torch.Tensor | list[torch.Tensor] | None = None
        else:
            previous_results = cast(
                GLM52DenseDecoderLayerOutput
                | GLM52DenseDecoderLayerMicroBatchOutput
                | GLM52MoEDecoderLayerOutput
                | GLM52MoEDecoderLayerMicroBatchOutput,
                previous_layer_results,
            )
            dsa_topk_ids = previous_results["dsa_topk_ids"]

        activation_offload = int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1
        dsa_topk_offload = int(os.getenv("XTUNER_DSA_TOPK_OFFLOAD", "0")) == 1
        offload_tensors: list[torch.Tensor] = []
        if activation_offload and layer_idx >= self.config.first_k_dense_replace:
            offload_tensors = list(hidden_states) if is_micro_batch else [cast(torch.Tensor, hidden_states)]
        if dsa_topk_offload and layer_idx in self._dsa_topk_last_consumers and dsa_topk_ids is not None:
            offload_tensors.extend(dsa_topk_ids if isinstance(dsa_topk_ids, list) else [dsa_topk_ids])

        # The offload context expects a dense zero-based block index across every
        # active activation or DSA-ID window, including dense GLM layers.
        offload_block_idx = sum(
            (activation_offload and previous_idx >= self.config.first_k_dense_replace)
            or (
                dsa_topk_offload
                and previous_idx in self._dsa_topk_last_consumers
                and self._dsa_topk_source_layers[previous_idx] != previous_idx
            )
            for previous_idx in (int(idx) for idx in self.layers)
            if previous_idx < layer_idx
        )
        decoder_forward = cast(
            Callable[
                ...,
                DenseDecoderLayerOutput
                | DenseDecoderLayerMicroBatchOutput
                | MoEDecoderLayerOutput
                | MoEDecoderLayerMicroBatchOutput,
            ],
            decoder_layer,
        )
        with self._saved_tensors_offload_ctx(offload_block_idx, offload_tensors):
            layer_results = decoder_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=dsa_topk_ids,
            )
        if layer_idx < self.config.first_k_dense_replace:
            if is_micro_batch:
                return cast(GLM52DenseDecoderLayerMicroBatchOutput, layer_results)
            return cast(GLM52DenseDecoderLayerOutput, layer_results)
        if is_micro_batch:
            return cast(GLM52MoEDecoderLayerMicroBatchOutput, layer_results)
        return cast(GLM52MoEDecoderLayerOutput, layer_results)

    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if key.startswith("mtp_block."):
            match = re.match(r"mtp_block\.layers\.(\d+)\.(.+)", key)
            assert match is not None, f"Unexpected GLM-5.2 MTP key: {key}"
            mtp_layer_idx = self.config.num_hidden_layers + int(match.group(1))
            key = f"layers.{mtp_layer_idx}.{match.group(2)}"
            # GLM HF stores the MTP decoder as the next layer after the main stack.
            # Only the MTP pre/post modules keep special names at that layer.
            key = key.replace(".decoder_layer.", ".")
            key = re.sub(r"layers\.(\d+)\.final_layernorm\.", r"layers.\1.shared_head.norm.", key)

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
    attention: DSAMLAConfig = DSAMLAConfig(
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
    index_share_for_mtp_iteration: bool = True
    num_nextn_predict_layers: int | None = 1
    mtp_config: MTPConfig | None = None

    @computed_field
    def num_key_value_heads(self) -> int:
        return self.attention.num_attention_heads

    def build(self) -> Glm52MoE:
        self._normalize_physical_mtp_indexer_types()
        return Glm52MoE(self)

    def _normalize_physical_mtp_indexer_types(self) -> None:
        if self.mtp_config is None:
            return

        indexer_types = self.attention.indexer_types
        if indexer_types is None:
            indexer_types = [
                "full"
                if dsa_topk_source_layer(
                    layer_idx=layer_idx,
                    indexer_types=None,
                    index_skip_topk_offset=self.attention.index_skip_topk_offset,
                    index_topk_freq=self.attention.index_topk_freq,
                )
                == layer_idx
                else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]
        else:
            indexer_types = list(indexer_types)

        num_physical_mtp_layers = 1 if self.mtp_config.share_weights else self.mtp_config.num_layers
        expected_mtp_types = ["full"] + ["shared"] * (num_physical_mtp_layers - 1)
        if len(indexer_types) == self.num_hidden_layers:
            indexer_types.extend(expected_mtp_types)
        elif (
            len(indexer_types) != self.num_hidden_layers + num_physical_mtp_layers
            or indexer_types[self.num_hidden_layers :] != expected_mtp_types
        ):
            raise ValueError(
                "GLM-5.2 physical MTP indexer_types must start with 'full' and share that indexer in any "
                f"remaining physical MTP layers; got {indexer_types[self.num_hidden_layers :]}."
            )

        # HF lists only the main stack. XTuner also builds the checkpoint-backed
        # physical MTP decoder, so attention construction needs its effective type.
        self.attention.indexer_types = indexer_types

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        cfg = HFGlmMoeDsaConfig.from_pretrained(hf_path)

        assert isinstance(cfg, HFGlmMoeDsaConfig)

        rope_parameters_cfg = RopeParametersConfig.from_hf_config(cfg)
        if getattr(cfg, "num_nextn_predict_layers", 0) and not cfg.index_share_for_mtp_iteration:
            raise ValueError("GLM-5.2 MTP requires index_share_for_mtp_iteration=True.")

        config = cls(
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
                indexer_types=list(cfg.indexer_types) if cfg.indexer_types is not None else None,
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
            index_share_for_mtp_iteration=cfg.index_share_for_mtp_iteration,
            num_nextn_predict_layers=getattr(cfg, "num_nextn_predict_layers", None),
            mtp_config=MTPConfig(
                num_layers=cfg.num_nextn_predict_layers,
                share_weights=True,
            )
            if getattr(cfg, "num_nextn_predict_layers", 0)
            else None,
        )
        config._normalize_physical_mtp_indexer_types()
        return config

    @property
    def hf_config(self) -> HFGlmMoeDsaConfig:
        """HuggingFace configuration."""
        assert isinstance(self.router, NoAuxRouterConfig), "Only support saving NoAuxRouter to HF GLM-5.2 format."
        attention = self.attention
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
            num_attention_heads=attention.num_attention_heads,
            num_key_value_heads=attention.num_attention_heads,
            head_dim=self.hf_head_dim,
            kv_lora_rank=attention.kv_lora_rank,
            q_lora_rank=attention.q_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=attention.v_head_dim,
            attention_bias=attention.qkv_bias or attention.o_bias,
            attention_dropout=attention.dropout,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.router.n_group,
            topk_group=self.router.topk_group,
            scoring_func=self.router.scoring_func,
            norm_topk_prob=self.router.norm_topk_prob,
            routed_scaling_factor=self.router.router_scaling_factor,
            tie_word_embeddings=self.tie_word_embeddings,
            index_topk=attention.index_topk,
            index_head_dim=attention.index_head_dim,
            index_n_heads=attention.index_n_heads,
            index_topk_freq=attention.index_topk_freq,
            index_skip_topk_offset=attention.index_skip_topk_offset,
            index_share_for_mtp_iteration=self.index_share_for_mtp_iteration,
            indexer_rope_interleave=attention.indexer_rope_interleave,
            indexer_types=(
                attention.indexer_types[: self.num_hidden_layers] if attention.indexer_types is not None else None
            ),
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            dtype=torch.bfloat16,
        )
