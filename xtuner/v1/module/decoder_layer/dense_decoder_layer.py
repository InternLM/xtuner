from typing import Literal, cast

import torch
import torch.nn as nn

from xtuner.v1.config import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module import (
    AttnOutputs,
    DSAMultiLatentAttention,
    GatedDeltaNetConfig,
    MHAConfig,
    MLAConfig,
    RMSNorm,
)
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import ForwardState

from ..linear import build_linear


class DenseMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        hidden_act: str,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.gate_proj = build_linear(hidden_size, intermediate_size, bias=bias, float8_cfg=float8_cfg)
        self.up_proj = build_linear(hidden_size, intermediate_size, bias=bias, float8_cfg=float8_cfg)
        self.down_proj = build_linear(intermediate_size, hidden_size, bias=bias, float8_cfg=float8_cfg)
        self.act_fn = get_act_fn(hidden_act)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DenseDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal["default", "zero_centered"] = "default",
        attention_config: MLAConfig | MHAConfig | GatedDeltaNetConfig,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        generate_config: GenerateConfig | None = None,
        float8_cfg: Float8Config | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = attention_config.build(
            hidden_size=hidden_size,
            layer_type=layer_type,
            layer_idx=layer_idx,
            rope_scaling_cfg=rope_scaling_cfg,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )
        self.mlp = DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=mlp_bias,
            hidden_act=hidden_act,
            float8_cfg=float8_cfg,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

    def forward(
        self,
        *layer_inputs: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run equal-shaped training micro-batches in one layer invocation.

        Keeping the micro-batch loop inside the decoder layer lets outer FSDP
        and checkpoint wrappers materialize the layer only once, while each
        attention call keeps its own ``SequenceContext``. DSA callers append
        explicit IDs after the hidden-state inputs; DSA results use the flat
        ``(hidden..., dsa_topk_ids...)`` layout required by reentrant checkpoint.
        """
        if isinstance(seq_ctx, SequenceContext):
            assert len(layer_inputs) in (1, 2), (
                "Single-microbatch DenseDecoderLayer expects hidden_states and optional dsa_topk_ids."
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
            return self._forward(
                hidden_states=layer_inputs[0],
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=layer_inputs[1] if len(layer_inputs) == 2 else None,
            )

        n = len(seq_ctx)
        assert len(layer_inputs) in (n, 2 * n), (
            f"Multi-microbatch DenseDecoderLayer expects {n} hidden states and optional {n} dsa_topk_ids."
        )
        hidden_states = layer_inputs[:n]
        dsa_topk_ids: tuple[torch.Tensor | None, ...]
        if len(layer_inputs) == n:
            dsa_topk_ids = (None,) * n
        else:
            dsa_topk_ids = layer_inputs[n:]

        assert isinstance(position_embeddings, list) and len(position_embeddings) == n
        assert all(hidden.shape == hidden_states[0].shape for hidden in hidden_states)
        layer_results = tuple(
            self._forward(
                hidden_states=hidden,
                position_embeddings=position_embedding,
                seq_ctx=context,
                dsa_topk_ids=topk_ids,
            )
            for hidden, topk_ids, position_embedding, context in zip(
                hidden_states, dsa_topk_ids, position_embeddings, seq_ctx
            )
        )
        if isinstance(layer_results[0], tuple):
            return tuple(result[0] for result in layer_results) + tuple(result[1] for result in layer_results)  # type: ignore[index]
        return layer_results  # type: ignore[return-value]

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        dsa_topk_ids: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if dsa_topk_ids is None:
            attn_outputs: AttnOutputs = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
        else:
            dsa_attn = cast(DSAMultiLatentAttention, self.self_attn)
            attn_outputs = dsa_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=dsa_topk_ids,
            )
        dsa_topk_ids = attn_outputs.get("dsa_topk_ids")
        hidden_states = attn_outputs["projected_output"]
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if dsa_topk_ids is None:
            return hidden_states
        return hidden_states, dsa_topk_ids

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn.prefilling(  # type: ignore
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn.decoding(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            state=ForwardState.DECODING,  # type: ignore   # TODO: Fix outdated interface
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(  # type: ignore
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )
