from typing import Literal, TypedDict, cast

import torch
import torch.nn as nn
from typing_extensions import NotRequired

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
from xtuner.v1.utils import ForwardState, checkpoint_record

from ..linear import build_linear


class DenseDecoderLayerOutput(TypedDict):
    """Per-micro-batch outputs of one :class:`DenseDecoderLayer` forward.

    A dense layer only produces hidden states, but it reports them through the same keyed contract
    as :class:`~xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer` so that the two
    layer families stay interchangeable to their callers. DSA layers additionally return the
    source layer's explicit top-k IDs.
    """

    hidden_states: torch.Tensor
    dsa_topk_ids: NotRequired[torch.Tensor]


class DenseDecoderLayerMicroBatchOutput(TypedDict):
    """Outputs of one :class:`DenseDecoderLayer` forward over several micro-
    batches.

    Each field holds one entry per micro-batch, in input order.
    """

    hidden_states: list[torch.Tensor]
    dsa_topk_ids: NotRequired[list[torch.Tensor]]


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
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        dsa_topk_ids: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> DenseDecoderLayerOutput | DenseDecoderLayerMicroBatchOutput:
        """Run equal-shaped training micro-batches in one layer invocation.

        Keeping the micro-batch loop inside the decoder layer lets outer FSDP and checkpointing
        materialize the layer only once, while each attention call keeps its own
        ``SequenceContext``.

        Args:
            hidden_states (torch.Tensor | list[torch.Tensor]): Input hidden states, one tensor per
                micro-batch.
            position_embeddings (tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]):
                Rotary position embeddings ``(cos, sin)``, aligned with ``hidden_states``.
            seq_ctx (SequenceContext | list[SequenceContext]): Sequence context, aligned with
                ``hidden_states``.
            dsa_topk_ids (torch.Tensor | list[torch.Tensor] | None): Explicit source-layer DSA
                top-k IDs, aligned with ``hidden_states``.

        Returns:
            DenseDecoderLayerOutput | DenseDecoderLayerMicroBatchOutput: Output hidden states. A
            single tensor for a single ``hidden_states`` tensor, a per-micro-batch list for a list
            of them. DSA layers additionally return explicit ``dsa_topk_ids``.
        """
        if not isinstance(hidden_states, list):
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
            assert isinstance(seq_ctx, SequenceContext)
            assert dsa_topk_ids is None or isinstance(dsa_topk_ids, torch.Tensor)
            return self._forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=dsa_topk_ids,
            )

        n = len(hidden_states)
        assert isinstance(position_embeddings, list) and len(position_embeddings) == n
        assert isinstance(seq_ctx, list) and len(seq_ctx) == n
        assert all(hidden.shape == hidden_states[0].shape for hidden in hidden_states)
        if dsa_topk_ids is None:
            dsa_topk_ids_list: list[torch.Tensor | None] = [None] * n
        else:
            assert isinstance(dsa_topk_ids, list) and len(dsa_topk_ids) == n
            dsa_topk_ids_list = list(dsa_topk_ids)

        layer_results = [
            self._forward(
                hidden_states=hidden,
                position_embeddings=position_embedding,
                seq_ctx=context,
                dsa_topk_ids=topk_ids,
            )
            for hidden, topk_ids, position_embedding, context in zip(
                hidden_states, dsa_topk_ids_list, position_embeddings, seq_ctx
            )
        ]
        output: DenseDecoderLayerMicroBatchOutput = {
            "hidden_states": [result["hidden_states"] for result in layer_results]
        }
        output_ids = [result.get("dsa_topk_ids") for result in layer_results]
        if any(topk_ids is not None for topk_ids in output_ids):
            assert all(topk_ids is not None for topk_ids in output_ids)
            output["dsa_topk_ids"] = [cast(torch.Tensor, topk_ids) for topk_ids in output_ids]
        return output

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        dsa_topk_ids: torch.Tensor | None,
    ) -> DenseDecoderLayerOutput:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        checkpoint_record("attn.begin")
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
        checkpoint_record("attn.end")
        hidden_states = residual + attn_outputs["projected_output"]

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        checkpoint_record("mlp.begin")
        hidden_states = self.mlp(hidden_states)
        checkpoint_record("mlp.end")
        hidden_states = residual + hidden_states

        output: DenseDecoderLayerOutput = {"hidden_states": hidden_states}
        if (output_ids := attn_outputs.get("dsa_topk_ids")) is not None:
            output["dsa_topk_ids"] = output_ids
        return output

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
