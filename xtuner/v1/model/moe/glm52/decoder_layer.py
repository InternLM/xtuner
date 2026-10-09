from typing import cast

import torch
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import AttnOutputs, RouterResults
from xtuner.v1.module.decoder_layer.dense_decoder_layer import (
    DenseDecoderLayer,
    DenseDecoderLayerMicroBatchOutput,
    DenseDecoderLayerOutput,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayer,
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)

from .dsa_mla import DSAMultiLatentAttention, GLM52AttnOutputs


class GLM52DenseDecoderLayerOutput(DenseDecoderLayerOutput):
    """GLM-5.2 dense-layer output with explicit DSA IDs."""

    dsa_topk_ids: torch.Tensor


class GLM52DenseDecoderLayerMicroBatchOutput(DenseDecoderLayerMicroBatchOutput):
    """GLM-5.2 dense-layer outputs for intra-layer micro-batches."""

    dsa_topk_ids: list[torch.Tensor]


class GLM52DenseDecoderLayer(DenseDecoderLayer):
    """Dense decoder layer that threads GLM-5.2 DSA IDs explicitly."""

    @override
    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        dsa_topk_ids: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> GLM52DenseDecoderLayerOutput | GLM52DenseDecoderLayerMicroBatchOutput:
        if not isinstance(hidden_states, list):
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
            assert isinstance(seq_ctx, SequenceContext)
            assert dsa_topk_ids is None or isinstance(dsa_topk_ids, torch.Tensor)
            return self._glm52_forward(
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
            self._glm52_forward(
                hidden_states=hidden,
                position_embeddings=position_embedding,
                seq_ctx=context,
                dsa_topk_ids=topk_ids,
            )
            for hidden, topk_ids, position_embedding, context in zip(
                hidden_states, dsa_topk_ids_list, position_embeddings, seq_ctx
            )
        ]
        return {
            "hidden_states": [result["hidden_states"] for result in layer_results],
            "dsa_topk_ids": [result["dsa_topk_ids"] for result in layer_results],
        }

    def _glm52_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        dsa_topk_ids: torch.Tensor | None,
    ) -> GLM52DenseDecoderLayerOutput:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention = cast(DSAMultiLatentAttention, self.self_attn)
        attn_outputs = attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            dsa_topk_ids=dsa_topk_ids,
        )
        hidden_states = residual + attn_outputs["projected_output"]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return {
            "hidden_states": hidden_states,
            "dsa_topk_ids": attn_outputs["dsa_topk_ids"],
        }


class GLM52MoEDecoderLayerOutput(MoEDecoderLayerOutput):
    """GLM-5.2 MoE-layer output with explicit DSA IDs."""

    dsa_topk_ids: torch.Tensor


class GLM52MoEDecoderLayerMicroBatchOutput(MoEDecoderLayerMicroBatchOutput):
    """GLM-5.2 MoE-layer outputs for intra-layer micro-batches."""

    dsa_topk_ids: list[torch.Tensor]


class GLM52MoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder layer that threads GLM-5.2 DSA IDs explicitly."""

    @override
    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        dsa_topk_ids: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> GLM52MoEDecoderLayerOutput | GLM52MoEDecoderLayerMicroBatchOutput:
        if not isinstance(hidden_states, list):
            assert isinstance(seq_ctx, SequenceContext)
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
            assert dsa_topk_ids is None or isinstance(dsa_topk_ids, torch.Tensor)
            return cast(
                GLM52MoEDecoderLayerOutput,
                self._forward(
                    hidden_states=hidden_states,
                    seq_ctx=seq_ctx,
                    position_embeddings=position_embeddings,
                    attention_kwargs={"dsa_topk_ids": dsa_topk_ids},
                ),
            )

        n = len(hidden_states)
        assert isinstance(seq_ctx, list) and len(seq_ctx) == n
        assert isinstance(position_embeddings, list) and len(position_embeddings) == n
        if dsa_topk_ids is None:
            dsa_topk_ids_list: list[torch.Tensor | None] = [None] * n
        else:
            assert isinstance(dsa_topk_ids, list) and len(dsa_topk_ids) == n
            dsa_topk_ids_list = list(dsa_topk_ids)

        return cast(
            GLM52MoEDecoderLayerMicroBatchOutput,
            self._micro_batch_forward(
                hidden_states_list=hidden_states,
                seq_ctx_list=seq_ctx,
                position_embeddings_list=position_embeddings,
                attention_kwargs_list=[{"dsa_topk_ids": topk_ids} for topk_ids in dsa_topk_ids_list],
            ),
        )

    @override
    def _build_output(
        self,
        *,
        hidden_states: torch.Tensor,
        router_results: RouterResults,
        attn_outputs: AttnOutputs,
    ) -> GLM52MoEDecoderLayerOutput:
        glm_attn_outputs = cast(GLM52AttnOutputs, attn_outputs)
        return {
            "hidden_states": hidden_states,
            "router_logits": router_results["logits"],
            "router_weights": router_results["router_weights"],
            "router_topk_ids": router_results["topk_ids"],
            "dsa_topk_ids": glm_attn_outputs["dsa_topk_ids"],
        }

    @override
    def _build_micro_batch_output(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        router_results_list: list[RouterResults],
        attn_outputs_list: list[AttnOutputs],
    ) -> GLM52MoEDecoderLayerMicroBatchOutput:
        glm_attn_outputs = [cast(GLM52AttnOutputs, output) for output in attn_outputs_list]
        return {
            "hidden_states": hidden_states_list,
            "router_logits": [result["logits"] for result in router_results_list],
            "router_weights": [result["router_weights"] for result in router_results_list],
            "router_topk_ids": [result["topk_ids"] for result in router_results_list],
            "dsa_topk_ids": [output["dsa_topk_ids"] for output in glm_attn_outputs],
        }
