from typing import cast

import torch
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.mtp import MTPBlock, MTPLayer
from xtuner.v1.module.mtp.mtp_block import MTPInternalOutput

from .decoder_layer import (
    GLM52MoEDecoderLayerMicroBatchOutput,
    GLM52MoEDecoderLayerOutput,
)


class GLM52MTPLayer(MTPLayer):
    """MTP layer whose wrapped GLM-5.2 decoder consumes explicit DSA IDs."""

    @override
    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        future_embeddings: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        dsa_topk_ids: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> GLM52MoEDecoderLayerOutput | GLM52MoEDecoderLayerMicroBatchOutput:
        if not isinstance(hidden_states, list):
            assert isinstance(future_embeddings, torch.Tensor)
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
            assert isinstance(seq_ctx, SequenceContext)
            assert dsa_topk_ids is None or isinstance(dsa_topk_ids, torch.Tensor)
            projected = self._preprocess(hidden_states=hidden_states, future_embeddings=future_embeddings)
            layer_results = cast(
                GLM52MoEDecoderLayerOutput,
                self.decoder_layer(
                    projected,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                    dsa_topk_ids=dsa_topk_ids,
                ),
            )
            return {
                "hidden_states": self.final_layernorm(layer_results["hidden_states"]),
                "router_logits": layer_results["router_logits"],
                "router_weights": layer_results["router_weights"],
                "router_topk_ids": layer_results["router_topk_ids"],
                "dsa_topk_ids": layer_results["dsa_topk_ids"],
            }

        n = len(hidden_states)
        assert isinstance(future_embeddings, list) and len(future_embeddings) == n
        assert isinstance(position_embeddings, list) and len(position_embeddings) == n
        assert isinstance(seq_ctx, list) and len(seq_ctx) == n
        if dsa_topk_ids is None:
            decoder_topk_ids = None
        else:
            assert isinstance(dsa_topk_ids, list) and len(dsa_topk_ids) == n
            decoder_topk_ids = dsa_topk_ids

        projected_list = [
            self._preprocess(hidden_states=hidden, future_embeddings=future)
            for hidden, future in zip(hidden_states, future_embeddings)
        ]
        micro_batch_results = cast(
            GLM52MoEDecoderLayerMicroBatchOutput,
            self.decoder_layer(
                projected_list,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=decoder_topk_ids,
            ),
        )
        return {
            "hidden_states": [self.final_layernorm(hidden) for hidden in micro_batch_results["hidden_states"]],
            "router_logits": micro_batch_results["router_logits"],
            "router_weights": micro_batch_results["router_weights"],
            "router_topk_ids": micro_batch_results["router_topk_ids"],
            "dsa_topk_ids": micro_batch_results["dsa_topk_ids"],
        }


class GLM52MTPBlock(MTPBlock):
    """MTP block that keeps DSA sharing private to GLM-5.2."""

    @override
    def _call_decoder_layer(
        self,
        layer: MTPLayer,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        previous_layer_results: MTPInternalOutput | None,
        future_embeddings: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> MTPInternalOutput:
        glm_layer = cast(GLM52MTPLayer, layer)
        previous_results = cast(
            GLM52MoEDecoderLayerOutput | GLM52MoEDecoderLayerMicroBatchOutput | None,
            previous_layer_results,
        )
        dsa_topk_ids = None if previous_results is None else previous_results["dsa_topk_ids"]

        return cast(
            MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput,
            glm_layer(
                hidden_states,
                future_embeddings=future_embeddings,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                dsa_topk_ids=dsa_topk_ids,
            ),
        )
