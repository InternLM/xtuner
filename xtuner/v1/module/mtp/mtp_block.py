"""Multi-Token Prediction (MTP) Block implementation."""

from typing import Callable, cast

import torch
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)

from .config import MTPConfig
from .mtp_layer import MTPLayer
from .utils import roll_sequence_context


MTPDepthOutput = MoEDecoderLayerOutput
"""One MTP depth produces the same keyed outputs as the decoder layer it
wraps."""

MTPInternalOutput = MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput


class MTPBlock(nn.Module):
    """Multi-Token Prediction (MTP) block containing multiple MTP layers.

    This block manages D sequential MTP layers, where each layer predicts
    a future token at increasing depths (i+1, i+2, ..., i+D).

    The k-th layer receives:
    - Hidden states from the (k-1)-th layer
    - Embeddings of tokens at position (i+k)

    This forms a sequential prediction chain where deeper layers build upon
    the predictions of shallower layers.

    Args:
        mtp_config (MTPConfig): MTP configuration.
        mtp_layers (list[MTPLayer]): List of MTP layers. Each layer should be a
            fully constructed MTPLayer instance. The number of layers determines
            the prediction depth (D).

    Example:
        >>> # Build MTP layers (typically done by Dense/MoE model)
        >>> mtp_layers = []
        >>> for i in range(2):
        ...     decoder_layer = build_decoder_layer(...)
        ...     mtp_layer = MTPLayer(
        ...         hidden_size=512,
        ...         rms_norm_eps=1e-6,
        ...         rms_norm_type="default",
        ...         decoder_layer=decoder_layer,
        ...     )
        ...     mtp_layers.append(mtp_layer)
        >>>
        >>> # Create MTP block
        >>> mtp_block = MTPBlock(mtp_config=config, mtp_layers=mtp_layers)
        >>>
        >>> # Single-microbatch forward
        >>> outputs = mtp_block(
        ...     h,
        ...     embed_tokens_fn=embed_fn,
        ...     position_embeddings=pos_emb,
        ...     seq_ctx=ctx,
        ... )
        >>> # outputs[0]: predictions for i+1
        >>> # outputs[1]: predictions for i+2
        >>>
        >>> # Multi-microbatch (domino EP) forward
        >>> outputs_per_mb = mtp_block(
        ...     [h0, h1],
        ...     embed_tokens_fn=embed_fn,
        ...     position_embeddings=[pos_emb_0, pos_emb_1],
        ...     seq_ctx=[ctx_0, ctx_1],
        ... )
        >>> # outputs_per_mb[mb_idx][depth_idx] -> MTPDepthOutput
    """

    def __init__(self, *, mtp_config: MTPConfig, mtp_layers: list[MTPLayer]):
        super().__init__()
        if not mtp_layers:
            raise ValueError("mtp_layers cannot be empty")

        if mtp_config.share_weights and len(mtp_layers) != 1:
            raise ValueError(f"share_weights mode requires exactly 1 MTP layer, got {len(mtp_layers)}")
        if not mtp_config.share_weights and len(mtp_layers) != mtp_config.num_layers:
            raise ValueError(f"Expected {mtp_config.num_layers} MTP layers, but got {len(mtp_layers)}")
        self.mtp_config = mtp_config
        self.layers = nn.ModuleList(mtp_layers)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> list[MTPDepthOutput] | list[list[MTPDepthOutput]]:
        """Forward pass through all MTP layers.

        Mirrors :meth:`MoEDecoderLayer.forward`: with a single hidden-state tensor it runs
        the regular per-microbatch path; with ``N`` hidden-state tensors it runs all
        micro-batches together at each MTP depth, so the inner MoE EP dispatch/combine of
        the wrapped decoder layer can be overlapped across micro-batches (domino EP).

        Args:
            hidden_states (torch.Tensor | list[torch.Tensor]): Hidden states from the main model,
                shape ``[batch, seq_len, hidden_size]`` each, one tensor per micro-batch.
            embed_tokens_fn (Callable): Function to embed tokens. Takes token IDs and returns
                embeddings. Should have signature ``embed_tokens_fn(token_ids: Tensor) -> Tensor``.
            position_embeddings (tuple | list[tuple]): Rotary position embeddings (cos, sin),
                aligned per-microbatch with ``hidden_states``.
            seq_ctx (SequenceContext | list[SequenceContext]): Sequence context per micro-batch.

        Returns:
            list[MTPDepthOutput] | list[list[MTPDepthOutput]]: For a single ``hidden_states``
            tensor, one entry per MTP depth ``D``, where ``outputs[k]`` is the prediction for
            token ``i+k+1``. For ``N`` micro-batches, ``outputs[mb_idx][depth_idx]``.
        """
        if not isinstance(hidden_states, list):
            assert isinstance(seq_ctx, SequenceContext), (
                "seq_ctx should be a SequenceContext instance in single-microbatch mode"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a (cos, sin) tuple in single-microbatch mode"
            )
            return self._forward(
                hidden_states=hidden_states,
                embed_tokens_fn=embed_tokens_fn,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )

        n = len(hidden_states)
        assert isinstance(seq_ctx, list) and len(seq_ctx) == n, (
            "seq_ctx should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(position_embeddings, list) and len(position_embeddings) == n, (
            "position_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        return self._micro_batch_forward(
            hidden_states_list=hidden_states,
            embed_tokens_fn=embed_tokens_fn,
            position_embeddings_list=position_embeddings,
            seq_ctx_list=seq_ctx,
        )

    def _forward(
        self,
        *,
        hidden_states: torch.Tensor,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> list[MTPDepthOutput]:
        """Forward pass through all MTP layers.

        Args:
            hidden_states (torch.Tensor): Hidden states from the main model,
                shape [batch, seq_len, hidden_size].
            embed_tokens_fn (Callable): Function to embed tokens. Takes token IDs
                and returns embeddings. Should have signature:
                    embed_tokens_fn(token_ids: Tensor) -> Tensor
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Rotary position
                embeddings (cos, sin).
            seq_ctx (SequenceContext): Sequence context containing input_ids, position_ids,
                attention mask, etc.

        Returns:
            list[MTPDepthOutput]: One entry per MTP depth.
                Length equals num_layers.
                - outputs[0]: Outputs for predicting token at position (i+1)
                - outputs[k]: Outputs for predicting token at position (i+k+1)
        """
        mtp_outputs: list[MTPDepthOutput] = []
        current_hidden_states = hidden_states.detach() if self.mtp_config.detach_mtp_inputs else hidden_states
        current_seq_ctx = seq_ctx
        previous_layer_results: MTPInternalOutput | None = None

        num_steps = self.mtp_config.num_layers
        for step in range(num_steps):
            layer = cast(MTPLayer, self.layers[0] if self.mtp_config.share_weights else self.layers[step])
            # Roll each packed sequence independently so we get the (i+k)-th token while
            # respecting per-sequence boundaries inside the packed batch.
            current_seq_ctx = roll_sequence_context(current_seq_ctx, shifts=-1)
            future_embeddings = self._embed_future(current_seq_ctx, embed_tokens_fn)

            if self.mtp_config.detach_mtp_inputs:
                future_embeddings = future_embeddings.detach()

            layer_results = cast(
                MoEDecoderLayerOutput,
                self._call_decoder_layer(
                    layer,
                    current_hidden_states,
                    previous_layer_results=previous_layer_results,
                    future_embeddings=future_embeddings,
                    position_embeddings=position_embeddings,
                    seq_ctx=current_seq_ctx,
                ),
            )
            previous_layer_results = layer_results
            current_hidden_states = layer_results["hidden_states"]
            mtp_outputs.append(
                {
                    "hidden_states": current_hidden_states,
                    "router_logits": layer_results["router_logits"],
                    "router_weights": layer_results["router_weights"],
                    "router_topk_ids": layer_results["router_topk_ids"],
                }
            )

        return mtp_outputs

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
        """Call one MTP decoder layer.

        Subclasses can consume model-specific fields from ``previous_layer_results`` while the
        generic MTP input and public output stay model-agnostic.
        """
        return layer(
            hidden_states,
            future_embeddings=future_embeddings,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )

    def _micro_batch_forward(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx_list: list[SequenceContext],
    ) -> list[list[MTPDepthOutput]]:
        n = len(hidden_states_list)
        # Per-microbatch outputs accumulated across MTP depths; final outer/inner layout is
        # outputs_per_mb[mb_idx][depth_idx] to match the single-microbatch API shape.
        outputs_per_mb: list[list[MTPDepthOutput]] = [[] for _ in range(n)]
        current_hidden_states_list = list(hidden_states_list)
        current_seq_ctx_list = list(seq_ctx_list)
        previous_layer_results: MTPInternalOutput | None = None

        num_steps = self.mtp_config.num_layers
        for step in range(num_steps):
            layer = cast(MTPLayer, self.layers[0] if self.mtp_config.share_weights else self.layers[step])

            current_seq_ctx_list = [roll_sequence_context(ctx, shifts=-1) for ctx in current_seq_ctx_list]
            future_embeddings_list = [self._embed_future(ctx, embed_tokens_fn) for ctx in current_seq_ctx_list]

            layer_results = cast(
                MoEDecoderLayerMicroBatchOutput,
                self._call_decoder_layer(
                    layer,
                    current_hidden_states_list,
                    previous_layer_results=previous_layer_results,
                    future_embeddings=future_embeddings_list,
                    position_embeddings=position_embeddings_list,
                    seq_ctx=current_seq_ctx_list,
                ),
            )
            previous_layer_results = layer_results

            for mb_idx in range(n):
                outputs_per_mb[mb_idx].append(
                    {
                        "hidden_states": layer_results["hidden_states"][mb_idx],
                        "router_logits": layer_results["router_logits"][mb_idx],
                        "router_weights": layer_results["router_weights"][mb_idx],
                        "router_topk_ids": layer_results["router_topk_ids"][mb_idx],
                    }
                )

            current_hidden_states_list = layer_results["hidden_states"]

        return outputs_per_mb

    @staticmethod
    def _embed_future(
        seq_ctx: SequenceContext,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if seq_ctx.inputs_embeds is None:
            return embed_tokens_fn(seq_ctx.input_ids)  # type: ignore[arg-type]
        return seq_ctx.inputs_embeds
