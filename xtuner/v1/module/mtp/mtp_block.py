"""Multi-Token Prediction (MTP) Block implementation."""

from typing import Callable

import torch
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext

from .mtp_layer import MTPLayer
from .utils import roll_sequence_context


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
        >>> mtp_block = MTPBlock(mtp_layers=mtp_layers)
        >>>
        >>> # Forward pass
        >>> outputs = mtp_block(
        ...     hidden_states=h,
        ...     input_ids=ids,
        ...     position_ids=pos,
        ...     embed_tokens_fn=embed_fn,
        ...     position_embeddings=pos_emb,
        ...     seq_ctx=ctx,
        ... )
        >>> # outputs[0]: predictions for i+1
        >>> # outputs[1]: predictions for i+2
    """

    def __init__(self, *, mtp_layers: list[MTPLayer]):
        super().__init__()
        if not mtp_layers:
            raise ValueError("mtp_layers cannot be empty")

        self.layers = nn.ModuleList(mtp_layers)
        self.num_layers = len(mtp_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: List of 3-tuples
                (hidden_states, router_weights, router_results) for each MTP depth.
                Length equals num_layers.
                - outputs[0]: Outputs for predicting token at position (i+1)
                - outputs[k]: Outputs for predicting token at position (i+k+1)
        """
        mtp_outputs = []
        current_hidden_states = hidden_states
        current_seq_ctx = seq_ctx

        for layer in self.layers:
            # Roll sequence context to get future tokens
            # This shifts each packed sequence independently, respecting boundaries
            current_seq_ctx = roll_sequence_context(current_seq_ctx, shifts=-1)

            # Get embeddings for future tokens
            if current_seq_ctx.inputs_embeds is None:
                future_embeddings = embed_tokens_fn(current_seq_ctx.input_ids)  # type: ignore[arg-type]
            else:
                future_embeddings = current_seq_ctx.inputs_embeds

            # Forward through MTP layer
            current_hidden_states = layer(
                hidden_states=current_hidden_states,
                future_embeddings=future_embeddings,
                position_embeddings=position_embeddings,
                seq_ctx=current_seq_ctx,
            )
            # Save output for this depth
            mtp_outputs.append(current_hidden_states)

        return mtp_outputs
