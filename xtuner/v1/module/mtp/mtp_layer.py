"""Multi-Token Prediction (MTP) Layer implementation."""

from typing import Literal

import torch
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.linear import build_linear


class MTPLayer(nn.Module):
    """Single Multi-Token Prediction (MTP) layer.

    MTP Layer wraps a standard decoder layer with MTP-specific preprocessing
    and postprocessing. The structure is:

        [enorm + hnorm + projection] → [DecoderLayer] → [final_layernorm]

    The k-th MTP layer predicts the (i+k)-th token by combining:
    1. Hidden states from the previous MTP layer (or main model)
    2. Embedding of the future token at position (i+k)

    Note: The decoder layer's internal normalization (input_layernorm) is preserved
    for simplicity and modularity. While this adds a small computational overhead,
    it allows MTP to work with any decoder layer implementation (Dense, MoE, etc.)
    without modification.

    Args:
        hidden_size (int): Hidden dimension size.
        rms_norm_eps (float): Epsilon for RMSNorm.
        rms_norm_type (str): Type of RMSNorm ("default" or "zero_centered").
        decoder_layer (nn.Module): A fully constructed decoder layer instance.
            This can be DenseDecoderLayer, MoEDecoderLayer, or any custom decoder layer
            that implements the standard forward signature.
        float8_cfg: Float8 configuration for the projection layer.

    Example:
        >>> from xtuner.v1.module.decoder_layer import DenseDecoderLayer
        >>> decoder_layer = DenseDecoderLayer(
        ...     hidden_size=512,
        ...     intermediate_size=2048,
        ...     ...
        ... )
        >>> mtp_layer = MTPLayer(
        ...     hidden_size=512,
        ...     rms_norm_eps=1e-6,
        ...     rms_norm_type="default",
        ...     decoder_layer=decoder_layer,
        ... )
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        rms_norm_type: Literal["default", "zero_centered"],
        decoder_layer: nn.Module,
        float8_cfg=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # MTP-specific preprocessing components
        self.enorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.hnorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.eh_proj = build_linear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )

        # Core decoder layer (Dense, MoE, or any custom implementation)
        self.decoder_layer = decoder_layer

        # MTP-specific postprocessing component
        self.final_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the MTP layer.

        Args:
            hidden_states (torch.Tensor): Hidden states from previous layer,
                shape [batch, seq_len, hidden_size].
            future_embeddings (torch.Tensor): Embeddings of future tokens,
                shape [batch, seq_len, hidden_size].
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Rotary position
                embeddings (cos, sin).
            seq_ctx (SequenceContext): Sequence context containing attention mask, etc.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A 3-tuple of
                (hidden_states, router_weights, router_results) where each tensor
                has shape [batch, seq_len, ...].
        """
        # Step 1: Normalize embeddings and hidden states separately
        # This ensures both inputs are in the same numerical range
        normalized_embedding = self.enorm(future_embeddings)
        normalized_hidden = self.hnorm(hidden_states)

        # Step 2: Concatenate and project to combine information
        # [B, S, H] + [B, S, H] → [B, S, 2H] → [B, S, H]
        combined = torch.cat([normalized_embedding, normalized_hidden], dim=-1)
        projected = self.eh_proj(combined)

        # Step 3: Pass through the standard decoder layer
        # This includes attention, MLP, and their respective normalizations
        # TODO: TMP hardcode here.
        hidden_states, router_results, router_weights = self.decoder_layer(
            projected,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )

        # Step 4: Final normalization before output
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, router_results, router_weights
