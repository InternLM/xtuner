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
        *hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, ...]:
        """Forward pass through the MTP layer.

        Mirrors :meth:`MoEDecoderLayer.forward`: when a single ``hidden_states`` tensor is
        provided, the layer runs the regular single-microbatch path and returns a 4-tuple
        ``(hidden, router_logits, router_weights, router_topk_ids)``. When ``N`` hidden states are provided
        (intra-layer micro-batching / domino EP), ``future_embeddings``, ``position_embeddings``
        and ``seq_ctx`` must be lists of length ``N``; the per-microbatch preprocessing
        (enorm/hnorm/eh_proj) is run independently and a single underlying decoder forward
        is issued so the inner MoE EP communication can be overlapped across micro-batches.

        Args:
            hidden_states (torch.Tensor): One or more hidden state tensors. A single tensor
                triggers the single-microbatch path; multiple tensors trigger the
                multi-microbatch path.
            future_embeddings (torch.Tensor | list[torch.Tensor]): Embeddings of the future
                tokens, aligned per-microbatch with ``hidden_states``.
            position_embeddings (tuple | list[tuple]): Rotary position embeddings (cos, sin),
                aligned per-microbatch with ``hidden_states``.
            seq_ctx (SequenceContext | list[SequenceContext]): Sequence context per micro-batch.

        Returns:
            tuple: For single-microbatch input, a 4-tuple
                ``(hidden_states, router_logits, router_weights, router_topk_ids)``.
                For ``N`` micro-batches, a flat tuple of length ``4 * N`` matching the
                convention used by :meth:`MoEDecoderLayer._micro_batch_forward`:
                ``(hidden_0, ..., hidden_{N-1}, router_logits_0, ...,
                router_weights_{N-1}, router_topk_ids_0, ..., router_topk_ids_{N-1})``.
        """
        if len(hidden_states) == 1:
            assert isinstance(future_embeddings, torch.Tensor), (
                "future_embeddings should be a Tensor in single-microbatch mode"
            )
            assert isinstance(seq_ctx, SequenceContext), (
                "seq_ctx should be a SequenceContext instance in single-microbatch mode"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a (cos, sin) tuple in single-microbatch mode"
            )
            return self._forward(
                hidden_states=hidden_states[0],
                future_embeddings=future_embeddings,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )

        assert isinstance(future_embeddings, list), (
            "future_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(seq_ctx, list), (
            "seq_ctx should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(position_embeddings, list), (
            "position_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        return self._micro_batch_forward(
            hidden_states_list=list(hidden_states),
            future_embeddings_list=future_embeddings,
            position_embeddings_list=position_embeddings,
            seq_ctx_list=seq_ctx,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self._preprocess(hidden_states=hidden_states, future_embeddings=future_embeddings)

        hidden_states, router_results, router_weights, router_topk_ids = self.decoder_layer(
            projected,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, router_results, router_weights, router_topk_ids

    def _micro_batch_forward(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        future_embeddings_list: list[torch.Tensor],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx_list: list[SequenceContext],
    ) -> tuple[torch.Tensor, ...]:
        n = len(hidden_states_list)
        assert len(future_embeddings_list) == n and len(position_embeddings_list) == n and len(seq_ctx_list) == n, (
            "All per-microbatch inputs must share the same length"
        )

        # Run MTP preprocessing eagerly across all micro-batches so the underlying decoder
        # layer can overlap its EP communication in a single fused forward.
        projected_list = [
            self._preprocess(hidden_states=h, future_embeddings=e)
            for h, e in zip(hidden_states_list, future_embeddings_list)
        ]

        layer_results = self.decoder_layer(
            *projected_list,
            position_embeddings=position_embeddings_list,
            seq_ctx=seq_ctx_list,
        )
        assert isinstance(layer_results, tuple) and len(layer_results) == 4 * n, (
            "Multi-microbatch MTP requires the wrapped decoder layer to return a flat "
            f"(hidden..., router_logits..., router_weights..., router_topk_ids...) tuple of length {4 * n}; "
            f"got length {len(layer_results) if isinstance(layer_results, tuple) else type(layer_results)}"
        )

        hidden_out = [self.final_layernorm(h) for h in layer_results[:n]]
        router_logits = list(layer_results[n : 2 * n])
        router_weights = list(layer_results[2 * n : 3 * n])
        router_topk_ids = list(layer_results[3 * n :])
        return tuple(hidden_out + router_logits + router_weights + router_topk_ids)

    def _preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize embeddings and hidden states separately so both inputs share a numerical
        # range, then concatenate along the last dim and project back to ``hidden_size``.
        normalized_embedding = self.enorm(future_embeddings)
        normalized_hidden = self.hnorm(hidden_states)
        combined = torch.cat([normalized_embedding, normalized_hidden], dim=-1)
        return self.eh_proj(combined)
