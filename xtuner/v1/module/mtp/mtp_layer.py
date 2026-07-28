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
        *layer_inputs: torch.Tensor,
        future_embeddings: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, ...]:
        """Forward pass through the MTP layer.

        Mirrors :meth:`MoEDecoderLayer.forward`: DSA calls append explicit
        ``dsa_topk_ids`` to both the flat inputs and results. The enclosing
        :class:`MTPBlock` consumes that extra category internally and keeps the
        public per-depth output unchanged.

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
            tuple: A flat ``4 * N`` result for non-DSA decoders or ``5 * N``
                result for DSA decoders, with ``dsa_topk_ids`` as the final
                category.
        """
        if isinstance(seq_ctx, SequenceContext):
            assert len(layer_inputs) in (1, 2), (
                "Single-microbatch MTPLayer expects hidden_states and optional dsa_topk_ids."
            )
            assert isinstance(future_embeddings, torch.Tensor), (
                "future_embeddings should be a Tensor in single-microbatch mode"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a (cos, sin) tuple in single-microbatch mode"
            )
            return self._forward(
                hidden_states=layer_inputs[0],
                dsa_topk_ids=layer_inputs[1] if len(layer_inputs) == 2 else None,
                future_embeddings=future_embeddings,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )

        n = len(seq_ctx)
        assert len(layer_inputs) in (n, 2 * n), (
            f"Multi-microbatch MTPLayer expects {n} hidden states and optional {n} dsa_topk_ids."
        )
        assert isinstance(future_embeddings, list), (
            "future_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(position_embeddings, list), (
            "position_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        dsa_topk_ids_list: list[torch.Tensor | None]
        if len(layer_inputs) == n:
            dsa_topk_ids_list = [None] * n
        else:
            dsa_topk_ids_list = list(layer_inputs[n:])
        return self._micro_batch_forward(
            hidden_states_list=list(layer_inputs[:n]),
            dsa_topk_ids_list=dsa_topk_ids_list,
            future_embeddings_list=future_embeddings,
            position_embeddings_list=position_embeddings,
            seq_ctx_list=seq_ctx,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        dsa_topk_ids: torch.Tensor | None,
        future_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> tuple[torch.Tensor, ...]:
        projected = self._preprocess(hidden_states=hidden_states, future_embeddings=future_embeddings)

        decoder_inputs = (projected,) if dsa_topk_ids is None else (projected, dsa_topk_ids)
        layer_results = self.decoder_layer(
            *decoder_inputs,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        assert len(layer_results) in (4, 5)
        hidden_states, router_results, router_weights, router_topk_ids = layer_results[:4]

        hidden_states = self.final_layernorm(hidden_states)
        mtp_results = (hidden_states, router_results, router_weights, router_topk_ids)
        if len(layer_results) == 4:
            return mtp_results
        return (*mtp_results, layer_results[4])

    def _micro_batch_forward(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        dsa_topk_ids_list: list[torch.Tensor | None],
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

        decoder_inputs = list(projected_list)
        if any(dsa_topk_ids is not None for dsa_topk_ids in dsa_topk_ids_list):
            assert all(dsa_topk_ids is not None for dsa_topk_ids in dsa_topk_ids_list)
            decoder_inputs.extend(dsa_topk_ids for dsa_topk_ids in dsa_topk_ids_list if dsa_topk_ids is not None)
        layer_results = self.decoder_layer(
            *decoder_inputs,
            position_embeddings=position_embeddings_list,
            seq_ctx=seq_ctx_list,
        )
        assert isinstance(layer_results, tuple) and len(layer_results) in (4 * n, 5 * n), (
            "Multi-microbatch MTP requires the wrapped decoder layer to return a flat "
            "(hidden..., router_logits..., router_weights..., router_topk_ids..., optional dsa_topk_ids...) "
            f"tuple of length {4 * n} or {5 * n}; "
            f"got length {len(layer_results) if isinstance(layer_results, tuple) else type(layer_results)}"
        )

        hidden_out = [self.final_layernorm(h) for h in layer_results[:n]]
        router_logits = list(layer_results[n : 2 * n])
        router_weights = list(layer_results[2 * n : 3 * n])
        router_topk_ids = list(layer_results[3 * n : 4 * n])
        mtp_results = hidden_out + router_logits + router_weights + router_topk_ids
        if len(layer_results) == 4 * n:
            return tuple(mtp_results)
        return tuple(mtp_results + list(layer_results[4 * n :]))

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
