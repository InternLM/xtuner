# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# One DeepSeek-V4-Flash decoder layer, split out of ``model/moe/deepseek_v4.py``.
# Holds the HC residual-mix + DSA attention + MoE-FFN stack for a single layer
# (``V4DecoderLayer``) plus the FFN-dispatch carry-state (``V4FFNState``). The
# model module builds these per layer and drives them from its ``_forward`` /
# ``_micro_batch_forward``; this file owns no model-level state — embed, rope and
# the final norm all stay in the model.
# ============================================================================

from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import HashRouterConfig, NoAuxRouterConfig, RouterResults
from xtuner.v1.module.attention.dsa import DeepSeekSparseAttention
from .hc_block import HCWrapperConfig, _unshard_hc_params, hc_post, hc_pre
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEBlock,
    MoEGate,
    MoEMLP,
)
from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rms_norm import RMSNorm


class V4FFNState(NamedTuple):
    """State threaded across the FFN-dispatch boundary in V4 Domino EP.

    Captured by :meth:`V4DecoderLayer._forward_pre_ffn_dispatch` and consumed by
    :meth:`V4DecoderLayer._forward_post_ffn_combine`. Holds every tensor / shape /
    Sinkhorn weight needed to finish the HC-wrapped FFN block after the MoE
    dispatcher returns. Defined as a ``NamedTuple`` (not a dataclass) so it stays
    immutable and structurally hashable for downstream tracing.

    Args:
        residual (torch.Tensor): HC streams entering ``hc_pre_ffn``, consumed by
            ``hc_post_ffn`` as the additive residual after the FFN block.
        post_f (torch.Tensor): Per-stream ``post`` weights from the FFN-side
            ``hc_pre`` Sinkhorn (shape ``[1, S, hc_mult]``).
        comb_f (torch.Tensor): Doubly-stochastic ``comb`` matrix from the
            FFN-side ``hc_pre`` Sinkhorn (shape ``[1, S, hc_mult, hc_mult]``).
        h_normed (torch.Tensor): Output of ``post_attention_layernorm`` (the
            ``h`` that the dispatcher flattens for transport), fed back into
            ``_ffn_post_compute`` so shared experts see the same activations the
            routed experts did.
        origin_shape (torch.Size): Shape of ``h_normed`` before flatten, used to
            ``.view()`` the dispatcher's combined output back into rank order.
        router_results (RouterResults): Output of the gate; carries ``topk_ids``
            / ``topk_weights`` for the dispatcher and ``logits`` / ``router_weights``
            for aux-loss accumulation.
    """

    residual: torch.Tensor
    post_f: torch.Tensor
    comb_f: torch.Tensor
    h_normed: torch.Tensor
    origin_shape: torch.Size
    router_results: RouterResults


class V4DecoderLayer(nn.Module):
    """One DeepSeek-V4-Flash decoder layer: HC residual mix + DSA attn + MoE ffn.

    Owns every submodule and parameter for one layer directly — no inner /
    outer wrapper, no inheritance through ``MoEDecoderLayer``, no
    ``set_context`` side-channel. The previous three-class layout
    (``_V4InnerBlock(MoEDecoderLayer)`` → ``HCDecoderLayer`` → ``V4DecoderLayer``)
    was a chain of "narrow generic protocol + adapter + bridge" that had exactly
    one user (V4); inlining the chain removes the ``set_context`` /
    ``_last_router_results`` mutable state, the dual-registration assert
    (``hc_layer.inner is inner``) and the inherited-but-never-called
    ``MoEDecoderLayer.forward``.

    Forward contract:

        layer(hidden_states,
              *, position_embeddings, position_embeddings_compressed,
                 seq_ctx, input_ids)
            -> (hidden_states_out, router_logits, router_weights)

    All inputs flow in through arguments; router results flow out through
    the tuple. No hidden state on ``self`` between forward calls.

    Compile boundaries are exposed as separate private methods so the V4
    compile cfg can target them individually:

    * :meth:`_attn_compute`     — ``input_layernorm`` + DSA (heavy, has matmul + sparse_attn).
    * :meth:`_ffn_pre_compute`  — ``post_attention_layernorm`` + gate (lite).
    * :meth:`_ffn_post_compute` — ``+ shared_experts`` + ``* hidden_factor`` (lite).
    * :meth:`_shared_experts_forward` — shared expert FFN ± gate (called by post).

    Args:
        compress_ratio (int): Per-layer DSA mode (0 / 4 / 128).
        layer_idx (int): Position in the decoder stack.
        hidden_size (int): One stream's hidden size.
        moe_intermediate_size (int): MoE expert FFN intermediate dim.
        hidden_act (str): MoE expert activation name.
        num_experts_per_tok (int): Routed experts per token.
        n_routed_experts (int): Total routed experts.
        n_shared_experts (int): Shared-expert count (0 disables).
        moe_act_fn_cfg (MoEActFnConfig): Expert activation policy.
        router_config (NoAuxRouterConfig | HashRouterConfig): Router topology.
            HashRouter for the first ``num_hash_layers`` layers; NoAux for
            the rest.
        dispatcher (str | None): EP dispatcher backend; ``None`` for non-EP.
        ep_mesh (DeviceMesh | None): EP device mesh.
        hc_cfg (HCWrapperConfig): ``hc_mult`` / ``hc_eps`` /
            ``hc_sinkhorn_iters``. ``hc_mult == 1`` degenerates to a plain
            pre-norm residual block (kept for parity-anchor testing).
        attention_module (nn.Module): Pre-built ``DeepSeekSparseAttention``.
            DSAConfig.build requires a per-layer ``compress_ratio`` so the
            caller has to construct DSA outside this class.
        rms_norm_eps (float): RMSNorm epsilon.
        rms_norm_type ("default" | "zero_centered"): RMSNorm variant.
        mlp_bias (bool): Bias on shared experts' MLP linears.
        gate_bias (bool): Bias on the routing gate.
        moe_bias (bool): Bias on expert linears.
        with_shared_expert_gate (bool): Whether to add a sigmoid gate over
            shared experts.
        hidden_factor (float): Scalar applied to the combined FFN output
            before the HC ``post`` residual mix.
        router_compute_dtype ("float32" | "native"): Router math precision.
        float8_cfg (Float8Config | None): FP8 expert/grouped-linear config.
    """

    def __init__(
        self,
        *,
        compress_ratio: int,
        layer_idx: int,
        hidden_size: int,
        moe_intermediate_size: int,
        hidden_act: str,
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        moe_act_fn_cfg: MoEActFnConfig,
        router_config: NoAuxRouterConfig | HashRouterConfig,
        dispatcher: str | None,
        ep_mesh: DeviceMesh | None,
        hc_cfg: HCWrapperConfig,
        attention_module: nn.Module,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: str = "default",
        mlp_bias: bool = False,
        gate_bias: bool = False,
        moe_bias: bool = False,
        with_shared_expert_gate: bool = False,
        hidden_factor: float = 1.0,
        router_compute_dtype: str = "float32",
        float8_cfg: Any = None,
    ) -> None:
        super().__init__()

        self.compress_ratio = compress_ratio
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.with_shared_expert_gate = with_shared_expert_gate
        self.hidden_factor = hidden_factor
        self.ep_mesh = ep_mesh

        # ─── HC parameters ───
        # V4 stores HC parameters in fp32 even when the rest of the model is
        # bf16 (V4 reference: `Block.__init__` uses `with set_dtype(torch.float32)`)
        # because the 20-iteration Sinkhorn is bf16-unstable.
        self.hc_mult = hc_cfg.hc_mult
        self.hc_eps = hc_cfg.hc_eps
        self.hc_sinkhorn_iters = hc_cfg.hc_sinkhorn_iters
        mix_dim = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * hidden_size
        fp32 = torch.float32
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_dim, hc_dim, dtype=fp32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_dim, dtype=fp32))
        self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=fp32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_dim, hc_dim, dtype=fp32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_dim, dtype=fp32))
        self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=fp32))
        # Degenerate-safe init: scale[0]=1 keeps the pre-weight derivative
        # non-zero so training can escape the all-zero attractor; scale[1] /
        # scale[2] = 0 starts post and comb from constant uniform values rather
        # than random softmax noise.
        with torch.no_grad():
            self.hc_attn_scale[0] = 1.0
            self.hc_ffn_scale[0] = 1.0

        # ─── Norms + attention ───
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)  # type: ignore[arg-type]
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)  # type: ignore[arg-type]
        # DSA is built outside this class because DSAConfig.build needs a
        # per-layer ``compress_ratio`` that the generic
        # ``attention_config.build`` chain in ``MoEDecoderLayer`` doesn't know
        # how to thread. ``_build_one_layer`` constructs DSA and hands it here.
        self.self_attn = attention_module

        # ─── MoE routing + experts + dispatcher ───
        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_config=router_config,
            gate_bias=gate_bias,
            router_compute_dtype=router_compute_dtype,  # type: ignore[arg-type]
        )
        self.experts = MoEBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
        )
        process_group = ep_mesh.get_group() if ep_mesh is not None else None
        self.dispatcher = build_dispatcher(
            dispatcher=dispatcher,  # type: ignore[arg-type]
            n_routed_experts=n_routed_experts,
            ep_group=process_group,
            training_dtype="fp8" if float8_cfg is not None else "bf16",
            generate_dtype="bf16",
        )

        # ─── Shared experts (optional) ───
        self.shared_experts: MoEMLP | None
        self.shared_expert_gate: nn.Module | None
        if n_shared_experts > 0:
            self.shared_experts = MoEMLP(
                hidden_size=hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
            )
            self.shared_expert_gate = build_linear(hidden_size, 1, bias=False) if with_shared_expert_gate else None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

    # ───────── public forward ─────────

    def forward(
        self,
        *hidden_states: torch.Tensor,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor]
        | None
        | list[tuple[torch.Tensor, torch.Tensor] | None],
        input_ids: torch.Tensor | list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, ...]:
        """One V4-Flash decoder pass — single-MB or Domino-EP multi-MB.

        Dispatches on the variadic ``hidden_states`` count: a single tensor
        runs the sequential :meth:`_forward`, ``N >= 2`` tensors plus aligned
        list-typed sibling args run :meth:`_micro_batch_forward`'s wave
        pipeline. Variadic positional ``*hidden_states`` mirrors
        :meth:`MoEDecoderLayer.forward` and is required for FSDP2 correctness:
        the multi-MB wave pipeline must execute **inside** a single
        ``forward()`` call so the pre/post forward hooks ``fully_shard``
        registers bracket the whole multi-MB pass exactly once.

        Args:
            hidden_states (torch.Tensor): One or more HC-expanded packed
                varlen activations, each shape ``[1, total_tokens, hc_mult, hidden_size]``.
                A single tensor selects the sequential path; ``>= 2`` select
                the Domino multi-MB path.
            seq_ctx (SequenceContext | list[SequenceContext]): Single
                ``SequenceContext`` for sequential, aligned list for multi-MB.
                Carries ``cu_seq_lens`` for varlen and
                ``rollout_routed_experts`` for the gate fast-path.
            position_embeddings (tuple[torch.Tensor, torch.Tensor] | list[...]):
                Dense rope basis ``(cos, sin)`` for DSA sliding-window heads,
                single tuple for sequential, aligned list for multi-MB.
            position_embeddings_compressed (tuple[torch.Tensor, torch.Tensor] | None | list[...]):
                Compressed rope basis ``(cos, sin)`` for the Indexer; ``None``
                when this layer's ``compress_ratio != 4``. Aligned list for
                multi-MB.
            input_ids (torch.Tensor | list[torch.Tensor] | None): Per-token
                ids consumed by :class:`HashRouter` in the first
                ``num_hash_layers`` layers; ignored by ``NoAuxRouter``.
                Aligned list for multi-MB or single tensor for sequential.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, ...]:
                Sequential: ``(hidden_states, router_logits, router_weights)``.
                Multi-MB: a flat tuple of length ``3 * N`` —
                ``(h0, ..., h_{N-1}, rl0, ..., rl_{N-1}, rw0, ..., rw_{N-1})``,
                mirroring :meth:`MoEDecoderLayer._micro_batch_forward`'s
                return contract so callers unpack identically across the two
                layer flavours.
        """
        if len(hidden_states) == 1:
            assert isinstance(seq_ctx, SequenceContext), (
                f"Single-MB forward expects `seq_ctx` as a SequenceContext, got {type(seq_ctx).__name__}"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "Single-MB forward expects `position_embeddings` as a (cos, sin) tuple"
            )
            assert position_embeddings_compressed is None or (
                isinstance(position_embeddings_compressed, tuple) and len(position_embeddings_compressed) == 2
            ), "Single-MB forward expects `position_embeddings_compressed` as a (cos, sin) tuple or None"
            assert input_ids is None or isinstance(input_ids, torch.Tensor), (
                "Single-MB forward expects `input_ids` as a torch.Tensor or None"
            )
            return self._forward(
                hidden_states[0],
                position_embeddings=position_embeddings,
                position_embeddings_compressed=position_embeddings_compressed,
                seq_ctx=seq_ctx,
                input_ids=input_ids,
            )

        n = len(hidden_states)
        assert isinstance(seq_ctx, list) and len(seq_ctx) == n, (
            f"Multi-MB forward expects `seq_ctx` as a list of length {n}"
        )
        assert isinstance(position_embeddings, list) and len(position_embeddings) == n, (
            f"Multi-MB forward expects `position_embeddings` as a list of length {n}"
        )
        assert isinstance(position_embeddings_compressed, list) and len(position_embeddings_compressed) == n, (
            f"Multi-MB forward expects `position_embeddings_compressed` as a list of length {n}"
        )
        if input_ids is not None:
            assert isinstance(input_ids, list) and len(input_ids) == n, (
                f"Multi-MB forward expects `input_ids` as a list of length {n} (or None)"
            )
        return self._micro_batch_forward(
            hidden_states_list=list(hidden_states),
            seq_ctx_list=seq_ctx,
            position_embeddings_list=position_embeddings,
            position_embeddings_compressed_list=position_embeddings_compressed,
            input_ids_list=input_ids,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential single-MB layer pass — HC-wrapped attn then HC-wrapped ffn.

        Body of the pre-Domino ``forward()`` — preserved bit-for-bit so the
        single-MB code path stays unchanged when ``forward()`` dispatches a
        single-tensor call here.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ``hidden_states`` ``[1, total_tokens, hc_mult, hidden_size]``
                - ``router_logits`` (passed to ``MoE.aux_loss.accumulate``)
                - ``router_weights`` (passed to ``MoE.aux_loss.accumulate``)
        """
        if self.hc_mult == 1:
            # Degenerate path: HC carries no mixing information at H=1; this
            # branch falls back to the plain pre-norm residual that non-HC
            # decoders use. Kept as a structural parity anchor for testing.
            return self._plain_residual_forward(
                hidden_states,
                position_embeddings,
                position_embeddings_compressed,
                seq_ctx,
                input_ids,
            )

        # ─── HC-wrapped attention ───
        # Hoist DTensor.full_tensor() out of any compile region: doing it here
        # (eager) lets the downstream ``hc_pre`` stay a single contiguous
        # compiled graph instead of breaking three times mid-trace for each
        # HC param under FSDP/EP.
        attn_fn, attn_scale, attn_base = _unshard_hc_params(self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        residual = hidden_states
        x_reduced, post_a, comb_a = hc_pre(
            hidden_states,
            attn_fn,
            attn_scale,
            attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        attn_out = self._attn_compute(x_reduced, position_embeddings, position_embeddings_compressed, seq_ctx)
        hidden_states = hc_post(attn_out, residual, post_a, comb_a)

        # ─── HC-wrapped FFN (MoE) ───
        ffn_fn, ffn_scale, ffn_base = _unshard_hc_params(self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        residual = hidden_states
        x_reduced, post_f, comb_f = hc_pre(
            hidden_states,
            ffn_fn,
            ffn_scale,
            ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        ffn_out, router_results = self._ffn_compute(x_reduced, seq_ctx, input_ids)
        hidden_states = hc_post(ffn_out, residual, post_f, comb_f)

        return hidden_states, router_results["logits"], router_results["router_weights"]

    # ───────── Domino-EP staged forward halves ─────────
    #
    # ``forward()`` is the single-MB entry point and runs the full layer end-to-end.
    # For Domino EP we need to split that forward at the FFN-dispatch boundary so
    # the outer micro-batch driver in :meth:`DeepSeekV4._domino_micro_batch_forward`
    # can interleave dispatcher async ops across MBs. The two halves below carry
    # exactly the same compute as ``forward()``; only the dispatcher chain
    # (``dispatch_preprocess`` → ``combine_postprocess``) is lifted out into the
    # outer driver. ``hc_mult == 1`` (the degenerate ``_plain_residual_forward``
    # path) is intentionally not supported here — V4 production runs all use
    # ``hc_mult == 4`` and the degenerate path has no FFN dispatch overlap to win.

    def _forward_pre_ffn_dispatch(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, V4FFNState]:
        """First half of a Domino-EP layer pass: everything before FFN dispatch.

        Runs HC-wrapped attention end-to-end, then HC-pre for the FFN side, then
        ``_ffn_pre_compute`` (post-attn LN + gate). Returns the flattened hidden
        states ready for :meth:`GenericDispatcher.dispatch_preprocess` and a
        :class:`V4FFNState` carrying the state the outer driver must thread back
        into :meth:`_forward_post_ffn_combine` once the dispatcher finishes.

        Args:
            hidden_states (torch.Tensor): HC-expanded streams, shape
                ``[1, total_tokens, hc_mult, hidden_size]``.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Dense rope
                basis ``(cos, sin)`` for DSA sliding-window heads.
            position_embeddings_compressed (tuple[torch.Tensor, torch.Tensor] | None):
                Compressed rope basis ``(cos, sin)`` for the Indexer; ``None``
                when this layer's ``compress_ratio != 4``.
            seq_ctx (SequenceContext): Carries ``cu_seq_lens`` /
                ``rollout_routed_experts`` (the latter is sliced per layer here
                so the dispatcher sees a single layer's pre-routed expert ids).
            input_ids (torch.Tensor | None): Per-token ids consumed by the hash
                router in the first ``num_hash_layers`` layers.

        Returns:
            tuple[torch.Tensor, V4FFNState]:
                - Flattened, dispatch-ready hidden states with shape
                  ``[total_tokens, hidden_size]``.
                - Carry-state for :meth:`_forward_post_ffn_combine`.
        """
        assert self.hc_mult > 1, (
            "Domino-EP staged forward only supports hc_mult > 1; hc_mult == 1 "
            "must use the synchronous forward() / _plain_residual_forward path."
        )

        # ─── HC-wrapped attention (identical to forward()) ───
        attn_fn, attn_scale, attn_base = _unshard_hc_params(self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        residual = hidden_states
        x_reduced, post_a, comb_a = hc_pre(
            hidden_states,
            attn_fn,
            attn_scale,
            attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        attn_out = self._attn_compute(x_reduced, position_embeddings, position_embeddings_compressed, seq_ctx)
        hidden_states = hc_post(attn_out, residual, post_a, comb_a)

        # ─── HC-pre for FFN (mirrors forward()) ───
        ffn_fn, ffn_scale, ffn_base = _unshard_hc_params(self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        residual_ffn = hidden_states
        x_reduced, post_f, comb_f = hc_pre(
            hidden_states,
            ffn_fn,
            ffn_scale,
            ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )

        # ─── FFN pre-compute (gate + post-attn LN), same slicing as _ffn_compute ───
        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]
        else:
            rollout_routed_experts = None
        h, router_results = self._ffn_pre_compute(x_reduced, rollout_routed_experts, input_ids)

        state = V4FFNState(
            residual=residual_ffn,
            post_f=post_f,
            comb_f=comb_f,
            h_normed=h,
            origin_shape=h.shape,
            router_results=router_results,
        )
        return h.view(-1, h.shape[-1]), state

    def _forward_post_ffn_combine(
        self,
        post_combined_hidden_states: torch.Tensor,
        state: V4FFNState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Second half of a Domino-EP layer pass: everything after FFN combine.

        Takes the dispatcher's ``combine_postprocess`` output plus the state
        captured by :meth:`_forward_pre_ffn_dispatch` and finishes the layer:
        ``_ffn_post_compute`` (shared experts + scalar scale) and HC-post for
        the FFN side. Returns the same triple as :meth:`forward`.

        Args:
            post_combined_hidden_states (torch.Tensor): Output of
                :meth:`GenericDispatcher.combine_postprocess`'s ``hidden_states``
                field, shape ``[total_tokens, hidden_size]``.
            state (V4FFNState): Carry-state from
                :meth:`_forward_pre_ffn_dispatch`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Same contract as
                :meth:`forward` — ``(hidden_states_out, router_logits, router_weights)``.
        """
        combined_hidden_states = post_combined_hidden_states.view(*state.origin_shape)
        ffn_out = self._ffn_post_compute(combined_hidden_states, state.h_normed)
        hidden_states = hc_post(ffn_out, state.residual, state.post_f, state.comb_f)
        return (
            hidden_states,
            state.router_results["logits"],
            state.router_results["router_weights"],
        )

    def _micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        position_embeddings_compressed_list: list[tuple[torch.Tensor, torch.Tensor] | None],
        input_ids_list: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Domino-EP wave pipeline across micro-batches for one V4 decoder layer.

        Runs inside :meth:`forward` (via the variadic ``*hidden_states``
        dispatch) so the single ``fully_shard`` pre/post-forward hook pair
        brackets the entire multi-MB pass — required for correct parameter
        all-gather, resharding and gradient accumulation under FSDP2. The
        schedule mirrors :meth:`MoEDecoderLayer._micro_batch_forward`:

            Phase A — for each MB sequentially:
                HC-attn + HC-pre-FFN + ffn_pre_compute (via
                :meth:`_forward_pre_ffn_dispatch`), then async
                ``dispatch_preprocess``. Queues the MB-side compute on the
                main stream and the dispatch op on ``_comm_stream``.
            Phase B1 — for each MB sequentially:
                ``dispatch`` → ``dispatch_postprocess`` → ``experts`` →
                ``combine_preprocess``, all with ``async_op=True``. Combine
                is **not** issued here so the comm stream sees
                ``D0, D1, ..., DN-1`` back-to-back, which is essential for
                backward overlap (see below).
            Phase B2 — for each MB sequentially:
                ``combine`` with ``async_op=True``. Comm stream now sees
                ``C0, C1, ..., CN-1`` back-to-back.
            Phase C — for each MB sequentially:
                ``combine_postprocess`` → ``_forward_post_ffn_combine``
                (FFN-post + HC-post-FFN).

        Why split combine into its own phase: interleaving dispatch and
        combine per-MB (``D0, C0, D1, C1, ...``) leaves the comm stream in
        a state where, in backward, ``C0.bwd`` is queued **behind**
        ``D1.bwd`` — but ``D1.bwd`` itself has to wait for the main-stream
        backward chain (``E1.bwd → DP1.bwd``) to complete before it can
        fire. That stalls ``C0.bwd`` even though its grad from
        ``CPo0.bwd`` has been ready for a while. Splitting into
        ``D0, ..., DN-1, C0, ..., CN-1`` matches :class:`MoEDecoderLayer`'s
        layout and puts ``C.bwd`` ops back-to-back in the backward queue,
        so they can stream through comm while the main-stream PMF/CPo
        backwards run in parallel.

        Output is bit-identical to issuing the same MBs through :meth:`forward`
        sequentially — the schedule reorder is a CUDA-stream issue order only.

        Args:
            hidden_states_list (list[torch.Tensor]): Per-MB HC-expanded
                streams, each shape ``[1, total_tokens, hc_mult, hidden_size]``.
            seq_ctx_list (list[SequenceContext]): Aligned per-MB sequence contexts.
            position_embeddings_list (list[tuple[torch.Tensor, torch.Tensor]]):
                Aligned per-MB dense rope ``(cos, sin)`` for DSA sliding-window heads.
            position_embeddings_compressed_list (list[tuple[torch.Tensor, torch.Tensor] | None]):
                Aligned per-MB compressed rope ``(cos, sin)`` for the
                Indexer; ``None`` slots when this layer's ``compress_ratio != 4``.
            input_ids_list (list[torch.Tensor] | None): Aligned per-MB
                ``input_ids`` for :class:`HashRouter`; ``None`` for score-routed layers.

        Returns:
            tuple[torch.Tensor, ...]: Flat ``3 * N``-tuple
                ``(h0, ..., h_{N-1}, rl0, ..., rl_{N-1}, rw0, ..., rw_{N-1})``,
                same layout as :meth:`MoEDecoderLayer._micro_batch_forward`.
        """
        n = len(hidden_states_list)
        # Pad input_ids alignment so the zip below stays straight when the
        # caller (score-routed model) doesn't pass any. Mirrors
        # :meth:`MoEDecoderLayer._micro_batch_forward`'s input_ids_iter pattern.
        if input_ids_list is None:
            input_ids_iter: list[torch.Tensor | None] = [None] * n
        else:
            input_ids_iter = list(input_ids_list)

        # Phase A — per-MB attn block + HC-pre-FFN + ffn_pre_compute + async dispatch_preprocess.
        state_list: list[V4FFNState] = []
        pre_dispatched_list: list[Any] = []
        for hs, sc, pe, pec, ids in zip(
            hidden_states_list,
            seq_ctx_list,
            position_embeddings_list,
            position_embeddings_compressed_list,
            input_ids_iter,
        ):
            collapsed, state = self._forward_pre_ffn_dispatch(
                hs,
                position_embeddings=pe,
                position_embeddings_compressed=pec,
                seq_ctx=sc,
                input_ids=ids,
            )
            pre_dispatched = self.dispatcher.dispatch_preprocess(
                hidden_states=collapsed,
                topk_ids=state.router_results["topk_ids"],
                async_op=True,
            )
            state_list.append(state)
            pre_dispatched_list.append(pre_dispatched)

        # Phase B1 — per-MB dispatch + dispatch_post + experts + combine_pre (all async).
        # Combine is deliberately deferred to Phase B2 so the comm stream
        # sees all dispatches back-to-back; mirrors :class:`MoEDecoderLayer`
        # at ``moe_decoder_layer.py:570-603``. The dispatcher's
        # forward_finished_event chain lets ``_comm_stream`` overlap across
        # MBs without CPU sync (modulo the TorchAll2AllDispatcher
        # ``.tolist()`` block in ``_dispatch`` — a PyTorch NCCL-binding
        # constraint; DeepEP backend is fully CPU-non-blocking and wins the
        # actual overlap here).
        dispatched_list: list[Any] = []
        post_dispatched_list: list[Any] = []
        pre_combined_list: list[Any] = []
        for i in range(n):
            state = state_list[i]
            dispatched = self.dispatcher.dispatch(
                pre_dispatched=pre_dispatched_list[i],
                topk_weights=state.router_results["topk_weights"],
                decoding=False,
                async_op=True,
            )
            post_dispatched = self.dispatcher.dispatch_postprocess(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched,
                async_op=True,
            )
            experts_out = self.experts(
                post_dispatched["hidden_states"],
                post_dispatched["tokens_per_expert"],
                decoding=False,
            )
            pre_combined = self.dispatcher.combine_preprocess(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                decoding=False,
                async_op=True,
            )
            dispatched_list.append(dispatched)
            post_dispatched_list.append(post_dispatched)
            pre_combined_list.append(pre_combined)

        # Phase B2 — per-MB combine (async). Issued back-to-back on the comm
        # stream after every MB's experts + combine_pre is in flight, so the
        # backward queue has ``C.bwd`` ops contiguous (see docstring).
        combined_list: list[Any] = []
        for i in range(n):
            combined = self.dispatcher.combine(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                pre_combined=pre_combined_list[i],
                decoding=False,
                async_op=True,
            )
            combined_list.append(combined)

        # Phase C — per-MB combine_postprocess + FFN-post + HC-post-FFN.
        hidden_states_out_list: list[torch.Tensor] = []
        router_logits_list: list[torch.Tensor] = []
        router_weights_list: list[torch.Tensor] = []
        for i in range(n):
            post_combined = self.dispatcher.combine_postprocess(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                pre_combined=pre_combined_list[i],
                combined=combined_list[i],
                async_op=True,
            )
            h_out, r_logits, r_weights = self._forward_post_ffn_combine(
                post_combined["hidden_states"],
                state_list[i],
            )
            hidden_states_out_list.append(h_out)
            router_logits_list.append(r_logits)
            router_weights_list.append(r_weights)

        # Flat tuple matches MoEDecoderLayer._micro_batch_forward return contract
        # so callers can unpack ``result[:n] / result[n:2n] / result[2n:3n]``
        # uniformly across layer flavours.
        return tuple(hidden_states_out_list + router_logits_list + router_weights_list)

    # ───────── compile-target sub-graphs ─────────

    def _attn_compute(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        # Compile-friendly: pre-norm + DSA. Pure tensor ops, no all2all, no
        # mutable state. ``DSA.forward`` is itself a compile target so this
        # graph effectively brackets the layernorm + the o-proj epilogue
        # around an opaque DSA call.
        h = self.input_layernorm(x)
        dsa = cast(DeepSeekSparseAttention, self.self_attn)
        attn = dsa(
            hidden_states=h,
            position_embeddings=position_embeddings,
            position_embeddings_compressed=position_embeddings_compressed,
            seq_ctx=seq_ctx,
        )
        return attn["projected_output"]

    def _ffn_pre_compute(
        self,
        x: torch.Tensor,
        rollout_routed_experts: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RouterResults]:
        # Compile-friendly: post-attn-norm + gate. Fuses RMSNorm with the
        # gate projection / softmax / top-k pick.
        h = self.post_attention_layernorm(x)
        router_results: RouterResults = self.gate(h, rollout_routed_experts, input_ids=input_ids)
        return h, router_results

    def _ffn_post_compute(
        self,
        combined_hidden_states: torch.Tensor,
        h_normed: torch.Tensor,
    ) -> torch.Tensor:
        # Compile-friendly: combined + shared experts + scalar scale. HC owns
        # the residual add (it happens outside this method, in ``forward``),
        # so we deliberately skip it here — unlike the parent's
        # ``MoEDecoderLayer._post_moe_forward`` which folds the residual in.
        if self.n_shared_experts > 0:
            shared_out = self._shared_experts_forward(h_normed)
            combined_hidden_states = combined_hidden_states + shared_out
        return combined_hidden_states * self.hidden_factor

    def _shared_experts_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.shared_experts is not None
        shared_out = self.shared_experts(hidden_states)
        if self.with_shared_expert_gate:
            assert self.shared_expert_gate is not None
            shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return shared_out

    # ───────── private orchestrator (FFN dispatch chain) ─────────

    def _ffn_compute(
        self,
        x: torch.Tensor,
        seq_ctx: SequenceContext,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RouterResults]:
        # Eager orchestrator that brackets compile-friendly sub-graphs around
        # the deepep dispatcher chain. The split exists so dynamo never traces
        # across the dispatcher boundary — its data-dependent post-all2all
        # token count would either bake into inductor codegen (specialised on
        # the first batch's routing) or, with unbacked symints, crash
        # inductor's range heuristics. See ``V4_EP_COMPILE_CFG`` comment.
        if seq_ctx.rollout_routed_experts is not None and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]
        else:
            rollout_routed_experts = None

        h, router_results = self._ffn_pre_compute(x, rollout_routed_experts, input_ids)

        origin_shape = h.shape
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=h.view(-1, h.shape[-1]),
            topk_ids=router_results["topk_ids"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=router_results["topk_weights"],
            decoding=False,
        )
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            decoding=False,
        )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            decoding=False,
        )
        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        combined_hidden_states = post_combined["hidden_states"].view(*origin_shape)

        ffn_out = self._ffn_post_compute(combined_hidden_states, h)
        return ffn_out, router_results

    # ───────── hc_mult == 1 degenerate path ─────────

    def _plain_residual_forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hc_mult==1 carries no HC mixing information. The input arrives as
        # ``[1, S, 1, D]`` (the model-level HC expand is unconditional); we
        # squeeze the singleton hc axis, run the plain pre-norm residual
        # pattern, then re-add the axis so downstream code stays
        # rank-invariant.
        x_single = x.squeeze(-2)
        attn_out = self._attn_compute(x_single, position_embeddings, position_embeddings_compressed, seq_ctx)
        x_single = x_single + attn_out
        ffn_out, router_results = self._ffn_compute(x_single, seq_ctx, input_ids)
        x_single = x_single + ffn_out
        return (
            x_single.unsqueeze(-2),
            router_results["logits"],
            router_results["router_weights"],
        )
