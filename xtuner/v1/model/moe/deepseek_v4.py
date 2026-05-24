# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Portions of this file (class structure, parameter shapes, HF key mapping) are
# adapted from DeepSeek-V4-Flash `inference/model.py` (`Transformer`, `Block`,
# `MTPBlock`, `ParallelHead`), Copyright (c) DeepSeek-AI, released under the
# MIT License.
# Upstream reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# Local cache: .dev_scripts/deepseek_v4_reference/model.py
#
# The training path retained here strips inference-time machinery (kv_cache,
# block_table, FP4/FP8 quant, TileLang kernels, tensor-parallel collectives)
# and substitutes XTuner's varlen-packed primitives for the V4 reference's
# fixed-batch tensors.
# ============================================================================
"""DeepSeekV4 model glue: ties DSA attention, hash routing, NoAux sqrt-softplus
routing, dual rope and Hyper-Connections together into a working
:class:`MoEConfig` / :class:`MoE` pair for DeepSeek-V4-Flash."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn
from pydantic import Field
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self, override

from transformers import AutoConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import HashRouterConfig, NoAuxRouterConfig, RouterResults
from xtuner.v1.module.attention.dsa import DeepSeekSparseAttention, DSAConfig
from xtuner.v1.module.decoder_layer.hc_block import HCWrapperConfig, _unshard_hc_params, hc_post, hc_pre
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEBlock,
    MoEDecoderLayer,
    MoEGate,
    MoEMLP,
)
from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rms_norm import RMSNorm
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.utils import get_logger

from xtuner.v1.model.base import HFSaveCfg, TorchCompileOption

from .moe import MOE_EP_COMPILE_CFG, MOE_NON_EP_COMPILE_CFG, BalancingLossConfig, MoE, MoEConfig, ZLossConfig


logger = get_logger()


# V4 compile strategy — mirrors the parent ``MoEDecoderLayer`` pattern of
# splitting the layer forward into ``compile-friendly compute subs`` +
# ``eager dispatcher orchestration``:
#
#   * ``hc_pre`` / ``hc_post`` (top-level fns in ``hc_block.py``) carry the HC
#     residual-mixing matmul + sinkhorn + RMS rescale. Decorated with
#     ``@maybe_compile``; entered here so the runtime compile pass enables them.
#   * ``V4DecoderLayer._attn_compute`` runs ``input_layernorm`` + DSA. Pure
#     compute (no dispatcher), safe in both EP and non-EP. Required the
#     ``xtuner.v1.utils.compile._patch_sympy_mod_eval_negative_subs`` upstream
#     workaround so inductor's coalescing analysis on the Indexer's symbolic
#     Mod expressions doesn't crash under EP's dynamic seq_ctx symbols.
#   * ``V4DecoderLayer._ffn_pre_compute`` runs ``post_attention_layernorm`` +
#     gate (norm+softmax fused into one kernel — the ``vectorized_add`` /
#     ``FillFunctor`` storm in the trace came from these tiny ops being eager).
#   * ``V4DecoderLayer._ffn_post_compute`` runs ``+ shared_experts`` and
#     ``* hidden_factor`` (HC owns the final residual add).
#   * ``DeepSeekSparseAttention.forward`` is the V4 attention path itself,
#     wrapped as a separate compile target inside ``_attn_compute``.
#   * Parent's ``MoEBlock.forward`` covers the expert GEMM (already covered
#     through ``MOE_(NON_)EP_COMPILE_CFG``).
#
# ``V4DecoderLayer.forward`` and ``V4DecoderLayer._ffn_compute`` are the
# orchestrators; they MUST stay eager because ``_ffn_compute`` enters the
# deepep dispatcher whose ``moe::permute/unpermute`` fakes report a
# data-dependent output dim that inductor can't reason about (either
# specialises on the first batch's routing or trips its range heuristics
# with unbacked symints).
# ``dynamic=True`` makes dynamo trace with symbolic shapes from the first
# pass instead of specialising on concrete sizes — critical here because
# ``intra_layer_micro_batch=2`` plus packed varlen data feeds two distinct
# ``seq_ctx`` shapes into every step, and the inductor autotune cache
# specialises per shape variant; without ``dynamic=True`` each new
# (cu_seq_lens layout, total_tokens) combination triggers a fresh autotune
# search (visible as 30-70 s step-time spikes between 5 s cache hits).
#
# ``coordinate_descent_tuning`` and ``shape_padding`` were tried but
# multiplied first-compile cost (steps 5-13 oscillated 4-70 s while the
# tuner searched per shape variant); they are net-negative when the workload
# has shape diversity. ``epilogue_fusion`` is cheap (no extra autotune) and
# enables matmul-epilogue fusion in the hc_pre / hc_post / attn_block
# matmul-dominant graphs, which is where most launch-storm reduction lives.
_HEAVY_INDUCTOR_OPTIONS: dict[str, int | bool | str] = {
    "epilogue_fusion": True,
    "triton.unique_kernel_names": True,
}
_HEAVY = TorchCompileOption(fullgraph=False, dynamic=True, options=_HEAVY_INDUCTOR_OPTIONS)
_LITE = TorchCompileOption(fullgraph=False, dynamic=True)

# V4-specific compile targets that are safe under both EP and non-EP. Every
# entry here is a pure-Tensor sub-graph: no MoE dispatcher (all2all) callbacks,
# no DTensor unshard inside the traced region, no data-dependent control flow.
# These get layered on top of the parent's MoE compile cfgs.
_V4_LAYER_TARGETS: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.decoder_layer.hc_block.hc_pre": _HEAVY,
    # ``hc_post`` is now an eager dispatcher: the default bf16 path calls the
    # ``xtuner::hc_post_fwd`` Triton custom op (self-optimized, opaque to
    # compile), and only the ``_HC_HF_PARITY`` fp32 fallback
    # (``_hc_post_eager``) benefits from inductor fusion — so we compile that
    # one instead of the dispatcher.
    "xtuner.v1.module.decoder_layer.hc_block._hc_post_eager": _HEAVY,
    "xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer._attn_compute": _HEAVY,
    "xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer._ffn_pre_compute": _LITE,
    "xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer._ffn_post_compute": _LITE,
    "xtuner.v1.model.moe.deepseek_v4.DeepSeekV4._hc_head_reduce_compute": _LITE,
    "xtuner.v1.module.attention.dsa.DeepSeekSparseAttention.forward": _HEAVY,
    # The compressor's scatter + softmax + sum + RMSNorm chain is exactly the
    # ~50-elementwise-op storm that showed up in the rank0 trace under EP. The
    # ``int(cu_seq_lens_out[-1].item())`` sync at the head of forward breaks
    # the graph once; ``fullgraph=False`` (in ``_LITE``) accepts that break
    # and still fuses the two halves on either side of it.
    "xtuner.v1.module.attention.kv_compressor.KVCompressor.forward": _LITE,
}

V4_NON_EP_COMPILE_CFG: dict[str, TorchCompileOption] = MOE_NON_EP_COMPILE_CFG | _V4_LAYER_TARGETS

# EP-safe variant: identical V4 layer-internal targets, but built on top of
# ``MOE_EP_COMPILE_CFG`` which already drops ``MoEDecoderLayer.forward``
# (since the full layer forward enters the all2all dispatcher under EP, and
# inductor can't trace the deepep ``moe::permute/unpermute`` fakes).
#
# History: V4_EP_COMPILE_CFG was previously ``{}`` because of a recompute-time
# 130 GiB fp32 allocation that hit "across DSA, hc_pre/post, and shared/expert
# paths". The shared upstream cause was the native Indexer materialising a
# ``[1, S_i, n_heads, T_i]`` fp32 score tensor inside the autograd graph;
# under varlen + dynamic-shape compile + activation-checkpoint recompute, that
# 4-5 GB-per-layer tensor multiplied across shape-variant retraces until it
# evicted everything else. The Indexer is now invoked under
# ``torch.no_grad()`` from a Triton-backed top-k path
# (:mod:`xtuner.v1.module.attention._indexer_topk_triton`), so the fp32 score
# tensor never enters autograd. ``DSAConfig.indexer_backend`` defaults to
# ``"triton"`` to ensure this code path is the one taken.
V4_EP_COMPILE_CFG: dict[str, TorchCompileOption] = MOE_EP_COMPILE_CFG | _V4_LAYER_TARGETS


# V4-Flash ships its `compress_ratios` as a vector of length `num_hidden_layers + 1`
# (43 + 1 = 44): indices 0..42 cover the main transformer stack, index 43 is the
# trailing MTP layer. We keep the default factory consistent with the released
# config.json so meta-device construction works without an HF round-trip.
_DEFAULT_COMPRESS_RATIOS: list[int] = [0, 0] + [4, 128] * 20 + [4, 0]


def _build_compressed_position_embeddings(
    rotary_emb,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    # DualRotaryEmbedding emits two rope bases (`rope_theta` and
    # `compress_rope_theta`) via the `use_compressed` keyword; the regular base
    # is consumed by sliding-window heads and the compressed base by the
    # Indexer / compressor heads. We materialise both up-front in DeepSeekV4
    # forward so each layer can pick the one matching its `compress_ratio`
    # without re-running the rope kernel.
    if hasattr(rotary_emb, "inv_freq_compressed"):
        return rotary_emb(hidden_states, position_ids, use_compressed=True)
    return None


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
            self.shared_expert_gate = (
                build_linear(hidden_size, 1, bias=False) if with_shared_expert_gate else None
            )
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
        attn_fn, attn_scale, attn_base = _unshard_hc_params(
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
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
        attn_out = self._attn_compute(
            x_reduced, position_embeddings, position_embeddings_compressed, seq_ctx
        )
        hidden_states = hc_post(attn_out, residual, post_a, comb_a)

        # ─── HC-wrapped FFN (MoE) ───
        ffn_fn, ffn_scale, ffn_base = _unshard_hc_params(
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
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
        attn_fn, attn_scale, attn_base = _unshard_hc_params(
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
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
        attn_out = self._attn_compute(
            x_reduced, position_embeddings, position_embeddings_compressed, seq_ctx
        )
        hidden_states = hc_post(attn_out, residual, post_a, comb_a)

        # ─── HC-pre for FFN (mirrors forward()) ───
        ffn_fn, ffn_scale, ffn_base = _unshard_hc_params(
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
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
        if (
            seq_ctx.rollout_routed_experts is not None
            and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]
        ):
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
        if (
            seq_ctx.rollout_routed_experts is not None
            and self.layer_idx < seq_ctx.rollout_routed_experts.shape[1]
        ):
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
        attn_out = self._attn_compute(
            x_single, position_embeddings, position_embeddings_compressed, seq_ctx
        )
        x_single = x_single + attn_out
        ffn_out, router_results = self._ffn_compute(x_single, seq_ctx, input_ids)
        x_single = x_single + ffn_out
        return (
            x_single.unsqueeze(-2),
            router_results["logits"],
            router_results["router_weights"],
        )


class DeepSeekV4Config(MoEConfig):
    """Configuration for DeepSeek-V4-Flash.

    Mirrors :class:`DeepSeekV3Config` but uses :class:`DSAConfig` (sparse
    attention with grouped O-LoRA + attention sink), ``"sqrtsoftplus"`` NoAux
    scoring, hash routing for the first ``num_hash_layers`` layers, dual rope
    (``rope_theta`` + ``compress_rope_theta``), and Hyper-Connections
    wrappers around every decoder layer.
    """

    vocab_size: int = 129280
    max_position_embeddings: int = 1048576
    pad_token_id: int | None = None
    eos_token_id: int = 1
    bos_token_id: int = 0
    num_hidden_layers: int = 43
    # V4 has no dense-replace prefix: every layer in the main stack is MoE; the
    # `first_k_dense_replace` mechanism inherited from V3 stays at 0.
    first_k_dense_replace: int = 0
    num_hash_layers: int = 3
    hidden_size: int = 4096
    # Unused by V4 (MoE expert dim is `moe_intermediate_size`); kept for
    # parent-config compatibility — MoEConfig.intermediate_size is non-Optional.
    intermediate_size: int = 0
    moe_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    # SwiGLU clamp limit applied to expert intermediate activations
    # (DeepSeekV4 inference `Expert.forward` clamps with min=-limit, max=limit).
    swiglu_limit: float = 10.0
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(
            rope_theta=10000.0,
            rope_type="yarn",
            beta_fast=32,
            beta_slow=1,
            factor=16,
            original_max_position_embeddings=65536,
            compress_rope_theta=160000.0,
            compress_ratios=list(_DEFAULT_COMPRESS_RATIOS),
        )
    )
    hidden_act: str = "silu"
    attention: DSAConfig = Field(
        default_factory=lambda: DSAConfig(
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=1024,
            o_lora_rank=1024,
            o_groups=8,
            sliding_window=128,
            use_attn_sink=True,
            index_head_dim=128,
            index_n_heads=64,
            index_topk=512,
            rms_norm_eps=1e-6,
        )
    )
    tie_word_embeddings: bool = False
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    hidden_factor: float = 1.0
    router: NoAuxRouterConfig = Field(
        default_factory=lambda: NoAuxRouterConfig(
            n_group=8,
            topk_group=4,
            scoring_func="sqrtsoftplus",
            norm_topk_prob=True,
            router_scaling_factor=1.5,
        )
    )
    hc_cfg: HCWrapperConfig = Field(
        default_factory=lambda: HCWrapperConfig(
            hc_mult=4,
            hc_eps=1e-6,
            hc_sinkhorn_iters=20,
        )
    )
    # V4-Flash does not train an aux balancing loss (the HF config exposes
    # `routed_scaling_factor` but no balancing field, and inference/model.py
    # has no aux-loss path). HashRouter's dummy logits would also be invalid
    # input to BalancingLoss anyway — we keep both nullable losses off by default.
    balancing_loss_cfg: BalancingLossConfig | None = None
    z_loss_cfg: ZLossConfig | None = None
    mtp_config: MTPConfig | None = Field(default_factory=lambda: MTPConfig(num_layers=1, loss_scaling_factor=0.1))
    moe_act_fn_cfg: MoEActFnConfig = Field(
        default_factory=lambda: MoEActFnConfig(
            act_type="clipped_swiglu",
            # V4's `swiglu_limit=10` is symmetric, matching XTuner's
            # `native_clipped_swiglu(limit=...)` clamp range.
            clip_alpha=1.0,
            clip_limit=10.0,
        )
    )
    # HC mixing parameters live in fp32 (the 20-iter Sinkhorn is bf16-unstable).
    # We mark their HF keys here so :meth:`XTunerBaseModel._fully_shard` keeps
    # them as Replicate-on-world DTensors (added to FSDP's ``ignored_params``)
    # instead of sharding them like every other parameter. Without this the
    # per-forward ``_unshard_hc_params`` ``.full_tensor()`` call does an
    # allgather every forward (~100 KB × 86 calls/step, plus syncs); the
    # ignored path turns the call into a no-op on the replicated DTensor.
    #
    # Pattern covers all 9 HC parameters: per-layer
    # ``layers.<L>.hc_(attn|ffn)_(fn|base|scale)`` and model-top-level
    # ``hc_head_(fn|base|scale)``. The trailing ``$`` keeps the match precise
    # (re.search is substring by default).
    hf_save_cfg: HFSaveCfg = Field(
        default_factory=lambda: HFSaveCfg(
            fp32_keys_pattern=[r"hc_(attn|ffn|head)_(fn|base|scale)$"],
        )
    )

    def build(self) -> "DeepSeekV4":
        return DeepSeekV4(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        """Construct a :class:`DeepSeekV4Config` from the HF release directory.

        Args:
            hf_path (str | Path): Path containing ``config.json``.

        Returns:
            DeepSeekV4Config: XTuner-side config mirroring the HF fields.
        """
        # transformers<5.10 does not ship a `DeepseekV4Config`. We try AutoConfig
        # first (with trust_remote_code) so a HF release shipping `*.py` modeling
        # files just works; if that fails we fall back to reading `config.json`
        # directly into a SimpleNamespace, since every field DeepSeekV4Config
        # consumes from `cfg` is a plain attribute lookup with no Python-side
        # validation.
        try:
            cfg = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        except (KeyError, ValueError):
            config_json_path = Path(hf_path) / "config.json"
            with open(config_json_path, encoding="utf-8") as f:
                cfg = SimpleNamespace(**json.load(f))
        assert getattr(cfg, "model_type", None) == "deepseek_v4", (
            f"Expected `model_type == 'deepseek_v4'`, got {getattr(cfg, 'model_type', None)!r}"
        )

        default_rope_params = (
            cls.model_fields["rope_parameters_cfg"].get_default(call_default_factory=True).model_dump()
        )
        rope_parameters_cfg = RopeParametersConfig.from_hf_config(cfg, default_rope_params)
        assert rope_parameters_cfg is not None, "DeepSeek-V4 HF config must define rope parameters"

        # ``num_hash_layers`` field handling. transformers <5.9 stored it directly
        # on the config. transformers >=5.9.0 ``DeepseekV4Config.__post_init__``
        # consumes the legacy ``num_hash_layers`` kwarg from ``config.json`` and
        # converts it into the per-layer ``mlp_layer_types`` list
        # (``["hash_moe"] * num_hash_layers + ["moe"] * rest``), then drops the
        # scalar attribute. We recover the count by counting the leading
        # ``"hash_moe"`` entries — V4 always puts hash-routed layers at the
        # front so this matches ``DeepSeekV4._should_compute_aux_loss``'s
        # ``layer_idx < num_hash_layers`` semantics.
        num_hash_layers = getattr(cfg, "num_hash_layers", None)
        if num_hash_layers is None:
            mlp_layer_types = getattr(cfg, "mlp_layer_types", None) or []
            num_hash_layers = 0
            for t in mlp_layer_types:
                if t != "hash_moe":
                    break
                num_hash_layers += 1

        attention = DSAConfig(
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            o_lora_rank=cfg.o_lora_rank,
            o_groups=cfg.o_groups,
            sliding_window=cfg.sliding_window,
            use_attn_sink=True,
            index_head_dim=cfg.index_head_dim,
            index_n_heads=cfg.index_n_heads,
            index_topk=cfg.index_topk,
            rms_norm_eps=cfg.rms_norm_eps,
        )

        router = NoAuxRouterConfig(
            # V4 inference uses 8 groups / top-4 (model.py:Gate.__init__);
            # HF config doesn't expose them, so we hard-pin the documented values.
            n_group=8,
            topk_group=4,
            scoring_func=cfg.scoring_func,
            norm_topk_prob=cfg.norm_topk_prob,
            router_scaling_factor=cfg.routed_scaling_factor,
        )

        hc_cfg = HCWrapperConfig(
            hc_mult=cfg.hc_mult,
            hc_eps=cfg.hc_eps,
            hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        )

        num_nextn = getattr(cfg, "num_nextn_predict_layers", 0) or 0
        mtp_config: MTPConfig | None = (
            MTPConfig(num_layers=num_nextn, loss_scaling_factor=0.1) if num_nextn > 0 else None
        )

        moe_act_fn_cfg = MoEActFnConfig(
            act_type="clipped_swiglu",
            clip_alpha=1.0,
            clip_limit=float(getattr(cfg, "swiglu_limit", 10.0)),
        )

        return cls(
            vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings,
            pad_token_id=getattr(cfg, "pad_token_id", None),
            eos_token_id=cfg.eos_token_id,
            bos_token_id=getattr(cfg, "bos_token_id", 0),
            num_hidden_layers=cfg.num_hidden_layers,
            num_hash_layers=num_hash_layers,
            hidden_size=cfg.hidden_size,
            moe_intermediate_size=cfg.moe_intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            swiglu_limit=float(getattr(cfg, "swiglu_limit", 10.0)),
            rope_parameters_cfg=rope_parameters_cfg,
            hidden_act=cfg.hidden_act,
            attention=attention,
            tie_word_embeddings=cfg.tie_word_embeddings,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            router=router,
            hc_cfg=hc_cfg,
            mtp_config=mtp_config,
            moe_act_fn_cfg=moe_act_fn_cfg,
        )

    @property
    def hf_config(self):
        # V4 has no transformers-built-in config class in older releases; let `save_hf`
        # fall back to copying the original `*.py` files from `self._hf_path`.
        return None


class DeepSeekV4(MoE):
    """DeepSeek-V4-Flash transformer for training.

    Departures from the base :class:`MoE` worth flagging:

    * **Per-layer router/attention dispatch** — the first ``num_hash_layers``
      MoE gates use :class:`HashRouter`, the rest use sqrt-softplus
      :class:`NoAuxRouter`; every layer's attention is
      :class:`DeepSeekSparseAttention` with a per-layer
      ``compress_ratio in {0, 4, 128}`` pulled from
      ``rope_parameters_cfg.compress_ratios``.
    * **Hyper-Connections** — every decoder layer is a :class:`V4DecoderLayer`
      which inlines the HC residual mix, so the model carries ``hc_mult`` residual streams.
      The embedding is expanded to ``[B, S, hc_mult, D]`` and reduced back to
      ``[B, S, D]`` via a learned ``hc_head_*`` triple before the final norm.
    * **Forward override** — the standard :class:`MoE._forward` assumes
      ``decoder_layer(...)`` returns ``(hidden, logits, weights)`` and treats
      ``hidden`` as ``[B, S, D]``. V4 layers operate on ``[B, S, hc_mult, D]``
      and need an extra ``position_embeddings_compressed`` rope tuple, so
      ``_forward`` is replaced here.

    Args:
        config (DeepSeekV4Config): Model configuration.
    """

    config: DeepSeekV4Config

    def __init__(self, config: DeepSeekV4Config) -> None:
        self._hc_mult = config.hc_cfg.hc_mult
        super().__init__(config)
        # `hc_head_*` reduces `[B, S, hc_mult, D]` back to `[B, S, D]` before the
        # final RMSNorm + lm_head. Shape matches V4 reference ParallelHead/Transformer
        # (model.py L797): `[hc_mult, hc_mult * D]`. Stored fp32 for sinkhorn stability.
        # Registered AFTER super().__init__ — registering earlier would be wiped
        # by MoE.__init__ → nn.Module.__init__ resetting `_parameters = {}`. We
        # then re-run `_init_load_spec` so the freshly-registered params land in
        # `load_spec_mapping` (the parent's first call built it without them).
        hc_mult = config.hc_cfg.hc_mult
        hc_dim = hc_mult * config.hidden_size
        fp32 = torch.float32
        self.hc_head_fn = nn.Parameter(torch.zeros(hc_mult, hc_dim, dtype=fp32))
        self.hc_head_base = nn.Parameter(torch.zeros(hc_mult, dtype=fp32))
        # V4 reference uses a scalar scale for hc_head (model.py L799); we keep
        # the same shape for HF key parity.
        self.hc_head_scale = nn.Parameter(torch.zeros(1, dtype=fp32))
        self._init_load_spec()

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        # See V4_NON_EP_COMPILE_CFG / V4_EP_COMPILE_CFG at module level for the
        # rationale on which V4-specific class forwards are added on top of the
        # parent MoE config. `@property` mirrors the base / parent decorator —
        # `BaseModel._resolve_compile_cfg` reads `self.default_compile_cfg`
        # without parens.
        if self.config.ep_size > 1:
            return V4_EP_COMPILE_CFG
        return V4_NON_EP_COMPILE_CFG

    @override
    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        # The parent MoE.build_layers does its own per-layer wiring for
        # MLA/MHA/GatedDeltaNet attention plus score-only routing. V4 needs per-layer
        # `compress_ratio`, per-layer router type, and an HC wrapper — wholly
        # different control flow. We re-implement here rather than threading the
        # decisions into the parent loop because every per-layer branch in the parent
        # would otherwise need to know about DSA / hash / HC.
        v4_cfg = cast(DeepSeekV4Config, config)
        compress_ratios = v4_cfg.rope_parameters_cfg.compress_ratios
        assert compress_ratios is not None and len(compress_ratios) >= v4_cfg.num_hidden_layers, (
            f"compress_ratios (len={len(compress_ratios) if compress_ratios else 0}) must cover "
            f"all {v4_cfg.num_hidden_layers} hidden layers"
        )

        layers = nn.ModuleDict()
        for layer_idx in range(v4_cfg.num_hidden_layers):
            layers[str(layer_idx)] = self._build_one_layer(v4_cfg, layer_idx, compress_ratios[layer_idx])
        return layers

    @override
    def build_mtp_block(self, config: MoEConfig):
        # The V4 MTP block has its own HC head + e_proj/h_proj/enorm/hnorm chain
        # (model.py:MTPBlock L738-766) and uses the same per-layer DSA + hash/score
        # routing dispatch as the main stack. Reusing the parent's MTPBlock builder
        # would route through MoEDecoderLayer with the default attention_config.build
        # path, which crashes for DSAConfig (no compress_ratio). PR9 leaves MTP as a
        # TODO follow-up: the structural pieces (``V4DecoderLayer`` reusable for
        # the MTP body) are in place, but the MTP-specific
        # e_proj / h_proj / enorm / hnorm / hc_head_* glue and its
        # ``compress_ratios[-1]``-driven attention mode need their own wiring pass.
        if config.mtp_config is not None:
            logger.warning(
                "DeepSeekV4: mtp_config is set but MTP forward + parameter wiring is not "
                "implemented in PR9. The model will build without MTP and skip MTP loss; "
                "follow-up PR will add the V4 MTPBlock with HC + e_proj/h_proj. "
                f"(num_layers={config.mtp_config.num_layers})"
            )
        return None

    @override
    def build_embeddings(self, config: MoEConfig):
        # We rely on the parent's plain `nn.Embedding` and broadcast to `hc_mult`
        # streams in `_forward`. Keeping the embedding module untouched means HF
        # weight loading for `embed.weight` works without any layout massage.
        return nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def to_hf_key_list(self, key: str) -> list[str]:
        """Translate an XTuner-side parameter name to its HF counterpart(s).

        DeepSeek-V4 HF keys carry no ``model.`` prefix and no ``mlp.`` infix; experts
        are named ``w1/w2/w3``; HC parameters sit at ``layers.N.hc_*`` (top-level on
        the wrapper); MTP layers live at ``mtp.M.*``. Fused expert weights expand
        to one HF key per expert.

        Args:
            key (str): XTuner-side parameter name.

        Returns:
            list[str]: One or more HF parameter names; lists with multiple entries
            arise when an XTuner-side fused tensor maps to per-expert HF tensors.
        """
        # Top-level HC head parameters and final norm stay as-is.
        if key in {"hc_head_fn", "hc_head_base", "hc_head_scale"}:
            return [key]
        if key == "norm.weight":
            return ["norm.weight"]
        if key == "embed_tokens.weight":
            return ["embed.weight"]
        if key == "lm_head.weight":
            return ["head.weight"]

        n_routed_experts = self.config.n_routed_experts

        # Layers prefix — strip XTuner's wrapper layout (`.inner.<...>`) to match HF's
        # flat layout, then translate XTuner module names to HF names.
        m = re.match(r"^layers\.(\d+)\.(.+)$", key)
        if m:
            layer_idx, tail = m.group(1), m.group(2)
            return [
                f"layers.{layer_idx}.{hf_tail}"
                for hf_tail in self._translate_layer_tail(tail, layer_idx, n_routed_experts)
            ]

        # MTP block — XTuner stores it under `mtp_block.layers.M.<...>`. HF flat
        # layout is `mtp.M.<...>`. The MTP layer body has the same structure as
        # a main-stack layer plus the extra e_proj/h_proj/enorm/hnorm/norm fields
        # and its own hc_head_*.
        m = re.match(r"^mtp_block\.layers\.(\d+)\.(.+)$", key)
        if m:
            mtp_idx, tail = m.group(1), m.group(2)
            return [f"mtp.{mtp_idx}.{hf_tail}" for hf_tail in self._translate_mtp_tail(tail, n_routed_experts)]

        return [key]

    @override
    def _should_compute_aux_loss(self, layer_idx: int) -> bool:
        # Hash-routed layers emit a `[1]` dummy logits placeholder; feeding that
        # to AuxLossContext.accumulate's `index_select(0, nonpad_indices)` would
        # raise an out-of-range error. Skip the aux-loss accumulation for those
        # layers entirely. Score-routed layers (idx >= num_hash_layers) keep the
        # default behaviour.
        return layer_idx >= self.config.num_hash_layers

    @override
    def _forward(self, seq_ctx, loss_ctx, return_router_logits: bool = False):  # type: ignore[override]
        # We replace MoE._forward outright because the V4 forward graph has three
        # invariants that don't fit the parent:
        #   1) Decoder layers operate on `[B, S, hc_mult, D]`, not `[B, S, D]`.
        #   2) Each layer needs `position_embeddings_compressed` in addition to the
        #      dense rope.
        #   3) The final norm runs *after* an hc_head reduction back to `[B, S, D]`.
        # Note: MTP forward is omitted in this PR; the V4 MTP block has its own
        # HC head + e/h proj + enorm/hnorm chain that must be wired separately
        # (tracked as PR9 follow-up).
        from xtuner.v1.loss import LMHeadLossContext
        from xtuner.v1.model.utils import ModelForwardExtraLogInfo

        from .moe import MoEModelOutputs

        assert seq_ctx.position_ids is not None
        assert seq_ctx.input_ids is not None, "DeepSeekV4 requires input_ids (HashRouter consumes them)"
        hidden_states = self.embed_tokens(seq_ctx.input_ids)
        # Dense rope (sliding-window heads) and compressed rope (Indexer) are both
        # produced from the same DualRotaryEmbedding instance; we pre-compute both
        # so each layer can pick the matching pair without branching on layer type.
        position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids, use_compressed=False)
        position_embeddings_compressed = _build_compressed_position_embeddings(
            self.rotary_emb, hidden_states, seq_ctx.position_ids
        )

        # Expand `[B, S, D]` → `[B, S, hc_mult, D]`. `.contiguous()` is essential
        # because downstream HC ops (`flatten(2)`, `.view(shape)`) assume a dense
        # layout; the expand-without-copy would alias.
        hidden_states = hidden_states.unsqueeze(-2).expand(-1, -1, self._hc_mult, -1).contiguous()

        output: dict = {}
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        keep_router = self.config.return_router_results or return_router_logits
        if keep_router:
            output["router_logits"] = {}
            output["router_weights"] = {}
        else:
            output["router_logits"] = None
            output["router_weights"] = None

        balancing_ctx, z_ctx = self._extract_aux_loss_ctx(loss_ctx)
        nonpad_indices = torch.nonzero(seq_ctx.mask, as_tuple=True)[1]
        non_pad_token = nonpad_indices.numel()
        num_tokens_global, z_world_size = self._z_loss_dist_token_count(z_ctx, non_pad_token, seq_ctx.mask.device)

        offload_active = int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1
        for idx, decoder_layer in self.layers.items():
            v4_layer = cast(V4DecoderLayer, decoder_layer)
            # Mirror the parent's per-layer activation offload window: with the HC-expanded
            # `[B, S, hc_mult, D]` activation (4× the parent's `[B, S, D]`), staging each
            # layer's residual on CPU is the main lever for fitting 256-expert layers on
            # bf16 with full pack_max_length. Gated on XTUNER_ACTIVATION_OFFLOAD=1.
            if offload_active:
                from xtuner.v1.utils.activation_offload import async_save_on_cpu

                with async_save_on_cpu(
                    h2d_stream=self.offload_stream,
                    d2h_stream=self.offload_stream,
                    block_idx=int(idx),
                    group="text",
                    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
                ):
                    hidden_states, router_logits, router_weights = v4_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        position_embeddings_compressed=position_embeddings_compressed,
                        seq_ctx=seq_ctx,
                        input_ids=seq_ctx.input_ids,
                    )
            else:
                hidden_states, router_logits, router_weights = v4_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    position_embeddings_compressed=position_embeddings_compressed,
                    seq_ctx=seq_ctx,
                    input_ids=seq_ctx.input_ids,
                )
            if keep_router:
                output["router_logits"][f"layer{idx}"] = self._maybe_offload_router(router_logits)
                output["router_weights"][f"layer{idx}"] = self._maybe_offload_router(router_weights)
            if self._should_compute_aux_loss(int(idx)):
                hidden_states = self.aux_loss.accumulate(
                    selected_router_weights=router_weights.index_select(0, nonpad_indices).contiguous().float(),
                    selected_router_logits=router_logits.index_select(0, nonpad_indices).contiguous().float(),
                    hidden_states=hidden_states,
                    balancing_ctx=balancing_ctx,
                    z_ctx=z_ctx,
                    num_tokens_local=non_pad_token,
                    num_tokens_global=num_tokens_global,
                    world_size=z_world_size,
                )
            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        # Reduce `[B, S, hc_mult, D]` → `[B, S, D]` via the model-level hc_head
        # triple, then apply the standard final RMSNorm + lm_head.
        hidden_states = self._hc_head_reduce(hidden_states)
        hidden_states = self.norm(hidden_states)

        lm_loss_ctx = loss_ctx["lm"] if loss_ctx is not None else None
        loss, (logits, extra_info) = self.lm_head(hidden_states, cast(LMHeadLossContext, lm_loss_ctx))  # type: ignore[arg-type]
        output["loss"] = loss
        output["logits"] = logits
        output["extra_info"] = extra_info if extra_info is not None else ModelForwardExtraLogInfo()

        # Hash-routed layers don't accumulate routing stats (see
        # `_should_compute_aux_loss`). When `num_hash_layers >= num_hidden_layers`
        # (legal for sub-stack smoke configs like 2 layers + 3 hash layers from
        # the release config), every layer is hash-routed and aux_loss has nothing
        # to finalize — calling finalize would raise from `_cal_tokens_per_expert`.
        # Skip the call and emit None aux outputs; `internal_metrics.py` already
        # treats `tokens_per_expert_global is None` as "no MoE load this step".
        if self.config.num_hash_layers < self.config.num_hidden_layers:
            balancing_loss, z_loss, tokens_per_expert_global = self.aux_loss.finalize(
                balancing_ctx=balancing_ctx,
                z_ctx=z_ctx,
                non_pad_token=non_pad_token,
            )
            if balancing_loss is not None:
                output["balancing_loss"] = balancing_loss
            if z_loss is not None:
                output["z_loss"] = z_loss
            output["tokens_per_expert_global"] = tokens_per_expert_global
        else:
            output["tokens_per_expert_global"] = None

        if keep_router:
            for layer_name, router_logits_t in output["router_logits"].items():
                output["router_logits"][layer_name] = router_logits_t.detach().unsqueeze(0)

        return MoEModelOutputs(**output)

    @override
    def _micro_batch_forward(  # type: ignore[override]
        self,
        seq_ctx_list,
        loss_ctx_list,
        return_router_logits: bool = False,
    ):
        # V4 needs a fresh micro-batch forward (rather than inheriting MoE's) for the
        # same three reasons `_forward` overrides the parent:
        #   1) `[B, S, hc_mult, D]` HC-expanded activations
        #   2) dual rope (`position_embeddings` + `position_embeddings_compressed`)
        #   3) `_hc_head_reduce` before the final norm
        # When ``config.domino`` is set (and ep_size > 1, no offload), the layer
        # loop below calls ``v4_layer(*hidden_states_list, seq_ctx=ctx_list, ...)``
        # once per layer; ``V4DecoderLayer.forward`` dispatches on the variadic
        # ``*hidden_states`` length to :meth:`V4DecoderLayer._micro_batch_forward`,
        # running the 3-phase Domino wave INSIDE the layer's forward so FSDP2's
        # pre/post-forward hooks bracket the whole pass exactly once.
        from xtuner.v1.loss import LMHeadLossContext
        from xtuner.v1.model.utils import ModelForwardExtraLogInfo

        from .moe import MoEModelOutputs

        if self.config.return_hidden_states:
            raise NotImplementedError("return_hidden_states is not supported in V4 micro-batch forward")
        assert len(seq_ctx_list) == len(loss_ctx_list), "seq_ctx and loss_ctx must have same length"

        n_mb = len(seq_ctx_list)

        # Per-MB: embed → dual rope → HC expand. Each MB stays as its own tensor in
        # the list; we never cat across MBs along the seq dim because `V4DecoderLayer`
        # is called once per MB anyway, and a cat-then-chunk round-trip would only
        # add the same `i.clone()` workaround the parent needs for `async_save_on_cpu`.
        hidden_states_list: list[torch.Tensor] = []
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        position_embeddings_compressed_list: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for seq_ctx in seq_ctx_list:
            assert seq_ctx.position_ids is not None
            assert seq_ctx.input_ids is not None, "DeepSeekV4 requires input_ids (HashRouter consumes them)"
            h = self.embed_tokens(seq_ctx.input_ids)
            pos_emb = self.rotary_emb(h, seq_ctx.position_ids, use_compressed=False)
            pos_emb_compressed = _build_compressed_position_embeddings(self.rotary_emb, h, seq_ctx.position_ids)
            h = h.unsqueeze(-2).expand(-1, -1, self._hc_mult, -1).contiguous()
            hidden_states_list.append(h)
            position_embeddings_list.append(pos_emb)
            position_embeddings_compressed_list.append(pos_emb_compressed)

        # Aux-loss state is computed over the union of all MBs' tokens (matches the
        # parent's behaviour and what the global mean CE in `ce_loss.py` expects).
        balancing_ctx, z_ctx = self._extract_aux_loss_ctx(loss_ctx_list)
        cat_mask = torch.cat([ctx.mask for ctx in seq_ctx_list], dim=1)
        nonpad_indices_cat = torch.nonzero(cat_mask, as_tuple=True)[1]
        non_pad_token = nonpad_indices_cat.numel()
        num_tokens_global, z_world_size = self._z_loss_dist_token_count(z_ctx, non_pad_token, cat_mask.device)

        output: dict = {}
        keep_router = self.config.return_router_results or return_router_logits
        router_logits_per_mb: list[dict[str, torch.Tensor]] = [{} for _ in range(n_mb)] if keep_router else []

        offload_active = int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1
        # ``_micro_batch_forward`` is reached only when the caller passed a list of
        # SequenceContexts (i.e. ``intra_layer_micro_batch >= 2``). That flag exists
        # solely to trigger the layer-internal Domino wave pipeline, which needs
        # an all2all to overlap with expert compute — so ep_size > 1 is implied.
        # ep_size == 1 + intra_layer_micro_batch >= 2 is a user-error config:
        # ``NaiveDispatcher`` at ep=1 doesn't support ``async_op=True`` and will
        # raise inside the dispatcher; we don't paper over that with a sequential
        # fallback because the parent :class:`MoE` doesn't, and the dead branch
        # only adds maintenance surface.
        #
        # Offload composes orthogonally: when active, wrap the whole multi-MB
        # layer call in one ``async_save_on_cpu`` context (matching
        # :class:`MoE`'s parent pattern at ``moe.py:542-552``). The layer is
        # called exactly once per layer, so the offload buffer ring only needs
        # a per-layer ``block_idx``; ``custom_check_fn`` skips the n_mb input
        # tensors via ``data_ptr in [...]``.

        for idx, decoder_layer in self.layers.items():
            v4_layer = cast(V4DecoderLayer, decoder_layer)

            # One layer call per layer carrying all MBs. The variadic
            # ``*hidden_states`` signature routes to
            # :meth:`V4DecoderLayer._micro_batch_forward`, whose 3-phase wave
            # pipeline runs entirely INSIDE the layer's ``forward`` — so FSDP2's
            # pre/post-forward hooks bracket the whole multi-MB pass exactly once
            # (one all-gather, one reshard). Calling the staged halves from out
            # here would bypass those hooks and break param management; see the
            # :meth:`V4DecoderLayer._micro_batch_forward` docstring.
            layer_call_kwargs = dict(
                seq_ctx=seq_ctx_list,
                position_embeddings=position_embeddings_list,
                position_embeddings_compressed=position_embeddings_compressed_list,
                input_ids=[sc.input_ids for sc in seq_ctx_list],
            )
            if offload_active:
                from xtuner.v1.utils.activation_offload import async_save_on_cpu

                with async_save_on_cpu(
                    h2d_stream=self.offload_stream,
                    d2h_stream=self.offload_stream,
                    block_idx=int(idx),
                    group="text",
                    custom_check_fn=lambda x, _hs=hidden_states_list: x.data_ptr()
                    in [h.data_ptr() for h in _hs],
                    prefetch=True,
                    reserve_pin_memory=True,
                ):
                    layer_out = v4_layer(*hidden_states_list, **layer_call_kwargs)
            else:
                layer_out = v4_layer(*hidden_states_list, **layer_call_kwargs)

            # Layer returns flat ``3 * n_mb`` tuple: hidden_states ... router_logits ... router_weights.
            new_hidden_states_list = list(layer_out[:n_mb])
            layer_router_logits = list(layer_out[n_mb : 2 * n_mb])
            layer_router_weights = list(layer_out[2 * n_mb : 3 * n_mb])
            for mb_idx in range(n_mb):
                hidden_states_list[mb_idx] = new_hidden_states_list[mb_idx]

            if keep_router:
                for mb_idx in range(n_mb):
                    router_logits_per_mb[mb_idx][f"layer{idx}"] = self._maybe_offload_router(
                        layer_router_logits[mb_idx]
                    )

            if self._should_compute_aux_loss(int(idx)):
                # Concatenate router stats across MBs so aux_loss sees the same global
                # token set the parent path does. Pin the z-loss carrier to MB0's
                # hidden_states to mirror the parent — `total_loss.backward()` traverses
                # MB0's path exactly once.
                cat_router_weights = torch.cat(layer_router_weights, dim=0)
                cat_router_logits = torch.cat(layer_router_logits, dim=0)
                hidden_states_list[0] = self.aux_loss.accumulate(
                    selected_router_weights=cat_router_weights.index_select(0, nonpad_indices_cat)
                    .contiguous()
                    .float(),
                    selected_router_logits=cat_router_logits.index_select(0, nonpad_indices_cat).contiguous().float(),
                    hidden_states=hidden_states_list[0],
                    balancing_ctx=balancing_ctx,
                    z_ctx=z_ctx,
                    num_tokens_local=non_pad_token,
                    num_tokens_global=num_tokens_global,
                    world_size=z_world_size,
                )

        # MTP omitted to match `_forward`: V4 MTP wiring (HC head + e_proj/h_proj +
        # enorm/hnorm) is the PR9 follow-up. When it lands, mirror the parent's
        # per-MB MTP-loss aggregation here.
        if self.mtp_block is not None:
            raise NotImplementedError(
                "V4 micro-batch forward does not wire MTP yet (same TODO as `_forward`); "
                "see DeepSeekV4.build_mtp_block."
            )

        # HC head reduce + final norm + lm_head: cat once across MBs so lm_head runs
        # as a single GEMM (matches parent's perf).
        cat_hidden_states = torch.cat(hidden_states_list, dim=1)
        cat_hidden_states = self._hc_head_reduce(cat_hidden_states)
        cat_hidden_states = self.norm(cat_hidden_states)

        lm_loss_ctx_list = [loss_ctx_dict["lm"] for loss_ctx_dict in loss_ctx_list]
        cat_loss_ctx = type(lm_loss_ctx_list[0]).cat(lm_loss_ctx_list)
        loss, (logits, extra_info) = self.lm_head(cat_hidden_states, cast(LMHeadLossContext, cat_loss_ctx))

        output["loss"] = loss.sum()
        moe_extra_info = ModelForwardExtraLogInfo()
        if extra_info:
            moe_extra_info.append(extra_info)
        output["extra_info"] = moe_extra_info

        # Same `num_hash_layers >= num_hidden_layers` guard as `_forward`: skip
        # finalize when no layer accumulated routing stats so the smoke configs
        # (e.g. release `num_hash_layers=3` with `num_hidden_layers=2`) don't crash.
        if self.config.num_hash_layers < self.config.num_hidden_layers:
            balancing_loss, z_loss, tokens_per_expert_global = self.aux_loss.finalize(
                balancing_ctx=balancing_ctx,
                z_ctx=z_ctx,
                non_pad_token=non_pad_token,
            )
            if balancing_loss is not None:
                output["balancing_loss"] = balancing_loss
            if z_loss is not None:
                output["z_loss"] = z_loss
            output["tokens_per_expert_global"] = tokens_per_expert_global
        else:
            output["tokens_per_expert_global"] = None

        if keep_router:
            # Stack per-MB router logits into the same `[1, n_mb, ...]` layout the
            # parent emits, so downstream consumers don't need a V4-specific branch.
            router_logits_dict: dict[str, torch.Tensor] = {}
            layer_names = list(router_logits_per_mb[0].keys())
            for layer_name in layer_names:
                stacked = torch.stack(
                    [router_logits_per_mb[mb][layer_name].detach() for mb in range(n_mb)],
                    dim=0,
                ).unsqueeze(0)
                router_logits_dict[layer_name] = stacked
            output["router_logits"] = router_logits_dict

        return MoEModelOutputs(**output, logits=logits)

    def _hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # Eager wrapper: unshard the DTensor params once, then enter the
        # compile-friendly compute. Splitting at this boundary mirrors the
        # ``V4DecoderLayer.forward → _unshard_hc_params → hc_pre`` pattern.
        from torch.distributed.tensor import DTensor as _DTensor

        hc_head_fn = self.hc_head_fn.full_tensor() if isinstance(self.hc_head_fn, _DTensor) else self.hc_head_fn
        hc_head_scale = (
            self.hc_head_scale.full_tensor() if isinstance(self.hc_head_scale, _DTensor) else self.hc_head_scale
        )
        hc_head_base = (
            self.hc_head_base.full_tensor() if isinstance(self.hc_head_base, _DTensor) else self.hc_head_base
        )
        return self._hc_head_reduce_compute(x, hc_head_fn, hc_head_scale, hc_head_base)

    def _hc_head_reduce_compute(
        self,
        x: torch.Tensor,
        hc_head_fn: torch.Tensor,
        hc_head_scale: torch.Tensor,
        hc_head_base: torch.Tensor,
    ) -> torch.Tensor:
        # Port of `ParallelHead.hc_head` (model.py L728-735). Mirrors `hc_pre`'s
        # RMS-rescaled mixing but skips Sinkhorn — head reduce uses a per-stream
        # sigmoid weight, not a doubly-stochastic mix. Compile-friendly: takes
        # already-unsharded HC params and contains only tensor ops, so dynamo
        # traces it as one contiguous graph (matmul + rsqrt + sigmoid + sum
        # fuse into ~2-3 kernels instead of 8 eager small ops).
        #
        # Same fp32-upcast avoidance as :func:`hc_pre`: keep activations in
        # bf16, only reduce/accumulate in fp32, and run the gate linear in
        # bf16 (cuBLAS internally accumulates in fp32). The tiny ``mixes``
        # output is upcast for the sigmoid + per-stream bias, then auto-
        # promoted on the final ``pre × x_flat`` multiplication.
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)
        sq_mean = (x_flat * x_flat).mean(-1, keepdim=True, dtype=torch.float32)
        rsqrt = torch.rsqrt(sq_mean + self.config.rms_norm_eps)
        mixes = torch.nn.functional.linear(x_flat, hc_head_fn.to(x_flat.dtype)).float() * rsqrt
        pre = torch.sigmoid(mixes * hc_head_scale.float() + hc_head_base.float()) + self.config.hc_cfg.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
        return y.to(dtype)

    def _build_one_layer(
        self,
        config: DeepSeekV4Config,
        layer_idx: int,
        compress_ratio: int,
    ) -> V4DecoderLayer:
        # Pick the router topology per-layer; hash layers do not need group/topk_group
        # because they bypass scoring entirely.
        router_config: NoAuxRouterConfig | HashRouterConfig
        if layer_idx < config.num_hash_layers:
            router_config = HashRouterConfig(
                vocab_size=config.vocab_size,
                n_routed_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
        else:
            router_config = config.router

        # DSAConfig.build requires per-layer `compress_ratio` which the generic
        # `attention_config.build` chain doesn't know about, so DSA is
        # constructed here and handed to V4DecoderLayer via the
        # `attention_module` kwarg.
        attention_module = config.attention.build(
            hidden_size=config.hidden_size,
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
        )

        return V4DecoderLayer(
            compress_ratio=compress_ratio,
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            num_experts_per_tok=config.num_experts_per_tok,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            moe_act_fn_cfg=config.moe_act_fn_cfg,
            router_config=router_config,
            dispatcher=config.dispatcher,
            ep_mesh=self.ep_mesh,
            hc_cfg=config.hc_cfg,
            attention_module=attention_module,
            rms_norm_eps=config.rms_norm_eps,
            rms_norm_type=config.rms_norm_type,
            mlp_bias=config.mlp_bias,
            gate_bias=False,
            moe_bias=config.moe_bias,
            with_shared_expert_gate=config.with_shared_expert_gate,
            hidden_factor=config.hidden_factor,
            router_compute_dtype=config.router_compute_dtype,
            float8_cfg=config.float8_cfg,
        )

    def _translate_layer_tail(self, tail: str, layer_idx: str, n_routed_experts: int) -> list[str]:
        # XTuner module names → HF key fragments (per BF16 safetensors index).
        # ``V4DecoderLayer`` owns every parameter directly, so XTuner-side
        # tails arrive flat: ``layers.L.hc_attn_fn``, ``layers.L.input_layernorm.weight``,
        # ``layers.L.experts.fused_w1w3.weight`` etc. HF also keeps things flat
        # under ``layers.L.`` but renames some of the slots. The mapping below
        # is the authoritative bridge.
        del layer_idx  # only used for the outer prefix already prepended

        # HC mix parameters: name 1:1 with HF (no prefix change).
        if tail.startswith("hc_attn_") or tail.startswith("hc_ffn_"):
            return [tail]

        # Pre-attention / post-attention layernorms map to `attn_norm` / `ffn_norm`.
        if tail == "input_layernorm.weight":
            return ["attn_norm.weight"]
        if tail == "post_attention_layernorm.weight":
            return ["ffn_norm.weight"]

        # Attention: XTuner `self_attn.*` → HF `attn.*`.
        if tail.startswith("self_attn."):
            return ["attn." + tail[len("self_attn.") :]]

        # MoE gate: XTuner `gate.weight` is the linear projection; `gate.router.*` are
        # the router-specific buffers (tid2eid for hash, e_score_correction_bias for noaux).
        if tail == "gate.weight":
            return ["ffn.gate.weight"]
        if tail == "gate.bias":
            return ["ffn.gate.bias"]
        if tail == "gate.router.tid2eid":
            return ["ffn.gate.tid2eid"]
        if tail == "gate.router.e_score_correction_bias":
            return ["ffn.gate.bias"]

        # Experts: fused tensors expand to per-expert HF keys.
        if tail == "experts.fused_w1w3.weight":
            keys: list[str] = []
            for i in range(n_routed_experts):
                keys.append(f"ffn.experts.{i}.w1.weight")
                keys.append(f"ffn.experts.{i}.w3.weight")
            return keys
        if tail == "experts.fused_w2.weight":
            return [f"ffn.experts.{i}.w2.weight" for i in range(n_routed_experts)]

        # Shared experts: XTuner uses gate_proj/up_proj/down_proj names; HF uses w1/w3/w2.
        if tail == "shared_experts.gate_proj.weight":
            return ["ffn.shared_experts.w1.weight"]
        if tail == "shared_experts.up_proj.weight":
            return ["ffn.shared_experts.w3.weight"]
        if tail == "shared_experts.down_proj.weight":
            return ["ffn.shared_experts.w2.weight"]

        # Fallback: pass through (catches anything we missed — surfaces as a missing
        # key in `from_hf` rather than silent skipping).
        return [tail]

    def _translate_mtp_tail(self, tail: str, n_routed_experts: int) -> list[str]:
        # XTuner MTP layer wraps a decoder_layer + extra fields (e_proj/h_proj/enorm/
        # hnorm/norm/hc_head_*). HF mirrors the body of a main layer under `mtp.M.*`
        # plus the extras at the same level. The exact MTPLayer attribute names
        # depend on the cherry-picked PR; this translator is best-effort for
        # `to_hf_key_list_coverage` and may need a follow-up once V4 MTP is wired
        # to the new HC head pattern.
        if tail.startswith("decoder_layer."):
            inner_tail = tail[len("decoder_layer.") :]
            return self._translate_layer_tail(inner_tail, "0", n_routed_experts)
        return [tail]
