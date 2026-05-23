# Copyright (c) OpenMMLab. All rights reserved.
"""Decoder-layer level forward parity vs HuggingFace's :mod:`transformers.models.deepseek_v4`.

We construct matched small DeepSeek-V4 configs on both sides (HF reference and XTuner),
random-init the HF layer, copy parameters across with an explicit name / layout map,
and compare layer outputs on the same input.

Three layer types are covered:

* **sliding-only** (``compress_ratio=0``, ``layer_types[i]="sliding_attention"``) —
  validates attention, HC, MoE, and the FFN path without any compressor / Indexer
  in the way.
* **CSA** (``compress_ratio=4``, ``layer_types[i]="compressed_sparse_attention"``) —
  adds the KVCompressor + Indexer top-K to the sliding-attn path.
* **HCA** (``compress_ratio=128``, ``layer_types[i]="heavily_compressed_attention"``) —
  KVCompressor with deterministic positional gather (no Indexer).

Why the tolerance is non-zero. The XTuner forward and the HF forward execute the
same math but in different orders (HF builds one extended ``[K_window + K_compressed]``
KV stream and runs a single dense softmax with a mask; XTuner gathers the per-query
top-K via ``sparse_attn`` and runs an online softmax over them). Both are
mathematically equivalent up to floating-point reduction order; we therefore allow
~1e-2 absolute / 1e-2 relative tolerance in bf16, the standard parity budget used
elsewhere in the repo (see ``test_qwen3_moe.py``).

Cost. Each test loads two ~5 MB models (toy dims), no checkpoint needed; ~few
seconds per test.
"""
from __future__ import annotations

import os

# The default ``m_grouped_gemm_TMA_triton3_4`` MoE expert kernel fails to compile
# under Triton 3.5.1 with ``PassManager::run failed`` (``ttng.tensormap_create``
# pipeliner missing); the same env var the V4 CI config uses to swap in the
# cutlass-backed grouped GEMM unblocks ``MoEBlock.forward`` here. Set before any
# xtuner / triton import so the import-time dispatcher picks up the override.
os.environ.setdefault("XTUNER_USE_CUTLASS_GROUP_GEMM", "1")

import pytest
import torch

# HF V4 reference (transformers >= 5.9.0)
hf_v4 = pytest.importorskip("transformers.models.deepseek_v4")
from transformers.models.deepseek_v4 import DeepseekV4Config as HFDeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer as HFDecoderLayer,
    DeepseekV4Model as HFV4Model,
)

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.deepseek_v4 import DeepSeekV4Config, V4DecoderLayer
from xtuner.v1.module.attention.dsa import DSAConfig
from xtuner.v1.module.decoder_layer.hc_block import HCWrapperConfig, hc_post, hc_pre
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.module.router.hash_router import HashRouterConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


# ─── Small-config knobs shared across tests ─────────────────────────────────────
_VOCAB = 256
_HIDDEN = 64
_MOE_INTER = 32             # used as ``intermediate_size`` on HF, ``moe_intermediate_size`` on XTuner
_N_HEADS = 8
_HEAD_DIM = 32              # MLA full head_dim
# XTuner V4 now uses HF's interleaved RoPE convention (DualRotaryEmbedding
# emits half-dim cos/sin; DSA's ``_apply_rope`` applies adjacent-pair
# rotation matching HF's ``DeepseekV4RotaryEmbedding`` +
# ``apply_rotary_pos_emb``). This means we can exercise rope at production-
# like dims here without weight permutation. We pick 16 (not the V4-Flash
# default 64) just to keep this test fixture small.
_QK_ROPE = 16
_SLIDING = 32               # sliding window length (multiple of 8 keeps FlashAttention happy if ever enabled)
_INDEX_TOPK = 8
_INDEX_HEAD_DIM = 16
_INDEX_N_HEADS = 4
_N_ROUTED = 4
_N_SHARED = 1
_N_EXPERTS_PER_TOK = 2
_O_GROUPS = 2
_O_LORA = 16
_Q_LORA = 32
_HC_MULT = 4
_HC_SINKHORN_ITERS = 3      # cut down for test speed
_RMS_EPS = 1e-6
_DTYPE = torch.bfloat16


# ─── HF / XTuner config builders ────────────────────────────────────────────────


def _build_hf_config(layer_types: list[str], num_hash_layers: int) -> HFDeepseekV4Config:
    """Build a small HF DeepseekV4Config with the given per-layer types.

    Args:
        layer_types (list[str]): One of ``"sliding_attention"``,
            ``"compressed_sparse_attention"``, ``"heavily_compressed_attention"``
            per layer.
        num_hash_layers (int): How many leading layers use the hash router.

    Returns:
        HFDeepseekV4Config: A small config that matches the XTuner config built
        by :func:`_build_xtuner_config` 1:1 in dimensions and per-layer mode.
    """
    n_layers = len(layer_types)
    mlp_layer_types = ["hash_moe"] * num_hash_layers + ["moe"] * max(0, n_layers - num_hash_layers)

    cfg = HFDeepseekV4Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_MOE_INTER,
        num_hidden_layers=n_layers,
        num_attention_heads=_N_HEADS,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=_QK_ROPE / _HEAD_DIM,
        sliding_window=_SLIDING,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        n_routed_experts=_N_ROUTED,
        num_experts_per_tok=_N_EXPERTS_PER_TOK,
        n_shared_experts=_N_SHARED,
        num_local_experts=_N_ROUTED,
        o_groups=_O_GROUPS,
        o_lora_rank=_O_LORA,
        q_lora_rank=_Q_LORA,
        index_topk=_INDEX_TOPK,
        index_head_dim=_INDEX_HEAD_DIM,
        index_n_heads=_INDEX_N_HEADS,
        hc_mult=_HC_MULT,
        hc_sinkhorn_iters=_HC_SINKHORN_ITERS,
        hc_eps=_RMS_EPS,
        rms_norm_eps=_RMS_EPS,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        swiglu_limit=10.0,
        max_position_embeddings=2048,
        pad_token_id=None,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_xtuner_config(compress_ratios: list[int], num_hash_layers: int) -> DeepSeekV4Config:
    """Build a small XTuner DeepSeekV4Config that mirrors :func:`_build_hf_config`.

    Args:
        compress_ratios (list[int]): One of ``0`` / ``4`` / ``128`` per layer.
        num_hash_layers (int): How many leading layers use the hash router.

    Returns:
        DeepSeekV4Config: Matches the HF config 1:1 in dims and per-layer mode.
    """
    n_layers = len(compress_ratios)
    cfg = DeepSeekV4Config(
        num_hidden_layers=n_layers,
        num_hash_layers=num_hash_layers,
        n_routed_experts=_N_ROUTED,
        n_shared_experts=_N_SHARED,
        num_experts_per_tok=_N_EXPERTS_PER_TOK,
        hidden_size=_HIDDEN,
        moe_intermediate_size=_MOE_INTER,
        vocab_size=_VOCAB,
        mtp_config=None,
        hc_cfg=HCWrapperConfig(hc_mult=_HC_MULT, hc_eps=_RMS_EPS, hc_sinkhorn_iters=_HC_SINKHORN_ITERS),
        rms_norm_eps=_RMS_EPS,
    )
    cfg.attention = DSAConfig(
        num_attention_heads=_N_HEADS,
        num_key_value_heads=1,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE,
        q_lora_rank=_Q_LORA,
        o_lora_rank=_O_LORA,
        o_groups=_O_GROUPS,
        sliding_window=_SLIDING,
        use_attn_sink=True,
        index_head_dim=_INDEX_HEAD_DIM,
        index_n_heads=_INDEX_N_HEADS,
        index_topk=_INDEX_TOPK,
        indexer_backend="native",  # _INDEX_N_HEADS=4 is below the triton kernel's tensor-core floor
        backend="native",          # avoid FlashMLA / cudnn dependencies in unit tests
    )
    cfg.rope_parameters_cfg = RopeParametersConfig(
        rope_theta=10000.0,
        rope_type="yarn",
        beta_fast=32,
        beta_slow=1,
        factor=16,
        original_max_position_embeddings=65536,
        compress_rope_theta=160000.0,
        compress_ratios=compress_ratios + [0],   # trailing 0 for the (unused) MTP slot
    )
    cfg.router = NoAuxRouterConfig(
        n_group=2,
        topk_group=2,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    )
    cfg.dispatcher = None   # eager experts, no all2all
    cfg.compile_cfg = False
    # HF V4's ``DeepseekV4TopKRouter`` / ``DeepseekV4HashRouter`` run the gate
    # ``F.linear`` in the input dtype (bf16), then ``score_fn(logits)`` in the
    # same dtype. XTuner's MoEGate defaults to ``router_compute_dtype="float32"``
    # — it upcasts hidden_states and the gate weight to fp32 before the linear
    # for routing-stability. Bit-identical parity requires matching HF's bf16
    # compute. (For production training the fp32 path is a *real* improvement
    # over bf16 routing; we only pin to ``"native"`` here to make parity
    # measurable.)
    cfg.router_compute_dtype = "native"
    return cfg


# ─── HF → XTuner weight copy ────────────────────────────────────────────────────


def _copy_hf_to_xtuner_layer(
    hf_layer: HFDecoderLayer,
    xtuner_layer: V4DecoderLayer,
    *,
    compress_ratio: int,
) -> None:
    """Copy every parameter from a HF DeepseekV4DecoderLayer into the matched
    XTuner V4DecoderLayer.

    The two implementations have different module nesting (HF nests the Indexer
    *inside* the CSA compressor; XTuner has the Indexer as a sibling of the
    main compressor on ``self_attn``) and a different MoE expert storage layout
    (HF: per-expert 3D ``nn.Parameter`` ``[E, 2*I, H]``; XTuner: flattened
    ``[E*2*I, H]`` via ``build_grouped_linear``). The mapping below mirrors
    ``DeepSeekV4._translate_layer_tail`` going the other direction.

    Args:
        hf_layer (HFDecoderLayer): Source.
        xtuner_layer (V4DecoderLayer): Target. Modified in-place.
        compress_ratio (int): ``0`` / ``4`` / ``128`` — selects which compressor
            sub-tree to copy (only one is present on each layer).
    """
    # HC parameters (mHC residual mix).
    xtuner_layer.hc_attn_fn.data.copy_(hf_layer.attn_hc.fn.data)
    xtuner_layer.hc_attn_base.data.copy_(hf_layer.attn_hc.base.data)
    xtuner_layer.hc_attn_scale.data.copy_(hf_layer.attn_hc.scale.data)
    xtuner_layer.hc_ffn_fn.data.copy_(hf_layer.ffn_hc.fn.data)
    xtuner_layer.hc_ffn_base.data.copy_(hf_layer.ffn_hc.base.data)
    xtuner_layer.hc_ffn_scale.data.copy_(hf_layer.ffn_hc.scale.data)

    # Pre / post attention norms.
    xtuner_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    xtuner_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)

    # Attention core (Q/KV/O LoRA chain + per-head attention sink).
    xa = xtuner_layer.self_attn
    ha = hf_layer.self_attn
    xa.wq_a.weight.data.copy_(ha.q_a_proj.weight.data)
    xa.q_norm.weight.data.copy_(ha.q_a_norm.weight.data)
    xa.wq_b.weight.data.copy_(ha.q_b_proj.weight.data)
    xa.wkv.weight.data.copy_(ha.kv_proj.weight.data)
    xa.kv_norm.weight.data.copy_(ha.kv_norm.weight.data)
    # wo_a is "grouped low-rank": both HF (``DeepseekV4GroupedLinear``) and XTuner
    # store the weight as flat ``[o_groups * o_lora_rank, head_dim_per_group]``
    # and reshape to ``[o_groups, o_lora_rank, head_dim_per_group]`` per-group at
    # forward time — bit-identical weight layouts, no transpose / reshape needed.
    xa.wo_a.weight.data.copy_(ha.o_a_proj.weight.data)
    xa.wo_b.weight.data.copy_(ha.o_b_proj.weight.data)
    xa.attn_sink.data.copy_(ha.sinks.data.to(xa.attn_sink.dtype))

    if compress_ratio in (4, 128):
        # The two compressor flavours (``DeepseekV4HCACompressor`` /
        # ``DeepseekV4CSACompressor``) expose the same outer surface:
        # ``kv_proj`` / ``gate_proj`` / ``position_bias`` / ``kv_norm``. CSA
        # additionally owns an ``indexer`` submodule.
        hf_comp = ha.compressor
        x_comp = xa.compressor
        x_comp.wkv.weight.data.copy_(hf_comp.kv_proj.weight.data)
        x_comp.wgate.weight.data.copy_(hf_comp.gate_proj.weight.data)
        x_comp.ape.data.copy_(hf_comp.position_bias.data)
        x_comp.norm.weight.data.copy_(hf_comp.kv_norm.weight.data)

    if compress_ratio == 4:
        # XTuner: ``self_attn.indexer`` (sibling of ``compressor``) with a nested
        # ``self_attn.indexer.compressor`` (KVCompressor with rotate=True).
        # HF:     ``self_attn.compressor.indexer`` (nested in CSA compressor),
        #         whose compression weights are flat fields on the Indexer (not a
        #         nested compressor module).
        hf_idx = ha.compressor.indexer
        x_idx = xa.indexer
        x_idx.wq_b.weight.data.copy_(hf_idx.q_b_proj.weight.data)
        x_idx.weights_proj.weight.data.copy_(hf_idx.weights_proj.weight.data)
        x_idx.compressor.wkv.weight.data.copy_(hf_idx.kv_proj.weight.data)
        x_idx.compressor.wgate.weight.data.copy_(hf_idx.gate_proj.weight.data)
        x_idx.compressor.ape.data.copy_(hf_idx.position_bias.data)
        x_idx.compressor.norm.weight.data.copy_(hf_idx.kv_norm.weight.data)

    # MoE: router + experts + shared experts.
    hf_mlp = hf_layer.mlp
    xtuner_layer.gate.weight.data.copy_(hf_mlp.gate.weight.data)
    if hf_mlp.is_hash:
        # HashRouter: deterministic ``tid2eid[input_ids]`` selection.
        xtuner_layer.gate.router.tid2eid.data.copy_(hf_mlp.gate.tid2eid.data)
    else:
        # NoAuxRouter: per-expert score bias.
        xtuner_layer.gate.router.e_score_correction_bias.data.copy_(
            hf_mlp.gate.e_score_correction_bias.data.to(xtuner_layer.gate.router.e_score_correction_bias.dtype)
        )

    # Experts: HF stores ``gate_up_proj`` as 3D ``[E, 2*I, H]`` with gate / up
    # stacked on dim 1 in that order (``F.linear(x, w)`` then ``.chunk(2, -1)``).
    # XTuner stores ``fused_w1w3.weight`` as ``[E * 2*I, H]`` — same memory layout
    # once the leading two dims are flattened. ``down_proj``: HF ``[E, H, I]``,
    # XTuner ``[E*H, I]`` — same flatten.
    n_routed = _N_ROUTED
    hf_gate_up = hf_mlp.experts.gate_up_proj.data    # [E, 2*I, H]
    hf_down = hf_mlp.experts.down_proj.data          # [E, H, I]
    xtuner_layer.experts.fused_w1w3.weight.data.copy_(
        hf_gate_up.reshape(n_routed * 2 * _MOE_INTER, _HIDDEN)
    )
    xtuner_layer.experts.fused_w2.weight.data.copy_(hf_down.reshape(n_routed * _HIDDEN, _MOE_INTER))

    # Shared experts: XTuner ``MoEMLP`` uses ``gate_proj`` / ``up_proj`` / ``down_proj``,
    # same as HF.
    if xtuner_layer.shared_experts is not None:
        xtuner_layer.shared_experts.gate_proj.weight.data.copy_(hf_mlp.shared_experts.gate_proj.weight.data)
        xtuner_layer.shared_experts.up_proj.weight.data.copy_(hf_mlp.shared_experts.up_proj.weight.data)
        xtuner_layer.shared_experts.down_proj.weight.data.copy_(hf_mlp.shared_experts.down_proj.weight.data)


# ─── Test anchor: swap XTuner attention with HF's naive attention ──────────────


def _install_hf_attention_fallback(
    xtuner_layer: V4DecoderLayer,
    hf_layer: HFDecoderLayer,
    hf_model: HFV4Model,
) -> None:
    """Monkey-patch ``xtuner_layer.self_attn.__call__`` to delegate to HF's
    ``DeepseekV4Attention``. After this, ``xtuner_layer.forward(...)`` produces
    the same attention output as HF would (bit-identical, weights already
    copied via :func:`_copy_hf_to_xtuner_layer`). The HC / norms / MoE wrappers
    around the attention stay XTuner-native.

    The XTuner DSA's call signature is
    ``forward(hidden_states, position_embeddings, position_embeddings_compressed, seq_ctx)``
    and returns an ``AttnOutputs`` dict. We adapt those to HF's
    ``forward(hidden_states, position_embeddings, position_ids, attention_mask, past_key_values)``
    signature in the patch, wrapping HF's returned ``(attn_output, attn_weights)``
    back into an ``AttnOutputs``-shaped dict.

    Use this when the parity test wants 0 attention-path error to isolate
    other divergence sources (MoE kernel reduction order, etc.).

    The XTuner DSA module is kept alive (its weights remain registered as
    submodule parameters and continue to be copied from HF), so toggling this
    monkey-patch off restores the XTuner path without state loss.

    Args:
        xtuner_layer: target XTuner V4DecoderLayer. Its ``self_attn`` is patched
            in-place.
        hf_layer: matched HF DecoderLayer holding the source ``DeepseekV4Attention``.
        hf_model: HF model whose ``rotary_emb`` produces the dual rope cos/sin
            in HF's interleaved (half-dim) format.
    """
    hf_attn = hf_layer.self_attn
    sliding = hf_model.config.sliding_window

    def _hf_naive_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx,
    ):
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        # XTuner's ``DualRotaryEmbedding`` now emits D-dim pre-arranged
        # ``(cos_full, sin_full_signed)`` so the per-layer ``_apply_rope`` is
        # one fused ``x * cos + flip_pairs(x) * sin``. HF still consumes the
        # half-dim layout, so undo the precompute here for the fallback:
        #   ``cos_half[..., i]  = cos_full[..., 2i]``        (even positions; odd dup)
        #   ``sin_half[..., i]  = sin_full_signed[..., 2i+1]`` (odd positions; even = -sin)
        def _to_hf_half(pe: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            cos_full, sin_full_signed = pe
            return cos_full[..., 0::2].contiguous(), sin_full_signed[..., 1::2].contiguous()

        position_embeddings_hf = {
            "main": _to_hf_half(position_embeddings),
            "compress": _to_hf_half(position_embeddings_compressed) if position_embeddings_compressed is not None
                        else _to_hf_half(position_embeddings),
        }
        if seq_ctx is not None and getattr(seq_ctx, "position_ids", None) is not None:
            position_ids = seq_ctx.position_ids
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = _build_sliding_causal_mask(
            seq_len, sliding, dtype=hidden_states.dtype, device=device
        )
        attn_output, attn_weights = hf_attn(
            hidden_states,
            position_embeddings=position_embeddings_hf,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        # XTuner DSA returns an ``AttnOutputs`` TypedDict with these keys.
        # Only ``projected_output`` is consumed by V4DecoderLayer._attn_compute,
        # so we fill the others with sentinels.
        return {
            "projected_output": attn_output,
            "raw_output": attn_output,    # unused downstream in this test
            "softmax_lse": attn_weights,  # unused downstream
        }

    xtuner_layer.self_attn.forward = _hf_naive_forward   # type: ignore[method-assign]


def _install_hf_moe_fallback(
    xtuner_layer: V4DecoderLayer,
    hf_layer: HFDecoderLayer,
) -> None:
    """Monkey-patch ``xtuner_layer``'s MoE block to delegate to
    ``hf_layer.mlp`` (HF's ``DeepseekV4SparseMoeBlock``). After this, the
    XTuner V4DecoderLayer produces HF-bit-identical MoE outputs.

    The patch hooks ``_ffn_compute`` because ``V4DecoderLayer.forward`` calls
    that method to produce the FFN output + ``router_results`` — replacing
    it means we don't need to mock the dispatcher / experts call chain
    individually. ``_ffn_pre_compute`` / ``_ffn_post_compute`` stay as
    XTuner-native (the post-norm + ``hidden_factor`` scale).

    Together with :func:`_install_hf_attention_fallback` this gives true
    zero-error parity at the layer output: only HC's two ``hc_pre`` /
    ``hc_post`` calls (already proven bit-identical to HF in
    ``test_subcomponent_probe``) are XTuner-native, and they match HF
    exactly.

    Args:
        xtuner_layer: target V4DecoderLayer to patch.
        hf_layer: matched HF decoder layer providing the source ``mlp``.
    """
    hf_mlp = hf_layer.mlp
    is_hash = hf_mlp.is_hash

    def _hf_naive_ffn_compute(
        x: torch.Tensor,
        seq_ctx,
        input_ids: torch.Tensor | None,
    ):
        # XTuner's ``_ffn_compute`` internally calls ``_ffn_pre_compute(x)``
        # which applies ``post_attention_layernorm`` before the gate +
        # experts. HF's ``DeepseekV4SparseMoeBlock.forward`` does NOT do that
        # norm itself — it's done outside by ``DeepseekV4DecoderLayer.forward``
        # (line 1136: ``self.mlp(self.post_attention_layernorm(collapsed), ...)``).
        # So when we delegate ``_ffn_compute`` to HF's ``mlp``, we must apply
        # ``post_attention_layernorm`` here ourselves; otherwise the MoE sees
        # un-normalised input and the layer output drifts by ~1 bf16 ULP.
        x_normed = xtuner_layer.post_attention_layernorm(x)
        # Resolve hash router's input_ids (HF expects [B, S] not flat).
        if is_hash:
            ids = input_ids.view(1, -1) if input_ids is not None else None
            mlp_out = hf_mlp(x_normed, input_ids=ids)
            # HF's HashRouter forward signature still emits the same router_results
            # tuple. Call gate directly to populate router_results.
            logits, weights, indices = hf_mlp.gate(x_normed, ids)
        else:
            mlp_out = hf_mlp(x_normed, input_ids=None)
            logits, weights, indices = hf_mlp.gate(x_normed)
        # XTuner's ``_ffn_post_compute(combined, h_normed)`` scales by
        # ``hidden_factor`` — but HF's ``mlp.forward`` does NOT do that scaling
        # (V4 config defaults ``hidden_factor=1.0`` so it's a no-op anyway).
        # Apply XTuner's scaling here to keep the call-site contract.
        ffn_out = mlp_out * xtuner_layer.hidden_factor
        router_results = {
            "logits": logits,
            "router_weights": weights,
            "topk_weights": weights,
            "topk_ids": indices,
            "topkens_per_expert": torch.histc(
                indices, bins=_N_ROUTED, min=0, max=_N_ROUTED
            ),
        }
        return ffn_out, router_results

    xtuner_layer._ffn_compute = _hf_naive_ffn_compute   # type: ignore[method-assign]


# ─── Forward driver ─────────────────────────────────────────────────────────────


def _build_sliding_causal_mask(
    seq_len: int,
    sliding_window: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``[1, 1, S, S]`` causal + sliding-window additive mask.

    Cell ``(i, j)`` is ``0.0`` (allowed) iff ``j <= i`` and ``i - j < window``,
    else ``-inf``. This matches what
    :func:`transformers.create_sliding_window_causal_mask` produces in
    ``DeepseekV4Model.forward`` — we build it directly here instead of going
    through that helper because the helper expects a full ``inputs_embeds``
    tensor and a ``past_key_values`` cache and we already have everything we
    need.

    Required because HF's ``eager_attention_forward`` does:
        ``if attention_mask is not None: attn_weights += attention_mask``
    — i.e. ``None`` means *no* mask, so HF would attend over the full
    sequence (non-causal). XTuner's ``sparse_attn`` is causal by construction
    (the topk_idxs only includes ``j <= i`` positions in its sliding window),
    so to match HF must receive an explicit causal+window mask.
    """
    positions = torch.arange(seq_len, device=device)
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)   # rel[i, j] = j - i
    valid = (rel <= 0) & (rel > -sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~valid, float("-inf"))
    return mask.view(1, 1, seq_len, seq_len)


def _run_hf_layer(
    hf_model: HFV4Model,
    layer_idx: int,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one HF DeepseekV4DecoderLayer in isolation.

    Args:
        hf_model (HFV4Model): Source of the layer + the rotary basis.
        layer_idx (int): Layer to invoke.
        hidden_states (torch.Tensor): ``[B, S, hc_mult, hidden]`` HC-expanded input.
        input_ids (torch.Tensor): ``[B, S]`` consumed by HashRouter; ignored elsewhere.
        position_ids (torch.Tensor): ``[B, S]`` token positions.

    Returns:
        torch.Tensor: ``[B, S, hc_mult, hidden]`` layer output.
    """
    # Same pre-layer rope setup as ``DeepseekV4Model.forward``.
    position_embeddings = {
        "main": hf_model.rotary_emb(hidden_states.flatten(2), position_ids=position_ids, layer_type="main"),
        "compress": hf_model.rotary_emb(hidden_states.flatten(2), position_ids=position_ids, layer_type="compress"),
    }
    causal_mask = _build_sliding_causal_mask(
        position_ids.shape[1],
        hf_model.config.sliding_window,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    return hf_model.layers[layer_idx](
        hidden_states,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=causal_mask,
        input_ids=input_ids,
        past_key_values=None,
    )


def _hf_rotary_to_xtuner_format(
    hf_rotary: torch.nn.Module,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute D-dim ``(cos_full, sin_full_signed)`` sharing inv_freq with HF.

    XTuner's ``DualRotaryEmbedding`` pre-arranges cos/sin into the layout
    the per-layer ``_apply_rope`` consumes directly: D-dim ``cos_full`` and
    ``sin_full_signed`` (sign pattern ``[-, +, -, +, ...]`` folded in). This
    helper reproduces that precompute from HF's half-dim ``inv_freq`` so the
    two sides see bit-identical angles AND identical input layout to DSA.
    """
    inv_freq = getattr(hf_rotary, f"{layer_type}_inv_freq")  # [qk_rope_head_dim/2]
    scaling = getattr(hf_rotary, f"{layer_type}_attention_scaling", 1.0)
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.float()
    cos_half = freqs.cos() * scaling
    sin_half = freqs.sin() * scaling
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full_signed = torch.stack([-sin_half, sin_half], dim=-1).flatten(-2)
    return cos_full.to(hidden_states_2d.dtype), sin_full_signed.to(hidden_states_2d.dtype)


def _run_xtuner_layer(
    xtuner_layer: V4DecoderLayer,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    hf_model: HFV4Model,
) -> torch.Tensor:
    """Run the matched XTuner V4DecoderLayer with the same inputs.

    XTuner's forward signature differs from HF's:
    * takes ``position_embeddings`` (dense rope) + ``position_embeddings_compressed``
      (yarn rope) as separate tuples rather than a dict,
    * takes a ``seq_ctx`` carrying ``cu_seq_lens`` (varlen packing — irrelevant
      here with a single sample but required by the API),
    * always treats the input as packed varlen ``[1, total_tokens, hc_mult, D]``,
    * needs XTuner-format (cat-style) cos/sin while HF's rotary_emb emits
      half-dim interleaved-style cos/sin.

    We share the inv_freq between the two sides via
    :func:`_hf_rotary_to_xtuner_format` so both rotations use the same
    underlying angles.
    """
    hidden_2d = hidden_states.flatten(2)
    # HF's rotary emits half-dim ``(cos, sin)``. XTuner's ``DualRotaryEmbedding``
    # now emits the pre-arranged D-dim ``(cos_full, sin_full_signed)`` layout
    # the per-layer ``_apply_rope`` consumes directly — convert HF's output to
    # match. The conversion is bit-identical to the precompute step in
    # ``DualRotaryEmbedding.forward`` (no extra precision loss).
    cos_main_half, sin_main_half = hf_model.rotary_emb(hidden_2d, position_ids=position_ids, layer_type="main")
    cos_comp_half, sin_comp_half = hf_model.rotary_emb(hidden_2d, position_ids=position_ids, layer_type="compress")
    cos_main = cos_main_half.repeat_interleave(2, dim=-1)
    sin_main = torch.stack([-sin_main_half, sin_main_half], dim=-1).flatten(-2)
    cos_comp = cos_comp_half.repeat_interleave(2, dim=-1)
    sin_comp = torch.stack([-sin_comp_half, sin_comp_half], dim=-1).flatten(-2)
    bsz, seq_len = position_ids.shape
    assert bsz == 1, "XTuner V4DecoderLayer is hardcoded to packed-varlen with batch=1"
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=hidden_states.device)
    seq_ctx = SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=seq_len,
        max_length_k=seq_len,
        device=str(hidden_states.device),
    )
    seq_ctx.position_ids = position_ids
    out, _, _ = xtuner_layer(
        hidden_states,
        position_embeddings=(cos_main, sin_main),
        position_embeddings_compressed=(cos_comp, sin_comp),
        seq_ctx=seq_ctx,
        input_ids=input_ids.view(-1),
    )
    return out


# ─── Test cases ─────────────────────────────────────────────────────────────────


_LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


@pytest.mark.gpu
class TestV4DecoderLayerParity:
    """Run identical inputs through a matched HF + XTuner decoder layer and
    require their outputs to match within bf16 reduction-order tolerance."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        torch.manual_seed(0)

    def _setup(self, layer_type: str, *, num_hash_layers: int, layer_idx: int):
        """Build matched models, copy HF → XTuner, return (hf_model, xtuner_layer, layer_idx)."""
        ratio = _LAYER_TYPE_TO_RATIO[layer_type]
        # We only need ``layer_idx + 1`` layers but build a few extras so the
        # tested layer is not the last one (the model's final-layer reshard
        # path differs slightly).
        n_layers = layer_idx + 1
        # Pad layer_types so the desired one is at ``layer_idx``.
        layer_types = ["sliding_attention"] * n_layers
        layer_types[layer_idx] = layer_type
        compress_ratios = [_LAYER_TYPE_TO_RATIO[t] for t in layer_types]

        hf_cfg = _build_hf_config(layer_types, num_hash_layers)
        xtuner_cfg = _build_xtuner_config(compress_ratios, num_hash_layers)

        device = torch.device("cuda")
        hf_model = HFV4Model(hf_cfg).to(device=device, dtype=_DTYPE).eval()
        xtuner_model = xtuner_cfg.build().to(device=device, dtype=_DTYPE).eval()

        # Copy parameters layer by layer so the routed-experts indexing matches.
        for i, lt in enumerate(layer_types):
            _copy_hf_to_xtuner_layer(
                hf_model.layers[i],
                xtuner_model.layers[str(i)],
                compress_ratio=_LAYER_TYPE_TO_RATIO[lt],
            )

        return hf_model, xtuner_model.layers[str(layer_idx)], layer_idx, ratio

    def _common_inputs(self, seq_len: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random ``hidden_states`` (HC-expanded), ``input_ids``, ``position_ids``."""
        device = torch.device("cuda")
        hidden_states = torch.randn(1, seq_len, _HC_MULT, _HIDDEN, device=device, dtype=_DTYPE)
        input_ids = torch.randint(0, _VOCAB, (1, seq_len), device=device, dtype=torch.long)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        return hidden_states, input_ids, position_ids

    def test_sliding_attention_parity(self) -> None:
        """No compressor / no Indexer — exercises Q/KV/O LoRA, attn sink, HC, MoE."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "sliding_attention", num_hash_layers=0, layer_idx=0
        )
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=32)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=1e-2, rtol=1e-2)

    def test_csa_parity(self) -> None:
        """compress_ratio=4 (Indexer top-K + KVCompressor) + sliding window."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "compressed_sparse_attention", num_hash_layers=0, layer_idx=1
        )
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=64)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=2e-2, rtol=2e-2)

    def test_hca_parity(self) -> None:
        """compress_ratio=128 (deterministic positional gather + KVCompressor)."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "heavily_compressed_attention", num_hash_layers=0, layer_idx=1
        )
        # Pack large enough for HCA to actually have non-trivial chunks:
        # ratio=128 → at S=256 the last query sees 2 compressed chunks.
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=256)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=2e-2, rtol=2e-2)

    def test_hash_router_parity(self) -> None:
        """Hash-routed sliding layer (layer_idx < num_hash_layers)."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "sliding_attention", num_hash_layers=1, layer_idx=0
        )
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=32)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=1e-2, rtol=1e-2)

    def test_csa_parity_with_hf_attention_anchor(self) -> None:
        """CSA layer with attention DELEGATED to HF — measures the rest of
        the XTuner layer (HC pre/post, norms, MoE) in isolation.

        After ``_install_hf_attention_fallback``, the XTuner V4DecoderLayer's
        attention sub-path is bit-identical to HF's. Any remaining diff in
        the layer output comes from the non-attention parts (HC residual mix,
        MoE expert dispatch). With all other numerical alignments in place
        (commit ``c64c89fc``) the residual should land in bf16 MoE-kernel
        noise (~3e-2 abs), but the attention path contributes 0 by
        construction.
        """
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "compressed_sparse_attention", num_hash_layers=0, layer_idx=1
        )
        _install_hf_attention_fallback(xtuner_layer, hf_model.layers[layer_idx], hf_model)
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=64)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=4e-2, rtol=4e-2)

    def test_hca_parity_with_hf_attention_anchor(self) -> None:
        """HCA layer with attention delegated to HF — see
        :meth:`test_csa_parity_with_hf_attention_anchor`."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "heavily_compressed_attention", num_hash_layers=0, layer_idx=1
        )
        _install_hf_attention_fallback(xtuner_layer, hf_model.layers[layer_idx], hf_model)
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=256)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=4e-2, rtol=4e-2)

    def test_csa_parity_full_hf_anchor(self) -> None:
        """CSA layer with BOTH attention and MoE delegated to HF — XTuner HC
        pre/post + V4DecoderLayer wrapping stay live. After aligning
        ``hc_split_sinkhorn`` to use ``torch.softmax`` (matching HF's
        ``DeepseekV4HyperConnection.forward``), the XTuner forward is
        bit-identical to HF at this anchor (every fp32 + bf16 step has
        the same reduction order). Tolerance is ``atol=0.0``.
        """
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "compressed_sparse_attention", num_hash_layers=0, layer_idx=1
        )
        hf_layer = hf_model.layers[layer_idx]
        _install_hf_attention_fallback(xtuner_layer, hf_layer, hf_model)
        _install_hf_moe_fallback(xtuner_layer, hf_layer)
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=64)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=0.0, rtol=0.0)

    def test_hca_parity_full_hf_anchor(self) -> None:
        """HCA layer with BOTH attention and MoE delegated to HF — see
        :meth:`test_csa_parity_full_hf_anchor`. ``atol=0.0``."""
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "heavily_compressed_attention", num_hash_layers=0, layer_idx=1
        )
        hf_layer = hf_model.layers[layer_idx]
        _install_hf_attention_fallback(xtuner_layer, hf_layer, hf_model)
        _install_hf_moe_fallback(xtuner_layer, hf_layer)
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=256)
        with torch.no_grad():
            hf_out = _run_hf_layer(hf_model, layer_idx, hidden_states, input_ids, position_ids)
            xt_out = _run_xtuner_layer(xtuner_layer, hidden_states, input_ids, position_ids, hf_model)
        torch.testing.assert_close(xt_out, hf_out, atol=0.0, rtol=0.0)

    def test_subcomponent_probe(self, capsys) -> None:
        """Walk through the sliding-attention forward step by step and print
        the first sub-component where HF and XTuner diverge.

        Both sides share weights (copied via ``_copy_hf_to_xtuner_layer``) so
        every step should match to within bf16 reduction-order tolerance
        (~1e-4 abs). The first step that exceeds that bound is the bug site.

        We instrument the *sliding-only* case (no compressor, no Indexer) to
        keep the surface small — once that path matches we can extend the
        probe to CSA / HCA.
        """
        hf_model, xtuner_layer, layer_idx, _ = self._setup(
            "sliding_attention", num_hash_layers=0, layer_idx=0
        )
        hidden_states, input_ids, position_ids = self._common_inputs(seq_len=32)

        bsz, seq_len = position_ids.shape
        device = hidden_states.device
        n_heads = _N_HEADS
        head_dim = _HEAD_DIM
        qk_rope_head_dim = _QK_ROPE
        hidden_dim = _HIDDEN

        hf_layer = hf_model.layers[layer_idx]
        ha = hf_layer.self_attn
        xa = xtuner_layer.self_attn

        steps: list[tuple[str, torch.Tensor, torch.Tensor]] = []  # (name, hf_t, xt_t)

        with torch.no_grad():
            # ─── Step 1: HC pre on attention ─────────────────────────────────
            # HF: attn_hc(hidden_states) -> (post, comb, collapsed)
            # XTuner: hc_pre(hidden_states, hc_attn_fn, hc_attn_scale, hc_attn_base, ...)
            #         -> (collapsed, post, comb)  (different return order!)
            hf_post, hf_comb, hf_collapsed = hf_layer.attn_hc(hidden_states)
            xt_collapsed, xt_post, xt_comb = hc_pre(
                hidden_states,
                xtuner_layer.hc_attn_fn,
                xtuner_layer.hc_attn_scale,
                xtuner_layer.hc_attn_base,
                xtuner_layer.hc_mult,
                xtuner_layer.hc_sinkhorn_iters,
                xtuner_layer.hc_eps,
            )
            steps.append(("hc_pre.collapsed", hf_collapsed, xt_collapsed))
            steps.append(("hc_pre.post", hf_post, xt_post))
            steps.append(("hc_pre.comb", hf_comb, xt_comb))

            # Use HF's collapsed for both sides downstream so divergence is
            # attributed to the SUB-STEP, not to upstream cascade.
            x = hf_collapsed

            # ─── Step 2: input_layernorm ─────────────────────────────────────
            hf_norm = hf_layer.input_layernorm(x)
            xt_norm = xtuner_layer.input_layernorm(x)
            steps.append(("input_layernorm", hf_norm, xt_norm))
            x = hf_norm   # reuse downstream

            # ─── Step 3: Q-LoRA chain ────────────────────────────────────────
            hf_q_a = ha.q_a_proj(x)
            xt_q_a = xa.wq_a(x)
            steps.append(("q_a_proj", hf_q_a, xt_q_a))

            hf_q_a_n = ha.q_a_norm(hf_q_a)
            xt_q_a_n = xa.q_norm(hf_q_a)
            steps.append(("q_a_norm", hf_q_a_n, xt_q_a_n))

            q_lowrank = hf_q_a_n

            hf_q_b = ha.q_b_proj(q_lowrank).view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
            xt_q_b = xa.wq_b(q_lowrank).unflatten(-1, (n_heads, head_dim))
            # Compare in matching layout: HF [B, H, S, D], XTuner [B, S, H, D]
            steps.append(("q_b_proj", hf_q_b.transpose(1, 2), xt_q_b))

            # ─── Step 4: per-head RMSNorm on Q ───────────────────────────────
            # Both HF (``DeepseekV4UnweightedRMSNorm``) and XTuner DSA's inline
            # per-head RMS compute the square in fp32 (matched after the
            # ``[Fix] V4 q_b_norm/hc_pre: fp32 square`` commit). We replicate
            # XTuner's exact math here rather than calling a submodule, since
            # XTuner does this inline in DSA.forward.
            q_for_norm = hf_q_b   # [B, H, S, D] layout from HF
            hf_q_normed = ha.q_b_norm(q_for_norm)
            xt_q_inv = torch.rsqrt(q_for_norm.float().square().mean(-1, keepdim=True) + _RMS_EPS).to(
                q_for_norm.dtype
            )
            xt_q_normed = q_for_norm * xt_q_inv
            steps.append(("q_b_norm (per-head)", hf_q_normed, xt_q_normed))

            # ─── Step 5: KV path ─────────────────────────────────────────────
            hf_kv = ha.kv_norm(ha.kv_proj(x)).view(bsz, seq_len, 1, head_dim).transpose(1, 2)
            xt_kv = xa.kv_norm(xa.wkv(x)).unflatten(-1, (1, head_dim))
            steps.append(("kv_proj+norm", hf_kv.transpose(1, 2), xt_kv))

            # ─── Step 6: Attention end-to-end (DSA.forward vs HF attn) ───────
            # Feed both sides the SAME ``x`` (post-input_layernorm hidden states)
            # and compare the projected output (post-O-LoRA). This bundles
            # rope + QK^T + softmax + V + O-LoRA into one comparison; any
            # divergence here means one of those is the bug. Bisection
            # continues below if this step fails.
            position_embeddings_hf = {
                "main": hf_model.rotary_emb(x, position_ids=position_ids, layer_type="main"),
                "compress": hf_model.rotary_emb(x, position_ids=position_ids, layer_type="compress"),
            }
            causal_mask = _build_sliding_causal_mask(
                seq_len,
                hf_model.config.sliding_window,
                dtype=x.dtype,
                device=device,
            )
            hf_attn_out, _ = ha(
                x,
                position_embeddings=position_embeddings_hf,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=None,
            )

            cos_main, sin_main = _hf_rotary_to_xtuner_format(
                hf_model.rotary_emb, x, position_ids, "main"
            )
            cos_comp, sin_comp = _hf_rotary_to_xtuner_format(
                hf_model.rotary_emb, x, position_ids, "compress"
            )
            cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            xt_seq_ctx = SequenceContext(
                input_ids=input_ids,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=seq_len,
                max_length_k=seq_len,
                device=str(device),
            )
            xt_seq_ctx.position_ids = position_ids
            xt_attn = xa(
                x,
                position_embeddings=(cos_main, sin_main),
                position_embeddings_compressed=(cos_comp, sin_comp),
                seq_ctx=xt_seq_ctx,
            )
            xt_attn_out = xt_attn["projected_output"]
            steps.append(("attention end-to-end", hf_attn_out, xt_attn_out))

            # ─── Step 7: HC-post for attention ───────────────────────────────
            # Feed both sides the SAME attention output + SAME (post, comb,
            # residual) so divergence here is the hc_post implementation only.
            hf_post_a, hf_comb_a, hf_collapsed_a = hf_layer.attn_hc(hidden_states)
            # HF inline (decoder_layer.forward lines 1131-1133):
            hf_hc_post = hf_post_a.to(_DTYPE).unsqueeze(-1) * hf_attn_out.unsqueeze(-2) + torch.matmul(
                hf_comb_a.to(_DTYPE).transpose(-1, -2), hidden_states
            )
            # XTuner hc_post call:
            from xtuner.v1.module.decoder_layer.hc_block import hc_post as _hc_post
            xt_hc_post = _hc_post(hf_attn_out, hidden_states, hf_post_a, hf_comb_a)
            steps.append(("hc_post (attn)", hf_hc_post, xt_hc_post))

            # ─── Step 8: post_attention_layernorm ────────────────────────────
            # Feed both sides the SAME hc_post_a output, take a single HC
            # stream (collapsed for ffn block) — but for parity, we apply
            # post_attention_layernorm to the same input.
            # Use HF's collapsed for the ffn-block input:
            hf_post_f, hf_comb_f, hf_collapsed_f = hf_layer.ffn_hc(hf_hc_post)
            ffn_in = hf_collapsed_f
            hf_pln = hf_layer.post_attention_layernorm(ffn_in)
            xt_pln = xtuner_layer.post_attention_layernorm(ffn_in)
            steps.append(("post_attention_layernorm", hf_pln, xt_pln))

            # ─── Step 9a: MoE router ─────────────────────────────────────────
            # Compare router weights and chosen expert indices.
            hf_mlp_gate = hf_layer.mlp.gate
            if hf_layer.mlp.is_hash:
                hf_logits, hf_weights, hf_indices = hf_mlp_gate(hf_pln, input_ids)
            else:
                hf_logits, hf_weights, hf_indices = hf_mlp_gate(hf_pln)
            steps.append(("router.logits", hf_logits, hf_logits))  # self-check (skip)

            xt_router_in = hf_pln   # both sides take same input
            xt_router_results = xtuner_layer.gate(
                xt_router_in,
                None,
                input_ids=(input_ids.view(-1) if hf_layer.mlp.is_hash else None),
            )
            xt_logits = xt_router_results["logits"]
            xt_topk_ids = xt_router_results["topk_ids"]
            xt_topk_w = xt_router_results["topk_weights"]
            # Compare logits, weights, indices. Indices must match exactly.
            steps.append(("router.logits", hf_logits, xt_logits.view(hf_logits.shape)))
            steps.append(("router.topk_weights", hf_weights, xt_topk_w.view(hf_weights.shape)))
            # Indices: HF gives [B*S, top_k], XTuner gives [B*S, top_k]. Bool diff.
            idx_diff = (hf_indices != xt_topk_ids.view(hf_indices.shape)).float().sum().item()
            if idx_diff > 0:
                print(f"[FAIL] router.topk_ids: {int(idx_diff)} elements differ between HF and XTuner")

            # ─── Step 9b: routed experts (use HF's indices on both sides) ────
            # Bypass routing differences (HF's topk(sorted=False) + XTuner's
            # topk(sorted=True) can return the same SET of indices in
            # different orders for tied scores, which masks any expert-side
            # divergence). Compute the routed-expert output manually using
            # HF's chosen indices on both sides.
            hf_top_k_idx = hf_indices.view(seq_len, -1)  # [S, top_k]
            hf_top_k_w = hf_weights.view(seq_len, -1)    # [S, top_k]
            flat = hf_pln.view(-1, hf_pln.shape[-1])

            # HF experts: explicit per-expert loop with gate_up_proj / down_proj.
            hf_routed_out = hf_layer.mlp.experts(flat, hf_top_k_idx, hf_top_k_w).view_as(hf_pln)

            # XTuner experts: use HF's indices on the SAME per-expert loop,
            # exposing the fused expert weights as 3D views (HF's native
            # ``[E, 2*I, H]`` / ``[E, H, I]`` layout).
            xt_gate_up = xtuner_layer.experts.fused_w1w3.weight.view(
                _N_ROUTED, 2 * _MOE_INTER, _HIDDEN
            )
            xt_down = xtuner_layer.experts.fused_w2.weight.view(_N_ROUTED, _HIDDEN, _MOE_INTER)
            from transformers.activations import ACT2FN as _ACT2FN

            act_fn = _ACT2FN["silu"]
            limit = float(hf_model.config.swiglu_limit)
            flat = hf_pln.view(-1, hf_pln.shape[-1])
            xt_routed_out = torch.zeros_like(flat)
            expert_mask = torch.nn.functional.one_hot(hf_top_k_idx, num_classes=_N_ROUTED).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in hit:
                eidx = int(expert_idx[0])
                if eidx == _N_ROUTED:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[eidx])
                current = torch.nn.functional.linear(flat[token_idx], xt_gate_up[eidx])
                gate, up = current.chunk(2, dim=-1)
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
                current = act_fn(gate) * up
                current = torch.nn.functional.linear(current, xt_down[eidx])
                current = current * hf_top_k_w[token_idx, top_k_pos, None]
                xt_routed_out.index_add_(0, token_idx, current.to(xt_routed_out.dtype))
            xt_routed_out = xt_routed_out.view_as(hf_pln)
            steps.append(("routed experts (same indices)", hf_routed_out, xt_routed_out))

            # ─── Step 9c: shared experts ─────────────────────────────────────
            hf_shared = hf_layer.mlp.shared_experts(hf_pln)
            xt_shared = xtuner_layer._shared_experts_forward(hf_pln)
            steps.append(("shared_experts", hf_shared, xt_shared))


            # ─── Step 9d: MoE end-to-end ─────────────────────────────────────
            hf_mlp_out = hf_layer.mlp(hf_pln, input_ids=input_ids)

            # XTuner has no equivalent single-method MoE call; mimic the
            # ffn_block path:
            rollout_routed_experts = None   # no rollout in test
            xt_input_ids = input_ids.view(-1) if hf_layer.mlp.is_hash else None
            h_normed, router_results = xtuner_layer._ffn_pre_compute(
                ffn_in, rollout_routed_experts, xt_input_ids
            )
            # h_normed here equals xt_pln above (already verified). Run the
            # MoE expert dispatch path manually (no all2all since dispatcher=None
            # in the test config).
            origin_shape = h_normed.shape
            dispatcher = xtuner_layer.dispatcher
            pre_disp = dispatcher.dispatch_preprocess(
                hidden_states=h_normed.view(-1, h_normed.shape[-1]),
                topk_ids=router_results["topk_ids"],
            )
            disp = dispatcher.dispatch(
                pre_dispatched=pre_disp,
                topk_weights=router_results["topk_weights"],
                decoding=False,
            )
            post_disp = dispatcher.dispatch_postprocess(pre_dispatched=pre_disp, dispatched=disp)
            experts_out = xtuner_layer.experts(
                post_disp["hidden_states"],
                post_disp["tokens_per_expert"],
                decoding=False,
            )
            pre_comb = dispatcher.combine_preprocess(
                hidden_states=experts_out,
                pre_dispatched=pre_disp,
                dispatched=disp,
                post_dispatched=post_disp,
                decoding=False,
            )
            combined = dispatcher.combine(
                pre_dispatched=pre_disp,
                dispatched=disp,
                post_dispatched=post_disp,
                pre_combined=pre_comb,
                decoding=False,
            )
            post_comb = dispatcher.combine_postprocess(
                pre_dispatched=pre_disp,
                dispatched=disp,
                post_dispatched=post_disp,
                pre_combined=pre_comb,
                combined=combined,
            )
            xt_routed = post_comb["hidden_states"].view(*origin_shape)
            xt_mlp_out = xtuner_layer._ffn_post_compute(xt_routed, h_normed)
            # Isolate dispatcher + grouped-GEMM kernel diff vs the per-expert
            # manual loop on the same indices/weights (``xt_routed_out``).
            steps.append(("xtuner dispatcher+grouped vs per-expert", xt_routed_out, xt_routed))
            steps.append(("MoE end-to-end", hf_mlp_out, xt_mlp_out))

            # ─── Step 10: HC-post for FFN ────────────────────────────────────
            hf_hc_ffn_post = hf_post_f.to(_DTYPE).unsqueeze(-1) * hf_mlp_out.unsqueeze(-2) + torch.matmul(
                hf_comb_f.to(_DTYPE).transpose(-1, -2), hf_hc_post
            )
            xt_hc_ffn_post = _hc_post(hf_mlp_out, hf_hc_post, hf_post_f, hf_comb_f)
            steps.append(("hc_post (ffn)", hf_hc_ffn_post, xt_hc_ffn_post))

        # Threshold tiers (per-step max abs diff):
        #   <= 1e-5 → fp32 ULP noise (single-op level)
        #   <= 1e-3 → bf16 multi-op chain noise (e.g. attention / RMSNorm with
        #             many bf16 reductions); not a bug
        #   >  1e-3 → real divergence, needs investigation
        BF16_FLOOR = 1e-3
        for name, hf_t, xt_t in steps:
            if hf_t.shape != xt_t.shape:
                print(f"[FAIL] {name}: shape mismatch hf={tuple(hf_t.shape)} xt={tuple(xt_t.shape)}")
                continue
            diff = (hf_t.float() - xt_t.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            tag = "OK" if max_diff <= BF16_FLOOR else "DIFF"
            print(f"[{tag}] {name:30s} max={max_diff:.4e}  mean={mean_diff:.4e}  shape={tuple(hf_t.shape)}")

        # Force capsys to flush
        captured = capsys.readouterr()
        # Re-print so pytest -s users see it; also fail if any step is off
        print(captured.out)
        bad = [
            (name, hf_t, xt_t)
            for name, hf_t, xt_t in steps
            if hf_t.shape != xt_t.shape
            or (hf_t.float() - xt_t.float()).abs().max().item() > BF16_FLOOR
        ]
        assert not bad, (
            f"Sub-component probe found {len(bad)} divergent step(s) above {BF16_FLOOR:.0e}: "
            f"{[name for name, _, _ in bad]}"
        )
