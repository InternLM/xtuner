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
    """Compute half-dim cos/sin sharing inv_freq with HF.

    Both XTuner's ``DualRotaryEmbedding`` and HF's
    ``DeepseekV4RotaryEmbedding`` now emit half-dim cos/sin
    (``[B, S, qk_rope_head_dim/2]``) in the interleaved RoPE convention —
    after the ``[Fix] V4: switch RoPE to interleaved convention`` commit,
    the two are layout- and convention-equivalent. We still build cos/sin
    via this helper rather than calling HF's rotary directly so we can pass
    them straight into XTuner's DSA, sharing HF's ``inv_freq`` for
    bit-identical angles.
    """
    inv_freq = getattr(hf_rotary, f"{layer_type}_inv_freq")  # [qk_rope_head_dim/2]
    scaling = getattr(hf_rotary, f"{layer_type}_attention_scaling", 1.0)
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.float()
    return (freqs.cos() * scaling).to(hidden_states_2d.dtype), (freqs.sin() * scaling).to(hidden_states_2d.dtype)


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
    cos_main, sin_main = _hf_rotary_to_xtuner_format(hf_model.rotary_emb, hidden_2d, position_ids, "main")
    cos_comp, sin_comp = _hf_rotary_to_xtuner_format(hf_model.rotary_emb, hidden_2d, position_ids, "compress")
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
            # HF uses ``DeepseekV4UnweightedRMSNorm`` (no weight); XTuner inlines
            # ``rsqrt(q_sq.mean) + multiply`` directly. Same math.
            q_for_norm = hf_q_b   # [B, H, S, D] layout from HF
            hf_q_normed = ha.q_b_norm(q_for_norm)
            # Inline XTuner version, applied to the same layout.
            q_sq = q_for_norm * q_for_norm
            q_inv = torch.rsqrt(q_sq.mean(-1, keepdim=True, dtype=torch.float32) + _RMS_EPS).to(q_for_norm.dtype)
            xt_q_normed = q_for_norm * q_inv
            steps.append(("q_b_norm (per-head)", hf_q_normed, xt_q_normed))

            # ─── Step 5: KV path ─────────────────────────────────────────────
            hf_kv = ha.kv_norm(ha.kv_proj(x)).view(bsz, seq_len, 1, head_dim).transpose(1, 2)
            xt_kv = xa.kv_norm(xa.wkv(x)).unflatten(-1, (1, head_dim))
            steps.append(("kv_proj+norm", hf_kv.transpose(1, 2), xt_kv))

        # Print first 3 mismatches above 1e-4 (bf16 noise floor) so we can see
        # how far down the chain the agreement holds.
        printed = 0
        for name, hf_t, xt_t in steps:
            if hf_t.shape != xt_t.shape:
                print(f"[FAIL] {name}: shape mismatch hf={tuple(hf_t.shape)} xt={tuple(xt_t.shape)}")
                printed += 1
                continue
            diff = (hf_t.float() - xt_t.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            tag = "OK" if max_diff < 1e-4 else "DIFF"
            print(f"[{tag}] {name:30s} max={max_diff:.4e}  mean={mean_diff:.4e}  shape={tuple(hf_t.shape)}")
            if max_diff > 1e-4:
                printed += 1
            if printed >= 5:
                break

        # Force capsys to flush
        captured = capsys.readouterr()
        # Re-print so pytest -s users see it; also fail if any step is off
        print(captured.out)
        any_diff = any(
            (hf_t.shape == xt_t.shape and (hf_t.float() - xt_t.float()).abs().max().item() > 1e-4)
            or hf_t.shape != xt_t.shape
            for _, hf_t, xt_t in steps
        )
        assert not any_diff, (
            "Sub-component probe found divergence — see printed output above for first offending step."
        )
