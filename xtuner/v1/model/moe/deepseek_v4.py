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
from typing import cast

import torch
import torch.nn as nn
from pydantic import Field
from typing_extensions import Self, override

from transformers import AutoConfig
from xtuner.v1.model.base import HFSaveCfg, TorchCompileOption
from xtuner.v1.module import HashRouterConfig, NoAuxRouterConfig
from xtuner.v1.module.attention.dsa import DSAConfig
from xtuner.v1.module.attention.dsa.kv_compressor import KVCompressor
from xtuner.v1.module.decoder_layer.deepseek_v4 import HCWrapperConfig, V4DecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
)
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.hf_parity import hf_parity_enabled

from .moe import (
    MOE_EP_COMPILE_CFG,
    MOE_NON_EP_COMPILE_CFG,
    BalancingLossConfig,
    LayerInput,
    LayerInputMB,
    MoE,
    MoEConfig,
    ZLossConfig,
)


logger = get_logger()


# DeepSeek TileKernels mhc backend (https://github.com/deepseek-ai/TileKernels). Opt-in via
# ``XTUNER_USE_MHC_KERNELS=1``. Default OFF: the existing HF-parity ``sigmoid(scale * mixes +
# base) + eps`` path keeps producing the bit-equivalent result it has been validated against.
# When ON, we route the per-token mix-gate computation in ``_hc_head_reduce_compute`` through
# ``tile_kernels.mhc.head_compute_mix_kernel._mhc_head_compute_mix_fwd`` (Hopper SM90+ TileLang
# JIT). Kernel instances are keyed by ``(hc_mult, hc_eps, token_block_size)`` and JIT-built
# once per process — the ``lru_cache`` makes that explicit.
_USE_MHC_KERNELS = os.environ.get("XTUNER_USE_MHC_KERNELS", "0") == "1"


def _expand_hc_streams(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Expand ``[B, S, D]`` → ``[B, S, hc_mult, D]`` for the HC residual
    stream.

    Default path is the standard ``unsqueeze + expand + contiguous`` chain. When
    ``XTUNER_USE_MHC_KERNELS=1`` we route through :func:`xtuner.v1.ops.mhc.mhc_expand`,
    which wraps the TileLang fwd/bwd kernels behind a ``torch.library.custom_op`` so
    autograd flows back to ``hidden_states`` via the reduce-sum bwd kernel.
    """
    if not _USE_MHC_KERNELS or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        return hidden_states.unsqueeze(-2).expand(-1, -1, hc_mult, -1).contiguous()
    from xtuner.v1.ops.mhc import mhc_expand

    return mhc_expand(hidden_states, hc_mult)


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
    "xtuner.v1.module.decoder_layer.deepseek_v4.hc_block.hc_pre": _HEAVY,
    # ``hc_post`` is now an eager dispatcher: the default bf16 path calls the
    # ``xtuner::hc_post_fwd`` Triton custom op (self-optimized, opaque to
    # compile), and only the ``_HC_HF_PARITY`` fp32 fallback
    # (``_hc_post_eager``) benefits from inductor fusion — so we compile that
    # one instead of the dispatcher.
    "xtuner.v1.module.decoder_layer.deepseek_v4.hc_block._hc_post_eager": _HEAVY,
    "xtuner.v1.module.decoder_layer.deepseek_v4.decoder_layer.V4DecoderLayer._attn_compute": _HEAVY,
    "xtuner.v1.module.decoder_layer.deepseek_v4.decoder_layer.V4DecoderLayer._ffn_pre_compute": _LITE,
    "xtuner.v1.module.decoder_layer.deepseek_v4.decoder_layer.V4DecoderLayer._ffn_post_compute": _LITE,
    "xtuner.v1.model.moe.deepseek_v4.DeepSeekV4._hc_head_reduce_compute": _LITE,
    "xtuner.v1.module.attention.dsa.dsa.DeepSeekSparseAttention.forward": _HEAVY,
    # The compressor's scatter + softmax + sum + RMSNorm chain is exactly the
    # ~50-elementwise-op storm that showed up in the rank0 trace under EP. The
    # ``int(cu_seq_lens_out[-1].item())`` sync at the head of forward breaks
    # the graph once; ``fullgraph=False`` (in ``_LITE``) accepts that break
    # and still fuses the two halves on either side of it.
    "xtuner.v1.module.attention.dsa.kv_compressor.KVCompressor.forward": _LITE,
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
# (:mod:`xtuner.v1.module.attention.dsa._indexer_topk_triton`), so the fp32 score
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


# The rope base a layer's *main* q/kv use is decided by whether the layer compresses, not
# by whether it owns an Indexer: the V4 reference builds ``Attention.freqs_cis`` with
# ``(original_seq_len, compress_rope_theta)`` when ``compress_ratio != 0`` and with
# ``(0, rope_theta)`` — YaRN disabled — otherwise. HF mirrors this with
# ``DeepseekV4Attention.rope_layer_type = "main" if sliding else "compress"``. Compressed
# layers therefore rotate q, the window KV *and* the compressed KV against one shared
# base, which is what makes q·k depend only on relative distance across both streams.


def _select_attention_rope(
    compress_ratio: int,
    layer_input: "V4LayerInput",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the rope basis a layer's main attention q/kv must rotate against.

    Args:
        compress_ratio (int): The layer's DSA mode (``0`` for pure sliding-window).
        layer_input (V4LayerInput): Carries both precomputed bases.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` for this layer.
    """
    if compress_ratio == 0:
        return layer_input["position_embeddings"]
    compressed = layer_input["position_embeddings_compressed"]
    assert compressed is not None, f"compress_ratio={compress_ratio} layer needs the compressed rope basis"
    return compressed


def _select_attention_rope_mb(
    compress_ratio: int,
    layer_input: "V4LayerInputMB",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Multi-micro-batch counterpart of :func:`_select_attention_rope`.

    Args:
        compress_ratio (int): The layer's DSA mode (``0`` for pure sliding-window).
        layer_input (V4LayerInputMB): Carries both precomputed bases, one entry per MB.

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: Per-MB ``(cos, sin)`` for this layer.
    """
    if compress_ratio == 0:
        return layer_input["position_embeddings"]
    compressed = layer_input["position_embeddings_compressed"]
    assert all(c is not None for c in compressed), (
        f"compress_ratio={compress_ratio} layer needs the compressed rope basis for every micro-batch"
    )
    return cast(list[tuple[torch.Tensor, torch.Tensor]], compressed)


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
            # The V4 reference floors / ceils the YaRN correction range
            # (`find_correction_range` in `precompute_freqs_cis`), and HF's
            # `_compute_yarn_parameters` defaults `truncate` to True for the same reason.
            # `RopeParametersConfig` defaults it to False, and HF configs never ship the
            # key, so V4 has to pin it here or every compressed layer's `inv_freq` lands on
            # a different interpolation ramp than the reference.
            truncate=True,
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
            # No group-limited routing — see the `from_hf` router comment.
            n_group=1,
            topk_group=1,
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
            # `clamped_swiglu`, not gpt-oss's `clipped_swiglu`: both apply V4's asymmetric
            # `swiglu_limit` clamp, but `clipped_swiglu` also carries gpt-oss's `(up + 1)`
            # residual term, which V4's `Expert.forward` / HF's `DeepseekV4MLP` do not have.
            act_type="clamped_swiglu",
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
            # V4 does no group-limited routing: its reference `Gate.forward` selects experts
            # with a plain `scores.topk(topk)` over all `n_routed_experts` (the `n_groups`
            # field on that class is `o_groups`, the attention O-LoRA grouping, not an expert
            # grouping), and the HF `DeepseekV4TopKRouter` matches. `n_group == topk_group == 1`
            # is how XTuner spells "no grouping".
            n_group=1,
            topk_group=1,
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
            act_type="clamped_swiglu",
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


class V4LayerInput(LayerInput):
    """DeepSeek-V4's :class:`LayerInput`: adds the compressed rope and raw
    ``input_ids`` that every V4 layer consumes (HashRouter reads ``input_ids``;
    the Indexer reads the compressed rope)."""

    position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None
    input_ids: torch.Tensor


class V4LayerInputMB(LayerInputMB):
    """Micro-batch counterpart of :class:`V4LayerInput`; every added field is a
    per-microbatch list."""

    position_embeddings_compressed: list[tuple[torch.Tensor, torch.Tensor] | None]
    input_ids: list[torch.Tensor]


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
        # Distinct positive compress_ratios across the stack. The model forward builds one
        # ``cu_seq_lens_out`` per ratio and caches it on the SequenceContext, so every layer of
        # that ratio reuses the cumsum + H2D instead of recomputing it inside its KVCompressor.
        self._compressor_ratios = sorted({r for r in compress_ratios[: v4_cfg.num_hidden_layers] if r > 0})

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

    # ---- Single-sequence ``_forward`` seams (the MoE skeleton drives these) ----
    # V4 keeps the parent's embed→loop→norm→lm_head skeleton and only overrides the
    # points where its forward graph diverges: an HC-expanded `[B, S, hc_mult, D]`
    # residual stream, a second (compressed) rope per layer, raw input_ids for the
    # HashRouter, and an hc_head reduction before the final norm. MTP is skipped
    # automatically — V4's mtp_block is None (build_mtp_block returns None) — so the
    # parent's MTP branch is a no-op (PR9 follow-up wires the V4-specific MTP head).

    def _assign_compressed_cu_seq_lens(self, seq_ctx) -> None:
        # Build ``cu_seq_lens_out`` once per distinct compress_ratio and cache it on the
        # SequenceContext, so the per-layer KVCompressor (DSA + Indexer) reuses it instead of
        # re-running the cumsum + H2D every call. Keyed by ratio because the chunk count is
        # ``ceil(L_i / ratio)`` — different for the ratio-4 and ratio-128 layers.
        if not self._compressor_ratios:
            return
        seq_ctx.compressed_cu_seq_lens = {
            ratio: KVCompressor.build_cu_seq_lens_out(seq_ctx.cu_seq_lens_q, seq_ctx.cu_seq_lens_q_cpu, ratio)[0]
            for ratio in self._compressor_ratios
        }

    @override
    def _prepare_hidden_states(self, seq_ctx) -> V4LayerInput:  # type: ignore[override]
        assert seq_ctx.position_ids is not None
        assert seq_ctx.input_ids is not None, "DeepSeekV4 requires input_ids (HashRouter consumes them)"
        self._assign_compressed_cu_seq_lens(seq_ctx)
        hidden_states = self.embed_tokens(seq_ctx.input_ids)
        # Dense rope (sliding-window heads) and compressed rope (Indexer) both come
        # from the same DualRotaryEmbedding; precompute both so each layer picks the
        # matching pair without branching on layer type.
        position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids, use_compressed=False)
        position_embeddings_compressed = _build_compressed_position_embeddings(
            self.rotary_emb, hidden_states, seq_ctx.position_ids
        )
        # Expand `[B, S, D]` → `[B, S, hc_mult, D]`. `.contiguous()` is essential
        # because downstream HC ops (`flatten(2)`, `.view(shape)`) assume a dense
        # layout; the expand-without-copy would alias.
        # NB: unlike the parent, V4 does not call `_mark_dynamic` here — its
        # aggressive-compile cfg drives dynamic-shape tracing instead.
        hidden_states = _expand_hc_streams(hidden_states, self._hc_mult)
        return {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "position_embeddings_compressed": position_embeddings_compressed,
            "input_ids": seq_ctx.input_ids,
        }

    @override
    def _call_decoder_layer(self, decoder_layer, idx, hidden_states, seq_ctx, layer_input: V4LayerInput):  # type: ignore[override]
        # Every V4 layer routes (first_k_dense_replace == 0) and needs the compressed
        # rope + raw input_ids that the parent contract omits.
        v4_layer = cast(V4DecoderLayer, decoder_layer)
        hidden_states, router_logits, router_weights = v4_layer(
            hidden_states,
            position_embeddings=_select_attention_rope(v4_layer.compress_ratio, layer_input),
            position_embeddings_compressed=layer_input["position_embeddings_compressed"],
            seq_ctx=seq_ctx,
            input_ids=layer_input["input_ids"],
        )
        return hidden_states, router_logits, router_weights

    @override
    def _finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Collapse `[B, S, hc_mult, D]` back to `[B, S, D]` via the model-level
        # hc_head triple before the standard final RMSNorm.
        return self._hc_head_reduce(hidden_states)

    @override
    def _should_finalize_aux_loss(self) -> bool:
        # When every layer is hash-routed (num_hash_layers >= num_hidden_layers,
        # legal for sub-stack smoke configs like 2 layers + 3 hash layers from the
        # release config), no layer accumulated routing stats and aux_loss.finalize
        # would raise from `_cal_tokens_per_expert`. `internal_metrics.py` already
        # treats `tokens_per_expert_global is None` as "no MoE load this step".
        return self.config.num_hash_layers < self.config.num_hidden_layers

    # ---- Micro-batch seams (same MoE skeleton; V4 keeps a per-MB list throughout) ----
    # The base mb path runs dense-prefix layers on the concatenated batch and splits to a
    # per-MB list at the MoE boundary. V4 has no dense prefix (first_k_dense_replace == 0) and
    # an HC-expanded per-MB residual stream, so its prepare returns the per-MB list directly and
    # the base's dense phase is a no-op — no cat-then-chunk round-trip. The tail (cat → finalize
    # → norm → lm_head → aux finalize) is the shared base implementation.

    @override
    def _prepare_hidden_states_mb(self, seq_ctx_list) -> V4LayerInputMB:  # type: ignore[override]
        # Per-MB embed → dual rope → HC expand. Each MB stays its own tensor (no cat across MBs):
        # V4DecoderLayer is called once per layer carrying all MBs anyway, and a cat-then-chunk
        # round-trip would only add the base's `i.clone()` workaround. No `_mark_dynamic` — V4's
        # aggressive-compile cfg drives dynamic-shape tracing.
        hidden_states_list: list[torch.Tensor] = []
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        position_embeddings_compressed_list: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        input_ids_list: list[torch.Tensor] = []
        for seq_ctx in seq_ctx_list:
            assert seq_ctx.position_ids is not None
            assert seq_ctx.input_ids is not None, "DeepSeekV4 requires input_ids (HashRouter consumes them)"
            self._assign_compressed_cu_seq_lens(seq_ctx)
            h = self.embed_tokens(seq_ctx.input_ids)
            pos_emb = self.rotary_emb(h, seq_ctx.position_ids, use_compressed=False)
            pos_emb_compressed = _build_compressed_position_embeddings(self.rotary_emb, h, seq_ctx.position_ids)
            h = _expand_hc_streams(h, self._hc_mult)
            hidden_states_list.append(h)
            position_embeddings_list.append(pos_emb)
            position_embeddings_compressed_list.append(pos_emb_compressed)
            input_ids_list.append(seq_ctx.input_ids)
        return {
            "hidden_states_list": hidden_states_list,
            "position_embeddings": position_embeddings_list,
            "position_embeddings_compressed": position_embeddings_compressed_list,
            "input_ids": input_ids_list,
        }

    @override
    def _call_decoder_layer_mb(
        self,
        decoder_layer,
        idx,
        hidden_states_list,
        seq_ctx_list,
        layer_input: V4LayerInputMB,  # type: ignore[override]
    ):
        # One call per layer carrying all MBs: V4DecoderLayer dispatches on the variadic
        # `*hidden_states` length to its internal 3-phase Domino wave, so FSDP2's pre/post-forward
        # hooks bracket the whole multi-MB pass exactly once. Adds the compressed rope + raw
        # input_ids the base contract omits.
        n_mb = len(hidden_states_list)
        v4_layer = cast(V4DecoderLayer, decoder_layer)
        layer_out = v4_layer(
            *hidden_states_list,
            position_embeddings=_select_attention_rope_mb(v4_layer.compress_ratio, layer_input),
            position_embeddings_compressed=layer_input["position_embeddings_compressed"],
            seq_ctx=seq_ctx_list,
            input_ids=layer_input["input_ids"],
        )
        return (
            list(layer_out[:n_mb]),
            list(layer_out[n_mb : 2 * n_mb]),
            list(layer_out[2 * n_mb :]),
        )

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
        # Two paths, the same split ``hc_pre`` makes — and driven by the same
        # ``XTUNER_V4_HF_PARITY`` switch, which is documented as *the* bit-exact escape
        # hatch for the whole HC family. This site used to ignore it, so flipping the flag
        # left the final stream collapse on the fast path and capped whole-model parity at
        # ~1.6e-2 no matter what the layers did.
        #
        # Default: keep activations in bf16, reduce/accumulate in fp32, and run the gate
        # linear in bf16 (cuBLAS accumulates in fp32 internally). Note the rescale is
        # applied *after* the linear rather than to its input — algebraically identical
        # because ``rsqrt`` is a per-token scalar and ``F.linear`` is homogeneous, and it
        # keeps the normalized full-width copy off HBM.
        #
        # Parity: normalize first, then run the linear in fp32, exactly as HF's
        # ``DeepseekV4HyperHead.forward`` orders it.
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)
        if hf_parity_enabled():
            x_flat_f32 = x_flat.float()
            rsqrt = torch.rsqrt(x_flat_f32.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
            mixes = torch.nn.functional.linear(x_flat_f32 * rsqrt, hc_head_fn.float())
        else:
            sq_mean = (x_flat * x_flat).mean(-1, keepdim=True, dtype=torch.float32)
            rsqrt = torch.rsqrt(sq_mean + self.config.rms_norm_eps)
            mixes = torch.nn.functional.linear(x_flat, hc_head_fn.to(x_flat.dtype)).float() * rsqrt
        pre = self._hc_head_sigmoid_gate(mixes, hc_head_scale, hc_head_base)
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
        return y.to(dtype)

    def _hc_head_sigmoid_gate(
        self,
        mixes: torch.Tensor,
        hc_head_scale: torch.Tensor,
        hc_head_base: torch.Tensor,
    ) -> torch.Tensor:
        # ``mixes`` is fp32 ``[B, S, hc_mult]`` and the reference math is
        # ``sigmoid(mixes * scale + base) + hc_eps``.
        #
        # Default (HF-parity) path uses the pure-PyTorch fused-op chain that has been the
        # validated reference all along. The TileKernels override is opt-in via
        # ``XTUNER_USE_MHC_KERNELS=1``; we keep both side-by-side so future numerical regressions
        # have a known-good fallback to bisect against, and so installations without TileLang
        # remain functional out of the box. Parity was validated at fp32 ``atol=rtol=1e-6`` on
        # SM90 (one ULP of ``sigmoid``).
        hc_eps = self.config.hc_cfg.hc_eps
        if not _USE_MHC_KERNELS or not mixes.is_cuda:
            return torch.sigmoid(mixes * hc_head_scale.float() + hc_head_base.float()) + hc_eps

        # ``mhc_head_compute_mix`` wraps the TileLang fwd/bwd kernels behind a
        # ``torch.library.custom_op``; the bwd reduces the per-SM scale/base partials and
        # routes them back to the original parameters via autograd.
        from xtuner.v1.ops.mhc import mhc_head_compute_mix

        return mhc_head_compute_mix(mixes, hc_head_scale.float().view(1), hc_head_base.float(), hc_eps)

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
            # Scoring is shared with the score-routed layers by construction — hash
            # routing overrides only which experts a token reaches, not how their
            # outputs are weighted.
            router_config = HashRouterConfig(
                vocab_size=config.vocab_size,
                n_routed_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                scoring_func=config.router.scoring_func,
                router_scaling_factor=config.router.router_scaling_factor,
                norm_topk_prob=config.router.norm_topk_prob,
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
