# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Portions of this file are adapted from DeepSeek-V4-Flash `inference/model.py`
# (class `Attention` lines 436-543, helpers `get_window_topk_idxs` 254-265,
# `get_compress_topk_idxs` 268-276, `apply_rotary_emb` 232-244), Copyright (c)
# DeepSeek-AI, released under the MIT License.
# Upstream reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# Local cache: .dev_scripts/deepseek_v4_reference/model.py
#
# Only the start_pos == 0 (prefill / training) branch is retained; all
# inference-time kv_cache, freqs_cis buffers, FP4/FP8 quantization and tensor
# parallelism paths are intentionally dropped. Packed varlen samples are
# processed one at a time via cu_seq_lens (matching KVCompressor / Indexer
# convention) rather than the fixed-batch tensors used by the upstream
# reference. The complex-pair RoPE convention of the upstream reference is
# replaced by the rotate-half convention to stay compatible with XTuner's
# `RotaryEmbedding` cos/sin output shape `[B, S, rope_head_dim]`.
# ============================================================================

from typing import Annotated, Literal

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import get_logger


try:
    import quack as _quack

    _HAS_QUACK = True
except ImportError:
    _HAS_QUACK = False

from ...rms_norm import RMSNorm
from ._dsa_rope import _apply_rope_inverse_split, _apply_rope_split
from ._dsa_topk import (
    _build_compress_topk_idxs_varlen,
    _build_window_topk_idxs_varlen,
    _interleave_window_compressed_kv,
    _shift_topk_to_global,
)
from ._flash_mla_sparse_attn import (
    _FLASH_MLA_TOPK_ALIGN,
    _flash_mla_topk_ok,
)
from ._flash_mla_sparse_attn import (
    cudnn_sparse_attn_apply as _cudnn_sparse_attn_apply,
)
from ._flash_mla_sparse_attn import (
    flash_mla_sparse_attn_apply as _flash_mla_sparse_attn_apply,
)
from ..attn_outputs import AttnOutputs
from .indexer import Indexer, IndexerConfig
from .kv_compressor import KVCompressor
from .sparse_attn import sparse_attn as _native_sparse_attn


logger = get_logger()


class DSAConfig(BaseModel):
    """Configuration for DeepSeek-V4 Sparse Attention.

    Sibling of :class:`MLAConfig` rather than a subclass: V4-Flash has no
    latent KV projection (``wkv`` goes straight from ``hidden_size`` to
    ``head_dim``), uses MQA (single KV head), splits ``head_dim`` into a
    NoPE prefix and a ``qk_rope_head_dim`` suffix, and replaces the single
    ``o_proj`` with a grouped low-rank ``wo_a → wo_b`` chain. Forcing these
    differences into ``MLAConfig`` via optional fields would require
    branching on ``kv_lora_rank is None`` throughout the V3 path (see design
    doc §4.3).

    Args:
        num_attention_heads (int): Number of query heads (64 in V4).
        num_key_value_heads (int): Number of KV heads (1 in V4 — MQA).
        head_dim (int): Per-head dimension (512 in V4). The final
            ``qk_rope_head_dim`` of these dims carry rotary positional info,
            the remaining ``head_dim - qk_rope_head_dim`` are NoPE.
        qk_rope_head_dim (int): RoPE-carrying suffix length (64 in V4).
        q_lora_rank (int): Width of the low-rank Q intermediate
            ``q_norm(wq_a(x))`` (1024 in V4).
        o_lora_rank (int): Per-group rank of the grouped output LoRA
            (1024 in V4).
        o_groups (int): Number of output groups for grouped O-LoRA (8 in V4).
            ``num_attention_heads * head_dim`` must be divisible by this.
        sliding_window (int): Local-attention window size (128 in V4). Every
            DSA layer applies this window; compressed layers stack the
            compressed-KV stream on top of it.
        use_attn_sink (bool): Whether to allocate a learnable per-head
            ``attn_sink`` parameter. Defaults to ``True``.
        index_head_dim (int): Per-head dimension of the Indexer's internal
            stream (128 in V4).
        index_n_heads (int): Number of Indexer scoring heads (64 in V4).
        index_topk (int): Top-k budget per query in the Indexer (512 in V4).
        rms_norm_eps (float): Epsilon used by every RMSNorm in DSA
            (``q_norm``, ``kv_norm``, plus any Compressor and Indexer norms).
            Defaults to ``1e-6``.
    """

    model_config = ConfigDict(title="DeepSeek Sparse Attention config", extra="forbid")
    num_attention_heads: Annotated[int, Parameter(group="attention")]
    num_key_value_heads: Annotated[int, Parameter(group="attention")] = 1
    head_dim: Annotated[int, Parameter(group="attention")]
    qk_rope_head_dim: Annotated[int, Parameter(group="attention")]
    q_lora_rank: Annotated[int, Parameter(group="attention")]
    o_lora_rank: Annotated[int, Parameter(group="attention")]
    o_groups: Annotated[int, Parameter(group="attention")]
    sliding_window: Annotated[int, Parameter(group="attention")]
    use_attn_sink: bool = True
    index_head_dim: Annotated[int, Parameter(group="attention")] = 128
    index_n_heads: Annotated[int, Parameter(group="attention")] = 64
    index_topk: Annotated[int, Parameter(group="attention")] = 512
    rms_norm_eps: float = 1e-6
    # Backend selecting the sparse-attention kernel used for the layer's
    # ``sparse_attn`` call:
    #   * ``"native"``  — the pure-PyTorch reference in :mod:`sparse_attn`.
    #     Slow but autograd-correct, has no extra deps. Default.
    #   * ``"flash_mla"`` — Phase-1 fused backend: forward uses
    #     :func:`flash_mla.flash_mla_sparse_fwd` (DeepSeek-AI's FlashMLA C++
    #     kernel, requires SM90/SM100 and ``pip install`` of the FlashMLA repo);
    #     backward re-runs the native sparse_attn under an autograd.Function so
    #     numerical gradients match ``"native"`` exactly. Future phases will
    #     swap the backward path to cudnn-frontend ``SparseAttentionBackward``.
    #   * ``"cudnn"``     — Phase-2 fused backend: forward stays on FlashMLA
    #     (same as ``"flash_mla"``); backward uses cudnn-frontend's
    #     ``sparse_attention_backward_wrapper`` (DSA bwd, FlashMLA-shape,
    #     SM90/SM100). Requires ``pip install nvidia-cudnn-frontend>=1.24.0``.
    backend: Annotated[Literal["native", "flash_mla", "cudnn"], Parameter(group="attention")] = "native"
    # Static upper bound on the packed query length (``DataloaderConfig.pack_max_length``).
    # HCA (``compress_ratio=128``) needs it to pre-allocate the constant ``-1``
    # pad buffer that brings the per-layer ``topk`` width to a multiple of
    # FlashMLA's 128 alignment — HCA's natural width
    # ``sliding_window + pack // 128 + 1`` is never aligned. When ``None``,
    # HCA layers on a flash_mla/cudnn backend fall back to native sparse_attn
    # (slower; see :meth:`DeepSeekSparseAttention._resolve_sparse_attn_fn`).
    # CSA (``compress_ratio=4``) and sliding-only (``compress_ratio=0``)
    # layers do not consult this field.
    pack_max_length: Annotated[int | None, Parameter(group="attention")] = None
    # Backend for the Indexer's top-k score-and-select path. ``"triton"`` runs
    # the varlen tensor-core kernel in :mod:`._indexer_topk_triton` under
    # ``torch.no_grad()`` (the indexer's output indices have no gradient anyway
    # because the downstream ``gather`` blocks it). At V4 production dims this
    # is ~1.6× faster than the native einsum loop AND avoids materialising the
    # ``[1, S_i, n_heads, T_i]`` fp32 score tensor (~4.5 GB per layer at
    # pack=4096) that is the root cause of the recompute-time 130 GB OOM that
    # forced :data:`V4_EP_COMPILE_CFG` to be empty. ``"native"`` keeps the
    # pure-PyTorch reference for parity testing.
    indexer_backend: Annotated[Literal["native", "triton"], Parameter(group="attention")] = "triton"

    def build(
        self,
        *,
        hidden_size: int,
        layer_idx: int,
        compress_ratio: int,
    ) -> "DeepSeekSparseAttention":
        return DeepSeekSparseAttention(
            dsa_cfg=self,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            compress_ratio=compress_ratio,
        )


class DeepSeekSparseAttention(nn.Module):
    """DeepSeek-V4 Sparse Attention with grouped O-LoRA and learnable sink.

    Each layer combines:

    * **Q-LoRA**: ``wq_a → q_norm → wq_b`` produces the multi-head queries.
    * **MQA K/V**: a single ``wkv`` projects hidden states straight to one
      shared key/value vector of width ``head_dim``.
    * **Per-layer sparsity** (driven by ``compress_ratio``):
        - ``0``: pure sliding-window attention, no Compressor, no Indexer.
        - ``4``: window + compressed-KV from a dedicated :class:`KVCompressor`
          + per-query top-k chosen by :class:`Indexer`.
        - ``128``: window + compressed-KV + deterministic positional top-k
          (no Indexer).
    * **Grouped O-LoRA**: ``wo_a`` weight is reshaped into
      ``[o_groups, o_lora_rank, head_dim_per_group]`` and contracted with
      grouped attention output, then ``wo_b`` projects back to
      ``hidden_size``.
    * **Attention sink**: a learnable per-head fp32 logit concatenated to the
      sparse-attn softmax so the softmax can route probability mass onto a
      "no-op" target when no KV slot is relevant.

    Args:
        dsa_cfg (DSAConfig): Static DSA hyper-parameters.
        hidden_size (int): Model hidden size.
        layer_idx (int): Layer index, used only for module naming.
        compress_ratio (int): Per-layer compression mode pulled from
            ``RopeParametersConfig.compress_ratios[layer_idx]``. Must be in
            ``{0, 4, 128}``.
    """

    def __init__(
        self,
        dsa_cfg: DSAConfig,
        hidden_size: int,
        layer_idx: int,
        compress_ratio: int,
    ) -> None:
        super().__init__()
        if compress_ratio not in (0, 4, 128):
            raise ValueError(f"compress_ratio must be one of {{0, 4, 128}}; got {compress_ratio}")
        if dsa_cfg.head_dim <= dsa_cfg.qk_rope_head_dim:
            raise ValueError(
                f"head_dim ({dsa_cfg.head_dim}) must exceed qk_rope_head_dim ({dsa_cfg.qk_rope_head_dim})"
            )
        if dsa_cfg.qk_rope_head_dim % 2 != 0:
            raise ValueError(f"qk_rope_head_dim must be even for rotate-half rope; got {dsa_cfg.qk_rope_head_dim}")
        n_heads_x_dim = dsa_cfg.num_attention_heads * dsa_cfg.head_dim
        if n_heads_x_dim % dsa_cfg.o_groups != 0:
            raise ValueError(
                f"num_attention_heads * head_dim ({n_heads_x_dim}) must be divisible by o_groups ({dsa_cfg.o_groups})"
            )

        self.name = f"layers.{layer_idx}.self_attn"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.compress_ratio = compress_ratio
        self.backend = dsa_cfg.backend
        self.num_attention_heads = dsa_cfg.num_attention_heads
        self.num_key_value_heads = dsa_cfg.num_key_value_heads
        self.head_dim = dsa_cfg.head_dim
        self.qk_rope_head_dim = dsa_cfg.qk_rope_head_dim
        self.q_lora_rank = dsa_cfg.q_lora_rank
        self.o_lora_rank = dsa_cfg.o_lora_rank
        self.o_groups = dsa_cfg.o_groups
        self.sliding_window = dsa_cfg.sliding_window
        self.rms_norm_eps = dsa_cfg.rms_norm_eps
        # why: V4 uses dense softmax_scale = head_dim ** -0.5 (model.py L464);
        # YaRN-style mscale is folded into the rope cos/sin via
        # `attention_scaling` upstream, so we do not multiply it in twice here.
        self.softmax_scale = dsa_cfg.head_dim**-0.5

        # Q-LoRA path: hidden → q_lora_rank → (n_heads, head_dim).
        self.wq_a = nn.Linear(hidden_size, dsa_cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(dsa_cfg.q_lora_rank, eps=dsa_cfg.rms_norm_eps)
        self.wq_b = nn.Linear(dsa_cfg.q_lora_rank, dsa_cfg.num_attention_heads * dsa_cfg.head_dim, bias=False)

        # MQA K/V: single head shared across all queries.
        self.wkv = nn.Linear(hidden_size, dsa_cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(dsa_cfg.head_dim, eps=dsa_cfg.rms_norm_eps)

        # Grouped O-LoRA. `wo_a` is conceptually a [o_groups, o_lora_rank,
        # head_dim_per_group] tensor that we'll reshape at forward time; the
        # storage layout matches V4 reference L462.
        head_dim_per_group = dsa_cfg.num_attention_heads * dsa_cfg.head_dim // dsa_cfg.o_groups
        self.head_dim_per_group = head_dim_per_group
        self.wo_a = nn.Linear(head_dim_per_group, dsa_cfg.o_groups * dsa_cfg.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(dsa_cfg.o_groups * dsa_cfg.o_lora_rank, hidden_size, bias=False)

        # why: V4 stores `attn_sink` in fp32 (model.py L456) because the sink
        # logit competes head-to-head with attention logits during the softmax,
        # and fp16/bf16 underflow on a single per-head scalar would silently
        # eliminate the sink's effect. Keep it fp32 even when the rest of the
        # module runs in bf16.
        self.use_attn_sink = dsa_cfg.use_attn_sink
        if self.use_attn_sink:
            self.attn_sink = nn.Parameter(torch.zeros(dsa_cfg.num_attention_heads, dtype=torch.float32))
        else:
            self.register_parameter("attn_sink", None)

        # Compressor + Indexer are layer-mode dependent. Compressed layers
        # (ratio in {4, 128}) own a private KVCompressor — note this is
        # **separate** from the Indexer's internal Compressor (which also
        # exists at ratio == 4 with Hadamard rotation enabled).
        if compress_ratio > 0:
            self.compressor = KVCompressor(
                hidden_size=hidden_size,
                head_dim=dsa_cfg.head_dim,
                compress_ratio=compress_ratio,
                overlap=(compress_ratio == 4),
                rotate=False,
                qk_rope_head_dim=dsa_cfg.qk_rope_head_dim,
                rms_norm_eps=dsa_cfg.rms_norm_eps,
            )
        else:
            self.compressor = None  # type: ignore[assignment]

        if compress_ratio == 4:
            self.indexer = Indexer(
                IndexerConfig(
                    hidden_size=hidden_size,
                    q_lora_rank=dsa_cfg.q_lora_rank,
                    index_n_heads=dsa_cfg.index_n_heads,
                    index_head_dim=dsa_cfg.index_head_dim,
                    rope_head_dim=dsa_cfg.qk_rope_head_dim,
                    index_topk=dsa_cfg.index_topk,
                    compress_ratio=4,
                    rms_norm_eps=dsa_cfg.rms_norm_eps,
                    backend=dsa_cfg.indexer_backend,
                )
            )
        else:
            self.indexer = None  # type: ignore[assignment]

        # Bind the sparse-attn callable once, at construction time. The previous
        # design called ``cudnn_sparse_attn`` / ``flash_mla_sparse_attn`` and let
        # them do a runtime check
        # ``if not _flash_mla_topk_ok(topk_idxs.size(-1)): return _native_sparse_attn(...)``
        # — which is fine in eager but a trap under ``torch.compile`` +
        # ``dynamic=True``: dynamo treats ``topk_idxs.size(-1)`` as a symbolic
        # int (the size comes from ``cat(window_topk, compress_topk).size(-1)``
        # whose dims are module attributes), can't constant-fold ``% 128``, and
        # ends up baking the native-fallback branch into the compiled graph even
        # for layers whose topk *does* satisfy the alignment requirement. The
        # baked-in native branch then triggers the 32 GiB ``expand+gather``
        # materialisation in inductor codegen.
        #
        # Decide statically here: we know ``sliding_window`` and ``index_topk``
        # at __init__ time, so we know exactly which topk_max each layer will
        # see and whether FlashMLA can run it. The forward path becomes a single
        # function-pointer call with no branch.
        self._sparse_attn_fn = self._resolve_sparse_attn_fn(dsa_cfg)

        # HCA pad buffer: sized once from ``pack_max_length`` so the forward
        # path is one slice + cat — no Python-side cache check, no
        # ``torch.compiler.disable`` helper, no graph break. The
        # ``_hca_uses_padded_path`` flag is purely a const at trace time, so
        # compile sees a straight tensor-op pipeline. Pad cols (and the
        # static-max compress-topk width that pairs with it) are computed
        # here so forward can use Python ints — see below.
        self._hca_uses_padded_path = (
            compress_ratio == 128
            and dsa_cfg.pack_max_length is not None
            and self._sparse_attn_fn is not _native_sparse_attn
        )
        if self._hca_uses_padded_path:
            assert dsa_cfg.pack_max_length is not None  # narrow for mypy
            pack_max_length = dsa_cfg.pack_max_length
            # Build ``compress_topk`` at this static width every forward,
            # regardless of actual ``total_tokens`` ≤ pack_max_length. Tokens
            # short of the natural horizon already get ``-1`` from
            # ``_build_compress_topk_idxs_varlen``'s clamp, so the only
            # consequence is some extra -1 entries inside the valid block,
            # which sparse_attn masks the same way as the trailing pad.
            self._hca_max_compress_w = pack_max_length // compress_ratio + 1
            natural_topk = dsa_cfg.sliding_window + self._hca_max_compress_w
            padded_topk = ((natural_topk + _FLASH_MLA_TOPK_ALIGN - 1) // _FLASH_MLA_TOPK_ALIGN) * _FLASH_MLA_TOPK_ALIGN
            self._hca_pad_cols = padded_topk - natural_topk
            # ``_shift_topk_to_global(...).int()`` is int32, so the cat partner
            # must also be int32. At V4 prod dims (pack=16384, pad_cols≈127)
            # this is ~8 MB per HCA layer.
            self.register_buffer(
                "_compress_topk_pad",
                torch.full(
                    (1, pack_max_length, self._hca_pad_cols),
                    -1,
                    dtype=torch.int32,
                ),
                persistent=False,
            )
        else:
            self._hca_max_compress_w = 0
            self._hca_pad_cols = 0

    def _resolve_sparse_attn_fn(self, dsa_cfg: "DSAConfig"):
        # Per-layer topk_max for the FlashMLA-alignment decision:
        #   compress_ratio == 0   : window_topk only
        #                           → topk_max = sliding_window
        #   compress_ratio == 4   : window + Indexer top-K (CSA, content-adaptive)
        #                           → topk_max = sliding_window + index_topk
        #   compress_ratio == 128 : window + deterministic positional top-K (HCA)
        #                           → natural width sliding_window + pack/128 + 1
        #                             is never 128-aligned. With
        #                             ``pack_max_length`` known at config time
        #                             we ceil-round to the next multiple of 128
        #                             and pad the extra compress-topk columns
        #                             with ``-1`` (mask slot). Without
        #                             ``pack_max_length`` we cannot size the
        #                             static pad buffer, so HCA falls back to
        #                             native.
        if self.compress_ratio == 0:
            topk_max = dsa_cfg.sliding_window
            flashmla_compatible = _flash_mla_topk_ok(topk_max)
        elif self.compress_ratio == 4:
            topk_max = dsa_cfg.sliding_window + dsa_cfg.index_topk
            flashmla_compatible = _flash_mla_topk_ok(topk_max)
        else:
            # compress_ratio == 128 (HCA)
            if dsa_cfg.pack_max_length is None:
                topk_max = -1
                flashmla_compatible = False
            else:
                natural = dsa_cfg.sliding_window + dsa_cfg.pack_max_length // self.compress_ratio + 1
                topk_max = ((natural + _FLASH_MLA_TOPK_ALIGN - 1) // _FLASH_MLA_TOPK_ALIGN) * _FLASH_MLA_TOPK_ALIGN
                flashmla_compatible = _flash_mla_topk_ok(topk_max)

        # Pick the function. When the user asked for flash_mla / cudnn but
        # this specific layer can't run it, warn explicitly so the user
        # knows which layer fell back and why — silent fallback inside the
        # forward (the previous design) hid this.
        if dsa_cfg.backend in ("flash_mla", "cudnn") and not flashmla_compatible:
            if self.compress_ratio == 128 and dsa_cfg.pack_max_length is None:
                reason = (
                    "compress_ratio=128 (HCA) needs DSAConfig.pack_max_length set "
                    "to size the static -1 pad buffer; without it the topk width is "
                    "pack-dependent and never 128-aligned. Set "
                    "``moe_cfg.attention.pack_max_length = "
                    "dataloader_cfg.pack_max_length`` in your training config to "
                    "enable the HCA padded path"
                )
            else:
                reason = f"topk_max={topk_max} is not a multiple of FlashMLA's 128-alignment"
            logger.warning(
                "DSA layer %d: backend=%r requested but %s — falling back to native for this layer.",
                self.layer_idx,
                dsa_cfg.backend,
                reason,
            )
            return _native_sparse_attn

        if dsa_cfg.backend == "flash_mla":
            return _flash_mla_sparse_attn_apply
        if dsa_cfg.backend == "cudnn":
            return _cudnn_sparse_attn_apply
        return _native_sparse_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        """Run DSA forward over a packed varlen batch.

        Args:
            hidden_states (torch.Tensor): Packed hidden states shaped
                ``[1, total_tokens, hidden_size]``.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]):
                ``(cos, sin)`` for the sliding-window rope basis
                (``rope_theta`` without yarn). Each tensor is shaped
                ``[1, total_tokens, qk_rope_head_dim]`` and uses the
                rotate-half convention emitted by
                :class:`xtuner.v1.module.rope.RotaryEmbedding`.
            position_embeddings_compressed (tuple[torch.Tensor, torch.Tensor] | None):
                ``(cos, sin)`` for the yarn'd ``compress_rope_theta`` basis,
                same shape as ``position_embeddings``. Required when
                ``compress_ratio == 4`` (consumed by the Indexer); ignored
                when ``compress_ratio in {0, 128}``.
            seq_ctx (SequenceContext): Carries the per-sample ``cu_seq_lens``
                used to chunk the packed batch into independent samples.

        Returns:
            AttnOutputs: Dict with ``raw_output`` (pre-O-LoRA per-head
            output), ``projected_output`` (after grouped O-LoRA, the only
            tensor consumed by ``MoEDecoderLayer`` in training), and
            ``softmax_lse=None`` (the pure-PyTorch sparse_attn does not emit
            an LSE).
        """
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise ValueError(
                "DSA expects packed varlen hidden states with shape "
                f"[1, total_tokens, hidden_size]; got {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"hidden_states last dim {hidden_states.size(-1)} != hidden_size {self.hidden_size}")
        if self.compress_ratio > 0 and position_embeddings_compressed is None:
            raise ValueError(
                "position_embeddings_compressed is required for compress_ratio > 0 "
                "(the KVCompressor rotates compressed-kv with the compressed-rope basis to "
                "match V4 reference Compressor.forward, mirroring HF "
                "DeepseekV4{CSA,HCA}Compressor)"
            )

        cos, sin = position_embeddings
        total_tokens = hidden_states.size(1)
        if cos.shape[-2] != total_tokens or sin.shape[-2] != total_tokens:
            raise ValueError(
                "position_embeddings must carry one (cos, sin) row per token; "
                f"got cos {tuple(cos.shape)}, sin {tuple(sin.shape)}, expected token dim {total_tokens}"
            )
        # ``DualRotaryEmbedding`` now emits D-dim cos/sin pre-arranged so the
        # per-layer ``_apply_rope`` is one ``x * cos + flip_pairs(x) * sin``
        # (sign pattern is folded into ``sin``). See the rope module's
        # ``DualRotaryEmbedding.forward`` docstring for the layout.
        if cos.size(-1) != self.qk_rope_head_dim:
            raise ValueError(
                "position_embeddings last dim must equal qk_rope_head_dim "
                f"({self.qk_rope_head_dim}); got {cos.size(-1)}"
            )

        # 1. Q-LoRA path. `q_lowrank` is reused by the Indexer (V4 reuses
        # the same low-rank Q stream for the score path).
        q_lowrank = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_lowrank).unflatten(-1, (self.num_attention_heads, self.head_dim))
        # V4 applies a second per-head RMSNorm to q (model.py L498), separate
        # from ``q_norm`` which acted on the low-rank stream. HF's
        # ``DeepseekV4UnweightedRMSNorm`` (modeling_deepseek_v4.py:66) computes
        # the square in fp32 — ``x.float().square().mean(-1) + eps`` — and only
        # casts the rsqrt back to bf16 before the multiply. We match that here.
        #
        # Under ``torch.compile`` the fp32 ``q.float().square()`` does not
        # materialise to HBM (inductor fuses square + mean + rsqrt + multiply
        # into one kernel, the fp32 intermediate stays in registers). In eager
        # this is ~128 MB transient at pack=4096 (one fp32 copy of q), short-
        # lived (freed after the rsqrt). The earlier bf16-square shortcut here
        # introduced a ~3% absolute divergence vs HF on this op alone — see
        # ``test_subcomponent_probe`` in
        # ``tests/model/test_deepseek_v4_decoder_layer_parity.py``.
        if _HAS_QUACK:
            q = _quack.rmsnorm(q, weight=None, eps=self.rms_norm_eps)
        else:
            q_inv_norm = torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(q.dtype)
            q = q * q_inv_norm
        q = _apply_rope_split(q, cos, sin, self.qk_rope_head_dim)

        # 2. KV (single head, MQA).
        kv = self.kv_norm(self.wkv(hidden_states))  # [1, S, head_dim]
        kv = _apply_rope_split(kv, cos, sin, self.qk_rope_head_dim)

        # 3. Single varlen forward over the whole pack. The three sparse-attn
        # backends (native / flash_mla / cudnn) all gather kv by global index
        # and have no notion of samples; cross-sample isolation is the caller's
        # responsibility — i.e. *ours* — and is achieved by
        #   (a) laying ``kv_full`` out as ``[W_0, C_0, W_1, C_1, ...]`` so each
        #       sample owns a contiguous slab, and
        #   (b) building ``topk_idxs`` whose entries are sample-local then
        #       shifted by ``cu_packed[sample_id]`` (+ ``q_len[sample_id]`` for
        #       the compressed half), so no token can attend across sample
        #       boundaries.
        # This replaces an old per-sample Python loop that called the compressor,
        # indexer, and sparse_attn N times with one .cpu()/.tolist() sync at the
        # top — N times per layer, per micro-batch, per step.
        cu_q = seq_ctx.cu_seq_lens_q
        device = q.device

        window_topk = _build_window_topk_idxs_varlen(self.sliding_window, cu_q, total_tokens)

        if self.compress_ratio > 0:
            # Compressor / Indexer accept cu_seq_lens by API today but still
            # loop per-sample internally; Phase 2/3 will vectorise them. Even
            # so, the single outer call eliminates N entry/exit pairs and
            # batches the GEMMs at the layer boundary.
            assert self.compressor is not None  # compress_ratio > 0 always materialises it
            # Compressed boundaries are built once per compress_ratio at model forward and cached
            # on seq_ctx (DeepSeekV4._assign_compressed_cu_seq_lens); reuse this layer's instead of
            # recomputing the cumsum + H2D. ``None`` (e.g. standalone DSA in a unit test) falls back
            # to the compressor building it. The Indexer's internal compressor shares this ratio's
            # value. ``.get`` over the constant ``self.compress_ratio`` key stays compile-traceable.
            cu_seq_lens_out = (
                seq_ctx.compressed_cu_seq_lens.get(self.compress_ratio)
                if seq_ctx.compressed_cu_seq_lens is not None
                else None
            )
            kv_compressed, cu_c = self.compressor(
                hidden_states,
                cu_q,
                cu_seq_lens_cpu=seq_ctx.cu_seq_lens_q_cpu,
                position_embeddings_compressed=position_embeddings_compressed,
                cu_seq_lens_out=cu_seq_lens_out,
            )  # [1, total_c, D], [B+1]
            if self.compress_ratio == 4:
                # ``DualRotaryEmbedding`` already emits half-dim cos/sin in the
                # interleaved convention the Indexer expects; pass through.
                cos_c, sin_c = position_embeddings_compressed  # type: ignore[misc]
                assert self.indexer is not None  # compress_ratio == 4 always materialises it
                # ``self.indexer`` emits the top-k *indices* that feed into
                # sparse_attn's ``gather``; ``gather`` propagates no gradient
                # through its index argument, so nothing inside the Indexer
                # (wq_b, weights_proj, internal KVCompressor, RoPE, the score
                # path) ever receives backprop in the current V4 design. Running
                # the entire call under ``torch.no_grad()`` saves the autograd
                # state these ops would otherwise stash (Linear-input saves on
                # ``q_lowrank`` / ``hidden_states``, the compressor's scatter
                # buffers, the softmax outputs) without changing any tensor that
                # the rest of the model can see — and gives the surrounding
                # ``DSA.forward`` compile graph one clean ``no_grad`` subregion
                # to fold into. Remove this wrap if a future variant adds an
                # auxiliary loss directly on Indexer outputs.
                with torch.no_grad():
                    compress_topk = self.indexer(
                        hidden_states,
                        q_lowrank,
                        (cos_c, sin_c),
                        cu_q,
                        cu_seq_lens_cpu=seq_ctx.cu_seq_lens_q_cpu,
                        cu_seq_lens_out=cu_seq_lens_out,
                    )
            else:
                # compress_ratio == 128: deterministic positional top-k.
                #
                # On the HCA padded path (``pack_max_length`` set + flash_mla/
                # cudnn backend), pin the build width to the static
                # ``_hca_max_compress_w`` so the trailing pad slice is also
                # static and ``torch.cat`` lowers cleanly. Tokens below the
                # natural horizon already get ``-1`` from
                # ``_build_compress_topk_idxs_varlen``'s clamp; the only
                # consequence of the larger build is extra -1 entries inside
                # the valid block, which sparse_attn masks identically to the
                # trailing pad.
                #
                # On the native path, use the historic dynamic width — native
                # has no alignment constraint and a smaller build saves a few
                # KB on short batches.
                if self._hca_uses_padded_path:
                    max_compressed_width = self._hca_max_compress_w
                else:
                    max_compressed_width = total_tokens // self.compress_ratio + 1
                compress_topk = _build_compress_topk_idxs_varlen(
                    self.compress_ratio, cu_q, cu_c, total_tokens, max_compressed_width
                )
        else:
            kv_compressed = kv.new_zeros(1, 0, kv.size(-1))
            cu_c = torch.zeros_like(cu_q)
            compress_topk = None

        # Sample-interleaved kv_full layout and the cumulative offsets we shift
        # topk indices against.
        q_lens = cu_q[1:] - cu_q[:-1]
        c_lens = cu_c[1:] - cu_c[:-1]
        packed_lens = q_lens + c_lens
        cu_packed = torch.zeros(packed_lens.numel() + 1, dtype=cu_q.dtype, device=device)
        cu_packed[1:] = torch.cumsum(packed_lens, dim=0)

        kv_full = _interleave_window_compressed_kv(kv, kv_compressed, cu_q, cu_c, cu_packed)
        topk_idxs = _shift_topk_to_global(window_topk, compress_topk, cu_q, cu_packed).int()

        # HCA padded path: slice the pre-allocated ``[1, pack_max_length, _hca_pad_cols]``
        # ``-1`` buffer down to the current ``total_tokens`` and cat it onto
        # ``topk_idxs`` to lift the trailing dim to a multiple of FlashMLA's
        # 128 alignment. The buffer is registered at ``__init__`` so this is a
        # view + a single ``torch.cat`` — no Python-side check, no graph break.
        # ``self._hca_uses_padded_path`` is a constant attribute; dynamo folds
        # the ``if`` away at trace time.
        if self._hca_uses_padded_path:
            pad_slice = self._compress_topk_pad[:, :total_tokens, :]
            topk_idxs = torch.cat([topk_idxs, pad_slice], dim=-1)

        # Under ep_size > 1, ``_replicate_other_params`` wraps ``self.attn_sink``
        # as a Replicate-on-ep DTensor. ``sparse_attn`` cats it with plain-tensor
        # logits via ``torch.cat``, which would crash with "mixed Tensor and DTensor".
        # ``.to_local()`` is a no-op on plain tensors (ep=1 path).
        if self.attn_sink is None:
            attn_sink = q.new_zeros(self.num_attention_heads, dtype=torch.float32)
        else:
            from torch.distributed.tensor import DTensor as _DTensor

            attn_sink = self.attn_sink.to_local() if isinstance(self.attn_sink, _DTensor) else self.attn_sink

        attn_out = self._sparse_attn_fn(q, kv_full, attn_sink, topk_idxs, self.softmax_scale, cu_q)

        # 4. De-rotate the rope tail on the output (V4 reference L534) over the
        # whole packed batch; the op is per-token so it cancels rope cleanly on
        # each sample without needing per-sample slicing.
        raw_output = _apply_rope_inverse_split(attn_out, cos, sin, self.qk_rope_head_dim)

        # 5. Grouped O-LoRA. `wo_a` is conceptually
        # `[o_groups, o_lora_rank, head_dim_per_group]`; the einsum mirrors
        # V4 reference L538-541. We access `.weight` directly (bypassing the
        # Linear forward) so the patched `nn.Linear.forward` `.to_local()`
        # shim doesn't fire here — do the unwrap explicitly to keep ep>1 working.
        from torch.distributed.tensor import DTensor as _DTensor

        wo_a_weight = self.wo_a.weight
        if isinstance(wo_a_weight, _DTensor):
            wo_a_weight = wo_a_weight.to_local()
        o_grouped = raw_output.reshape(1, raw_output.size(1), self.o_groups, self.head_dim_per_group)
        wo_a_view = wo_a_weight.view(self.o_groups, self.o_lora_rank, self.head_dim_per_group)
        o_proj = torch.einsum("bsgd,grd->bsgr", o_grouped, wo_a_view)  # [1, S, o_groups, o_lora_rank]
        projected_output = self.wo_b(o_proj.flatten(2))  # [1, S, hidden_size]

        attn_outputs: AttnOutputs = {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": None,
        }
        return attn_outputs

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.wq_a.weight)
        nn.init.xavier_uniform_(self.wq_b.weight)
        nn.init.xavier_uniform_(self.wkv.weight)
        nn.init.xavier_uniform_(self.wo_a.weight)
        nn.init.xavier_uniform_(self.wo_b.weight)
        self.q_norm.init_weights()
        self.kv_norm.init_weights()
        if self.attn_sink is not None:
            nn.init.zeros_(self.attn_sink)
        if self.compressor is not None:
            self.compressor.init_weights()
        if self.indexer is not None:
            self.indexer.init_weights()
