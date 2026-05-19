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

from typing import Annotated

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn

from xtuner.v1.data_proto import SequenceContext

from ..rms_norm import RMSNorm
from .attn_outputs import AttnOutputs
from .indexer import Indexer, IndexerConfig
from .kv_compressor import KVCompressor
from .sparse_attn import sparse_attn


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
                )
            )
        else:
            self.indexer = None  # type: ignore[assignment]

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
        if self.compress_ratio == 4 and position_embeddings_compressed is None:
            raise ValueError("position_embeddings_compressed is required for compress_ratio == 4 (Indexer rope)")

        cos, sin = position_embeddings
        total_tokens = hidden_states.size(1)
        if cos.shape[-2] != total_tokens or sin.shape[-2] != total_tokens:
            raise ValueError(
                "position_embeddings must carry one (cos, sin) row per token; "
                f"got cos {tuple(cos.shape)}, sin {tuple(sin.shape)}, expected token dim {total_tokens}"
            )
        if cos.size(-1) != self.qk_rope_head_dim:
            raise ValueError(
                "position_embeddings last dim must equal qk_rope_head_dim "
                f"({self.qk_rope_head_dim}); got {cos.size(-1)}"
            )

        # 1. Q-LoRA path. `q_lowrank` is reused by the Indexer (V4 reuses
        # the same low-rank Q stream for the score path).
        q_lowrank = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_lowrank).unflatten(-1, (self.num_attention_heads, self.head_dim))
        # why: V4 applies a second per-head RMSNorm to q (model.py L498),
        # separate from `q_norm` which acted on the low-rank stream. This
        # stabilises the dot-product scale across heads.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(q.dtype)
        q = _apply_rope_split(q, cos, sin, self.qk_rope_head_dim)

        # 2. KV (single head, MQA).
        kv = self.kv_norm(self.wkv(hidden_states))  # [1, S, head_dim]
        kv = _apply_rope_split(kv, cos, sin, self.qk_rope_head_dim)

        # 3. Per-sample loop. We deviate from the V4 reference (which assumes
        # one contiguous sequence) because XTuner packs multiple samples per
        # forward; matching KVCompressor / Indexer per-sample convention is
        # the only way to avoid cross-sample contamination in the window /
        # compressed-KV indexing.
        cu = seq_ctx.cu_seq_lens_q.detach().cpu().tolist()
        num_samples = len(cu) - 1
        output_pieces: list[torch.Tensor] = []

        for i in range(num_samples):
            s0, s1 = cu[i], cu[i + 1]
            sample_len = s1 - s0
            if sample_len == 0:
                continue
            sample_cu = torch.tensor([0, sample_len], dtype=torch.int32, device=hidden_states.device)

            q_s = q[:, s0:s1]
            kv_s = kv[:, s0:s1]
            x_s = hidden_states[:, s0:s1]
            cos_s = cos[:, s0:s1]
            sin_s = sin[:, s0:s1]

            window_topk = _build_window_topk_idxs(self.sliding_window, sample_len, q.device)

            if self.compress_ratio > 0:
                kv_compressed, _ = self.compressor(x_s, sample_cu)  # [1, T_c, head_dim]
                t_c = kv_compressed.size(1)
                if self.compress_ratio == 4:
                    qlr_s = q_lowrank[:, s0:s1]
                    cos_c_full, sin_c_full = position_embeddings_compressed  # type: ignore[misc]
                    # why: the Indexer's internal RoPE uses the complex-pair
                    # (`view_as_complex`) convention from the V4 reference and
                    # therefore expects half-dim cos/sin. XTuner's dual-rope
                    # module emits full-dim rotate-half cos/sin which carries
                    # the same frequency vector duplicated as
                    # `cat((freqs, freqs), dim=-1)`; the first half is exactly
                    # the half-dim cos/sin the Indexer needs.
                    half = self.qk_rope_head_dim // 2
                    cos_c_s = cos_c_full[:, s0:s1, :half]
                    sin_c_s = sin_c_full[:, s0:s1, :half]
                    compress_topk = self.indexer(x_s, qlr_s, (cos_c_s, sin_c_s), sample_cu)
                else:
                    # compress_ratio == 128: deterministic positional top-k.
                    compress_topk = _build_compress_topk_idxs(self.compress_ratio, sample_len, t_c, q.device)

                # Compressed indices live in [0, t_c); after concatenating
                # window KV (length `sample_len`) and compressed KV in that
                # order, shift compressed entries by `sample_len`. -1 stays
                # -1 (masked-out slot in sparse_attn).
                compress_topk_shifted = torch.where(compress_topk == -1, compress_topk, compress_topk + sample_len)
                kv_full = torch.cat([kv_s, kv_compressed], dim=1)
                topk_idxs = torch.cat([window_topk, compress_topk_shifted], dim=-1)
            else:
                kv_full = kv_s
                topk_idxs = window_topk

            topk_idxs = topk_idxs.int()

            attn_sink = (
                self.attn_sink
                if self.attn_sink is not None
                else q_s.new_zeros(self.num_attention_heads, dtype=torch.float32)
            )

            o_s = sparse_attn(
                q_s,
                kv_full,
                attn_sink,
                topk_idxs,
                self.softmax_scale,
                sample_cu,
            )  # [1, sample_len, H, head_dim]

            # 4. De-rotate the rope tail on the output (V4 reference L534).
            # Forward and inverse cancel on the rope-carrying dims so the
            # output's rope tail is positionally neutral before O-LoRA.
            o_s = _apply_rope_inverse_split(o_s, cos_s, sin_s, self.qk_rope_head_dim)

            output_pieces.append(o_s)

        # Concatenate per-sample outputs back into the packed layout.
        if len(output_pieces) == 0:
            raw_output = q.new_zeros(1, 0, self.num_attention_heads, self.head_dim)
        else:
            raw_output = torch.cat(output_pieces, dim=1)

        # 5. Grouped O-LoRA. `wo_a` is conceptually
        # `[o_groups, o_lora_rank, head_dim_per_group]`; the einsum mirrors
        # V4 reference L538-541.
        o_grouped = raw_output.reshape(1, raw_output.size(1), self.o_groups, self.head_dim_per_group)
        wo_a_view = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, self.head_dim_per_group)
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


def _apply_rope_split(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    # Apply rotate-half rope to the final `rope_head_dim` slice of `x`,
    # leaving the NoPE prefix untouched. Concat is done at the end so the
    # output shape matches `x`.
    nope = x[..., : x.size(-1) - rope_head_dim]
    rope_tail = x[..., x.size(-1) - rope_head_dim :]
    rope_tail = _apply_rope(rope_tail, cos, sin)
    return torch.cat([nope, rope_tail], dim=-1)


def _apply_rope_inverse_split(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    nope = x[..., : x.size(-1) - rope_head_dim]
    rope_tail = x[..., x.size(-1) - rope_head_dim :]
    rope_tail = _apply_rope_inverse(rope_tail, cos, sin)
    return torch.cat([nope, rope_tail], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [..., S, *, rope_head_dim]; cos/sin: [B, S, rope_head_dim].
    # The cos/sin tensors from XTuner's RotaryEmbedding are already
    # `cat((freqs, freqs))` — i.e. each half spans the full rope dim — so the
    # rotate-half formula `x * cos + rotate_half(x) * sin` applies directly.
    cos_b, sin_b = _broadcast_cos_sin(cos, sin, x)
    return x * cos_b + _rotate_half(x) * sin_b


def _apply_rope_inverse(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # The inverse of rotate-half rope is `x * cos - rotate_half(x) * sin`,
    # derived by inverting the 2-D rotation per dim-pair (sin → -sin).
    cos_b, sin_b = _broadcast_cos_sin(cos, sin, x)
    return x * cos_b - _rotate_half(x) * sin_b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Standard rotate-half: split last dim in two halves, rotate by 90 deg
    # in the (x1, x2) plane.
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _broadcast_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin are [B, S, D]; x is either [B, S, D] (KV, MQA) or
    # [B, S, H, D] (queries / output). Insert a head axis when needed so the
    # broadcast multiplication doesn't accidentally flatten the heads.
    if x.dim() == cos.dim():
        return cos, sin
    if x.dim() == cos.dim() + 1:
        return cos.unsqueeze(-2), sin.unsqueeze(-2)
    raise ValueError(f"Cannot broadcast cos/sin {tuple(cos.shape)} against x {tuple(x.shape)}")


def _build_window_topk_idxs(window_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
    # Ports `get_window_topk_idxs` (model.py L262-264) for the start_pos == 0
    # branch. For each query position we list the indices of the up-to-
    # `window_size` preceding KV positions; entries that would point below 0
    # are masked with -1 and dropped by sparse_attn.
    base = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(1)
    matrix = (base - window_size + 1).clamp(min=0) + torch.arange(
        min(seqlen, window_size), device=device, dtype=torch.long
    )
    matrix = torch.where(matrix > base, torch.full_like(matrix, -1), matrix)
    return matrix.unsqueeze(0)  # [1, seqlen, min(seqlen, window_size)]


def _build_compress_topk_idxs(
    ratio: int,
    seqlen: int,
    t_compressed: int,
    device: torch.device,
) -> torch.Tensor:
    # Ports `get_compress_topk_idxs` (model.py L273-275) for the
    # start_pos == 0 branch. Each query position s (1-indexed) can attend to
    # compressed positions in [0, (s + 1) // ratio); anything outside this
    # horizon is marked -1. We deliberately clamp the column count to
    # `t_compressed` (the actual compressed length emitted by KVCompressor for
    # this sample) rather than `seqlen // ratio` — they differ when the sample
    # length is not a multiple of `ratio`, because KVCompressor pads to a
    # whole group.
    matrix = torch.arange(t_compressed, device=device, dtype=torch.long).repeat(seqlen, 1)
    horizon = torch.arange(1, seqlen + 1, device=device, dtype=torch.long).unsqueeze(1) // ratio
    matrix = torch.where(matrix >= horizon, torch.full_like(matrix, -1), matrix)
    return matrix.unsqueeze(0)  # [1, seqlen, t_compressed]
