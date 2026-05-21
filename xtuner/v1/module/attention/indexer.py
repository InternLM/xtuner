# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Portions of this file are adapted from DeepSeek-V4-Flash `inference/model.py`
# (function `rotate_activation` lines 247-253, class `Indexer` lines 380-433),
# Copyright (c) DeepSeek-AI, released under the MIT License.
# Upstream reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# Local cache: .dev_scripts/deepseek_v4_reference/model.py
#
# Only the start_pos == 0 (prefill / training) branch is retained; all
# inference-time kv_cache, freqs_cis buffers, FP4/FP8 quantization, and tensor
# parallelism are intentionally dropped. Packed varlen samples are processed
# one at a time via cu_seq_lens (matching KVCompressor's per-sample loop)
# rather than the fixed-batch tensors used by the upstream reference.
# ============================================================================

from typing import Annotated, Literal

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn

from .kv_compressor import KVCompressor


class IndexerConfig(BaseModel):
    """Configuration for the DeepSeek-V4 sparse-attention Indexer.

    The Indexer scores compressed-KV positions for each query and selects the
    top-``index_topk`` per query for the downstream sparse-attention path.
    Only DSA layers with ``compress_ratio == 4`` build an Indexer; layers with
    ``compress_ratio == 128`` use a deterministic positional top-k instead.

    Args:
        hidden_size (int): Input feature size of the packed hidden states
            (e.g. 4096 in V4).
        q_lora_rank (int): Width of the low-rank Q output produced by DSA's
            ``q_norm(wq_a(x))``; the Indexer expands it back to head space.
        index_n_heads (int): Number of scoring heads (64 in V4).
        index_head_dim (int): Per-head dimension used for both the Q
            projection and the compressed-KV stream (128 in V4).
        rope_head_dim (int): Subset of each head dim that carries rotary
            position info (64 in V4); the remaining dims are NoPE.
        index_topk (int): Maximum number of compressed positions selected per
            query (512 in V4). The effective k is clamped to the per-sample
            compressed length.
        compress_ratio (int): Compression ratio of the internal Compressor.
            Must be ``4`` in V4 — only ratio-4 DSA layers carry an Indexer.
        rms_norm_eps (float): Epsilon used by the internal Compressor's
            trailing RMSNorm. Defaults to ``1e-6``.
    """

    model_config = ConfigDict(title="DeepSeek-V4 Indexer config for xtuner", extra="forbid")
    hidden_size: Annotated[int, Parameter(group="indexer")]
    q_lora_rank: Annotated[int, Parameter(group="indexer")]
    index_n_heads: Annotated[int, Parameter(group="indexer")]
    index_head_dim: Annotated[int, Parameter(group="indexer")]
    rope_head_dim: Annotated[int, Parameter(group="indexer")]
    index_topk: Annotated[int, Parameter(group="indexer")]
    compress_ratio: Annotated[int, Parameter(group="indexer")]
    rms_norm_eps: float = 1e-6
    # Forward backend selector:
    #   * "native"  — pure-PyTorch per-sample loop (default, gradient-safe).
    #   * "triton"  — varlen fused kernel; eliminates the [S_i, n_heads, T_i]
    #     fp32 intermediate (17 GiB at pack=8192 / n_heads=64). FORWARD ONLY,
    #     wraps the call in ``torch.no_grad()``; intended for V4 fine-tuning
    #     where Indexer params arrive pre-trained and the topk → gather path
    #     blocks gradient flow anyway. Switch off for any setup that adds an
    #     aux loss against the indexer scores.
    backend: Annotated[Literal["native", "triton"], Parameter(group="indexer")] = "native"


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation along the last dim of ``x``.

    Spreads activation magnitude evenly across feature dims, which would
    normally improve FP4/FP8 quantization quality. We keep the rotation in
    training because the Indexer's scoring path must match inference layout
    bit-for-bit at deployment time.

    Tries the ``hadamard-transform`` PyPI package first (O(d log d) via the
    fast Walsh-Hadamard transform). Falls back to a dense Hadamard-matrix
    multiply if the package is missing; the fallback is O(d^2) but
    ``index_head_dim`` is 128 in V4, so 128^2 = 16384 muls per element is
    cheap relative to the surrounding einsums.

    Args:
        x (torch.Tensor): Tensor whose last dim is a power of two. Any shape
            is accepted; rotation is applied independently per row.

    Returns:
        torch.Tensor: Same shape and dtype as ``x``, rotated and rescaled by
        ``1 / sqrt(d)`` so the rotation is unitary.
    """
    d = x.size(-1)
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(f"rotate_activation requires last dim be a power of two; got {d}")

    try:
        from hadamard_transform import hadamard_transform  # type: ignore[import-not-found]

        return hadamard_transform(x, scale=d**-0.5)
    except ImportError:
        pass

    # Fallback: dense Hadamard via Sylvester's construction, cached per
    # (dim, dtype, device). Kept here rather than at module scope because
    # the cache only needs to materialize when the fast package is absent.
    h = _build_hadamard_matrix(d, x.dtype, x.device) * (d**-0.5)
    return torch.matmul(x, h)


_HADAMARD_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _build_hadamard_matrix(d: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]], H_1 = [[1]].
    # Cached so repeated calls with the same (d, dtype, device) don't rebuild
    # the matrix; the cache is small because every Indexer instance shares the
    # same head_dim.
    key = (d, dtype, device)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h = torch.ones((1, 1), dtype=dtype, device=device)
    size = 1
    while size < d:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
        size *= 2
    _HADAMARD_CACHE[key] = h
    return h


class Indexer(nn.Module):
    """Scores compressed-KV positions and emits top-k indices per query.

    Used by DeepSeek-V4 sparse-attention layers with ``compress_ratio == 4``.
    The module owns a private :class:`KVCompressor` (with ``overlap=True`` and
    ``rotate=True``) so it can build a Hadamard-rotated compressed-KV stream
    independent of the DSA layer's main Compressor.

    Args:
        config (IndexerConfig): Configuration carrying the head dims, rope
            split, top-k budget, and compression ratio.
    """

    def __init__(self, config: IndexerConfig) -> None:
        super().__init__()
        if config.compress_ratio != 4:
            raise ValueError(f"Indexer is only built for compress_ratio == 4 layers; got {config.compress_ratio}")
        if config.index_head_dim < config.rope_head_dim:
            raise ValueError(
                f"index_head_dim ({config.index_head_dim}) must be >= rope_head_dim ({config.rope_head_dim})"
            )

        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.compress_ratio = config.compress_ratio
        self.backend = config.backend
        # why: matches V4 reference `softmax_scale = head_dim ** -0.5` at
        # L395; the extra `n_heads ** -0.5` is fused into `weights_proj`
        # at forward time to keep the score on a stable scale.
        self.softmax_scale = config.index_head_dim**-0.5

        self.wq_b = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)
        # why: V4 stores weights_proj in bf16 (see model.py L394). We declare
        # the layer with the project default dtype here; the actual bf16
        # casting lives in the DSA layer's autocast / parameter-init policy
        # (PR5) so the Indexer stays dtype-neutral for the training reference.
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = KVCompressor(
            hidden_size=config.hidden_size,
            head_dim=config.index_head_dim,
            compress_ratio=config.compress_ratio,
            overlap=True,
            rotate=True,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lowrank: torch.Tensor,
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor],
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-query top-k compressed-KV indices.

        Args:
            hidden_states (torch.Tensor): Packed varlen hidden states with
                shape ``[1, total_tokens, hidden_size]`` (XTuner convention).
            q_lowrank (torch.Tensor): Low-rank query stream from DSA's
                ``q_norm(wq_a(x))`` with shape ``[1, total_tokens, q_lora_rank]``.
            position_embeddings_compressed (tuple[torch.Tensor, torch.Tensor]):
                ``(cos, sin)`` pair for the compressed-rope basis
                (``compress_rope_theta``). Each tensor is shaped
                ``[1, total_tokens, rope_head_dim]`` and is produced by the
                DSA layer's dual-rope module.
            cu_seq_lens (torch.Tensor): 1D int32 cumulative per-sample token
                counts with length ``num_samples + 1``.

        Returns:
            torch.Tensor: Top-k indices shaped ``[1, total_tokens, index_topk]``
            in int64. Entries that fall outside the query's causal horizon or
            beyond the per-sample compressed length are ``-1``.
        """
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise ValueError(
                "hidden_states must be packed varlen with shape [1, total_tokens, hidden_size]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if q_lowrank.dim() != 3 or q_lowrank.size(0) != 1 or q_lowrank.size(1) != hidden_states.size(1):
            raise ValueError(
                "q_lowrank must match hidden_states along (batch, tokens); "
                f"got q_lowrank {tuple(q_lowrank.shape)} vs hidden_states {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"hidden_states last dim {hidden_states.size(-1)} != hidden_size {self.hidden_size}")
        if q_lowrank.size(-1) != self.q_lora_rank:
            raise ValueError(f"q_lowrank last dim {q_lowrank.size(-1)} != q_lora_rank {self.q_lora_rank}")
        if cu_seq_lens.dim() != 1 or cu_seq_lens.numel() < 2:
            raise ValueError(f"cu_seq_lens must be 1D with at least 2 entries; got shape {tuple(cu_seq_lens.shape)}")

        cos, sin = position_embeddings_compressed
        total_tokens = hidden_states.size(1)
        if cos.shape[-2] != total_tokens or sin.shape[-2] != total_tokens:
            raise ValueError(
                "position_embeddings_compressed must carry one (cos, sin) row per query token; "
                f"got cos {tuple(cos.shape)}, sin {tuple(sin.shape)}, expected token dim {total_tokens}"
            )
        if cos.size(-1) != self.rope_head_dim // 2:
            raise ValueError(
                "position_embeddings_compressed last dim must equal rope_head_dim // 2 "
                f"({self.rope_head_dim // 2}); got {cos.size(-1)}"
            )

        # Step 1-2: expand qr to (n_heads, head_dim) and rotate the rope tail.
        q = self.wq_b(q_lowrank).unflatten(-1, (self.n_heads, self.head_dim))
        q_nope_tail = q[..., : self.head_dim - self.rope_head_dim]
        q_rope_tail = q[..., self.head_dim - self.rope_head_dim :]
        q_rope_tail = _apply_rope(q_rope_tail, cos, sin)
        q = torch.cat([q_nope_tail, q_rope_tail], dim=-1)

        # Step 3: Hadamard rotation across the head_dim axis.
        q = rotate_activation(q)

        # Step 4: build the per-sample compressed-KV stream. The compressor is
        # now varlen (single wkv/wgate GEMM over the full pack); we still need
        # Python-int boundaries for the per-sample score loop below.
        kv_compressed, compressed_cu_seq_lens = self.compressor(hidden_states, cu_seq_lens)

        # Step 5: gate weights, scaled exactly as V4 reference L418.
        weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.n_heads**-0.5)

        if self.backend == "triton":
            # Fused varlen kernel — no [total_q, n_heads, total_c] intermediate.
            # ``torch.no_grad`` is correct here: Indexer outputs flow into
            # sparse_attn's ``gather`` which has no gradient w.r.t. indices, so
            # there is no useful gradient to back-propagate through wq_b /
            # weights_proj / the internal Compressor on this path.
            from ._indexer_topk_triton import indexer_topk_triton

            with torch.no_grad():
                return indexer_topk_triton(
                    q,
                    kv_compressed,
                    weights,
                    cu_seq_lens,
                    compressed_cu_seq_lens,
                    ratio=self.compress_ratio,
                    index_topk=self.index_topk,
                    softmax_scale=1.0,  # already folded into ``weights``
                )

        # Single D2H transfer for both boundary tensors — one cudaMemcpy + one
        # stream sync instead of two. The per-sample loop below cannot be
        # vectorised cheaply: a full-pack ``[total_q, total_c]`` score matrix
        # would compute every cross-sample (q, kv) pair, only to mask them out
        # — at typical V4 pack/ratio settings (pack=8192, n_heads=64,
        # total_c=2048) that is a 137 GFLOPS Indexer per layer vs the
        # block-diagonal 8 GFLOPS we keep here. The remaining sync is the only
        # graph break inside this module.
        cu_stacked = torch.stack([cu_seq_lens, compressed_cu_seq_lens]).detach().cpu().tolist()
        boundaries, compressed_boundaries = cu_stacked[0], cu_stacked[1]
        num_samples = len(boundaries) - 1

        topk_pad = self.index_topk
        topk_idxs = q.new_full((1, total_tokens, topk_pad), -1, dtype=torch.long)

        ratio = self.compress_ratio
        for i in range(num_samples):
            q_start, q_end = boundaries[i], boundaries[i + 1]
            c_start, c_end = compressed_boundaries[i], compressed_boundaries[i + 1]
            sample_seqlen = q_end - q_start
            sample_clen = c_end - c_start
            if sample_seqlen == 0 or sample_clen == 0:
                continue

            q_i = q[:, q_start:q_end]
            kv_i = kv_compressed[:, c_start:c_end]
            w_i = weights[:, q_start:q_end]

            # Step 6: per-head dot, ReLU clipped (V4 reference L420-421).
            index_score = torch.einsum("bshd,btd->bsht", q_i, kv_i).relu_()
            # Step 7: head-axis weighted sum -> [1, S_i, T_i].
            index_score = (index_score * w_i.unsqueeze(-1)).sum(dim=2)

            # Step 8: causal mask. A query at offset s (1-indexed) may attend
            # to compressed positions strictly below `(s + 1) // ratio`,
            # matching the floor-division boundary in V4 reference L425.
            q_pos = torch.arange(1, sample_seqlen + 1, device=index_score.device).unsqueeze(1)
            c_pos = torch.arange(sample_clen, device=index_score.device).unsqueeze(0)
            causal_mask = c_pos >= (q_pos // ratio)
            index_score = index_score.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

            # Step 9: top-k along the compressed axis, clamped to the
            # per-sample compressed length to avoid out-of-range indices when
            # a sample is shorter than `index_topk * ratio`.
            k = min(self.index_topk, sample_clen)
            sample_topk = index_score.topk(k, dim=-1).indices

            # Step 10: re-apply the causal horizon to the chosen indices and
            # mark out-of-horizon picks as -1. This mirrors V4 L429-430 and
            # also defangs any -inf positions that tie at the top of `topk`.
            horizon = q_pos // ratio  # [S_i, 1]
            out_of_horizon = sample_topk >= horizon.unsqueeze(0)
            sample_topk = torch.where(out_of_horizon, torch.full_like(sample_topk, -1), sample_topk)

            # Pad the tail with -1 when the sample's compressed length is
            # shorter than the configured budget; downstream sparse_attn
            # treats -1 as a masked-out slot.
            if k < topk_pad:
                pad = sample_topk.new_full((1, sample_seqlen, topk_pad - k), -1)
                sample_topk = torch.cat([sample_topk, pad], dim=-1)

            topk_idxs[:, q_start:q_end, :] = sample_topk

        return topk_idxs

    def init_weights(self) -> None:
        nn.init.xavier_uniform_(self.wq_b.weight)
        nn.init.xavier_uniform_(self.weights_proj.weight)
        self.compressor.init_weights()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [1, S, H, rope_head_dim]; cos/sin: [1, S, rope_head_dim // 2].
    # Mirrors the V4 reference's `view_as_complex` rotation (model.py L235-
    # 243): consecutive pairs along the last dim of x are treated as the real
    # and imaginary parts of a complex number, and (cos[k], sin[k]) carry the
    # k-th frequency's rotation angle.
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(f"_apply_rope expects cos/sin of rank 3; got cos {tuple(cos.shape)}, sin {tuple(sin.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must share shape; got {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if x.size(-1) != 2 * cos.size(-1):
        raise ValueError(f"rope dim mismatch: x last dim {x.size(-1)} != 2 * cos last dim {2 * cos.size(-1)}")

    cos_b = cos.unsqueeze(2)  # [1, S, 1, rope_head_dim // 2]
    sin_b = sin.unsqueeze(2)
    x_pairs = x.float().unflatten(-1, (-1, 2))
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b
    rotated = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
    return rotated.to(x.dtype)
