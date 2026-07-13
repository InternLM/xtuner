# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Portions of this file are adapted from DeepSeek-V4-Flash `inference/model.py`
# (class `Compressor`, lines 279-379), Copyright (c) DeepSeek-AI, released
# under the MIT License.
# Upstream reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# Local cache: .dev_scripts/deepseek_v4_reference/model.py
#
# Only the start_pos == 0 (prefill / training) branch is retained; all
# inference-time KV cache, score state, RoPE and FP4/FP8 quantization paths
# are intentionally dropped. Per-sample boundaries are enforced via the
# XTuner-standard `cu_seq_lens` cumulative-length tensor instead of the
# fixed-batch tensors used by the upstream reference.
# ============================================================================

import torch
from torch import nn
from torch.distributed.tensor import DTensor as _DTensor

from ...rms_norm import RMSNorm
from ._dsa_rope import _apply_rope_split


class KVCompressor(nn.Module):
    """Learned gated pooling that compresses a packed KV sequence by
    `compress_ratio`.

    Used in DeepSeek-V4 sparse attention: once on the main attention path to
    produce a compressed KV stream for the sparse-attn kernel, and once inside
    the Indexer (with Hadamard-rotated input) to score candidate positions.

    The compressor projects each token into a key/value vector (`wkv`) and a
    scalar-style gate (`wgate`), groups consecutive ``compress_ratio`` tokens
    together, applies a learned absolute positional embedding (``ape``) inside
    each group, softmax-weights the gate over the group axis, and emits one
    pooled vector per group followed by RMSNorm.

    When ``overlap=True`` (only meaningful for ``compress_ratio == 4``), each
    compressed token additionally attends to the previous group; this is
    implemented by doubling the gate/value width (`coff = 1 + overlap`) and
    splicing the two halves across adjacent groups, matching the upstream
    ``Compressor.overlap_transform``.

    Args:
        hidden_size (int): Input feature size of the packed hidden states.
        head_dim (int): Per-head size of the compressed output (matches the
            attention head dim, e.g. 128 for the Indexer).
        compress_ratio (int): Number of input tokens collapsed into one
            compressed token. DeepSeek-V4 uses 4 (overlapping) and 128
            (non-overlapping).
        overlap (bool): Enable overlapping windows. Should only be set when
            ``compress_ratio == 4``. Defaults to ``False``.
        rotate (bool): Records whether the caller (the Indexer) will apply a
            Hadamard rotation to the inputs before forward. The rotation
            itself is performed by the caller, not by this module; we keep the
            flag purely so the caller can branch on a single, recorded source
            of truth. Defaults to ``False``.
        rms_norm_eps (float): Epsilon used by the trailing RMSNorm. Defaults
            to ``1e-6``.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        overlap: bool = False,
        rotate: bool = False,
        qk_rope_head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        # why: Hadamard rotation is applied by the caller (Indexer) before
        # forward; we just record the flag for caller-side branching.
        self.rotate = rotate
        # ``qk_rope_head_dim`` carries the length of the rope-tail slice that the
        # V4 reference rotates *inside* the compressor (HF
        # ``DeepseekV4{CSA,HCA}Compressor.forward`` apply ``apply_rotary_pos_emb``
        # to ``compressed[..., -rope_dim:]`` right after RMSNorm). When ``None``
        # the compressor stays rope-free for backward compatibility with the
        # pre-fix call sites; downstream code must then handle the rope itself.
        self.qk_rope_head_dim = qk_rope_head_dim

        coff = 1 + int(overlap)
        self._coff = coff

        self.wkv = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, eps=rms_norm_eps)
        # APE is added to the gate logits before softmax so the network can
        # learn position-within-window biases independent of input content.
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim))

    @staticmethod
    def build_cu_seq_lens_out(
        cu_seq_lens: torch.Tensor,
        cu_seq_lens_cpu: torch.Tensor | None,
        ratio: int,
    ) -> tuple[torch.Tensor, int]:
        """Per-sample compressed-chunk boundaries for a packed varlen batch.

        Hoisted out of :meth:`forward` so the model forward can compute it once per
        ``compress_ratio`` and hand it to every layer via
        :attr:`SequenceContext.compressed_cu_seq_lens` instead of every
        ``KVCompressor`` recomputing the cumsum + H2D per call.

        Args:
            cu_seq_lens (torch.Tensor): 1D int tensor, length ``num_samples + 1``,
                cumulative per-sample token counts (GPU).
            cu_seq_lens_cpu (torch.Tensor | None): CPU mirror of ``cu_seq_lens``.
                When provided, the cumsum runs on CPU and only the
                ``num_samples + 1`` result reaches the GPU via a non-blocking H2D,
                avoiding the host-blocking ``.item()`` D2H of the GPU fallback.
            ratio (int): ``compress_ratio`` (chunk width).

        Returns:
            tuple[torch.Tensor, int]: ``(cu_seq_lens_out, total_c)`` — the GPU
            compressed-boundary tensor (length ``num_samples + 1``) and the total
            compressed-chunk count.
        """
        if cu_seq_lens_cpu is not None:
            c_lens_cpu = (cu_seq_lens_cpu[1:] - cu_seq_lens_cpu[:-1] + ratio - 1) // ratio
            total_c = int(c_lens_cpu.sum())
            cu_out_cpu = torch.cat([torch.zeros(1, dtype=cu_seq_lens_cpu.dtype), torch.cumsum(c_lens_cpu, dim=0)])
            cu_seq_lens_out = cu_out_cpu.to(device=cu_seq_lens.device, dtype=cu_seq_lens.dtype, non_blocking=True)
        else:
            c_lens = (cu_seq_lens[1:] - cu_seq_lens[:-1] + ratio - 1) // ratio
            cu_seq_lens_out = torch.zeros(c_lens.numel() + 1, dtype=cu_seq_lens.dtype, device=cu_seq_lens.device)
            cu_seq_lens_out[1:] = torch.cumsum(c_lens, dim=0)
            total_c = int(cu_seq_lens_out[-1].item())
        return cu_seq_lens_out, total_c

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        cu_seq_lens_cpu: torch.Tensor | None = None,
        position_embeddings_compressed: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seq_lens_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a packed varlen sequence sample-by-sample.

        Args:
            hidden_states (torch.Tensor): Packed hidden states shaped
                ``[1, total_tokens, hidden_size]`` (XTuner varlen convention,
                see ``xtuner/v1/module/attention/mla.py`` ``forward_training``).
                A 2D ``[total_tokens, hidden_size]`` tensor is also accepted
                and is promoted to the 3D form before processing.
            cu_seq_lens (torch.Tensor): 1D int32 tensor of length
                ``num_samples + 1`` giving cumulative per-sample token counts.
            cu_seq_lens_cpu (torch.Tensor | None): Optional CPU mirror of
                ``cu_seq_lens`` (see :class:`SequenceContext.cu_seq_lens_q_cpu`).
                When provided, the per-chunk count and ``total_c`` are derived
                from this CPU copy, avoiding the host-blocking ``.item()`` D2H
                that the fallback path triggers.
            position_embeddings_compressed (tuple[torch.Tensor, torch.Tensor] | None):
                Optional ``(cos, sin)`` rope table for the compressed-chunk rope.
            cu_seq_lens_out (torch.Tensor | None): Optional precomputed compressed
                boundaries (see :meth:`build_cu_seq_lens_out`). When the model
                forward already built it for this ``compress_ratio``, pass it here
                so the cumsum + H2D are skipped; ``total_c`` is still derived from
                ``cu_seq_lens_cpu`` (cheap, no D2H). When ``None`` it is built here.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(compressed, cu_seq_lens_out)``
            where ``compressed`` has shape
            ``[1, total_compressed_tokens, head_dim]`` matching the input
            rank, and ``cu_seq_lens_out`` is a 1D int32 tensor of length
            ``num_samples + 1`` carrying the compressed sample boundaries.
        """
        if hidden_states.dim() == 2:
            input_was_2d = True
            packed = hidden_states.unsqueeze(0)
        elif hidden_states.dim() == 3:
            input_was_2d = False
            packed = hidden_states
        else:
            raise ValueError(
                f"hidden_states must be 2D [S, D] or 3D [1, S, D]; got shape {tuple(hidden_states.shape)}"
            )

        if packed.size(0) != 1:
            raise ValueError(f"KVCompressor expects packed varlen input with batch dim 1; got batch={packed.size(0)}")
        if packed.size(-1) != self.hidden_size:
            raise ValueError(f"hidden_states last dim {packed.size(-1)} != hidden_size {self.hidden_size}")
        if cu_seq_lens.dim() != 1 or cu_seq_lens.numel() < 2:
            raise ValueError(f"cu_seq_lens must be 1D with at least 2 entries; got shape {tuple(cu_seq_lens.shape)}")

        ratio = self.compress_ratio
        head_dim = self.head_dim
        coff = self._coff
        device = packed.device
        total_q = packed.size(1)

        # 1. Per-sample compressed-chunk boundaries (``cu_seq_lens_out``) + ``total_c``.
        #
        # Preferred path: the model forward already built ``cu_seq_lens_out`` once for this
        # ``compress_ratio`` (same value for every layer of that ratio) and passed it in, so we
        # skip the per-call cumsum + H2D and only derive ``total_c`` from the CPU mirror (cheap,
        # no D2H — ``total_c`` cannot ride along on ``cu_seq_lens_out`` because it must stay a
        # Python int and would force a recompile if threaded through the compiled attn graph).
        #
        # Fallback path: build both here via :meth:`build_cu_seq_lens_out` (CPU cumsum + H2D when
        # ``cu_seq_lens_cpu`` is available, else a GPU cumsum with a host-blocking ``.item()``).
        # The compressor stays eager either way, so any sync only graph-breaks compile *across*
        # this call, not inside it.
        if cu_seq_lens_out is not None and cu_seq_lens_cpu is not None:
            c_lens_cpu = (cu_seq_lens_cpu[1:] - cu_seq_lens_cpu[:-1] + ratio - 1) // ratio
            total_c = int(c_lens_cpu.sum())
        else:
            cu_seq_lens_out, total_c = self.build_cu_seq_lens_out(cu_seq_lens, cu_seq_lens_cpu, ratio)

        # 2. Project every token in two GEMMs over the full pack — no per-sample padding.
        # Each input token at sample-local position ``s`` lands in chunk ``s // ratio``
        # at slot ``s % ratio``; tokens that don't fill the tail of a chunk leave
        # that slot at the buffer's init value (kv=0 / score=-inf), reproducing the
        # original code's "zero-pad the input + softmax masks it out" behaviour
        # without ever materialising the padding.
        kv_proj = self.wkv(packed).view(total_q, coff * head_dim)
        score_proj = self.wgate(packed).view(total_q, coff * head_dim)

        # 3. Per-token chunk assignment.
        pos = torch.arange(total_q, device=device, dtype=torch.long)
        sample_id = torch.searchsorted(cu_seq_lens, pos, right=True) - 1
        in_sample_pos = pos - cu_seq_lens[sample_id]
        chunk_in_sample = in_sample_pos // ratio
        pos_in_chunk = in_sample_pos % ratio
        global_chunk_id = cu_seq_lens_out[sample_id] + chunk_in_sample
        flat_idx = global_chunk_id * ratio + pos_in_chunk

        # 4. Scatter into chunk layout. Use index_put on a fresh buffer so this
        # stays a pure functional op (autograd-friendly, no in-place aliasing).
        chunk_dim = coff * head_dim
        kv_chunks_flat = kv_proj.new_zeros(total_c * ratio, chunk_dim)
        score_chunks_flat = score_proj.new_full((total_c * ratio, chunk_dim), float("-inf"))
        kv_chunks_flat = kv_chunks_flat.index_put((flat_idx,), kv_proj)
        score_chunks_flat = score_chunks_flat.index_put((flat_idx,), score_proj)
        kv_chunks = kv_chunks_flat.view(1, total_c, ratio, chunk_dim)
        score_chunks = score_chunks_flat.view(1, total_c, ratio, chunk_dim)

        # 5. APE + DTensor unwrap (same EP rationale as the original).
        ape = self.ape.to_local() if isinstance(self.ape, _DTensor) else self.ape
        score_chunks = score_chunks + ape

        # 6. Overlap (compress_ratio == 4 path). The "previous chunk" link is
        # per-sample: chunk-0 of each sample has no predecessor inside that
        # sample, so its first-half slot is filled with the masking value
        # (0 for kv, -inf for score) rather than the previous sample's last chunk.
        if self.overlap:
            kv_chunks = self._overlap_transform_varlen(kv_chunks, cu_seq_lens_out, fill_value=0.0)
            score_chunks = self._overlap_transform_varlen(score_chunks, cu_seq_lens_out, fill_value=float("-inf"))

        # 7. Softmax + weighted sum + norm.
        weights = score_chunks.softmax(dim=2)
        compressed = (kv_chunks * weights).sum(dim=2)
        compressed = self.norm(compressed)  # [1, total_c, head_dim]

        # 8. RoPE on the rope-tail of each compressed chunk, mirroring HF
        # ``DeepseekV4{CSA,HCA}Compressor.forward`` (modeling_deepseek_v4.py
        # L424 / L668). The HF reference computes
        # ``positions = arange(n_windows) * compress_rate + first_window_position``
        # (window starting token, ``first_window_position == 0`` in the prefill
        # path xtuner uses) and rotates ``compressed[..., -rope_dim:]`` with the
        # ``compress``-base cos/sin at those positions.
        #
        # We accept the caller's full per-query ``(cos, sin)`` table (xtuner's
        # ``_build_compressed_position_embeddings`` already materialises it per
        # forward) and gather the entries at the first-token global index of
        # each compressed chunk: ``first_token_per_chunk[c] = cu_seq_lens[s] +
        # (c - cu_seq_lens_out[s]) * compress_ratio`` where ``s`` is the sample
        # owning chunk ``c``. The gather is a single ``index_select`` per cos /
        # sin so the autograd path stays clean.
        if position_embeddings_compressed is not None and self.qk_rope_head_dim is not None:
            cos_q, sin_q = position_embeddings_compressed
            if cos_q.dim() != 3 or cos_q.size(0) != 1:
                raise ValueError(
                    f"position_embeddings_compressed cos must be [1, total_q, rope_head_dim]; got {tuple(cos_q.shape)}"
                )
            if cos_q.size(-1) != self.qk_rope_head_dim:
                raise ValueError(
                    f"position_embeddings_compressed last dim {cos_q.size(-1)} != qk_rope_head_dim {self.qk_rope_head_dim}"
                )
            chunk_idx = torch.arange(total_c, device=device, dtype=cu_seq_lens.dtype)
            # Map each chunk to its owning sample with the same idiom as step 3's
            # per-token assignment (``searchsorted(cu, ., right=True) - 1``): the
            # owner of chunk ``c`` is the sample ``s`` with ``cu_seq_lens_out[s] <=
            # c < cu_seq_lens_out[s+1]``. ``right=True`` is load-bearing — a chunk
            # sitting exactly on a boundary is the *first* chunk of the next sample,
            # so it must map to that sample, not the previous one. Mapping it to the
            # previous sample makes ``chunk_in_sample_local`` overshoot, pushing
            # ``first_token_per_chunk`` past the sample's last token and indexing
            # ``cos_q`` out of bounds.
            sample_of_chunk = torch.searchsorted(cu_seq_lens_out, chunk_idx, right=True) - 1
            chunk_in_sample_local = chunk_idx - cu_seq_lens_out[sample_of_chunk]
            first_token_per_chunk = cu_seq_lens[sample_of_chunk] + chunk_in_sample_local * ratio
            # ``cos_q[0]`` is [total_q, rope_head_dim]; gather along the token axis.
            cos_kv = cos_q[0].index_select(0, first_token_per_chunk.long()).unsqueeze(0)
            sin_kv = sin_q[0].index_select(0, first_token_per_chunk.long()).unsqueeze(0)
            compressed = _apply_rope_split(compressed, cos_kv, sin_kv, self.qk_rope_head_dim)

        if input_was_2d:
            compressed = compressed.squeeze(0)
        return compressed, cu_seq_lens_out

    def _overlap_transform_varlen(
        self,
        tensor: torch.Tensor,
        cu_chunks: torch.Tensor,
        fill_value: float,
    ) -> torch.Tensor:
        """Varlen replacement for :meth:`_overlap_transform`.

        Same math as the original (chunk's own second-half stays; previous
        chunk's first-half is prepended), but the "previous chunk" link is
        sample-aware: each sample's first chunk gets ``fill_value`` for its
        first-half slot instead of leaking from the previous sample's last
        chunk.

        Args:
            tensor (torch.Tensor): ``[1, total_c, ratio, 2*head_dim]`` chunk-laid input.
            cu_chunks (torch.Tensor): ``[B+1]`` cumulative compressed-chunk counts;
                ``cu_chunks[i]`` is the global chunk index where sample ``i`` begins.
            fill_value (float): What to write into the first-half slots of every
                sample-first chunk (``0.0`` for kv, ``-inf`` for score).

        Returns:
            torch.Tensor: ``[1, total_c, 2*ratio, head_dim]``.
        """
        bsz, total_c, ratio, two_d = tensor.shape
        head_dim = self.head_dim
        assert two_d == 2 * head_dim, f"overlap_transform expects last dim {2 * head_dim}, got {two_d}"
        device = tensor.device

        # Identify the sample-first chunk: ``chunk_id == cu_chunks[sample_of_chunk]``.
        chunk_pos = torch.arange(total_c, device=device, dtype=torch.long)
        chunk_sample_id = torch.searchsorted(cu_chunks, chunk_pos, right=True) - 1
        is_first_in_sample = chunk_pos == cu_chunks[chunk_sample_id]

        # Gather the previous chunk's first-half slice. ``prev_idx`` is clamped
        # so chunk 0 doesn't index out-of-range; the value at index 0 is
        # overwritten by ``fill_value`` via the mask below anyway.
        prev_idx = (chunk_pos - 1).clamp(min=0)
        prev_half = tensor[0].index_select(0, prev_idx)[:, :, :head_dim]  # [total_c, ratio, head_dim]
        prev_half_masked = torch.where(
            is_first_in_sample.view(-1, 1, 1),
            torch.full_like(prev_half, fill_value),
            prev_half,
        )

        # Current chunk's own second-half slice — straight slice, no gather.
        cur_half = tensor[0, :, :, head_dim:]  # [total_c, ratio, head_dim]

        return torch.cat([prev_half_masked, cur_half], dim=1).unsqueeze(0)

    def init_weights(self) -> None:
        nn.init.zeros_(self.ape)
        nn.init.xavier_uniform_(self.wkv.weight)
        nn.init.xavier_uniform_(self.wgate.weight)
        self.norm.init_weights()
