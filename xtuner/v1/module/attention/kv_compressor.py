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

from ..rms_norm import RMSNorm


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

        coff = 1 + int(overlap)
        self._coff = coff

        self.wkv = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(hidden_size, coff * head_dim, bias=False)
        self.norm = RMSNorm(head_dim, eps=rms_norm_eps)
        # APE is added to the gate logits before softmax so the network can
        # learn position-within-window biases independent of input content.
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coff * head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seq_lens: torch.Tensor,
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

        boundaries = cu_seq_lens.detach().cpu().tolist()
        num_samples = len(boundaries) - 1

        compressed_chunks: list[torch.Tensor] = []
        compressed_lengths: list[int] = [0]
        running_total = 0
        for i in range(num_samples):
            start, end = boundaries[i], boundaries[i + 1]
            sample = packed[:, start:end, :]
            compressed_sample = self._compress_sample(sample)
            compressed_chunks.append(compressed_sample)
            running_total += compressed_sample.size(1)
            compressed_lengths.append(running_total)

        compressed = torch.cat(compressed_chunks, dim=1)
        cu_seq_lens_out = torch.tensor(
            compressed_lengths,
            dtype=cu_seq_lens.dtype,
            device=cu_seq_lens.device,
        )

        if input_was_2d:
            compressed = compressed.squeeze(0)
        return compressed, cu_seq_lens_out

    def _compress_sample(self, sample: torch.Tensor) -> torch.Tensor:
        # sample: [1, S_i, hidden_size]; S_i can be < ratio (single short
        # sample still produces one compressed token after padding).
        ratio = self.compress_ratio
        overlap = self.overlap
        head_dim = self.head_dim
        seq_len = sample.size(1)

        # Pad each sample to a multiple of ratio. We deviate from the upstream
        # prefill (which stashes the remainder in kv_state for the next
        # forward call): in training there is no cross-call state, so we
        # emit a partial-but-zero-padded final group instead.
        remainder = seq_len % ratio
        if remainder != 0:
            pad_len = ratio - remainder
            sample = torch.nn.functional.pad(sample, (0, 0, 0, pad_len))
            seq_len = sample.size(1)

        num_chunks = seq_len // ratio
        kv = self.wkv(sample)
        score = self.wgate(sample)

        # [1, num_chunks, ratio, coff * head_dim]
        kv = kv.unflatten(1, (num_chunks, ratio))
        score = score.unflatten(1, (num_chunks, ratio)) + self.ape

        if overlap:
            kv = self._overlap_transform(kv, fill_value=0.0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        weights = score.softmax(dim=2)
        compressed = (kv * weights).sum(dim=2)
        compressed = self.norm(compressed)
        # Output rank matches the input: caller passed a 3D packed tensor
        # so we keep batch dim here; the public forward strips it again
        # only if the original input was 2D.
        return compressed.view(1, num_chunks, head_dim)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        # Mirrors `Compressor.overlap_transform` from the V4 reference:
        # doubles the per-chunk window so each compressed token sees its own
        # group (placed in the second half of the new ratio axis) plus the
        # previous group's tokens (first half). The chunk with no predecessor
        # is filled with `fill_value` to be a no-op under softmax (-inf) or
        # weighted sum (0.0).
        bsz, num_chunks, ratio, two_d = tensor.shape
        head_dim = self.head_dim
        assert two_d == 2 * head_dim, f"overlap_transform expects last dim {2 * head_dim}, got {two_d}"
        new_tensor = tensor.new_full((bsz, num_chunks, 2 * ratio, head_dim), fill_value)
        # Second half of the new ratio axis: current chunk's "own" half-dim slice.
        new_tensor[:, :, ratio:, :] = tensor[:, :, :, head_dim:]
        # First half of the new ratio axis: previous chunk's "shared" half-dim slice.
        if num_chunks > 1:
            new_tensor[:, 1:, :ratio, :] = tensor[:, :-1, :, :head_dim]
        return new_tensor

    def init_weights(self) -> None:
        nn.init.zeros_(self.ape)
        nn.init.xavier_uniform_(self.wkv.weight)
        nn.init.xavier_uniform_(self.wgate.weight)
        self.norm.init_weights()
