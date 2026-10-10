# Copyright (c) OpenMMLab. All rights reserved.
"""Fixed-layout gate prefix sums for KDA's 64-token chunks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_cumsum_kernel(
    g: tl.tensor,
    output: tl.tensor,
    cu_seqlens: tl.tensor,
    chunk_indices: tl.tensor,
    T: int,
    H: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    VARLEN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    feature_block, chunk, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head = batch_head % H
    if VARLEN:
        document = tl.load(chunk_indices + 2 * chunk).to(tl.int32)
        chunk = tl.load(chunk_indices + 2 * chunk + 1).to(tl.int32)
        start = tl.load(cu_seqlens + document).to(tl.int32)
        stop = tl.load(cu_seqlens + document + 1).to(tl.int32)
    else:
        start = batch_head // H * T
        stop = start + T
    token = start + chunk * BT + tl.arange(0, BT)
    feature = feature_block * BK + tl.arange(0, BK)
    offsets = (token[:, None] * H + head) * K + feature[None, :]
    valid = (token[:, None] < stop) & (feature[None, :] < K)
    values = tl.load(g + offsets, mask=valid, other=0).to(tl.float32)
    # Scan before scaling, as in FLA. Changing this order changes rounding.
    prefix = tl.cumsum(values, axis=0) * SCALE
    tl.store(output + offsets, prefix, mask=valid)


def chunk_cumsum(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Compute FP32 prefixes without selecting a new scan layout for each head count.

    Args:
        g (torch.Tensor): Contiguous log gates of shape ``[B, T, H, K]``.
        cu_seqlens (torch.Tensor | None): Packed document boundaries, with batch size one.
        chunk_indices (torch.Tensor | None): Document/chunk pairs prepared with chunk size 64.
        scale (float): Multiplier applied after the prefix sum.

    Returns:
        torch.Tensor: FP32 prefixes with the same shape as ``g``.
    """
    batch, length, heads, features = g.shape
    if (cu_seqlens is None) != (chunk_indices is None):
        raise ValueError("Packed KDA prefixes require both cu_seqlens and chunk_indices")
    if cu_seqlens is not None and batch != 1:
        raise ValueError("Packed KDA prefixes require batch size one")
    g = g.contiguous()
    chunks = triton.cdiv(length, 64) if chunk_indices is None else chunk_indices.shape[0]
    output = torch.empty_like(g, dtype=torch.float32)
    # FLA autotunes the scan using H in its key. Ulysses changes H, which can select
    # a different FP32 addition order and perturb all subsequent recurrent states.
    _chunk_cumsum_kernel[(triton.cdiv(features, 32), chunks, batch * heads)](
        g,
        output,
        cu_seqlens,
        chunk_indices,
        length,
        heads,
        features,
        scale,
        VARLEN=cu_seqlens is not None,
        BT=64,
        BK=32,
        num_warps=2,
    )
    return output
