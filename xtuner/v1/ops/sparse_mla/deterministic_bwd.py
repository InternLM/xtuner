# Copyright (c) OpenMMLab. All rights reserved.
"""Ordered segmented reduction for the deterministic sparse-MLA backward.

cuDNN's sparse-attention backward accumulates dKV with unordered atomics, and PyTorch's
deterministic ``index_add_`` pays for that with an internal sort + serial reduction that
costs ~1.5s and a ~64GB workspace at GLM-5.3-Flash's DSA shape. This module replaces the
accumulation with an explicit sort + a Triton kernel that reduces each segment serially
in sorted order, so the accumulation order is fully fixed (bitwise repeatable) while the
kernel stays memory-bandwidth bound (~20ms per layer instead of ~1.5s).

Determinism argument: within one call, segment ``s`` owns the sorted row range
``[offsets[s], offsets[s+1])`` and a single program accumulates those rows in ascending
position order (row tiles of ``BLOCK_M``, each tile tree-reduced with a fixed shape). The
pre-existing accumulator value is loaded as the init of that running sum, so across calls
(e.g. once per query chunk) chunk partials are accumulated in call order. Every step's
order is fixed by the launch configuration, so repeated runs are bitwise identical.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _ordered_segment_reduce_kernel(
    contrib_ptr,  # [n_rows, DIM] contribution rows (bf16/fp32), in original (unsorted) order
    perm_ptr,  # [n_rows] int64: sorted position -> source row in contrib
    offsets_ptr,  # [num_segments + 1] int64: segment boundaries in the sorted order
    accum_ptr,  # [num_segments, DIM] fp32 accumulator, read-modify-write in place
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
):
    seg = tl.program_id(0)
    start = tl.load(offsets_ptr + seg)
    end = tl.load(offsets_ptr + seg + 1)
    cols = tl.arange(0, DIM)
    acc = tl.load(accum_ptr + seg * DIM + cols)
    for m in range(start, end, BLOCK_M):
        pos = m + tl.arange(0, BLOCK_M)
        in_seg = pos < end
        rows = tl.load(perm_ptr + pos, mask=in_seg, other=0)
        tile = tl.load(contrib_ptr + rows[:, None] * DIM + cols[None, :], mask=in_seg[:, None], other=0.0)
        acc += tl.sum(tile.to(tl.float32), axis=0)
    tl.store(accum_ptr + seg * DIM + cols, acc)


def ordered_segment_reduce_(
    accum: Tensor,
    contrib: Tensor,
    seg_ids: Tensor,
    block_m: int = 32,
    num_warps: int = 4,
) -> None:
    """Accumulate contribution rows into ``accum`` segment by segment, in a fixed order.

    ``accum[seg] += sum of contrib rows r with seg_ids[r] == seg``. Rows within one
    segment are summed in ascending original-row order (stable sort), one program per
    segment, serial row tiles -- so the result is bitwise repeatable across runs and
    across processes on the same hardware. ``accum`` is updated in place; calling this
    repeatedly (e.g. once per query chunk) accumulates chunk partials in call order.

    Rows with a negative segment id (sparse-attention's -1 padding) are dropped before
    the reduction; callers must ensure those rows carry an exact-zero contribution.
    Dropping them is not just a performance fix: padding rows would otherwise share one
    clamped segment and form a single serial monster segment (~half of all rows for a
    causal topk window), while a true zero never changes the FP32 sum.

    Args:
        accum (Tensor): ``[num_segments, dim]`` fp32 accumulator, updated in place.
        contrib (Tensor): ``[n_rows, dim]`` contribution rows, bf16 or fp32; bf16 rows
            are upcast to fp32 before accumulation.
        seg_ids (Tensor): ``[n_rows]`` int64 segment index per row; values in
            ``[0, num_segments)`` select a segment, negative values skip the row.
        block_m (int): Rows per inner tile of the serial segment loop.
        num_warps (int): Triton launch warps per program.
    """
    n_rows, dim = contrib.shape
    num_segments = accum.shape[0]
    assert dim & (dim - 1) == 0, f"dim must be a power of two, got {dim}"
    assert contrib.is_cuda and accum.is_cuda and seg_ids.is_cuda
    sorted_ids, perm = torch.sort(seg_ids, stable=True)
    # Drop negative (padding) ids: after the sort they form a prefix of the sorted order.
    first_valid = torch.searchsorted(sorted_ids, 0)
    sorted_ids = sorted_ids[first_valid:]
    perm = perm[first_valid:]
    # Integer bincount is exact under any atomic order, so this stays deterministic.
    counts = torch.bincount(sorted_ids, minlength=num_segments)
    offsets = counts.new_zeros(num_segments + 1)
    torch.cumsum(counts, 0, out=offsets[1:])
    _ordered_segment_reduce_kernel[(num_segments,)](
        contrib,
        perm,
        offsets,
        accum,
        BLOCK_M=block_m,
        DIM=dim,
        num_warps=num_warps,
    )
