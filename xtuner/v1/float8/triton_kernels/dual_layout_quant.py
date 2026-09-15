# Copyright (c) OpenMMLab. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _group_metadata_kernel(
    size_per_group_ptr,
    size_per_group_expand_ptr,
    group_pad_offsets_ptr,
    token_cumdiffs_ptr,
    group_token_starts_ptr,
    token_ends_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_GROUPS
    sizes = tl.load(size_per_group_ptr + offsets, mask=mask, other=0)
    sizes_expand = (sizes + GROUP_SIZE - 1) // GROUP_SIZE * GROUP_SIZE
    group_ends = tl.cumsum(sizes, axis=0)
    group_ends_expand = tl.cumsum(sizes_expand, axis=0)
    group_starts = group_ends - sizes
    group_starts_expand = group_ends_expand - sizes_expand

    tl.store(size_per_group_expand_ptr + offsets, sizes_expand, mask=mask)
    tl.store(group_pad_offsets_ptr, 0)
    tl.store(group_pad_offsets_ptr + offsets + 1, group_ends_expand, mask=mask)
    tl.store(token_cumdiffs_ptr + offsets, group_starts_expand - group_starts, mask=mask)
    tl.store(group_token_starts_ptr + offsets, group_starts, mask=mask)
    tl.store(token_ends_ptr + offsets, group_ends, mask=mask)


@triton.jit
def _group_ids_per_tile_kernel(
    group_pad_offsets_ptr,
    group_ids_ptr,
    NUM_TILES,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Map each padded token tile to its expert without an O(experts) scan.

    ``group_pad_offsets`` is monotonically increasing.  The previous
    implementation compared every tile against every expert boundary, which
    made this metadata launch scale linearly with the expert count.  The
    binary search keeps the result identical while reducing the comparisons
    to ``ceil(log2(NUM_GROUPS + 1))`` per tile.
    """
    tile_ids = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tile_starts = tile_ids * GROUP_SIZE
    low = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    high = tl.full((BLOCK_SIZE,), NUM_GROUPS, dtype=tl.int32)
    for _ in tl.static_range(0, (NUM_GROUPS + 1).bit_length()):
        # Clamp the probe so the fixed-iteration search never reads the
        # sentinel one-past-the-end entry when low == high == NUM_GROUPS.
        probe = tl.minimum((low + high) // 2, NUM_GROUPS - 1)
        group_end = tl.load(group_pad_offsets_ptr + probe + 1)
        take_upper = tile_starts >= group_end
        low = tl.where(take_upper, probe + 1, low)
        high = tl.where(take_upper, high, probe)
    group_ids = low
    mask = (tile_ids < NUM_TILES) & (tile_starts < tl.load(group_pad_offsets_ptr + NUM_GROUPS))
    tl.store(group_ids_ptr + tile_ids, group_ids, mask=mask)


@triton.jit
def _per_tile_quant_with_trans_per_block_kernel(
    input_ptr,
    row_output_ptr,
    row_scale_ptr,
    trans_output_ptr,
    trans_scale_ptr,
    group_pad_offsets_ptr,
    group_ids_per_tile_ptr,
    token_cumdiffs_ptr,
    token_ends_ptr,
    stride_input_m: tl.constexpr,
    stride_input_n: tl.constexpr,
    stride_row_output_m: tl.constexpr,
    stride_row_output_n: tl.constexpr,
    stride_row_scale_m: tl.constexpr,
    stride_row_scale_n: tl.constexpr,
    N: tl.constexpr,
    M_EXPAND,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    m_pad = tl.load(group_pad_offsets_ptr + NUM_GROUPS)
    if pid_m * GROUP_SIZE >= m_pad:
        hidden_offsets = pid_n * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        token_offsets_expand = pid_m * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        trans_output_offsets = (
            trans_output_ptr
            + hidden_offsets[:, None].to(tl.int64) * M_EXPAND
            + token_offsets_expand[None, :].to(tl.int64)
        )
        trans_scale_offset = trans_scale_ptr + pid_n * (M_EXPAND // GROUP_SIZE) + pid_m

        zeros = tl.zeros((GROUP_SIZE, GROUP_SIZE), dtype=trans_output_ptr.dtype.element_ty)
        tl.store(trans_output_offsets, zeros)
        tl.store(trans_scale_offset, 0.0)
        return

    group_id = tl.load(group_ids_per_tile_ptr + pid_m)
    token_cumdiff = tl.load(token_cumdiffs_ptr + group_id)
    token_end = tl.load(token_ends_ptr + group_id)

    token_offsets = pid_m * GROUP_SIZE - token_cumdiff + tl.arange(0, GROUP_SIZE)
    hidden_offsets = pid_n * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_offsets_expand = pid_m * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_mask = token_offsets < token_end
    hidden_mask = hidden_offsets < N
    input_mask = token_mask[:, None] & hidden_mask[None, :]

    input_offsets = (
        input_ptr
        + token_offsets[:, None].to(tl.int64) * stride_input_m
        + hidden_offsets[None, :].to(tl.int64) * stride_input_n
    )
    input_block = tl.load(input_offsets, mask=input_mask, other=0.0).to(tl.float32)

    reciprocal_fp8_max = 1.0 / FP8_MAX
    row_scale = tl.max(tl.abs(input_block), axis=1) * reciprocal_fp8_max
    row_scale = tl.clamp(row_scale, 1e-12, 3e38)
    row_output = input_block / row_scale[:, None]
    row_output = tl.clamp(row_output, FP8_MIN, FP8_MAX).to(row_output_ptr.dtype.element_ty)

    row_output_offsets = (
        row_output_ptr
        + token_offsets[:, None].to(tl.int64) * stride_row_output_m
        + hidden_offsets[None, :].to(tl.int64) * stride_row_output_n
    )
    row_scale_offsets = (
        row_scale_ptr
        + token_offsets.to(tl.int64) * stride_row_scale_m
        + pid_n * stride_row_scale_n
    )
    tl.store(row_output_offsets, row_output, mask=input_mask)
    tl.store(row_scale_offsets, row_scale, mask=token_mask)

    trans_scale = tl.max(tl.max(tl.abs(input_block), axis=0), axis=0) / FP8_MAX
    trans_scale = tl.clamp(trans_scale, 1e-12, 3e38)
    trans_output = input_block / trans_scale
    trans_output = tl.clamp(trans_output, FP8_MIN, FP8_MAX).trans(1, 0).to(trans_output_ptr.dtype.element_ty)

    trans_output_offsets = (
        trans_output_ptr
        + hidden_offsets[:, None].to(tl.int64) * M_EXPAND
        + token_offsets_expand[None, :].to(tl.int64)
    )
    trans_scale_offset = trans_scale_ptr + pid_n * (M_EXPAND // GROUP_SIZE) + pid_m
    tl.store(trans_output_offsets, trans_output, mask=hidden_mask[:, None])
    tl.store(trans_scale_offset, trans_scale)


@triton.jit
def _per_tile_quant_with_trans_per_tile_kernel(
    input_ptr,
    row_output_ptr,
    row_scale_ptr,
    trans_output_ptr,
    trans_scale_ptr,
    group_pad_offsets_ptr,
    group_ids_per_tile_ptr,
    token_cumdiffs_ptr,
    token_ends_ptr,
    stride_input_m: tl.constexpr,
    stride_input_n: tl.constexpr,
    stride_row_output_m: tl.constexpr,
    stride_row_output_n: tl.constexpr,
    stride_row_scale_m: tl.constexpr,
    stride_row_scale_n: tl.constexpr,
    M_EXPAND,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    m_pad = tl.load(group_pad_offsets_ptr + NUM_GROUPS)
    if pid_m * GROUP_SIZE >= m_pad:
        return

    group_id = tl.load(group_ids_per_tile_ptr + pid_m)
    token_cumdiff = tl.load(token_cumdiffs_ptr + group_id)
    token_end = tl.load(token_ends_ptr + group_id)
    token_offsets = pid_m * GROUP_SIZE - token_cumdiff + tl.arange(0, GROUP_SIZE)
    hidden_offsets = pid_n * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_offsets_expand = pid_m * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_mask = token_offsets < token_end
    hidden_mask = hidden_offsets < N
    input_offsets = (
        input_ptr
        + token_offsets[:, None].to(tl.int64) * stride_input_m
        + hidden_offsets[None, :].to(tl.int64) * stride_input_n
    )
    input_block = tl.load(input_offsets, mask=token_mask[:, None] & hidden_mask[None, :], other=0.0).to(tl.float32)

    reciprocal_fp8_max = 1.0 / FP8_MAX
    row_scale = tl.max(tl.abs(input_block), axis=1) * reciprocal_fp8_max
    row_scale = tl.clamp(row_scale, 1e-12, 3e38)
    row_output = input_block / row_scale[:, None]
    row_output = tl.clamp(row_output, FP8_MIN, FP8_MAX).to(row_output_ptr.dtype.element_ty)
    row_output_offsets = (
        row_output_ptr
        + token_offsets[:, None].to(tl.int64) * stride_row_output_m
        + hidden_offsets[None, :].to(tl.int64) * stride_row_output_n
    )
    row_scale_offsets = (
        row_scale_ptr
        + token_offsets.to(tl.int64) * stride_row_scale_m
        + pid_n * stride_row_scale_n
    )
    tl.store(row_output_offsets, row_output, mask=token_mask[:, None] & hidden_mask[None, :])
    tl.store(row_scale_offsets, row_scale, mask=token_mask)

    trans_scale = tl.max(tl.abs(input_block), axis=0) / FP8_MAX
    trans_scale = tl.clamp(trans_scale, 1e-12, 3e38)
    trans_output = input_block / trans_scale[None, :]
    trans_output = tl.clamp(trans_output, FP8_MIN, FP8_MAX).to(trans_output_ptr.dtype.element_ty)
    trans_output_offsets = (
        trans_output_ptr
        + hidden_offsets[None, :].to(tl.int64) * M_EXPAND
        + token_offsets_expand[:, None].to(tl.int64)
    )
    trans_scale_offsets = trans_scale_ptr + hidden_offsets * (M_EXPAND // GROUP_SIZE) + pid_m
    tl.store(trans_output_offsets, trans_output, mask=hidden_mask[None, :])
    tl.store(trans_scale_offsets, trans_scale, mask=hidden_mask)


def _group_metadata(
    size_per_group: torch.Tensor, group_size: int, num_tiles: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_groups = size_per_group.shape[0]
    size_per_group_expand = torch.empty_like(size_per_group)
    group_pad_offsets = size_per_group.new_empty((num_groups + 1,), dtype=torch.int32)
    token_cumdiffs = torch.empty_like(size_per_group)
    group_token_starts = torch.empty_like(size_per_group)
    token_ends = torch.empty_like(size_per_group)
    _group_metadata_kernel[(1,)](
        size_per_group,
        size_per_group_expand,
        group_pad_offsets,
        token_cumdiffs,
        group_token_starts,
        token_ends,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=triton.next_power_of_2(num_groups),
        num_warps=1,
    )
    group_ids_per_tile = size_per_group.new_empty((num_tiles,), dtype=torch.int32)
    block_size = 64
    _group_ids_per_tile_kernel[(triton.cdiv(num_tiles, block_size),)](
        group_pad_offsets,
        group_ids_per_tile,
        NUM_TILES=num_tiles,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=block_size,
        num_warps=2,
    )
    return size_per_group_expand, group_pad_offsets, group_ids_per_tile, token_cumdiffs, group_token_starts, token_ends


@torch.library.custom_op("float8::per_tile_quant_with_trans_per_block", mutates_args=())
def per_tile_quant_with_trans_per_block(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor into row-wise and transposed block-wise FP8 layouts.

    Args:
        input_tensor (torch.Tensor): Two-dimensional BF16 activation tensor.
        size_per_group (torch.Tensor): Number of rows belonging to each expert.
        group_size (int): Quantization group size. Only 128 is supported.
        dtype (torch.dtype): FP8 output dtype. Only E4M3FN is supported.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Row-major FP8 data and
        scales, transposed block-wise FP8 data and scales, and padded expert sizes.
    """
    m, n = input_tensor.shape
    assert n % group_size == 0

    num_groups = size_per_group.shape[0]
    m_expand = m + group_size * num_groups - m % group_size
    num_tiles = m_expand // group_size
    (
        size_per_group_expand,
        group_pad_offsets,
        group_ids_per_tile,
        token_cumdiffs,
        _,
        token_ends,
    ) = _group_metadata(size_per_group, group_size, num_tiles)
    row_output = torch.empty_like(input_tensor, dtype=dtype)
    row_scales = input_tensor.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = input_tensor.new_empty((n, m_expand), dtype=dtype)
    trans_scales = input_tensor.new_empty((n // group_size, m_expand // group_size), dtype=torch.float32)

    grid = (triton.cdiv(n, group_size), triton.cdiv(m_expand, group_size))
    _per_tile_quant_with_trans_per_block_kernel[grid](
        input_tensor,
        row_output,
        row_scales,
        trans_output,
        trans_scales,
        group_pad_offsets,
        group_ids_per_tile,
        token_cumdiffs,
        token_ends,
        input_tensor.stride(0),
        input_tensor.stride(1),
        row_output.stride(0),
        row_output.stride(1),
        row_scales.stride(0),
        row_scales.stride(1),
        N=n,
        M_EXPAND=m_expand,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        FP8_MIN=torch.finfo(dtype).min,
        FP8_MAX=torch.finfo(dtype).max,
        num_warps=4,
        num_stages=3,
    )
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand


@per_tile_quant_with_trans_per_block.register_fake
def _(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = input_tensor.shape
    num_groups = size_per_group.shape[0]
    m_expand = m + group_size * num_groups - m % group_size
    row_output = torch.empty_like(input_tensor, dtype=dtype)
    row_scales = input_tensor.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = input_tensor.new_empty((n, m_expand), dtype=dtype)
    trans_scales = input_tensor.new_empty((n // group_size, m_expand // group_size), dtype=torch.float32)
    size_per_group_expand = torch.empty_like(size_per_group)
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand


@torch.library.custom_op("float8::per_tile_quant_with_trans_per_tile", mutates_args=())
def per_tile_quant_with_trans_per_tile(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a tensor into row-wise and transposed row-wise FP8 layouts.

    Args:
        input_tensor (torch.Tensor): Two-dimensional BF16 gradient tensor.
        size_per_group (torch.Tensor): Number of rows belonging to each expert.
        group_size (int): Quantization group size. Only 128 is supported.
        dtype (torch.dtype): FP8 output dtype. Only E4M3FN is supported.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Row-major FP8 data and
        scales, transposed row-wise FP8 data and scales, and padded expert sizes.
    """
    m, n = input_tensor.shape
    assert n % group_size == 0

    num_groups = size_per_group.shape[0]
    m_expand = m + group_size * num_groups - m % group_size
    num_tiles = m_expand // group_size
    (
        size_per_group_expand,
        group_pad_offsets,
        group_ids_per_tile,
        token_cumdiffs,
        _,
        group_ends,
    ) = _group_metadata(size_per_group, group_size, num_tiles)
    row_output = torch.empty_like(input_tensor, dtype=dtype)
    row_scales = input_tensor.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = input_tensor.new_empty((n, m_expand), dtype=dtype)
    trans_scales = input_tensor.new_empty((n, m_expand // group_size), dtype=torch.float32)

    grid = (triton.cdiv(n, group_size), triton.cdiv(m_expand, group_size))
    _per_tile_quant_with_trans_per_tile_kernel[grid](
        input_tensor,
        row_output,
        row_scales,
        trans_output,
        trans_scales,
        group_pad_offsets,
        group_ids_per_tile,
        token_cumdiffs,
        group_ends,
        input_tensor.stride(0),
        input_tensor.stride(1),
        row_output.stride(0),
        row_output.stride(1),
        row_scales.stride(0),
        row_scales.stride(1),
        M_EXPAND=m_expand,
        N=n,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        FP8_MIN=torch.finfo(dtype).min,
        FP8_MAX=torch.finfo(dtype).max,
        num_warps=4,
        num_stages=3,
    )
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand


@per_tile_quant_with_trans_per_tile.register_fake
def _(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = input_tensor.shape
    num_groups = size_per_group.shape[0]
    m_expand = m + group_size * num_groups - m % group_size
    row_output = torch.empty_like(input_tensor, dtype=dtype)
    row_scales = input_tensor.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = input_tensor.new_empty((n, m_expand), dtype=dtype)
    trans_scales = input_tensor.new_empty((n, m_expand // group_size), dtype=torch.float32)
    size_per_group_expand = torch.empty_like(size_per_group)
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand
