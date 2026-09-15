# Copyright (c) OpenMMLab. All rights reserved.

import torch
import triton
import triton.language as tl

from .dual_layout_quant import _group_metadata


@triton.jit
def _swiglu_per_tile_quant_with_trans_per_block_kernel(
    gate_up_ptr,
    row_output_ptr,
    row_scale_ptr,
    trans_output_ptr,
    trans_scale_ptr,
    group_pad_offsets_ptr,
    group_ids_per_tile_ptr,
    token_cumdiffs_ptr,
    token_ends_ptr,
    stride_gate_up_m: tl.constexpr,
    stride_gate_up_n: tl.constexpr,
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

    gate_offsets = (
        gate_up_ptr
        + token_offsets[:, None].to(tl.int64) * stride_gate_up_m
        + hidden_offsets[None, :].to(tl.int64) * stride_gate_up_n
    )
    up_offsets = gate_offsets + N * stride_gate_up_n
    gate = tl.load(gate_offsets, mask=input_mask, other=0.0).to(tl.float32)
    up = tl.load(up_offsets, mask=input_mask, other=0.0).to(tl.float32)

    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    act = (silu.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)

    reciprocal_fp8_max = 1.0 / FP8_MAX
    row_scale = tl.max(tl.abs(act), axis=1) * reciprocal_fp8_max
    row_scale = tl.clamp(row_scale, 1e-12, 3e38)
    row_output = act / row_scale[:, None]
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

    trans_scale = tl.max(tl.max(tl.abs(act), axis=0), axis=0) / FP8_MAX
    trans_scale = tl.clamp(trans_scale, 1e-12, 3e38)
    trans_output = act / trans_scale
    trans_output = tl.clamp(trans_output, FP8_MIN, FP8_MAX).trans(1, 0).to(trans_output_ptr.dtype.element_ty)

    trans_output_offsets = (
        trans_output_ptr
        + hidden_offsets[:, None].to(tl.int64) * M_EXPAND
        + token_offsets_expand[None, :].to(tl.int64)
    )
    trans_scale_offset = trans_scale_ptr + pid_n * (M_EXPAND // GROUP_SIZE) + pid_m
    tl.store(trans_output_offsets, trans_output, mask=hidden_mask[:, None])
    tl.store(trans_scale_offset, trans_scale)


@torch.library.custom_op("float8::swiglu_per_tile_quant_with_trans_per_block", mutates_args=())
def swiglu_per_tile_quant_with_trans_per_block(
    gate_up: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, double_n = gate_up.shape
    n = double_n // 2
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
    row_output = gate_up.new_empty((m, n), dtype=dtype)
    row_scales = gate_up.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = gate_up.new_empty((n, m_expand), dtype=dtype)
    trans_scales = gate_up.new_empty((n // group_size, m_expand // group_size), dtype=torch.float32)

    grid = (triton.cdiv(n, group_size), triton.cdiv(m_expand, group_size))
    _swiglu_per_tile_quant_with_trans_per_block_kernel[grid](
        gate_up,
        row_output,
        row_scales,
        trans_output,
        trans_scales,
        group_pad_offsets,
        group_ids_per_tile,
        token_cumdiffs,
        token_ends,
        gate_up.stride(0),
        gate_up.stride(1),
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
        num_warps=8,
        num_stages=3,
    )
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand


@swiglu_per_tile_quant_with_trans_per_block.register_fake
def _(
    gate_up: torch.Tensor,
    size_per_group: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, double_n = gate_up.shape
    n = double_n // 2
    num_groups = size_per_group.shape[0]
    m_expand = m + group_size * num_groups - m % group_size
    row_output = gate_up.new_empty((m, n), dtype=dtype)
    row_scales = gate_up.new_empty((m, n // group_size), dtype=torch.float32)
    trans_output = gate_up.new_empty((n, m_expand), dtype=dtype)
    trans_scales = gate_up.new_empty((n // group_size, m_expand // group_size), dtype=torch.float32)
    size_per_group_expand = torch.empty_like(size_per_group)
    return row_output, row_scales, trans_output, trans_scales, size_per_group_expand

