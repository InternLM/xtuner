# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from xtuner._lite.accelerate.float8_gmm.float8_utils import to_fp8_saturated


@triton.jit
def trans_per_block_quant_expand_128x_kernel(
    input_ptr,
    output_ptr,
    output_scales_ptr,
    seq_start_per_expert,
    seq_end_per_expert,
    seq_start_per_expert_expand,
    seq_end_per_expert_expand,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_n,
    stride_out_m: tl.constexpr,
    stride_out_scale_n,
    stride_out_scale_m: tl.constexpr,
    M,
    N: tl.constexpr,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequant+transpose+quant kernel."""
    pid = tl.program_id(axis=0)
    pid_e = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    m = pid // num_pid_n
    n = pid % num_pid_n

    seq_start_expand = tl.load(seq_start_per_expert_expand + pid_e)
    seq_start = tl.load(seq_start_per_expert + pid_e)
    seq_end = tl.load(seq_end_per_expert + pid_e)
    seq_len = seq_end - seq_start

    if m * BLOCK_SIZE >= seq_len:
        return

    offs_m = m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_m = offs_m < seq_len
    mask_n = offs_n < N

    input_offset = (
        input_ptr
        + seq_start * stride_in_m
        + offs_m[:, None] * stride_in_m
        + offs_n[None, :] * stride_in_n
    )
    output_offset = (
        output_ptr
        + offs_n[None, :] * stride_out_n
        + offs_m[:, None] * stride_out_m
        + seq_start_expand * stride_out_m
    )

    output_scale_offset = (
        output_scales_ptr
        + (seq_start_expand // BLOCK_SIZE) * stride_out_scale_m
        + m * stride_out_scale_m
        + n * stride_out_scale_n
    )

    input_block = tl.load(
        input_offset, mask=mask_m[:, None] & mask_n[None, :], other=0.0
    )
    output_block_scale = tl.max(tl.max(tl.abs(input_block), 0), 0) / fmax
    output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
    input_block = input_block / output_block_scale
    input_block = tl.clamp(input_block, fmin, fmax).to(output_ptr.dtype.element_ty)
    tl.store(output_offset, input_block, mask=mask_n[None, :])
    tl.store(output_scale_offset, output_block_scale)


@triton_op("myfp8::trans_per_block_quant_expand_128x", mutates_args={})
def trans_per_block_quant_expand_128x(
    input_tensor: torch.Tensor,
    seq_len_per_expert: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequant a per-group per-tile quantized tensor. Do per-block quantization
    along another dimension. And finally transpose it.

    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N]. It is
            per-tile quantized along N dim.
        input_scales (Tensor): The input scale. Shape [M, group_num_n].
            group_num_n=N//group_size.
        seq_len_per_expert (Tensor): The seq length of each expert. The sum of it
            should be equal to M.
        group_size (int): The group size of the quantization. Default to 128.

    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-tile
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, group_num_m].
            group_num_m=(M_EXPAND+group_size-1)//group_size.
    """
    M, N = input_tensor.shape
    assert N % group_size == 0
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    num_experts = seq_len_per_expert.shape[0]
    seq_len_per_expert_expand = triton.cdiv(seq_len_per_expert, 128) * 128
    seq_end_per_expert_expand = seq_len_per_expert_expand.cumsum(0)
    seq_start_per_expert_expand = seq_end_per_expert_expand - seq_len_per_expert_expand
    seq_end_per_expert = seq_len_per_expert.cumsum(0)
    seq_start_per_expert = seq_end_per_expert - seq_len_per_expert
    M_EXPAND = M + 128 * num_experts - M % 128

    output_tensor = input_tensor.new_empty((N, M_EXPAND), dtype=dtype)
    output_scales = input_tensor.new_empty(
        (N // group_size, triton.cdiv(M_EXPAND, group_size)), dtype=torch.float32
    )

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]) * triton.cdiv(N, meta["BLOCK_SIZE"]),
        num_experts,
    )

    wrap_triton(trans_per_block_quant_expand_128x_kernel)[grid](
        input_tensor,
        output_tensor,
        output_scales,
        seq_start_per_expert,
        seq_end_per_expert,
        seq_start_per_expert_expand,
        seq_end_per_expert_expand,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_scales.stride(0),
        output_scales.stride(1),
        M,
        N,
        fmax=fmax,
        fmin=fmin,
        BLOCK_SIZE=group_size,
    )

    return output_tensor, output_scales, seq_len_per_expert_expand


def per_block_trans_quant_expand_per_expert(
    x, eps=1e-12, block_size=128, quant_dtype=torch.float8_e4m3fn
):
    x = x.T
    dim, seq = x.shape
    seq_expand = triton.cdiv(seq, block_size) * block_size
    x_expand = torch.cat([x, x.new_zeros((dim, seq_expand - seq))], dim=-1)

    x_expand = (
        x_expand.reshape(
            dim // block_size, block_size, seq_expand // block_size, block_size
        )
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    amax = x_expand.abs().amax(-1, True)
    scale = amax.float() / torch.finfo(quant_dtype).max
    x_fp8 = to_fp8_saturated(x_expand / scale, quant_dtype)
    x_fp8 = (
        x_fp8.reshape(
            dim // block_size, seq_expand // block_size, block_size, block_size
        )
        .transpose(1, 2)
        .reshape(dim, seq_expand)
    )
    scale = scale.reshape(dim // block_size, seq_expand // block_size)

    return x_fp8, scale


def per_block_trans_quant_expand_torch(x, tokens_per_expert):
    M = x.shape[0]
    ne = tokens_per_expert.shape[0]

    x_list = torch.split(x, tokens_per_expert.tolist(), dim=0)
    x_trans_quant_list, x_trans_quant_scale_list = [], []
    for x in x_list:
        x_fp8, scale = per_block_trans_quant_expand_per_expert(x)
        x_trans_quant_list.append(x_fp8)
        x_trans_quant_scale_list.append(scale)
    x_trans_quant = torch.cat(x_trans_quant_list, dim=-1)
    x_trans_quant_scale = torch.cat(x_trans_quant_scale_list, dim=-1)

    pad_len = M + 128 * ne - M % 128 - x_trans_quant.shape[1]
    pad = x_trans_quant.new_zeros((x_trans_quant.shape[0], pad_len))
    x_trans_quant = torch.cat([x_trans_quant, pad], dim=1)

    pad_len = (M + 128 * ne - M % 128) // 128 - x_trans_quant_scale.shape[1]
    pad = x_trans_quant_scale.new_zeros((x_trans_quant_scale.shape[0], pad_len))
    x_trans_quant_scale = torch.cat([x_trans_quant_scale, pad], dim=1)

    return x_trans_quant, x_trans_quant_scale, triton.cdiv(tokens_per_expert, 128) * 128


if __name__ == "__main__":
    import copy

    torch.manual_seed(0)
    din = 5120
    dout = 3072
    ne = 32
    seq = 8192
    x_temp = torch.randn(
        seq * 8, din, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    x = copy.deepcopy(x_temp)

    bias = 3
    tokens_per_expert = (
        torch.tensor(
            [512 - bias, 2 * 2048 - 512 + bias, 128 - bias, 2 * 2048 - 128 + bias] * 8
        )
        .cuda()
        .long()
    )

    (
        output_tensor,
        output_scales,
        seq_len_per_expert_expand,
    ) = trans_per_block_quant_expand_128x(x, tokens_per_expert)
    (
        output_tensor_ref,
        output_scales_ref,
        seq_len_per_expert_expand_ref,
    ) = per_block_trans_quant_expand_torch(x, tokens_per_expert)
    breakpoint()
