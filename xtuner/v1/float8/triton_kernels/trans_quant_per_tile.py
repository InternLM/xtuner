# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def get_cuda_autotune_config():
    return [
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
            },
            num_stages=4,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["N"])
@triton.jit
def trans_per_tile_quant_expand_128x_kernel(
    input_ptr,
    output_ptr,
    output_scales_ptr,
    group_start,
    group_end,
    group_start_expand,
    group_end_expand,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_n,
    stride_out_m: tl.constexpr,
    stride_out_scale_n,
    stride_out_scale_m: tl.constexpr,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dequant+transpose+quant kernel."""
    pid_n = tl.program_id(axis=0)
    group_id = tl.program_id(axis=1)

    group_start = tl.load(group_start + group_id)
    group_end = tl.load(group_end + group_id)
    group_size = group_end - group_start

    if group_size <= 0:
        return

    group_start_expand = tl.load(group_start_expand + group_id)
    group_end_expand = tl.load(group_end_expand + group_id)

    # load a block inside a group
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = group_start + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N

    offs_m_expand = group_start_expand + tl.arange(0, BLOCK_M)
    for i in tl.range(0, tl.cdiv(group_size, BLOCK_M)):
        # avoid int32 overflow when seqlen is large
        input_offset = (
            input_ptr + offs_m[:, None].to(tl.int64) * stride_in_m + offs_n[None, :].to(tl.int64) * stride_in_n
        )
        input_block = tl.load(input_offset, mask=(offs_m < group_end)[:, None] & mask_n[None, :], other=0).to(
            tl.float32
        )
        output_block_scale = tl.max(tl.abs(input_block), 0) / fmax
        output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
        scales_block_ptr = (
            output_scales_ptr
            + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_out_scale_n
            + group_start_expand // 128
            + i
        )
        tl.store(scales_block_ptr, output_block_scale, mask=mask_n)

        input_quant = input_block / output_block_scale[None, :]
        # avoid int32 overflow when seqlen is large
        output_offset = (
            output_ptr
            + offs_n[None, :].to(tl.int64) * stride_out_n
            + offs_m_expand[:, None].to(tl.int64) * stride_out_m
        )
        output_block = tl.clamp(input_quant, fmin, fmax).to(output_ptr.dtype.element_ty)
        tl.store(
            output_offset,
            output_block,
            mask=(offs_m_expand < group_end_expand)[:, None] & mask_n[None, :],
        )
        offs_m += BLOCK_M
        offs_m_expand += BLOCK_M


@triton_op("float8::trans_per_tile_quant_expand_128x", mutates_args={})
def trans_per_tile_quant_expand_128x(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequant a stacked per-channel quantized tensor. Do per-channel
    quantization along another dimension. And finally transpose it.
    input_tensor and input_scales should be colom major. While the output
    tensor would be row major.

    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N].
        size_per_group (Tensor): The token numbers for each expert. Shape [num_groups].
            size_per_group.sum() == M.
        dtype (dtype): The torch dtype of fp8. Default to float8_e4m3fn.

    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-channel
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, 1].
    """
    M, N = input_tensor.shape
    num_groups = size_per_group.shape[0]
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    size_per_group_expand = triton.cdiv(size_per_group, 128) * 128
    group_end_expand = size_per_group_expand.cumsum(0)
    group_start_expand = group_end_expand - size_per_group_expand
    output_tensor = input_tensor.new_empty((N, M + 128 * num_groups - M % 128), dtype=dtype)
    output_scales = input_tensor.new_empty((N, (M + 128 * num_groups - M % 128) // 128), dtype=torch.float32)

    group_end = size_per_group.cumsum(0)
    group_start = group_end - size_per_group
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), num_groups)  # noqa: E731
    wrap_triton(trans_per_tile_quant_expand_128x_kernel)[grid](
        input_tensor,
        output_tensor,
        output_scales,
        group_start,
        group_end,
        group_start_expand,
        group_end_expand,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_scales.stride(0),
        output_scales.stride(1),
        fmax=fmax,
        fmin=fmin,
        eps=eps,
        N=N,
    )

    return output_tensor, output_scales, size_per_group_expand


# def per_tile_quant(x : torch.Tensor):
#     m, n = x.shape
#     m_aligned = (m + 127) // 128 * 128
#     out = torch.zeros((n, m_aligned),device="cuda",dtype=x.dtype)
#     out[:n,:m] = x.t()
#     out = out.reshape(-1,128)
#     scale = out.abs().float().amax(-1,True) / 448.0
#     dtype = torch.float8_e4m3fn
#     dmax = torch.finfo(dtype).max if dtype != torch.int8 else 127
#     dmin = torch.finfo(dtype).min if dtype != torch.int8 else -127
#     out = (out / scale).clamp(dmin,dmax).to(torch.float8_e4m3fn)
#     out = out.reshape(n,m_aligned)
#     return out, scale.reshape(n,-1)


def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
    max_value = torch.finfo(float8_dtype).max
    x = x.clamp(min=-max_value, max=max_value)
    return x.to(float8_dtype)


def per_tile_trans_quant_per_expert(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x = x.T
    dim, seq = x.shape
    seq_expand = triton.cdiv(seq, 128) * 128
    x_expand = torch.cat([x, x.new_zeros((dim, seq_expand - seq))], dim=-1)

    x_expand = x_expand.view(-1, 128)
    amax = x_expand.abs().amax(-1, True).to(torch.float64)
    scale = torch.clamp(amax, min=eps) / torch.finfo(quant_dtype).max
    scale = scale.float()
    x_fp8 = to_fp8_saturated(x_expand.float() / scale, quant_dtype).view(dim, seq_expand)
    scale = scale.view(dim, -1)

    return x_fp8, scale


def trans_quant_expand_128x_per_tile_torch(x, tokens_per_expert):
    M = x.shape[0]
    ne = tokens_per_expert.shape[0]

    x_list = torch.split(x, tokens_per_expert.tolist(), dim=0)
    x_trans_quant_list, x_trans_quant_scale_list = [], []
    for x in x_list:
        x_fp8, scale = per_tile_trans_quant_per_expert(x)
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
    # Example usage
    torch.manual_seed(0)
    din = 5120
    dout = 3072
    # ne = 8
    ne = 32
    x = torch.randn(8192 * 8, din, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    bias = 3
    if ne == 8:
        tokens_per_expert = (
            torch.tensor(
                [
                    2 * 1024 - bias,
                    8 * 1024 + bias,
                    14 * 1024 - bias,
                    7 * 1024 + bias,
                    8 * 1024 - bias,
                    9 * 1024 + bias,
                    7.5 * 1024 + bias,
                    8.5 * 1024 - bias,
                ]
            )
            .cuda()
            .long()
        )
        # tokens_per_expert = torch.tensor([1, 1, 1, 1, 1, 1, 1, 65536 - 7]).cuda().long()
    elif ne == 32:
        # tokens_per_expert = torch.tensor(
        #   [512 - bias, 2 * 2048 - 512 + bias, 128 - bias, 2 * 2048 - 128 + bias] * 8).cuda().long()
        # tokens_per_expert = torch.tensor([1] * 31 + [65536 - 31]).cuda().long()
        tokens_per_expert = torch.tensor([0, 2 * 2048, 128 - bias, 2 * 2048 - 128 + bias] * 8).cuda().long()
    else:
        raise NotImplementedError

    (
        x_trans_fp8,
        x_trans_scale,
        tokens_per_expert_expand,
    ) = trans_per_tile_quant_expand_128x(x, tokens_per_expert)

    (
        x_trans_fp8_torch,
        x_trans_scale_torch,
        tokens_per_expert_expand_torch,
    ) = trans_quant_expand_128x_per_tile_torch(x, tokens_per_expert)
    breakpoint()
