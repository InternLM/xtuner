from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from xtuner.v1.float8.float8_utils import to_fp8_saturated


@triton.jit
def trans_per_block_quant_gemm_kernel(
    input_ptr,
    output_ptr,
    output_scales_ptr,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_n,
    stride_out_m: tl.constexpr,
    stride_out_scale_n,
    stride_out_scale_m: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, GROUP_SIZE)
    m = pid // num_pid_n
    n = pid % num_pid_n
    offs_m = m * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    offs_n = n * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask_m = offs_m < M
    mask_n = offs_n < N
    # avoid int32 overflow although it is not expected to happen in linear module.
    # upcast to int64 just in case.
    input_offset = input_ptr + offs_m[:, None].to(tl.int64) * stride_in_m + offs_n[None, :].to(tl.int64) * stride_in_n
    output_offset = (
        output_ptr + offs_n[None, :].to(tl.int64) * stride_out_n + offs_m[:, None].to(tl.int64) * stride_out_m
    )
    output_scale_offset = output_scales_ptr + m * stride_out_scale_m + n * stride_out_scale_n
    input_block = tl.load(input_offset, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    output_block_scale = tl.max(tl.abs(input_block)) / fmax
    output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
    input_block = input_block / output_block_scale
    input_block = tl.clamp(input_block, fmin, fmax).to(output_ptr.dtype.element_ty)
    tl.store(output_offset, input_block, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(output_scale_offset, output_block_scale)


@torch.library.custom_op("float8::trans_per_block_quant_gemm", mutates_args=())
def trans_per_block_quant_gemm(
    input_tensor: torch.Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = input_tensor.shape
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max
    output_tensor = input_tensor.new_empty((N, M), dtype=dtype)
    output_scales = input_tensor.new_empty(
        (triton.cdiv(N, group_size), triton.cdiv(M, group_size)), dtype=torch.float32
    )
    grid = (triton.cdiv(M, group_size) * triton.cdiv(N, group_size),)
    trans_per_block_quant_gemm_kernel[grid](
        input_tensor,
        output_tensor,
        output_scales,
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
        GROUP_SIZE=group_size,
    )
    return output_tensor, output_scales


@trans_per_block_quant_gemm.register_fake
def _(
    input_tensor: torch.Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = input_tensor.shape
    output_tensor = input_tensor.new_empty((N, M), dtype=dtype)
    output_scales = input_tensor.new_empty(
        (triton.cdiv(N, group_size), triton.cdiv(M, group_size)), dtype=torch.float32
    )
    return output_tensor, output_scales


@torch.no_grad()
def per_block_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    dout, din = x.shape
    block_size = 128
    x = (
        x.view(dout // block_size, block_size, din // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)

    x_quanted = (
        x_quanted.view(dout // block_size, din // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dout, din)
    )
    x_scales = x_scales.view(dout // block_size, din // block_size)
    return x_quanted, x_scales


@torch.no_grad()
@torch.compile(fullgraph=True)
def per_block_trans_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_trans = x.transpose(0, 1).contiguous()
    return per_block_quant(x_trans, eps, quant_dtype)


if __name__ == "__main__":
    x = torch.randn(32768, 10240, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        out, scale = trans_per_block_quant_gemm(x, 128, torch.float8_e4m3fn)
        out_ref, scale_ref = per_block_trans_quant(x)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(2):
            out, scale = trans_per_block_quant_gemm(x, 128, torch.float8_e4m3fn)
        for _ in range(2):
            out_ref, scale_ref = per_block_trans_quant(x)

    prof.export_chrome_trace("trans_per_block_quant_triton_vs_compile.json")
    breakpoint()
