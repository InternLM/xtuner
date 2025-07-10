from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from xtuner.v1.float8.float8_utils import to_fp8_saturated


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_N": 128,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_N": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["K"])
@triton.jit
def trans_per_tile_quant_gemm_kernel(
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
    BLOCK_N: tl.constexpr,
):
    """Transpose+quant kernel."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    m = pid // num_pid_n
    n = pid % num_pid_n
    offs_m = m * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    # avoid int32 overflow although it is not expected to happen in linear module.
    # upcast to int64 just in case.
    input_offset = input_ptr + offs_m[:, None].to(tl.int64) * stride_in_m + offs_n[None, :].to(tl.int64) * stride_in_n
    output_offset = (
        output_ptr + offs_n[None, :].to(tl.int64) * stride_out_n + offs_m[:, None].to(tl.int64) * stride_out_m
    )
    output_scale_offset = output_scales_ptr + m * stride_out_scale_m + offs_n * stride_out_scale_n
    input_block = tl.load(input_offset, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    output_block_scale = tl.max(tl.abs(input_block), 0) / fmax
    output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
    input_block = input_block / output_block_scale[None, :]
    input_block = tl.clamp(input_block, fmin, fmax).to(output_ptr.dtype.element_ty)
    tl.store(output_offset, input_block, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(output_scale_offset, output_block_scale, mask=mask_n)


@torch.library.custom_op("float8::trans_per_tile_quant_gemm", mutates_args=())
def trans_per_tile_quant_gemm(
    input_tensor: torch.Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Do per-tile quantization along the first dimension.

    And finally transpose it.
    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N]. It is
            per-tile quantized along N dim.
        group_size (int): The group size of the quantization. Default to 128.
        dtype (torch.dtype): The output tensor dtype.
    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-tile
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, group_num_m].
            group_num_m=(M+group_size-1)//group_size.
    """
    M, N = input_tensor.shape
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max
    output_tensor = input_tensor.new_empty((N, M), dtype=dtype)
    output_scales = input_tensor.new_empty((N, triton.cdiv(M, group_size)), dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["GROUP_SIZE"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    trans_per_tile_quant_gemm_kernel[grid](
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


@trans_per_tile_quant_gemm.register_fake
def _(
    input_tensor: torch.Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = input_tensor.shape
    output_tensor = input_tensor.new_empty((N, M), dtype=dtype)
    output_scales = input_tensor.new_empty((N, triton.cdiv(M, group_size)), dtype=torch.float32)
    return output_tensor, output_scales


@torch.no_grad()
def per_tile_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    seq, dim = x.shape
    x = x.view(-1, 128)
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    x_quanted = x_quanted.view(seq, dim)
    x_scales = x_scales.view(seq, -1)
    return x_quanted, x_scales


@torch.no_grad()
@torch.compile(fullgraph=True)
def per_tile_trans_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_trans = x.transpose(0, 1).contiguous()
    return per_tile_quant(x_trans, eps, quant_dtype)


if __name__ == "__main__":
    x = torch.randn(32768, 10240, device="cuda", dtype=torch.bfloat16)

    for _ in range(2):
        out, scale = trans_per_tile_quant_gemm(x, 128, torch.float8_e4m3fn)
        out_ref, scale_ref = per_tile_trans_quant(x)

    print(trans_per_tile_quant_gemm_kernel.best_config)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(2):
            out, scale = trans_per_tile_quant_gemm(x, 128, torch.float8_e4m3fn)
        for _ in range(2):
            out_ref, scale_ref = per_tile_trans_quant(x)

    prof.export_chrome_trace("trans_per_tile_quant_triton_vs_compile.json")
    breakpoint()
