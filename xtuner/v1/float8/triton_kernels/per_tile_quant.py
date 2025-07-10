# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from xtuner.v1.float8.float8_utils import to_fp8_saturated


def get_cuda_autotune_config():
    return [
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_M": 128,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["K"])
@triton.jit
def per_tile_quant_kernel(
    a_ptr,
    out_ptr,
    scale_ptr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    stride_sg: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    M,
    K,
    BLOCK_M: tl.constexpr,
):
    group_id = tl.program_id(1)
    m_id = tl.program_id(0)
    k_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    m_offs = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    k_mask = k_offs < K
    mk_mask = m_mask[:, None] & k_mask[None, :]
    # avoid int32 overflow although it is not expected to happen in linear module.
    # upcast to int64 just in case.
    a_ptrs = a_ptr + m_offs[:, None].to(tl.int64) * stride_am + k_offs[None, :].to(tl.int64) * stride_ak
    o_ptrs = out_ptr + m_offs[:, None].to(tl.int64) * stride_om + k_offs[None, :].to(tl.int64) * stride_ok
    s_ptr = scale_ptr + m_offs * stride_sm + group_id * stride_sg
    rfp8_max = 1 / fp8_max
    a = tl.load(a_ptrs, mask=mk_mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(a), 1) * rfp8_max
    scale = tl.clamp(scale, 1e-12, 3e38)
    out = a / scale[:, None]
    out = tl.clamp(out, fp8_min, fp8_max)
    out = out.to(out_ptr.dtype.element_ty)
    tl.store(o_ptrs, out, mask=mk_mask)
    tl.store(s_ptr, scale, mask=m_mask)


@torch.library.custom_op("float8::per_tile_quant", mutates_args=())
def per_tile_quant(
    A: Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert A.dim() == 2
    M, K = A.shape
    num_groups = triton.cdiv(K, group_size)
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max
    out = torch.empty_like(A, dtype=dtype)
    scales = A.new_empty(M, num_groups, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), num_groups)  # noqa: E731
    per_tile_quant_kernel[grid](
        A,
        out,
        scales,
        fp8_min=fmin,
        fp8_max=fmax,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_om=out.stride(0),
        stride_ok=out.stride(1),
        stride_sm=scales.stride(0),
        stride_sg=scales.stride(1),
        GROUP_SIZE=group_size,
        M=M,
        K=K,
    )
    return out, scales


@per_tile_quant.register_fake
def _(A: Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = A.shape
    num_groups = triton.cdiv(K, group_size)
    out = torch.empty_like(A, dtype=dtype)
    scales = A.new_empty(M, num_groups, dtype=torch.float32)
    return out, scales


@torch.compile(fullgraph=True)
def per_tile_quant_torch(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    seq, dim = x.shape
    x = x.view(-1, 128)
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    x_quanted = x_quanted.view(seq, dim)
    x_scales = x_scales.view(seq, -1)
    return x_quanted, x_scales


if __name__ == "__main__":
    x = torch.randn(32768, 10240, device="cuda", dtype=torch.bfloat16)
    out, scale = per_tile_quant(x, 128, torch.float8_e4m3fn)
    out_ref, scale_ref = per_tile_quant_torch(x)

    for _ in range(4):
        out, scale = per_tile_quant(x, 128, torch.float8_e4m3fn)
        out_ref, scale_ref = per_tile_quant_torch(x)

    print(per_tile_quant_kernel.best_config)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(2):
            out, scale = per_tile_quant(x, 128, torch.float8_e4m3fn)
        for _ in range(2):
            out_ref, scale_ref = per_tile_quant_torch(x)

    prof.export_chrome_trace("per_tile_quant_triton_vs_compile.json")
    breakpoint()
