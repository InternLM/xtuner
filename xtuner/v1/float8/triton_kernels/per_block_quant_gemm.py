# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.profiler import ProfilerActivity, profile

from xtuner.v1.float8.float8_utils import to_fp8_saturated
from xtuner.v1.utils import maybe_compile


@triton.jit
def per_block_quant_gemm_kernel(
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
    M: tl.constexpr,
    K: tl.constexpr,
):
    """Quant fp8 kernel.

    Args:
        a_ptr: the input tensor for quantization. Shape [M, K].
        out_ptr: the output tensor after quantization. Shape [M, K].
        scale_ptr: the output scale. Shape [M, K // group_size].
        fp8_min: the minimal value of fp8 dtype.
        fp8_max: the maximum value of fp8 dtype.
        stride_am: the stride of a_ptr along M dim. Equal to K.
        stride_ak: the stride of a_ptr along K dim. Equal to 1.
        stride_om: the stride of out_ptr along M dim. Equal to K.
        stride_ok: the stride of out_ptr along K dim. Equal to 1.
        stride_sm: the stride of scale_ptr along M dim. Equal to K//group_size.
        stride_sg: the stride of scale_ptr along group_num dim. Equal to 1.
        GROUP_SIZE: the group size for quantization.
        M: shape[0] of a_ptr.
        K: shape[1] of a_ptr.
    """
    group_id = tl.program_id(1)
    m_id = tl.program_id(0)
    k_offs = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    m_offs = m_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    m_mask = m_offs < M
    k_mask = k_offs < K
    mk_mask = m_mask[:, None] & k_mask[None, :]
    # avoid int32 overflow although it is not expected to happen in linear module.
    # upcast to int64 just in case.
    a_ptrs = a_ptr + m_offs[:, None].to(tl.int64) * stride_am + k_offs[None, :].to(tl.int64) * stride_ak
    o_ptrs = out_ptr + m_offs[:, None].to(tl.int64) * stride_om + k_offs[None, :].to(tl.int64) * stride_ok
    s_ptr = scale_ptr + m_id * stride_sm + group_id * stride_sg
    rfp8_max = 1 / fp8_max
    a = tl.load(a_ptrs, mask=mk_mask, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(a)) * rfp8_max
    scale = tl.clamp(scale, 1e-12, 3e38)
    out = a / scale
    out = tl.clamp(out, fp8_min, fp8_max)
    out = out.to(out_ptr.dtype.element_ty)
    tl.store(o_ptrs, out, mask=mk_mask)
    tl.store(s_ptr, scale)


@triton_op("float8::per_block_quant_gemm", mutates_args={})
def per_block_quant_gemm(
    A: Tensor, group_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per tile quant online.

    Args:
        A (Tensor): The input tensor with shape [M, K]. Quantization along
            K dim with a fixed group size.
        group_size (int): The group size for per tile quantization.
        dtype (torch.dtype): The output tensor dtype.
    Returns:
        out (Tensor): The output quantized tensor with shape [M, K].
        scales (Tensor): The scales of the quantized tensor. Shape
            [M, K // group_size].
    """
    assert A.dim() == 2
    M, K = A.shape
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max
    out = torch.empty_like(A, dtype=dtype)
    scales = A.new_empty(triton.cdiv(M, group_size), triton.cdiv(K, group_size), dtype=torch.float32)
    grid = (triton.cdiv(M, group_size), triton.cdiv(K, group_size))
    wrap_triton(per_block_quant_gemm_kernel)[grid](
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


def _get_min_alignment(size: int, alignment_value: int) -> int:
    return (1 + ((size - 1) // alignment_value)) * alignment_value


def pad_tensor_for_matmul(tensor, dims) -> torch.Tensor:
    assert tensor.dim() == 2
    dim1, dim2 = tensor.shape

    if isinstance(dims, int):
        dims = (dims,)

    dim1_aligned = _get_min_alignment(dim1, 128) if 0 in dims else dim1
    dim2_aligned = _get_min_alignment(dim2, 128) if 1 in dims else dim2

    pad_dim1 = dim1_aligned - dim1
    pad_dim2 = dim2_aligned - dim2

    return torch.nn.functional.pad(tensor, (0, pad_dim2, 0, pad_dim1))


@torch.no_grad()
@maybe_compile(fullgraph=True)
def per_block_quant_torch(tensor: torch.Tensor, block_size=128, float8_dtype=torch.float8_e4m3fn):
    dim0, dim1 = tensor.shape
    tensor_pad = pad_tensor_for_matmul(tensor, (0, 1))
    dim0_pad, dim1_pad = tensor_pad.shape
    tensor_pad = (
        tensor_pad.view(dim0_pad // block_size, block_size, dim1_pad // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    amax = tensor_pad.abs().amax(-1, True)
    amax = amax.to(torch.float64)
    scales = amax / torch.finfo(float8_dtype).max
    scales = scales.to(torch.float32)
    tensor_pad_scaled = tensor_pad.float() / scales
    tensor_pad_bits_fp8 = to_fp8_saturated(tensor_pad_scaled, float8_dtype)
    tensor_pad_bits_fp8 = (
        tensor_pad_bits_fp8.view(dim0_pad // block_size, dim1_pad // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dim0_pad, dim1_pad)
    )
    scales = scales.view(dim0_pad // block_size, dim1_pad // block_size)
    tensor_pad_bits_fp8 = tensor_pad_bits_fp8[:dim0, :dim1]
    return tensor_pad_bits_fp8, scales


if __name__ == "__main__":
    x = torch.randn(512 + 64, 2048, device="cuda", dtype=torch.bfloat16)
    # x = torch.randn(32768, 10240, device="cuda", dtype=torch.bfloat16)
    out, scale = per_block_quant_gemm(x, 128, torch.float8_e4m3fn)
    out_ref, scale_ref = per_block_quant_torch(x)
    print(f"(out_ref - out).abs().max(): {(out_ref.float() - out.float()).abs().max()}")
    print(f"(scale_ref - scale).abs().max(): {(scale_ref - scale).abs().max()}")

    for _ in range(4):
        out, scale = per_block_quant_gemm(x, 128, torch.float8_e4m3fn)
        out_ref, scale_ref = per_block_quant_torch(x)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(2):
            out, scale = per_block_quant_gemm(x, 128, torch.float8_e4m3fn)
        for _ in range(2):
            out_ref, scale_ref = per_block_quant_torch(x)

    prof.export_chrome_trace("per_block_quant_triton_vs_compile.json")
    breakpoint()
