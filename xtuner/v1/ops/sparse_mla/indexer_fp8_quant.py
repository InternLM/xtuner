# Copyright (c) OpenMMLab. All rights reserved.
"""FP8 quantization for the GLM-5.2 Indexer.

The Indexer has a fixed 128-wide projection dimension. Quantize each post-RoPE Q head row and each K row independently
with an E4M3 value and an UE8M0 (power-of-two) scale. Keep this quantizer next to the Indexer rather than extending
XTuner's generic Linear/MoE FP8 helper: the generic helper's historical scale contract must remain unchanged. The row-
wise format follows LMDeploy's FP8 Indexer quantization contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


INDEXER_HEAD_DIM = 128
FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _fast_log2_ceil(x):
    bits_x = tl.cast(x, tl.uint32, bitcast=True)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return tl.cast(exp_x - 127 + tl.where(man_bits != 0, 1, 0), tl.int32)


@triton.jit
def _fast_pow2(x):
    bits_x = (x + 127) << 23
    return tl.cast(bits_x, tl.float32, bitcast=True)


@triton.jit
def _fast_round_scale(amax, fp8_max_inv):
    return _fast_pow2(_fast_log2_ceil(amax * fp8_max_inv))


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_stages=3, num_warps=8),
    ]


@triton.autotune(configs=_autotune_configs(), key=["K"])
@triton.jit
def _indexer_fp8_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_om,
    stride_ok: tl.constexpr,
    stride_sm,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    block_m = tl.program_id(0)
    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # ``K`` is a constexpr kernel argument.  Referencing the Python module
    # constant directly from a Triton JIT function is rejected by recent
    # Triton versions unless that global is wrapped in ``tl.constexpr``.
    # Using the already-validated constexpr ``K`` also keeps this kernel
    # portable across Triton releases.
    cols = tl.arange(0, K)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    # Use 64-bit address arithmetic: a long-context GLM-5.2 batch can have
    # more than 2^31 Q elements when rows are flattened as S*H.
    row_offsets = rows.to(tl.int64)
    col_offsets = cols.to(tl.int64)
    input_ptrs = input_ptr + row_offsets[:, None] * stride_am + col_offsets[None, :] * stride_ak
    output_ptrs = output_ptr + row_offsets[:, None] * stride_om + col_offsets[None, :] * stride_ok
    values = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    scale = _fast_round_scale(tl.maximum(amax, 1e-6), 1 / fp8_max)
    # Keep reciprocal-then-multiply order for UE8M0 scales. This avoids a
    # needless division-rounding difference in FP8 bytes at the edge of a
    # representable value.
    values = tl.clamp(values * (1.0 / scale[:, None]), fp8_min, fp8_max)
    tl.store(output_ptrs, values.to(output_ptr.dtype.element_ty), mask=mask)
    tl.store(scale_ptr + row_offsets * stride_sm, scale, mask=rows < M)


@torch.library.custom_op("sparse_mla::indexer_fp8_quant", mutates_args=())
def indexer_fp8_quant(x: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize BF16 Indexer rows to E4M3 values and UE8M0 scales.

    ``x`` may have arbitrary leading dimensions but its final dimension must
    be the GLM-5.2 Indexer width (128).  The returned scale has the same
    leading dimensions as ``x``; one scale is emitted for every final-dim
    row.  No dequantization is performed here. The row-wise rule matches the
    format used by LMDeploy's FP8 Indexer.
    """

    if x.ndim < 2 or x.shape[-1] != INDEXER_HEAD_DIM:
        raise ValueError(f"Indexer FP8 quantizer expects [..., 128], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Indexer FP8 quantizer expects BF16 input, got {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError("Indexer FP8 quantizer requires CUDA")
    if not x.is_contiguous():
        raise ValueError("Indexer FP8 quantizer requires contiguous input")

    shape = x.shape
    rows = x.numel() // INDEXER_HEAD_DIM
    flat = x.view(rows, INDEXER_HEAD_DIM)
    output = torch.empty_like(flat, dtype=FP8_DTYPE)
    scales = torch.empty((rows,), device=x.device, dtype=torch.float32)
    # Triton does not accept a zero-sized launch grid.  Empty packed batches
    # are valid at the operator boundary, so return their correctly shaped
    # outputs without launching a kernel.
    if rows == 0:
        return output.view(shape), scales.view(shape[:-1])

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)  # noqa: E731
    _indexer_fp8_quant_kernel[grid](
        flat,
        output,
        scales,
        fp8_min=torch.finfo(FP8_DTYPE).min,
        fp8_max=torch.finfo(FP8_DTYPE).max,
        stride_am=flat.stride(0),
        stride_ak=flat.stride(1),
        stride_om=output.stride(0),
        stride_ok=output.stride(1),
        stride_sm=scales.stride(0),
        M=rows,
        K=INDEXER_HEAD_DIM,
    )
    return output.view(shape), scales.view(shape[:-1])


@indexer_fp8_quant.register_fake
def _(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.ndim < 2 or x.shape[-1] != INDEXER_HEAD_DIM:
        raise ValueError(f"Indexer FP8 quantizer expects [..., 128], got {tuple(x.shape)}")
    return torch.empty_like(x, dtype=FP8_DTYPE), torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)


__all__ = ["FP8_DTYPE", "INDEXER_HEAD_DIM", "indexer_fp8_quant"]
