# Copyright (c) OpenMMLab. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def quantize_per_row(input_block, fp8_min: tl.constexpr, fp8_max: tl.constexpr):
    """Quantize rows in FP32; leave layout and output casting to the caller."""
    reciprocal_fp8_max = 1.0 / fp8_max
    scale = tl.max(tl.abs(input_block), axis=1) * reciprocal_fp8_max
    scale = tl.clamp(scale, 1e-12, 3e38)
    output = input_block / scale[:, None]
    return tl.clamp(output, fp8_min, fp8_max), scale


@triton.jit
def quantize_per_column(input_block, fp8_min: tl.constexpr, fp8_max: tl.constexpr):
    """Quantize columns, preserving the transposed per-tile scale formula."""
    scale = tl.max(tl.abs(input_block), axis=0) / fp8_max
    scale = tl.clamp(scale, 1e-12, 3e38)
    output = input_block / scale[None, :]
    return tl.clamp(output, fp8_min, fp8_max), scale


@triton.jit
def quantize_per_block(
    input_block,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    REDUCE_ALL: tl.constexpr = False,
    USE_RECIPROCAL: tl.constexpr = False,
):
    """Keep each caller's historical reduction and scale arithmetic."""
    if REDUCE_ALL:
        amax = tl.max(tl.abs(input_block))
    else:
        amax = tl.max(tl.max(tl.abs(input_block), axis=0), axis=0)
    if USE_RECIPROCAL:
        reciprocal_fp8_max = 1.0 / fp8_max
        scale = amax * reciprocal_fp8_max
    else:
        scale = amax / fp8_max
    scale = tl.clamp(scale, 1e-12, 3e38)
    output = input_block / scale
    return tl.clamp(output, fp8_min, fp8_max), scale
