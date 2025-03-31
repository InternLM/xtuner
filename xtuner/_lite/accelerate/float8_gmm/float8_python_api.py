# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_python_api.py
"""This file defines the Python functions for float8 which expect inputs of
class `Float8Tensor`.

This is a thin wrapper on top of the aten API to simplify the product code.
"""

from typing import Optional

import torch


# [Note] Usage of scales
# The meaning of scale in this library can be found in the definition of the Float8Tensor
# Cublas defines scale to always mean a multiplicative factor for the respective matrices
# For a,b going from fp8 -> fp32 we multiple by the inverse of the scale
# For output going from fp32 -> fp8 we multiply by the scale
def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """This is the unwrapped version of addmm_float8, which does not take in
    Float8Tensors as inputs.

    This is used to standardize the logic between subclassed and non subclassed versions of the linear module.
    """

    post_inverse_scale = None
    if (
        a_scale.shape == (a_data.shape[0], 1)
        and b_scale.shape == (1, b_data.shape[1])
        and not use_fast_accum
    ):
        # The rowwise CUTLASS-based kernel is so slow without fast-accum that
        # we'd rather use the tensorwise cuBLAS-based kernel and do the scaling
        # manually afterwards (hoping Inductor will be able to fuse it).
        post_inverse_scale = a_scale * b_scale
        a_scale = a_scale.new_ones(())
        b_scale = a_scale.new_ones(())

    post_bias = None
    if output_dtype == torch.float32:
        # Bias is not supported by _scaled_mm when output is fp32
        post_bias = bias
        bias = None

    output = torch._scaled_mm(
        a_data,
        b_data,
        scale_a=a_scale,
        scale_b=b_scale,
        bias=bias,
        scale_result=output_scale,
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )

    if post_inverse_scale is not None:
        output *= post_inverse_scale
    if post_bias is not None:
        output += post_bias

    return output
