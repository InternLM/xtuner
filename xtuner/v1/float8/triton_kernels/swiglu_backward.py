# Copyright (c) OpenMMLab. All rights reserved.

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu_backward_kernel(
    gate_up_ptr,
    grad_act_ptr,
    grad_gate_up_ptr,
    stride_gate_up_m: tl.constexpr,
    stride_gate_up_n: tl.constexpr,
    stride_grad_act_m: tl.constexpr,
    stride_grad_act_n: tl.constexpr,
    stride_grad_gate_up_m: tl.constexpr,
    stride_grad_gate_up_n: tl.constexpr,
    N: tl.constexpr,
    NUM_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ELEMENTS
    rows = offsets // N
    cols = offsets % N

    gate_offsets = gate_up_ptr + rows.to(tl.int64) * stride_gate_up_m + cols.to(tl.int64) * stride_gate_up_n
    up_offsets = gate_offsets + N * stride_gate_up_n
    grad_act_offsets = (
        grad_act_ptr + rows.to(tl.int64) * stride_grad_act_m + cols.to(tl.int64) * stride_grad_act_n
    )

    gate = tl.load(gate_offsets, mask=mask).to(tl.float32)
    up = tl.load(up_offsets, mask=mask).to(tl.float32)
    grad_act = tl.load(grad_act_offsets, mask=mask).to(tl.float32)

    exp_neg_gate = libdevice.exp(-gate)
    sigmoid = libdevice.div_rn(1.0, 1.0 + exp_neg_gate)
    silu = libdevice.div_rn(gate, 1.0 + exp_neg_gate).to(tl.bfloat16).to(tl.float32)
    grad_silu = (grad_act * up).to(tl.bfloat16).to(tl.float32)
    grad_gate = grad_silu * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_up = grad_act * silu

    grad_gate_offsets = (
        grad_gate_up_ptr
        + rows.to(tl.int64) * stride_grad_gate_up_m
        + cols.to(tl.int64) * stride_grad_gate_up_n
    )
    grad_up_offsets = grad_gate_offsets + N * stride_grad_gate_up_n
    tl.store(grad_gate_offsets, grad_gate, mask=mask)
    tl.store(grad_up_offsets, grad_up, mask=mask)


@torch.library.custom_op("float8::swiglu_backward", mutates_args=())
def swiglu_backward(gate_up: torch.Tensor, grad_act: torch.Tensor) -> torch.Tensor:
    m, double_n = gate_up.shape
    n = double_n // 2
    grad_gate_up = torch.empty_like(gate_up)
    num_elements = m * n
    if num_elements == 0:
        return grad_gate_up

    block_size = 2048
    grid = (triton.cdiv(num_elements, block_size),)
    _swiglu_backward_kernel[grid](
        gate_up,
        grad_act,
        grad_gate_up,
        gate_up.stride(0),
        gate_up.stride(1),
        grad_act.stride(0),
        grad_act.stride(1),
        grad_gate_up.stride(0),
        grad_gate_up.stride(1),
        N=n,
        NUM_ELEMENTS=num_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )
    return grad_gate_up


@swiglu_backward.register_fake
def _(gate_up: torch.Tensor, grad_act: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(gate_up)
