# Copyright (c) OpenMMLab. All rights reserved.
"""Private CUDA layout kernels, imported only when a CUDA custom op
executes."""

import triton
import triton.language as tl


@triton.jit
def _transpose_2d_kernel(x_ptr: tl.tensor, y_ptr: tl.tensor, S: int, D: int, BLOCK: tl.constexpr) -> None:
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_s = (pid_s * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    offs_d = (pid_d * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = (offs_s[:, None] < S) & (offs_d[None, :] < D)
    value = tl.load(x_ptr + offs_s[:, None] * D + offs_d[None, :], mask=mask, other=0)
    tl.store(y_ptr + offs_d[:, None] * S + offs_s[None, :], tl.trans(value), mask=tl.trans(mask))


@triton.jit
def _swap_groups_kernel(
    X: tl.tensor,
    Y: tl.tensor,
    A: int,
    C: int,
    N: int,
    S0: int,
    S1: int,
    S2: int,
    BLOCK: tl.constexpr,
) -> None:
    A = tl.cast(A, tl.int64)
    C = tl.cast(C, tl.int64)
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    a = index // C % A
    b = index // (A * C)
    c = index % C
    value = tl.load(X + a * S0 + b * S1 + c * S2, mask=index < N, other=0)
    tl.store(Y + index, value, mask=index < N)
