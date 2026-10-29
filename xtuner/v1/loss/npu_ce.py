# Copyright (c) OpenMMLab. All rights reserved.
"""Plain Ascend CE backward with one output rounding per gradient element.

The target subtraction must happen in FP32, before storing to a BF16/FP16
output. Storing ``p * scale`` and then updating the target in a second pass
loses significant bits when the target probability is close to one.
"""

import triton
import triton.language as tl


@triton.jit
def cross_entropy_backward_kernel_no_weight(
    X_ptr,
    X_stride,
    Y_ptr,
    lse_ptr,
    grad_output_ptr,
    grad_output_stride,
    dX_ptr,
    dX_stride,
    n_cols,
    n_rows,
    ce_stats_ptr,
    ignore_index,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    """Match the plain Liger Ascend launch ABI; reject extras in the caller."""
    pid = tl.program_id(0)
    rows_per_program = tl.cdiv(n_rows, tl.num_programs(0))
    row_start = pid * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)
    inv_n = tl.load(ce_stats_ptr).to(tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    for row in range(row_start, row_end):
        row_i64 = row.to(tl.int64)
        target = tl.load(Y_ptr + row_i64)
        upstream = tl.load(
            grad_output_ptr + row_i64 * grad_output_stride if reduction == "none" else grad_output_ptr
        ).to(tl.float32)
        scale = upstream * inv_n
        if target == ignore_index:
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offsets
                tl.store(dX_ptr + row_i64 * dX_stride + cols, 0.0, cols < n_cols)
        else:
            lse = tl.load(lse_ptr + row_i64).to(tl.float32)
            if not HAS_LSE:
                target_logit = tl.load(X_ptr + row_i64 * X_stride + target).to(tl.float32)
                if reduction == "mean":
                    lse = lse / inv_n
                lse = lse + target_logit
            for start in range(0, n_cols, BLOCK_SIZE):
                cols = start + offsets
                logits = tl.load(X_ptr + row_i64 * X_stride + cols, cols < n_cols, other=float("-inf")).to(tl.float32)
                probability = tl.exp(logits - lse)
                # Subtract while still FP32, then scale and round only at the store.
                grad = (probability - tl.where(cols == target, 1.0, 0.0)) * scale
                tl.store(dX_ptr + row_i64 * dX_stride + cols, grad, cols < n_cols)
