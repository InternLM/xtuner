# Copied from https://github.com/deepseek-ai/DeepSeek-V3/blob/4a65fd9221103ff03864337453c238e65d1f4a1b/inference/kernel.py

import torch
import triton
import triton.language as tl


@triton.jit
def per_block_dequant_gemm_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Dequantizes weights using the provided scaling factors and stores the
    result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def per_block_dequant_gemm(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M//block_size, N//block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))  # noqa: E731
    per_block_dequant_gemm_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y
