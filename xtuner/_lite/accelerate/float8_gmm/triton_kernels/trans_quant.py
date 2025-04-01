# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def get_cuda_autotune_config():
    return [
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256,}, num_stages=4,
        #               num_warps=8),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
            },
            num_stages=4,
            num_warps=8,
        ),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128,}, num_stages=4,
        #               num_warps=8),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
            },
            num_stages=4,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["N"])
@triton.jit
def trans_quant_kernel(
    input_ptr,
    output_ptr,
    output_scales_ptr,
    group_start,
    group_end,
    group_start_expand,
    group_end_expand,
    stride_in_m: tl.constexpr,
    stride_in_n: tl.constexpr,
    stride_out_n,
    stride_out_m: tl.constexpr,
    stride_out_scale_n,
    stride_out_scale_m: tl.constexpr,
    fmax: tl.constexpr,
    fmin: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dequant+transpose+quant kernel."""
    pid_n = tl.program_id(axis=0)
    group_id = tl.program_id(axis=1)

    group_start = tl.load(group_start + group_id)
    group_end = tl.load(group_end + group_id)
    group_size = group_end - group_start

    group_start_expand = tl.load(group_start_expand + group_id)
    group_end_expand = tl.load(group_end_expand + group_id)

    # load a block inside a group
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = group_start + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    max_val_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if group_size <= 0:
        output_scales_ptr_block = (
            output_scales_ptr
            + offs_n * stride_out_scale_n
            + group_id * stride_out_scale_m
        )
        output_block_scale = tl.zeros((BLOCK_N,), dtype=tl.float32) + 1e-12
        tl.store(output_scales_ptr_block, output_block_scale, mask=mask_n)
        return

    for i in tl.range(0, tl.cdiv(group_size, BLOCK_M)):
        input_offset = (
            input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
        )
        input_block = tl.load(
            input_offset, mask=(offs_m < group_end)[:, None] & mask_n[None, :], other=0
        ).to(tl.float32)
        max_val_block = tl.maximum(max_val_block, tl.abs(input_block))
        offs_m += BLOCK_M

    # compute scale
    output_block_scale = tl.max(max_val_block, 0) / fmax
    output_block_scale = tl.clamp(output_block_scale, 1e-12, 3e38)
    output_scales_ptr = (
        output_scales_ptr + offs_n * stride_out_scale_n + group_id * stride_out_scale_m
    )

    # group_end_expand = group_start_expand+group_size
    offs_m = group_start + tl.arange(0, BLOCK_M)
    offs_m_expand = group_start_expand + tl.arange(0, BLOCK_M)
    for i in tl.range(0, tl.cdiv(group_size, BLOCK_M)):
        input_offset = (
            input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
        )
        output_offset = (
            output_ptr
            + offs_n[None, :] * stride_out_n
            + offs_m_expand[:, None] * stride_out_m
        )
        input_block = tl.load(
            input_offset,
            mask=(offs_m < group_end)[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        input_block = input_block / output_block_scale[None, :]
        output_block = tl.clamp(input_block, fmin, fmax).to(output_ptr.dtype.element_ty)
        tl.store(
            output_offset,
            output_block,
            mask=(offs_m_expand < group_end_expand)[:, None] & mask_n[None, :],
        )
        offs_m += BLOCK_M
        offs_m_expand += BLOCK_M
    tl.store(output_scales_ptr, output_block_scale, mask=mask_n)


@triton_op("myfp8::trans_quant_expand_128x", mutates_args={})
def trans_quant_expand_128x(
    input_tensor: torch.Tensor,
    size_per_group: torch.Tensor,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequant a stacked per-channel quantized tensor. Do per-channel
    quantization along another dimension. And finally transpose it.
    input_tensor and input_scales should be colom major. While the output
    tensor would be row major.

    Args:
        input_tensor (Tensor): The input quantized tensor. Shape [M, N].
        size_per_group (Tensor): The token numbers for each expert. Shape [num_groups].
            size_per_group.sum() == M.
        dtype (dtype): The torch dtype of fp8. Default to float8_e4m3fn.

    Returns:
        output_tensor (Tensor): The output tensor. Shape [N, M]. It is per-channel
            quantized along M dim.
        output_scales (Tensor): The output scales. Shape [N, 1].
    """
    M, N = input_tensor.shape
    num_groups = size_per_group.shape[0]
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    size_per_group_expand = triton.cdiv(size_per_group, 128) * 128
    group_end_expand = size_per_group_expand.cumsum(0)
    group_start_expand = group_end_expand - size_per_group_expand
    output_tensor = input_tensor.new_empty(
        (N, M + 128 * num_groups - M % 128), dtype=dtype
    )
    # output_tensor = input_tensor.new_zeros((N, M + 128 * num_groups - M % 128), dtype=dtype)
    output_scales = input_tensor.new_empty((N, num_groups), dtype=torch.float32)

    group_end = size_per_group.cumsum(0)
    group_start = group_end - size_per_group
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), num_groups)
    wrap_triton(trans_quant_kernel)[grid](
        input_tensor,
        output_tensor,
        output_scales,
        group_start,
        group_end,
        group_start_expand,
        group_end_expand,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_scales.stride(0),
        output_scales.stride(1),
        fmax=fmax,
        fmin=fmin,
        eps=eps,
        N=N,
    )

    return output_tensor, output_scales, size_per_group_expand


if __name__ == "__main__":
    # Example usage
    dtype = torch.float8_e4m3fn
    dmax = torch.finfo(dtype).max if dtype != torch.int8 else 127
    dmin = torch.finfo(dtype).min if dtype != torch.int8 else -127
    group_original_list = []
    group_input_list = []
    group_input_scale_list = []
    num_groups = 8
    tokens_per_expert = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for i in range(num_groups):
        input_tensor = torch.randn(1000 + i * 100, 5120, device="cuda")
        group_input_list.append(input_tensor.to(torch.float32))
        tokens_per_expert[i] = input_tensor.shape[0]
    group_input_tensor = torch.cat(group_input_list, 0)
    output_tensor, output_scales, size_per_group_expand = trans_quant_expand_128x(
        group_input_tensor, tokens_per_expert, dtype=dtype
    )
    output_tensor, output_scales, size_per_group_expand = trans_quant_expand_128x(
        group_input_tensor, tokens_per_expert, dtype=dtype
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output_tensor, output_scales, size_per_group_expand = trans_quant_expand_128x(
        group_input_tensor, tokens_per_expert, dtype=dtype
    )

    # output_tensor = backend.gmm(a, b, batch_sizes, True)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")

    output_list = output_tensor.split_with_sizes(size_per_group_expand.tolist(), -1)
    input_list = group_input_tensor.split_with_sizes(tokens_per_expert.tolist(), 0)
    for i in range(num_groups):
        out_dequant = (
            output_list[i].transpose(0, 1).contiguous().float() * output_scales[:, i]
        )
        input_tensor = input_list[i].float()
        torch.testing.assert_close(
            out_dequant[: tokens_per_expert[i]], input_tensor, atol=0.3, rtol=1e-4
        )
        assert out_dequant[tokens_per_expert[i] :].sum().item() == 0
