# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


def get_cuda_autotune_config():
    return [
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_N": 256,
                "BLOCK_K": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_N": 128,
                "BLOCK_K": 128,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_N": 64,
                "BLOCK_K": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 256,
                "BLOCK_K": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 128,
                "BLOCK_K": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 64,
                "BLOCK_K": 64,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 128,
                "BLOCK_K": 64,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_N": 32,
                "BLOCK_K": 64,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.jit
def grouped_launch(
    pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr
):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit
def get_group_id(m, group_offsets, g_start, num_groups):
    id = 0
    off_out = 0
    offnxt_out = 0
    for group_id in tl.range(g_start, num_groups):
        group_off = tl.load(group_offsets + group_id)
        group_off_nxt = tl.load(group_offsets + group_id + 1)
        if m >= group_off and m < group_off_nxt:
            id = group_id
            off_out = group_off
            offnxt_out = group_off_nxt
    return id, off_out, offnxt_out


@triton.autotune(configs=get_cuda_autotune_config(), key=["N", "K"])
@triton.jit
def gmm_fp8_act_per_channel_w_per_expert_kernel(
    A,
    a_scale_ptr,
    B,
    b_scale_ptr,
    C,
    group_pad_offs,
    token_cumdiffs,
    token_pad_ends,
    num_groups,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        # pid_m = tile_id // num_pid_n
        # pid_n = tile_id % num_pid_n

        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group, group_pad_off, group_pad_off_nxt = get_group_id(
            pid_m * BLOCK_M, group_pad_offs, 0, num_groups
        )
        token_cumdiff = tl.load(token_cumdiffs + group)
        token_pad_end = tl.load(token_pad_ends + group)

        offs_am = pid_m * BLOCK_M - token_cumdiff + tl.arange(0, BLOCK_M)
        offs_bn = group * N + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = B + ((offs_bn)[:, None] * K + offs_k[None, :])

        as_ptrs = a_scale_ptr + offs_am
        bs_ptrs = b_scale_ptr + offs_bn
        as_mask = offs_am < token_pad_end
        # bs_mask = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) < N
        a_scale = tl.load(as_ptrs, mask=as_mask, other=1.0)
        b_scale = tl.load(bs_ptrs)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            # hint to Triton compiler to do proper loop pipelining
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])
            # load ab
            a = tl.load(
                a_ptrs,
                mask=(offs_k[None, :] < K - k * BLOCK_K)
                & (offs_am[:, None] < token_pad_end),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            # mma
            accumulator = tl.dot(a, b.T, acc=accumulator)
            # update scales and ratio
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        c = accumulator * a_scale[:, None] * b_scale
        offs_cm = pid_m * BLOCK_M - token_cumdiff + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < token_pad_end) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton_op("myfp8::gmm_fp8_act_per_channel_w_per_expert", mutates_args={})
def gmm_fp8_act_per_channel_w_per_expert(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: torch.Tensor,
    size_per_group: torch.Tensor,
    dtype_out: torch.dtype = torch.float16,
) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape
    num_groups, N, BK = B.shape
    assert BK == K, f"{BK} {K}"

    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert A.stride(-1) == 1, "Please make sure A is K-major"

    group_pad_off = torch.zeros(
        size_per_group.shape[0] + 1, device="cuda", dtype=torch.int32
    )

    BLOCK_M = 128
    # size_per_group_padding = ((size_per_group + BLOCK_M -1 ) / BLOCK_M).int() * BLOCK_M
    size_per_group_padding = triton.cdiv(size_per_group, BLOCK_M) * BLOCK_M
    group_pad_off[1:] = size_per_group_padding.cumsum(0)

    C = A.new_empty(M, N, dtype=dtype_out)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count - 24

    def grid(META):
        assert N % META["BLOCK_N"] == 0, "Only support when N is a multiple of BLOCK_N"

        return (NUM_SMS,)

    # M_pad = int(sum(size_per_group_padding))
    M_pad = size_per_group_padding.sum()

    token_diff = size_per_group_padding - size_per_group
    token_cumdiff = token_diff.cumsum(0)
    token_pad_end = size_per_group_padding.cumsum(0) - token_cumdiff

    token_cumdiff = token_diff.cumsum(0) - token_diff

    wrap_triton(gmm_fp8_act_per_channel_w_per_expert_kernel)[grid](
        A,
        A_scale,
        B,
        B_scale,
        C,
        group_pad_off,
        token_cumdiff,
        token_pad_end,
        num_groups,
        M_pad,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        GROUP_M=8,
    )

    return C


if __name__ == "__main__":
    from typing import Tuple

    def ceil_div(x: int, y: int) -> int:
        """Perform ceiling division of two integers.

        Args:
            x: the dividend.
            y: the divisor.

        Returns:
            The result of the ceiling division.
        """
        return (x + y - 1) // y

    def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
            m, n
        ), (x_amax / 448.0).view(m, -1)

    def per_channel_cast_to_fp8(
        x: torch.Tensor, dtype=torch.float8_e4m3fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, n)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        if dtype == torch.float8_e4m3fn:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
                m, n
            ), (x_amax / fmax).view(m, -1)
        else:
            fmax = torch.finfo(torch.float8_e5m2).max
            return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(
                m, n
            ), (x_amax / fmax).view(m, -1)

    def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros(
            (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
            dtype=x.dtype,
            device=x.device,
        )
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
            x_view.size(0), x_view.size(2)
        )

    def per_expert_cast_to_fp8(
        x: torch.Tensor, dtype=torch.float8_e4m3fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 3
        num_groups, m, n = x.shape
        x_padded = torch.zeros(
            (num_groups, ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
            dtype=x.dtype,
            device=x.device,
        )
        x_padded[:, :m, :n] = x
        x_view = x_padded.view(num_groups, m, 1, n)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        if dtype == torch.float8_e4m3fn:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e4m3fn)
            return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (
                x_amax / fmax
            ).view(x_view.size(0), x_view.size(2))
        else:
            fmax = torch.finfo(torch.float8_e5m2).max
            x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e5m2)
            return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (
                x_amax / fmax
            ).view(x_view.size(0), x_view.size(2))

    def gen_data_fwd(
        M,
        N,
        K,
        tokens_per_expert,
        dtype_out=torch.bfloat16,
        dtype_a=torch.float8_e4m3fn,
        dtype_b=torch.float8_e4m3fn,
    ):
        ref_dw = torch.empty(M, N, device="cuda", dtype=dtype_out)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        num_groups = len(tokens_per_expert)

        weights = torch.randn(num_groups, N, K, device="cuda", dtype=torch.bfloat16)

        x_fp8, x_scale = per_channel_cast_to_fp8(x, dtype_a)
        weights_fp8, weights_scale = per_expert_cast_to_fp8(weights, dtype_b)

        # prepare data
        t_start = 0
        for i, tokens in enumerate(tokens_per_expert):
            tokens = int(tokens)
            x_tmp = x[t_start : t_start + tokens]
            weight = weights[i]

            ref_dw[t_start : t_start + tokens] = x_tmp @ weight.T

            t_start += tokens

        # breakpoint()
        return x_fp8, x_scale, weights_fp8, weights_scale, ref_dw

    # Example usage
    # dtype_a = torch.float8_e4m3fn
    # dtype_b = torch.float8_e4m3fn

    dtype_a = torch.float8_e5m2
    dtype_b = torch.float8_e4m3fn

    dtype_out = torch.bfloat16

    # from helper import gen_data_fwd

    bias = 3
    # tokens_per_expert = [2047, 2048] * 1
    tokens_per_expert = [
        512 - bias,
        2 * 2048 - 512 + bias,
        128 - bias,
        2 * 2048 - 128 + bias,
    ] * 8
    # tokens_per_expert  = [1] * 31 + [65536 - 31]
    num_groups = len(tokens_per_expert)

    M = sum(tokens_per_expert)
    N = 6144
    K = 5120

    # N = 5120
    # K = 3072

    x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd = gen_data_fwd(
        M,
        N,
        K,
        tokens_per_expert,
        dtype_out=dtype_out,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
    )
    size_per_group = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int)
    weights_scale = weights_scale.view(-1, 1).repeat(1, N)

    for i in range(3):
        output_tensor = gmm_fp8_act_per_channel_w_per_expert(
            x_fp8,
            x_scale,
            weights_fp8,
            weights_scale,
            size_per_group,
            dtype_out=dtype_out,
        )
        # breakpoint()

    amax = max(output_tensor.abs().max(), ref_fwd.abs().max())
    adiffmax = (output_tensor - ref_fwd).abs().max()
    rdiffmax = adiffmax / amax
    print(f"max relative difference of the layer is {rdiffmax}")

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # with_stack = True,
        # with_modules = True,
        record_shapes=True,
    ) as prof:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output_tensor = gmm_fp8_act_per_channel_w_per_expert(
            x_fp8,
            x_scale,
            weights_fp8,
            weights_scale,
            size_per_group,
            dtype_out=dtype_out,
        )

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    trace = "../trace/fwd_groupedK_gemm2.json"
    prof.export_chrome_trace(trace)
    # Get time from trace
    import json

    with open(trace) as file:
        data = json.load(file)

    kernel_time = 0
    for event in data["traceEvents"]:
        if "gmm_fp8_act_per_channel_w_per_expert_kernel" in event["name"]:
            kernel_time += event["dur"] / 1000
    print(
        f"\nPure kernel Elapsed time {round((kernel_time), 1)} ms, "
        f"{round((2 * M * N * K)/(kernel_time)/10**9, 0)} tflops"
    )
