# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


SM_MARGIN = int(os.environ.get("XTUNER_SM_MARGIN", 0))


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 6}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 10}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 14}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 18}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 18}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 22}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 22}, num_stages=3, num_warps=8),
    ]


@triton.jit
def grouped_launch(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    remian_pid = pid - group_id * width
    pid_m = group_id * group_m + (remian_pid % group_size)

    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.autotune(configs=get_cuda_autotune_config(), key=["N", "K"])
@triton.jit
def m_grouped_gemm_bKmajor_kernel(
    A,
    B,
    C,
    pad_starts,
    pad_ends,
    group_starts,
    group_ends,
    m_indices_pad,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr,
    dtype_c: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16
    """Gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m).to(tl.int32)
        pad_off = tl.load(pad_starts + group).to(tl.int32)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        offs_am = 0
        offs_bn = (pid_n * BLOCK_N).to(tl.int32)
        offs_k = 0

        a_ptr = A.to(tl.pointer_type(dtypeA))
        b_ptr = B.to(tl.pointer_type(dtypeB))
        c_ptr = C.to(tl.pointer_type(dtypeC))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[group_end, K],
            strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[(group + 1) * N, K],
            strides=[K, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[group_end, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([group_start + offs_am, offs_k])
            b = b_desc.load([group * N + offs_bn, offs_k])

            # mma
            accumulator = tl.dot(a, b.T, acc=accumulator, input_precision="tf32x3")
            offs_k += BLOCK_K

        c = accumulator.to(dtypeC)
        offs_cm = group_start
        offs_cn = (pid_n * BLOCK_N).to(tl.int32)
        c_desc.store([offs_cm, offs_cn], c)


@triton.autotune(configs=get_cuda_autotune_config(), key=["N", "K"])
@triton.jit
def m_grouped_gemm_bNmajor_kernel(
    A,
    B,
    C,
    pad_starts,
    pad_ends,
    group_starts,
    group_ends,
    m_indices_pad,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr,
    dtype_c: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16
    """Gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)

        group = tl.load(m_indices_pad + pid_m).to(tl.int32)
        pad_off = tl.load(pad_starts + group).to(tl.int32)

        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off)).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        offs_am = 0
        offs_bn = (pid_n * BLOCK_N).to(tl.int32)
        offs_k = 0
        offs_bk = 0
        a_ptr = A.to(tl.pointer_type(dtypeA))
        b_ptr = B.to(tl.pointer_type(dtypeB))
        c_ptr = C.to(tl.pointer_type(dtypeC))

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[group_end, K],
            strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[(group + 1) * K, N],
            strides=[N, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[group_end, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([group_start + offs_am, offs_k])
            b = b_desc.load([group * K + offs_bk, offs_bn])
            # mma
            accumulator = tl.dot(a, b, acc=accumulator, input_precision="tf32x3")
            offs_k += BLOCK_K
            offs_bk += BLOCK_K

        c = accumulator.to(dtypeC)
        offs_cm = group_start

        offs_cn = (pid_n * BLOCK_N).to(tl.int32)
        c_desc.store([offs_cm, offs_cn], c)


@triton.jit
def repeat_interleave_kernel(
    group_ptr,
    repeats_ptr,
    repeat_cum_ptr,
    output_ptr,
):
    pid = tl.program_id(axis=0)
    repeat = tl.load(repeats_ptr + pid)
    start = tl.load(repeat_cum_ptr + pid) - repeat
    group = tl.load(group_ptr + pid)

    for r in range(repeat):
        tl.store(output_ptr + start + r, group)


def repeat_interleave(
    group_indices: Tensor,
    repeats: Tensor,
    repeat_cum: Tensor,
    m_indices_pad: Tensor,
) -> None:
    grid = lambda args: (len(repeats),)  # noqa: E731
    repeat_interleave_kernel[grid](group_indices, repeats, repeat_cum, m_indices_pad)
    return


@torch.library.custom_op("moe::m_grouped_gemm", mutates_args=())
def m_grouped_gemm(A: Tensor, B: Tensor, size_per_group: torch.Tensor, trans_b: bool = False) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 3

    M, K = A.shape

    assert A.stride(-1) == 1, "Please make sure A is K-major"
    if trans_b:
        num_groups, N, BK = B.shape
        strideBN, strideBK = B.stride(1), B.stride(2)
    else:
        num_groups, BK, N = B.shape
        strideBK, strideBN = B.stride(1), B.stride(2)

    assert BK == K, "K of A should be equal to K of B"
    C = A.new_empty(M, N)

    BLOCK_M = 128
    m_per_group_padding = triton.cdiv(size_per_group, BLOCK_M) * BLOCK_M
    M_pad = m_per_group_padding.sum()

    repeats = (m_per_group_padding // BLOCK_M).to(torch.int32)
    m_indices_pad = torch.empty(M // BLOCK_M + num_groups, device=size_per_group.device, dtype=torch.int64)
    repeat_interleave(
        torch.arange(num_groups, device="cuda").to(torch.int32), repeats, repeats.cumsum(0), m_indices_pad
    )

    pad_start = m_per_group_padding.cumsum(0) - m_per_group_padding
    pad_end = m_per_group_padding.cumsum(0)

    group_end = size_per_group.cumsum(0) - size_per_group + size_per_group
    group_start = size_per_group.cumsum(0) - size_per_group

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count - SM_MARGIN

    dtype_mapping = {torch.bfloat16: 0, torch.float16: 1}
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)

    def grid(META):
        # assert N % META["BLOCK_N"] == 0, "Only support when N is a multiple of BLOCK_N"

        return (NUM_SMS,)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    m_grouped_gemm_kernel = m_grouped_gemm_bKmajor_kernel if trans_b else m_grouped_gemm_bNmajor_kernel

    m_grouped_gemm_kernel[grid](
        A,
        B,
        C,
        pad_start,
        pad_end,
        group_start,
        group_end,
        m_indices_pad,
        M_pad,
        M,
        N,
        K,
        dtype_a,
        dtype_b,
        dtype_c,
        strideBN,
        strideBK,
        BLOCK_M=BLOCK_M,
    )
    return C


@m_grouped_gemm.register_fake
def _(A: Tensor, B: Tensor, size_per_group: torch.Tensor, trans_b: bool = False) -> Tensor:
    M, K = A.shape
    if trans_b:
        num_groups, N, BK = B.shape
    else:
        num_groups, BK, N = B.shape
    C = A.new_empty(M, N)
    return C


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile
    from utils import generate_random_list, row_max_normalization

    def gmm(a, b, batch_sizes, trans_b=False):
        batch_sizes = batch_sizes.numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start : start + size, :] @ rhs)
            start += size
        return torch.cat(out)

    groups = 64
    z = groups
    trans_b = True
    print(f"{trans_b = }")
    batch_sizes = torch.Tensor(generate_random_list(groups, groups * 4096)).cuda().to(torch.int64)
    batch_sizes[1] = 0
    batch_sizes[10] = 0
    print(f"{batch_sizes = }")

    batch_sizes_cpu = batch_sizes.cpu()
    M = batch_sizes.sum().item()

    for n, k in ((256 + 32, 256 + 32), (768 * 2, 2048), (2048, 768), (1536 * 2, 4096), (4096 + 32, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(M, k, dtype=torch.bfloat16, device="cuda").view(-1, k).requires_grad_(True)  # type: ignore
        b = (
            torch.randn(z, n, k, dtype=torch.bfloat16, device="cuda")
            if trans_b
            else torch.randn(z, k, n, dtype=torch.bfloat16, device="cuda").requires_grad_(True)
        )
        out_ref = gmm(a, b, batch_sizes.cpu(), trans_b)
        out_cublas = out_ref.new_empty(out_ref.size())
        # out_cutlass = out_ref.new_empty(out_ref.size())

        for i in range(3):
            out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)
            # backend.gmm(a, b, out_cublas, batch_sizes_cpu, False, trans_b)
            # backend.gmm(a, b, out_cutlass, batch_sizes, False, trans_b)

        from pathlib import Path

        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent.parent
        trace_file = f"{parent_dir}/trace/gmm_triton_cublas_cutlass_N{n}_K{k}" + ".json"
        import os

        Path(os.path.join(parent_dir, "trace")).mkdir(parents=True, exist_ok=True)

        def trace_handler(prof):
            prof.export_chrome_trace(trace_file)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=1, repeat=0),
            on_trace_ready=trace_handler,
            with_modules=True,
            record_shapes=True,
        ) as prof:
            for i in range(5):
                out_triton = m_grouped_gemm(a, b, batch_sizes, trans_b)
                # with record_function(f"Cublas_record"):
                #     backend.gmm(a, b, out_cublas, batch_sizes_cpu, False, trans_b)
                # with record_function(f"Cutlass_record"):
                #     backend.gmm(a, b, out_cutlass, batch_sizes, False, trans_b)
                prof.step()
        # breakpoint()
        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        # out_cublas = row_max_normalization(out_cublas)
        # out_cutlass = row_max_normalization(out_cutlass)
        out_ref = row_max_normalization(out_ref)
        torch.cuda.empty_cache()

        torch.testing.assert_close(out_triton, out_ref, rtol=1e-02, atol=1e-02)
        # torch.testing.assert_close(out_cublas, out_ref, rtol = 5e-03, atol = 5e-03)
        # torch.testing.assert_close(out_cutlass, out_ref, rtol = 1e-02, atol = 1e-02)

        print(f"{n = }, {k = }, {M = }, {trace_file = }")

        import json

        with open(trace_file) as file:
            data = json.load(file)

        triton_time = 0
        cublas_time = 0
        cutlass_time = 0
        for event in data["traceEvents"]:
            try:
                if "m_grouped_gemm_" in event["name"]:
                    triton_time = event["dur"] / 1000
                if "Cublas_record" in event["name"] and "gpu_user_annotation" in event["cat"]:
                    cublas_time = event["dur"] / 1000
                # if "Cutlass_record" in event["name"] and "gpu_user_annotation" in event["cat"]:
                #     cutlass_time = event["dur"] / 1000
            except:  # noqa: E722
                pass
        print(
            f"    Pure kernel Elapsed time {round((triton_time), 2)} ms, {round((2 * M * n * k) / (triton_time) / 10**9, 0)} tflops"
        )
