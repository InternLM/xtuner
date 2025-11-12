import torch
import triton
import triton.language as tl
from torch import Tensor

from .utils import TmaAutoTuneHelper


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
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


@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N"])
@triton.jit
def k_grouped_gemm_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    A,
    B,
    C,
    group_starts,
    group_ends,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr,
    dtype_c: tl.constexpr,
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
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n * num_groups

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        group = tile_id // (num_pid_m * num_pid_n)
        group_start = tl.load(group_starts + group).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        tokens = group_end - group_start

        num_pid_k = tl.cdiv(tokens, BLOCK_K)
        offs_k = group_start

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in range(0, num_pid_k - 1):
            a = tl._experimental_descriptor_load(a_desc_ptr, [offs_k, offs_am], [BLOCK_K, BLOCK_M], dtypeA)
            b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], dtypeB)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision="tf32x3")
            offs_k += BLOCK_K
        if tokens > 0:
            offs_k_final = group_start + (num_pid_k - 1) * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = A + ((offs_am + tl.arange(0, BLOCK_M))[None, :] + offs_k_final[:, None] * M)
            b_ptrs = B + ((offs_bn + tl.arange(0, BLOCK_N))[None, :] + offs_k_final[:, None] * N)
            maskA = offs_k_final[:, None] < group_end
            maskB = offs_k_final[:, None] < group_end
            a = tl.load(a_ptrs, mask=maskA, other=0.0)
            b = tl.load(b_ptrs, mask=maskB, other=0.0)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision="tf32x3")

        c = accumulator.to(dtypeC)
        off_row = offs_am + group * M
        off_col = offs_bn
        tl._experimental_descriptor_store(c_desc_ptr, c, [off_row, off_col])


@torch.library.custom_op("moe::k_grouped_gemm", mutates_args=())
def k_grouped_gemm(A: Tensor, B: Tensor, size_per_group: torch.Tensor) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 2

    K, M = A.shape
    K_, N = B.shape

    assert A.stride(-1) == 1, "Please make sure A is K-major"
    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    # assert K * A.element_size() % 128 == 0, "A and B should be 128-byte aligned"
    num_groups = size_per_group.shape[0]

    C = A.new_empty(num_groups, M, N)
    group_end = size_per_group.cumsum(0) - size_per_group + size_per_group
    group_start = size_per_group.cumsum(0) - size_per_group

    dtype_mapping = {torch.bfloat16: 0, torch.float16: 1}
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)

    assert dtype_a >= 0, f"data type {A.dtype} not supported"
    assert dtype_b >= 0, f"data type {B.dtype} not supported"
    assert dtype_c >= 0, f"data type {C.dtype} not supported"

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        assert N % META["BLOCK_N"] == 0, "Only support when N is a multiple of BLOCK_N"
        assert M % META["BLOCK_M"] == 0, "Only support when M is a multiple of BLOCK_M"
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            A.data_ptr(),
            K,
            M,
            META["BLOCK_K"],
            META["BLOCK_M"],
            A.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            B.data_ptr(),
            K,
            N,
            META["BLOCK_K"],
            META["BLOCK_N"],
            B.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            C.data_ptr(),
            C.shape[0] * C.shape[1],
            C.shape[2],
            META["BLOCK_M"],
            META["BLOCK_N"],
            C.element_size(),
        )

        return (NUM_SMS,)

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    k_grouped_gemm_kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        A,
        B,
        C,
        group_start,
        group_end,
        num_groups,
        M,
        N,
        K,
        dtype_a,
        dtype_b,
        dtype_c,
    )
    return C


@k_grouped_gemm.register_fake
def _(A: Tensor, B: Tensor, size_per_group: torch.Tensor) -> Tensor:
    K, M = A.shape
    K_, N = B.shape
    num_groups = size_per_group.shape[0]
    C = A.new_empty(num_groups, M, N)
    return C


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function
    from utils import generate_random_list, row_max_normalization

    def gmm_dw(a, b, batch_sizes):
        K, M = a.shape
        K_, N = b.shape

        assert a.stride(-1) == 1, "Please make sure A is K-major"
        assert b.stride(-1) == 1, "Please make sure B is K-major"
        assert K == K_, "Please make sure that A and B have the same seqlen"
        num_groups = batch_sizes.shape[0]

        out = a.new_empty(num_groups, M, N)

        group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
        group_start = batch_sizes.cumsum(0) - batch_sizes
        for g, (start, end) in enumerate(zip(group_start, group_end)):
            rhs = b[start:end, :]
            lhs = a[start:end, :]
            out[g] = lhs.T @ rhs
        return out.contiguous()

    groups = 128
    z = groups
    trans_b = False
    trans_a = True
    batch_sizes = torch.Tensor(generate_random_list(groups, groups * 4096)).cuda().to(torch.int64).abs()
    # batch_sizes = batch_sizes // 512 * 512
    batch_sizes_cpu = batch_sizes.cpu()
    K = batch_sizes.sum().item()

    for m, n in ((768 * 2, 2048), (2048, 768), (1536 * 2, 4096), (4096, 1536)):
        torch.cuda.empty_cache()
        a = torch.randn(K, m, dtype=torch.bfloat16, device="cuda").view(K, -1)  # type: ignore
        b = torch.randn(K, n, dtype=torch.bfloat16, device="cuda").view(K, -1)  # type: ignore
        out_ref = gmm_dw(a, b, batch_sizes.cpu())
        out_cublas = torch.empty_like(out_ref)
        out_cutlass = torch.empty_like(out_ref)

        from pathlib import Path

        script_path = Path(__file__).resolve()
        parent_dir = script_path.parent.parent
        trace_file = f"{parent_dir}/trace/gmm_dw_triton_cublas_cutlass_M{m}_N{n}" + ".json"
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
                with record_function("Triton_record"):
                    out_triton = k_grouped_gemm(a, b, batch_sizes)
                torch.cuda.synchronize()
                # with record_function(f"Cublas_record"):
                #     backend.gmm(a, b, out_cublas, batch_sizes_cpu, trans_a, trans_b, -1, False)
                # torch.cuda.synchronize()
                # with record_function(f"Cutlass_record"):
                #     backend.gmm(a, b, out_cutlass, batch_sizes, trans_a, trans_b, -1, True)
                # torch.cuda.synchronize()
                prof.step()

        # post-process, row normalization
        out_triton = row_max_normalization(out_triton)
        out_cublas = row_max_normalization(out_cublas)
        out_cutlass = row_max_normalization(out_cutlass)
        out_ref = row_max_normalization(out_ref)

        torch.testing.assert_close(out_triton, out_ref, rtol=0.001, atol=0.01)
        torch.testing.assert_close(out_cublas, out_ref, rtol=0.001, atol=0.01)
        torch.testing.assert_close(out_cutlass, out_ref, rtol=0.001, atol=0.01)

        print(f"{m = }, {n = }, {K = }")

        import json

        with open(trace_file) as file:
            data = json.load(file)

        triton_time = 0
        cublas_time = 0
        cutlass_time = 0
        for event in data["traceEvents"]:
            try:
                if "k_grouped_gemm" in event["name"]:
                    triton_time += event["dur"] / 1000
                # if "Cublas_record" in event["name"] and "gpu_user_annotation" in event["cat"]:
                #     cublas_time = event["dur"] / 1000
                # if "Cutlass_record" in event["name"] and "gpu_user_annotation" in event["cat"]:
                #     cutlass_time = event["dur"] / 1000
            except:  # noqa: E722
                pass
        print(
            f"    Pure Triton kernel Elapsed time {round((triton_time), 2)} ms, {round((2 * m * n * K) / (triton_time) / 10**9, 0)} tflops"
        )
