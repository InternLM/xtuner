# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 3},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 16},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=5,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 3},
            num_stages=5,
            num_warps=8,
        ),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, "GROUP_M": 8}, num_stages=7,
        #               num_warps=8,),
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


@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "N"])
@triton.jit
def gemm_kernel_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    a_scale,
    b_scale,
    tokens_per_expert,
    tokens_off,
    num_groups,
    M,
    N,
    K,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n * num_groups

    dtypeA = tl.float8e4nv if dtype_a == 1 else tl.float8e5
    dtypeB = tl.float8e4nv if dtype_b == 1 else tl.float8e5

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        g = tile_id // (num_pid_m * num_pid_n)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        offs_k = tl.load(tokens_off + g)

        offs_as = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) + g * M
        offs_bs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) + g * N
        as_ptrs = a_scale + offs_as[:, None]
        bs_ptrs = b_scale + offs_bs[:, None]

        tokens = tl.load(tokens_per_expert + g)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_pid_k = tl.cdiv(tokens, BLOCK_K)

        a_s = tl.load(as_ptrs)
        b_s = tl.load(bs_ptrs)

        for kk in range(0, num_pid_k):
            a = tl._experimental_descriptor_load(
                a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtypeA
            )
            b = tl._experimental_descriptor_load(
                b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtypeB
            )
            accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
            offs_k += BLOCK_K
        accumulator *= a_s * b_s.T
        accumulator = accumulator.to(tl.bfloat16)
        tl._experimental_descriptor_store(
            c_desc_ptr, accumulator, [offs_am + g * M, offs_bn]
        )


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)


# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 512

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


class KernelParamWrapper:
    def __init__(self, desc):
        self.desc = desc

    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


DTYPE_MAPPING = {
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 2,
}


def gmm_dw_fp8_act_per_channel_w_per_expert(
    x_fp8: torch.Tensor,  # (M, K)
    x_scale: torch.Tensor,  # (M, ne)
    y_fp8: torch.Tensor,  # (N, K)
    y_scale: torch.Tensor,  # (N, ne)
    tokens_per_expert: torch.Tensor,  # (ne, )
) -> torch.Tensor:
    desc_helper = TmaAutoTuneHelper()

    dtype_a = DTYPE_MAPPING[x_fp8.dtype]
    dtype_b = DTYPE_MAPPING[y_fp8.dtype]
    M, K = x_fp8.shape
    N, K_ = y_fp8.shape
    assert K == K_
    num_groups = tokens_per_expert.shape[0]
    output = torch.empty((num_groups, M, N), dtype=torch.bfloat16, device="cuda")
    token_off = tokens_per_expert.cumsum(0) - tokens_per_expert
    token_off = token_off.int()

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count - 24

    def grid(META):
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            x_fp8.data_ptr(),
            M,
            K,
            META["BLOCK_M"],
            META["BLOCK_K"],
            x_fp8.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            y_fp8.data_ptr(),
            N,
            K,
            META["BLOCK_N"],
            META["BLOCK_K"],
            y_fp8.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "c",
            output.data_ptr(),
            num_groups * M,
            N,
            META["BLOCK_M"],
            META["BLOCK_N"],
            output.element_size(),
        )
        # return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        return (NUM_SMS,)

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
    gemm_kernel_tma[grid](
        desc_a,
        desc_b,
        desc_c,
        x_scale,
        y_scale,
        tokens_per_expert,
        token_off,
        num_groups,
        M,
        N,
        K,
        dtype_a,
        dtype_b,
        NUM_SMS=NUM_SMS,
    )

    # print(f"best config {gemm_kernel_tma.best_config}", flush = True)
    return output
