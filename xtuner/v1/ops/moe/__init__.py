import torch

from .protocol import (
    GroupGemmProtocol,
    MoePermuteProtocol,
    MoeUnpermuteProtocol,
    cpu_group_gemm,
    cpu_permute,
    cpu_unpermute,
)


def get_group_gemm() -> GroupGemmProtocol:
    if torch.accelerator.is_available() is False:
        return cpu_group_gemm
    elif torch.accelerator.current_accelerator().type == "cuda":
        import os

        if os.getenv("XTUNER_GROUP_GEMM", "triton") == "cutlass":
            from .cuda import cutlass_group_gemm as cuda_group_gemm
        else:
            from .cuda import triton_group_gemm as cuda_group_gemm

        return cuda_group_gemm

    elif torch.accelerator.current_accelerator().type == "npu":
        from .npu import npu_group_gemm

        return npu_group_gemm
    else:
        raise NotImplementedError


def get_token_permute() -> MoePermuteProtocol:
    if torch.accelerator.is_available() is False:
        return cpu_permute

    elif torch.accelerator.current_accelerator().type == "cuda":
        from .cuda import cuda_token_permute

        return cuda_token_permute
    elif torch.accelerator.current_accelerator().type == "npu":
        from .npu import npu_token_permute

        return npu_token_permute
    else:
        raise NotImplementedError


def get_token_unpermute() -> MoeUnpermuteProtocol:
    if torch.accelerator.is_available() is False:
        return cpu_unpermute
    elif torch.accelerator.current_accelerator().type == "cuda":
        from .cuda import cuda_token_unpermute

        return cuda_token_unpermute
    if torch.accelerator.current_accelerator().type == "npu":
        from .npu import npu_token_unpermute

        return npu_token_unpermute
    else:
        raise NotImplementedError


group_gemm = get_group_gemm()
permute = get_token_permute()
unpermute = get_token_unpermute()
