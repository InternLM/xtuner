# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.v1.utils.env_check import (
    check_torch_accelerator_available,
    check_triton_available,
    get_env_not_available_func,
)


if check_torch_accelerator_available() and check_triton_available():
    # Import Triton kernels only if torch.accelerator is available and Triton is installed
    from .k_grouped_gemm_TMA import k_grouped_gemm
    from .m_grouped_gemm_TMA import m_grouped_gemm
else:
    env_not_available_func = get_env_not_available_func(["torch.accelerator", "triton"])
    k_grouped_gemm = env_not_available_func
    m_grouped_gemm = env_not_available_func


__all__ = ["k_grouped_gemm", "m_grouped_gemm"]
