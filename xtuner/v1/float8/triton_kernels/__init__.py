# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.v1.utils.env_check import (
    check_torch_accelerator_available,
    check_triton_available,
    get_env_not_available_func,
)


if check_torch_accelerator_available() and check_triton_available():
    # Import Triton kernels only if torch.accelerator is available and Triton is installed
    from .per_block_dequant_gemm import per_block_dequant_gemm
    from .per_block_quant_gemm import per_block_quant_gemm
    from .per_tile_quant import per_tile_quant
    from .trans_per_block_quant_gemm import trans_per_block_quant_gemm
    from .trans_per_tile_quant_gemm import trans_per_tile_quant_gemm
    from .trans_quant_per_block import trans_per_block_quant_expand_128x
    from .trans_quant_per_tile import trans_per_tile_quant_expand_128x
else:
    env_not_available_func = get_env_not_available_func(["torch.accelerator", "triton"])
    per_block_dequant_gemm = env_not_available_func
    per_block_quant_gemm = env_not_available_func
    per_tile_quant = env_not_available_func
    trans_per_block_quant_gemm = env_not_available_func
    trans_per_tile_quant_gemm = env_not_available_func
    trans_per_block_quant_expand_128x = env_not_available_func
    trans_per_tile_quant_expand_128x = env_not_available_func

__all__ = [
    "trans_per_block_quant_expand_128x",
    "trans_per_tile_quant_expand_128x",
    "per_block_quant_gemm",
    "per_tile_quant",
    "trans_per_block_quant_gemm",
    "trans_per_tile_quant_gemm",
    "per_block_dequant_gemm",
]
