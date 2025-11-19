import torch

from .protocol import FlashAttnVarlenProtocol, cpu_flash_varlen_attn


def get_flash_attn_varlen() -> FlashAttnVarlenProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_flash_varlen_attn

    elif device == "npu":
        from .npu import npu_flash_varlen_attn

        return npu_flash_varlen_attn
    elif device == "cuda":
        # TODO: #  Uniofiy "1" and "true"
        # For the optional feature, we should collect all information to logs
        import os

        if os.environ.get("XTUNER_USE_FA3", "0") == "1":
            from .gpu import gpu_flash_varlen_attn_v3 as flash_attn_varlen_func
        else:
            from .gpu import flash_attn_varlen_func_v2 as flash_attn_varlen_func  # type: ignore

        return flash_attn_varlen_func
    else:
        raise RuntimeError


# TODO: Enhance the optional requirement for flash attention
try:
    flash_attn_varlen_func = get_flash_attn_varlen()
except ImportError:
    flash_attn_varlen_func = None  # type: ignore[assignment]
