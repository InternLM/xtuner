import torch

from .protocol import FlashAttnVarlenProtocol, cpu_flash_varlen_attn


def get_flash_attn_varlen() -> FlashAttnVarlenProtocol:
    from xtuner.v1.utils.device import get_device

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
            from flash_attn_interface import flash_attn_varlen_func
        else:
            from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    else:
        raise RuntimeError


flash_attn_varlen_func = get_flash_attn_varlen()
