import torch

from .protocol import FlashAttnVarlenProtocol, cpu_flash_varlen_attn


def get_flash_attn_varlen() -> FlashAttnVarlenProtocol:
    if not torch.accelerator.is_available():
        return cpu_flash_varlen_attn

    elif torch.accelerator.current_accelerator().type == "npu":
        from .npu import npu_flash_varlen_attn

        return npu_flash_varlen_attn
    elif torch.accelerator.current_accelerator().type == "cuda":
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
