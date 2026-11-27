# Copyright (c) OpenMMLab. All rights reserved.

from typing import Literal

import torch


SUPPORTED_DEVICES = ("cpu", "cuda", "npu")
_DEVICE = None
_MEMORY_UNIT_DIVISORS: dict[Literal["MB", "GB"], float] = {
    "MB": float(1024**2),
    "GB": float(1024**3),
}


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        try:
            if torch.accelerator.is_available():
                device = torch.accelerator.current_accelerator().type
                if device == "npu":
                    from torch_npu.contrib import transfer_to_npu  # noqa
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

        if device not in SUPPORTED_DEVICES:
            raise NotImplementedError(
                "Supports only CPU, CUDA or NPU. If your accelerator is CUDA or NPU, "
                "please make sure that your environmental settings are "
                "configured correctly."
            )

        _DEVICE = device

    return _DEVICE


def get_torch_device_module():
    device = get_device()
    if device == "cuda":
        return torch.cuda
    elif device == "npu":
        return torch.npu
    elif device == "mlu":
        return torch.mlu
    elif device == "cpu":
        return torch.cpu
    else:
        raise NotImplementedError


def get_device_memory_stats(
    *,
    peak: bool = False,
    unit: Literal["MB", "GB"] = "GB",
    as_str: bool = False,
) -> dict[str, float] | str:
    """Collect PyTorch allocator and device driver (``mem_get_info``) memory
    statistics."""
    device = get_device()
    device_module = get_torch_device_module()
    divisor = _MEMORY_UNIT_DIVISORS[unit]

    stats: dict[str, float] = {}
    if peak:
        if hasattr(device_module, "max_memory_allocated"):
            stats["torch_max_allocated"] = device_module.max_memory_allocated() / divisor
        if hasattr(device_module, "max_memory_reserved"):
            stats["torch_max_reserved"] = device_module.max_memory_reserved() / divisor
    else:
        if hasattr(device_module, "memory_allocated"):
            stats["torch_allocated"] = device_module.memory_allocated() / divisor
        if hasattr(device_module, "memory_reserved"):
            stats["torch_reserved"] = device_module.memory_reserved() / divisor

    if device != "cpu" and hasattr(device_module, "mem_get_info"):
        free_bytes, total_bytes = device_module.mem_get_info()
        stats[f"{device}_used"] = (total_bytes - free_bytes) / divisor
        stats[f"{device}_free"] = free_bytes / divisor
        stats[f"{device}_total"] = total_bytes / divisor

    if not as_str:
        return stats

    return ", ".join(f"{k}: {v:.2f} {unit}" for k, v in stats.items())
