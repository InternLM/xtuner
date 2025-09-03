# Copyright (c) OpenMMLab. All rights reserved.

import torch


SUPPORTED_DEVICES = ("cpu", "cuda", "npu")
_DEVICE = None


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
