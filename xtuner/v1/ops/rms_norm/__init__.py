from functools import partial

import torch

from .protocol import RMSNormProtocol


def native_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from torch.nn import functional as F

    return F.rms_norm(x, weight.shape, weight, epsilon)


def npu_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


def gpu_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from .gpu import rms_norm_fn

    return rms_norm_fn(x, weight, bias=None, eps=epsilon)


def get_rms_norm_fn() -> RMSNormProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return native_rms_norm
    elif device == "npu":
        return npu_rms_norm
    else:
        return gpu_rms_norm


rms_norm = get_rms_norm_fn()
