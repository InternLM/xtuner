import os

import torch

from .protocol import RMSNormProtocol


def native_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from torch.nn import functional as F

    return F.rms_norm(x, weight.shape, weight, epsilon)


def npu_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    from .gpu import rms_norm_fn

    return rms_norm_fn(x, weight, bias=None, eps=epsilon)

def native_zero_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    # TODO: is native_rms_norm ?
    def _norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
    output = _norm(x.float())
    # Llama does x.to(float16) * w whilst Qwen3_5Moe is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def get_rms_norm_fn() -> RMSNormProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device in ["cpu", "cuda"]:
        # TODO: control triton rmsnorm by model config rather than env var
        if os.getenv("XTUNER_USE_NATIVE_RMSNORM", "1") == "0" and device == "cuda":
            return _triton_rms_norm
        else:
            return native_rms_norm
    elif device == "npu":
        return npu_rms_norm
    else:
        raise NotImplementedError(f"RMSNorm is not implemented on {device}")

def get_zero_centered_rms_norm_fn() -> RMSNormProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device in ["cpu", "cuda"]:
        # TODO: control triton rmsnorm by model config rather than env var
        if os.getenv("XTUNER_USE_NATIVE_RMSNORM", "1") == "0" and device == "cuda":
            raise NotImplementedError("Zero-centered RMSNorm is not implemented in triton")
        else:
            return native_zero_centered_rms_norm
    elif device == "npu":
        raise NotImplementedError("Zero-centered RMSNorm is not implemented on NPU")
    else:
        raise NotImplementedError(f"RMSNorm is not implemented on {device}")


rms_norm = get_rms_norm_fn()
zero_centered_rms_norm = get_zero_centered_rms_norm_fn()