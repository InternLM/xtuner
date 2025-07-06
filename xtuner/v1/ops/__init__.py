from .comm import all_to_all_single_autograd, ulysses_all_to_all


__all__ = ["all_to_all_single_autograd", "ulysses_all_to_all"]

try:
    from .moe_gemm import moe_grouped_gemm as grouped_gemm
    from .moe_permute import permute, unpermute
except ImportError:
    ...
else:
    __all__.extend(["permute", "unpermute", "grouped_gemm"])


try:
    from .comm.deepep_op import (
        buffer_capture,
        deep_ep_combine,
        deep_ep_dispatch,
        get_deepep_buffer,
        get_low_latency_buffer,
    )
except ImportError:
    ...
else:
    __all__.extend(
        ["deep_ep_combine", "deep_ep_dispatch", "get_deepep_buffer", "get_low_latency_buffer", "buffer_capture"]
    )


def __getattr__(name: str):
    if name in ["permute", "unpermute", "grouped_gemm"]:
        # TODO: (yehaochen) replace install url
        raise ImportError(
            f"{name} is not available, please install `grouped_gemm` from https://github.com/fanshiqing/grouped_gemm"
        )
    elif name in [
        "deep_ep_dispatch",
        "deep_ep_combine",
        "get_low_latency_buffer",
        "get_deepep_buffer",
        "buffer_capture",
    ]:
        raise ImportError(
            f"{name} is not available, please install `deep_ep` from https://github.com/deepseek-ai/DeepEP"
        )
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
