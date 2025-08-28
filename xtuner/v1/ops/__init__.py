from .comm import all_to_all_single_autograd, ulysses_all_to_all


__all__ = ["all_to_all_single_autograd", "ulysses_all_to_all"]

from .flash_attn import flash_attn_varlen_func
from .moe import group_gemm, permute, unpermute
from .rms_norm import rms_norm
from .rotary_emb import apply_rotary_pos_emb
from .swiglu import swiglu
from .tensor_parallel import attn_column_parallel, attn_row_parallel


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
