from .comm import all_to_all_single_autograd, ulysses_all_to_all


__all__ = ["all_to_all_single_autograd", "ulysses_all_to_all"]

try:
    from .moe_permute import permute, unpermute
except ImportError:
    ...
else:
    __all__.extend(["permute", "unpermute"])


def __getattr__(name: str):
    if name in ["permute", "unpermute"]:
        # TODO: (yehaochen) replace install url
        raise ImportError(
            f"{name} is not available, please install `grouped_gemm` from https://github.com/fanshiqing/grouped_gemm"
        )
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
