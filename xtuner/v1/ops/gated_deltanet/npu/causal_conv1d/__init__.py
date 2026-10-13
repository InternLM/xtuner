from .causal_conv1d_triton_ascend import get_num_cores
from .interface import CausalConv1dFunction, causal_conv1d_triton, causal_conv1d_triton_native


__all__ = ["CausalConv1dFunction", "causal_conv1d_triton", "causal_conv1d_triton_native", "get_num_cores"]
