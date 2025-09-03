import torch

from .protocol import (
    AttnColwiseParallelProtocol,
    AttnRowwiseParallelProtocol,
    cpu_attn_row_parallel_forward,
    cpu_column_parallel_forward,
)


def get_attn_colwise_parallel() -> AttnColwiseParallelProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_column_parallel_forward

    elif device == "cuda":
        from .cuda import attn_column_parallel_forward

        return attn_column_parallel_forward
    elif device == "npu":
        from .npu import attn_column_parallel_forward

        return attn_column_parallel_forward
    else:
        raise NotImplementedError


def get_attn_rowwise_parallel() -> AttnRowwiseParallelProtocol:
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "cpu":
        return cpu_attn_row_parallel_forward

    elif device == "cuda":
        from .cuda import attn_row_parallel_forward

        return attn_row_parallel_forward
    elif device == "npu":
        from .npu import attn_row_parallel_forward

        return attn_row_parallel_forward
    else:
        raise NotImplementedError


attn_row_parallel = get_attn_rowwise_parallel()
attn_column_parallel = get_attn_colwise_parallel()
