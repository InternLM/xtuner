import torch

from .protocol import (
    AttnColwiseParallelProtocol,
    AttnRowwiseParallelProtocol,
    cpu_attn_row_parallel_forward,
    cpu_column_parallel_forward,
)


def get_attn_colwise_parallel() -> AttnColwiseParallelProtocol:
    if torch.accelerator.is_available() is False:
        return cpu_column_parallel_forward

    elif torch.accelerator.current_accelerator().type == "cuda":
        from .cuda import attn_column_parallel_forward

        return attn_column_parallel_forward
    elif torch.accelerator.current_accelerator().type == "npu":
        from .npu import attn_column_parallel_forward

        return attn_column_parallel_forward
    else:
        raise NotImplementedError


def get_attn_rowwise_parallel() -> AttnRowwiseParallelProtocol:
    if torch.accelerator.is_available() is False:
        return cpu_attn_row_parallel_forward

    elif torch.accelerator.current_accelerator().type == "cuda":
        from .cuda import attn_row_parallel_forward

        return attn_row_parallel_forward
    if torch.accelerator.current_accelerator().type == "npu":
        from .npu import attn_row_parallel_forward

        return attn_row_parallel_forward
    else:
        raise NotImplementedError


attn_row_parallel = get_attn_rowwise_parallel()
attn_column_parallel = get_attn_colwise_parallel()
