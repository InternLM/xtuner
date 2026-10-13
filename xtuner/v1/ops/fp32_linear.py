# Copyright (c) OpenMMLab. All rights reserved.
"""FP32 projections with a token-row geometry independent of SP degree."""

import torch
from torch.nn import functional as F


def fp32_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_size: int = 0,
) -> torch.Tensor:
    """Project in FP32, optionally padding every GEMM to fixed token rows.

    A positive block size trades launches and padding for consistent forward
    GEMM shapes across token shards. It does not make gradient reductions or
    unrelated operators deterministic. Weights must already be materialized.

    Args:
        inputs: Token features with the feature dimension last.
        weight: Ordinary floating-point linear weight, not a Float8Tensor.
        bias: Optional linear bias.
        block_size: Fixed GEMM row count; zero keeps the native row count.

    Returns:
        FP32 projected features with the original token dimensions.
    """
    if block_size < 0:
        raise ValueError("block_size must be nonnegative")
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        x = inputs.float()
        w = weight.float()
        b = bias.float() if bias is not None else None
        if block_size == 0 or x.numel() == 0:
            return F.linear(x, w, b)
        rows = x.reshape(-1, x.shape[-1])
        blocks = []
        for part in rows.split(block_size, dim=0):
            padded = F.pad(part, (0, 0, 0, block_size - part.shape[0]))
            blocks.append(F.linear(padded, w, b)[: part.shape[0]])
        return torch.cat(blocks, dim=0).reshape(*x.shape[:-1], w.shape[0])
