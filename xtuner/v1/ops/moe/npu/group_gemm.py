import torch


def npu_group_gemm(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    from mindspeed.core.fusions.grouped_matmul import Ops

    weights = weights.transpose(1, 2)

    out = Ops.gmm(x, weights, split_sizes, trans_b=False)

    return out
