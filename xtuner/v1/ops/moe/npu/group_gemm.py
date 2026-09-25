from collections.abc import Sequence

import torch
from mindspeed.core.fusions.grouped_matmul import Ops


def npu_group_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    split_sizes: torch.Tensor,
    tokens_per_expert_cpu: torch.Tensor | None = None,
    replica_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    replica_grad: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    if replica_weight is not None:
        raise NotImplementedError("UltraEP grouped GEMM is CUDA-only")
    weights = weights.transpose(1, 2)

    out = Ops.gmm(x, weights, split_sizes, trans_b=False)

    return out
