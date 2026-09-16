from collections.abc import Sequence
from typing import Protocol, Tuple

import torch


class GroupGemmProtocol(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        split_sizes: torch.Tensor,
        tokens_per_expert_cpu: torch.Tensor | None = None,
        replica_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
        replica_grad: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor: ...


class MoePermuteProtocol(Protocol):
    def __call__(
        self,
        input_act: torch.Tensor,
        indices: torch.Tensor,
        num_topK: int | None = None,
        num_out_tokens: int | None = None,
        num_negative_one_in_indices: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


class MoeUnpermuteProtocol(Protocol):
    def __call__(
        self, input_act: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor | None = None
    ) -> torch.Tensor: ...


def cpu_group_gemm(
    x: torch.Tensor,
    weights: torch.Tensor,
    split_sizes: torch.Tensor,
    tokens_per_expert_cpu: torch.Tensor | None = None,
    replica_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    replica_grad: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    if replica_weight is not None:
        raise NotImplementedError("UltraEP grouped GEMM is CUDA-only")
    raise NotImplementedError("CPU GroupGemm is not implemented yet.")


def cpu_permute(
    input_act: torch.Tensor,
    indices: torch.Tensor,
    num_topK: int | None = None,
    num_out_tokens: int | None = None,
    num_negative_one_in_indices: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("CPU Permute is not implemented yet.")


def cpu_unpermute(
    input_act: torch.Tensor, row_id_map: torch.Tensor, probs: torch.Tensor | None = None
) -> torch.Tensor:
    raise NotImplementedError("CPU Unpermute is not implemented yet.")
