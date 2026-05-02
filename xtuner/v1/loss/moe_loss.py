from typing import Annotated, Literal

import torch
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        tensor = all_reduce(tensor, op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)


def all_reduce_autograd(tensor, op, group):
    return _AllReduce.apply(op, group, tensor)


class BalancingLossConfig(BaseModel):
    """Balancing loss configuration for MoE models.

    Args:
        balancing_loss_alpha (float): Weight for the balancing loss. Defaults to 0.001.
        balancing_loss_global_average (bool): Whether to perform global averaging across all ranks.
            Defaults to True.
        router_scoring_func (str): Router scoring function type. Options are "sigmoid" and "softmax".
            Defaults to "softmax".
    """

    model_config = ConfigDict(extra="forbid")
    balancing_loss_alpha: Annotated[float, Parameter(help="weight for balancing loss")] = 0.001
    balancing_loss_global_average: Annotated[bool, Parameter(help="global average for balancing loss")] = True
    router_scoring_func: Annotated[Literal["sigmoid", "softmax"], Parameter(help="router scoring function")] = (
        "softmax"
    )

    def build(self) -> "BalancingLossContext":
        """Build BalancingLossContext.

        Returns:
            BalancingLossContext: Built loss context.
        """
        loss_kwargs = BalancingLossKwargs()
        return BalancingLossContext(self, loss_kwargs)


class BalancingLossKwargs(BaseModel):
    """Keyword arguments for balancing loss computation.

    This class is empty as all parameters are passed to forward().
    """

    model_config = ConfigDict(title="balancing loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)


class BalancingLossContext(nn.Module):
    """Balancing loss context for MoE models.

    Args:
        loss_cfg (BalancingLossConfig): The configuration for the balancing loss.
        loss_kwargs (BalancingLossKwargs): The keyword arguments for the balancing loss.
    """

    def __init__(self, loss_cfg: BalancingLossConfig, loss_kwargs: BalancingLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        self._batch_size = 1
        # Per-layer differentiable accumulator. tokens_per_expert is owned by AuxLossContext
        # and passed in at finalize() time to avoid duplicate storage / duplicate all_reduce.
        self.routing_weights_sum_list: list[torch.Tensor] = []

    @staticmethod
    def build_batches(
        loss_ctx_list: list["BalancingLossContext"],
    ) -> list["BalancingLossContext"]:
        """Build batches for balancing loss contexts.

        For balancing loss, we set the batch size for proper gradient accumulation.

        Args:
            loss_ctx_list (list[BalancingLossContext]): List of loss contexts.

        Returns:
            list[BalancingLossContext]: The same list with batch_size set.
        """
        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
        return loss_ctx_list

    def accumulate(
        self,
        *,
        router_weights: torch.Tensor,
    ) -> None:
        """Update the per-layer differentiable accumulator for balancing loss.

        Args:
            router_weights (torch.Tensor): Router weights with non-padding tokens already selected.
                Shape: ``(non_pad, n_routed_experts)``.
        """
        # router_weights.sum(dim=0) is [n_routed_experts]; sum's backward does not save the input
        # tensor, so the [non_pad, n_routed_experts] activation is not pinned by this accumulator.
        self.routing_weights_sum_list.append(router_weights.sum(dim=0))

    def finalize(
        self,
        *,
        tokens_per_expert_local: torch.Tensor,
        tokens_per_expert_global: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
        non_pad_token: int,
    ) -> torch.Tensor:
        """Finalize balancing loss from accumulators.

        Args:
            tokens_per_expert_local (torch.Tensor): Per-layer expert token counts on this rank,
                ``(num_layers, n_routed_experts)``. Used by the non-global-average branch.
            tokens_per_expert_global (torch.Tensor): All-reduced ``tokens_per_expert``,
                ``(num_layers, n_routed_experts)``. Used by the global-average branch.
            n_routed_experts (int): Number of routed experts.
            num_experts_per_tok (int): Number of experts selected per token.
            non_pad_token (int): Number of non-padding tokens on this rank.

        Returns:
            torch.Tensor: Final balancing loss.
        """
        routing_weights_sum_list = self.routing_weights_sum_list
        self.routing_weights_sum_list = []
        if self.loss_cfg.balancing_loss_alpha == 0 or not routing_weights_sum_list:
            return torch.tensor(0.0, device=tokens_per_expert_local.device, dtype=torch.float32)

        local_gating_sum = torch.stack(routing_weights_sum_list, dim=0)

        if self.loss_cfg.balancing_loss_global_average and dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok

            routing_weights_sum_global = all_reduce_autograd(local_gating_sum, "sum", group)
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
            tokens_per_expert_for_loss = tokens_per_expert_global
        else:
            valid_tokens = max(non_pad_token, 1)
            scale_global = n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean_global = local_gating_sum / valid_tokens
            tokens_per_expert_for_loss = tokens_per_expert_local

        loss = scale_global * (tokens_per_expert_for_loss * routing_weights_mean_global).sum(-1)
        loss = loss.sum() * self.loss_cfg.balancing_loss_alpha
        return loss / self._batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size


class ZLossConfig(BaseModel):
    """Z-loss configuration for MoE models.

    Args:
        z_loss_alpha (float): Weight for the z-loss. Defaults to 0.001.
        z_loss_global_average (bool): Whether to perform global averaging across all ranks.
            Defaults to True.
    """

    model_config = ConfigDict(extra="forbid")
    z_loss_alpha: Annotated[float, Parameter(help="weight for z-loss")] = 0.001
    z_loss_global_average: Annotated[bool, Parameter(help="global average for z-loss")] = True

    def build(self) -> "ZLossContext":
        """Build ZLossContext.

        Returns:
            ZLossContext: Built loss context.
        """
        loss_kwargs = ZLossKwargs()
        return ZLossContext(self, loss_kwargs)


class ZLossKwargs(BaseModel):
    """Keyword arguments for z-loss computation."""

    model_config = ConfigDict(title="z-loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)


class ZLossContext(nn.Module):
    """Z-loss context for MoE models.

    Args:
        loss_cfg (ZLossConfig): The configuration for the z-loss.
        loss_kwargs (ZLossKwargs): The keyword arguments for the z-loss.
    """

    def __init__(self, loss_cfg: ZLossConfig, loss_kwargs: ZLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        self._batch_size = 1
        self.z_loss_logsum_list: list[torch.Tensor] = []
        self.z_loss_token_count_list: list[torch.Tensor] = []

    @staticmethod
    def build_batches(
        loss_ctx_list: list["ZLossContext"],
    ) -> list["ZLossContext"]:
        """Build batches for z-loss contexts.

        For z-loss, we set the batch size for proper gradient accumulation.

        Args:
            loss_ctx_list (list[ZLossContext]): List of loss contexts.

        Returns:
            list[ZLossContext]: The same list with batch_size set.
        """
        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
        return loss_ctx_list

    def accumulate(
        self,
        *,
        router_logits: torch.Tensor,
    ) -> None:
        """Update z-loss accumulators for one layer.

        Args:
            router_logits (torch.Tensor): Router logits with non-padding tokens selected.
        """
        # TODO: z-loss currently keeps autograd dependency on router_logits through logsumexp,
        # which may retain extra graph state and increase activation memory. Unlike balancing-loss
        # path (where sequence-wise reductions are mostly cheap to keep), this path may benefit
        # from a memory-optimized implementation (e.g., custom autograd with recomputation/chunking).
        self.z_loss_token_count_list.append(torch.tensor(router_logits.shape[0], device=router_logits.device))
        self.z_loss_logsum_list.append(torch.logsumexp(router_logits, dim=-1).square().sum())

    def finalize(self) -> torch.Tensor:
        """Finalize z-loss from split-aux accumulators.

        Returns:
            torch.Tensor: Final z-loss.
        """
        z_loss_logsum = self.z_loss_logsum_list
        z_loss_token_count = self.z_loss_token_count_list
        self.z_loss_logsum_list = []
        self.z_loss_token_count_list = []
        if self.loss_cfg.z_loss_alpha == 0:
            device = z_loss_logsum[0].device if z_loss_logsum else DEVICE
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        if not z_loss_logsum:
            return torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

        active_token_count = torch.stack(z_loss_token_count, dim=0)
        active_logsum = torch.stack(z_loss_logsum, dim=0)
        token_count = torch.clamp(active_token_count, min=1)
        loss = active_logsum / token_count.to(active_logsum.dtype)

        if self.loss_cfg.z_loss_global_average and dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            token_count_global = all_reduce(active_token_count, "sum", group)
            token_count_global = torch.clamp(token_count_global, min=1)
            world_size = dist.get_world_size()
            loss = loss * active_token_count.to(active_logsum.dtype) * world_size / token_count_global

        loss = loss.sum() * self.loss_cfg.z_loss_alpha
        return loss / self._batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size
