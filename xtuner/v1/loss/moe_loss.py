from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh

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


class BalancingLoss(nn.Module):
    def __init__(
        self,
        balancing_loss_alpha: float,
        balancing_loss_global_average: bool,
        router_scoring_func: Literal["sigmoid", "softmax"],
    ) -> None:
        super().__init__()
        self.loss_weight = balancing_loss_alpha
        self.global_average = balancing_loss_global_average

    def forward(self, router_weights, n_routed_experts, num_experts_per_tok):
        if self.loss_weight == 0:
            return torch.tensor(0.0, device=router_weights.device, dtype=torch.float32)

        num_layers = router_weights.shape[0]
        router_weights = router_weights.float()  # (nlayers, seq, ne)
        _, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_weights.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)

        tokens_per_expert_global = tokens_per_expert.to(router_weights.dtype)  # (nlayers, ne)
        if self.global_average and dist.is_initialized():
            tokens_per_expert_global = all_reduce(tokens_per_expert_global, "sum", dist.group.WORLD)  # (nlayers, ne)
            tokens_global = tokens_per_expert_global.sum(-1)  # (nlayers, )
            seqlen_global = tokens_global // num_experts_per_tok
            routing_weights_sum_global = all_reduce_autograd(
                router_weights.sum(dim=1), "sum", dist.group.WORLD
            )  # (nlayers, )
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            scale_global = n_routed_experts / (router_weights.shape[1] * num_experts_per_tok)
            routing_weights_mean_global = router_weights.mean(dim=1)
        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        loss = loss.sum()
        # from xtuner.v1.profiler.prober import ProberList
        # ProberList.record_tensor(routing_weights_mean_global, "[balancing_loss][after]routing_weights_mean_global")
        # ProberList.record_tensor(tokens_per_expert_global, "[balancing_loss][after]tokens_per_expert_global")
        # ProberList.record_tensor(scale_global, "[balancing_loss][after]scale_global")
        return loss * self.loss_weight


def z_loss(router_logits: torch.Tensor, global_average: bool = False):
    router_logits = router_logits.float()  # (nlayers, seq, ne)
    num_seq = max(1, router_logits.shape[1])
    logsum_square = z_loss = torch.logsumexp(router_logits, dim=-1).square()
    z_loss = (logsum_square.sum(dim=-1) / num_seq).sum()

    if global_average and dist.is_initialized():
        unmasked_num = router_logits.shape[1]
        unmasked_num_rank = torch.tensor(unmasked_num, device=router_logits.device, dtype=torch.int64)
        unmasked_num_global = all_reduce(unmasked_num_rank, "sum", dist.group.WORLD)  # type: ignore
        world_size = dist.get_world_size()
        z_loss = z_loss * unmasked_num * world_size / unmasked_num_global

    return z_loss


class ZLoss(nn.Module):
    def __init__(
        self,
        z_loss_alpha: float,
        z_loss_global_average: bool,
    ) -> None:
        super().__init__()
        self.loss_weight = z_loss_alpha
        self.global_average = z_loss_global_average

    def forward(self, router_logits):
        if self.loss_weight == 0:
            return torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)
        loss = z_loss(router_logits, self.global_average)
        return loss * self.loss_weight


# ==================== New LossContext-based implementation ====================


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

    def forward(
        self,
        router_weights: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Compute balancing loss.

        Args:
            router_weights (torch.Tensor): Router weights. Shape: (num_layers, seq_len, num_experts).
            n_routed_experts (int): Number of routed experts.
            num_experts_per_tok (int): Number of experts per token.

        Returns:
            torch.Tensor: Balancing loss value.
        """
        if self.loss_cfg.balancing_loss_alpha == 0:
            return torch.tensor(0.0, device=router_weights.device, dtype=torch.float32)

        num_layers = router_weights.shape[0]
        router_weights = router_weights.float()  # (nlayers, seq, ne)
        _, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_weights.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)

        tokens_per_expert_global = tokens_per_expert.to(router_weights.dtype)  # (nlayers, ne)
        if self.loss_cfg.balancing_loss_global_average and dist.is_initialized():
            tokens_per_expert_global = all_reduce(tokens_per_expert_global, "sum", dist.group.WORLD)
            tokens_global = tokens_per_expert_global.sum(-1)  # (nlayers, )
            seqlen_global = tokens_global // num_experts_per_tok
            routing_weights_sum_global = all_reduce_autograd(router_weights.sum(dim=1), "sum", dist.group.WORLD)
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            scale_global = n_routed_experts / (router_weights.shape[1] * num_experts_per_tok)
            routing_weights_mean_global = router_weights.mean(dim=1)

        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        loss = loss.sum() * self.loss_cfg.balancing_loss_alpha

        # Normalize by batch size for proper gradient accumulation
        loss = loss / self._batch_size

        return loss

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

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute z-loss.

        Args:
            router_logits (torch.Tensor): Router logits. Shape: (num_layers, seq_len, num_experts).

        Returns:
            torch.Tensor: Z-loss value.
        """
        if self.loss_cfg.z_loss_alpha == 0:
            return torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)

        router_logits = router_logits.float()  # (nlayers, seq, ne)
        num_seq = max(1, router_logits.shape[1])
        logsum_square = torch.logsumexp(router_logits, dim=-1).square()
        loss = (logsum_square.sum(dim=-1) / num_seq).sum()

        if self.loss_cfg.z_loss_global_average and dist.is_initialized():
            unmasked_num = router_logits.shape[1]
            unmasked_num_rank = torch.tensor(unmasked_num, device=router_logits.device, dtype=torch.int64)
            unmasked_num_global = all_reduce(unmasked_num_rank, "sum", dist.group.WORLD)
            world_size = dist.get_world_size()
            loss = loss * unmasked_num * world_size / unmasked_num_global

        loss = loss * self.loss_cfg.z_loss_alpha

        # Normalize by batch size for proper gradient accumulation
        loss = loss / self._batch_size

        return loss

    @property
    def batch_size(self) -> int:
        return self._batch_size
