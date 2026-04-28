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
        self.local_load_list: list[torch.Tensor] = []
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

    def forward(
        self,
        router_weights: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Compute balancing loss.

        TODO: `Qwen3VLTextMoE._forward` in xtuner/v1/model/moe/qwen3vl_text.py still uses
        this legacy forward path. This method and that usage are planned to be removed.

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
            tokens_per_expert_global = all_reduce(tokens_per_expert_global, "sum", dist.group.WORLD)  # type: ignore
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

    def accumulate(
        self,
        *,
        router_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> None:
        """Update split-aux balancing accumulators for one layer.

        Args:
            router_weights (torch.Tensor): Router weights with non-padding tokens selected.
            tokens_per_expert (torch.Tensor): Per-layer expert token counts.
        """
        # TODO: Consider merging `local_load_list` with the aux-loss-side
        # `_local_load_logits_list` so tokens-per-expert statistics are stored only
        # once and reused by both finalize paths.
        self.local_load_list.append(tokens_per_expert)
        self.routing_weights_sum_list.append(router_weights.sum(dim=0))

    def finalize(
        self,
        *,
        n_routed_experts: int,
        num_experts_per_tok: int,
        non_pad_token: int,
    ) -> torch.Tensor:
        """Finalize balancing loss from split-aux accumulators.

        Args:
            local_load (torch.Tensor): Per-layer expert token counts accumulator.
            routing_weights_sum_list (list[torch.Tensor]): Per-layer router-weight sums.
            active_layers (int): Number of active MoE layers observed in this forward.
            n_routed_experts (int): Number of routed experts.
            num_experts_per_tok (int): Number of experts selected per token.
            non_pad_token (int): Number of non-padding tokens.

        Returns:
            torch.Tensor: Final balancing loss.
        """
        local_load = self.local_load_list
        routing_weights_sum_list = self.routing_weights_sum_list
        self.local_load_list = []
        self.routing_weights_sum_list = []
        if self.loss_cfg.balancing_loss_alpha == 0:
            device = local_load[0].device if local_load else DEVICE
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        if not local_load or not routing_weights_sum_list:
            device = routing_weights_sum_list[0].device if routing_weights_sum_list else DEVICE
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        local_gating_sum = torch.stack(routing_weights_sum_list, dim=0)
        active_local_load = torch.stack(local_load, dim=0)

        if self.loss_cfg.balancing_loss_global_average and dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            tokens_per_expert_global = all_reduce(active_local_load, "sum", group)
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok

            routing_weights_sum_global = all_reduce_autograd(local_gating_sum, "sum", group)
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            tokens_per_expert_global = active_local_load
            valid_tokens = max(non_pad_token, 1)
            scale_global = n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean_global = local_gating_sum / valid_tokens

        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
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

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute z-loss.

        TODO: `Qwen3VLTextMoE._forward` in xtuner/v1/model/moe/qwen3vl_text.py still uses
        this legacy forward path. This method and that usage are planned to be removed.

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
            group = dist.group.WORLD
            assert group is not None
            unmasked_num_global = all_reduce(unmasked_num_rank, "sum", group)
            world_size = dist.get_world_size()
            loss = loss * unmasked_num * world_size / unmasked_num_global

        loss = loss * self.loss_cfg.z_loss_alpha

        # Normalize by batch size for proper gradient accumulation
        loss = loss / self._batch_size

        return loss

    def accumulate(
        self,
        *,
        router_logits: torch.Tensor,
    ) -> None:
        """Update split-aux z-loss accumulators for one layer.

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
