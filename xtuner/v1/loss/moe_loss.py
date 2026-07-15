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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            tuple[torch.Tensor, torch.Tensor]: ``(balancing_loss, local_balancing_loss)``. The first
            carries the autograd graph used for backward; in the global-average branch it is reduced
            across ranks via ``all_reduce_autograd``. The second is a detached, per-rank local
            component for logging whose cross-rank SUM reproduces the first (used by the display
            pipeline so that logged curves stay global once backward switches to the local component).
        """
        routing_weights_sum_list = self.routing_weights_sum_list
        self.routing_weights_sum_list = []
        if self.loss_cfg.balancing_loss_alpha == 0 or not routing_weights_sum_list:
            zero = torch.tensor(0.0, device=tokens_per_expert_local.device, dtype=torch.float32)
            return zero, zero

        local_gating_sum = torch.stack(routing_weights_sum_list, dim=0)
        alpha = self.loss_cfg.balancing_loss_alpha

        if self.loss_cfg.balancing_loss_global_average and dist.is_initialized():
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok
            scale_global = n_routed_experts / tokens_global

            # Under reduce-sum gradients the loss stays as this rank's local component: use this
            # rank's own gating sum with the global denominators, without any cross-rank all_reduce.
            # Cross-rank aggregation happens on the gradients (FSDP / scale_and_reduce_grad SUM);
            # since every other factor here is detached and global, summing the loss over ranks
            # reproduces the global balancing loss.
            routing_weights_mean = local_gating_sum / seqlen_global.unsqueeze(-1)
            loss_vec = scale_global * (tokens_per_expert_global * routing_weights_mean).sum(-1)
            local_vec = loss_vec.detach()
        else:
            valid_tokens = max(non_pad_token, 1)
            scale_global = n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean_global = local_gating_sum / valid_tokens
            loss_vec = scale_global * (tokens_per_expert_local * routing_weights_mean_global).sum(-1)
            # No cross-rank term in this branch; the loss is already a per-rank quantity.
            local_vec = loss_vec.detach()

        loss = loss_vec.sum() * alpha / self._batch_size
        local_loss = local_vec.sum() * alpha / self._batch_size
        return loss, local_loss.detach()

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
        # Z-loss is folded into a running detached scalar for logging only. The differentiable
        # per-layer scalar is injected back into the main forward graph via AuxLossScaler at
        # accumulate() time, so we never need to keep per-layer logsum tensors around. This is
        # the memory-saving pattern adapted from Megatron's MoEAuxLossAutoScaler.
        self._running_loss_for_log: torch.Tensor | None = None
        # Detached per-rank local component (the same running scalar but WITHOUT the `× world_size`
        # global-average factor). The display pipeline restores the global value by SUM-reducing
        # this across ranks, which keeps logged curves global once backward drops the factor.
        self._local_running_loss_for_log: torch.Tensor | None = None

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
        num_tokens_local: int,
        num_tokens_global: torch.Tensor | None,
        world_size: int,
    ) -> torch.Tensor:
        """Compute z-loss for one layer and return it as a scalar with autograd
        attached.

        The caller is expected to inject the returned scalar back into the main forward graph
        via :class:`AuxLossScaler`; the autograd graph behind this scalar (logsumexp -> square ->
        sum -> scaling) is then released as soon as the corresponding layer is reached during
        backward, instead of being pinned until a global ``finalize()``.

        Args:
            router_logits (torch.Tensor): Router logits with non-padding tokens already selected.
                Shape ``(non_pad, n_routed_experts)``.
            num_tokens_local (int): Number of non-padding tokens on this rank for the current
                forward (constant across MoE layers in a single forward).
            num_tokens_global (torch.Tensor | None): All-reduced non-padding token count across
                ranks, as an int64 scalar tensor. ``None`` when ``z_loss_global_average`` is off
                or the process group is not initialized.
            world_size (int): Number of ranks contributing to ``num_tokens_global``. Ignored when
                ``num_tokens_global`` is ``None``.

        Returns:
            torch.Tensor: Per-layer z-loss as a 0-d tensor with autograd graph back to
            ``router_logits``.
        """
        if self.loss_cfg.z_loss_alpha == 0:
            zero = torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)
            self._update_running(zero)
            self._update_local_running(zero)
            return zero

        denom_local = max(num_tokens_local, 1)
        base = torch.logsumexp(router_logits, dim=-1).square().sum() / denom_local

        loss = base
        if self.loss_cfg.z_loss_global_average and num_tokens_global is not None:
            # Under reduce-sum gradients the injected z-loss stays as this rank's local component
            # (its share of the global-average z-loss, WITHOUT any `× world_size`). Cross-rank
            # aggregation happens on the gradients via the FSDP SUM reduce-scatter; summing this
            # local component over ranks reproduces the global z-loss.
            denom_global = torch.clamp(num_tokens_global, min=1)
            loss = base * num_tokens_local / denom_global

        loss = loss * self.loss_cfg.z_loss_alpha / self._batch_size
        self._update_running(loss.detach())
        self._update_local_running(loss.detach())
        return loss

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the accumulated z-loss as detached scalars for logging only.

        The differentiable contribution has already been injected into the main forward graph at
        each ``accumulate()`` call, so these values carry no autograd graph; they exist purely to
        populate logging.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(z_loss, local_z_loss)``. The first is the current
            global-average running scalar (kept for the ``z_loss`` output field); the second is the
            detached per-rank local component (without ``× world_size``) whose cross-rank SUM equals
            the first, used by the display pipeline.
        """
        value = self._running_loss_for_log
        local_value = self._local_running_loss_for_log
        self._running_loss_for_log = None
        self._local_running_loss_for_log = None
        zero = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        return (
            value if value is not None else zero,
            local_value if local_value is not None else zero,
        )

    def _update_running(self, value: torch.Tensor) -> None:
        if self._running_loss_for_log is None:
            self._running_loss_for_log = value.clone()
        else:
            self._running_loss_for_log = self._running_loss_for_log + value

    def _update_local_running(self, value: torch.Tensor) -> None:
        if self._local_running_loss_for_log is None:
            self._local_running_loss_for_log = value.clone()
        else:
            self._local_running_loss_for_log = self._local_running_loss_for_log + value

    @property
    def batch_size(self) -> int:
        return self._batch_size
