from typing import Any, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.utils import get_torch_device_module
from xtuner.v1.utils.router_offload import AsyncOffloadedTensor, async_offload_to_cpu, wait_async_offload


DEVICE_MODULE = get_torch_device_module()


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


def select_nonpad(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Select non-padding positions from tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
        mask (torch.Tensor): Attention mask.
        dim (int): Select dimension.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Selected tensor and selected indices.
    """
    indices = torch.nonzero(mask, as_tuple=True)[1]
    selected = torch.index_select(tensor, dim, indices).contiguous().float()
    return selected, indices


def maybe_offload_tensor(
    tensor: torch.Tensor,
    split_bal_loss: bool,
    offload_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Offload tensor to CPU only when split_bal_loss is enabled.

    When split layer balancing is enabled, return a materialized CPU tensor so
    downstream Pydantic models only see plain ``torch.Tensor`` values.
    """
    if split_bal_loss:
        return wait_async_offload(async_offload_to_cpu(tensor, offload_stream))
    return tensor


def maybe_wait_offload_tensor(
    tensor: torch.Tensor | AsyncOffloadedTensor,
    split_bal_loss: bool,
) -> torch.Tensor:
    """Return a plain tensor for both eager and offloaded inputs."""
    if split_bal_loss and isinstance(tensor, AsyncOffloadedTensor):
        return wait_async_offload(tensor)
    return cast(torch.Tensor, tensor).detach()


class LayerBalancingLossConfig(BaseModel):
    """Configuration for layer-wise split balancing loss."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    num_layers: int | None = None
    n_routed_experts: int | None = None
    device: torch.device | str | int | None = None

    def build(
        self,
        *,
        num_layers: int | None = None,
        n_routed_experts: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> "LayerBalancingLoss":
        """Build layer balancing loss context.

        Args:
            num_layers (int | None): Number of layers. Fallback to config value when None.
            n_routed_experts (int | None): Number of routed experts. Fallback to config value when None.
            device (torch.device | str | None): Device used for internal accumulators.
                Fallback order: argument -> config field -> DEVICE_MODULE.current_device().

        Returns:
            LayerBalancingLoss: Built context.
        """
        resolved_num_layers = num_layers if num_layers is not None else self.num_layers
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_num_layers is not None, "num_layers must be provided either in config or build()."
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_device = device if device is not None else self.device
        if resolved_device is None:
            resolved_device = DEVICE_MODULE.current_device()

        return LayerBalancingLoss(
            num_layers=resolved_num_layers,
            n_routed_experts=resolved_n_routed_experts,
            device=resolved_device,
        )


class LayerBalancingLossKwargs(BaseModel):
    """Keyword arguments for layer-wise split balancing loss context."""

    model_config = ConfigDict(
        title="layer balancing loss keyword arguments", extra="forbid", arbitrary_types_allowed=True
    )
    device: torch.device | str | int | None


class LayerBalancingLossContext(nn.Module):
    """Layer-wise balancing loss accumulator used by split_bal_loss mode."""

    def __init__(self, loss_cfg: LayerBalancingLossConfig, loss_kwargs: LayerBalancingLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        num_layers = self.loss_cfg.num_layers
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert num_layers is not None
        assert n_routed_experts is not None

        self.local_load = torch.zeros(num_layers, n_routed_experts, device=loss_kwargs.device)
        self.routing_weights_sum_list: list[torch.Tensor] = []
        self.local_load_logits = torch.zeros(
            num_layers,
            n_routed_experts,
            dtype=torch.int64,
            device=loss_kwargs.device,
        )

    def update(
        self,
        layer_idx: int,
        router_weights: torch.Tensor,
        num_experts_per_tok: int,
        router_logits: torch.Tensor,
    ) -> None:
        """Update accumulators for one layer.

        Args:
            layer_idx (int): Layer index.
            router_weights (torch.Tensor): Router weights, shape (1, non_pad_seq, n_experts).
            num_experts_per_tok (int): Number of experts selected per token.
            router_logits (torch.Tensor): Router logits, shape (1, non_pad_seq, n_experts).
        """
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert n_routed_experts is not None

        _, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
        tokens_per_expert = torch.histc(
            selected_experts.view(-1),
            bins=n_routed_experts,
            min=0,
            max=n_routed_experts,
        ).float()
        self.local_load[layer_idx] = tokens_per_expert
        self.routing_weights_sum_list.append(router_weights.sum(dim=0))

        _, selected_experts = torch.topk(router_logits, num_experts_per_tok, dim=-1)
        tokens_per_expert_logits = torch.histc(
            selected_experts.view(-1),
            bins=n_routed_experts,
            min=0,
            max=n_routed_experts,
        ).to(torch.long)
        self.local_load_logits[layer_idx] = tokens_per_expert_logits

    def finalize(
        self,
        dist_init: bool,
        num_experts_per_tok: int,
        non_pad_token: int,
        balancing_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        """Finalize layer-wise balancing loss.

        Args:
            dist_init (bool): Whether to use distributed global average mode.
            num_experts_per_tok (int): Number of experts selected per token.
            non_pad_token (int): Number of non-padding tokens.
            balancing_loss_weight (float): Balancing loss weight.

        Returns:
            torch.Tensor: Final balancing loss.
        """
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert n_routed_experts is not None
        local_gating_sum = torch.stack(self.routing_weights_sum_list, dim=0)

        if dist_init:
            group = dist.group.WORLD
            assert group is not None
            tokens_per_expert_global = all_reduce(self.local_load, "sum", group)
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok

            routing_weights_sum_global = all_reduce_autograd(local_gating_sum, "sum", group)
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = n_routed_experts / tokens_global
        else:
            tokens_per_expert_global = self.local_load
            valid_tokens = max(non_pad_token, 1)
            scale_global = n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean_global = local_gating_sum / valid_tokens

        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        return loss.sum() * balancing_loss_weight

    def cal_tokens_per_expert(self) -> torch.Tensor:
        """Get tokens-per-expert tensor for logging/bias update."""
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            return all_reduce(self.local_load_logits, "sum", group)
        return self.local_load_logits


class LayerBalancingLoss(LayerBalancingLossContext):
    """Backward-compatible wrapper keeping the original constructor usage."""

    def __init__(self, num_layers: int, n_routed_experts: int, device: torch.device | str | int):
        cfg = LayerBalancingLossConfig(num_layers=num_layers, n_routed_experts=n_routed_experts, device=device)
        kwargs = LayerBalancingLossKwargs(device=device)
        super().__init__(cfg, kwargs)


def prepare_layer_balancing_loss(
    layer_balancing_cfg: LayerBalancingLossConfig | None,
    *,
    num_layers: int,
    n_routed_experts: int,
) -> LayerBalancingLoss | None:
    """Build layer balancing loss object from config.

    Args:
        layer_balancing_cfg (LayerBalancingLossConfig | None): Layer balancing config.
        num_layers (int): Number of decoder layers.
        n_routed_experts (int): Number of routed experts.

    Returns:
        LayerBalancingLoss | None: Built object when enabled, else None.
    """
    if layer_balancing_cfg is None:
        return None

    return layer_balancing_cfg.build(
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
    )


def accumulate_layer_balancing_loss(
    layer_balancing_loss: LayerBalancingLoss | None,
    *,
    layer_idx: int,
    router_weights: torch.Tensor,
    router_logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    num_experts_per_tok: int,
) -> None:
    """Accumulate per-layer balancing statistics.

    This is a no-op when layer balancing loss is disabled.
    """
    if layer_balancing_loss is None:
        return

    router_weights_selected, _ = select_nonpad(router_weights, mask, dim=dim)
    router_logits_selected, _ = select_nonpad(router_logits, mask, dim=dim)
    layer_balancing_loss.update(
        layer_idx,
        router_weights_selected,
        num_experts_per_tok,
        router_logits_selected,
    )


def finalize_layer_balancing_loss(
    layer_balancing_loss: LayerBalancingLoss | None,
    *,
    balancing_ctx: Any,
    num_experts_per_tok: int,
    non_pad_token: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Finalize balancing loss and tokens-per-expert from accumulated layer
    stats.

    Returns None when layer balancing is disabled or balancing_ctx is None.
    """
    if layer_balancing_loss is None or balancing_ctx is None:
        return None

    dist_init = balancing_ctx.loss_cfg.balancing_loss_global_average and dist.is_initialized()
    balancing_loss = layer_balancing_loss.finalize(
        dist_init,
        num_experts_per_tok,
        non_pad_token=non_pad_token,
        balancing_loss_weight=balancing_ctx.loss_cfg.balancing_loss_alpha,
    )
    tokens_per_expert_global = layer_balancing_loss.cal_tokens_per_expert()
    return balancing_loss, tokens_per_expert_global
