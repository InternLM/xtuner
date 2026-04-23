from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.utils import get_torch_device_module


if TYPE_CHECKING:
    from xtuner.v1.loss.base_loss_ctx import BaseLossContext
    from xtuner.v1.loss.moe_loss import BalancingLossContext, ZLossContext
    from xtuner.v1.model.moe.moe import MoELossContextDict


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


class AuxLossConfig(BaseModel):
    """Configuration for layer-wise split MoE auxiliary loss."""

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
    ) -> "AuxLoss":
        """Build a layer-wise MoE auxiliary loss context."""
        resolved_num_layers = num_layers if num_layers is not None else self.num_layers
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_num_layers is not None, "num_layers must be provided either in config or build()."
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_device = device if device is not None else self.device
        if resolved_device is None:
            resolved_device = DEVICE_MODULE.current_device()

        return AuxLoss(
            num_layers=resolved_num_layers,
            n_routed_experts=resolved_n_routed_experts,
            device=resolved_device,
        )


class AuxLossKwargs(BaseModel):
    """Keyword arguments for layer-wise split MoE auxiliary loss context."""

    model_config = ConfigDict(title="layer moe loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)
    device: torch.device | str | int | None


class AuxLossContext(nn.Module):
    """Layer-wise MoE auxiliary loss accumulator used by split aux-loss
    mode."""

    def __init__(self, loss_cfg: AuxLossConfig, loss_kwargs: AuxLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        num_layers = self.loss_cfg.num_layers
        n_routed_experts = self.loss_cfg.n_routed_experts
        assert num_layers is not None, "num_layers must be resolved before creating AuxLossContext."
        assert n_routed_experts is not None, "n_routed_experts must be resolved before creating AuxLossContext."
        self.num_layers: int = num_layers
        self.n_routed_experts: int = n_routed_experts

        self.local_load: torch.Tensor | None = None
        self.routing_weights_sum_list: list[torch.Tensor] = []
        self.local_load_logits: torch.Tensor | None = None
        self.z_loss_logsum: torch.Tensor | None = None
        self.z_loss_token_count: torch.Tensor | None = None
        self._balancing_enabled = False
        self._z_loss_enabled = False
        self._active_layers = 0

    def configure_runtime(
        self,
        *,
        loss_ctx: "list[MoELossContextDict] | MoELossContextDict | None",
    ) -> None:
        """Configure per-forward split-aux branches.

        Args:
                loss_ctx (list[MoELossContextDict] | MoELossContextDict | None): Generic loss context (single dict or list of dicts).
        """
        balancing_ctx, z_loss_ctx = self._extract_aux_ctx(loss_ctx)
        balancing_enabled = balancing_ctx is not None
        z_loss_enabled = z_loss_ctx is not None

        self._balancing_enabled = balancing_enabled
        self._z_loss_enabled = z_loss_enabled
        self.local_load_logits = torch.zeros(
            self.num_layers,
            self.n_routed_experts,
            dtype=torch.int64,
            device=self.loss_kwargs.device,
        )
        if self._balancing_enabled:
            self.local_load = torch.zeros(self.num_layers, self.n_routed_experts, device=self.loss_kwargs.device)
        else:
            self.local_load = None
        if self._z_loss_enabled:
            self.z_loss_logsum = torch.zeros(self.num_layers, dtype=torch.float32, device=self.loss_kwargs.device)
            self.z_loss_token_count = torch.zeros(self.num_layers, dtype=torch.int64, device=self.loss_kwargs.device)

    def _extract_aux_ctx(
        self, loss_ctx: "list[MoELossContextDict] | MoELossContextDict | None"
    ) -> tuple["BaseLossContext | None", "BaseLossContext | None"]:
        """Extract balancing and z-loss contexts from one or many loss
        contexts."""
        if loss_ctx is None:
            return None, None

        if isinstance(loss_ctx, list):
            balancing_ctx = next((ctx.get("balancing") for ctx in loss_ctx if ctx.get("balancing") is not None), None)
            z_loss_ctx = next((ctx.get("z_loss") for ctx in loss_ctx if ctx.get("z_loss") is not None), None)
            return balancing_ctx, z_loss_ctx

        if hasattr(loss_ctx, "get"):
            balancing_ctx = loss_ctx.get("balancing")
            z_loss_ctx = loss_ctx.get("z_loss")
            return balancing_ctx, z_loss_ctx

        return None, None

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
                router_weights (torch.Tensor): Router weights, shape (non_pad_seq, n_experts).
                num_experts_per_tok (int): Number of experts selected per token.
                router_logits (torch.Tensor): Router logits, shape (non_pad_seq, n_experts).
        """
        local_load_logits = self.local_load_logits
        assert local_load_logits is not None

        _, selected_experts = torch.topk(router_logits, num_experts_per_tok, dim=-1)
        tokens_per_expert_logits = torch.histc(
            selected_experts.view(-1),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        ).to(torch.long)
        local_load_logits[layer_idx] = tokens_per_expert_logits

        if self._balancing_enabled:
            local_load = self.local_load
            assert local_load is not None

            _, selected_experts = torch.topk(router_weights, num_experts_per_tok, dim=-1)
            tokens_per_expert = torch.histc(
                selected_experts.view(-1),
                bins=self.n_routed_experts,
                min=0,
                max=self.n_routed_experts,
            ).float()
            local_load[layer_idx] = tokens_per_expert
            self.routing_weights_sum_list.append(router_weights.sum(dim=0))

        self._active_layers = max(self._active_layers, layer_idx + 1)

        if self._z_loss_enabled:
            z_loss_token_count = self.z_loss_token_count
            z_loss_logsum = self.z_loss_logsum
            assert z_loss_token_count is not None
            assert z_loss_logsum is not None
            z_loss_token_count[layer_idx] = router_logits.shape[0]
            z_loss_logsum[layer_idx] = torch.logsumexp(router_logits, dim=-1).square().sum()

    def accumulate(
        self,
        *,
        layer_idx: int,
        router_weights: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts_per_tok: int,
        mask: torch.Tensor,
        dim: int = 1,
    ) -> None:
        """Accumulate routing statistics for one layer.

        This keeps the split MoE auxiliary-loss path per-layer and avoids retaining a
        full ``(num_layers, seq_len, num_experts)`` router tensor in memory.
        """
        selected_router_weights, _ = select_nonpad(router_weights, mask, dim=dim)
        selected_router_logits, _ = select_nonpad(router_logits, mask, dim=dim)
        self.update(
            layer_idx,
            selected_router_weights,
            num_experts_per_tok,
            selected_router_logits,
        )

    def finalize_balance_loss(
        self,
        dist_init: bool,
        num_experts_per_tok: int,
        non_pad_token: int,
        moe_loss_weight: float = 1.0,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Finalize the layer-wise MoE auxiliary loss term.

        Args:
                dist_init (bool): Whether to use distributed global average mode.
                num_experts_per_tok (int): Number of experts selected per token.
                non_pad_token (int): Number of non-padding tokens.
                moe_loss_weight (float): Auxiliary loss weight.
                batch_size (int): Number of micro-batches calibrated together for gradient accumulation.

        Returns:
                torch.Tensor: Final MoE auxiliary loss.
        """
        if not self._balancing_enabled:
            return torch.tensor(0.0, device=self.loss_kwargs.device, dtype=torch.float32)
        if self._active_layers == 0 or not self.routing_weights_sum_list:
            return torch.tensor(0.0, device=self.loss_kwargs.device, dtype=torch.float32)

        active_layers = self._active_layers
        local_gating_sum = torch.stack(self.routing_weights_sum_list, dim=0)
        local_gating_sum = local_gating_sum[:active_layers]
        local_load = self.local_load
        assert local_load is not None
        local_load = local_load[:active_layers]

        if dist_init:
            group = dist.group.WORLD
            assert group is not None
            tokens_per_expert_global = all_reduce(local_load, "sum", group)
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok

            routing_weights_sum_global = all_reduce_autograd(local_gating_sum, "sum", group)
            routing_weights_mean_global = routing_weights_sum_global / seqlen_global.unsqueeze(-1)
            scale_global = self.n_routed_experts / tokens_global
        else:
            tokens_per_expert_global = local_load
            valid_tokens = max(non_pad_token, 1)
            scale_global = self.n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean_global = local_gating_sum / valid_tokens

        loss = scale_global * (tokens_per_expert_global * routing_weights_mean_global).sum(-1)
        loss = loss.sum() * moe_loss_weight
        return loss / batch_size

    def cal_tokens_per_expert(self) -> torch.Tensor:
        """Get tokens-per-expert tensor for logging/bias update."""
        local_load_logits = self.local_load_logits
        assert local_load_logits is not None
        if self._active_layers == 0:
            return torch.zeros(0, self.n_routed_experts, dtype=torch.int64, device=self.loss_kwargs.device)
        active_load_logits = local_load_logits[: self._active_layers]
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            return all_reduce(active_load_logits, "sum", group)
        return active_load_logits

    def finalize_z_loss(
        self,
        *,
        global_average: bool,
        z_loss_weight: float = 1.0,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Finalize the split z-loss accumulated across all layers.

        Args:
                global_average (bool): Whether to use distributed global average mode.
                z_loss_weight (float): Z-loss weight.
                batch_size (int): Number of micro-batches calibrated together for gradient accumulation.

        Returns:
                torch.Tensor: Final z-loss value.
        """
        if not self._z_loss_enabled or z_loss_weight == 0:
            return torch.tensor(0.0, device=self.loss_kwargs.device, dtype=torch.float32)

        z_loss_token_count = self.z_loss_token_count
        z_loss_logsum = self.z_loss_logsum
        assert z_loss_token_count is not None
        assert z_loss_logsum is not None
        if self._active_layers == 0:
            return torch.tensor(0.0, device=self.loss_kwargs.device, dtype=torch.float32)
        active_layers = self._active_layers
        z_loss_token_count = z_loss_token_count[:active_layers]
        z_loss_logsum = z_loss_logsum[:active_layers]

        token_count = torch.clamp(z_loss_token_count, min=1)
        loss = z_loss_logsum / token_count.to(z_loss_logsum.dtype)
        if global_average and dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            token_count_global = all_reduce(z_loss_token_count, "sum", group)
            token_count_global = torch.clamp(token_count_global, min=1)
            world_size = dist.get_world_size()
            loss = loss * z_loss_token_count.to(z_loss_logsum.dtype) * world_size / token_count_global

        loss = loss.sum()

        loss = loss * z_loss_weight
        return loss / batch_size

    def reset(self) -> None:
        """Clear accumulated state before a new forward pass."""
        if self.local_load is not None:
            self.local_load.zero_()
        self.routing_weights_sum_list.clear()
        if self.local_load_logits is not None:
            self.local_load_logits.zero_()
        self._balancing_enabled = False
        self._z_loss_enabled = False
        self._active_layers = 0
        if self.z_loss_logsum is not None and self.z_loss_token_count is not None:
            self.z_loss_logsum.zero_()
            self.z_loss_token_count.zero_()

    def finalize(
        self,
        *,
        num_experts_per_tok: int,
        non_pad_token: int,
        balancing_ctx: "BaseLossContext | None" = None,
        z_ctx: "BaseLossContext | None" = None,
        loss_ctx: "list[MoELossContextDict] | MoELossContextDict | None" = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor] | None:
        """Finalize split auxiliary losses and expert counts from runtime
        contexts."""
        if loss_ctx is not None:
            balancing_ctx, z_ctx = self._extract_aux_ctx(loss_ctx)

        balancing_loss = None
        if balancing_ctx is not None:
            balancing_loss_ctx = cast("BalancingLossContext", balancing_ctx)
            dist_init = balancing_loss_ctx.loss_cfg.balancing_loss_global_average and dist.is_initialized()
            balancing_loss = self.finalize_balance_loss(
                dist_init,
                num_experts_per_tok,
                non_pad_token=non_pad_token,
                moe_loss_weight=balancing_loss_ctx.loss_cfg.balancing_loss_alpha,
                batch_size=balancing_loss_ctx.batch_size,
            )

        z_loss = None
        if z_ctx is not None:
            z_loss_ctx = cast("ZLossContext", z_ctx)
            z_loss = self.finalize_z_loss(
                global_average=z_loss_ctx.loss_cfg.z_loss_global_average,
                z_loss_weight=z_loss_ctx.loss_cfg.z_loss_alpha,
                batch_size=z_loss_ctx.batch_size,
            )

        tokens_per_expert_global = self.cal_tokens_per_expert()
        return balancing_loss, z_loss, tokens_per_expert_global


class AuxLoss(AuxLossContext):
    """Unified layer-wise MoE auxiliary loss wrapper."""

    def __init__(self, num_layers: int, n_routed_experts: int, device: torch.device | str | int):
        cfg = AuxLossConfig(num_layers=num_layers, n_routed_experts=n_routed_experts, device=device)
        kwargs = AuxLossKwargs(device=device)
        super().__init__(cfg, kwargs)

    @classmethod
    def prepare(
        cls,
        aux_loss_cfg: AuxLossConfig | None,
        *,
        num_layers: int,
        n_routed_experts: int,
    ) -> "AuxLoss | None":
        """Build a split MoE auxiliary loss object from config when enabled."""
        if aux_loss_cfg is None:
            return None

        return aux_loss_cfg.build(
            num_layers=num_layers,
            n_routed_experts=n_routed_experts,
        )
