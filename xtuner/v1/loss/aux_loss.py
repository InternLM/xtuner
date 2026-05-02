import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.loss.moe_loss import BalancingLossContext, ZLossContext
from xtuner.v1.utils import get_torch_device_module


DEVICE_MODULE = get_torch_device_module()


class AuxLossConfig(BaseModel):
    """Configuration for layer-wise split MoE auxiliary loss."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    device: torch.device | str | int | None = None

    def build(
        self,
        *,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> "AuxLoss":
        """Build a layer-wise MoE auxiliary loss context."""
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_num_experts_per_tok = (
            num_experts_per_tok if num_experts_per_tok is not None else self.num_experts_per_tok
        )
        assert resolved_num_experts_per_tok is not None, (
            "num_experts_per_tok must be provided either in config or build()."
        )

        resolved_device = device if device is not None else self.device
        if resolved_device is None:
            resolved_device = DEVICE_MODULE.current_device()

        return AuxLoss(
            n_routed_experts=resolved_n_routed_experts,
            num_experts_per_tok=resolved_num_experts_per_tok,
            device=resolved_device,
        )


class AuxLossKwargs(BaseModel):
    """Keyword arguments for layer-wise split MoE auxiliary loss context."""

    model_config = ConfigDict(title="layer moe loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)
    device: torch.device | str | int | None


class AuxLossContext(nn.Module):
    """Layer-wise split MoE auxiliary loss dispatcher.

    Owns the per-layer ``tokens_per_expert`` accumulator used both by logging / bias update and by
    ``BalancingLossContext.finalize``. Sub-context accumulators (router_weights_sum for balancing,
    logsum / token_count for z-loss) live inside their respective contexts.
    """

    def __init__(self, loss_cfg: AuxLossConfig, loss_kwargs: AuxLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        n_routed_experts = self.loss_cfg.n_routed_experts
        num_experts_per_tok = self.loss_cfg.num_experts_per_tok
        assert n_routed_experts is not None, "n_routed_experts must be resolved before creating AuxLossContext."
        assert num_experts_per_tok is not None, "num_experts_per_tok must be resolved before creating AuxLossContext."
        self.n_routed_experts: int = n_routed_experts
        self.num_experts_per_tok: int = num_experts_per_tok
        self._local_load_logits_list: list[torch.Tensor] = []

    def accumulate(
        self,
        *,
        selected_router_weights: torch.Tensor,
        selected_router_logits: torch.Tensor,
        balancing_ctx: list[BalancingLossContext] | BalancingLossContext | None = None,
        z_ctx: list[ZLossContext] | ZLossContext | None = None,
    ) -> None:
        """Accumulate routing statistics for one layer.

        Args:
            selected_router_weights (torch.Tensor): Router weights with non-padding tokens already
                selected. Shape: ``(non_pad, n_routed_experts)``.
            selected_router_logits (torch.Tensor): Router logits with non-padding tokens already
                selected. Shape: ``(non_pad, n_routed_experts)``.
            balancing_ctx (list[BalancingLossContext] | BalancingLossContext | None): Balancing loss
                context(s) to fan-out to. ``None`` to skip.
            z_ctx (list[ZLossContext] | ZLossContext | None): Z-loss context(s) to fan-out to.
                ``None`` to skip.
        """
        # tokens_per_expert is non-differentiable (topk + histc) and shared between
        # logging output and BalancingLossContext.finalize. Owned here as the single source of truth.
        _, selected_experts = torch.topk(selected_router_weights, self.num_experts_per_tok, dim=-1)
        tokens_per_expert_l = torch.histc(
            selected_experts.view(-1),
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        ).to(torch.long)
        self._local_load_logits_list.append(tokens_per_expert_l)

        for ctx in _as_list(balancing_ctx):
            ctx.accumulate(router_weights=selected_router_weights)

        for ctx in _as_list(z_ctx):
            ctx.accumulate(router_logits=selected_router_logits)

    def finalize(
        self,
        *,
        balancing_ctx: list[BalancingLossContext] | BalancingLossContext | None,
        z_ctx: list[ZLossContext] | ZLossContext | None,
        non_pad_token: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Finalize split auxiliary losses and expert counts from runtime
        state."""
        tokens_per_expert_local, tokens_per_expert_global = self._cal_tokens_per_expert()

        balancing_loss: torch.Tensor | None = None
        balancing_list = _as_list(balancing_ctx)
        if balancing_list:
            partials = [
                ctx.finalize(
                    tokens_per_expert_local=tokens_per_expert_local,
                    tokens_per_expert_global=tokens_per_expert_global,
                    n_routed_experts=self.n_routed_experts,
                    num_experts_per_tok=self.num_experts_per_tok,
                    non_pad_token=non_pad_token,
                )
                for ctx in balancing_list
            ]
            balancing_loss = partials[0] if len(partials) == 1 else torch.stack(partials).sum(dim=0)

        z_loss: torch.Tensor | None = None
        z_list = _as_list(z_ctx)
        if z_list:
            partials = [ctx.finalize() for ctx in z_list]
            z_loss = partials[0] if len(partials) == 1 else torch.stack(partials).sum(dim=0)

        return balancing_loss, z_loss, tokens_per_expert_global

    def _cal_tokens_per_expert(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack per-layer expert counts and produce both local and globally
        reduced views.

        The local view is needed by BalancingLossContext's non-global-average branch (per-rank scaling); the global
        view is what the consumer (logging / bias update) wants.
        """
        local_load_logits = self._local_load_logits_list
        self._local_load_logits_list = []

        if not local_load_logits:
            raise RuntimeError(
                "No MoE routing statistics were accumulated before finalize(). "
                "This usually means the model has no MoE layers or finalize() was called "
                "without a preceding accumulate()."
            )
        tokens_per_expert_local = torch.stack(local_load_logits, dim=0)
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            tokens_per_expert_global = all_reduce(tokens_per_expert_local, "sum", group)
        else:
            tokens_per_expert_global = tokens_per_expert_local
        return tokens_per_expert_local, tokens_per_expert_global


class AuxLoss(AuxLossContext):
    """Unified MoE auxiliary loss wrapper."""

    def __init__(self, n_routed_experts: int, num_experts_per_tok: int, device: torch.device | str | int):
        cfg = AuxLossConfig(
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            device=device,
        )
        kwargs = AuxLossKwargs(device=device)
        super().__init__(cfg, kwargs)


def _as_list(
    ctx: list | object | None,
) -> list:
    if ctx is None:
        return []
    if isinstance(ctx, list):
        return ctx
    return [ctx]
