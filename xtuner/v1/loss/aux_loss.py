import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce

from xtuner.v1.loss.moe_loss import BalancingLossContext, ZLossContext


class AuxLossScaler(torch.autograd.Function):
    """Inject an auxiliary loss into the main forward graph as a passthrough.

    ``apply(carrier, aux_loss)`` returns ``carrier`` unchanged in the forward, but registers
    ``aux_loss`` in autograd so that when the main loss backward traverses this node it injects
    ``ones_like(aux_loss)`` into the aux_loss subgraph. This triggers the per-layer aux_loss
    backward inline with the main backward at the corresponding layer, instead of holding all
    layers' aux_loss saved tensors alive until a global ``finalize`` node fires.

    Adapted from Megatron-LM's ``MoEAuxLossAutoScaler``.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss)


class AuxLossConfig(BaseModel):
    """Configuration for layer-wise split MoE auxiliary loss."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None

    def build(
        self,
        *,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
    ) -> "AuxLossContext":
        """Build a layer-wise MoE auxiliary loss context."""
        resolved_n_routed_experts = n_routed_experts if n_routed_experts is not None else self.n_routed_experts
        assert resolved_n_routed_experts is not None, "n_routed_experts must be provided either in config or build()."

        resolved_num_experts_per_tok = (
            num_experts_per_tok if num_experts_per_tok is not None else self.num_experts_per_tok
        )
        assert resolved_num_experts_per_tok is not None, (
            "num_experts_per_tok must be provided either in config or build()."
        )

        return AuxLossContext(
            AuxLossConfig(
                n_routed_experts=resolved_n_routed_experts,
                num_experts_per_tok=resolved_num_experts_per_tok,
            )
        )


class AuxLossContext(nn.Module):
    """Layer-wise split MoE auxiliary loss dispatcher.

    Owns the per-layer ``tokens_per_expert`` accumulator used both by logging / bias update and by
    ``BalancingLossContext.finalize``. Sub-context accumulators (router_weights_sum for balancing,
    logsum / token_count for z-loss) live inside their respective contexts.
    """

    def __init__(self, loss_cfg: AuxLossConfig):
        super().__init__()
        self.loss_cfg = loss_cfg
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
        selected_experts: torch.Tensor,
        hidden_states: torch.Tensor,
        balancing_ctx: list[BalancingLossContext] | BalancingLossContext | None = None,
        z_ctx: list[ZLossContext] | ZLossContext | None = None,
        num_tokens_local: int = 0,
        num_tokens_global: torch.Tensor | None = None,
        world_size: int = 1,
    ) -> torch.Tensor:
        """Accumulate routing statistics for one layer and inject z-loss into
        the main graph.

        Args:
            selected_router_weights (torch.Tensor): Router weights with non-padding tokens already
                selected. Shape: ``(non_pad, n_routed_experts)``.
            selected_router_logits (torch.Tensor): Router logits with non-padding tokens already
                selected. Shape: ``(non_pad, n_routed_experts)``.
            selected_experts (torch.Tensor): Expert IDs assigned by the router, with non-padding
                tokens already selected. Shape: ``(non_pad, num_experts_per_tok)``.
            hidden_states (torch.Tensor): A carrier tensor on the main forward path. Z-loss is
                attached to it via :class:`AuxLossScaler` so that backward through the main loss
                releases this layer's logsumexp saved tensor inline.
            balancing_ctx (list[BalancingLossContext] | BalancingLossContext | None): Balancing loss
                context(s) to fan-out to. ``None`` to skip.
            z_ctx (list[ZLossContext] | ZLossContext | None): Z-loss context(s) to fan-out to.
                ``None`` to skip.
            num_tokens_local (int): Non-padding token count on this rank for the current forward.
                Required when any z-loss context is provided.
            num_tokens_global (torch.Tensor | None): All-reduced non-padding token count across
                ranks (int64 scalar). Pass ``None`` when ``z_loss_global_average`` is off or no
                process group is initialized.
            world_size (int): World size that produced ``num_tokens_global``.

        Returns:
            torch.Tensor: ``hidden_states`` augmented with the per-layer z-loss autograd hook.
            Identical in value to the input; the caller must replace its handle so the hook is
            preserved on the main forward graph.
        """
        # The router is the single source of truth for assignments: grouped or
        # rollout routing is not equivalent to another global Top-K over weights.
        tokens_per_expert_l = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.n_routed_experts,
        )
        self._local_load_logits_list.append(tokens_per_expert_l)

        for ctx in _as_list(balancing_ctx):
            ctx.accumulate(router_weights=selected_router_weights)

        for ctx in _as_list(z_ctx):
            z_loss_l = ctx.accumulate(
                router_logits=selected_router_logits,
                num_tokens_local=num_tokens_local,
                num_tokens_global=num_tokens_global,
                world_size=world_size,
            )
            hidden_states = AuxLossScaler.apply(hidden_states, z_loss_l)

        return hidden_states

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


def _as_list(
    ctx: list | object | None,
) -> list:
    if ctx is None:
        return []
    if isinstance(ctx, list):
        return ctx
    return [ctx]
