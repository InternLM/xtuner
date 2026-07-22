from typing import cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import TypedDict

from xtuner.v1.loss.moe_loss import BalancingLossConfig, BalancingLossContext, ZLossConfig, ZLossContext
from xtuner.v1.loss.utils import sp_split


class AuxLossFinalizeOutput(TypedDict):
    """Backward-relevant outputs of :meth:`AuxLossContext.finalize`.

    Per-rank display values are not returned here; each loss context exposes them via ``calibrate()``.
    """

    balancing_loss: torch.Tensor | None
    z_loss: torch.Tensor | None
    tokens_per_expert_global: torch.Tensor


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
    """Configuration and factory for the per-micro-batch MoE auxiliary-loss
    hub.

    Besides the routing dimensions, it aggregates the optional balancing / z-loss sub-configs so that
    :meth:`build` can turn one micro-batch's data into a fully data-bound :class:`AuxLossContext`
    (non-padding indices + token-count denominators + sub-contexts) in a single call, mirroring the
    ``BaseLossConfig.build(data, sp_mesh)`` contract used by CE / MTP. ``n_routed_experts`` /
    ``num_experts_per_tok`` and the sub-configs are populated from the owning model config
    (see ``MoEConfig``); aux losses are configured through the model config, not here.

    Args:
        n_routed_experts (int | None): Number of routed experts (set from the model config).
        num_experts_per_tok (int | None): Experts selected per token (set from the model config).
        balancing_loss_cfg (BalancingLossConfig | None): Balancing sub-config, or ``None`` when disabled.
        z_loss_cfg (ZLossConfig | None): Z-loss sub-config, or ``None`` when disabled.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    balancing_loss_cfg: BalancingLossConfig | None = None
    z_loss_cfg: ZLossConfig | None = None

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "AuxLossContext":
        """Build a data-bound per-micro-batch MoE auxiliary-loss hub.

        Reads the padding mask from ``data["seq_ctx"]`` (SP-split when a sequence-parallel mesh is
        given, mirroring CE) to derive this micro-batch's non-padding indices and the token-count
        denominators, builds the optional balancing / z-loss sub-contexts from the aggregated configs,
        and returns a hub already carrying all of them -- so the forward runs no aux-token collective.
        With no balancing / z sub-configs the result is a tokens-only hub (still produces
        ``tokens_per_expert`` for maxvio / bias update, no aux backward, no collective).

        Args:
            data (dict): Micro-batch data; must contain ``"seq_ctx"`` (a ``SequenceContext``).
            sp_mesh (DeviceMesh | None): Sequence-parallel mesh; the mask is SP-split when its size > 1.

        Returns:
            AuxLossContext: The per-micro-batch auxiliary-loss hub bound to this data.
        """
        assert self.n_routed_experts is not None, "n_routed_experts must be set on AuxLossConfig before build()."
        assert self.num_experts_per_tok is not None, "num_experts_per_tok must be set on AuxLossConfig before build()."

        balancing_ctx = self.balancing_loss_cfg.build() if self.balancing_loss_cfg is not None else None
        z_ctx = self.z_loss_cfg.build() if self.z_loss_cfg is not None else None

        mask = data["seq_ctx"].mask
        if sp_mesh is not None and sp_mesh.size() > 1:
            mask = sp_split(mask, sp_mesh=sp_mesh, split_dim=1, padding_value=False)

        aux_ctx = AuxLossContext(self, balancing_ctx=balancing_ctx, z_ctx=z_ctx)
        aux_ctx.set_nonpad_list([torch.nonzero(mask, as_tuple=True)[1]])

        # Token-count denominators (SP-local count + its cross-rank total) are set here so the forward
        # runs no aux collective, mirroring CELossContext.build_batches. Only the sub-contexts consume
        # them, so a tokens-only hub skips this entirely.
        if balancing_ctx is not None or z_ctx is not None:
            num_tokens_local = int(mask.sum())
            if balancing_ctx is not None:
                balancing_ctx.set_non_pad_token(num_tokens_local)
            if z_ctx is not None:
                z_ctx.set_token_counts(
                    num_tokens_local, self._global_token_count(z_ctx, num_tokens_local, mask.device)
                )
        return aux_ctx

    @staticmethod
    def _global_token_count(
        z_ctx: ZLossContext,
        num_tokens_local: int,
        device: torch.device | str | int,
    ) -> torch.Tensor | None:
        # Cross-rank non-padding token count for z-loss global averaging. None when no process group is
        # initialized (single-process reference / eval, handled as world size 1).
        if not dist.is_initialized():
            return None
        n = torch.tensor(num_tokens_local, device=device, dtype=torch.int64)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        return n


class AuxLossContext(nn.Module):
    """Per-batch MoE auxiliary-loss hub.

    Owns everything the model needs to turn a layer's raw router outputs into aux-loss backward and
    routing statistics: the optional balancing / z-loss sub-contexts, the per-micro-batch non-padding
    indices, and the per-layer ``tokens_per_expert`` accumulator used by logging / bias update and by
    ``BalancingLossContext.finalize``. The model feeds one layer's raw router outputs for every
    micro-batch at once (as lists); padding removal, histogram accumulation, balancing / z dispatch,
    and z-loss injection all happen inside :meth:`accumulate`.

    One hub is built per micro-batch at ``build_loss_ctx_batch`` time, carrying that micro-batch's
    non-padding indices and (via the sub-contexts) token counts. Under ``intra_layer_micro_batch`` the
    per-micro-batch hubs are merged with :meth:`cat` into a single whole-batch hub whose
    ``nonpad_list`` holds the indices in micro-batch order, so :meth:`accumulate` aligns each
    micro-batch's nonpad with its router outputs by a single internal ``zip`` (no micro-batch index
    is threaded through the model).

    Args:
        loss_cfg (AuxLossConfig): Resolved aux config (``n_routed_experts`` / ``num_experts_per_tok``).
        balancing_ctx (BalancingLossContext | None): Balancing sub-context, or ``None`` when disabled.
        z_ctx (ZLossContext | None): Z-loss sub-context, or ``None`` when disabled.
    """

    def __init__(
        self,
        loss_cfg: AuxLossConfig,
        balancing_ctx: BalancingLossContext | None = None,
        z_ctx: ZLossContext | None = None,
    ):
        super().__init__()
        self.loss_cfg = loss_cfg
        n_routed_experts = self.loss_cfg.n_routed_experts
        num_experts_per_tok = self.loss_cfg.num_experts_per_tok
        assert n_routed_experts is not None, "n_routed_experts must be resolved before creating AuxLossContext."
        assert num_experts_per_tok is not None, "num_experts_per_tok must be resolved before creating AuxLossContext."
        self.n_routed_experts: int = n_routed_experts
        self.num_experts_per_tok: int = num_experts_per_tok
        self.balancing_ctx = balancing_ctx
        self.z_ctx = z_ctx
        # Non-padding indices (SP-local) per micro-batch, in micro-batch order. A per-micro-batch hub
        # built at build time holds a single-element list; cat() concatenates them into the whole-batch
        # order that accumulate() zips against its router-output lists.
        self.nonpad_list: list[torch.Tensor] = []
        # Per-layer tokens_per_expert accumulator keyed by dense MoE-layer ordinal. Same-layer
        # micro-batches add into the same slot (histogram counts are additive over the token pool).
        self._local_load_logits: dict[int, torch.Tensor] = {}

    @classmethod
    def cat(cls, chunks: list["AuxLossContext"]) -> "AuxLossContext":
        """Merge per-micro-batch aux hubs into one whole-batch hub.

        Concatenates each chunk's non-padding indices into ``nonpad_list`` (micro-batch order) and
        merges the balancing / z-loss sub-contexts (which sum their per-micro-batch token counts). The
        differentiable accumulators are empty at merge time -- accumulation runs per layer afterwards.

        Args:
            chunks (list[AuxLossContext]): Per-micro-batch hubs (identical config).

        Returns:
            AuxLossContext: A single whole-batch hub.
        """
        assert len(chunks) > 0, "chunks must not be empty."
        balancing_ctx = (
            BalancingLossContext.cat([cast(BalancingLossContext, c.balancing_ctx) for c in chunks])
            if chunks[0].balancing_ctx is not None
            else None
        )
        z_ctx = (
            ZLossContext.cat([cast(ZLossContext, c.z_ctx) for c in chunks]) if chunks[0].z_ctx is not None else None
        )
        merged = cls(chunks[0].loss_cfg, balancing_ctx=balancing_ctx, z_ctx=z_ctx)
        merged.set_nonpad_list([nonpad for chunk in chunks for nonpad in chunk.nonpad_list])
        return merged

    def set_nonpad_list(self, nonpad_list: list[torch.Tensor]) -> None:
        """Set the per-micro-batch non-padding indices (SP-local), in micro-
        batch order.

        Args:
            nonpad_list (list[torch.Tensor]): One 1-D index tensor per micro-batch, ordered to match
                the router-output lists passed to :meth:`accumulate`.
        """
        self.nonpad_list = nonpad_list

    def accumulate(
        self,
        *,
        layer_idx: int,
        router_weights_list: list[torch.Tensor],
        router_logits_list: list[torch.Tensor],
        hidden_states_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Accumulate one layer's router statistics across all micro-batches
        and inject z-loss into the main graph.

        For each micro-batch: strips padding with that micro-batch's nonpad indices, updates the
        per-layer ``tokens_per_expert`` and balancing accumulators (all micro-batches of this layer
        add into the same slot, so the result equals concatenating them and accumulating once), and
        injects this micro-batch's z-loss onto its own ``hidden_states`` carrier via
        :class:`AuxLossScaler`. Token counts were set on the sub-contexts at build time.

        Args:
            layer_idx (int): Dense MoE-layer ordinal (0-based); all micro-batches share this slot.
            router_weights_list (list[torch.Tensor]): Raw router weights per micro-batch, each shape
                ``(num_tokens, n_routed_experts)``.
            router_logits_list (list[torch.Tensor]): Raw router logits per micro-batch, same shapes.
            hidden_states_list (list[torch.Tensor]): Per-micro-batch carrier tensors on the main
                forward path. Each micro-batch's z-loss is attached to its own carrier via
                :class:`AuxLossScaler` so backward releases this layer's logsumexp saved tensor inline.

        Returns:
            list[torch.Tensor]: ``hidden_states_list`` with each entry augmented by the per-layer
            z-loss autograd hook (identical in value); the caller must replace its handles so the
            hooks are preserved on the main forward graph.
        """
        assert len(self.nonpad_list) == len(router_weights_list), (
            "nonpad_list and router_weights_list must have one entry per micro-batch."
        )
        for i, nonpad_indices in enumerate(self.nonpad_list):
            selected_router_weights = router_weights_list[i].index_select(0, nonpad_indices).contiguous().float()

            # tokens_per_expert is non-differentiable (topk + histc) and shared between logging output
            # and BalancingLossContext.finalize. Owned here as the single source of truth.
            _, selected_experts = torch.topk(selected_router_weights, self.num_experts_per_tok, dim=-1)
            tokens_per_expert_l = torch.histc(
                selected_experts.view(-1),
                bins=self.n_routed_experts,
                min=0,
                max=self.n_routed_experts,
            ).to(torch.long)
            prev = self._local_load_logits.get(layer_idx)
            self._local_load_logits[layer_idx] = tokens_per_expert_l if prev is None else prev + tokens_per_expert_l

            if self.balancing_ctx is not None:
                self.balancing_ctx.accumulate(layer_idx=layer_idx, router_weights=selected_router_weights)

            if self.z_ctx is not None:
                selected_router_logits = router_logits_list[i].index_select(0, nonpad_indices).contiguous().float()
                z_loss_l = self.z_ctx.accumulate(router_logits=selected_router_logits)
                hidden_states_list[i] = AuxLossScaler.apply(hidden_states_list[i], z_loss_l)

        return hidden_states_list

    def finalize(self) -> "AuxLossFinalizeOutput":
        """Finalize auxiliary losses and expert counts from the accumulated
        state.

        Returns:
            AuxLossFinalizeOutput: ``balancing_loss`` / ``z_loss`` carry the backward graph and the
            globally reduced ``tokens_per_expert_global``. Per-rank display values are produced by
            :meth:`calibrate`, not returned here.
        """
        tokens_per_expert_local, tokens_per_expert_global = self._cal_tokens_per_expert()

        balancing_loss: torch.Tensor | None = None
        if self.balancing_ctx is not None:
            balancing_loss = self.balancing_ctx.finalize(
                tokens_per_expert_local=tokens_per_expert_local,
                tokens_per_expert_global=tokens_per_expert_global,
                n_routed_experts=self.n_routed_experts,
                num_experts_per_tok=self.num_experts_per_tok,
            )

        z_loss: torch.Tensor | None = None
        if self.z_ctx is not None:
            z_loss = self.z_ctx.finalize()

        return AuxLossFinalizeOutput(
            balancing_loss=balancing_loss,
            z_loss=z_loss,
            tokens_per_expert_global=tokens_per_expert_global,
        )

    def calibrate(self) -> dict[str, torch.Tensor]:
        """Collect the auxiliary losses' detached display values for logging.

        Owns the aux side of the display pipeline so the model does not hand-assemble it.
        ``balancing_loss`` / ``z_loss`` are this rank's own values; ``balancing_loss_global`` is the
        cross-rank balancing loss, since expert balance is only meaningful over the whole token pool.

        Returns:
            dict[str, torch.Tensor]: Aux display values keyed by loss name (only present terms).
        """
        calibrated: dict[str, torch.Tensor] = {}
        if self.balancing_ctx is not None:
            calibrated["balancing_loss"] = self.balancing_ctx.calibrate()
            calibrated["balancing_loss_global"] = self.balancing_ctx.global_calibrate()
        if self.z_ctx is not None:
            calibrated["z_loss"] = self.z_ctx.calibrate()
        return calibrated

    def _cal_tokens_per_expert(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack per-layer expert counts and produce both local and globally
        reduced views.

        The local view is needed by BalancingLossContext's single-process branch (per-rank scaling); the global view is
        what the consumer (logging / bias update) wants.
        """
        local_load_logits = self._local_load_logits
        self._local_load_logits = {}

        if not local_load_logits:
            raise RuntimeError(
                "No MoE routing statistics were accumulated before finalize(). "
                "This usually means the model has no MoE layers or finalize() was called "
                "without a preceding accumulate()."
            )
        # Stack in ascending layer-ordinal order; consumers (bias update / maxvio) index rows by the
        # dense MoE-layer ordinal, so the row order must follow the ordinal, not insertion order.
        tokens_per_expert_local = torch.stack([local_load_logits[k] for k in sorted(local_load_logits)], dim=0)
        if dist.is_initialized():
            group = dist.group.WORLD
            assert group is not None
            tokens_per_expert_global = all_reduce(tokens_per_expert_local, "sum", group)
        else:
            tokens_per_expert_global = tokens_per_expert_local
        return tokens_per_expert_local, tokens_per_expert_global
