from typing import Annotated, Literal, cast

import torch
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist

from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class BalancingLossConfig(BaseModel):
    """Balancing loss configuration for MoE models.

    Args:
        balancing_loss_alpha (float): Weight for the balancing loss. Defaults to 0.001.
        router_scoring_func (str): Router scoring function type. Options are "sigmoid" and "softmax".
            Defaults to "softmax".
    """

    model_config = ConfigDict(extra="forbid")
    balancing_loss_alpha: Annotated[float, Parameter(help="weight for balancing loss")] = 0.001
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
        # Per-layer differentiable accumulator keyed by dense MoE-layer ordinal. Under
        # intra_layer_micro_batch a layer is accumulated once per micro-batch; the per-micro-batch
        # router-weight sums add into the same slot (the sum is linear over the token pool), so the
        # result equals concatenating the micro-batches and accumulating once. tokens_per_expert is
        # owned by AuxLossContext and passed in at finalize() time to avoid duplicate storage /
        # duplicate all_reduce.
        self.routing_weights_sum: dict[int, torch.Tensor] = {}
        # Detached display values set by finalize(). `_calibrated` is this rank's local balancing loss
        # (calibrate()); `_calibrated_global` is the cross-rank balancing loss over the whole token
        # pool (global_calibrate()), the semantically meaningful load-balance signal.
        self._calibrated: torch.Tensor | None = None
        self._calibrated_global: torch.Tensor | None = None
        # This rank's non-padding token count, set at build time by set_non_pad_token()
        # (see MoE._build_aux_ctx) and summed across micro-batches in cat().
        self._non_pad_token: int = 1

    def set_non_pad_token(self, non_pad_token: int) -> None:
        """Set this rank's non-padding token count for the current batch.

        Args:
            non_pad_token (int): Non-padding token count on this rank.
        """
        self._non_pad_token = non_pad_token

    @staticmethod
    def cat(chunks: list["BalancingLossContext"]) -> "BalancingLossContext":
        """Merge per-micro-batch balancing contexts into one whole-batch
        context.

        Balancing loss is a whole-batch quantity (its expert-load statistics are bilinear in the
        token pool, so a per-micro-batch sum is not the batch value). Under
        ``intra_layer_micro_batch`` the micro-batches are concatenated into one forward and their
        router statistics are fed to a single context. The differentiable accumulators are still empty
        at merge time -- accumulation runs per layer afterwards -- but the build-time non-padding token
        count is per-micro-batch, so it is summed here to recover the whole-batch count.

        Args:
            chunks (list[BalancingLossContext]): Per-micro-batch contexts (identical config).

        Returns:
            BalancingLossContext: A single whole-batch context.
        """
        assert len(chunks) > 0, "chunks must not be empty."
        merged = BalancingLossContext(chunks[0].loss_cfg, BalancingLossKwargs())
        merged._non_pad_token = sum(chunk._non_pad_token for chunk in chunks)
        return merged

    def accumulate(
        self,
        *,
        layer_idx: int,
        router_weights: torch.Tensor,
    ) -> None:
        """Update the per-layer differentiable accumulator for balancing loss.

        Args:
            layer_idx (int): Dense MoE-layer ordinal (0-based). Same-layer micro-batches add into the
                same slot, so the accumulated sum matches concatenating them and accumulating once.
            router_weights (torch.Tensor): Router weights with non-padding tokens already selected.
                Shape: ``(non_pad, n_routed_experts)``.
        """
        # router_weights.sum(dim=0) is [n_routed_experts]; sum's backward does not save the input
        # tensor, so the [non_pad, n_routed_experts] activation is not pinned by this accumulator.
        layer_sum = router_weights.sum(dim=0)
        prev = self.routing_weights_sum.get(layer_idx)
        self.routing_weights_sum[layer_idx] = layer_sum if prev is None else prev + layer_sum

    def finalize(
        self,
        *,
        tokens_per_expert_local: torch.Tensor,
        tokens_per_expert_global: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Finalize balancing loss from accumulators.

        The non-padding token count was set at build time via ``set_non_pad_token()``.

        Args:
            tokens_per_expert_local (torch.Tensor): Per-layer expert token counts on this rank,
                ``(num_layers, n_routed_experts)``. Used by the single-process branch.
            tokens_per_expert_global (torch.Tensor): All-reduced ``tokens_per_expert``,
                ``(num_layers, n_routed_experts)``. Used by the global-average branch.
            n_routed_experts (int): Number of routed experts.
            num_experts_per_tok (int): Number of experts selected per token.

        Returns:
            torch.Tensor: This rank's balancing loss carrying the autograd graph for backward. Under
            reduce-sum it is computed from this rank's own ``local_gating_sum`` with global detached
            statistics; cross-rank aggregation happens on the gradients (FSDP / scale_and_reduce_grad
            SUM), so summing over ranks reproduces the global balancing loss. The per-rank display
            value is computed separately by ``calibrate()`` from local statistics.
        """
        routing_weights_sum = self.routing_weights_sum
        self.routing_weights_sum = {}
        if self.loss_cfg.balancing_loss_alpha == 0 or not routing_weights_sum:
            zero = torch.tensor(0.0, device=tokens_per_expert_local.device, dtype=torch.float32)
            self._calibrated = zero
            self._calibrated_global = zero
            return zero

        # Stack in ascending layer-ordinal order so the rows align with tokens_per_expert_local /
        # tokens_per_expert_global (both keyed by the same dense MoE-layer ordinal).
        local_gating_sum = torch.stack([routing_weights_sum[k] for k in sorted(routing_weights_sum)], dim=0)
        alpha = self.loss_cfg.balancing_loss_alpha

        if dist.is_initialized():
            tokens_global = tokens_per_expert_global.sum(-1)
            seqlen_global = tokens_global // num_experts_per_tok
            scale_global = n_routed_experts / tokens_global
            routing_weights_mean = local_gating_sum / seqlen_global.unsqueeze(-1)
            loss_vec = scale_global * (tokens_per_expert_global * routing_weights_mean).sum(-1)
        else:
            # Single-process path (no process group). Numerically identical to the distributed branch
            # at world size 1: tokens_per_expert_global == tokens_per_expert_local, tokens_global ==
            # valid_tokens * num_experts_per_tok, and seqlen_global == valid_tokens, so scale / mean /
            # loss match term for term. Kept so reference / eval (dist uninitialized) still works.
            valid_tokens = max(self._non_pad_token, 1)
            scale_global = n_routed_experts / (valid_tokens * num_experts_per_tok)
            routing_weights_mean = local_gating_sum / valid_tokens
            loss_vec = scale_global * (tokens_per_expert_local * routing_weights_mean).sum(-1)

        loss = loss_vec.sum() * alpha
        # Display values (detached). `_calibrated`: this rank's balancing loss from LOCAL statistics
        # (its own tokens_per_expert / seqlen), a readable per-rank number. `_calibrated_global`: the
        # balancing loss over the whole cross-rank token pool (one all_reduce of the gating sum), which
        # is the load-balance signal that actually matters -- a per-rank number cannot show whether
        # experts are balanced across EP ranks. Both are display-only, NOT backward modes; they do not
        # reintroduce the removed non-global averaging.
        self._calibrated = self._local_balancing_loss(
            local_gating_sum, tokens_per_expert_local, n_routed_experts, num_experts_per_tok
        )
        self._calibrated_global = self._global_balancing_loss(
            local_gating_sum, tokens_per_expert_global, n_routed_experts, num_experts_per_tok
        )
        return loss

    def _local_balancing_loss(
        self,
        local_gating_sum: torch.Tensor,
        tokens_per_expert_local: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        valid_tokens = max(self._non_pad_token, 1)
        scale_local = n_routed_experts / (valid_tokens * num_experts_per_tok)
        routing_weights_mean_local = local_gating_sum.detach() / valid_tokens
        loss = scale_local * (tokens_per_expert_local * routing_weights_mean_local).sum(-1)
        return (loss.sum() * self.loss_cfg.balancing_loss_alpha).detach()

    def calibrate(self) -> torch.Tensor:
        """This rank's balancing loss (from local statistics) for display
        (detached, no all_reduce).

        Returns:
            torch.Tensor: This rank's local balancing loss.
        """
        assert self._calibrated is not None, "finalize() must be called before calibrate()"
        return self._calibrated

    def global_calibrate(self) -> torch.Tensor:
        """The balancing loss over the whole cross-rank token pool for display
        (detached).

        Reflects whether experts are balanced across all ranks, which the per-rank ``calibrate()``
        value cannot show. Costs one all_reduce of the gating sum, done at ``finalize()`` time.

        Returns:
            torch.Tensor: The global balancing loss.
        """
        assert self._calibrated_global is not None, "finalize() must be called before global_calibrate()"
        return self._calibrated_global

    def _global_balancing_loss(
        self,
        local_gating_sum: torch.Tensor,
        tokens_per_expert_global: torch.Tensor,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        # Mirror finalize()'s global-average branch but on the ALL-REDUCED gating sum, so the value is
        # the true global balancing loss rather than this rank's backward component.
        global_gating_sum = local_gating_sum.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(global_gating_sum, op=dist.ReduceOp.SUM)
        tokens_global = tokens_per_expert_global.sum(-1).clamp_min(1)
        seqlen_global = (tokens_global // num_experts_per_tok).clamp_min(1)
        scale_global = n_routed_experts / tokens_global
        routing_weights_mean = global_gating_sum / seqlen_global.unsqueeze(-1)
        loss = scale_global * (tokens_per_expert_global * routing_weights_mean).sum(-1)
        return (loss.sum() * self.loss_cfg.balancing_loss_alpha).detach()


class ZLossConfig(BaseModel):
    """Z-loss configuration for MoE models.

    Args:
        z_loss_alpha (float): Weight for the z-loss. Defaults to 0.001.
    """

    model_config = ConfigDict(extra="forbid")
    z_loss_alpha: Annotated[float, Parameter(help="weight for z-loss")] = 0.001

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
        # Z-loss is backward-only: the differentiable per-layer scalar is injected into the main graph
        # via AuxLossScaler at accumulate() time (memory-saving pattern from Megatron's
        # MoEAuxLossAutoScaler). For display we self-maintain a single detached running scalar holding
        # THIS RANK's z-loss mean (the raw `logsumexp^2` mean per layer, without the reduce-sum
        # `num_tokens_local/denom_global` factor), i.e. a clean world-size-independent per-rank value.
        self._running_loss_for_log: torch.Tensor | None = None
        self._calibrated: torch.Tensor | None = None
        # Token counts set at build time by set_token_counts() (see MoE._build_aux_ctx) and summed
        # across micro-batches in cat(). Used as the z-loss normalization denominators.
        self._num_tokens_local: int = 1
        self._num_tokens_global: torch.Tensor | None = None

    @staticmethod
    def cat(chunks: list["ZLossContext"]) -> "ZLossContext":
        """Merge per-micro-batch z-loss contexts into one whole-batch context.

        Mirrors :meth:`BalancingLossContext.cat`: under ``intra_layer_micro_batch`` the concatenated
        router logits are fed to a single context, so the per-micro-batch contexts collapse to one.
        The differentiable accumulators are empty at merge time, but the build-time token counts are
        per-micro-batch, so both the local count and the (linear) cross-rank count are summed here to
        recover the whole-batch denominators.

        Args:
            chunks (list[ZLossContext]): Per-micro-batch contexts (identical config).

        Returns:
            ZLossContext: A single whole-batch context.
        """
        assert len(chunks) > 0, "chunks must not be empty."
        merged = ZLossContext(chunks[0].loss_cfg, ZLossKwargs())
        merged._num_tokens_local = sum(chunk._num_tokens_local for chunk in chunks)
        # num_tokens_global is None iff no process group is initialized -- identical across chunks
        # (shared config). all_reduce is linear, so summing per-chunk globals == all_reduce of the
        # summed local counts, i.e. the whole-batch global token total.
        if chunks[0]._num_tokens_global is None:
            merged._num_tokens_global = None
        else:
            merged._num_tokens_global = torch.stack(
                [cast(torch.Tensor, chunk._num_tokens_global) for chunk in chunks]
            ).sum()
        return merged

    def set_token_counts(self, num_tokens_local: int, num_tokens_global: torch.Tensor | None) -> None:
        """Set this rank's non-padding token counts (local and cross-rank) for
        the current batch.

        Args:
            num_tokens_local (int): Non-padding token count on this rank.
            num_tokens_global (torch.Tensor | None): All-reduced non-padding token count across ranks
                (int64 scalar), or ``None`` when no process group is initialized (single-process
                reference / eval).
        """
        self._num_tokens_local = num_tokens_local
        self._num_tokens_global = num_tokens_global

    def accumulate(
        self,
        *,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute z-loss for one layer and return it as a scalar with autograd
        attached.

        The caller is expected to inject the returned scalar back into the main forward graph
        via :class:`AuxLossScaler`; the autograd graph behind this scalar (logsumexp -> square ->
        sum -> scaling) is then released as soon as the corresponding layer is reached during
        backward, instead of being pinned until a global ``finalize()``. Token counts were set at
        build time via ``set_token_counts()``.

        Args:
            router_logits (torch.Tensor): Router logits with non-padding tokens already selected.
                Shape ``(non_pad, n_routed_experts)``.

        Returns:
            torch.Tensor: Per-layer z-loss as a 0-d tensor with autograd graph back to
            ``router_logits``.
        """
        num_tokens_local = self._num_tokens_local
        num_tokens_global = self._num_tokens_global
        if self.loss_cfg.z_loss_alpha == 0:
            zero = torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)
            self._update_running(zero)
            return zero

        denom_local = max(num_tokens_local, 1)
        base = torch.logsumexp(router_logits, dim=-1).square().sum() / denom_local

        # Single-process (dist uninitialized): num_tokens_global is None, so the loss stays `base`,
        # which equals the distributed branch at world size 1 (denom_global == num_tokens_local).
        loss = base
        if num_tokens_global is not None:
            # Under reduce-sum gradients the injected z-loss stays as this rank's local component
            # (its share of the global z-loss, WITHOUT any `× world_size`). Cross-rank aggregation
            # happens on the gradients via the FSDP SUM reduce-scatter; summing this local component
            # over ranks reproduces the global z-loss.
            denom_global = torch.clamp(num_tokens_global, min=1)
            loss = base * num_tokens_local / denom_global

        loss = loss * self.loss_cfg.z_loss_alpha
        # Display value: this rank's raw z-loss mean (drop the `num_tokens_local/denom_global`
        # reduce-sum factor), which is world-size independent and readable per rank.
        self._update_running((base * self.loss_cfg.z_loss_alpha).detach())
        return loss

    def finalize(self) -> torch.Tensor:
        """Return this rank's accumulated z-loss mean as a detached scalar for
        the output field.

        The differentiable contribution was injected into the main graph at each ``accumulate()``
        call, so this value carries no autograd graph. It is this rank's own z-loss mean (no
        cross-rank all_reduce); ``calibrate()`` returns the same value for the display pipeline.
        """
        value = self._running_loss_for_log
        self._running_loss_for_log = None
        self._calibrated = value if value is not None else torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        return self._calibrated

    def calibrate(self) -> torch.Tensor:
        """This rank's z-loss mean for display (detached, no cross-rank
        all_reduce)."""
        assert self._calibrated is not None, "finalize() must be called before calibrate()"
        return self._calibrated

    def _update_running(self, value: torch.Tensor) -> None:
        if self._running_loss_for_log is None:
            self._running_loss_for_log = value.clone()
        else:
            self._running_loss_for_log = self._running_loss_for_log + value
