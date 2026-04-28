"""TorchAll2AllTPEPDispatcher: EP AlltoAll dispatcher with TP AllGather/ReduceScatter.

Forward data flow (adds two TP collectives around the existing EP dispatcher):

    dispatch_preprocess : permute by expert (each TP rank independently, N_local tokens)
    dispatch            : EP AlltoAll (each TP rank independently, routing N_local token copies)
    dispatch_postprocess: TP AllGather → merge TP slices into M_total tokens
                          then permute by local expert (for grouped GEMM)
    [Expert GEMM]       : each TP rank computes full expert output (redundant across TP)
    combine_preprocess  : unpermute back to TP-AllGather order
                          then TP ReduceScatterMean → restore M_ep_recv per TP rank
    combine             : EP AlltoAll reverse (each TP rank independently)
    combine_postprocess : unpermute with topk_weights → [N_local, H] per TP rank

Design rationale (mirrors Megatron MoEAlltoAllTokenDispatcher with TP+EP):
  - Expert weights are NOT sharded by TP; each TP rank holds a full copy.
  - TP AllGather before experts and TP ReduceScatterMean after experts form a symmetric pair
    that keeps the forward values numerically identical to the EP-only case.
  - ReduceScatterMean (avg reduce) is used so that the redundant expert outputs from all TP
    ranks reduce back to the original values without a TP-factor scaling in the forward pass.
  - The backward of ReduceScatterMean (AllGather) and AllGather backward (AllReduce+slice)
    introduce a 1/TP scaling in the gradient. This is a known design trade-off consistent
    with the Megatron approach; the model learns to compensate via weight initialisation.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from typing_extensions import override

from xtuner.v1.ops import permute, unpermute

from . import XTUNER_DISPATCHER_DEBUG
from .torch_all2all import (
    TorchAll2AllDispatcher,
    TorchAll2AllDispatchResult,
    TorchAll2AllPostDispatchResult,
    TorchAll2AllPreCombineResult,
    TorchAll2AllPreDispatchResult,
    get_backward_hook,
    get_backward_pre_hook,
)


class TorchAll2AllTPEPPostDispatchResult(TorchAll2AllPostDispatchResult):
    """Post-dispatch result for TP+EP dispatcher.

    Extends the EP-only result with per-TP-rank token counts needed to perform the
    TP ReduceScatterMean in ``combine_preprocess``.
    """

    output_splits_tp: list[int]


class _TPAllGather(torch.autograd.Function):
    """TP AllGather with autograd support.

    Forward : ``all_gather`` across the TP group, concatenating along the token dim.
    Backward: ``all_reduce`` (SUM) the gradient then slice — equivalent to a reduce-scatter
              sum in the unequal-size case.  This introduces a 1/TP factor relative to the
              mathematically exact gradient when computation is redundant across TP ranks,
              consistent with the Megatron redundant-TP-expert design.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        all_sizes: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        chunks = [torch.empty(s, hidden.shape[1], dtype=hidden.dtype, device=hidden.device) for s in all_sizes]
        dist.all_gather(chunks, hidden.contiguous(), group=tp_group)
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.all_sizes = all_sizes
        return torch.cat(chunks, dim=0)

    @staticmethod
    def backward(
        ctx: Any,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        offset = sum(ctx.all_sizes[: ctx.tp_rank])
        return grad[offset : offset + ctx.all_sizes[ctx.tp_rank]].clone(), None, None, None, None


class _TPReduceScatterMean(torch.autograd.Function):
    """TP ReduceScatterMean with autograd support.

    Forward : ``all_reduce`` (SUM) / TP_size then slice — equivalent to a mean reduce-scatter.
              When all TP ranks hold identical tensors (redundant expert computation), this
              returns the original un-scaled value for each rank's slice.
    Backward: ``all_gather`` the gradient slices to reconstruct the full gradient tensor,
              then divide by TP_size (chain rule through the /TP_size division).
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        all_sizes: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        hidden = hidden.clone()
        dist.all_reduce(hidden, op=dist.ReduceOp.SUM, group=tp_group)
        hidden = hidden / tp_size
        offset = sum(all_sizes[:tp_rank])
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.all_sizes = all_sizes
        return hidden[offset : offset + all_sizes[tp_rank]].contiguous()

    @staticmethod
    def backward(
        ctx: Any,
        grad_slice: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        chunks = [
            torch.empty(s, grad_slice.shape[1], dtype=grad_slice.dtype, device=grad_slice.device)
            for s in ctx.all_sizes
        ]
        dist.all_gather(chunks, grad_slice.contiguous(), group=ctx.tp_group)
        full_grad = torch.cat(chunks, dim=0) / ctx.tp_size
        return full_grad, None, None, None, None


def _tp_all_gather(
    hidden: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[int]]:
    """All-gather ``hidden`` across the TP group and return the gathered tensor
    plus per-rank sizes."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return hidden, [hidden.shape[0]]

    tp_rank = dist.get_rank(group=tp_group)
    local_size = hidden.new_tensor([hidden.shape[0]], dtype=torch.long)
    all_sizes_t = hidden.new_empty([tp_size], dtype=torch.long)
    dist.all_gather_into_tensor(all_sizes_t, local_size, group=tp_group)
    all_sizes = [int(s) for s in all_sizes_t.tolist()]

    gathered = _TPAllGather.apply(hidden, all_sizes, tp_group, tp_size, tp_rank)
    return gathered, all_sizes


def _tp_reduce_scatter_mean(
    hidden: torch.Tensor,
    all_sizes: list[int],
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Mean-reduce-scatter ``hidden`` across the TP group, returning this
    rank's slice."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return hidden

    tp_rank = dist.get_rank(group=tp_group)
    return _TPReduceScatterMean.apply(hidden, all_sizes, tp_group, tp_size, tp_rank)


def _tp_all_gather_tokens_per_expert_group(
    tokens_per_expert_group: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather per-TP expert counts in the same TP-rank order as
    ``_tp_all_gather``."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return tokens_per_expert_group.unsqueeze(0)

    gathered = tokens_per_expert_group.new_empty((tp_size, *tokens_per_expert_group.shape))
    dist.all_gather_into_tensor(gathered, tokens_per_expert_group.contiguous(), group=tp_group)
    return gathered


class TorchAll2AllTPEPDispatcher(TorchAll2AllDispatcher):
    """TP+EP dispatcher: wraps ``TorchAll2AllDispatcher`` with TP AllGather and
    ReduceScatterMean.

    Overrides only ``dispatch_postprocess`` and ``combine_preprocess``; all other steps
    (dispatch_preprocess, dispatch, combine, combine_postprocess) are unchanged from the
    EP-only base class.

    Args:
        n_routed_experts (int): Total number of routed experts across all EP ranks.
        ep_group (dist.ProcessGroup): Expert parallel process group.
        tp_group (dist.ProcessGroup): Tensor parallel process group.
        training_dtype (str): Dtype for training, ``"bf16"`` or ``"fp8"``.
        generate_dtype (str): Dtype for generation, ``"bf16"`` or ``"fp8"``.
    """

    def __init__(
        self,
        *,
        n_routed_experts: int,
        ep_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ) -> None:
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=ep_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        self._tp_group = tp_group
        self._tp_size = tp_group.size()

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllTPEPPostDispatchResult:
        if async_op:
            # async_op for TP collectives is not yet implemented; fall back to synchronous.
            assert dispatched["forward_finished_event"] is not None, "Use async_op=True for dispatch!"
            self.wait_comm_stream(dispatched["forward_finished_event"])

        # TP AllGather: [M_ep_recv, H] → [M_total, H]; also returns per-TP-rank sizes.
        gathered_hidden, output_splits_tp = _tp_all_gather(
            dispatched["hidden_states"],
            tp_group=self._tp_group,
        )

        # Permute [M_total, H] into local-expert order for grouped GEMM.  Since
        # TP AllGather concatenates tp0_block | tp1_block | ..., expert counts
        # must be gathered in the same TP order before building the row labels.
        gathered_tokens_per_expert_group = _tp_all_gather_tokens_per_expert_group(
            dispatched["tokens_per_expert_group"],
            tp_group=self._tp_group,
        )
        token_counts = gathered_tokens_per_expert_group.ravel()
        local_expert_ids = self._expert_ids_per_ep_rank.repeat(self._tp_size)
        global_input_tokens_local_experts_indices = torch.repeat_interleave(
            local_expert_ids,
            token_counts,
            output_size=gathered_hidden.shape[0],
        )
        global_input_tokens, row_ids_map = permute(
            gathered_hidden,
            global_input_tokens_local_experts_indices.to(torch.int32),
        )
        tokens_per_expert = gathered_tokens_per_expert_group.sum(dim=(0, 1))

        if async_op:
            assert dispatched["backward_previous_event"] is not None, "Use async_op=True for dispatch!"
            if global_input_tokens.grad_fn is not None:
                global_input_tokens.grad_fn.register_hook(
                    get_backward_hook(
                        dispatched["backward_previous_event"],
                        name="TorchAll2AllTPEPDispatcher.dispatch_postprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        if decoding:
            raise NotImplementedError("Decoding is not yet supported for TorchAll2AllTPEPDispatcher.")

        return TorchAll2AllTPEPPostDispatchResult(
            hidden_states=global_input_tokens,
            row_ids_map=row_ids_map,
            tokens_per_expert=tokens_per_expert,
            output_splits_tp=output_splits_tp,
        )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        post_dispatched: TorchAll2AllPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllPreCombineResult:
        tpep_post = cast(TorchAll2AllTPEPPostDispatchResult, post_dispatched)
        # Unpermute [M_total, H] back to TP-AllGather order (tp0_block | tp1_block | ...).
        hidden_states = unpermute(hidden_states, tpep_post["row_ids_map"])

        # TP ReduceScatterMean: [M_total, H] → [M_ep_recv, H] for this TP rank.
        hidden_states = _tp_reduce_scatter_mean(
            hidden_states,
            all_sizes=tpep_post["output_splits_tp"],
            tp_group=self._tp_group,
        )

        if async_op:
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllTPEPDispatcher.combine_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            backward_previous_event = None
            forward_finished_event = None

        if decoding:
            raise NotImplementedError("Decoding is not yet supported for TorchAll2AllTPEPDispatcher.")

        return TorchAll2AllPreCombineResult(
            hidden_states=hidden_states,
            backward_previous_event=backward_previous_event,
            forward_finished_event=forward_finished_event,
        )
