"""TorchAll2AllTPEPDispatcher: EP AlltoAll dispatcher with TP AllGather/ReduceScatter.

Forward data flow (adds two TP collectives around the existing EP dispatcher):

    dispatch_preprocess : permute by expert (each TP rank independently, N_local tokens)
    dispatch            : EP AlltoAll → TP AllGather, merging TP token slices into M_total tokens
    dispatch_postprocess: permute by local expert (for grouped GEMM)
    [Expert GEMM]       : column-parallel gate/up + row-parallel down projection
    combine_preprocess  : unpermute back to TP-AllGather order
    combine             : TP ReduceScatterRowsSum → EP AlltoAll reverse
    combine_postprocess : unpermute with topk_weights → [N_local, H] per TP rank

Design rationale (mirrors Megatron MoEAlltoAllTokenDispatcher with TP+EP):
  - Expert weights are sharded by TP: gate/up use column parallelism, down uses row
    parallelism.
  - TP AllGather before experts gives every TP rank the same token batch for its local
    expert weight shard.
  - TP ReduceScatterRowsSum after the row-parallel down projection sums partial hidden states
    across TP ranks, then returns each rank's original token slice.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from typing_extensions import override

from xtuner.v1.ops import permute, unpermute

from . import XTUNER_DISPATCHER_DEBUG
from .torch_all2all import (
    TorchAll2AllCombineResult,
    TorchAll2AllDispatcher,
    TorchAll2AllDispatchResult,
    TorchAll2AllPostDispatchResult,
    TorchAll2AllPreCombineResult,
    TorchAll2AllPreDispatchResult,
    get_backward_hook,
    get_backward_pre_hook,
)


class TorchAll2AllTPEPDispatchResult(TorchAll2AllDispatchResult):
    """Dispatch result after EP AlltoAll and TP AllGather.

    ``tp_rank_row_counts`` records the pre-AllGather token count per TP rank.  The
    later combine phase uses it to restore this TP rank's slice after the
    row-parallel expert output is summed.

    中文注释：``tp_rank_row_counts`` 是每个 TP rank 在 AllGather 前的行数。例如 ``tp_size=2``，
    EP dispatch 后 TP rank0 的 hidden 是 ``[3, H]``，rank1 是 ``[5, H]``，
    两个 rank 都会拿到 ``tp_rank_row_counts=[3, 5]``。TP AllGather 用它把
    变长 hidden 拼成 ``[8, H]``，combine 再按相同边界切回本 rank 的
    ``[3, H]`` 或 ``[5, H]``。
    """


class TorchAll2AllTPEPPostDispatchResult(TorchAll2AllPostDispatchResult): ...


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


def _tp_all_gather_rows_forward_impl(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run TP AllGather forward and return tensors whose lifetime may need
    recording."""
    hidden = hidden.contiguous()
    chunks = [torch.empty(s, hidden.shape[1], dtype=hidden.dtype, device=hidden.device) for s in tp_rank_row_counts]
    dist.all_gather(chunks, hidden, group=tp_group)
    return torch.cat(chunks, dim=0), hidden, chunks


def _tp_all_gather_rows_backward_impl(
    grad: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_rank: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    return _tp_reduce_scatter_rows_sum_impl(grad, tp_rank_row_counts, tp_rank, tp_group)


def _tp_reduce_scatter_rows_sum_impl(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_rank: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run TP ReduceScatterRowsSum and return tensors whose lifetime may need
    recording."""
    hidden = hidden.contiguous()
    assert hidden.shape[0] == sum(tp_rank_row_counts), (
        "TP ReduceScatterRowsSum input rows must match tp_rank_row_counts."
    )

    out = hidden.new_empty((tp_rank_row_counts[tp_rank], *hidden.shape[1:]))
    if hidden.shape[0] == 0:
        # 中文注释：所有 TP rank 都没有 token 时没有实际通信量，直接返回合法的 0 行 slice。
        return out, hidden, []

    if all(size == tp_rank_row_counts[0] for size in tp_rank_row_counts):
        dist.reduce_scatter_tensor(out, hidden, op=dist.ReduceOp.SUM, group=tp_group)
        return out, hidden, []

    input_chunks = list(torch.split(hidden, tp_rank_row_counts, dim=0))
    dist.reduce_scatter(out, input_chunks, op=dist.ReduceOp.SUM, group=tp_group)
    return out, hidden, input_chunks


def _tp_reduce_scatter_rows_sum_forward_impl(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_rank: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    return _tp_reduce_scatter_rows_sum_impl(hidden, tp_rank_row_counts, tp_rank, tp_group)


def _tp_reduce_scatter_rows_sum_backward_impl(
    grad_slice: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    grad_slice = grad_slice.contiguous()
    chunks = [
        torch.empty(s, grad_slice.shape[1], dtype=grad_slice.dtype, device=grad_slice.device)
        for s in tp_rank_row_counts
    ]
    dist.all_gather(chunks, grad_slice, group=tp_group)
    return torch.cat(chunks, dim=0), grad_slice, chunks


class _TPAllGatherRows(torch.autograd.Function):
    """TP AllGather with autograd support.

    Forward : ``all_gather`` across the TP group, concatenating along the token dim.
    Backward: ``reduce_scatter`` (SUM) the gradient into the original local token slice.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        gathered, _, _ = _tp_all_gather_rows_forward_impl(hidden, tp_rank_row_counts, tp_group)
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.tp_rank_row_counts = tp_rank_row_counts
        return gathered

    @staticmethod
    def backward(
        ctx: Any,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        grad_input, _, _ = _tp_all_gather_rows_backward_impl(grad, ctx.tp_rank_row_counts, ctx.tp_rank, ctx.tp_group)
        return grad_input, None, None, None, None


class _AsyncTPAllGatherRows(torch.autograd.Function):
    """TP AllGather on dispatcher comm stream.

    Forward : wait for the previous event, then all-gather token slices.
    Backward: wait until post-dispatch grad is ready, then reduce-scatter grad
              into this TP rank's input slice.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            gathered, hidden_for_comm, chunks = _tp_all_gather_rows_forward_impl(hidden, tp_rank_row_counts, tp_group)

            # 中文注释：同步/异步共用 TP AllGather 核心逻辑；
            # 异步只额外管理 stream/event 生命周期。
            _record_stream((hidden_for_comm, chunks, gathered), comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return gathered

    @staticmethod
    def backward(
        ctx: Any,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None, None]:
        with torch.cuda.stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            grad_input, grad_for_comm, chunks = _tp_all_gather_rows_backward_impl(
                grad,
                ctx.tp_rank_row_counts,
                ctx.tp_rank,
                ctx.tp_group,
            )

            _record_stream((grad_for_comm, chunks, grad_input), ctx.comm_stream)
            ctx.backward_finished_event.record(ctx.comm_stream)

        return grad_input, None, None, None, None, None, None, None, None, None


class _TPReduceScatterRowsSum(torch.autograd.Function):
    """TP ReduceScatterRowsSum with autograd support.

    Forward : ``reduce_scatter`` (SUM) to this TP rank's local token slice.
    Backward: ``all_gather`` the gradient slices to reconstruct the full gradient tensor,
              matching the sum reduction in the forward pass.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        out, _, _ = _tp_reduce_scatter_rows_sum_forward_impl(hidden, tp_rank_row_counts, tp_rank, tp_group)
        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.tp_rank_row_counts = tp_rank_row_counts
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_slice: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        full_grad, _, _ = _tp_reduce_scatter_rows_sum_backward_impl(grad_slice, ctx.tp_rank_row_counts, ctx.tp_group)
        return full_grad, None, None, None, None


class _AsyncTPReduceScatterRowsSum(torch.autograd.Function):
    """TP ReduceScatterRowsSum on dispatcher comm stream."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            out, hidden_for_comm, chunks = _tp_reduce_scatter_rows_sum_forward_impl(
                hidden,
                tp_rank_row_counts,
                tp_rank,
                tp_group,
            )

            # 中文注释：同步/异步共用 TP ReduceScatter 核心逻辑；异步只额外管理 stream/event。
            _record_stream((hidden_for_comm, chunks, out), comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.tp_group = tp_group
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_slice: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None, None]:
        with torch.cuda.stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            full_grad, grad_slice_for_comm, chunks = _tp_reduce_scatter_rows_sum_backward_impl(
                grad_slice,
                ctx.tp_rank_row_counts,
                ctx.tp_group,
            )

            _record_stream((grad_slice_for_comm, chunks, full_grad), ctx.comm_stream)
            ctx.backward_finished_event.record(ctx.comm_stream)

        return full_grad, None, None, None, None, None, None, None, None, None


def _tp_gather_tp_rank_row_counts(
    hidden: torch.Tensor,
    tp_group: dist.ProcessGroup,
    stream: torch.cuda.Stream | None = None,
) -> list[int]:
    """Gather per-TP-rank token counts as host ints for variable-size
    gather."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return [hidden.shape[0]]

    if stream is None:
        local_size = hidden.new_tensor([hidden.shape[0]], dtype=torch.long)
        tp_rank_row_counts_t = hidden.new_empty([tp_size], dtype=torch.long)
        dist.all_gather_into_tensor(tp_rank_row_counts_t, local_size, group=tp_group)
    else:
        # 中文注释：尺寸通信不依赖计算流，避免为了取 Python list 等待前面的 compute kernel。
        with torch.cuda.stream(stream):
            local_size = hidden.new_tensor([hidden.shape[0]], dtype=torch.long)
            tp_rank_row_counts_t = hidden.new_empty([tp_size], dtype=torch.long)
            dist.all_gather_into_tensor(tp_rank_row_counts_t, local_size, group=tp_group)
            local_size.record_stream(stream)
            tp_rank_row_counts_t.record_stream(stream)
        stream.synchronize()
    return [int(s) for s in tp_rank_row_counts_t.tolist()]


def _tp_all_gather_rows(
    hidden: torch.Tensor,
    tp_group: dist.ProcessGroup,
    tp_rank_row_counts: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """All-gather ``hidden`` across the TP group and return the gathered tensor
    plus per-rank sizes."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return hidden, [hidden.shape[0]]

    tp_rank = dist.get_rank(group=tp_group)
    if tp_rank_row_counts is None:
        tp_rank_row_counts = _tp_gather_tp_rank_row_counts(hidden, tp_group)

    gathered = _TPAllGatherRows.apply(hidden, tp_rank_row_counts, tp_group, tp_size, tp_rank)
    return gathered, tp_rank_row_counts


def _async_tp_all_gather_rows(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
    forward_previous_event: torch.cuda.Event,
    forward_finished_event: torch.cuda.Event,
    backward_previous_event: torch.cuda.Event,
    backward_finished_event: torch.cuda.Event,
    comm_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Async TP AllGather wrapper used by Domino TP+EP path."""
    tp_size = tp_group.size()
    if tp_size == 1:
        forward_finished_event.record()
        return hidden

    tp_rank = dist.get_rank(group=tp_group)
    return _AsyncTPAllGatherRows.apply(
        hidden,
        tp_rank_row_counts,
        tp_group,
        tp_size,
        tp_rank,
        forward_previous_event,
        forward_finished_event,
        backward_previous_event,
        backward_finished_event,
        comm_stream,
    )


def _tp_reduce_scatter_rows_sum(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Sum-reduce-scatter ``hidden`` across the TP group, returning this rank's
    slice."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return hidden

    tp_rank = dist.get_rank(group=tp_group)
    return _TPReduceScatterRowsSum.apply(hidden, tp_rank_row_counts, tp_group, tp_size, tp_rank)


def _async_tp_reduce_scatter_rows_sum(
    hidden: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
    forward_previous_event: torch.cuda.Event,
    forward_finished_event: torch.cuda.Event,
    backward_previous_event: torch.cuda.Event,
    backward_finished_event: torch.cuda.Event,
    comm_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Async TP ReduceScatterRowsSum wrapper used by Domino TP+EP path."""
    tp_size = tp_group.size()
    if tp_size == 1:
        forward_finished_event.record()
        return hidden

    tp_rank = dist.get_rank(group=tp_group)
    return _AsyncTPReduceScatterRowsSum.apply(
        hidden,
        tp_rank_row_counts,
        tp_group,
        tp_size,
        tp_rank,
        forward_previous_event,
        forward_finished_event,
        backward_previous_event,
        backward_finished_event,
        comm_stream,
    )


def _tp_all_gather_per_rank_metadata(
    tokens_per_expert_group: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather per-TP expert counts in the same TP-rank order as
    ``_tp_all_gather_rows``."""
    tp_size = tp_group.size()
    if tp_size == 1:
        return tokens_per_expert_group.unsqueeze(0)

    gathered = tokens_per_expert_group.new_empty((tp_size, *tokens_per_expert_group.shape))
    dist.all_gather_into_tensor(gathered, tokens_per_expert_group.contiguous(), group=tp_group)
    return gathered


def _async_tp_all_gather_per_rank_metadata(
    tokens_per_expert_group: torch.Tensor,
    tp_group: dist.ProcessGroup,
    forward_previous_event: torch.cuda.Event,
    forward_finished_event: torch.cuda.Event,
    comm_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Async gather for routing counts; no autograd is needed for these
    counts."""
    tp_size = tp_group.size()
    if tp_size == 1:
        forward_finished_event.record()
        return tokens_per_expert_group.unsqueeze(0)

    gathered = tokens_per_expert_group.new_empty((tp_size, *tokens_per_expert_group.shape))
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(forward_previous_event)
        counts = tokens_per_expert_group.contiguous()
        dist.all_gather_into_tensor(gathered, counts, group=tp_group)
        counts.record_stream(comm_stream)
        gathered.record_stream(comm_stream)
        forward_finished_event.record(comm_stream)
    return gathered


class TorchAll2AllTPEPDispatcher(TorchAll2AllDispatcher):
    """TP+EP dispatcher: wraps ``TorchAll2AllDispatcher`` with TP AllGather and
    ReduceScatterRowsSum.

    Keeps ``dispatch_preprocess`` and ``combine_postprocess`` from the EP-only
    base class, and moves the TP collectives into the communication methods
    ``dispatch`` and ``combine``.

    Args:
        n_routed_experts (int): Total number of routed experts across all EP ranks.
        ep_group (dist.ProcessGroup): Expert parallel process group.
        tp_group (dist.ProcessGroup): Tensor parallel process group.
        training_dtype (str): Dtype for training, ``"bf16"`` or ``"fp8"``.
        generate_dtype (str): Dtype for generation, ``"bf16"`` or ``"fp8"``.
    """

    # 中文注释：_tp_row_count_stream 只跑 tp_rank_row_counts 这类小的尺寸 all_gather。
    # 尺寸结果要同步回 Python list；如果复用 _comm_stream，会连同前面排队的大块
    # EP AllToAll 一起等完，削弱 Domino 隐藏 TP/EP 通信的效果。
    _tp_row_count_stream: torch.cuda.Stream | None = None

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
        if TorchAll2AllTPEPDispatcher._tp_row_count_stream is None:
            TorchAll2AllTPEPDispatcher._tp_row_count_stream = torch.cuda.Stream()
        self._tp_row_count_stream = TorchAll2AllTPEPDispatcher._tp_row_count_stream

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllTPEPDispatchResult:
        ep_dispatched = super().dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            async_op=async_op,
            decoding=decoding,
        )

        if async_op:
            assert ep_dispatched["forward_finished_event"] is not None, "Use async_op=True for dispatch!"
            assert ep_dispatched["backward_previous_event"] is not None, "Use async_op=True for dispatch!"
            comm_stream = cast(torch.cuda.Stream, self._comm_stream)
            # 中文注释：只同步变长 all_gather 的尺寸；
            # 大块 TP hidden 通信放到 comm stream 中隐藏。
            # 这里刻意使用 _tp_row_count_stream，避免为了拿 tp_rank_row_counts 的 Python list
            # 去同步 _comm_stream 上已经排队的 EP hidden AllToAll。
            tp_rank_row_counts = _tp_gather_tp_rank_row_counts(
                ep_dispatched["hidden_states"],
                self._tp_group,
                stream=self._tp_row_count_stream,
            )
            tp_hidden_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            tp_counts_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            tp_backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            hidden_states = _async_tp_all_gather_rows(
                ep_dispatched["hidden_states"],
                tp_rank_row_counts=tp_rank_row_counts,
                tp_group=self._tp_group,
                forward_previous_event=ep_dispatched["forward_finished_event"],
                forward_finished_event=tp_hidden_finished_event,
                backward_previous_event=tp_backward_previous_event,
                backward_finished_event=ep_dispatched["backward_previous_event"],
                comm_stream=comm_stream,
            )
            tokens_per_expert_group = _async_tp_all_gather_per_rank_metadata(
                ep_dispatched["tokens_per_expert_group"],
                tp_group=self._tp_group,
                forward_previous_event=tp_hidden_finished_event,
                forward_finished_event=tp_counts_finished_event,
                comm_stream=comm_stream,
            )
            forward_finished_event = tp_counts_finished_event
            backward_previous_event = tp_backward_previous_event
        else:
            hidden_states, tp_rank_row_counts = _tp_all_gather_rows(
                ep_dispatched["hidden_states"],
                tp_group=self._tp_group,
            )
            tokens_per_expert_group = _tp_all_gather_per_rank_metadata(
                ep_dispatched["tokens_per_expert_group"],
                tp_group=self._tp_group,
            )
            forward_finished_event = None
            backward_previous_event = None

        if decoding:
            raise NotImplementedError("Decoding is not yet supported for TorchAll2AllTPEPDispatcher.")

        return TorchAll2AllTPEPDispatchResult(
            hidden_states=hidden_states,
            topk_weights=ep_dispatched["topk_weights"],
            tokens_per_expert_group=tokens_per_expert_group,
            input_splits=ep_dispatched["input_splits"],
            output_splits=ep_dispatched["output_splits"],
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
            tp_rank_row_counts=tp_rank_row_counts,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllTPEPPostDispatchResult:
        tpep_dispatched = cast(TorchAll2AllTPEPDispatchResult, dispatched)
        if async_op:
            assert tpep_dispatched["forward_finished_event"] is not None, "Use async_op=True for dispatch!"
            assert tpep_dispatched["backward_previous_event"] is not None, "Use async_op=True for dispatch!"
            self.wait_comm_stream(tpep_dispatched["forward_finished_event"])

        token_counts = tpep_dispatched["tokens_per_expert_group"].ravel().to(torch.long)
        local_expert_ids = self._expert_ids_per_ep_rank.repeat(self._tp_size)
        global_input_tokens_local_experts_indices = torch.repeat_interleave(
            local_expert_ids,
            token_counts,
            output_size=tpep_dispatched["hidden_states"].shape[0],
        )
        global_input_tokens, row_ids_map = permute(
            tpep_dispatched["hidden_states"],
            global_input_tokens_local_experts_indices.to(torch.int32),
        )
        tokens_per_expert = tpep_dispatched["tokens_per_expert_group"].sum(dim=(0, 1))

        if async_op:
            if global_input_tokens.grad_fn is not None:
                global_input_tokens.grad_fn.register_hook(
                    get_backward_hook(
                        cast(torch.cuda.Event, tpep_dispatched["backward_previous_event"]),
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
        # Unpermute [M_total, H] back to TP-AllGather order (tp0_block | tp1_block | ...).
        hidden_states = unpermute(hidden_states, post_dispatched["row_ids_map"])

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

    @override
    def combine(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        post_dispatched: TorchAll2AllPostDispatchResult,
        pre_combined: TorchAll2AllPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllCombineResult:
        tpep_dispatched = cast(TorchAll2AllTPEPDispatchResult, dispatched)

        if async_op:
            forward_previous_event = pre_combined["forward_finished_event"]
            backward_finished_event = pre_combined["backward_previous_event"]
            assert forward_previous_event is not None, "Use async_op=True for combine_preprocess!"
            assert backward_finished_event is not None, "Use async_op=True for combine_preprocess!"
            comm_stream = cast(torch.cuda.Stream, self._comm_stream)

            tp_forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            tp_backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            # 中文注释：TP ReduceScatter 属于 combine 通信段，EP combine 等它完成后再发起。
            hidden_states = _async_tp_reduce_scatter_rows_sum(
                pre_combined["hidden_states"],
                tp_rank_row_counts=tpep_dispatched["tp_rank_row_counts"],
                tp_group=self._tp_group,
                forward_previous_event=forward_previous_event,
                forward_finished_event=tp_forward_finished_event,
                backward_previous_event=tp_backward_previous_event,
                backward_finished_event=backward_finished_event,
                comm_stream=comm_stream,
            )
            pre_combined_for_ep = TorchAll2AllPreCombineResult(
                hidden_states=hidden_states,
                backward_previous_event=tp_backward_previous_event,
                forward_finished_event=tp_forward_finished_event,
            )
        else:
            hidden_states = _tp_reduce_scatter_rows_sum(
                pre_combined["hidden_states"],
                tp_rank_row_counts=tpep_dispatched["tp_rank_row_counts"],
                tp_group=self._tp_group,
            )
            pre_combined_for_ep = TorchAll2AllPreCombineResult(
                hidden_states=hidden_states,
                backward_previous_event=None,
                forward_finished_event=None,
            )

        return cast(
            TorchAll2AllCombineResult,
            super().combine(
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                pre_combined=pre_combined_for_ep,
                async_op=async_op,
                decoding=decoding,
            ),
        )
