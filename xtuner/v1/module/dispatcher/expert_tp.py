from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


def _tp_all_gather_rows_forward_impl(
    tensor: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    tensor = tensor.contiguous()
    chunks = [
        torch.empty((size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) for size in tp_rank_row_counts
    ]
    dist.all_gather(chunks, tensor, group=tp_group)
    return torch.cat(chunks, dim=0), tensor, chunks


def _tp_reduce_scatter_rows_sum_impl(
    tensor: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_rank: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    tensor = tensor.contiguous()
    assert tensor.shape[0] == sum(tp_rank_row_counts), (
        "TP ReduceScatterRowsSum input rows must match tp_rank_row_counts."
    )

    out = tensor.new_empty((tp_rank_row_counts[tp_rank], *tensor.shape[1:]))
    if tensor.shape[0] == 0:
        # 中文注释：所有 TP rank 都没有 token 时没有通信量，直接返回当前 rank 的 0 行 slice。
        return out, tensor, []

    if all(size == tp_rank_row_counts[0] for size in tp_rank_row_counts):
        dist.reduce_scatter_tensor(out, tensor, op=dist.ReduceOp.SUM, group=tp_group)
        return out, tensor, []

    input_chunks = list(torch.split(tensor, tp_rank_row_counts, dim=0))
    dist.reduce_scatter(out, input_chunks, op=dist.ReduceOp.SUM, group=tp_group)
    return out, tensor, input_chunks


def _tp_all_gather_rows_backward_impl(
    grad: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_rank: int,
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    return _tp_reduce_scatter_rows_sum_impl(grad, tp_rank_row_counts, tp_rank, tp_group)


def _tp_reduce_scatter_rows_sum_backward_impl(
    grad_slice: torch.Tensor,
    tp_rank_row_counts: list[int],
    tp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    grad_slice = grad_slice.contiguous()
    chunks = [
        torch.empty((size, *grad_slice.shape[1:]), dtype=grad_slice.dtype, device=grad_slice.device)
        for size in tp_rank_row_counts
    ]
    dist.all_gather(chunks, grad_slice, group=tp_group)
    return torch.cat(chunks, dim=0), grad_slice, chunks


class _TPAllGatherRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        gathered, _, _ = _tp_all_gather_rows_forward_impl(tensor, tp_rank_row_counts, tp_group)
        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.tp_group = tp_group
        ctx.tp_rank = tp_rank
        return gathered

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        grad_input, _, _ = _tp_all_gather_rows_backward_impl(grad, ctx.tp_rank_row_counts, ctx.tp_rank, ctx.tp_group)
        return grad_input, None, None, None, None


class _AsyncTPAllGatherRows(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
        forward_previous_event: torch.cuda.Event | None,
        forward_finished_event: torch.cuda.Event | None,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with torch.cuda.stream(comm_stream):
            if forward_previous_event is not None:
                comm_stream.wait_event(forward_previous_event)
            gathered, tensor_for_comm, chunks = _tp_all_gather_rows_forward_impl(tensor, tp_rank_row_counts, tp_group)
            # 中文注释：异步路径只增加 stream/event 管理；
            # collective 核心逻辑和同步路径一致。
            _record_stream((tensor_for_comm, chunks, gathered), comm_stream)
            if forward_finished_event is not None:
                forward_finished_event.record(comm_stream)

        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.tp_group = tp_group
        ctx.tp_rank = tp_rank
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return gathered

    @staticmethod
    def backward(
        ctx: Any,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None, None]:
        grad_ready_event = torch.cuda.Event()
        grad_ready_event.record()
        with torch.cuda.stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            ctx.comm_stream.wait_event(grad_ready_event)
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
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        tp_group: dist.ProcessGroup,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        out, _, _ = _tp_reduce_scatter_rows_sum_impl(tensor, tp_rank_row_counts, tp_rank, tp_group)
        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx: Any, grad_slice: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        full_grad, _, _ = _tp_reduce_scatter_rows_sum_backward_impl(grad_slice, ctx.tp_rank_row_counts, ctx.tp_group)
        return full_grad, None, None, None, None


class _AsyncTPReduceScatterRowsSum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
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
            out, tensor_for_comm, chunks = _tp_reduce_scatter_rows_sum_impl(
                tensor,
                tp_rank_row_counts,
                tp_rank,
                tp_group,
            )
            # 中文注释：TP ReduceScatterRowsSum 属于 combine 通信段；
            # 输出事件交给 combine_postprocess 等待。
            _record_stream((tensor_for_comm, chunks, out), comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.tp_rank_row_counts = tp_rank_row_counts
        ctx.tp_group = tp_group
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_slice: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None, None]:
        grad_ready_event = torch.cuda.Event()
        grad_ready_event.record()
        with torch.cuda.stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            ctx.comm_stream.wait_event(grad_ready_event)
            full_grad, grad_slice_for_comm, chunks = _tp_reduce_scatter_rows_sum_backward_impl(
                grad_slice,
                ctx.tp_rank_row_counts,
                ctx.tp_group,
            )
            _record_stream((grad_slice_for_comm, chunks, full_grad), ctx.comm_stream)
            ctx.backward_finished_event.record(ctx.comm_stream)

        return full_grad, None, None, None, None, None, None, None, None, None


class ExpertTP:
    """Token-sliced Expert TP collectives shared by dispatcher routing
    paths."""

    def __init__(self, tp_group: dist.ProcessGroup) -> None:
        self._tp_group = tp_group
        self._tp_size = tp_group.size()

    @property
    def size(self) -> int:
        return self._tp_size

    def gather_tp_rank_row_counts(self, tensor: torch.Tensor, stream: torch.cuda.Stream | None = None) -> list[int]:
        if self._tp_size == 1:
            return [tensor.shape[0]]

        if stream is None:
            local_size = tensor.new_tensor([tensor.shape[0]], dtype=torch.long)
            tp_rank_row_counts_t = tensor.new_empty([self._tp_size], dtype=torch.long)
            dist.all_gather_into_tensor(tp_rank_row_counts_t, local_size, group=self._tp_group)
        else:
            # 中文注释：行数要转成 Python list；单独 stream 避免同步
            # dispatcher comm stream 上的大 tensor 通信。
            with torch.cuda.stream(stream):
                local_size = tensor.new_tensor([tensor.shape[0]], dtype=torch.long)
                tp_rank_row_counts_t = tensor.new_empty([self._tp_size], dtype=torch.long)
                dist.all_gather_into_tensor(tp_rank_row_counts_t, local_size, group=self._tp_group)
                _record_stream((local_size, tp_rank_row_counts_t), stream)
            stream.synchronize()
        return [int(size) for size in tp_rank_row_counts_t.tolist()]

    def all_gather_rows(
        self,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        if self._tp_size == 1:
            return tensor, [tensor.shape[0]]

        if tp_rank_row_counts is None:
            tp_rank_row_counts = self.gather_tp_rank_row_counts(tensor)

        tp_rank = dist.get_rank(group=self._tp_group)
        gathered = _TPAllGatherRows.apply(tensor, tp_rank_row_counts, self._tp_group, self._tp_size, tp_rank)
        return gathered, tp_rank_row_counts

    def all_gather_row_metadata(self, tensor: torch.Tensor, tp_rank_row_counts: list[int]) -> torch.Tensor:
        # 中文注释：topk_ids/topk_weights 和 hidden 使用同一份
        # tp_rank_row_counts，保证 source token 对齐。
        gathered, _ = self.all_gather_rows(tensor, tp_rank_row_counts)
        return gathered

    def all_gather_per_rank_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        # 中文注释：tokens_per_expert_group 这类固定形状 meta
        # 不沿 token 维变长，使用独立 gather。
        if self._tp_size == 1:
            return tensor.unsqueeze(0)

        gathered = tensor.new_empty((self._tp_size, *tensor.shape))
        dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=self._tp_group)
        return gathered

    def async_all_gather_rows(
        self,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        forward_previous_event: torch.cuda.Event | None,
        forward_finished_event: torch.cuda.Event | None,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        if self._tp_size == 1:
            if forward_finished_event is not None:
                forward_finished_event.record()
            return tensor

        tp_rank = dist.get_rank(group=self._tp_group)
        return _AsyncTPAllGatherRows.apply(
            tensor,
            tp_rank_row_counts,
            self._tp_group,
            self._tp_size,
            tp_rank,
            forward_previous_event,
            forward_finished_event,
            backward_previous_event,
            backward_finished_event,
            comm_stream,
        )

    def async_all_gather_row_metadata(
        self,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        forward_previous_event: torch.cuda.Event | None,
        forward_finished_event: torch.cuda.Event | None,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        if self._tp_size == 1:
            if forward_finished_event is not None:
                forward_finished_event.record()
            return tensor

        with torch.cuda.stream(comm_stream):
            if forward_previous_event is not None:
                comm_stream.wait_event(forward_previous_event)
            gathered, tensor_for_comm, chunks = _tp_all_gather_rows_forward_impl(
                tensor,
                tp_rank_row_counts,
                self._tp_group,
            )
            _record_stream((tensor_for_comm, chunks, gathered), comm_stream)
            if forward_finished_event is not None:
                forward_finished_event.record(comm_stream)
        return gathered

    def async_all_gather_per_rank_metadata(
        self,
        tensor: torch.Tensor,
        forward_previous_event: torch.cuda.Event | None,
        forward_finished_event: torch.cuda.Event | None,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        if self._tp_size == 1:
            if forward_finished_event is not None:
                forward_finished_event.record()
            return tensor.unsqueeze(0)

        gathered = tensor.new_empty((self._tp_size, *tensor.shape))
        with torch.cuda.stream(comm_stream):
            if forward_previous_event is not None:
                comm_stream.wait_event(forward_previous_event)
            tensor_for_comm = tensor.contiguous()
            dist.all_gather_into_tensor(gathered, tensor_for_comm, group=self._tp_group)
            _record_stream((tensor_for_comm, gathered), comm_stream)
            if forward_finished_event is not None:
                forward_finished_event.record(comm_stream)
        return gathered

    def reduce_scatter_rows_sum(self, tensor: torch.Tensor, tp_rank_row_counts: list[int]) -> torch.Tensor:
        if self._tp_size == 1:
            return tensor

        tp_rank = dist.get_rank(group=self._tp_group)
        return _TPReduceScatterRowsSum.apply(tensor, tp_rank_row_counts, self._tp_group, self._tp_size, tp_rank)

    def async_reduce_scatter_rows_sum(
        self,
        tensor: torch.Tensor,
        tp_rank_row_counts: list[int],
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        if self._tp_size == 1:
            forward_finished_event.record()
            return tensor

        tp_rank = dist.get_rank(group=self._tp_group)
        return _AsyncTPReduceScatterRowsSum.apply(
            tensor,
            tp_rank_row_counts,
            self._tp_group,
            self._tp_size,
            tp_rank,
            forward_previous_event,
            forward_finished_event,
            backward_previous_event,
            backward_finished_event,
            comm_stream,
        )
