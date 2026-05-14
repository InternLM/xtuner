"""TPEP dispatcher TP collective refactor sketch.

这个文件是设计伪代码，不接入训练路径。它描述当前更轻量的改法：

1. 不引入额外的执行上下文概念。
2. 保留同步和异步两个 autograd Function，让流程仍然直观对应当前代码。
3. 只把 TP AllGather / ReduceScatter 的核心通信、拼接、切片逻辑抽成共享函数。
4. 异步 Function 只比同步 Function 多做 stream wait、event record、record_stream。
"""

from __future__ import annotations

from typing import Any


Tensor = Any
ProcessGroup = Any
CudaEvent = Any
CudaStream = Any


# =============================================================================
# 1. 共享核心实现：同步/异步都调用这些函数
# =============================================================================


def tp_all_gather_forward_impl(
    hidden: Tensor,
    all_sizes: list[int],
    tp_group: ProcessGroup,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """TP AllGather forward 的共享核心。

    中文注释：这里只表达数学和 collective：
    [M_local, H] -> all_gather -> [M_total, H]。
    它不关心是否异步，也不关心 CUDA event。
    """
    hidden_for_comm = hidden.contiguous()
    chunks = [empty_rows_like(hidden_for_comm, rows) for rows in all_sizes]
    dist_all_gather(chunks, hidden_for_comm, group=tp_group)
    gathered = cat_rows(chunks)
    return gathered, hidden_for_comm, chunks


def tp_all_gather_backward_impl(
    grad: Tensor,
    all_sizes: list[int],
    tp_rank: int,
    tp_group: ProcessGroup,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """TP AllGather backward 的共享核心。

    中文注释：AllGather backward 的语义就是 TP ReduceScatterSum，
    因此和 combine forward 共用同一个真正 reduce_scatter 实现。
    """
    return tp_reduce_scatter_sum_impl(grad, all_sizes, tp_rank, tp_group)


def tp_reduce_scatter_sum_impl(
    hidden: Tensor,
    all_sizes: list[int],
    tp_rank: int,
    tp_group: ProcessGroup,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """TP ReduceScatterSum 的共享核心。

    中文注释：等长时走 reduce_scatter_tensor fast path；变长时按 TP size meta
    split 成 input_list，走 torch.distributed.reduce_scatter。
    """
    hidden_for_comm = hidden.contiguous()
    out = empty_rows_like(hidden_for_comm, all_sizes[tp_rank])
    if all_rows_are_empty(all_sizes):
        return out, hidden_for_comm, []
    if all_splits_equal(all_sizes):
        dist_reduce_scatter_tensor(out, hidden_for_comm, group=tp_group)
        return out, hidden_for_comm, []

    input_chunks = split_rows(hidden_for_comm, all_sizes)
    dist_reduce_scatter(out, input_chunks, group=tp_group)
    return out, hidden_for_comm, input_chunks


def tp_reduce_scatter_sum_forward_impl(
    hidden: Tensor,
    all_sizes: list[int],
    tp_rank: int,
    tp_group: ProcessGroup,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """TP ReduceScatterSum forward 的共享核心。"""
    return tp_reduce_scatter_sum_impl(hidden, all_sizes, tp_rank, tp_group)


def tp_reduce_scatter_sum_backward_impl(
    grad_slice: Tensor,
    all_sizes: list[int],
    tp_group: ProcessGroup,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """TP ReduceScatterSum backward 的共享核心。"""
    grad_slice_for_comm = grad_slice.contiguous()
    chunks = [empty_rows_like(grad_slice_for_comm, rows) for rows in all_sizes]
    dist_all_gather(chunks, grad_slice_for_comm, group=tp_group)
    full_grad = cat_rows(chunks)
    return full_grad, grad_slice_for_comm, chunks


# =============================================================================
# 2. 同步 Function：只调用共享核心
# =============================================================================


class TPAllGather:
    """同步 TP AllGather 伪代码。真实代码继承 ``torch.autograd.Function``。"""

    @staticmethod
    def forward(ctx: Any, hidden: Tensor, all_sizes: list[int], tp_group: ProcessGroup, tp_rank: int) -> Tensor:
        gathered, _, _ = tp_all_gather_forward_impl(hidden, all_sizes, tp_group)
        ctx.all_sizes = all_sizes
        ctx.tp_rank = tp_rank
        ctx.tp_group = tp_group
        return gathered

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        grad_input, _, _ = tp_all_gather_backward_impl(grad, ctx.all_sizes, ctx.tp_rank, ctx.tp_group)
        return grad_input


class TPReduceScatterSum:
    """同步 TP ReduceScatterSum 伪代码。"""

    @staticmethod
    def forward(ctx: Any, hidden: Tensor, all_sizes: list[int], tp_group: ProcessGroup, tp_rank: int) -> Tensor:
        out, _, _ = tp_reduce_scatter_sum_forward_impl(hidden, all_sizes, tp_rank, tp_group)
        ctx.all_sizes = all_sizes
        ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx: Any, grad_slice: Tensor) -> Tensor:
        full_grad, _, _ = tp_reduce_scatter_sum_backward_impl(grad_slice, ctx.all_sizes, ctx.tp_group)
        return full_grad


# =============================================================================
# 3. 异步 Function：流程和同步一致，只额外包 stream/event
# =============================================================================


class AsyncTPAllGather:
    """异步 TP AllGather 伪代码。"""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        all_sizes: list[int],
        tp_group: ProcessGroup,
        tp_rank: int,
        forward_previous_event: CudaEvent,
        forward_finished_event: CudaEvent,
        backward_previous_event: CudaEvent,
        backward_finished_event: CudaEvent,
        comm_stream: CudaStream,
    ) -> Tensor:
        with cuda_stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            gathered, hidden_for_comm, chunks = tp_all_gather_forward_impl(hidden, all_sizes, tp_group)

            # 中文注释：异步路径不重写 TP AllGather 逻辑，只管理 stream/event 生命周期。
            record_stream((hidden_for_comm, chunks, gathered), comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.all_sizes = all_sizes
        ctx.tp_rank = tp_rank
        ctx.tp_group = tp_group
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return gathered

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> Tensor:
        with cuda_stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            grad_input, grad_for_comm, chunks = tp_all_gather_backward_impl(
                grad,
                ctx.all_sizes,
                ctx.tp_rank,
                ctx.tp_group,
            )
            record_stream((grad_for_comm, chunks, grad_input), ctx.comm_stream)
            ctx.backward_finished_event.record(ctx.comm_stream)
        return grad_input


class AsyncTPReduceScatterSum:
    """异步 TP ReduceScatterSum 伪代码。"""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: Tensor,
        all_sizes: list[int],
        tp_group: ProcessGroup,
        tp_rank: int,
        forward_previous_event: CudaEvent,
        forward_finished_event: CudaEvent,
        backward_previous_event: CudaEvent,
        backward_finished_event: CudaEvent,
        comm_stream: CudaStream,
    ) -> Tensor:
        with cuda_stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            out, hidden_for_comm, chunks = tp_reduce_scatter_sum_forward_impl(hidden, all_sizes, tp_rank, tp_group)

            # 中文注释：异步路径不重写 ReduceScatter 逻辑，只记录通信流持有的 tensor。
            record_stream((hidden_for_comm, chunks, out), comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.all_sizes = all_sizes
        ctx.tp_group = tp_group
        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.comm_stream = comm_stream
        return out

    @staticmethod
    def backward(ctx: Any, grad_slice: Tensor) -> Tensor:
        with cuda_stream(ctx.comm_stream):
            ctx.comm_stream.wait_event(ctx.backward_previous_event)
            full_grad, grad_slice_for_comm, chunks = tp_reduce_scatter_sum_backward_impl(
                grad_slice,
                ctx.all_sizes,
                ctx.tp_group,
            )
            record_stream((grad_slice_for_comm, chunks, full_grad), ctx.comm_stream)
            ctx.backward_finished_event.record(ctx.comm_stream)
        return full_grad


# =============================================================================
# 4. dispatcher 仍然保持当前显式流程
# =============================================================================


def dispatch_tpep_pseudocode(ep_dispatched: Any, tp_group: ProcessGroup, async_op: bool) -> Any:
    """EP dispatch 后做 TP AllGather；这里只展示同步/异步流程保持相似。"""
    all_sizes = gather_tp_sizes(ep_dispatched.hidden_states, tp_group)
    tp_rank = dist_get_rank(tp_group)

    if async_op:
        hidden_states = AsyncTPAllGather.forward(
            ctx=new_ctx(),
            hidden=ep_dispatched.hidden_states,
            all_sizes=all_sizes,
            tp_group=tp_group,
            tp_rank=tp_rank,
            forward_previous_event=ep_dispatched.forward_finished_event,
            forward_finished_event=new_cuda_event(),
            backward_previous_event=new_cuda_event(),
            backward_finished_event=ep_dispatched.backward_previous_event,
            comm_stream=get_comm_stream(),
        )
    else:
        hidden_states = TPAllGather.forward(
            ctx=new_ctx(),
            hidden=ep_dispatched.hidden_states,
            all_sizes=all_sizes,
            tp_group=tp_group,
            tp_rank=tp_rank,
        )
    return hidden_states


def migration_plan() -> list[str]:
    return [
        "保留现有同步/异步 autograd Function，不新增 stage/context 抽象。",
        "抽出 AllGather forward/backward 的共享核心函数。",
        "抽出真正 reduce_scatter 的 TP ReduceScatterSum 共享核心函数。",
        "异步 Function 只保留 wait_event、record_stream、record_event 这些异步胶水。",
        "dispatcher 的 dispatch/combine 调用形状保持不变。",
    ]


# =============================================================================
# 5. 伪代码占位函数
# =============================================================================


def empty_rows_like(tensor: Tensor, rows: int) -> Tensor:
    raise NotImplementedError


def dist_all_gather(chunks: list[Tensor], tensor: Tensor, *, group: ProcessGroup) -> None:
    raise NotImplementedError


def dist_reduce_scatter_tensor(output: Tensor, input: Tensor, *, group: ProcessGroup) -> None:
    raise NotImplementedError


def dist_reduce_scatter(output: Tensor, input_list: list[Tensor], *, group: ProcessGroup) -> None:
    raise NotImplementedError


def split_rows(tensor: Tensor, sizes: list[int]) -> list[Tensor]:
    raise NotImplementedError


def all_splits_equal(sizes: list[int]) -> bool:
    raise NotImplementedError


def all_rows_are_empty(sizes: list[int]) -> bool:
    raise NotImplementedError


def cat_rows(chunks: list[Tensor]) -> Tensor:
    raise NotImplementedError


def cuda_stream(stream: CudaStream) -> Any:
    raise NotImplementedError


def record_stream(value: Any, stream: CudaStream) -> None:
    raise NotImplementedError


def gather_tp_sizes(hidden: Tensor, tp_group: ProcessGroup) -> list[int]:
    raise NotImplementedError


def dist_get_rank(tp_group: ProcessGroup) -> int:
    raise NotImplementedError


def new_ctx() -> Any:
    raise NotImplementedError


def new_cuda_event() -> CudaEvent:
    raise NotImplementedError


def get_comm_stream() -> CudaStream:
    raise NotImplementedError
