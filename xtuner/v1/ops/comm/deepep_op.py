# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap
from deep_ep_cpp import EventHandle

from xtuner.v1.utils import get_logger


logger = get_logger()

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

_low_latency_buffer: Optional[Buffer] = None
# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_deepep_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer
    if _buffer is None:
        # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
        num_nvl_bytes, num_rdma_bytes = 0, 0
        for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
        ):
            num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                num_rdma_bytes,
            )

        # Allocate a buffer if not existed or not enough buffer size
        # NOTES: the adaptive routing configuration of the network **must be off**
        if dist.get_rank() % 8 == 0:
            logger.info(f"[DeepEP] num_nvl_bytes: {num_nvl_bytes}, num_rdma_bytes: {num_rdma_bytes}")
        if (
            _buffer is None
            or _buffer.group != group
            or _buffer.num_nvl_bytes < num_nvl_bytes
            or _buffer.num_rdma_bytes < num_rdma_bytes
        ):
            _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer


def get_low_latency_buffer(
    group: dist.ProcessGroup,
    hidden: int,
    num_experts: int,
    num_max_dispatch_tokens_per_rank: int = 128,
) -> Buffer:
    # NOTES: the low-latency mode will consume much more space than the normal mode
    # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be
    # less than 256
    global _buffer

    num_nvl_bytes = 0
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )
    # num_rdma_bytes = 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden * 2, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden * 2, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None:
        # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
        assert num_experts % group.size() == 0
        # _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
        _buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=max(num_experts // group.size(), Buffer.num_sms // 2),
        )
        logger.info(
            f"{num_nvl_bytes}, {_buffer.num_nvl_bytes}, {num_max_dispatch_tokens_per_rank}, {hidden}, {num_experts}, {group.size()}"
        )
    else:
        assert num_nvl_bytes <= _buffer.num_nvl_bytes, (
            f"{num_nvl_bytes}, {_buffer.num_nvl_bytes}, {num_max_dispatch_tokens_per_rank}, {hidden}, {num_experts}, {group.size()}"
        )
        assert num_rdma_bytes <= _buffer.num_rdma_bytes
    return _buffer


# Buffer.set_num_sms(24)
# Communication buffer (will allocate at runtime)
# _buffer: Optional[Buffer] = None
global_event_dict: Dict[str, EventOverlap] = dict()


def get_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


@lru_cache()
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def buffer_capture() -> EventOverlap:
    return EventOverlap(EventHandle())


def dispatch_forward(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor | None,
    num_experts: int,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
) -> Tuple[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    List,
    Tuple,
    EventOverlap,
]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    # _buffer = get_buffer(group, get_hidden_bytes(x))
    if isinstance(x, torch.Tensor):
        hidden_size = x.shape[-1]
    else:
        hidden_size = x[0].shape[-1]

    _buffer = get_low_latency_buffer(group, hidden=hidden_size, num_experts=num_experts)

    # Calculate layout before actual dispatch
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = _buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=previous_event is not None,
    )
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # For more advanced usages, please refer to the docs of the `dispatch` function
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    # For event management, please refer to the docs of the `EventOverlap` class

    # NOTE: Since we do not pass `handle` to deepep kernel here,
    # `recv_topk_weights` and `recv_topk_weights` will not be `None`

    recv_topk_idx = cast(torch.Tensor, recv_topk_idx)
    recv_topk_weights = cast(torch.Tensor, recv_topk_weights)

    return (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    )


def dispatch_backward(
    grad_recv_x: torch.Tensor,
    grad_recv_topk_weights: torch.Tensor,
    num_experts: int,
    handle: Tuple,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, EventOverlap]:
    hidden_size = grad_recv_topk_weights[0].shape[-1]
    _buffer = get_low_latency_buffer(group, hidden=hidden_size, num_experts=num_experts)

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle,
        topk_weights=grad_recv_topk_weights,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(
    x: torch.Tensor,
    num_experts: int,
    handle: Tuple,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
) -> Tuple[torch.Tensor, EventOverlap]:
    hidden_size = x.shape[-1]
    _buffer = get_low_latency_buffer(group, hidden=hidden_size, num_experts=num_experts)

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(
    grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    num_experts: int,
    handle: Tuple,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    hidden_size = grad_combined_x[0].shape[-1] if isinstance(grad_combined_x, tuple) else grad_combined_x.shape[-1]
    _buffer = get_low_latency_buffer(group, num_experts=num_experts, hidden=hidden_size)

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        async_finish=True,
        previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None,
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event


class DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        group: dist.ProcessGroup,
        previous_event: Optional[EventOverlap] = None,
        fwd_comm_dtype_fp8: bool = False,
    ) -> Tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap
    ]:
        if fwd_comm_dtype_fp8:
            # TODO(chenchiyu): quant fp8 for x
            raise NotImplementedError("fp8 quant not implemented for dispatch forward")
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = dispatch_forward(x, topk_idx, topk_weights, num_experts, group, previous_event)
        # save deep comm handle
        ctx.save_for_backward(*handle)
        ctx.group = group
        ctx.num_experts = num_experts
        if fwd_comm_dtype_fp8:
            # TODO(chenchiyu): Float8Tensor process for recv_x = (x, x_scale)
            pass
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    @staticmethod
    def backward(  # type: ignore[invalid-override]
        ctx,
        grad_recv_x: torch.Tensor,
        grad_recv_topk_idx: torch.Tensor,
        grad_recv_topk_weights: torch.Tensor,
        *args,
    ) -> Tuple[torch.Tensor, None, torch.Tensor | None, None, None, None, None]:
        # load saved comm handle
        handle = ctx.saved_tensors
        combined_grad_x, combined_grad_recv_topk_weights, event = dispatch_backward(
            grad_recv_x, grad_recv_topk_weights, ctx.num_experts, handle, ctx.group, buffer_capture()
        )
        event.current_stream_wait()
        return (
            combined_grad_x,
            None,
            combined_grad_recv_topk_weights,
            None,
            None,
            None,
            None,
        )


class DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        num_experts: int,
        handle: Tuple,
        group: dist.ProcessGroup,
        previous_event: Optional[EventOverlap] = None,
        bwd_comm_dtype_fp8: bool = False,
    ) -> Tuple[torch.Tensor, EventOverlap]:
        combined_x, event = combine_forward(x, num_experts, handle, group, previous_event)
        # save deep comm handle
        ctx.save_for_backward(*handle)
        ctx.group = group
        ctx.bwd_comm_dtype_fp8 = bwd_comm_dtype_fp8
        ctx.num_experts = num_experts
        return combined_x, event

    @staticmethod
    def backward(  # type: ignore[invalid-override]
        ctx, grad_combined_x: torch.Tensor, *args
    ) -> Tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None, None, None]:
        bwd_comm_dtype_fp8 = ctx.bwd_comm_dtype_fp8
        if bwd_comm_dtype_fp8:
            # TODO(chenchiyu): quant fp8 for grad_combined_x
            raise NotImplementedError("fp8 quant not implemented for combine backward")
        # load saved comm handle
        handle = ctx.saved_tensors
        grad_x, event = combine_backward(grad_combined_x, ctx.num_experts, handle, ctx.group, buffer_capture())
        event.current_stream_wait()
        if bwd_comm_dtype_fp8:
            # TODO(chenchiyu): Float8Tensor process for grad_x = (x, x_scale)
            pass
        return grad_x, None, None, None, None


def deep_ep_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_routed_experts: int,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
    fwd_comm_dtype_fp8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    return DeepEPDispatch.apply(
        x,
        topk_idx,
        topk_weights,
        num_routed_experts,
        group,
        previous_event,
        fwd_comm_dtype_fp8,
    )  # type: ignore[return-value]


def deep_ep_combine(
    x: torch.Tensor,
    num_experts: int,
    deepep_comm_handle: Tuple,
    group: dist.ProcessGroup,
    previous_event: Optional[EventOverlap] = None,
    bwd_comm_dtype_fp8: bool = False,
) -> Tuple[torch.Tensor, EventOverlap]:
    return DeepEPCombine.apply(x, num_experts, deepep_comm_handle, group, previous_event, bwd_comm_dtype_fp8)  # type: ignore[return-value]


class DeepEPDispatchBwdOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        num_experts: int,
        comm_handle: Tuple,
        group: dist.ProcessGroup,
        previous_event_key: str,
        finish_event_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # save deep comm handle
        ctx.save_for_backward(*comm_handle)
        ctx.previous_event_key = previous_event_key
        ctx.finish_event_key = finish_event_key
        ctx.group = group
        ctx.num_experts = num_experts
        return recv_x, recv_topk_weights

    @staticmethod
    def backward(  # type: ignore[invalid-override]
        ctx, grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, None, None, None, None, None, None]:
        global global_event_dict
        # load saved comm handle
        comm_handle = ctx.saved_tensors
        previous_event = global_event_dict.pop(ctx.previous_event_key)
        combined_grad_x, combined_grad_topk_weights, event = dispatch_backward(
            grad_recv_x, grad_recv_topk_weights, ctx.num_experts, comm_handle, ctx.group, previous_event
        )
        assert ctx.finish_event_key not in global_event_dict
        global_event_dict[ctx.finish_event_key] = event
        return (
            combined_grad_x,
            combined_grad_topk_weights,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DeepEPCombineBwdOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        combined_x: torch.Tensor,
        num_experts: int,
        comm_handle: Tuple,
        group: dist.ProcessGroup,
        previous_event_key: str,
        finish_event_key: str,
        bwd_comm_dtype_fp8: bool,
    ) -> torch.Tensor:
        # save deep comm handle
        ctx.save_for_backward(*comm_handle)
        ctx.previous_event_key = previous_event_key
        ctx.finish_event_key = finish_event_key
        ctx.group = group
        ctx.bwd_comm_dtype_fp8 = bwd_comm_dtype_fp8
        ctx.num_experts = num_experts
        return combined_x

    @staticmethod
    def backward(  # type: ignore[invalid-override]
        ctx, grad_combined_x: torch.Tensor
    ) -> Tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None, None, None, None, None]:
        global global_event_dict
        bwd_comm_dtype_fp8 = ctx.bwd_comm_dtype_fp8
        if bwd_comm_dtype_fp8:
            # TODO(chenchiyu): quant fp8 for grad_combined_x
            raise NotImplementedError("fp8 quant not implemented for combine backward")
        # load saved comm handle
        comm_handle = ctx.saved_tensors
        previous_event = global_event_dict.pop(ctx.previous_event_key)
        grad_x, event = combine_backward(grad_combined_x, ctx.num_experts, comm_handle, ctx.group, previous_event)
        assert ctx.finish_event_key not in global_event_dict
        global_event_dict[ctx.finish_event_key] = event
        if bwd_comm_dtype_fp8:
            # TODO(chenchiyu): Float8Tensor process for grad_x = (x, x_scale)
            pass
        return grad_x, None, None, None, None, None, None


class WaitEventBwdOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, event_key: str) -> torch.Tensor:
        ctx.event_key = event_key
        return input_

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None]:  # type: ignore[invalid-override]
        global global_event_dict
        event = global_event_dict.pop(ctx.event_key)
        event.current_stream_wait()
        return grad_output, None


class CaptureEventBwdOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, event_key: str) -> torch.Tensor:
        ctx.event_key = event_key
        return input_

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None]:  # type: ignore[invalid-override]
        global global_event_dict
        assert ctx.event_key not in global_event_dict
        global_event_dict[ctx.event_key] = buffer_capture()
        return grad_output, None


# TODO: (yehaochen) Forward and backward logic should be decoupled
def deep_ep_dispatch_fwd_bwd_only(  # type: ignore
    x: torch.Tensor,
    topk_idx: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    num_routed_experts: int,
    group: dist.ProcessGroup,
    previous_event_key: str,
    finish_event_key: str,
    fwd_comm_dtype_fp8: bool = False,
    is_forward: bool = True,
    # bwd only param
    deepep_comm_handle: Optional[Tuple] = None,
    recv_x: Optional[torch.Tensor] = None,
    recv_topk_weights: Optional[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, Tuple],
    Tuple[torch.Tensor, torch.Tensor],
]:
    if is_forward:
        global global_event_dict
        if fwd_comm_dtype_fp8:
            # TODO(chenchiyu): quant fp8 for x
            raise NotImplementedError("fp8 quant not implemented for dispatch forward")
        previous_event = global_event_dict.pop(previous_event_key)
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = dispatch_forward(x, topk_idx, topk_weights, num_routed_experts, group, previous_event)  # type: ignore
        # deepep kernel won't propagate requires_grad
        # TODO(chenchiyu): support fp8 with recv_x = (x, x_scale)
        if not isinstance(recv_x, tuple):
            recv_x.requires_grad_(x.requires_grad)  # type: ignore
        else:
            raise NotImplementedError("grads propagation not implemented for dispatch forward")
        recv_topk_weights.requires_grad_(topk_weights.requires_grad)
        if fwd_comm_dtype_fp8:
            # TODO(chenchiyu): Float8Tensor process for recv_x = (x, x_scale)
            pass
        assert finish_event_key not in global_event_dict
        global_event_dict[finish_event_key] = event
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
        )  # type: ignore
    else:
        return DeepEPDispatchBwdOnly.apply(
            x,
            topk_weights,
            recv_x,
            recv_topk_weights,
            num_routed_experts,
            deepep_comm_handle,
            group,
            previous_event_key,
            finish_event_key,
        )  # type: ignore


# TODO: (yehaochen) Forward and backward logic should be decoupled
def deep_ep_combine_fwd_bwd_only(
    x: torch.Tensor,
    num_experts: int,
    deepep_comm_handle: Tuple,
    group: dist.ProcessGroup,
    previous_event_key: str,
    finish_event_key: str,
    bwd_comm_dtype_fp8: bool = False,
    is_forward: bool = True,
    # bwd only param
    combined_x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if is_forward:
        global global_event_dict
        previous_event = global_event_dict.pop(previous_event_key)
        combined_x, event = combine_forward(x, num_experts, deepep_comm_handle, group, previous_event)
        # deepep kernel won't propagate requires_grad
        combined_x.requires_grad_(x.requires_grad)
        assert finish_event_key not in global_event_dict
        global_event_dict[finish_event_key] = event
        return combined_x
    else:
        return DeepEPCombineBwdOnly.apply(
            x,
            combined_x,
            num_experts,
            deepep_comm_handle,
            group,
            previous_event_key,
            finish_event_key,
            bwd_comm_dtype_fp8,
        )


def wait_event_fwd(event_key: str) -> None:
    global global_event_dict
    event = global_event_dict.pop(event_key)
    event.current_stream_wait()


def wait_event_bwd(input_: torch.Tensor, event_key: str) -> torch.Tensor:
    return WaitEventBwdOnly.apply(input_, event_key)


def capture_event_fwd(event_key: str) -> None:
    global global_event_dict
    assert event_key not in global_event_dict
    global_event_dict[event_key] = buffer_capture()


def capture_event_bwd(input_: torch.Tensor, event_key: str) -> torch.Tensor:
    return CaptureEventBwdOnly.apply(input_, event_key)
