from typing import Literal, TypeAlias, cast

import torch
import torch.distributed as dist
from deep_ep import EventOverlap
from mmengine.utils import is_installed
from typing_extensions import override

from xtuner.v1.ops import permute, unpermute
from xtuner.v1.ops.comm.deepep_op import (
    buffer_capture,
    combine_backward,
    combine_forward,
    dispatch_backward,
    dispatch_forward,
)
from xtuner.v1.utils import copy_method_signature, get_device, get_logger

from . import XTUNER_DISPATCHER_DEBUG
from .base import (
    CombineResult,
    DispatchResult,
    GenericDispatcher,
    PostCombineResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
)


if get_device() == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa


DEVICE = get_device()
logger = get_logger()


DeepEPHandle = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# DeepEP handle include 6 tensor:
# (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
class DeepEPPreDispatchResult(PreDispatchResult):
    backward_previous_event: EventOverlap | None
    forward_finished_event: EventOverlap | None


class DeepEPDispatchResult(DispatchResult):
    handle: DeepEPHandle
    topk_ids: torch.Tensor
    num_recv_tokens_per_expert_list: list[int]
    forward_finished_event: EventOverlap | None


class DeepEPPostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class DeepEPPreCombineResult(PreCombineResult):
    backward_previous_event: EventOverlap | None
    forward_finished_event: EventOverlap | None


class DeepEPCombineResult(CombineResult):
    forward_finished_event: EventOverlap | None
    backward_previous_event: EventOverlap | None


DeepEPPostCombineResult = PostCombineResult


HiddenStates: TypeAlias = torch.Tensor


class DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        group: dist.ProcessGroup,
        forward_previous_event: EventOverlap | None = None,
        backward_finished_event: EventOverlap | None = None,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list,
        tuple,
        EventOverlap,
    ]:
        if (forward_previous_event is None) != (backward_finished_event is None):
            raise ValueError(
                "Internal Error! `forward_previous_event` and `backward_finished_event` should be both None or both "
                "not None"
            )
        is_async = forward_previous_event is not None
        ctx.is_async = is_async
        if not is_async:
            forward_previous_event = buffer_capture()
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = dispatch_forward(x, topk_idx, topk_weights, num_experts, group, forward_previous_event)
        # save deep comm handle
        if not is_async:
            event.current_stream_wait()
        ctx.save_for_backward(*handle)
        ctx.group = group
        ctx.num_experts = num_experts
        ctx.backward_finished_event = backward_finished_event
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
    ) -> tuple[torch.Tensor, None, torch.Tensor | None, None, None, None, None, None, None]:
        # load saved comm handle
        handle = ctx.saved_tensors
        combined_grad_x, combined_grad_recv_topk_weights, event = dispatch_backward(
            grad_recv_x, grad_recv_topk_weights, ctx.num_experts, handle, ctx.group, buffer_capture()
        )
        if not ctx.is_async:
            event.current_stream_wait()
        else:
            ctx.backward_finished_event.event = event.event
        return (
            combined_grad_x,
            None,
            combined_grad_recv_topk_weights,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_async_dispatch = copy_method_signature(DeepEPDispatch.forward)(DeepEPDispatch.apply)


class DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        num_experts: int,
        handle: DeepEPHandle,
        group: dist.ProcessGroup,
        forward_previous_event: EventOverlap | None = None,
        backward_previous_event: EventOverlap | None = None,
        backward_finished_event: EventOverlap | None = None,
    ) -> tuple[torch.Tensor, EventOverlap]:
        if not (
            (forward_previous_event is None) == (backward_finished_event is None) == (backward_previous_event is None)
        ):
            raise ValueError(
                "Internal Error! `forward_previous_event`, `backward_finished_event` and `backward_previous_event` "
                "should be all None or all not None"
            )
        is_async = forward_previous_event is not None
        ctx.is_async = is_async

        if not is_async:
            forward_previous_event = buffer_capture()

        combined_x, event = combine_forward(x, num_experts, handle, group, forward_previous_event)

        if not is_async:
            event.current_stream_wait()

        # save deep comm handle
        ctx.save_for_backward(*handle)
        ctx.group = group
        ctx.num_experts = num_experts
        ctx.backward_finished_event = backward_finished_event
        ctx.backward_previous_event = backward_previous_event
        return combined_x, event

    @staticmethod
    def backward(  # type: ignore[invalid-override]
        ctx, grad_combined_x: torch.Tensor, *args
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None, None, None, None, None]:
        # load saved comm handle
        handle = ctx.saved_tensors
        if not ctx.is_async:
            previous_event = buffer_capture()
        else:
            previous_event = ctx.backward_previous_event

        grad_x, event = combine_backward(grad_combined_x, ctx.num_experts, handle, ctx.group, previous_event)

        if not ctx.is_async:
            event.current_stream_wait()
        else:
            ctx.backward_finished_event.event = event.event
        return grad_x, None, None, None, None, None, None


_async_combine = copy_method_signature(DeepEPCombine.forward)(DeepEPCombine.apply)


def get_backward_pre_hook(backward_previous_event: EventOverlap, name: str | None = None, debug: bool = False):
    def _backward_pre_hook(*_):
        if debug:
            logger.info(f"[{name}] backward pre hook")
        if backward_previous_event is not None:
            backward_previous_event.current_stream_wait()

    return _backward_pre_hook


def get_backward_hook(backward_finished_event: EventOverlap, name: str | None = None, debug: bool = False):
    def _backward_hook(*_):
        if debug:
            logger.info(f"[{name}] backward hook")
        if backward_finished_event is not None:
            event = buffer_capture()
            backward_finished_event.event = event.event

    return _backward_hook


class DeepEPDispatcher(
    GenericDispatcher[
        DeepEPPreDispatchResult,
        DeepEPDispatchResult,
        DeepEPPostDispatchResult,
        DeepEPPreCombineResult,
        DeepEPCombineResult,
        DeepEPPostCombineResult,
    ]
):
    _comm_stream = None
    _process_group: dist.ProcessGroup

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        if not is_installed("deep_ep"):
            raise RuntimeError("`DeepEP` is not installed!")
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        assert self._process_group is not None, (
            "Process group must be provided for `DeepEPDispatcher`. "
            "If you are training a MoE model, it means that `expert parallel` is not enabled in the config."
        )

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> DeepEPPreDispatchResult:
        if async_op:
            backward_previous_event = EventOverlap(None)
            forward_finished_event = buffer_capture()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllDispatcher.dispatch_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return DeepEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int64),
            backward_previous_event=backward_previous_event,
            forward_finished_event=forward_finished_event,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPDispatchResult:
        (
            dispatched_hidden_states,
            dispatched_topk_idx,
            dispatched_topk_weights,
            num_recv_tokens_per_expert_list,
            dispatch_handle,
            event,
        ) = _async_dispatch(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            topk_weights,
            self._n_routed_experts,
            self._process_group,
            pre_dispatched["forward_finished_event"],
            pre_dispatched["backward_previous_event"],
        )

        if not async_op:
            event.current_stream_wait()
            forward_finished_event = None
        else:
            forward_finished_event = event

        ret = DeepEPDispatchResult(
            hidden_states=cast(HiddenStates, dispatched_hidden_states),
            topk_weights=dispatched_topk_weights,
            topk_ids=dispatched_topk_idx,
            handle=dispatch_handle,
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
            forward_finished_event=forward_finished_event,
        )
        return ret

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatched: DeepEPDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPPostDispatchResult:
        if async_op:
            assert dispatched["forward_finished_event"] is not None, "Please use `async_op=True` for dispatch!"
            dispatched["forward_finished_event"].current_stream_wait()

        num_recv_tokens_per_expert_list = dispatched["num_recv_tokens_per_expert_list"]
        num_out_tokens = sum(dispatched["num_recv_tokens_per_expert_list"])
        recv_topk_idx_numel = dispatched["topk_ids"].numel()
        num_neg_one_idx = recv_topk_idx_numel - num_out_tokens

        permuted_hidden_states, row_ids_map = permute(
            dispatched["hidden_states"],
            dispatched["topk_ids"].int(),
            num_out_tokens=num_out_tokens,
            num_negative_one_in_indices=num_neg_one_idx,
        )
        tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            dtype=torch.long,
            device=dispatched["topk_weights"].device,
        )

        if decoding:
            raise NotImplementedError
        else:
            return DeepEPPostDispatchResult(
                hidden_states=permuted_hidden_states,
                row_ids_map=row_ids_map,
                tokens_per_expert=tokens_per_expert,
            )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatched: DeepEPDispatchResult,
        post_dispatched: DeepEPPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPPreCombineResult:
        hidden_states = unpermute(
            hidden_states,
            post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )

        if async_op:
            backward_previous_event = EventOverlap(None)
            forward_finished_event = buffer_capture()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllDispatcher.combine_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            backward_previous_event = None
            forward_finished_event = None

        if decoding:
            raise NotImplementedError
        else:
            return DeepEPPreCombineResult(
                hidden_states=hidden_states,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )

    @override
    def combine(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatched: DeepEPDispatchResult,
        post_dispatched: DeepEPPostDispatchResult,
        pre_combined: DeepEPPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> CombineResult:
        if async_op:
            backward_previous_event = EventOverlap(None)
            assert pre_combined["forward_finished_event"] is not None, "Please use `async_op=True` for combine!"
            pre_combined["forward_finished_event"].current_stream_wait()
        else:
            backward_previous_event = None

        combined_hidden_states, event = _async_combine(
            pre_combined["hidden_states"],
            self._n_routed_experts,
            dispatched["handle"],
            self._process_group,
            pre_combined["forward_finished_event"],
            backward_previous_event,
            pre_combined["backward_previous_event"],
        )
        if not async_op:
            event.current_stream_wait()

        if not decoding:
            return DeepEPCombineResult(
                hidden_states=combined_hidden_states,
                forward_finished_event=event,
                backward_previous_event=backward_previous_event,
            )
        else:
            raise NotImplementedError

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatched: DeepEPDispatchResult,
        post_dispatched: DeepEPPostDispatchResult,
        pre_combined: DeepEPPreCombineResult,
        combined: DeepEPCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        hidden_states = combined["hidden_states"]
        forward_previous_event = combined["forward_finished_event"]

        hidden_states = hidden_states.view_as(hidden_states)

        if hidden_states.grad_fn is not None:
            hidden_states.grad_fn.register_hook(
                get_backward_hook(
                    backward_finished_event=combined["backward_previous_event"],
                    name="DeeEPDispatcher.combine_postprocess",
                    debug=XTUNER_DISPATCHER_DEBUG,
                )
            )

        if async_op:
            assert forward_previous_event is not None, "Please use `async_op=True` for combine!"
            forward_previous_event.current_stream_wait()
        return PostCombineResult(hidden_states=hidden_states)
