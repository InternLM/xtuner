import os
from typing import Literal, TypeAlias, cast

import torch
import torch.distributed as dist
from torch.autograd.function import Function
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_tensor,
    all_gather_tensor_autograd,
    all_to_all_single_autograd,
    reduce_scatter_tensor,
    reduce_scatter_tensor_autograd,
)
from typing_extensions import override

from xtuner.v1.ops import permute, unpermute
from xtuner.v1.ops.comm import AllGatherManager, ReduceScatterManager, SymmBufferManager
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


DEVICE = get_device()
logger = get_logger()


USE_CUSTOM_AG = int(os.getenv("XTUNER_USE_CUSTOM_AG_IN_DISPATCHER", 0)) == 1
USE_CUSTOM_RS = int(os.getenv("XTUNER_USE_CUSTOM_RS_IN_DISPATCHER", 0)) == 1

ag_symm = None
rs_symm = None
ag_manager = None
rs_manager = None
rs_event = None


MoEAGRSHandle = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# MoEAGRS handle include 6 tensor:
# (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
class MoEAGRSPreDispatchResult(PreDispatchResult):
    backward_previous_event: torch.cuda.Event | None
    forward_finished_event: torch.cuda.Event | None


class MoEAGRSDispatchResult(DispatchResult):
    topk_ids: torch.Tensor
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class MoEAGRSPostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class MoEAGRSPreCombineResult(PreCombineResult):
    backward_previous_event: torch.cuda.Event | None
    forward_finished_event: torch.cuda.Event | None


class MoEAGRSCombineResult(CombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


MoEAGRSPostCombineResult = PostCombineResult


HiddenStates: TypeAlias = torch.Tensor


def get_backward_pre_hook(backward_previous_event: torch.cuda.Event, name: str | None = None, debug: bool = False):
    def _backward_pre_hook(*_):
        # if name == "TorchAll2AllDispatcher.dispatch_preprocess":
        #     torch.cuda.synchronize()
        if debug:
            logger.info(f"[{name}] backward pre hook")
        if backward_previous_event is not None:
            torch.cuda.current_stream().wait_event(backward_previous_event)

    return _backward_pre_hook


def get_backward_hook(backward_finished_event: torch.cuda.Event, name: str | None = None, debug: bool = False):
    def _backward_hook(*_):
        if debug:
            logger.info(f"[{name}] backward hook")
        if backward_finished_event is not None:
            backward_finished_event.record()

    return _backward_hook


class _AsyncDispatch(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)

            if USE_CUSTOM_AG:
                global ag_manager, ag_symm
                if ag_symm is None:
                    ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=2)
                if ag_manager is None:
                    # agrs dispatcher do not need to select comm sm
                    ag_manager = AllGatherManager(num_buffers=2, select_comm_sm=False)
                send_bytes = hidden_states.element_size() * hidden_states.numel()
                recv_bytes = send_bytes * process_group.size()
                recv_numel = hidden_states.numel() * process_group.size()

                ag_manager.prepare_allgather_objects(
                    send_bytes=send_bytes,
                    process_group=process_group,
                    all_gather_stream=comm_stream,
                    barrier_all=False,  # intra-node comm, no need to barrier across nodes
                )
                device = hidden_states.device
                dtype = hidden_states.dtype
                combined_grad_out_symm = ag_symm.get_buffer(bytes=recv_bytes, device=device)
                combined_grad_out_symm = combined_grad_out_symm.view(dtype)[:recv_numel]

                ag_manager.execute_allgather(
                    send_bytes=send_bytes,
                    all_gather_output=combined_grad_out_symm,
                    all_gather_input=hidden_states,
                    process_group=process_group,
                )
                dispatched_hidden_states = combined_grad_out_symm.view(-1, *hidden_states.shape[1:])
            else:
                dispatched_hidden_states = all_gather_tensor_autograd(hidden_states, gather_dim=0, group=process_group)
                if isinstance(dispatched_hidden_states, AsyncCollectiveTensor):
                    dispatched_hidden_states = dispatched_hidden_states.wait()

            # topk_ids (seq, topk)
            topk_ids = topk_ids.T.flatten()
            dispatched_topk_ids = torch.empty_like(topk_ids)
            dist.all_to_all_single(
                dispatched_topk_ids,
                topk_ids,
                group=process_group,
            )
            dispatched_topk_ids = dispatched_topk_ids.view(-1, 1)
            topk_weights = topk_weights.T.flatten()
            dispatched_topk_weights = torch.empty_like(topk_weights)
            dist.all_to_all_single(
                dispatched_topk_weights,
                topk_weights,
                group=process_group,
            )
            dispatched_topk_weights = dispatched_topk_weights.view(-1, 1)

            dispatched_hidden_states.record_stream(comm_stream)
            dispatched_topk_ids.record_stream(comm_stream)
            dispatched_topk_weights.record_stream(comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        ctx.process_group = process_group
        ctx.comm_stream = comm_stream
        return dispatched_hidden_states, dispatched_topk_ids, dispatched_topk_weights

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, grad_topk_ids: torch.Tensor | None, grad_topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, torch.Tensor | None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.process_group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)

            if USE_CUSTOM_RS:
                global rs_manager, rs_symm
                if rs_symm is None:
                    rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
                if rs_manager is None:
                    rs_manager = ReduceScatterManager(num_buffers=2, select_comm_sm=False)

                process_group = ctx.process_group
                comm_stream = ctx.comm_stream

                send_bytes = grad_output.element_size() * grad_output.numel()
                recv_bytes = send_bytes // process_group.size()
                send_numel = grad_output.numel()

                rs_manager.prepare_reducescatter_objects(
                    recv_bytes=recv_bytes,
                    process_group=process_group,
                    reduce_scatter_stream=comm_stream,
                    barrier_all=False,  # intra-node comm, no need to barrier across nodes
                )
                device = grad_output.device
                dtype = grad_output.dtype
                symm_input = rs_symm.get_buffer(bytes=send_bytes, device=device)
                symm_input = symm_input.view(dtype)[:send_numel]
                symm_input = symm_input.view(grad_output.shape)

                reduce_scatter_output_numel = recv_bytes // symm_input.element_size()
                reduce_output = symm_input.new_empty((reduce_scatter_output_numel,))
                combined_grad_output = rs_manager.execute_reducescatter(
                    recv_bytes=recv_bytes,
                    reduce_scatter_output=reduce_output,
                    reduce_scatter_input=symm_input,
                    reduce_scatter_group=process_group,
                    reduce_scatter_reduce_op=dist.ReduceOp.SUM,
                )
                combined_grad_output = combined_grad_output.view(-1, *grad_output.shape[1:])
            else:
                combined_grad_output = reduce_scatter_tensor(
                    grad_output, reduceOp="sum", scatter_dim=0, group=ctx.process_group
                )

            grad_output.record_stream(ctx.comm_stream)
            combined_grad_output.record_stream(ctx.comm_stream)

            world_size = dist.get_world_size(group=ctx.process_group)
            grad_topk_weights = grad_topk_weights.view(-1)
            combined_grad_topk_weights = torch.empty_like(grad_topk_weights)
            dist.all_to_all_single(
                combined_grad_topk_weights,
                grad_topk_weights,
                group=ctx.process_group,
            )
            combined_grad_topk_weights = combined_grad_topk_weights.view(world_size, -1)
            combined_grad_topk_weights = combined_grad_topk_weights.T.contiguous()

            # grad_topk_weights and combined_grad_topk_weights must record_stream, this is very important
            grad_topk_weights.record_stream(ctx.comm_stream)
            combined_grad_topk_weights.record_stream(ctx.comm_stream)
            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return combined_grad_output, None, combined_grad_topk_weights, None, None, None, None, None, None


_async_dispatch = copy_method_signature(_AsyncDispatch.forward)(_AsyncDispatch.apply)


class _AsyncCombine(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)

            if USE_CUSTOM_RS:
                global rs_manager, rs_symm, rs_event
                if rs_symm is None:
                    rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
                if rs_manager is None:
                    rs_manager = ReduceScatterManager(num_buffers=2, select_comm_sm=False)

                send_bytes = hidden_states.element_size() * hidden_states.numel()
                recv_bytes = send_bytes // process_group.size()
                send_numel = hidden_states.numel()

                rs_manager.prepare_reducescatter_objects(
                    recv_bytes=recv_bytes,
                    process_group=process_group,
                    reduce_scatter_stream=comm_stream,
                    barrier_all=False,  # intra-node comm, no need to barrier across nodes
                )
                device = hidden_states.device
                dtype = hidden_states.dtype
                symm_input = rs_symm.get_buffer(bytes=send_bytes, device=device)
                symm_input = symm_input.view(dtype)[:send_numel]
                symm_input = symm_input.view(hidden_states.shape)

                reduce_scatter_output_numel = recv_bytes // symm_input.element_size()
                reduce_output = symm_input.new_empty((reduce_scatter_output_numel,))
                rs_manager.execute_reducescatter(
                    recv_bytes=recv_bytes,
                    reduce_scatter_output=reduce_output,
                    reduce_scatter_input=symm_input,
                    reduce_scatter_group=process_group,
                    reduce_scatter_reduce_op=dist.ReduceOp.SUM,
                )
                combined_hidden_states = reduce_output.view(-1, *hidden_states.shape[1:])
                rs_event = comm_stream.record_event()
            else:
                combined_hidden_states = reduce_scatter_tensor_autograd(
                    hidden_states, reduceOp="sum", scatter_dim=0, group=process_group
                )
                if isinstance(combined_hidden_states, AsyncCollectiveTensor):
                    combined_hidden_states = combined_hidden_states.wait()

            forward_finished_event.record(comm_stream)

        ctx.comm_stream = comm_stream
        ctx.process_group = process_group

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event
        return combined_hidden_states

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, *args
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.process_group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)

            if USE_CUSTOM_AG:
                global ag_manager, ag_symm
                if ag_symm is None:
                    ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=2)
                if ag_manager is None:
                    # agrs dispatcher do not need to select comm sm
                    ag_manager = AllGatherManager(num_buffers=2, select_comm_sm=False)

                send_bytes = grad_output.element_size() * grad_output.numel()
                recv_bytes = send_bytes * ctx.process_group.size()
                recv_numel = grad_output.numel() * ctx.process_group.size()

                ag_manager.prepare_allgather_objects(
                    send_bytes=send_bytes,
                    process_group=ctx.process_group,
                    all_gather_stream=ctx.comm_stream,
                    barrier_all=False,  # intra-node comm, no need to barrier across nodes
                )
                device = grad_output.device
                dtype = grad_output.dtype
                combined_grad_out_symm = ag_symm.get_buffer(bytes=recv_bytes, device=device)
                combined_grad_out_symm = combined_grad_out_symm.view(dtype)[:recv_numel]

                ag_manager.execute_allgather(
                    send_bytes=send_bytes,
                    all_gather_output=combined_grad_out_symm,
                    all_gather_input=grad_output,
                    process_group=ctx.process_group,
                )
                combined_grad_output = combined_grad_out_symm.view(-1, *grad_output.shape[1:])
            else:
                combined_grad_output = all_gather_tensor(grad_output, gather_dim=0, group=ctx.process_group)

            grad_output.record_stream(ctx.comm_stream)
            combined_grad_output.record_stream(ctx.comm_stream)

            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return combined_grad_output, None, None, None, None, None, None


_async_combine = copy_method_signature(_AsyncCombine.forward)(_AsyncCombine.apply)


class MoEAGRSDispatcher(
    GenericDispatcher[
        MoEAGRSPreDispatchResult,
        MoEAGRSDispatchResult,
        MoEAGRSPostDispatchResult,
        MoEAGRSPreCombineResult,
        MoEAGRSCombineResult,
        MoEAGRSPostCombineResult,
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
        self._experts_per_rank = self._n_routed_experts // self._process_group.size()
        if MoEAGRSDispatcher._comm_stream is None:
            MoEAGRSDispatcher._comm_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=DEVICE))

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> MoEAGRSPreDispatchResult:
        if async_op:
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="AGRSDispatcher.dispatch_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return MoEAGRSPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int64),
            backward_previous_event=backward_previous_event,
            forward_finished_event=forward_finished_event,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSDispatchResult:
        if not async_op:
            hidden_states = pre_dispatched["hidden_states"]
            topk_ids = pre_dispatched["topk_ids"]

            dispatched_hidden_states = all_gather_tensor_autograd(
                hidden_states, gather_dim=0, group=self._process_group
            )
            if isinstance(dispatched_hidden_states, AsyncCollectiveTensor):
                dispatched_hidden_states = dispatched_hidden_states.wait()

            # topk_ids (seq, topk)
            topk_ids = topk_ids.T.flatten()
            dispatched_topk_ids = torch.empty_like(topk_ids)
            dist.all_to_all_single(
                dispatched_topk_ids,
                topk_ids,
                group=self._process_group,
            )
            dispatched_topk_ids = dispatched_topk_ids.view(-1, 1)
            topk_weights = topk_weights.T.flatten()
            dispatched_topk_weights = all_to_all_single_autograd(
                topk_weights,
                output_split_sizes=None,
                input_split_sizes=None,
                group=self._process_group,
            )
            dispatched_topk_weights = dispatched_topk_weights.view(-1, 1)

            return MoEAGRSDispatchResult(
                hidden_states=cast(HiddenStates, dispatched_hidden_states),
                topk_weights=dispatched_topk_weights,
                topk_ids=dispatched_topk_ids,
                forward_finished_event=None,
                backward_previous_event=None,
            )
        else:
            forward_previous_event = pre_dispatched["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_dispatched["backward_previous_event"]
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            dispatched_hidden_states, dispatched_topk_idx, dispatched_topk_weights = _async_dispatch(
                pre_dispatched["hidden_states"],
                pre_dispatched["topk_ids"],
                topk_weights,
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )
            return MoEAGRSDispatchResult(
                hidden_states=cast(HiddenStates, dispatched_hidden_states),
                topk_weights=dispatched_topk_weights,
                topk_ids=dispatched_topk_idx,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSPostDispatchResult:
        if async_op:
            assert dispatched["forward_finished_event"] is not None, "Please use `async_op=True` for dispatch!"
            self.wait_comm_stream(dispatched["forward_finished_event"])

        permuted_hidden_states, row_ids_map = permute(
            dispatched["hidden_states"],
            dispatched["topk_ids"].to(torch.int32),
        )

        topk_ids = dispatched["topk_ids"]
        rank = dist.get_rank(group=self._process_group)
        tokens_per_expert = torch.histc(
            topk_ids,
            bins=self._experts_per_rank,
            min=rank * self._experts_per_rank,
            max=(rank + 1) * self._experts_per_rank,
        )

        if async_op:
            assert dispatched["backward_previous_event"] is not None, "Please use `async_op=True` for dispatch!"
            if permuted_hidden_states.grad_fn is not None:
                permuted_hidden_states.grad_fn.register_hook(
                    get_backward_hook(
                        dispatched["backward_previous_event"],
                        name="TorchAll2AllDispatcher.dispatch_posAGRSrocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        return MoEAGRSPostDispatchResult(
            hidden_states=permuted_hidden_states,
            row_ids_map=row_ids_map,
            tokens_per_expert=tokens_per_expert,
        )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoEAGRSPreCombineResult:
        hidden_states = unpermute(
            hidden_states,
            post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )

        if async_op:
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllDispatcher.combine_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return MoEAGRSPreCombineResult(
            hidden_states=hidden_states,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        pre_combined: MoEAGRSPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> CombineResult:
        if async_op:
            forward_previous_event = pre_combined["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_combined["backward_previous_event"]
            assert forward_previous_event is not None, "Please use `async_op=True` for combine_preprocess!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine_preprocess!"

            combined_hidden_states = _async_combine(
                pre_combined["hidden_states"],
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )
        else:
            forward_finished_event = None
            backward_previous_event = None
            hidden_states = pre_combined["hidden_states"]
            combined_hidden_states = reduce_scatter_tensor_autograd(
                hidden_states, reduceOp="sum", scatter_dim=0, group=self._process_group
            )
            if isinstance(combined_hidden_states, AsyncCollectiveTensor):
                combined_hidden_states = combined_hidden_states.wait()

        return MoEAGRSCombineResult(
            hidden_states=combined_hidden_states,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: MoEAGRSPreDispatchResult,
        dispatched: MoEAGRSDispatchResult,
        post_dispatched: MoEAGRSPostDispatchResult,
        pre_combined: MoEAGRSPreCombineResult,
        combined: MoEAGRSCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        hidden_states = combined["hidden_states"]

        if async_op:
            forward_previous_event = combined["forward_finished_event"]
            backward_finished_event = combined["backward_previous_event"]
            assert forward_previous_event is not None, "Please use `async_op=True` for combine!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine!"
            self.wait_comm_stream(forward_previous_event)

            hidden_states = hidden_states.view_as(hidden_states)

            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(
                    get_backward_hook(
                        backward_finished_event=cast(torch.cuda.Event, combined["backward_previous_event"]),
                        name="DeeEPDispatcher.combine_postprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        return PostCombineResult(hidden_states=hidden_states)

    def wait_comm_stream(self, event: torch.cuda.Event):
        torch.cuda.current_stream().wait_event(event)
