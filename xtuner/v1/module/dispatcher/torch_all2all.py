from typing import Literal, TypeAlias, cast

import torch
import torch.distributed as dist
from torch.autograd.function import Function
from typing_extensions import override

from xtuner.v1.ops import permute, unpermute
from xtuner.v1.ops.comm.all_to_all import all_to_all_single_autograd
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


# The execution steps of Torch all to all are as follows:
# 1. (preprocess) Sort the hidden states along the expert dimension (permute) to facilitate torch all2all, and record the
#                 row id map, which is used to restore the hidden states after combine.
# 2. (preprocess) Communicate routing information among ranks within the ep group, calculate the input and output layout for torch all2all.
# 3. (dispatch) Call torch all2all for communication between experts, obtaining expert-ordered hidden states.
# 4. (dispatch) Call the permute function again to adjust the hidden states into an order suitable for grouped gemm forward. Also record the row_id_map
#               row_id_map is used to restore the hidden states after combine.
# 5. (combine) Unpermute the hidden states after experts forward to get the original hidden states.
# 6. (combine) Call torch all2all combine to merge hidden states from different ranks.
# 7. (post process) Unpermute the hidden states after combine to get the final hidden states.


class TorchAll2AllPreDispatchResult(PreDispatchResult):
    row_id_map: torch.Tensor
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class TorchAll2AllDispatchResult(DispatchResult):
    tokens_per_expert_group: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class TorchAll2AllPostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class TorchAll2AllPreCombineResult(PreCombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class TorchAll2AllCombineResult(CombineResult):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class TorchAll2AllPostCombineResult(PostCombineResult): ...


HiddenStates: TypeAlias = torch.Tensor


def _dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    n_routed_experts: int,
    process_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    ep_size = process_group.size()
    num_experts_per_rank = n_routed_experts // ep_size
    tokens_per_expert = torch.histc(topk_ids, bins=n_routed_experts, min=0, max=n_routed_experts)
    # self._comm_stream.wait_event(event)
    tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
    dist.all_to_all_single(
        tokens_per_expert_group,
        tokens_per_expert,
        group=process_group,
    )

    # (r0e0, r0e1, ..., r0ei-1,
    #  r1e0, r1e1, ..., r1ei-1,
    tokens_per_expert_group = tokens_per_expert_group.view(ep_size, -1)

    # Get number experts each group
    input_splits = (
        tokens_per_expert.reshape(ep_size, num_experts_per_rank).to(device=torch.device("cpu")).sum(dim=1).tolist()
    )
    output_splits = tokens_per_expert_group.to(device=torch.device("cpu")).sum(dim=-1).tolist()

    hidden_size = hidden_states.size(-1)
    out = hidden_states.new_empty(size=(sum(output_splits), hidden_size))

    hidden_states = hidden_states.contiguous()

    dist.all_to_all_single(
        out,
        hidden_states,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=process_group,
    )
    return out, tokens_per_expert_group, input_splits, output_splits


class _AsyncDispatch(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        n_routed_experts: int,
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            out, tokens_per_expert, input_splits, output_splits = _dispatch(
                hidden_states,
                topk_ids,
                n_routed_experts,
                process_group,
            )
            out.record_stream(comm_stream)
            tokens_per_expert.record_stream(comm_stream)
            forward_finished_event.record(comm_stream)

        ctx.input_shape = hidden_states.shape
        ctx.output_split_sizes = output_splits
        ctx.input_split_sizes = input_splits

        ctx.comm_stream = comm_stream
        ctx.group = process_group

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event

        return out, tokens_per_expert, input_splits, output_splits

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, *args
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            # ctx.comm_stream.wait_stream(compute_stream)
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            dist.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            # NOTE: During backward, `grad_output` will be freed before dispatch backward finished if we do not record
            # the `grad_output` here.
            grad_output.record_stream(ctx.comm_stream)
            out.record_stream(ctx.comm_stream)

            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return out, None, None, None, None, None, None, None, None


_async_dispatch = copy_method_signature(_AsyncDispatch.forward)(_AsyncDispatch.apply)


class _AsyncCombine(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
        forward_previous_event: torch.cuda.Event,
        forward_finished_event: torch.cuda.Event,
        backward_previous_event: torch.cuda.Event,
        backward_finished_event: torch.cuda.Event,
        comm_stream: torch.cuda.Stream,
        process_group: dist.ProcessGroup,
    ):
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(forward_previous_event)
            out = all_to_all_single_autograd(
                hidden_states,
                input_split_sizes=input_splits,
                output_split_sizes=output_splits,
                group=process_group,
            )
            forward_finished_event.record(comm_stream)

        ctx.input_shape = hidden_states.shape
        ctx.output_split_sizes = output_splits
        ctx.input_split_sizes = input_splits

        ctx.comm_stream = comm_stream
        ctx.group = process_group

        ctx.backward_previous_event = backward_previous_event
        ctx.backward_finished_event = backward_finished_event

        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, *args
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None, None, None]:
        world_size = dist.get_world_size(group=ctx.group)
        if world_size == 1:
            return grad_output, None, None, None, None, None, None, None, None

        with torch.cuda.stream(ctx.comm_stream):
            if ctx.backward_previous_event is not None:
                ctx.comm_stream.wait_event(ctx.backward_previous_event)
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            dist.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            # NOTE: During backward, `grad_output` will be freed before dispatch backward finished if we do not record
            # the `grad_output` here.
            grad_output.record_stream(ctx.comm_stream)
            out.record_stream(ctx.comm_stream)

            if ctx.backward_finished_event is not None:
                ctx.backward_finished_event.record(ctx.comm_stream)
        return out, None, None, None, None, None, None, None, None


_async_combine = copy_method_signature(_AsyncCombine.forward)(_AsyncCombine.apply)


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


class TorchAll2AllDispatcher(
    GenericDispatcher[
        TorchAll2AllPreDispatchResult,
        TorchAll2AllDispatchResult,
        TorchAll2AllPostDispatchResult,
        TorchAll2AllPreCombineResult,
        TorchAll2AllCombineResult,
        TorchAll2AllPostCombineResult,
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
            "Process group must be provided for `TorchAll2AllDispatcher`. "
            "If you are training a MoE model, it means that `expert parallel` is not enabled in the config."
        )
        self._experts_per_rank = self._n_routed_experts // self._process_group.size()
        # Repeat the experts indicex for all to all dispatch
        self._expert_ids_per_ep_rank = torch.tensor(
            [i % self._experts_per_rank for i in range(self._n_routed_experts)],
            dtype=torch.int32,
            device="cuda",
        )
        if TorchAll2AllDispatcher._comm_stream is None:
            TorchAll2AllDispatcher._comm_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=DEVICE))
        # if training_dtype == "fp8":
        #     raise NotImplementedError

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> TorchAll2AllPreDispatchResult:
        permuted_hidden_states, row_ids_map = permute(hidden_states, topk_ids.to(torch.int32))

        if async_op:
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            forward_finished_event.record()
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            if permuted_hidden_states.grad_fn is not None:
                permuted_hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="TorchAll2AllDispatcher.dispatch_preprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            forward_finished_event = None
            backward_previous_event = None

        return TorchAll2AllPreDispatchResult(
            hidden_states=permuted_hidden_states,
            row_id_map=row_ids_map,
            topk_ids=topk_ids,
            forward_finished_event=forward_finished_event,
            backward_previous_event=backward_previous_event,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllDispatchResult:
        if not async_op:
            hidden_states, tokens_per_expert_group, input_splits, output_splits = _dispatch(
                pre_dispatched["hidden_states"],
                pre_dispatched["topk_ids"],
                self._n_routed_experts,
                self._process_group,
            )
            if decoding:
                raise NotImplementedError
            else:
                return TorchAll2AllDispatchResult(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    tokens_per_expert_group=cast(torch.Tensor, tokens_per_expert_group),
                    input_splits=cast(list[int], input_splits),
                    output_splits=cast(list[int], output_splits),
                    forward_finished_event=None,
                    backward_previous_event=None,
                )
        else:
            forward_previous_event = pre_dispatched["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_dispatched["backward_previous_event"]
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())

            assert backward_finished_event is not None, "Please use `async_op=True` for dispatch_preprocess!"
            assert forward_previous_event is not None, "Please use `async_op=True` for dispatch_preprocess!"

            hidden_states, tokens_per_expert_group, input_splits, output_splits = _async_dispatch(
                pre_dispatched["hidden_states"],
                pre_dispatched["topk_ids"],
                self._n_routed_experts,
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )
            if decoding:
                raise NotImplementedError
            else:
                return TorchAll2AllDispatchResult(
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    tokens_per_expert_group=tokens_per_expert_group,
                    input_splits=cast(list[int], input_splits),
                    output_splits=cast(list[int], output_splits),
                    backward_previous_event=backward_previous_event,
                    forward_finished_event=forward_finished_event,
                )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> TorchAll2AllPostDispatchResult:
        if async_op:
            assert dispatched["forward_finished_event"] is not None, "Please use `async_op=True` for dispatch!"
            self.wait_comm_stream(dispatched["forward_finished_event"])

        tokens_per_expert_group = dispatched["tokens_per_expert_group"]
        token_counts = tokens_per_expert_group.ravel()
        global_input_tokens_local_experts_indices = torch.repeat_interleave(
            self._expert_ids_per_ep_rank, token_counts, output_size=sum(dispatched["output_splits"])
        )

        # The dispatch result is already permuted, so we can return it directly.
        global_input_tokens, row_ids_map = permute(
            dispatched["hidden_states"],
            global_input_tokens_local_experts_indices.to(torch.int32),
        )
        tokens_per_expert = tokens_per_expert_group.sum(dim=0)

        if async_op:
            assert dispatched["backward_previous_event"] is not None, "Please use `async_op=True` for dispatch!"

            if global_input_tokens.grad_fn is not None:
                global_input_tokens.grad_fn.register_hook(
                    get_backward_hook(
                        dispatched["backward_previous_event"],
                        name="TorchAll2AllDispatcher.dispatch_postprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        if decoding:
            raise NotImplementedError
        else:
            return TorchAll2AllPostDispatchResult(
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
        hidden_states = unpermute(
            hidden_states,
            post_dispatched["row_ids_map"],
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
            backward_previous_event = None
            forward_finished_event = None

        if decoding:
            raise NotImplementedError
        else:
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
    ) -> CombineResult:
        if not async_op:
            hidden_states = all_to_all_single_autograd(
                pre_combined["hidden_states"],
                input_split_sizes=dispatched["output_splits"],
                output_split_sizes=dispatched["input_splits"],
                group=self._process_group,
            )
            forward_finished_event = None
            backward_previous_event = None
        else:
            forward_previous_event = pre_combined["forward_finished_event"]
            forward_finished_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_previous_event = cast(torch.cuda.Event, torch.cuda.Event())
            backward_finished_event = pre_combined["backward_previous_event"]

            assert forward_previous_event is not None, "Please use `async_op=True` for combine_preprocess!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine_preprocess!"

            hidden_states = _async_combine(
                pre_combined["hidden_states"],
                dispatched["output_splits"],
                dispatched["input_splits"],
                forward_previous_event,
                forward_finished_event,
                backward_previous_event,
                backward_finished_event,
                self._comm_stream,
                self._process_group,
            )

        if not decoding:
            return TorchAll2AllCombineResult(
                hidden_states=hidden_states,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )
        else:
            raise NotImplementedError

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatched: TorchAll2AllDispatchResult,
        post_dispatched: TorchAll2AllPostDispatchResult,
        pre_combined: TorchAll2AllPreCombineResult,
        combined: TorchAll2AllCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        if not async_op:
            hidden_states = unpermute(
                combined["hidden_states"],
                pre_dispatched["row_id_map"],
                probs=dispatched["topk_weights"],
            )
            backward_finished_event = None
        else:
            forward_previous_event = combined["forward_finished_event"]
            backward_finished_event = combined["backward_previous_event"]
            assert forward_previous_event is not None, "Please use `async_op=True` for combine!"
            assert backward_finished_event is not None, "Please use `async_op=True` for combine!"
            self.wait_comm_stream(forward_previous_event)
            hidden_states = unpermute(
                combined["hidden_states"],
                pre_dispatched["row_id_map"],
                probs=dispatched["topk_weights"],
            )
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(
                    get_backward_hook(
                        backward_finished_event=backward_finished_event,
                        name="TorchAll2AllDispatcher.combine_postprocess",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        return PostCombineResult(hidden_states=hidden_states)

    def wait_comm_stream(self, event: torch.cuda.Event):
        torch.cuda.current_stream().wait_event(event)
