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
    # Final ``topk_weights`` fed to DeepEP. Equal to the caller's ``topk_weights`` for ep-only
    # routing; for virtual expert TP (``tp_size > 1``) it is ``repeat_interleave``'d here in
    # ``dispatch_preprocess`` so the expand kernel runs on the compute stream during Loop A
    # (overlapping the next microbatch's attention/gate) instead of inside ``dispatch``.
    topk_weights: torch.Tensor
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
        tp_size: int = 1,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        """DeepEP-backed MoE dispatcher.

        When ``tp_size > 1`` the dispatcher fuses expert-parallel dispatch and tensor-parallel
        token replication into a single DeepEP collective. The caller must:

          * Build the combined ``(ep × tp)`` process group via ``ep_tp_mesh._flatten().get_group()``
            (mesh dims ordered with ``tp`` as the inner/fastest dim) and pass it as
            ``process_group``. ``process_group.size() == ep_size * tp_size``.
          * Pass ``tp_size`` so this class can:
              - Treat the expert space as ``n_routed_experts * tp_size`` *virtual* experts.
                Each physical expert ``e`` gets ``tp_size`` virtual copies, one owned by each
                TP rank in the EP group ``e`` belongs to.
              - Expand caller-supplied ``topk_ids`` so a token routed to physical expert ``e``
                lands on **both** TP ranks within EP rank ``ep(e)`` — exactly what
                column-parallel ``fused_w1w3`` needs.

        DeepEP's NVL+RDMA path encodes destination as
        ``(rdma_rank, is_token_in_nvl_rank_bits)`` (see ``DeepEP/csrc/kernels/internode.cu``),
        so duplicated routings landing on the same node are sent as a single RDMA transfer
        with the appropriate NVL bitmask. Cross-node bandwidth therefore matches the
        ep-only case; only the local intra-node fan-out is doubled.
        """
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
        self._tp_size = tp_size
        assert process_group.size() % tp_size == 0, (
            f"process_group size {process_group.size()} must be a multiple of tp_size {tp_size}; "
            f"the caller is expected to pass the combined (ep × tp) group."
        )
        self._ep_size = process_group.size() // tp_size
        assert n_routed_experts % self._ep_size == 0, (
            f"n_routed_experts {n_routed_experts} must be divisible by ep_size {self._ep_size}"
        )
        self._local_experts = n_routed_experts // self._ep_size
        # Virtual expert count seen by DeepEP. Per-rank count
        # (= virtual_n_experts / process_group.size()) stays equal to ``_local_experts`` —
        # downstream ``permute`` / ``group_gemm`` consume ``num_recv_tokens_per_expert_list`` of
        # that fixed length and no aggregation is needed.
        self._virtual_n_experts = n_routed_experts * tp_size

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool = False,
    ) -> DeepEPPreDispatchResult:
        if async_op:
            backward_previous_event = EventOverlap(None)
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="DeepEPDispatcher.dispatch_preprocess.hidden_states",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )
        else:
            backward_previous_event = None

        topk_ids = topk_ids.to(torch.int64)
        if self._tp_size > 1:
            topk_ids = self._expand_topk_ids_for_tp(topk_ids)
            # ``topk_ids`` was duplicated tp_size× above; ``topk_weights`` must follow with
            # the SAME value per duplicate. No 1/tp scaling — the two TP partial outputs sum
            # to the full expert output, so weighting both by ``w_k`` already gives
            # ``w_k * full`` after combine.
            topk_weights = topk_weights.repeat_interleave(self._tp_size, dim=-1).contiguous()
            if async_op and topk_weights.grad_fn is not None:
                # Symmetric to the ``hidden_states`` prehook: the grad for ``topk_weights``
                # flows back through ``repeat_interleave_backward`` on the compute stream,
                # while DeepEP's dispatch backward writes that grad on the comm stream and
                # stamps the event into ``backward_previous_event``. Without this prehook
                # the compute-stream backward starts before that event fires and reads
                # stale grad memory — observed as ``grad_norm=NaN``.
                topk_weights.grad_fn.register_prehook(
                    get_backward_pre_hook(
                        backward_previous_event=backward_previous_event,
                        name="DeepEPDispatcher.dispatch_preprocess.topk_weights",
                        debug=XTUNER_DISPATCHER_DEBUG,
                    )
                )

        # Capture AFTER all compute-stream work above (topk_ids expand, topk_weights
        # repeat_interleave) so DeepEP's ``stream_wait(previous_event)`` covers those
        # kernels. Capturing before them leaves their writes outside the event, and
        # DeepEP's comm-stream dispatch may read stale memory — observed as NaN /
        # divergent loss under ``intra_layer_micro_batch>1`` with virtual expert TP.
        forward_finished_event = buffer_capture() if async_op else None

        return DeepEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            backward_previous_event=backward_previous_event,
            forward_finished_event=forward_finished_event,
        )

    def _expand_topk_ids_for_tp(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Map physical-expert ids to virtual-expert ids so DeepEP routes each token to every
        TP rank within its owning EP group.

        Virtual id layout (rank ``r = ep * tp + t`` owns ids ``[r * local, (r + 1) * local)``)::

            virtual_id(e, t) = (ep(e) * tp + t) * local_experts + (e mod local_experts)

        The two virtuals for the same physical expert sit on adjacent ranks (same EP, t=0/t=1),
        so DeepEP's NUMA layout collapses the cross-node copy to a single RDMA transfer with a
        2-bit NVL bitmask. ``-1`` (padding) is preserved.
        """
        local_experts = self._local_experts
        tp = self._tp_size

        ep_e = topk_ids // local_experts
        local_idx = topk_ids % local_experts
        tp_offsets = torch.arange(tp, device=topk_ids.device, dtype=topk_ids.dtype)
        virtual = (ep_e.unsqueeze(-1) * tp + tp_offsets) * local_experts + local_idx.unsqueeze(-1)
        # Preserve sentinel (-1) for padded slots after expansion.
        virtual = torch.where(topk_ids.unsqueeze(-1) < 0, topk_ids.unsqueeze(-1), virtual)
        out = virtual.reshape(*topk_ids.shape[:-1], topk_ids.shape[-1] * tp)
        return out.contiguous()

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        topk_weights: torch.Tensor,  # noqa: ARG002 — already expanded and stashed in pre_dispatched
        async_op: bool = False,
        decoding: bool = False,
    ) -> DeepEPDispatchResult:
        # ``topk_ids`` / ``topk_weights`` expansion and the cross-stream sync setup live in
        # ``dispatch_preprocess`` so they run on Loop A's compute stream and overlap with the
        # next microbatch's attention/gate. ``dispatch`` itself only kicks off DeepEP.
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
            pre_dispatched["topk_weights"],
            self._virtual_n_experts,
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
        # Per-call pinned allocation is cheap here: PyTorch's caching host
        # allocator pools pinned blocks, so `cudaHostAlloc` only fires on cold
        # start; steady-state cost is sub-microsecond.
        #
        # Do NOT "optimize" this by holding a single module-level pinned buffer
        # and writing into it in place. Under multi-microbatch overlap the CPU
        # thread runs well ahead of the GPU stream, and the next microbatch's
        # host write would clobber the source before the previous microbatch's
        # `non_blocking=True` H2D had actually executed. The fresh-tensor form
        # is safe because the caching allocator refuses to recycle a pinned
        # block until the CUDA events referencing it have completed — a
        # guarantee a manually held buffer does not get.
        tokens_per_expert = torch.tensor(
            num_recv_tokens_per_expert_list,
            dtype=torch.long,
            pin_memory=True,
        )
        # `non_blocking=True` is only safe because every downstream consumer of
        # `tokens_per_expert` (group GEMM, FP8 quant kernels, prober) runs on
        # the current CUDA stream, so stream ordering covers the H2D. If
        # consumption moves to a different stream, the consumer must wait on an
        # event recorded after this copy.
        tokens_per_expert = tokens_per_expert.to(dispatched["topk_weights"].device, non_blocking=True)

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
            self._virtual_n_experts,
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
        # Restored original wait order (after view_as) to test torch_compile interaction
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
