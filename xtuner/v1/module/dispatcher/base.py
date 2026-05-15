from abc import ABC, abstractmethod
from typing import (
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)

import torch
from typing_extensions import TypedDict, override

from xtuner.v1.ops import permute, unpermute

from .expert_tp import ExpertTP


HiddenStates: TypeAlias = torch.Tensor


def _get_backward_pre_hook(backward_previous_event: torch.cuda.Event):
    def _backward_pre_hook(*_):
        torch.cuda.current_stream().wait_event(backward_previous_event)

    return _backward_pre_hook


def _get_backward_hook(backward_finished_event: torch.cuda.Event):
    def _backward_hook(*_):
        backward_finished_event.record()

    return _backward_hook


class PreDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor


class DispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor


class PostDispatchResult(TypedDict):
    """Result of the dispatch operation during prefilling phase.

    This class holds the dispatched result during the prefilling phase. `hidden_states` and
    `tokens_per_expert` are used for the experts forwarding. `topk_weights` contains the
    routing weights. Some dispatcher could apply weighted sum during combining to reduce the communication,
    `handle` is used to facilitate the combination of expert outputs after processing.

    Attributes:
        hidden_states: The hidden states after expert token routing and dispatching.
        tokens_per_expert: Count of tokens assigned to each expert in the current batch.
        topk_weights: Expert routing weights used for scaling hidden states when combining results.
        handle: An object that facilitates the combination of expert outputs after processing.
    """

    # TODO:
    hidden_states: torch.Tensor
    tokens_per_expert: torch.Tensor


class PreCombineResult(TypedDict):
    hidden_states: torch.Tensor


class CombineResult(TypedDict):
    hidden_states: torch.Tensor


class PostCombineResult(TypedDict):
    hidden_states: torch.Tensor


PreDispatch = TypeVar("PreDispatch")
Dispatch = TypeVar("Dispatch")
PostDispatch = TypeVar("PostDispatch")
PreCombine = TypeVar("PreCombine")
Combine = TypeVar("Combine")
PostCombine = TypeVar("PostCombine")
# TODO: add DecodingPostDispatch if needed.


# Not using Protocol here since `__init__` is shared for all dispatchers.
class GenericDispatcher(
    ABC,
    Generic[
        PreDispatch,
        Dispatch,
        PostDispatch,
        PreCombine,
        Combine,
        PostCombine,
    ],
):
    _n_routed_experts: int
    _process_group: torch.distributed.ProcessGroup | None

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup | None = None,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        self._process_group = process_group
        self._n_routed_experts = n_routed_experts
        self._training_dtype = training_dtype
        self._generate_dtype = generate_dtype

    @abstractmethod
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatch,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> Dispatch: ...

    @abstractmethod
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        async_op: bool = False,
    ) -> PostDispatch: ...

    @abstractmethod
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> PreDispatch: ...

    @abstractmethod
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        async_op: bool = False,
        decoding: bool = False,
    ) -> PreCombine: ...

    @abstractmethod
    def combine(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        pre_combined: PreCombine,
        async_op: bool = False,
        decoding: bool = False,
    ) -> CombineResult: ...

    @abstractmethod
    def combine_postprocess(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatched: Dispatch,
        post_dispatched: PostDispatch,
        pre_combined: PreCombine,
        combined: Combine,
        async_op: bool = False,
    ) -> PostCombine: ...


class DispacherInterface(
    GenericDispatcher[
        PreDispatchResult,
        DispatchResult,
        PostDispatchResult,
        PreCombineResult,
        CombineResult,
        PostCombineResult,
    ],
): ...


class NaivePreDispatchResult(PreDispatchResult, total=False):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaiveDispatchResult(DispatchResult, total=False):
    topk_ids: torch.Tensor
    tp_size_meta: list[int]
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None
    topk_weights_backward_previous_event: torch.cuda.Event | None


class NaivePostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class NaivePreCombineResult(PreCombineResult, total=False):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaiveCombineResult(CombineResult, total=False):
    forward_finished_event: torch.cuda.Event | None
    backward_previous_event: torch.cuda.Event | None


class NaivePostCombineResult(PostCombineResult): ...


class NaiveDispatcher(
    GenericDispatcher[
        NaivePreDispatchResult,
        NaiveDispatchResult,
        NaivePostDispatchResult,
        NaivePreCombineResult,
        NaiveCombineResult,
        NaivePostCombineResult,
    ]
):
    _comm_stream: torch.cuda.Stream | None = None

    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        training_dtype: Literal["fp8", "bf16"] = "bf16",
        generate_dtype: Literal["fp8", "bf16"] = "bf16",
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            process_group=process_group,
            training_dtype=training_dtype,
            generate_dtype=generate_dtype,
        )
        if self._process_group is not None:
            assert self._process_group.size() == 1, "Naive dispatcher is only for ep=1."
        self._expert_tp = ExpertTP(tp_group) if tp_group is not None and tp_group.size() > 1 else None
        if self._expert_tp is not None and NaiveDispatcher._comm_stream is None:
            NaiveDispatcher._comm_stream = torch.cuda.Stream()

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> NaivePreDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

            forward_finished_event = torch.cuda.Event()
            forward_finished_event.record()
            backward_previous_event = torch.cuda.Event()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(_get_backward_pre_hook(backward_previous_event))

            return NaivePreDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                forward_finished_event=forward_finished_event,
                backward_previous_event=backward_previous_event,
            )

        return NaivePreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaiveDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

            forward_previous_event = pre_dispatched["forward_finished_event"]
            backward_finished_event = pre_dispatched["backward_previous_event"]
            assert forward_previous_event is not None, "Use async_op=True for dispatch_preprocess!"
            assert backward_finished_event is not None, "Use async_op=True for dispatch_preprocess!"
            assert self._comm_stream is not None

            tp_size_meta = self._expert_tp.gather_size_meta(pre_dispatched["hidden_states"])
            # 中文注释：dispatch 内部的 TP AllGather 都排在同一个 comm stream，
            # 互相不需要 event 串行化；只在 dispatch 阶段边界记录最终完成事件。
            forward_finished_event = torch.cuda.Event()
            hidden_backward_previous_event = torch.cuda.Event()
            topk_weights_backward_previous_event = torch.cuda.Event()
            topk_weights_backward_finished_event = torch.cuda.Event()
            if topk_weights.grad_fn is not None:
                topk_weights.grad_fn.register_prehook(_get_backward_pre_hook(topk_weights_backward_finished_event))

            hidden_states = self._expert_tp.async_all_gather(
                pre_dispatched["hidden_states"],
                all_sizes=tp_size_meta,
                forward_previous_event=forward_previous_event,
                forward_finished_event=None,
                backward_previous_event=hidden_backward_previous_event,
                backward_finished_event=backward_finished_event,
                comm_stream=self._comm_stream,
            )
            topk_ids = self._expert_tp.async_all_gather_metadata(
                pre_dispatched["topk_ids"],
                all_sizes=tp_size_meta,
                forward_previous_event=None,
                forward_finished_event=None,
                comm_stream=self._comm_stream,
            )
            topk_weights = self._expert_tp.async_all_gather(
                topk_weights,
                all_sizes=tp_size_meta,
                forward_previous_event=None,
                forward_finished_event=forward_finished_event,
                backward_previous_event=topk_weights_backward_previous_event,
                backward_finished_event=topk_weights_backward_finished_event,
                comm_stream=self._comm_stream,
            )

            return NaiveDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                tp_size_meta=tp_size_meta,
                forward_finished_event=forward_finished_event,
                backward_previous_event=hidden_backward_previous_event,
                topk_weights_backward_previous_event=topk_weights_backward_previous_event,
            )

        if self._expert_tp is not None:
            hidden_states, tp_size_meta = self._expert_tp.all_gather(pre_dispatched["hidden_states"])
            topk_ids = self._expert_tp.all_gather_metadata(pre_dispatched["topk_ids"], tp_size_meta)
            topk_weights = self._expert_tp.all_gather_metadata(topk_weights, tp_size_meta)
            return NaiveDispatchResult(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                tp_size_meta=tp_size_meta,
            )

        return NaiveDispatchResult(
            hidden_states=pre_dispatched["hidden_states"],
            topk_weights=topk_weights,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaivePostDispatchResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")
            forward_finished_event = dispatched["forward_finished_event"]
            assert forward_finished_event is not None, "Use async_op=True for dispatch!"
            torch.cuda.current_stream().wait_event(forward_finished_event)

        topk_ids = dispatched["topk_ids"] if self._expert_tp is not None else pre_dispatched["topk_ids"]
        hidden_states, row_id_maps = permute(
            dispatched["hidden_states"],
            topk_ids.to(torch.int32),
        )
        tokens_per_expert = torch.histc(topk_ids, bins=self._n_routed_experts, min=0, max=self._n_routed_experts)
        if async_op:
            backward_previous_event = dispatched["backward_previous_event"]
            assert backward_previous_event is not None, "Use async_op=True for dispatch!"
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(_get_backward_hook(backward_previous_event))

        if decoding:
            raise NotImplementedError
        else:
            return NaivePostDispatchResult(
                hidden_states=hidden_states,
                row_ids_map=row_id_maps,
                tokens_per_expert=tokens_per_expert,
            )

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaivePreCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

        hidden_states = unpermute(
            input_act=hidden_states,
            row_id_map=post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )
        if async_op:
            backward_previous_event = torch.cuda.Event()
            forward_finished_event = torch.cuda.Event()
            forward_finished_event.record()
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_prehook(_get_backward_pre_hook(backward_previous_event))
                topk_weights_backward_previous_event = dispatched["topk_weights_backward_previous_event"]
                assert topk_weights_backward_previous_event is not None, "Use async_op=True for dispatch!"
                hidden_states.grad_fn.register_hook(_get_backward_hook(topk_weights_backward_previous_event))
        else:
            backward_previous_event = None
            forward_finished_event = None

        if decoding:
            raise NotImplementedError("NaiveDispatcher does not support decoding.")
        else:
            return NaivePreCombineResult(
                hidden_states=hidden_states,
                backward_previous_event=backward_previous_event,
                forward_finished_event=forward_finished_event,
            )

    @override
    def combine(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        pre_combined: NaivePreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaiveCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")

        if decoding:
            raise NotImplementedError
        else:
            if self._expert_tp is not None:
                if async_op:
                    forward_previous_event = pre_combined["forward_finished_event"]
                    backward_finished_event = pre_combined["backward_previous_event"]
                    assert forward_previous_event is not None, "Use async_op=True for combine_preprocess!"
                    assert backward_finished_event is not None, "Use async_op=True for combine_preprocess!"
                    assert self._comm_stream is not None

                    forward_finished_event = torch.cuda.Event()
                    backward_previous_event = torch.cuda.Event()
                    hidden_states = self._expert_tp.async_reduce_scatter_sum(
                        pre_combined["hidden_states"],
                        all_sizes=dispatched["tp_size_meta"],
                        forward_previous_event=forward_previous_event,
                        forward_finished_event=forward_finished_event,
                        backward_previous_event=backward_previous_event,
                        backward_finished_event=backward_finished_event,
                        comm_stream=self._comm_stream,
                    )
                    return NaiveCombineResult(
                        hidden_states=hidden_states,
                        forward_finished_event=forward_finished_event,
                        backward_previous_event=backward_previous_event,
                    )

                hidden_states = self._expert_tp.reduce_scatter_sum(
                    pre_combined["hidden_states"],
                    dispatched["tp_size_meta"],
                )
                return NaiveCombineResult(hidden_states=hidden_states)

            return NaiveCombineResult(hidden_states=pre_combined["hidden_states"])

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: NaivePreDispatchResult,
        dispatched: NaiveDispatchResult,
        post_dispatched: NaivePostDispatchResult,
        pre_combined: NaivePreCombineResult,
        combined: NaiveCombineResult,
        async_op: bool = False,
    ) -> PostCombineResult:
        if async_op:
            if self._expert_tp is None:
                raise NotImplementedError("Naive dispatcher async_op=True requires ExpertTP.")
            forward_finished_event = combined["forward_finished_event"]
            backward_previous_event = combined["backward_previous_event"]
            assert forward_finished_event is not None, "Use async_op=True for combine!"
            assert backward_previous_event is not None, "Use async_op=True for combine!"
            torch.cuda.current_stream().wait_event(forward_finished_event)
            hidden_states = combined["hidden_states"].view_as(combined["hidden_states"])
            if hidden_states.grad_fn is not None:
                hidden_states.grad_fn.register_hook(_get_backward_hook(backward_previous_event))
            return PostCombineResult(hidden_states=hidden_states)

        return PostCombineResult(hidden_states=combined["hidden_states"])
