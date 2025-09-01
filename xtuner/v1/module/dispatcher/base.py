from abc import ABC, abstractmethod
from typing import (
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)

import torch
from typing_extensions import TypedDict, override

from xtuner.v1.config.base_model import MoEConfig
from xtuner.v1.ops import permute, unpermute


HiddenStates: TypeAlias = torch.Tensor


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
    _config: MoEConfig

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


class NaivePreDispatchResult(PreDispatchResult): ...


class NaiveDispatchResult(DispatchResult): ...


class NaivePostDispatchResult(PostDispatchResult):
    row_ids_map: torch.Tensor


class NaivePreCombineResult(PreCombineResult): ...


class NaiveCombineResult(CombineResult): ...


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
    def __init__(
        self,
        *,
        n_routed_experts: int,
        process_group: torch.distributed.ProcessGroup | None = None,
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

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        async_op: bool = False,
    ) -> PreDispatchResult:
        if async_op:
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

        return NaivePreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> NaiveDispatchResult:
        if async_op:
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

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
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

        hidden_states, row_id_maps = permute(
            dispatched["hidden_states"],
            pre_dispatched["topk_ids"].to(torch.int32),
        )
        topk_ids = pre_dispatched["topk_ids"]
        tokens_per_expert = torch.histc(topk_ids, bins=self._n_routed_experts, min=0, max=self._n_routed_experts)
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
    ) -> PreCombineResult:
        if async_op:
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

        hidden_states = unpermute(
            input_act=hidden_states,
            row_id_map=post_dispatched["row_ids_map"],
            probs=dispatched["topk_weights"],
        )
        if decoding:
            raise NotImplementedError("NaiveDispatcher does not support decoding.")
        else:
            return PreCombineResult(hidden_states=hidden_states)

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
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

        if decoding:
            raise NotImplementedError
        else:
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
            raise NotImplementedError("Naive dispatcher is only for ep=1.")

        return PostCombineResult(hidden_states=combined["hidden_states"])
