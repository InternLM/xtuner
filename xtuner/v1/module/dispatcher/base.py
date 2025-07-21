from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import torch
from typing_extensions import NotRequired, overload, override

from xtuner.v1.config.base_model import MoEConfig
from xtuner.v1.ops.moe_permute import permute, unpermute


HiddenStates: TypeAlias = torch.Tensor


class PreDispatchResult(TypedDict):
    """Result container for the pre-dispatch phase in MoE routing.

    This class encapsulates the outputs produced during the initial token routing stage
    before the actual dispatch of tokens to experts occurs. It includes routing weights,
    expert assignments, and other metadata needed for the subsequent dispatch operation.

    Some dispatcher needs to permute the hidden states before dispatching them to experts.
    This class is used to hold the permuted hidden states and the corresponding routing information.

    Attributes:
        hidden_states: Hidden states (permuted) that will be routed to experts.
        topk_ids: Indices (permuted) of the top-k experts selected for each token.
        topk_weights: Routing weights (permuted) for each token-expert pair in the top-k selection.
    """

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class DecodingDispatchResult(TypedDict):
    """Dispatched result for decoding.

    This class hold the dispatched result during the decoding phase. `hidden_states` and `tokens_per_experts`
    are used for the experts forwarding. If fp8 precision is used, `fp8_scale` is also included.

    Attributes:
        hidden_states: The tensor of hidden states that have been routed after the dispatching process.
        tokens_per_experts: The number of tokens assigned to each expert as a result of dispatching.
        fp8_scale: Scaling factor used specifically when working with FP8 precision during decoding.
        handle: A handle for combining results after dispatching.
    """

    hidden_states: torch.Tensor
    tokens_per_experts: torch.Tensor
    fp8_scale: NotRequired[torch.Tensor]
    handle: NotRequired[Any]


class PrefillingDispatchResult(TypedDict):
    """Result of the dispatch operation during prefilling phase.

    This class holds the dispatched result during the prefilling phase. `hidden_states` and
    `tokens_per_experts` are used for the experts forwarding. `topk_weights` contains the
    routing weights. Some dispatcher could apply weighted sum during combining to reduce the communication,
    `handle` is used to facilitate the combination of expert outputs after processing.

    Attributes:
        hidden_states: The hidden states after expert token routing and dispatching.
        tokens_per_experts: Count of tokens assigned to each expert in the current batch.
        topk_weights: Expert routing weights used for scaling hidden states when combining results.
        handle: An object that facilitates the combination of expert outputs after processing.
    """

    hidden_states: torch.Tensor
    tokens_per_experts: torch.Tensor
    topk_weights: torch.Tensor
    handle: Any


class PrefillingCombineResult(TypedDict):
    hidden_states: torch.Tensor


DecodingCombineResult = PrefillingCombineResult


PreDispatch = TypeVar("PreDispatch")
PrefillingDispatch = TypeVar("PrefillingDispatch")
DecodingDispatch = TypeVar("DecodingDispatch")
PrefillingCombine = TypeVar("PrefillingCombine")
DecodingCombine = TypeVar("DecodingCombine")


# Not using Protocol here since `__init__` is shared for all dispatchers.
class GenericDispatcher(
    ABC,
    Generic[
        PreDispatch,
        PrefillingDispatch,
        DecodingDispatch,
        PrefillingCombine,
        DecodingCombine,
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

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatch,
        decoding: Literal[True],
    ) -> DecodingDispatch: ...

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatch,
        decoding: Literal[False],
    ) -> PrefillingDispatch: ...

    @abstractmethod
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatch,
        decoding: bool = False,
    ) -> PrefillingDispatch | DecodingDispatch: ...

    @abstractmethod
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> PreDispatch: ...

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatch,
        dispatch_result: PrefillingDispatch,
        decoding: Literal[False],
    ) -> PrefillingCombine: ...

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatch,
        dispatch_result: DecodingDispatch,
        decoding: Literal[True],
    ) -> DecodingCombine: ...

    @abstractmethod
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatch,
        dispatch_result: PrefillingDispatch | DecodingDispatch,
        decoding: bool = False,
    ) -> DecodingCombine | PrefillingCombine: ...

    @abstractmethod
    def combine_post_process(
        self,
        *,
        pre_dispatched: PreDispatch,
        dispatch_result: PrefillingDispatch | DecodingDispatch,
        combine_result: PrefillingCombine | DecodingCombine,
    ) -> HiddenStates: ...

    # Async interface for inference.
    # @abstractmethod
    # def async_dispatch(self) -> Callable[[], PrefillingDispatch]: ...
    #
    # @abstractmethod
    # def async_dispatch_decoding(self) -> Callable[[], PrefillingDispatch]: ...
    #
    # @abstractmethod
    # def async_combine(self) -> Callable[[], PrefillingCombine]: ...
    #
    # @abstractmethod
    # def async_combine_decoding(self) -> Callable[[], PrefillingCombine]: ...

    ################################### Async interface for training ###################################
    # @abstractmethod
    # def async_dispatch_forward(self) -> Callable[[], PrefillingDispatchResult]: ...
    #
    # @abstractmethod
    # def async_dispatch_backward(self): ...
    #
    # @abstractmethod
    # def wait_dispatch_backward(self): ...
    #
    # @abstractmethod
    # def async_combine_forward(self) -> Callable[[], PrefillingCombineResult]: ...
    #
    # @abstractmethod
    # def async_combine_backward(self): ...
    #
    # @abstractmethod
    # def wait_combine_backward(self) -> None: ...
    #
    # @abstractmethod
    # def combine_postprocess(self) -> HiddenStates: ...
    #
    # @abstractmethod
    # def async_combine_postprocess(self) -> Callable[[], HiddenStates]: ...
    #
    # @abstractmethod
    # def async_dispatch_preprocess(
    #     self,
    #     *,
    #     hidden_states: torch.Tensor,
    #     topk_ids: torch.Tensor,
    # ) -> Callable[[], bool]: ...
    #
    #


class DispacherInterface(
    GenericDispatcher[
        PreDispatchResult,
        PrefillingDispatchResult,
        DecodingDispatchResult,
        PrefillingCombineResult,
        DecodingCombineResult,
    ],
): ...


class NonEPPrefillingDispatchResult(PrefillingDispatchResult):
    row_ids_map: torch.Tensor


class NonEPDecodingDispatchResult(DecodingDispatchResult):
    row_ids_map: torch.Tensor


class NaiveDispatcher(
    GenericDispatcher[
        PreDispatchResult,
        NonEPPrefillingDispatchResult,
        NonEPDecodingDispatchResult,
        PrefillingCombineResult,
        DecodingCombineResult,
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
        topk_weights: torch.Tensor,
    ) -> PreDispatchResult:
        return PreDispatchResult(hidden_states=hidden_states, topk_weights=topk_weights, topk_ids=topk_ids)

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatchResult,
        decoding: Literal[True],
    ) -> NonEPDecodingDispatchResult: ...

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatchResult,
        decoding: Literal[False],
    ) -> NonEPPrefillingDispatchResult: ...

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: PreDispatchResult,
        decoding: bool = False,
    ) -> NonEPPrefillingDispatchResult | NonEPDecodingDispatchResult:
        topk_ids = pre_dispatched["topk_ids"]
        tokens_per_expert = torch.histc(topk_ids, bins=self._n_routed_experts, min=0, max=self._n_routed_experts)
        hidden_states, row_id_maps = permute(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"].to(torch.int32),
        )
        if decoding:
            return NonEPDecodingDispatchResult(
                hidden_states=hidden_states,
                tokens_per_experts=tokens_per_expert,
                row_ids_map=row_id_maps,
            )
        else:
            return NonEPPrefillingDispatchResult(
                hidden_states=hidden_states,
                tokens_per_experts=tokens_per_expert,
                row_ids_map=row_id_maps,
                topk_weights=pre_dispatched["topk_weights"],
                handle=None,  # NaiveDispatcher do not need async communication.
            )

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: NonEPPrefillingDispatchResult,
        decoding: Literal[False],
    ) -> PrefillingCombineResult: ...

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: NonEPDecodingDispatchResult,
        decoding: Literal[True],
    ) -> DecodingCombineResult: ...

    @override
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: NonEPPrefillingDispatchResult | NonEPDecodingDispatchResult,
        decoding: bool = False,
    ) -> PrefillingCombineResult | DecodingCombineResult:
        hidden_states = unpermute(
            input_act=hidden_states,
            row_id_map=dispatch_result["row_ids_map"],
            probs=pre_dispatched["topk_weights"],
        )
        if decoding:
            return DecodingCombineResult(
                hidden_states=hidden_states,
            )
        else:
            return PrefillingCombineResult(
                hidden_states=hidden_states,
            )

    @override
    def combine_post_process(
        self,
        *,
        pre_dispatched: PreDispatchResult,
        dispatch_result: NonEPPrefillingDispatchResult | NonEPDecodingDispatchResult,
        combine_result: PrefillingCombineResult | DecodingCombineResult,
    ) -> HiddenStates:
        return combine_result["hidden_states"]
