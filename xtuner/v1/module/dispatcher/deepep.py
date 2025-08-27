# type: ignore
from typing import Literal, TypeAlias, cast

import torch
from mmengine.utils import is_installed
from typing_extensions import Required, overload, override

from xtuner.v1.ops import buffer_capture, deep_ep_combine, deep_ep_dispatch, get_low_latency_buffer
from xtuner.v1.ops.moe_permute import permute, unpermute

from .base import (
    DecodingCombineResult,
    DecodingDispatchResult,
    GenericDispatcher,
    HiddenStates,
    PreDispatchResult,
    PrefillingCombineResult,
    PrefillingDispatchResult,
)


# DeepEP handle include 6 tensor:
# (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
DeepEPHandle = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

DeepEPPreDispatchResult: TypeAlias = PreDispatchResult


# TODO: Broken inheritance, we should fix it later
class DeepEPPrefillingDispatchResult(PrefillingDispatchResult):
    handle: Required[DeepEPHandle]  # type: ignore
    row_id_map: torch.Tensor


# TODO: Broken inheritance, we should fix it later
class DeepEPDecodingDispatchResult(DecodingDispatchResult):
    handle: DeepEPHandle  # type: ignore


DeepEPPrefillingCombineResult: TypeAlias = PrefillingCombineResult
DeepEPDecodingCombineResult: TypeAlias = DecodingCombineResult


class DeepEPDispatcher(
    GenericDispatcher[
        DeepEPPreDispatchResult,
        DeepEPPrefillingDispatchResult,
        DeepEPDecodingDispatchResult,
        DeepEPPrefillingCombineResult,
        DeepEPDecodingCombineResult,
    ]
):
    _process_group: torch.distributed.ProcessGroup

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

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        decoding: Literal[True],
    ) -> DeepEPDecodingDispatchResult: ...

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        decoding: Literal[False],
    ) -> DeepEPPrefillingDispatchResult: ...

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        decoding: bool = False,
    ) -> DeepEPPrefillingDispatchResult | DeepEPDecodingDispatchResult:
        if not decoding:
            return self._dispatch_prefilling(
                hidden_states=pre_dispatched["hidden_states"],
                topk_weights=pre_dispatched["topk_weights"],
                topk_ids=pre_dispatched["topk_ids"],
            )
        else:
            return self._dispatch_decoding(
                hidden_states=pre_dispatched["hidden_states"],
                topk_ids=pre_dispatched["topk_ids"],
            )

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DeepEPPreDispatchResult:
        return DeepEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: DeepEPPrefillingDispatchResult,
        decoding: Literal[False],
    ) -> DeepEPPrefillingCombineResult: ...

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: DeepEPDecodingDispatchResult,
        decoding: Literal[True],
    ) -> DeepEPDecodingCombineResult: ...

    @override
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: PreDispatchResult,
        dispatch_result: DeepEPPrefillingDispatchResult | DeepEPDecodingDispatchResult,
        decoding: bool = False,
    ) -> DeepEPPrefillingCombineResult | DeepEPDecodingCombineResult:
        if not decoding:
            return self._combine_prefilling(
                hidden_states=hidden_states,
                dispatched_result=cast(DeepEPPrefillingDispatchResult, dispatch_result),
            )
        else:
            return self._combine_decoding(
                hidden_states=hidden_states,
                pre_dispatched=cast(DeepEPPreDispatchResult, dispatch_result),
                dispatched_result=cast(DeepEPDecodingDispatchResult, dispatch_result),
            )

    @override
    def combine_post_process(
        self,
        *,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatch_result: DeepEPPrefillingDispatchResult | DeepEPDecodingDispatchResult,
        combine_result: DeepEPPrefillingCombineResult | DeepEPDecodingCombineResult,
    ) -> HiddenStates:
        return combine_result["hidden_states"]

    def _dispatch_decoding(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> DeepEPDecodingDispatchResult:
        hidden_size = hidden_states.shape[-1]
        x = hidden_states.view(-1, hidden_states.size()[-1])
        _buffer = get_low_latency_buffer(self._process_group, hidden=hidden_size, num_experts=self._n_routed_experts)

        # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
        recv_x, tokens_per_expert, handle, _, _ = _buffer.low_latency_dispatch(
            x,
            topk_ids,
            x.size(0),
            self._n_routed_experts,
            async_finish=False,
            use_fp8=self._training_dtype == "fp8",
            return_recv_hook=False,
        )

        # NOTES: the actual tensor will not be received only if you call `hook()`,
        # it is useful for double-batch overlapping, but **without any SM occupation**
        # If you don't want to overlap, please set `return_recv_hook=False`
        # Later, you can use our GEMM library to do the computation with this specific format
        if self._training_dtype == "fp8":
            assert isinstance(recv_x, tuple), "When using FP8, `recv_x` should be a tuple."
            hidden_states, fp_8_scale = recv_x
            return DeepEPDecodingDispatchResult(
                hidden_states=hidden_states,
                tokens_per_experts=tokens_per_expert,
                fp8_scale=fp_8_scale,
                handle=handle,
            )
        else:
            raise NotImplementedError("DeepEP decoding dispatch only supports FP8 for now.")

    def _dispatch_prefilling(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> DeepEPPrefillingDispatchResult:
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)

        topk_ids = topk_ids.to(torch.int64)

        # TODO: Maybe we should decouple the sync and async interface of deepep
        previous_event = buffer_capture()
        (
            dispatched_hidden_states,
            dispatched_topk_idx,
            dispatched_topk_weights,
            num_recv_tokens_per_expert_list,
            dispatch_handle,
            event,
        ) = deep_ep_dispatch(
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_routed_experts=self._n_routed_experts,
            group=self._process_group,
            previous_event=previous_event,
        )
        event.current_stream_wait()

        permuted_hidden_states, row_id_map = self._training_permute_dispatched(
            dispatched_hidden_states=dispatched_hidden_states,
            dispatched_topk_ids=dispatched_topk_idx,
            num_recv_tokens_per_expert_list=num_recv_tokens_per_expert_list,
        )
        tokens_per_experts = torch.tensor(
            num_recv_tokens_per_expert_list,
            dtype=torch.long,
            device=topk_weights.device,
        )
        return DeepEPPrefillingDispatchResult(
            hidden_states=permuted_hidden_states,
            tokens_per_experts=tokens_per_experts,
            handle=dispatch_handle,
            topk_weights=dispatched_topk_weights,
            row_id_map=row_id_map,
        )

    def _training_permute_dispatched(
        self,
        *,
        dispatched_hidden_states: torch.Tensor,
        dispatched_topk_ids: torch.Tensor,
        num_recv_tokens_per_expert_list,
    ):
        num_out_tokens = sum(num_recv_tokens_per_expert_list)
        recv_topk_idx_numel = dispatched_topk_ids.numel()
        num_neg_one_idx = recv_topk_idx_numel - num_out_tokens

        permuted_hidden_states, row_id_map = permute(
            dispatched_hidden_states,
            dispatched_topk_ids.int(),
            num_out_tokens=num_out_tokens,
            num_negative_one_in_indices=num_neg_one_idx,
        )
        return permuted_hidden_states, row_id_map

    def _combine_prefilling(
        self,
        hidden_states: torch.Tensor,
        dispatched_result: DeepEPPrefillingDispatchResult,
    ):
        unpermuted_hidden_states = self._training_unpermute_activation(
            hidden_states=hidden_states,
            row_id_map=dispatched_result["row_id_map"],
            topk_weights=dispatched_result["topk_weights"],
        )

        # TODO: Maybe we should decouple the sync and async interface of deepep
        event = buffer_capture()
        combined_hidden_states, event = deep_ep_combine(
            x=unpermuted_hidden_states,
            num_experts=self._n_routed_experts,
            deepep_comm_handle=dispatched_result["handle"],
            group=self._process_group,
        )
        event.current_stream_wait()
        # For event management, please refer to the docs of the `EventOverlap` class
        return DeepEPPrefillingCombineResult(
            hidden_states=combined_hidden_states,
        )

    def _combine_decoding(
        self,
        hidden_states: torch.Tensor,
        pre_dispatched: DeepEPPreDispatchResult,
        dispatched_result: DeepEPDecodingDispatchResult,
    ):
        hidden_size = hidden_states.shape[-1]
        _buffer = get_low_latency_buffer(
            self._process_group,
            hidden=hidden_size,
            num_experts=self._n_routed_experts,
        )

        # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
        combined_x, _, _ = _buffer.low_latency_combine(
            x=hidden_states,
            topk_idx=pre_dispatched["topk_ids"],
            topk_weights=pre_dispatched["topk_weights"],
            handle=dispatched_result["handle"],
            async_finish=False,
            return_recv_hook=False,
        )

        return DeepEPDecodingCombineResult(hidden_states=combined_x)

    def _training_unpermute_activation(
        self,
        hidden_states: torch.Tensor,
        row_id_map: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
    ):
        # assert self.ep_mesh.size() > 1
        activation = unpermute(
            input_act=hidden_states,
            row_id_map=row_id_map,
            probs=topk_weights,
        )
        return activation
