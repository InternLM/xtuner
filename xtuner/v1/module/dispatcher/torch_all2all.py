from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd
from typing_extensions import overload, override

from xtuner.v1.config.base_model import MoEConfig
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
    """Encapsulates pre-processed data required for AllToAll dispatch
    operations.

    This class stores the intermediate results generated during the pre-dispatch phase
    of the AllToAll communication pattern used in Mixture of Experts (MoE) models.

    Attributes:
        row_id_map (torch.Tensor): Mapping information needed to unpermute hidden states
            after AllToAll operations are complete.
        tokens_per_experts_group (torch.Tensor): Number of tokens assigned to each expert
            group across all processes in the communication group.
        input_splits (torch.Tensor): Input tensor splits defining how the data will be
            partitioned for the AllToAll collective operation.
        output_splits (torch.Tensor): Output tensor splits defining how the data will be
            recombined after the AllToAll collective operation.
    """

    row_id_map: torch.Tensor
    tokens_per_experts_group: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


class TorchAll2AllPrefillingDispatchResult(PrefillingDispatchResult):
    row_id_map: torch.Tensor


class TorchAll2AllDecodingDispatchResult(DecodingDispatchResult):
    row_id_map: torch.Tensor


TorchAll2AllPrefillingCombineResult = PrefillingCombineResult
TorchAll2AllDecodingCombineResult = DecodingCombineResult


class TorchAll2AllDispatcher(
    GenericDispatcher[
        TorchAll2AllPreDispatchResult,
        TorchAll2AllPrefillingDispatchResult,
        TorchAll2AllDecodingDispatchResult,
        TorchAll2AllPrefillingCombineResult,
        TorchAll2AllDecodingCombineResult,
    ]
):
    _process_group: dist.ProcessGroup

    def __init__(
        self,
        *,
        process_group: torch.distributed.ProcessGroup,
        config: MoEConfig,
    ):
        super().__init__(
            process_group=process_group,
            config=config,
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
        if config.training_dtype == "fp8":
            raise NotImplementedError

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        decoding: Literal[True],
    ) -> TorchAll2AllDecodingDispatchResult: ...

    @overload
    def dispatch(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        decoding: Literal[False],
    ) -> TorchAll2AllPrefillingDispatchResult: ...

    def dispatch(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        decoding: bool = False,
    ) -> TorchAll2AllPrefillingDispatchResult | TorchAll2AllDecodingDispatchResult:
        token_per_expert_group = pre_dispatched["tokens_per_experts_group"]
        token_counts = token_per_expert_group.ravel()
        global_input_tokens_local_experts_indices = torch.repeat_interleave(self._expert_ids_per_ep_rank, token_counts)
        global_input_tokens = all_to_all_single_autograd(
            pre_dispatched["hidden_states"],
            output_split_sizes=pre_dispatched["output_splits"],
            input_split_sizes=pre_dispatched["input_splits"],
            group=self._process_group,
        )
        global_input_tokens, row_id_map = permute(
            global_input_tokens,
            global_input_tokens_local_experts_indices.to(torch.int32),
        )
        tokens_per_experts = token_per_expert_group.sum(dim=0)
        if not decoding:
            return TorchAll2AllPrefillingDispatchResult(
                hidden_states=global_input_tokens,
                tokens_per_experts=tokens_per_experts,
                topk_weights=pre_dispatched["topk_weights"],
                row_id_map=row_id_map,
            )
        else:
            return TorchAll2AllDecodingDispatchResult(
                hidden_states=global_input_tokens,
                tokens_per_experts=tokens_per_experts,
                row_id_map=row_id_map,
            )

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> TorchAll2AllPreDispatchResult:
        ep_size = self._process_group.size()
        num_experts_per_rank = self._n_routed_experts // ep_size

        tokens_per_expert = torch.histc(topk_ids, bins=self._n_routed_experts, min=0, max=self._n_routed_experts)
        # permute output 相同 expert 的tokens在一起
        permuted_hidden_states, row_ids_map = permute(hidden_states, topk_ids.to(torch.int32))
        input_splits = (
            tokens_per_expert.reshape(ep_size, num_experts_per_rank).sum(dim=1).to(device=torch.device("cpu")).tolist()
        )
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(
            tokens_per_expert_group,
            tokens_per_expert,
            group=self._process_group,
        )

        # (r0e0, r0e1, ..., r0ei-1,
        #  r1e0, r1e1, ..., r1ei-1,
        tokens_per_expert_group = tokens_per_expert_group.view(ep_size, -1)

        # Get number experts each group
        output_splits = tokens_per_expert_group.sum(dim=-1).to(device=torch.device("cpu")).tolist()

        return TorchAll2AllPreDispatchResult(
            hidden_states=permuted_hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights.to(torch.float32),
            row_id_map=row_ids_map,
            tokens_per_experts_group=tokens_per_expert_group,
            input_splits=input_splits,
            output_splits=output_splits,
        )

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatch_result: TorchAll2AllPrefillingDispatchResult,
        decoding: Literal[False],
    ) -> TorchAll2AllPrefillingCombineResult: ...

    @overload
    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatch_result: TorchAll2AllDecodingDispatchResult,
        decoding: Literal[True],
    ) -> TorchAll2AllDecodingCombineResult: ...

    def combine(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatch_result: TorchAll2AllPrefillingDispatchResult | TorchAll2AllDecodingDispatchResult,
        decoding: bool = False,
    ) -> TorchAll2AllPrefillingCombineResult | TorchAll2AllDecodingCombineResult:
        hidden_states = unpermute(
            hidden_states,
            dispatch_result["row_id_map"],
        )

        hidden_states = all_to_all_single_autograd(
            hidden_states,
            input_split_sizes=pre_dispatched["output_splits"],
            output_split_sizes=pre_dispatched["input_splits"],
            group=self._process_group,
        )
        if not decoding:
            return TorchAll2AllPrefillingCombineResult(
                hidden_states=hidden_states,
            )
        else:
            return TorchAll2AllDecodingCombineResult(
                hidden_states=hidden_states,
            )

    def combine_post_process(
        self,
        *,
        pre_dispatched: TorchAll2AllPreDispatchResult,
        dispatch_result: TorchAll2AllPrefillingDispatchResult | TorchAll2AllDecodingDispatchResult,
        combine_result: TorchAll2AllDecodingCombineResult | TorchAll2AllPrefillingCombineResult,
    ) -> HiddenStates:
        hidden_states = unpermute(
            combine_result["hidden_states"],
            pre_dispatched["row_id_map"],
            probs=pre_dispatched["topk_weights"],
        )
        return hidden_states
