from unittest import TestCase
from unittest.mock import Mock

import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from xtuner.v1.module.dispatcher.base import NaiveDispatcher, GenericDispatcher
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher
from xtuner.v1.config import MoEConfig
import parametrize


import os


EP_SIZE = 8


def mock_experts(hidden_states: torch.Tensor, tokens_per_exprts: torch.Tensor):
    return hidden_states


class TestNoETorchAll2AllDispatcher(DistributedTestBase):
    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    def test_dispatch_and_combine(self, dtype, device):
        self.create_pg(device)
        num_experts = 16
        noep_dispatcher = NaiveDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
        )

        all2all_dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=torch.distributed.group.WORLD
        )

        seq_len = 32
        hidden_size = 128
        topk_experts = 4
        hidden_states = torch.rand(seq_len, hidden_size).to(device).to(dtype)
        topk_idx = torch.randint(0, num_experts, (seq_len, topk_experts)).to(device).to(torch.int32)
        topk_weights = torch.ones(seq_len, topk_experts).to(device).to(torch.float32)

        noep_results = self._dispatcher_call(
            dispatcher=noep_dispatcher,
            hidden_states=hidden_states,
            topk_ids=topk_idx,
            topk_weights=topk_weights
        )
        all2all_results = self._dispatcher_call(
            dispatcher=all2all_dispatcher,
            hidden_states=hidden_states,
            topk_ids=topk_idx,
            topk_weights=topk_weights
        )

        self.assertTrue(torch.allclose(noep_results, all2all_results, atol=1e-6, rtol=1e-4))

    def _dispatcher_call(
            self,
            dispatcher: GenericDispatcher,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor
    ):
        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        dispatched = dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=False,
        )
        experts_results = mock_experts(
            hidden_states=dispatched["hidden_states"],
            tokens_per_exprts=dispatched["tokens_per_experts"],
        )
        combined = dispatcher.combine( hidden_states=experts_results,
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            decoding=False,
        )
        return dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
