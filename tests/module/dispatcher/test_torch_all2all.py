from unittest import TestCase
from unittest.mock import Mock

import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from xtuner.v1.module.dispatcher.base import NaiveDispatcher, DispacherInterface
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

        self.assertTrue(torch.allclose(noep_results["hidden_states"], all2all_results["hidden_states"], atol=1e-6, rtol=1e-4))

    def _dispatcher_call(
            self,
            dispatcher: DispacherInterface,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor
    ):
        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
        )
        dispatched = dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            decoding=False,
        )
        post_dispatched = dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        experts_results = mock_experts(
            hidden_states=post_dispatched["hidden_states"],
            tokens_per_exprts=post_dispatched["tokens_per_expert"],
        )
        pre_combined = dispatcher.combine_preprocess(
            hidden_states=experts_results,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
        )
        combined = dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        return dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
