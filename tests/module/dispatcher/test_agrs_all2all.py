import unittest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from xtuner.v1.module.dispatcher.base import NaiveDispatcher, DispacherInterface
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher
import parametrize
from xtuner.v1.module.dispatcher.agrs import MoEAGRSDispatcher
from xtuner.v1.module.router.greedy import GreedyGroupedRouter


import os


EP_SIZE = 8


def mock_experts(hidden_states: torch.Tensor, tokens_per_exprts: torch.Tensor):
    return hidden_states


class TestNoETorchAll2AllDispatcher(DistributedTestBase):
    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    # @unittest.skipIf(True, "none")
    def test_dispatch_and_combine(self, dtype, device):
        self.create_pg(device)
        num_experts = 128

        all2all_dispatcher = TorchAll2AllDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=torch.distributed.group.WORLD
        )

        agrs_dispatcher = MoEAGRSDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=torch.distributed.group.WORLD
        )

        seq_len = 32
        hidden_size = 128
        topk_experts = 8

        router = GreedyGroupedRouter(
            router_n_groups=topk_experts,
            n_routed_experts=num_experts,
            num_experts_per_tok=topk_experts,
            norm_topk_prob=True,
        )

        logits = torch.randn(seq_len, num_experts).cuda()
        router_out = router(logits)

        hidden_states = torch.rand(seq_len, hidden_size).to(device).to(dtype)

        all2all_results = self._dispatcher_call(
            dispatcher=all2all_dispatcher,
            hidden_states=hidden_states,
            topk_ids=router_out["topk_ids"],
            topk_weights=router_out["topk_weights"]
        )

        agrs_results = self._dispatcher_call(
            dispatcher=agrs_dispatcher,
            hidden_states=hidden_states,
            topk_ids=router_out["topk_ids"],
            topk_weights=router_out["topk_weights"]
        )

        self.assertTrue(torch.allclose(all2all_results["hidden_states"], agrs_results["hidden_states"], atol=1e-2, rtol=1e-2))

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
            decoding=False,
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
