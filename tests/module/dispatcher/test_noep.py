from unittest import TestCase
from unittest.mock import Mock

import torch
from xtuner.v1.module.dispatcher.base import NaiveDispatcher
from xtuner.v1.config import MoEConfig
import parametrize


def mock_experts(hidden_states: torch.Tensor, tokens_per_exprts: torch.Tensor):
    return hidden_states


class TestNoEPDispatcher(TestCase):
    def setUp(self):
        n_routed_experts = 4
        self.dispatcher = NaiveDispatcher(
            n_routed_experts=n_routed_experts,
        )

    @parametrize.parametrize("dtype,device", [(torch.bfloat16, "cuda")])
    def test_dispatch_and_combine(self, dtype, device):
        # seq len 16, hidden size 4
        # [0, 1, 2, 3]
        hidden_states = torch.arange(4).unsqueeze(1).to(device).to(dtype).repeat(1, 32)
        topk_ids = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ]
        ).to(device)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        # results:
        # [
        #   e(h[0]) + e(h[1]), -> 0
        #   e(h[1]) + e(h[2]), -> 2
        #   e(h[2]) + e(h[3]), -> 4
        #   e(h[3]) + e(h[0]), -> 6
        # ]
        target_results = torch.tensor(
            [
                [0],
                [2],
                [4],
                [6],
            ]
        ).to(device).to(dtype).repeat(1, 32)
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            decoding=False,
        )
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        experts_results = mock_experts(
            hidden_states=post_dispatched["hidden_states"],
            tokens_per_exprts=post_dispatched["tokens_per_expert"],
        )

        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_results,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
        )
        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        result = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        self.assertTrue(torch.equal(result["hidden_states"], target_results))