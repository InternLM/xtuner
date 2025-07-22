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
            topk_weights=topk_weights,
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=False,
        )
        experts_results = mock_experts(
            hidden_states=dispatched["hidden_states"],
            tokens_per_exprts=dispatched["tokens_per_experts"],
        )
        combined = self.dispatcher.combine(
            hidden_states=experts_results,
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            decoding=False,
        )
        result = self.dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )
        self.assertTrue(torch.equal(result, target_results))
