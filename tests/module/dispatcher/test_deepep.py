from unittest.mock import Mock
from typing import cast

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.module.dispatcher.deepep import DeepEPDispatcher
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.module.dispatcher.base import NaiveDispatcher, GenericDispatcher
import parametrize


import os


def mock_experts(hidden_states: torch.Tensor, tokens_per_exprts: torch.Tensor):
    return hidden_states


class TestMoETorchAll2AllDispatcher(DistributedTestBase):
    @parametrize.parametrize(
        "dtype,device,async_op",
        [
            (torch.bfloat16, "cuda", False),
            (torch.bfloat16, "cuda", True),
        ]
    )
    def test_dispatch_and_combine(self, dtype, device, async_op):
        self.create_pg(device)
        num_experts = 16

        noep_dispatcher = NaiveDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
        )

        all2all_dispatcher = DeepEPDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=cast(dist.ProcessGroup, dist.group.WORLD)
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
            topk_weights=topk_weights,
            async_op=async_op,
        )

        self.assertTrue(torch.allclose(noep_results, all2all_results, atol=1e-6, rtol=1e-4))

    def test_sync_dispatch_and_combine_buffers_are_reusable_once_freed(self):
        # A synchronous DeepEP call already orders the compute stream after the communication kernels, so its
        # buffers must neither live in a comm-stream allocator pool nor stay pending on comm-stream events after
        # the host frees them. At 512K/SP8 such pending multi-GiB buffers made the next large allocation miss and
        # forced a device-wide `release_cached_blocks`.
        self.create_pg("cuda")
        num_experts = 16
        dispatcher = DeepEPDispatcher(
            n_routed_experts=num_experts,
            training_dtype="bf16",
            process_group=cast(dist.ProcessGroup, dist.group.WORLD),
        )
        hidden_states = torch.rand(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        topk_idx = torch.randint(0, num_experts, (32, 4), device="cuda", dtype=torch.int32)
        topk_weights = torch.ones(32, 4, device="cuda", dtype=torch.float32, requires_grad=True)

        def fwd_bwd():
            out = self._dispatcher_call(dispatcher, hidden_states, topk_idx, topk_weights)
            # DeepEP requires a contiguous gradient, which `out.sum().backward()` would not give.
            out.backward(torch.ones_like(out))
            torch.cuda.synchronize()

        # Warm up so that the DeepEP buffer is created outside the recorded window.
        fwd_bwd()
        torch.cuda.memory._record_memory_history(max_entries=100000)
        try:
            fwd_bwd()
            snapshot = torch.cuda.memory._snapshot()
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

        events = [event for trace in snapshot["device_traces"] for event in trace]
        compute_stream = torch.cuda.current_stream().cuda_stream
        self.assertEqual({event["stream"] for event in events if event["action"] == "alloc"}, {compute_stream})
        # Without cross-stream uses, the allocator completes a free right when the host requests it.
        pending_frees = [
            (event["size"], event["addr"])
            for index, event in enumerate(events)
            if event["action"] == "free_requested"
            and not (events[index + 1]["action"] == "free_completed" and events[index + 1]["addr"] == event["addr"])
        ]
        self.assertEqual(pending_frees, [])

    def _dispatcher_call(
        self,
        dispatcher: GenericDispatcher,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool=False
    ):
        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            async_op=async_op,
        )
        dispatched = dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            decoding=False,
            async_op=async_op,
        )
        post_dispatched = dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            async_op=async_op,
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
            async_op=async_op,
        )
        combined = dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            async_op=async_op,
        )
        post_combined = dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
            async_op=async_op,
        )
        return post_combined["hidden_states"]

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
