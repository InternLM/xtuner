import os
import unittest

import torch
import torch.distributed as dist
from torch.testing._comparison import default_tolerances

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.dispatcher.deepep import DeepEPDispatcher


BF16_RTOL, BF16_ATOL = default_tolerances(torch.bfloat16)
FLOAT32_RTOL, FLOAT32_ATOL = default_tolerances(torch.float32)


def _source_payload(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = rank + 2
    hidden_size = 128
    token_markers = torch.arange(rows, device=device, dtype=torch.float32) + rank * 10
    hidden = token_markers.unsqueeze(1) + torch.arange(hidden_size, device=device, dtype=torch.float32) / 100
    topk_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64).expand(rows, -1).contiguous()
    slot_offsets = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float32)
    topk_weights = token_markers.unsqueeze(1) / 1000 + slot_offsets
    return hidden.to(torch.bfloat16), topk_ids, topk_weights


def _build_ep_tp_groups(ep_size: int, tp_size: int, backend: str = "nccl"):
    ep_groups = [
        dist.new_group([ep_rank * tp_size + tp_rank for ep_rank in range(ep_size)], backend=backend)
        for tp_rank in range(tp_size)
    ]
    tp_groups = [
        dist.new_group([ep_rank * tp_size + tp_rank for tp_rank in range(tp_size)], backend=backend)
        for ep_rank in range(ep_size)
    ]
    return ep_groups, tp_groups


@unittest.skipIf(not torch.cuda.is_available(), "CUDA/NCCL is required for real DeepEP ExpertTP validation.")
class TestDeepEPExpertTPDispatcher(DeterministicDDPTestCase):
    def test_sync_virtual_expert_path_preserves_output_and_gradients(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())

        ep_size = 2
        tp_size = 2
        ep_rank = rank // tp_size
        tp_rank = rank % tp_size
        ep_groups, tp_groups = _build_ep_tp_groups(ep_size, tp_size)
        ep_group = ep_groups[tp_rank]
        tp_group = tp_groups[ep_rank]

        dispatcher = build_dispatcher(
            dispatcher="deepep",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
            ep_tp_group=dist.group.WORLD,
        )
        assert isinstance(dispatcher, DeepEPDispatcher)

        local_hidden, local_topk_ids, local_topk_weights = _source_payload(rank, device)
        hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)

        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_leaf,
            topk_ids=local_topk_ids,
            topk_weights=topk_weights_leaf,
            tokens_per_expert=torch.bincount(local_topk_ids.flatten(), minlength=4),
        )
        expected_virtual_ids = torch.tensor(
            [0, 2, 1, 3, 4, 6, 5, 7],
            device=device,
            dtype=torch.int64,
        ).expand(local_topk_ids.shape[0], -1)
        torch.testing.assert_close(
            pre_dispatched["topk_ids"],
            expected_virtual_ids,
        )
        torch.testing.assert_close(
            pre_dispatched["topk_weights"],
            topk_weights_leaf.repeat_interleave(tp_size, dim=-1),
        )

        result = self._run_public_api(
            dispatcher=dispatcher,
            hidden_states=hidden_leaf,
            topk_ids=local_topk_ids,
            topk_weights=topk_weights_leaf,
            tp_size=tp_size,
            async_op=False,
        )

        expected_output = (
            hidden_leaf.detach().float() * topk_weights_leaf.detach().sum(dim=1, keepdim=True)
        ).to(result["hidden_states"].dtype)
        torch.testing.assert_close(
            result["hidden_states"],
            expected_output,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        result["hidden_states"].float().sum().backward()
        assert hidden_leaf.grad is not None
        assert topk_weights_leaf.grad is not None
        expected_hidden_grad = topk_weights_leaf.detach().sum(dim=1, keepdim=True).expand_as(hidden_leaf)
        expected_hidden_grad = expected_hidden_grad.to(hidden_leaf.grad.dtype)
        expected_topk_grad = hidden_leaf.detach().float().sum(dim=1, keepdim=True).expand_as(topk_weights_leaf)
        torch.testing.assert_close(
            hidden_leaf.grad,
            expected_hidden_grad,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
        torch.testing.assert_close(
            topk_weights_leaf.grad,
            expected_topk_grad,
            atol=FLOAT32_ATOL,
            rtol=FLOAT32_RTOL,
        )

        dist.barrier()
        for group in ep_groups + tp_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    def test_async_path_matches_sync_output_and_gradients(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())

        ep_size = 2
        tp_size = 2
        ep_rank = rank // tp_size
        tp_rank = rank % tp_size
        ep_groups, tp_groups = _build_ep_tp_groups(ep_size, tp_size)
        ep_group = ep_groups[tp_rank]
        tp_group = tp_groups[ep_rank]

        dispatcher = build_dispatcher(
            dispatcher="deepep",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
            ep_tp_group=dist.group.WORLD,
        )

        local_hidden, local_topk_ids, local_topk_weights = _source_payload(rank, device)

        sync_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        sync_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        sync_result = self._run_public_api(
            dispatcher=dispatcher,
            hidden_states=sync_hidden_leaf * 1.25,
            topk_ids=local_topk_ids,
            topk_weights=sync_topk_weights_leaf * 0.5,
            tp_size=tp_size,
            async_op=False,
        )
        sync_result["hidden_states"].float().sum().backward()

        async_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        async_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        async_result = self._run_public_api(
            dispatcher=dispatcher,
            hidden_states=async_hidden_leaf * 1.25,
            topk_ids=local_topk_ids,
            topk_weights=async_topk_weights_leaf * 0.5,
            tp_size=tp_size,
            async_op=True,
        )
        async_result["hidden_states"].float().sum().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            async_result["hidden_states"],
            sync_result["hidden_states"],
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
        assert sync_hidden_leaf.grad is not None
        assert async_hidden_leaf.grad is not None
        assert sync_topk_weights_leaf.grad is not None
        assert async_topk_weights_leaf.grad is not None
        torch.testing.assert_close(
            async_hidden_leaf.grad,
            sync_hidden_leaf.grad,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )
        torch.testing.assert_close(
            async_topk_weights_leaf.grad,
            sync_topk_weights_leaf.grad,
            atol=FLOAT32_ATOL,
            rtol=FLOAT32_RTOL,
        )

        dist.barrier()
        for group in ep_groups + tp_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    def test_async_path_accepts_topk_weights_without_gradients(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())

        ep_size = 2
        tp_size = 2
        ep_rank = rank // tp_size
        tp_rank = rank % tp_size
        ep_groups, tp_groups = _build_ep_tp_groups(ep_size, tp_size)
        ep_group = ep_groups[tp_rank]
        tp_group = tp_groups[ep_rank]

        dispatcher = build_dispatcher(
            dispatcher="deepep",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
            ep_tp_group=dist.group.WORLD,
        )

        local_hidden, local_topk_ids, local_topk_weights = _source_payload(rank, device)
        hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        topk_weights = local_topk_weights.detach().clone()
        assert topk_weights.requires_grad is False

        result = self._run_public_api(
            dispatcher=dispatcher,
            hidden_states=hidden_leaf,
            topk_ids=local_topk_ids,
            topk_weights=topk_weights,
            tp_size=tp_size,
            async_op=True,
        )

        assert result["hidden_states"].shape == local_hidden.shape
        result["hidden_states"].float().sum().backward()
        torch.cuda.synchronize()
        assert hidden_leaf.grad is not None

        dist.barrier()
        for group in ep_groups + tp_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    def _run_public_api(
        self,
        *,
        dispatcher,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tp_size: int,
        async_op: bool,
    ) -> dict[str, torch.Tensor]:
        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            tokens_per_expert=torch.bincount(topk_ids.flatten(), minlength=4),
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
        # 中文注释：测试 dispatcher public API，不模拟真实 row-parallel expert；
        # 每个 ExpertTP rank 产出 1/tp_size partial，combine 应归约回完整输出。
        expert_output = post_dispatched["hidden_states"] / tp_size
        pre_combined = dispatcher.combine_preprocess(
            hidden_states=expert_output,
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
            decoding=False,
            async_op=async_op,
        )
        return dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
            async_op=async_op,
        )

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))
