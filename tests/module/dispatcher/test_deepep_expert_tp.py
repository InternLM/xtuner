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
    def test_sync_path_uses_deepep_received_source_rows_for_expert_tp(self) -> None:
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
        )
        assert isinstance(dispatcher, DeepEPDispatcher)

        local_hidden, local_topk_ids, local_topk_weights = _source_payload(rank, device)
        hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)

        pre_dispatched = dispatcher.dispatch_preprocess(
            hidden_states=hidden_leaf,
            topk_ids=local_topk_ids,
        )
        dispatched = dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights_leaf,
            decoding=False,
        )

        # 中文注释：DeepEP + ExpertTP 的 TP row counts 描述 DeepEP dispatch
        # 收到的 source-token rows，不是 topK 展开后的 route-copy rows。
        expected_tp_rank_row_counts = [
            sum(ep * tp_size + expected_tp_rank + 2 for ep in range(ep_size))
            for expected_tp_rank in range(tp_size)
        ]
        assert dispatched["tp_rank_row_counts"] == expected_tp_rank_row_counts
        assert dispatched["hidden_states"].shape[0] == sum(expected_tp_rank_row_counts)
        assert dispatched["topk_ids"].shape[0] == sum(expected_tp_rank_row_counts)
        assert dispatched["topk_weights"].shape[0] == sum(expected_tp_rank_row_counts)

        token_markers = dispatched["hidden_states"][:, 0].float()
        expected_gathered_weights = token_markers.unsqueeze(1) / 1000 + torch.tensor(
            [0.1, 0.2, 0.3, 0.4],
            device=device,
            dtype=torch.float32,
        )
        valid_topk_slots = dispatched["topk_ids"] >= 0
        torch.testing.assert_close(
            dispatched["topk_weights"][valid_topk_slots],
            expected_gathered_weights[valid_topk_slots],
        )

        post_dispatched = dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )

        raw_counts_by_tp_rank: list[list[int] | None] = [None for _ in range(tp_size)]
        dist.all_gather_object(raw_counts_by_tp_rank, dispatched["num_recv_tokens_per_expert_list"], group=tp_group)
        expected_tokens_per_expert = torch.tensor(
            raw_counts_by_tp_rank,
            dtype=torch.long,
            device=device,
        ).sum(dim=0)
        torch.testing.assert_close(post_dispatched["tokens_per_expert"], expected_tokens_per_expert)
        assert dispatched["num_recv_tokens_per_expert_list"] == raw_counts_by_tp_rank[tp_rank]
        assert int(post_dispatched["tokens_per_expert"].sum().item()) > sum(dispatched["tp_rank_row_counts"])

        # 中文注释：dispatcher 测试不模拟真实 row-parallel expert 权重；
        # 每个 ExpertTP rank 产出 1/tp_size partial，combine 的 ReduceScatterRowsSum
        # 应恢复完整 expert output 后再交给 DeepEP combine。
        expert_output = post_dispatched["hidden_states"] / tp_size
        pre_combined = dispatcher.combine_preprocess(
            hidden_states=expert_output,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
        )
        assert pre_combined["hidden_states"].shape[0] == sum(expected_tp_rank_row_counts)

        combined = dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        assert combined["hidden_states"].shape == local_hidden.shape

        post_combined = dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )

        expected_output = (
            hidden_leaf.detach().float() * topk_weights_leaf.detach().sum(dim=1, keepdim=True)
        ).to(post_combined["hidden_states"].dtype)
        torch.testing.assert_close(
            post_combined["hidden_states"],
            expected_output,
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

        post_combined["hidden_states"].float().sum().backward()
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

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))
