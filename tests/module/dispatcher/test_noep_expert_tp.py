import os
import unittest

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.dispatcher.base import NaiveDispatcher


def _payload_for_rank(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = rank + 2
    hidden_size = 8
    start = sum(i + 2 for i in range(rank))
    token_ids = torch.arange(start, start + rows, device=device)
    hidden = token_ids.to(torch.float32).unsqueeze(1) * 10 + torch.arange(hidden_size, device=device)
    topk_ids = torch.stack((token_ids % 4, (token_ids + 1) % 4), dim=1).to(torch.int64)
    topk_weights = torch.stack(
        (
            torch.full((rows,), 1.0, device=device),
            torch.full((rows,), 0.25 * (rank + 1), device=device),
        ),
        dim=1,
    )
    return hidden, topk_ids, topk_weights


def _run_dispatcher(
    dispatcher,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_scale: float = 1.0,
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
    # 中文注释：dispatcher 测试不跑真实 row-parallel expert；
    # 每个 TP rank 提供 1/tp_size 的 partial output，真实 ReduceScatterSum 后应回到 baseline。
    experts_results = post_dispatched["hidden_states"] * expert_scale
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
    result = dispatcher.combine_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        combined=combined,
    )
    return result, dispatched, post_dispatched, pre_combined, combined


@unittest.skipIf(not torch.cuda.is_available(), "CUDA/NCCL is required for real ExpertTP dispatcher validation.")
class TestNaiveExpertTPDispatcher(DeterministicDDPTestCase):
    def test_sync_path_uses_real_tp_collectives(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())

        ep_groups = [dist.new_group([ep_rank], backend="nccl") for ep_rank in range(world_size)]
        ep_group = ep_groups[rank]

        local_hidden, local_topk_ids, local_topk_weights = _payload_for_rank(rank, device)
        full_payloads = [_payload_for_rank(tp_rank, device) for tp_rank in range(world_size)]
        full_hidden = torch.cat([payload[0] for payload in full_payloads], dim=0)
        full_topk_ids = torch.cat([payload[1] for payload in full_payloads], dim=0)
        full_topk_weights = torch.cat([payload[2] for payload in full_payloads], dim=0)

        baseline = NaiveDispatcher(n_routed_experts=4)
        baseline_result, _, baseline_post, _, _ = _run_dispatcher(
            baseline,
            full_hidden,
            full_topk_ids,
            full_topk_weights,
        )

        dispatcher = build_dispatcher(
            dispatcher=None,
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=dist.group.WORLD,
        )
        result, dispatched, post_dispatched, pre_combined, combined = _run_dispatcher(
            dispatcher,
            local_hidden,
            local_topk_ids,
            local_topk_weights,
            expert_scale=1.0 / world_size,
        )

        all_sizes = [tp_rank + 2 for tp_rank in range(world_size)]
        slice_start = sum(all_sizes[:rank])
        slice_end = slice_start + all_sizes[rank]

        torch.testing.assert_close(dispatched["hidden_states"], full_hidden)
        torch.testing.assert_close(dispatched["topk_ids"], full_topk_ids)
        torch.testing.assert_close(dispatched["topk_weights"], full_topk_weights)
        torch.testing.assert_close(post_dispatched["tokens_per_expert"], baseline_post["tokens_per_expert"])
        torch.testing.assert_close(pre_combined["hidden_states"], baseline_result["hidden_states"] / world_size)
        torch.testing.assert_close(combined["hidden_states"], baseline_result["hidden_states"][slice_start:slice_end])
        torch.testing.assert_close(result["hidden_states"], baseline_result["hidden_states"][slice_start:slice_end])

        dist.barrier()
        for group in ep_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
