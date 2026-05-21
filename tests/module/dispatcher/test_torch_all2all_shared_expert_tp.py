import unittest

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.dispatcher.base import DispacherInterface
from xtuner.v1.module.dispatcher.torch_all2all import TorchAll2AllDispatcher
from xtuner.v1.module.dispatcher.torch_all2all_tpep import TorchAll2AllTPEPDispatcher


def _build_ep_tp_groups(
    ep_size: int,
    tp_size: int,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup, list[dist.ProcessGroup]]:
    all_groups = []
    ep_groups = []
    tp_groups = []
    for tp_rank in range(tp_size):
        group = dist.new_group([ep_rank * tp_size + tp_rank for ep_rank in range(ep_size)], backend="nccl")
        ep_groups.append(group)
        all_groups.append(group)
    for ep_rank in range(ep_size):
        group = dist.new_group([ep_rank * tp_size + tp_rank for tp_rank in range(tp_size)], backend="nccl")
        tp_groups.append(group)
        all_groups.append(group)

    rank = dist.get_rank()
    return ep_groups[rank % tp_size], tp_groups[rank // tp_size], all_groups


def _payload_for_rank(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = rank + 2
    hidden_size = 8
    token_ids = torch.arange(sum(i + 2 for i in range(rank)), sum(i + 2 for i in range(rank + 1)), device=device)
    hidden = token_ids.to(torch.float32).unsqueeze(1) * 10 + torch.arange(hidden_size, device=device)
    topk_ids = torch.stack((token_ids % 4, (token_ids + 1) % 4), dim=1).to(torch.int64)
    topk_weights = torch.stack(
        (
            torch.full((rows,), 1.0, device=device),
            torch.full((rows,), 0.2 * (rank + 1), device=device),
        ),
        dim=1,
    )
    return hidden, topk_ids, topk_weights


def _run_dispatcher(
    dispatcher: DispacherInterface,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    async_op: bool = False,
):
    pre_dispatched = dispatcher.dispatch_preprocess(hidden_states=hidden_states, topk_ids=topk_ids, async_op=async_op)
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
    # 中文注释：dispatcher 级别不跑真实 row-parallel expert，
    # 两个 TP rank 各提供一半 partial output。
    experts_results = post_dispatched["hidden_states"] / 2
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
        decoding=False,
        async_op=async_op,
    )
    result = dispatcher.combine_postprocess(
        pre_dispatched=pre_dispatched,
        dispatched=dispatched,
        post_dispatched=post_dispatched,
        pre_combined=pre_combined,
        combined=combined,
        async_op=async_op,
    )
    return result, dispatched, post_dispatched, pre_combined, combined


def _record_shared_expert_tp_stages(dispatcher: TorchAll2AllDispatcher) -> dict[str, list[str | int]]:
    stages: dict[str, list[str | int]] = {
        "async_op_true": [],
        "async_all_gather_rows": [],
        "async_all_gather_per_rank_metadata": [],
        "async_reduce_scatter_rows_sum": [],
        "comm_stream": [],
    }
    current_stage: list[str] = []
    expert_tp = dispatcher._expert_tp
    assert expert_tp is not None

    for stage_name in (
        "dispatch_preprocess",
        "dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "combine",
        "combine_postprocess",
    ):
        original_stage = getattr(dispatcher, stage_name)

        def stage_wrapper(*args, _original_stage=original_stage, _stage_name=stage_name, **kwargs):
            if kwargs.get("async_op", False):
                stages["async_op_true"].append(_stage_name)
            current_stage.append(_stage_name)
            try:
                return _original_stage(*args, **kwargs)
            finally:
                current_stage.pop()

        setattr(dispatcher, stage_name, stage_wrapper)

    for collective_name in (
        "async_all_gather_rows",
        "async_all_gather_per_rank_metadata",
        "async_reduce_scatter_rows_sum",
    ):
        original_collective = getattr(expert_tp, collective_name)

        def collective_wrapper(
            *args,
            _original_collective=original_collective,
            _collective_name=collective_name,
            **kwargs,
        ):
            stages[_collective_name].append(current_stage[-1] if current_stage else "<outside>")
            stages["comm_stream"].append(kwargs["comm_stream"].cuda_stream)
            return _original_collective(*args, **kwargs)

        setattr(expert_tp, collective_name, collective_wrapper)

    return stages


def _assert_shared_expert_tp_async_stages(
    stages: dict[str, list[str | int]],
    dispatcher: TorchAll2AllDispatcher,
) -> None:
    assert set(stages["async_op_true"]) == {
        "dispatch_preprocess",
        "dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "combine",
        "combine_postprocess",
    }
    assert stages["async_all_gather_rows"] == ["dispatch"]
    assert stages["async_all_gather_per_rank_metadata"] == ["dispatch"]
    assert stages["async_reduce_scatter_rows_sum"] == ["combine"]
    assert set(stages["comm_stream"]) == {dispatcher._comm_stream.cuda_stream}


@unittest.skipIf(not torch.cuda.is_available(), "CUDA/NCCL is required for real All2All ExpertTP validation.")
class TestTorchAll2AllSharedExpertTP(DeterministicDDPTestCase):
    def test_build_dispatcher_uses_shared_all2all_expert_tp(self) -> None:
        pg = self.create_pg("cuda")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        ep_group, tp_group, all_groups = _build_ep_tp_groups(ep_size=2, tp_size=2)

        dispatcher = build_dispatcher(
            dispatcher="all2all",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
        )

        assert isinstance(dispatcher, TorchAll2AllDispatcher)
        assert not isinstance(dispatcher, TorchAll2AllTPEPDispatcher)
        assert dispatcher._expert_tp is not None

        dist.barrier()
        for group in all_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    def test_sync_shared_all2all_matches_legacy_tpep(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
        ep_group, tp_group, all_groups = _build_ep_tp_groups(ep_size=2, tp_size=2)

        shared_dispatcher = build_dispatcher(
            dispatcher="all2all",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
        )
        legacy_dispatcher = TorchAll2AllTPEPDispatcher(
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
        )

        local_hidden, local_topk_ids, local_topk_weights = _payload_for_rank(rank, device)
        shared_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        shared_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        shared_result, shared_dispatched, shared_post, shared_pre_combined, shared_combined = _run_dispatcher(
            shared_dispatcher,
            shared_hidden_leaf * 1.25,
            local_topk_ids,
            shared_topk_weights_leaf * 0.5,
        )
        shared_loss = shared_result["hidden_states"].square().sum()
        shared_loss.backward()

        legacy_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        legacy_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        legacy_result, legacy_dispatched, legacy_post, legacy_pre_combined, legacy_combined = _run_dispatcher(
            legacy_dispatcher,
            legacy_hidden_leaf * 1.25,
            local_topk_ids,
            legacy_topk_weights_leaf * 0.5,
        )
        legacy_loss = legacy_result["hidden_states"].square().sum()
        legacy_loss.backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(shared_dispatched["hidden_states"], legacy_dispatched["hidden_states"])
        torch.testing.assert_close(
            shared_dispatched["tokens_per_expert_group"],
            legacy_dispatched["tokens_per_expert_group"],
        )
        assert shared_dispatched["tp_rank_row_counts"] == legacy_dispatched["tp_rank_row_counts"]
        torch.testing.assert_close(shared_post["tokens_per_expert"], legacy_post["tokens_per_expert"])
        torch.testing.assert_close(shared_pre_combined["hidden_states"], legacy_pre_combined["hidden_states"])
        torch.testing.assert_close(shared_combined["hidden_states"], legacy_combined["hidden_states"])
        torch.testing.assert_close(shared_result["hidden_states"], legacy_result["hidden_states"])
        assert shared_hidden_leaf.grad is not None
        assert legacy_hidden_leaf.grad is not None
        assert shared_topk_weights_leaf.grad is not None
        assert legacy_topk_weights_leaf.grad is not None
        torch.testing.assert_close(shared_hidden_leaf.grad, legacy_hidden_leaf.grad)
        torch.testing.assert_close(shared_topk_weights_leaf.grad, legacy_topk_weights_leaf.grad)

        dist.barrier()
        for group in all_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    def test_async_shared_all2all_uses_dispatcher_comm_stream(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
        ep_group, tp_group, all_groups = _build_ep_tp_groups(ep_size=2, tp_size=2)

        sync_dispatcher = build_dispatcher(
            dispatcher="all2all",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
        )
        async_dispatcher = build_dispatcher(
            dispatcher="all2all",
            n_routed_experts=4,
            ep_group=ep_group,
            tp_group=tp_group,
        )
        assert isinstance(sync_dispatcher, TorchAll2AllDispatcher)
        assert isinstance(async_dispatcher, TorchAll2AllDispatcher)
        stages = _record_shared_expert_tp_stages(async_dispatcher)

        local_hidden, local_topk_ids, local_topk_weights = _payload_for_rank(rank, device)
        sync_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        sync_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        sync_result, *_ = _run_dispatcher(
            sync_dispatcher,
            sync_hidden_leaf * 1.25,
            local_topk_ids,
            sync_topk_weights_leaf * 0.5,
        )
        sync_result["hidden_states"].square().sum().backward()

        async_hidden_leaf = local_hidden.detach().clone().requires_grad_(True)
        async_topk_weights_leaf = local_topk_weights.detach().clone().requires_grad_(True)
        async_result, *_ = _run_dispatcher(
            async_dispatcher,
            async_hidden_leaf * 1.25,
            local_topk_ids,
            async_topk_weights_leaf * 0.5,
            async_op=True,
        )
        async_result["hidden_states"].square().sum().backward()
        torch.cuda.synchronize()

        _assert_shared_expert_tp_async_stages(stages, async_dispatcher)
        torch.testing.assert_close(async_result["hidden_states"], sync_result["hidden_states"])
        assert sync_hidden_leaf.grad is not None
        assert async_hidden_leaf.grad is not None
        assert sync_topk_weights_leaf.grad is not None
        assert async_topk_weights_leaf.grad is not None
        torch.testing.assert_close(async_hidden_leaf.grad, sync_hidden_leaf.grad)
        torch.testing.assert_close(async_topk_weights_leaf.grad, sync_topk_weights_leaf.grad)

        dist.barrier()
        for group in all_groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
