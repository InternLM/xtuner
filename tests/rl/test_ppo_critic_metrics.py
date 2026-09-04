"""Regression tests for PPO critic metric reduction."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import xtuner.v1.rl.trainer.worker as worker_module
from xtuner.v1.rl.trainer.worker import TrainingWorker


def _worker(loss_type: str = "clipped") -> TrainingWorker:
    worker = TrainingWorker.__new__(TrainingWorker)
    worker.config = SimpleNamespace(critic_cfg=SimpleNamespace(loss_cfg=SimpleNamespace(loss_type=loss_type)))
    return worker


def test_empty_rank_still_joins_critic_metric_collective() -> None:
    """A rank-local empty metric dict must not skip the world-size collective."""
    global_totals = torch.tensor([2.0, 1.0, 0.5, 2.0, 2.5, 0.5, 1.0], dtype=torch.float64)

    def fake_all_reduce(totals: torch.Tensor, *, op: object) -> None:
        del op
        # Stand in for rank 0's non-empty contribution reaching this empty
        # rank. The production regression was exactly one rank at seqnum 669
        # while every empty rank stopped at seqnum 668.
        totals.copy_(global_totals)

    with (
        patch.object(worker_module, "DEVICE", torch.device("cpu")),
        patch.object(worker_module.dist, "is_initialized", return_value=True),
        patch.object(worker_module.dist, "all_reduce", side_effect=fake_all_reduce) as all_reduce,
    ):
        metrics = _worker()._finalize_critic_metrics({})

    all_reduce.assert_called_once()
    assert all_reduce.call_args.args[0].numel() == 7
    assert metrics["critic/value_mean"] == pytest.approx(0.5)
    assert metrics["critic/return_mean"] == pytest.approx(1.0)
    assert metrics["critic/value_mse"] == pytest.approx(0.25)
    assert metrics["critic/clip_frac"] == pytest.approx(0.5)


def test_unclipped_critic_uses_the_same_six_element_payload_on_empty_rank() -> None:
    payload_sizes: list[int] = []

    def fake_all_reduce(totals: torch.Tensor, *, op: object) -> None:
        del op
        payload_sizes.append(totals.numel())

    with (
        patch.object(worker_module, "DEVICE", torch.device("cpu")),
        patch.object(worker_module.dist, "is_initialized", return_value=True),
        patch.object(worker_module.dist, "all_reduce", side_effect=fake_all_reduce),
    ):
        metrics = _worker(loss_type="mse")._finalize_critic_metrics({})

    assert payload_sizes == [6]
    assert metrics == {}
