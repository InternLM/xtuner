"""Tests for LRConfig.build, the shared LR scheduler factory."""

import math

import pytest
import torch

from xtuner.v1.config import LRConfig


def _optimizer(lr: float = 1e-5) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _lr_trace(lr_cfg: LRConfig, total_steps: int, num_steps: int | None = None) -> list[float]:
    """Learning rate observed before each of ``num_steps`` optimizer updates."""
    optimizer = _optimizer()
    scheduler = lr_cfg.build(optimizer, total_steps)
    trace = []
    for _ in range(num_steps or total_steps):
        trace.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return trace


class TestWarmup:
    def test_warmup_ratio_is_a_fraction_of_total_steps(self) -> None:
        trace = _lr_trace(LRConfig(lr_type="constant", warmup_ratio=0.1), total_steps=100)
        # 10 warmup steps ramping linearly from 0 to the base lr.
        assert trace[0] == 0.0
        assert trace[5] == pytest.approx(0.5e-5)
        assert trace[10] == pytest.approx(1e-5)
        assert trace[50] == pytest.approx(1e-5)

    def test_warmup_ratio_above_one_is_an_absolute_step_count(self) -> None:
        trace = _lr_trace(LRConfig(lr_type="constant", warmup_ratio=20), total_steps=100)
        assert trace[10] == pytest.approx(0.5e-5)
        assert trace[20] == pytest.approx(1e-5)

    def test_zero_warmup_starts_at_the_base_lr(self) -> None:
        # A SequentialLR with a zero milestone would skip warmup and advance the
        # decay by one step, so the scheduler must be returned unwrapped.
        trace = _lr_trace(LRConfig(lr_type="constant", warmup_ratio=0.0), total_steps=10)
        assert trace[0] == pytest.approx(1e-5)
        assert all(value == pytest.approx(1e-5) for value in trace)

    def test_zero_warmup_does_not_skip_the_decay_start(self) -> None:
        trace = _lr_trace(LRConfig(lr_type="cosine", warmup_ratio=0.0, lr_min=0.0), total_steps=10)
        # Step 0 must still be the full base lr, not one cosine step in.
        assert trace[0] == pytest.approx(1e-5)

    def test_warmup_is_capped_at_total_steps(self) -> None:
        trace = _lr_trace(LRConfig(lr_type="constant", warmup_ratio=500), total_steps=10)
        assert all(value <= 1e-5 for value in trace)
        assert bool(all(math.isfinite(value) for value in trace))


class TestDecaySchedules:
    def test_constant_holds_the_base_lr(self) -> None:
        trace = _lr_trace(LRConfig(lr_type="constant", warmup_ratio=0.0), total_steps=50)
        assert all(value == pytest.approx(1e-5) for value in trace)

    def test_linear_decays_towards_lr_min(self) -> None:
        lr_min = 1e-6
        # total_steps + 1 so the final observation is the schedule's endpoint.
        trace = _lr_trace(LRConfig(lr_type="linear", warmup_ratio=0.0, lr_min=lr_min), total_steps=100, num_steps=101)
        assert trace[0] == pytest.approx(1e-5)
        assert trace[-1] == pytest.approx(lr_min)
        # Monotonically non-increasing after warmup.
        assert all(a >= b - 1e-12 for a, b in zip(trace, trace[1:]))

    def test_cosine_decays_towards_lr_min(self) -> None:
        lr_min = 1e-6
        trace = _lr_trace(LRConfig(lr_type="cosine", warmup_ratio=0.0, lr_min=lr_min), total_steps=100, num_steps=101)
        assert trace[0] == pytest.approx(1e-5)
        assert trace[-1] == pytest.approx(lr_min)
        assert all(a >= b - 1e-12 for a, b in zip(trace, trace[1:]))

    def test_cosine_midpoint_is_halfway(self) -> None:
        lr_min = 0.0
        trace = _lr_trace(LRConfig(lr_type="cosine", warmup_ratio=0.0, lr_min=lr_min), total_steps=100)
        assert trace[50] == pytest.approx(0.5e-5, rel=1e-3)

    def test_lr_holds_at_lr_min_past_total_steps(self) -> None:
        lr_min = 1e-6
        total_steps = 50
        warmup_steps = int(0.05 * total_steps)
        for lr_type in ("linear", "cosine"):
            trace = _lr_trace(
                LRConfig(lr_type=lr_type, warmup_ratio=0.05, lr_min=lr_min),
                total_steps=total_steps,
                num_steps=200,
            )
            # Past total_steps the schedule must hold at lr_min, not run past
            # the end of the curve and go negative or turn back upwards.
            after_warmup = trace[warmup_steps:]
            assert min(after_warmup) >= lr_min - 1e-12, lr_type
            assert trace[-1] == pytest.approx(lr_min), lr_type
            assert trace[total_steps] == pytest.approx(lr_min), lr_type


class TestValidation:
    def test_non_positive_total_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be positive"):
            LRConfig().build(_optimizer(), 0)

    @pytest.mark.parametrize("lr_type", ["constant", "linear", "cosine"])
    def test_scheduler_state_round_trips(self, lr_type: str) -> None:
        """Resuming must restore the exact position in the schedule.

        `SequentialLR.load_state_dict` restores counters without reapplying the
        learning rate, so a resumed run silently continues from the wrong point.
        A single LambdaLR is used precisely to avoid that.
        """
        cfg = LRConfig(lr_type=lr_type, warmup_ratio=0.1, lr_min=1e-6)

        optimizer = _optimizer()
        scheduler = cfg.build(optimizer, 100)
        for _ in range(30):
            optimizer.step()
            scheduler.step()
        state = scheduler.state_dict()
        expected = optimizer.param_groups[0]["lr"]

        resumed_optimizer = _optimizer()
        resumed_scheduler = cfg.build(resumed_optimizer, 100)
        resumed_scheduler.load_state_dict(state)

        assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected)

        # And it must keep tracking the original for the rest of the run.
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            resumed_optimizer.step()
            resumed_scheduler.step()
            assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
