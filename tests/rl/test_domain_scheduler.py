"""CPU checks for domain admission weights and concurrency budget invariants."""

import math
import random
from typing import Any

import pytest

from xtuner.v1.rl.agent_loop_manager.domain_scheduler import (
    AdaptiveDomainSchedulerConfig,
    DomainSchedulingState,
    solve_domain_budgets,
)


def config(**kwargs: Any) -> AdaptiveDomainSchedulerConfig:
    return AdaptiveDomainSchedulerConfig(
        **{**dict(target_groups={"slow": 1, "fast": 1}, initial_acceptance_rates={"slow": 0.5, "fast": 1.0}), **kwargs}
    )


def test_clipped_solver_matches_example_and_random_weighted_budget_invariants() -> None:
    solution = solve_domain_budgets(
        {"tau3": 12 / 0.6171735241502684, "automationbench": 25},
        {"tau3": 2975.0382000890872, "automationbench": 368.67213316666664},
        p_min=0,
        p_max=1.5,
        p_bar=0.75,
    )
    assert solution.group_caps == {"tau3": 48, "automationbench": 29}
    assert solution.c == pytest.approx(0.00316459, rel=1e-5)
    rng = random.Random(55)
    for _ in range(100):
        m = {str(i): rng.uniform(1, 100) for i in range(rng.randint(1, 8))}
        durations = {key: 10 ** rng.uniform(-2, 4) for key in m}
        p_min, p_bar, p_max = 0, rng.uniform(0, 2), 2
        solved = solve_domain_budgets(m, durations, p_min=p_min, p_max=p_max, p_bar=p_bar)
        assert sum(m[k] * solved.oversampling[k] for k in m) / sum(m.values()) == pytest.approx(p_bar)
        assert sum(solved.float_caps.values()) == pytest.approx((1 + p_bar) * sum(m.values()))
        assert all(0 <= solved.oversampling[k] <= 2 for k in m)
        assert all(solved.group_caps[k] == math.floor(solved.float_caps[k]) for k in m)
    for mean in (0, 1.5):
        solved = solve_domain_budgets({"a": 2, "b": 3}, {"a": 3, "b": 300}, p_min=0, p_max=1.5, p_bar=mean)
        assert all(p == mean for p in solved.oversampling.values())


def test_weighted_admission_uses_only_window_deficit_and_has_baseline_fallback() -> None:
    scheduler = DomainSchedulingState(config(selection_alpha=0, seed=2))
    scheduler.draw({"slow": 1, "fast": 0}, ["slow", "fast"])
    assert scheduler.selection_weights == {"slow": 0, "fast": 1}
    scheduler.draw({"slow": 1, "fast": 1}, ["slow", "fast"])
    # All eligible deficits are zero: retain B/r baseline admission.
    assert {scheduler.draw({"slow": 1, "fast": 1}, ["slow", "fast"]) for _ in range(100)} == {"slow", "fast"}
    other = DomainSchedulingState(config(selection_alpha=0, seed=2))
    a = DomainSchedulingState(config(seed=42))
    b = DomainSchedulingState(config(seed=42))
    assert [a.draw({}, ["slow", "fast"]) for _ in range(30)] == [b.draw({}, ["slow", "fast"]) for _ in range(30)]
    assert other.draw({}, []) is None
