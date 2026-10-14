"""Task failure scores and infrastructure failures have different semantics."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_trajectory_logging import TestSandboxArtifactOwnership as Fixtures

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import RolloutStatus
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import agent_in_sandbox_loop as impl
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import RolloutError


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(
    "failure", ["infra", "verifier", "missing_score", "nan", "transport_timeout", "cancel", "adapter"]
)
def test_invalid_attempt_never_exposes_a_reward(mode, failure):
    loop, state = Fixtures._make_loop(mode), Fixtures._make_state()
    item = Fixtures._make_item(RolloutStatus.COMPLETED)
    if failure in {"infra", "verifier"}:
        item.error = RolloutError(category=failure, message="failed")
        item.status = RolloutStatus.FAILED
    elif failure == "missing_score":
        item.reward = None
    elif failure == "nan":
        item.reward = float("nan")
    elif failure == "adapter":
        item.artifacts["response_message"]["extra_info"] = {"error": "connection failed"}
    else:
        item.artifacts["response_message"]["finish_reason"] = (
            "timeout" if failure == "transport_timeout" else "cancelled"
        )
    # A stale prior result must not survive a failed attempt.
    state.reward = {"score": 9}
    with patch.object(impl, "get_store") as store:
        result = asyncio.run(loop._build_rollout_states(state, item))
    assert result == [state]
    assert state.status == Status.FAILED
    assert state.reward is None
    assert state.error_msg
    assert "agent_artifacts" in state.extra_fields
    store.assert_not_called()


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("score", [0.0, 0.75])
def test_confirmed_task_timeout_uses_verifier_score(mode, score):
    loop, state = Fixtures._make_loop(mode), Fixtures._make_state()
    item = Fixtures._make_item(RolloutStatus.COMPLETED, segment_count=1)
    item.reward = score
    item.artifacts["response_message"].update(
        finish_reason="timeout", finish_info={"termination_reason": "task_budget_exhausted", "agent_stopped": True}
    )
    with patch.object(impl, "get_store", return_value=Fixtures._trace_store(1)):
        result = asyncio.run(loop._build_rollout_states(state, item))
    assert result[0].status == Status.COMPLETED
    assert result[0].reward["score"] == score
    assert result[0].finish_reason == "timeout"


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_execution_exception_discards_stale_reward_and_preserves_evidence(mode):
    loop, state = Fixtures._make_loop(mode), Fixtures._make_state()
    loop.logger = MagicMock()
    loop._run_item = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    state.extra_fields["rollout_item"] = Fixtures._make_item(RolloutStatus.COMPLETED)
    state.reward = {"score": 1}
    result = asyncio.run(loop.generate_sample(state))
    assert result[0].status == Status.FAILED
    assert result[0].reward is None
    assert result[0].extra_fields["agent_artifacts"]["response_message"]["content"] == "done"


def test_scored_episode_without_trace_is_not_trainable():
    loop, state = Fixtures._make_loop(), Fixtures._make_state()
    loop.logger = MagicMock()
    item = Fixtures._make_item(RolloutStatus.COMPLETED, segment_count=0)
    loop._run_item = AsyncMock(return_value=item)
    state.extra_fields["rollout_item"] = item
    result = asyncio.run(loop.generate_sample(state))
    assert result[0].status == Status.FAILED
    assert result[0].reward is None
