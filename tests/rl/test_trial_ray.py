"""Focused checks for Trial dispatch and cancellation ownership (no GPU)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from xtuner.v1.rl.agent_loop.sandbox_agent_loop import trial_ray
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem


def _item(pipeline):
    return AgentRolloutItem(id="trial", data_source="test", instruction="task", uid=17, pipeline=pipeline)


def test_local_adapter_preserves_preconstructed_runner():
    class Runner:
        async def run(self, item):
            item.reward = 0
            return item

    item = _item(Runner())
    assert asyncio.run(trial_ray.build_trial_executor("local")(item)) is item
    assert item.reward == 0


def test_ray_submission_keeps_item_fields_and_escapes_model_placement_group():
    async def scenario():
        item = _item({"type": "example.Runner"})
        item.metadata = {"provenance": "keep"}
        item.reward_model = {"answer": "reference"}
        result = item.model_copy(update={"reward": 0})
        future = asyncio.get_running_loop().create_future()
        future.set_result(result.model_dump(mode="python"))
        remote = MagicMock()
        remote.options.return_value.remote.return_value = future
        with patch.object(trial_ray, "_remote_item", remote):
            returned = await trial_ray.build_trial_executor("ray_task", 0.5)(item)
        assert returned.model_dump() == result.model_dump()
        strategy = remote.options.call_args.kwargs["scheduling_strategy"]
        assert strategy.placement_group is None
        assert strategy.placement_group_capture_child_tasks is False
        assert remote.options.call_args.kwargs["num_cpus"] == 0.5

    asyncio.run(scenario())


def test_cancellation_waits_for_original_remote_terminal_even_when_cancelled_twice():
    async def scenario():
        future = asyncio.get_running_loop().create_future()
        cancel_requested = asyncio.Event()
        events = []
        with patch.object(trial_ray.ray, "cancel", side_effect=lambda *a, **k: cancel_requested.set()):
            owner = asyncio.create_task(trial_ray._await_terminal(future))
            await asyncio.sleep(0)
            owner.cancel()
            await cancel_requested.wait()
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done()
            events.append("remote-finally")
            future.set_exception(RuntimeError("remote cancelled after cleanup"))
            with pytest.raises(asyncio.CancelledError):
                await owner
            events.append("parent-returned")
        assert events == ["remote-finally", "parent-returned"]

    asyncio.run(scenario())
