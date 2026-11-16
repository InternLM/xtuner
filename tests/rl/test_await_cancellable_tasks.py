"""真实执行边界的取消与完整清理测试（F1 / F2）。

Good Tests:
- 使用真实 asyncio 任务、真实 CPU Ray actor 和 RouterAgentLoop 观察取消传播与清理顺序。
- 只断言公开行为：清理完成、取消次数、worker 占用释放、最终异常类型。

Bad Tests:
- 不断言内部 Task 集合或 ray.cancel 的调用次数。
- 不把固定 sleep 当作完成凭据；事件只用于控制交错。
"""

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop.agent_loop import RouterAgentLoop
from xtuner.v1.rl.utils import await_cancellable_tasks


with patch.dict(
    sys.modules,
    {
        "lagent": MagicMock(),
        "lagent.utils": MagicMock(),
        "lagent.utils.rate_limiter": MagicMock(),
    },
):
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop


class TestAwaitCancellableTasks(unittest.TestCase):
    def test_local_children_finish_cleanup_before_cancellation_propagates(self):
        async def scenario() -> list[str]:
            events: list[str] = []
            started = [asyncio.Event(), asyncio.Event()]

            async def child(index: int, cleanup_delay: float) -> None:
                started[index].set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(cleanup_delay)
                    events.append(f"cleaned-{index}")

            owner = asyncio.create_task(await_cancellable_tasks([child(0, 0), child(1, 0.05)]))
            await asyncio.gather(*(event.wait() for event in started))
            owner.cancel()
            try:
                await owner
            except asyncio.CancelledError:
                events.append("owner-cancelled")
            return events

        events = asyncio.run(scenario())
        self.assertEqual(events, ["cleaned-0", "cleaned-1", "owner-cancelled"])

    def test_child_error_is_raised_only_after_siblings_finish(self):
        async def scenario() -> list[str]:
            events: list[str] = []
            started = asyncio.Event()

            async def failing() -> None:
                started.set()
                await asyncio.sleep(0)
                raise RuntimeError("generation failed")

            async def slow_sibling() -> None:
                await asyncio.sleep(0.05)
                events.append("sibling-cleaned")

            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                await await_cancellable_tasks([failing(), slow_sibling()])
            events.append("error-raised")
            return events

        self.assertEqual(asyncio.run(scenario()), ["sibling-cleaned", "error-raised"])


@ray.remote
class _SlowAgent:
    """Minimal async actor that mimics a remote Agent group with slow cleanup."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleanup_done = False

    async def generate_group(self, rollout_state, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)
            self.cleanup_done = True

    async def wait_started(self) -> None:
        await self.started.wait()

    def cleanup_finished(self) -> bool:
        return self.cleanup_done


class TestRemoteCancellation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                cls.started_ray = True
        except Exception as exc:
            raise unittest.SkipTest(f"Ray init failed for remote cancellation tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray and ray.is_initialized():
            ray.shutdown()

    def _state(self) -> RolloutState:
        return RolloutState(rollout_id=1, group_id=1, message=[], status=Status.INIT, prompt_ids=[1])

    def test_remote_object_ref_is_cancelled_and_cleanup_completes_before_return(self):
        async def scenario(actor) -> bool:
            ref = actor.generate_group.remote([self._state()])
            owner = asyncio.create_task(await_cancellable_tasks([ref]))
            await asyncio.wait_for(actor.wait_started.remote(), timeout=10)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            # The owner may return only after the remote finally completed.
            return ray.get(actor.cleanup_finished.remote())

        actor = _SlowAgent.remote()
        try:
            self.assertTrue(asyncio.run(scenario(actor)))
        finally:
            ray.kill(actor)

    def test_router_releases_worker_only_after_remote_cleanup(self):
        async def scenario(actor) -> tuple[bool, dict[str, int]]:
            router = RouterAgentLoop(workers=[actor], rollout_ctl=None)
            owner = asyncio.create_task(router.generate_group([self._state()]))
            await asyncio.wait_for(actor.wait_started.remote(), timeout=10)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            return ray.get(actor.cleanup_finished.remote()), router.get_worker_status()

        actor = _SlowAgent.remote()
        try:
            cleaned, loads = asyncio.run(scenario(actor))
        finally:
            ray.kill(actor)
        self.assertTrue(cleaned)
        self.assertEqual(list(loads.values()), [0])


class TestSandboxGroupCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_generate_group_waits_for_every_sample_cleanup(self):
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.sample_params = None
        loop._sample_semaphore = None
        loop.mode = "train"
        loop.is_valid_sample_fn = None
        loop.canonicalize_train_fields = lambda group: group
        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda sample: sample),
            on_group_ready=AsyncMock(side_effect=lambda samples: samples),
        )
        started = [asyncio.Event(), asyncio.Event()]
        cleaned: list[int] = []
        cancelled_counts = [0, 0]

        async def fake_generate_sample(state: RolloutState, **kwargs):
            index = state.rollout_id
            started[index].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_counts[index] += 1
                await asyncio.sleep(0.05 if index == 1 else 0)
                cleaned.append(index)
                raise
            return [state]

        loop.generate_sample = fake_generate_sample
        states = [
            RolloutState(rollout_id=0, group_id=0, message=[], status=Status.INIT, prompt_ids=[1]),
            RolloutState(rollout_id=1, group_id=0, message=[], status=Status.INIT, prompt_ids=[1]),
        ]

        owner = asyncio.create_task(loop.generate_group(states))
        await asyncio.gather(*(event.wait() for event in started))
        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner

        self.assertEqual(cleaned, [0, 1])
        self.assertEqual(cancelled_counts, [1, 1])


if __name__ == "__main__":
    unittest.main()
