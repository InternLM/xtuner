"""Shutdown reaches all loop owners without changing the sandbox Runner API."""

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import ray
import test_sandbox_pool as sandbox_fixtures

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop, AgentLoopActor, RouterAgentLoop, close_agent_loops
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.runner import Runner
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager
from xtuner.v1.rl.agent_loop_manager.disagg_agent_loop_manager import DisaggAgentLoopManager
from xtuner.v1.rl.utils.async_utils import CleanupError, await_cleanup
from xtuner.v1.train.rl_trainer import RLColocateTrainer, RLDisaggregatedTrainer


class TestLoopShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_base_close_is_optional_for_resource_free_loops(self):
        class Loop(AgentLoop):
            async def generate_sample(self, state, **kwargs):
                return state

        loop = Loop.__new__(Loop)
        await loop.close()
        await loop.close()

    async def test_managers_close_local_actor_and_router_owners_despite_failure(self):
        for manager_type in (AgentLoopManager, DisaggAgentLoopManager):
            with self.subTest(manager_type=manager_type.__name__):
                failure = RuntimeError("local cleanup failed")
                local = SimpleNamespace(close=AsyncMock(side_effect=failure))
                actor = MagicMock(spec=ray.actor.ActorHandle)
                actor.close = SimpleNamespace(remote=AsyncMock())
                worker = MagicMock(spec=ray.actor.ActorHandle)
                worker.close = SimpleNamespace(remote=AsyncMock())
                router = RouterAgentLoop([worker], MagicMock())
                manager = manager_type.__new__(manager_type)
                manager.task_runners = [SimpleNamespace(agent_loop=loop) for loop in (local, actor, router, actor)]
                with self.assertRaises(CleanupError) as caught:
                    await manager.close()
                self.assertEqual(caught.exception.errors, (failure,))
                local.close.assert_awaited_once()
                actor.close.remote.assert_awaited_once()
                worker.close.remote.assert_awaited_once()

    async def test_actor_forwards_close_to_its_local_loop(self):
        actor = AgentLoopActor.__new__(AgentLoopActor)
        actor.agent_loop = SimpleNamespace(close=AsyncMock())
        await actor.close()
        actor.agent_loop.close.assert_awaited_once()

    async def test_cancelled_owner_is_not_reported_as_success(self):
        other = SimpleNamespace(close=AsyncMock())
        cancelled = SimpleNamespace(close=AsyncMock(side_effect=asyncio.CancelledError()))
        with self.assertRaises(CleanupError) as caught:
            await close_agent_loops([cancelled, other])
        self.assertIsInstance(caught.exception.errors[0], asyncio.CancelledError)
        other.close.assert_awaited_once()

    async def test_repeated_cancellation_waits_for_all_owners_and_keeps_failure(self):
        started, release = asyncio.Event(), asyncio.Event()
        closed = []
        failure = RuntimeError("remote close failed")

        async def close():
            started.set()
            await release.wait()
            closed.append(True)
            raise failure

        task = asyncio.create_task(close_agent_loops([SimpleNamespace(close=close)]))
        await started.wait()
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 2)
        self.assertEqual(closed, [True])
        self.assertIsInstance(caught.exception.__cause__, CleanupError)
        self.assertEqual(caught.exception.__cause__.errors, (failure,))

    async def test_cleanup_can_itself_be_cancelled_without_spinning(self):
        task = asyncio.create_task(asyncio.sleep(60))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(await_cleanup(task), 2)

    async def test_sandbox_close_waits_for_original_runner_finally_once(self):
        events = []
        started, deleting, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Provider(sandbox_fixtures.FakeProvider):
            async def delete(self, env_id):
                deleting.set()
                await release.wait()
                await super().delete(env_id)

        class Block:
            async def __call__(self, client, item, record):
                started.set()
                await asyncio.Event().wait()

        pool = sandbox_fixtures.pool(Provider(events), sandbox_fixtures.spec())
        runner = Runner(pool=pool, infer=SandboxStage(pre=[Block()]), validate=Judger("score", SandboxStage()))
        item = AgentRolloutItem(
            id="shutdown", data_source="test", instruction="task.md", task_root=Path("/tmp"), uid=1, pipeline=runner
        )
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop._active_runs = set()
        loop._closing = False
        loop._close_task = None
        running = asyncio.create_task(loop._run_item(item))
        await asyncio.wait_for(started.wait(), 2)
        closing = asyncio.create_task(loop.close())
        await asyncio.wait_for(deleting.wait(), 2)
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0)
        self.assertFalse(closing.done())
        with self.assertRaisesRegex(RuntimeError, "closing"):
            await loop._run_item(item)
        with self.assertRaisesRegex(RuntimeError, "closing"):
            await loop.generate_sample(RolloutState(message=[]))
        with self.assertRaisesRegex(RuntimeError, "closing"):
            await loop.generate_group([])
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 2)
        with self.assertRaises(asyncio.CancelledError):
            await running
        await asyncio.gather(loop.close(), loop.close())
        self.assertEqual(events.count("delete:env-1-agent"), 1)
        self.assertEqual(events.count("close:env-1-agent"), 1)
        self.assertFalse(loop._active_runs)


class TestTrainerShutdown(unittest.TestCase):
    def test_fit_closes_train_and_eval_owners_for_every_exit(self):
        for trainer_type in (RLColocateTrainer, RLDisaggregatedTrainer):
            for outcome in (None, ValueError("training failed"), asyncio.CancelledError()):
                for cleanup_fails in (False, True):
                    with self.subTest(trainer=trainer_type.__name__, outcome=outcome, cleanup_fails=cleanup_fails):
                        trainer = trainer_type.__new__(trainer_type)
                        failure = RuntimeError("owner cleanup failed")
                        trainer.agent_loop_manager = SimpleNamespace(
                            close=AsyncMock(side_effect=failure if cleanup_fails else None)
                        )
                        trainer.eval_agent_loop_manager = SimpleNamespace(close=AsyncMock())
                        trainer.rl_health_manager = MagicMock()
                        trainer._exp_tracker = MagicMock()
                        fit_mock = AsyncMock if trainer_type is RLDisaggregatedTrainer else MagicMock
                        trainer._fit = fit_mock(side_effect=outcome)
                        with patch("xtuner.v1.train.rl_trainer.close_trace") as close_trace:
                            if outcome is not None:
                                with self.assertRaises(type(outcome)) as caught:
                                    trainer.fit()
                                self.assertIs(caught.exception, outcome)
                                if cleanup_fails:
                                    self.assertEqual(caught.exception.__cause__.errors, (failure,))
                            elif cleanup_fails:
                                with self.assertRaises(CleanupError) as caught:
                                    trainer.fit()
                                self.assertEqual(caught.exception.errors, (failure,))
                            else:
                                trainer.fit()
                            close_trace.assert_called_once()
                        trainer.agent_loop_manager.close.assert_awaited_once()
                        trainer.eval_agent_loop_manager.close.assert_awaited_once()
                        trainer._exp_tracker.close.assert_called_once()
                        if trainer_type is RLColocateTrainer:
                            trainer.rl_health_manager.stop.assert_called_once()

    def test_fit_handles_absent_eval_manager_and_shared_manager(self):
        for shared in (False, True):
            trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
            trainer._fit = AsyncMock()
            trainer._exp_tracker = MagicMock()
            trainer.agent_loop_manager = SimpleNamespace(close=AsyncMock())
            trainer.eval_agent_loop_manager = trainer.agent_loop_manager if shared else None
            with patch("xtuner.v1.train.rl_trainer.close_trace"):
                trainer.fit()
            trainer.agent_loop_manager.close.assert_awaited_once()

    def test_later_cleanup_failure_retains_existing_error_cause(self):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        training_error = ValueError("training failed")
        producer_error = RuntimeError("producer cleanup failed")
        owner_error = RuntimeError("owner cleanup failed")
        training_error.__cause__ = producer_error
        trainer._fit = AsyncMock(side_effect=training_error)
        trainer._exp_tracker = MagicMock()
        trainer.agent_loop_manager = SimpleNamespace(close=AsyncMock(side_effect=owner_error))
        trainer.eval_agent_loop_manager = None
        with patch("xtuner.v1.train.rl_trainer.close_trace"), self.assertRaises(ValueError) as caught:
            trainer.fit()
        self.assertIs(caught.exception, training_error)
        causes = caught.exception.__cause__.errors
        self.assertIs(causes[0], producer_error)
        self.assertEqual(causes[1].errors, (owner_error,))
