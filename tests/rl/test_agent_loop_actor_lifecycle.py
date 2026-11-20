"""Control RPCs drain generation on its owning loop even when its Ray group is full."""

import asyncio
import threading
from typing import Any

import ray

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop.agent_loop import AGENT_LOOP_CONCURRENCY_GROUP_GENERATE, AgentLoopActor


class TestAgentLoopActorLifecycle:
    def test_close_drains_saturated_generation_on_owner_loop(self) -> None:
        # Local classes are serialized by value: workers need no test-module imports.
        class Loop:
            def __init__(self, started: threading.Event) -> None:
                self.construct_loop = asyncio.get_event_loop()
                self.started = started
                self.cleaned = False

            def _set_teacher_scorer(self, scorer: Any) -> None:
                pass

            async def generate_sample(self, state: RolloutState, **kwargs: Any) -> RolloutState:
                self.generate_loop = asyncio.get_running_loop()
                self.pending = asyncio.create_task(self._wait())
                try:
                    await self.pending
                except asyncio.CancelledError:
                    # A drained generation is allowed to return its interrupted state.
                    pass
                return state

            async def _wait(self) -> None:
                try:
                    self.started.set()
                    await asyncio.Event().wait()
                finally:
                    self.cleaned = True

            async def close(self) -> None:
                self.close_loop = asyncio.get_running_loop()
                self.pending.cancel()
                await asyncio.gather(self.pending, return_exceptions=True)

        class Config:
            def __init__(self, started: threading.Event) -> None:
                self.started = started

            def build_local(self, **kwargs: Any) -> Loop:
                return Loop(self.started)

        class Actor(AgentLoopActor):
            def __init__(self) -> None:
                self.started = threading.Event()
                super().__init__(Config(self.started), None)

            async def wait_started(self) -> bool:
                return await asyncio.to_thread(self.started.wait, 20)

            async def snapshot(self) -> tuple[bool, bool]:
                loop = self.agent_loop
                return loop.cleaned, loop.construct_loop is loop.generate_loop is loop.close_loop

        ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2)
        actor = None
        try:
            actor = ray.remote(
                num_cpus=0,
                concurrency_groups={AGENT_LOOP_CONCURRENCY_GROUP_GENERATE: 1},
            )(Actor).remote()
            generation = actor.generate_sample.remote(RolloutState(message=[]))
            assert ray.get(actor.wait_started.remote(), timeout=30)
            # The sole generate slot stays occupied until close drains the pending task.
            ray.get(actor.close.remote(), timeout=30)
            ray.get(generation, timeout=30)
            assert ray.get(actor.snapshot.remote(), timeout=30) == (True, True)
        finally:
            if actor is not None:
                ray.kill(actor)
            ray.shutdown()
