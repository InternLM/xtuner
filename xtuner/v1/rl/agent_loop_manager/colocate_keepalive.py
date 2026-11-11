"""Opt-in persistent asyncio owner for alternating colocated RL.

The native asyncio_run loop is stable but stops during synchronous training.
This adapter keeps tool/client tasks progressing without enabling GPU inference.
Only the strategy's existing _PendingTasks owns live generation tasks.
"""

import asyncio
import threading


class LoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="xt-colocate-agent-loop", daemon=True)
        self.thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()
        self.loop.close()

    def submit(self, coro):
        if not self.thread.is_alive():
            coro.close()
            raise RuntimeError("Colocate Agent loop is closed")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def call(self, coro):
        return await asyncio.wrap_future(self.submit(coro))

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("Colocate Agent loop thread did not stop")


class ColocateKeepaliveManager:
    def __init__(self, inner, owner):
        self._inner, self._owner = inner, owner
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def produce_batch(self, *args, **kwargs):
        return await self._owner.call(self._inner.produce_batch(*args, **kwargs))

    async def prepare_initial_sync(self):
        # Servers were born closed, before their HTTP endpoint was published.
        # Build the ordinary fixed-topology barrier; no group is scheduled.
        return await self._owner.call(self._inner._keepalive.pause(self._inner._rollout_controller))

    async def publish_initial_version(self, model_step):
        return await self._owner.call(self._inner._keepalive.resume(self._inner._rollout_controller, model_step))

    async def save(self, *args, **kwargs):
        return await self._owner.call(self._inner.save(*args, **kwargs))

    async def resume(self, *args, **kwargs):
        return await self._owner.call(self._inner.resume(*args, **kwargs))

    async def close(self):
        if self._closed:
            return
        self._closed = True

        async def cleanup():
            cancellations = [
                asyncio.create_task(task.produce_strategy._pending_tasks.cancel_all())
                for task in self._inner.task_runners
            ]
            if cancellations:
                _, pending = await asyncio.wait(cancellations, timeout=120)
                if pending:
                    raise TimeoutError("Pending Agent cleanup exceeded 120 seconds")
            # Local AgentInSandboxLoop cancellation reaches original Runner
            # finally/pool cleanup. No duplicate sandbox owner is introduced.
            remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in remaining:
                task.cancel()
            if remaining:
                _, pending = await asyncio.wait(remaining, timeout=30)
                if pending:
                    raise TimeoutError("Agent client teardown exceeded 30 seconds")

        try:
            await self._owner.call(cleanup())
        finally:
            self._owner.stop()


def build_keepalive_manager(config, **kwargs):
    from .agent_turn_pause_policy import AgentTurnPausePolicy
    from .producer import AsyncProduceStrategyConfig

    tasks = config.tasks if isinstance(config.tasks, list) else [config.tasks]
    for task in tasks:
        strategy = task.produce_strategy_config
        if not isinstance(strategy, AsyncProduceStrategyConfig) or strategy.enable_partial_rollout:
            raise ValueError("Colocate turn keepalive requires AsyncProduceStrategy and token partial disabled")
        if not getattr(task.agent_loop_config, "require_turn_provenance", False):
            raise ValueError("Colocate turn keepalive requires real request provenance")
    owner = LoopThread()

    async def build():
        # Build clients, locks, sampler and tasks on the one persistent loop.
        inner = config.model_copy(update={"turn_keepalive": False}).build(**kwargs)
        inner._keepalive = AgentTurnPausePolicy(
            freeze_timeout=config.turn_keepalive_freeze_timeout,
            drain_timeout=config.turn_keepalive_drain_timeout,
            quiesce_timeout=config.turn_keepalive_quiesce_timeout,
        )
        for task in inner.task_runners:
            task.produce_strategy.retain_live_tasks = True
        return inner

    try:
        inner = owner.submit(build()).result()
    except BaseException:
        owner.stop()
        raise
    return ColocateKeepaliveManager(inner, owner)
