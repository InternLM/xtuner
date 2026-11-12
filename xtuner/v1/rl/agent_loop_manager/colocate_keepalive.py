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

        async def cleanup():
            errors = []
            for task in self._inner.task_runners:
                task.produce_strategy._pending_tasks.close_admission()
            finalizing = getattr(self._inner, "_finalizing_tasks", set())
            cancellations = getattr(self, "_cleanup_waiters", None)
            if cancellations is None:
                cancellations = [asyncio.create_task(task.produce_strategy._pending_tasks.cancel_all(),
                                                    name=f"cleanup:{task.task_name}")
                                 for task in self._inner.task_runners]
                self._cleanup_waiters = cancellations
            owned = set(finalizing) | set(cancellations)
            pending = set()
            if owned:
                done, pending = await asyncio.wait(owned, timeout=120)
                errors.extend(f"{t.get_name()}: {t.exception()!r}" for t in done
                              if not t.cancelled() and t.exception() is not None)
            if pending:
                # Do not stop this loop or cancel an already-running finally a
                # second time: platform cleanup must keep its original owner.
                return errors, [t.get_name() for t in pending]
            try:
                await self._inner.release_final_leftovers()
            except Exception as exc:
                errors.append(f"final trace cleanup: {exc!r}")
            remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in remaining:
                task.cancel()
            if remaining:
                done, pending = await asyncio.wait(remaining, timeout=30)
                errors.extend(f"{t.get_name()}: {t.exception()!r}" for t in done
                              if not t.cancelled() and t.exception() is not None)
            return errors, [t.get_name() for t in pending]

        errors, pending = await self._owner.call(cleanup())
        if pending:
            raise RuntimeError(f"Agent cleanup still owned by live loop: pending={pending} errors={errors}")
        self._owner.stop()
        self._closed = True
        if errors:
            raise RuntimeError(f"Agent cleanup failed: {errors}")


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
        inner._update_policy = config.turn_keepalive_update_policy
        inner._checkpoint_mode = config.turn_keepalive_checkpoint_mode
        if inner._update_policy == "batch_ready" and inner._checkpoint_mode != "model_export":
            raise ValueError("Batch-ready keepalive requires explicit model_export checkpoint mode")
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
