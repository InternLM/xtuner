"""Execute a complete sandbox Runner in a CPU Ray task.

The parent retains trace assembly, model services and route retirement. Workers
must have the pipeline's imports, environment and task paths through the Ray job
runtime environment; live Runner/client instances cannot cross this interface.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Callable, Literal

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .schemas import AgentRolloutItem


async def _run_local(item: AgentRolloutItem) -> AgentRolloutItem:
    from .agent_in_sandbox_loop import _resolve_runner

    runner = _resolve_runner(item.pipeline, str(item.uid))
    if runner is None:
        raise ValueError("AgentRolloutItem.pipeline is required.")
    return await runner.run(item)


def _execute_item(item_data: dict) -> dict:
    # A synchronous Ray task occupies one worker while asyncio.run gives this
    # invocation its own event loop. Construct all stateful clients in that loop.
    item = AgentRolloutItem.model_validate(item_data)
    async def run():
        loop = asyncio.get_running_loop()
        wakeup = None

        def wake():
            nonlocal wakeup
            wakeup = loop.call_later(1, wake)

        # Ray interrupts this synchronous worker with KeyboardInterrupt. A loop
        # waiting indefinitely in select otherwise cannot service that signal.
        wake()
        try:
            return await _run_local(item)
        finally:
            wakeup.cancel()

    return asyncio.run(run()).model_dump(mode="python")


_remote_item = ray.remote(num_cpus=1, num_gpus=0, max_retries=0, retry_exceptions=False)(_execute_item)


async def _await_terminal(ref) -> dict:
    future = asyncio.ensure_future(ref)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        try:
            ray.cancel(ref, force=False, recursive=False)
        except Exception:
            logging.getLogger(__name__).exception("Trial cancel request failed; waiting for original task termination")
        # Keep the parent's route lease until Runner.finally has completed.
        # Repeated parent cancellation must not abandon the original worker.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not future.cancelled():
            # Retrieve the remote cancellation/error, preserving parent cancel.
            future.exception()
        raise


async def _run_ray(item: AgentRolloutItem, *, num_cpus: float) -> AgentRolloutItem:
    if not isinstance(item.pipeline, (str, dict)):
        raise TypeError("ray_task requires a pipeline import path or declarative config, not a live Runner.")
    ref = _remote_item.options(
        num_cpus=num_cpus,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=None, placement_group_capture_child_tasks=False
        ),
    ).remote(item.model_dump(mode="python"))
    return AgentRolloutItem.model_validate(await _await_terminal(ref))


def build_trial_executor(mode: Literal["local", "ray_task"], num_cpus: float = 1) -> Callable:
    """Select execution once; both adapters accept and return AgentRolloutItem.

    Ray CPU credits must be available outside the model placement group. The
    default local adapter also supports existing preconstructed Runner objects.
    Cooperative cancellation waits for cleanup without force-killing workers;
    abrupt worker/node failure cannot guarantee remote sandbox deletion.
    """
    if mode == "local":
        return _run_local
    if mode == "ray_task":
        if num_cpus <= 0:
            raise ValueError("trial_cpus must be positive")
        return partial(_run_ray, num_cpus=num_cpus)
    raise ValueError(f"Unknown Trial dispatch mode: {mode!r}")
