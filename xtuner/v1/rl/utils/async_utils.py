import asyncio
from asyncio import AbstractEventLoop, Task
from typing import Any, Callable, Coroutine, List, Optional


_ASYNCIO_RUN_LOOP: AbstractEventLoop | None = None


def handle_task_exception(task: Task):
    """Handles exceptions from an asyncio Task.

    This function checks if a task has raised an exception and, if so,
    re-raises it. It ignores `asyncio.CancelledError`.

    Args:
        task (Task): The asyncio task to check for exceptions.

    Raises:
        Exception: The exception raised by the task.
    """
    try:
        exc = task.exception()
        if exc is not None:
            raise exc
    except asyncio.CancelledError:
        pass  # Task was cancelled, ignore


def create_task(
    coro: Coroutine,
    loop: Optional[AbstractEventLoop] = None,
    done_callbacks: Optional[List[Callable[[Task], object]]] = None,
) -> asyncio.tasks.Task:
    """Creates and configures an asyncio Task.

    This function creates a task from a coroutine and attaches specified
    done callbacks. By default, it includes a callback to handle exceptions.

    Args:
        coro (Coroutine): The coroutine to wrap in a task.
        loop (Optional[AbstractEventLoop], optional): The event loop to run
            the task in. If None, the current event loop is used.
            Defaults to None.
        done_callbacks (Optional[List[Callable[[Task], object]]], optional):
            A list of callbacks to add to the task. If None, a default
            exception handler is used. Defaults to None.

    Returns:
        asyncio.tasks.Task: The created asyncio task.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    if done_callbacks is None:
        done_callbacks = [handle_task_exception]
    task = loop.create_task(coro)
    for callback in done_callbacks:
        task.add_done_callback(callback)
    return task


def _get_default_asyncio_loop() -> AbstractEventLoop:
    """Get a module-level event loop reused by ``asyncio_run``."""
    global _ASYNCIO_RUN_LOOP
    if _ASYNCIO_RUN_LOOP is not None and not _ASYNCIO_RUN_LOOP.is_closed():
        return _ASYNCIO_RUN_LOOP

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    _ASYNCIO_RUN_LOOP = loop
    return _ASYNCIO_RUN_LOOP


def asyncio_run(coro: Coroutine, loop: Optional[AbstractEventLoop] = None) -> Any:
    """Synchronously run a coroutine on a shared/explicit event loop.

    This helper is used by `RLColocateTrainer.fit` for rollout collection:
    1) Trainer runs in sync code and repeatedly calls:
       - self.eval_agent_loop_manager.produce_batch(...)
       - self.agent_loop_manager.produce_batch(...)
    2) `produce_batch` is async, and internally runs `ProduceStrategy.produce_batch`,
       which launches many nested async tasks (`create_task`) and ultimately calls
       `AgentLoop.generate_group -> generate_sample`.
    3) In `VerlToolAgentLoop`, `generate_sample` awaits `self.verl_tool_agent_loop.run()`,
       where the tool loop stays on the same loop.

    In this pattern, if sync code uses `asyncio.run` every call, each invocation
    creates/closes a fresh loop, but `VerlToolAgentLoop` keeps internal work
    on one loop, the wrapped `generate_sample -> run -> Ray futures` chain can see
    mismatched loop ownership and trigger:
    ``Future attached to a different loop``.

    `asyncio_run` keeps calls bound to a stable loop instance so nested task/future
    chains stay compatible across repeated rollout phases.

    This helper is for sync-to-async boundaries only and should not be used from
    within an already running event loop.
    """
    if loop is None:
        loop = _get_default_asyncio_loop()
    if loop.is_running():
        raise RuntimeError("asyncio_run does not support being called from a running event loop.")
    return loop.run_until_complete(coro)
