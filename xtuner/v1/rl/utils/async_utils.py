import asyncio
import os
from asyncio import AbstractEventLoop, Task
from collections.abc import Awaitable, Iterable
from typing import Any, Callable, Coroutine, List, Optional, TypeVar, cast

import ray
from ray.exceptions import TaskCancelledError

from xtuner.v1.utils.logger import get_logger


T = TypeVar("T")


_ASYNCIO_RUN_LOOP: AbstractEventLoop | None = None
logger = get_logger()


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


async def cancel_and_drain(tasks: list[Task | None], *, timeout: float = 5.0) -> None:
    """Cancel tasks and consume their terminal exceptions.

    Use this when a parent coroutine handles cancellation itself but owns child tasks that must not keep running in the
    background.
    """
    tasks_to_drain = [task for task in tasks if task is not None]
    if not tasks_to_drain:
        return
    for task in tasks_to_drain:
        if task.done():
            continue
        task.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks_to_drain, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        return


async def await_owned_operation(operation: Awaitable[Any]) -> Any:
    """Await an operation this caller owns, including after cancellation.

    Cancelling the waiter must not hand the in-flight work to a loop that is about to close: a thread doing a
    training step or a checkpoint keeps using objects owned by this loop. On cancellation this waits for the
    operation to actually finish, reports its failure, then propagates ``CancelledError``.

    Args:
        operation (Awaitable[Any]): The owned operation, usually ``asyncio.to_thread``, an async cleanup, or a
            gathered Future.

    Returns:
        Any: The operation result when it completed without cancellation.
    """
    task = asyncio.ensure_future(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        try:
            task.result()
        except Exception:
            logger.exception("Owned operation failed while cancellation was pending")
        raise asyncio.CancelledError()
    return task.result()


async def await_cancellable_tasks(operations: Iterable[Awaitable[T]]) -> list[T]:
    """并发等待本次拥有的执行；收到取消后，等它们清理完再向外传播取消。

    为什么需要：普通 gather 可能在第一个子任务取消后提前结束，剩余任务还在 finally
    中；取消等待 Ray ObjectRef 的本地 Task，也不会自动停止远端 Agent。这个函数统一
    持有原执行，发送一次本地或远端取消，收齐结果，并延后传播调用方的取消。

    最小示例：两个子任务的清理一快一慢，父任务须等两个 finally 都结束。
        async def example():
            events = []
            started = [asyncio.Event(), asyncio.Event()]

            async def child(index, cleanup_delay):
                started[index].set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(cleanup_delay)
                    events.append(f"cleaned-{index}")

            owner = asyncio.create_task(await_cancellable_tasks([
                child(0, 0), child(1, 0.1)
            ]))
            await asyncio.gather(*(event.wait() for event in started))
            owner.cancel()
            try:
                await owner
            except asyncio.CancelledError:
                events.append("owner-cancelled")
            # 两个 cleaned 都先出现，owner-cancelled 最后出现。
            return events

    F1 的真实调用链：
        _PendingTasks.close 取消 BaseProduceContext.generate_group 的组级 Task，
        取消传到本地 AgentInSandboxLoop.generate_group。该方法改为：
            sample_groups = await await_cancellable_tasks(
                generate_one(state) for state in rollout_state
            )
        快 sample 的 Runner.pool.release_all 结束后，函数仍等待慢 sample 删除沙箱、
        关闭 client 和可选 route 退休；全部完成后才让组级 Task 结束，避免 asyncio.run
        退出时再次取消慢 sample 的 finally。

    F2 的真实调用链：
        BaseProduceContext 或 RouterAgentLoop 提交到实际 Ray worker 后，改为：
            ref = worker.generate_group.remote(rollout_state)
            (result,) = await await_cancellable_tasks([ref])
        收到取消时对原 ref 调用 ray.cancel，而不是只取消本地 waiter；随后继续等待
        原 ref 的终态。actor 内也须采用上述 F1 修复，才能由此确认全部 sample 清理已
        结束。RouterAgentLoop 随后才释放 worker 占用，manager.close 才能结束。

    本函数不撤销已完成工作，也不设清理总超时。普通异常在所有执行结束后再抛出；
    取消期间的意外异常会记录，最终仍传播调用方的 CancelledError。

    Args:
        operations (Iterable[Awaitable[T]]): 本次调用独占的本地执行或原 Ray ObjectRefs；
            不与其他 owner 共同管理取消，也不以会提前返回的 wrapper 代替真实执行。

    Returns:
        list[T]: 按输入顺序排列的结果；返回时所有输入执行均已结束。
    """
    operations = list(operations)  # Keep the original ObjectRefs; a local waiter cannot cancel remote work.
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    settled = asyncio.gather(*tasks, return_exceptions=True)
    cancelled = False
    cancel_errors: list[BaseException] = []

    try:
        await asyncio.shield(settled)
    except asyncio.CancelledError:
        cancelled = True
        for operation, task in zip(operations, tasks):
            if task.done():
                continue
            try:
                if isinstance(operation, ray.ObjectRef):
                    # Cancel the remote execution once; keep the waiter to observe its terminal state.
                    ray.cancel(operation, force=False, recursive=False)
                else:
                    task.cancel()
            except Exception as error:
                cancel_errors.append(error)  # Keep waiting for every other execution.
        try:
            await await_owned_operation(settled)
        except asyncio.CancelledError:
            pass  # A second cancellation cannot end the wait before ``settled`` really completes.

    results = settled.result()
    errors = [result for result in results if isinstance(result, BaseException)]
    if cancelled:
        unexpected = cancel_errors + [
            failure for failure in errors if not isinstance(failure, (asyncio.CancelledError, TaskCancelledError))
        ]
        for failure in unexpected:
            logger.error(
                "Owned execution failed while cancellation was pending",
                exc_info=(type(failure), failure, failure.__traceback__),
            )
        raise asyncio.CancelledError()
    if errors:
        raise errors[0]  # Ordinary errors also wait for every sibling to finish first.
    return cast(list[T], results)


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

    watchdog = None
    try:
        from .asyncio_diagnostics import install_asyncio_diagnostics, start_asyncio_run_watchdog

        if install_asyncio_diagnostics(loop):
            watchdog = start_asyncio_run_watchdog(loop=loop)
    except Exception as exc:
        if os.getenv("XTUNER_ASYNCIO_DIAGNOSTICS"):
            logger.warning(f"[AsyncioDiag] failed to start asyncio diagnostics: {type(exc).__name__}: {exc}")
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            from .asyncio_diagnostics import stop_asyncio_run_watchdog

            stop_asyncio_run_watchdog(watchdog)
        except Exception as exc:
            if os.getenv("XTUNER_ASYNCIO_DIAGNOSTICS"):
                logger.warning(f"[AsyncioDiag] failed to stop asyncio diagnostics: {type(exc).__name__}: {exc}")
