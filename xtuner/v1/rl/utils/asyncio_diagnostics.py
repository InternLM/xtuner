"""Asyncio diagnostics for Python 3.12 training hangs.

用法：
- 设置 `XTUNER_ASYNCIO_DIAGNOSTICS=1` 启用诊断。
- 看到 `[AsyncioDiag] ready ... SIGUSR1=ready` 日志后，再执行 `kill -USR1 <pid>`：
  把当前训练 event loop 上的 asyncio task、await chain、`task.get_stack()` 写入 dump 文件。
- `kill -USR2 <pid>`：用 `faulthandler` 把所有 Python 线程栈写入 dump 文件。
- dump 文件默认写到 `$WORK_DIR/asyncio_diagnostics_<timestamp>`；也可以用
  `XTUNER_ASYNCIO_DIAG_DIR=/path/to/dir` 覆盖。
- `XTUNER_ASYNCIO_RUN_WATCHDOG_S=600` 表示单次 `asyncio_run()` 超过 600 秒
  未返回时自动 dump 一次。
- `XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S=600` 表示这次 `asyncio_run()` 仍未返回时，
  每 600 秒继续自动 dump 一次。
- `XTUNER_ASYNCIO_DIAG_TASK_LIMIT=100`、`XTUNER_ASYNCIO_DIAG_STACK_LIMIT=8`、
  `XTUNER_ASYNCIO_DIAG_AWAIT_DEPTH=12` 控制单次 asyncio task dump 的详细程度。

检测原理：
- 训练代码通过 `xtuner.v1.rl.utils.asyncio_run()` 进入
  `loop.run_until_complete(coro)`。本模块在 `asyncio_run()` 拿到实际训练 loop 后
  安装 signal handler 和 watchdog，因此 `USR1` 看到的是训练正在使用的那个 loop。
- `USR1` 运行在 event loop 线程里，通过 `asyncio.all_tasks(loop)` 找到所有 task，
  再沿 coroutine 的 `cr_await` / `gi_yieldfrom` / `ag_await` 打印逻辑 await 链。
  这能补足 `pystack` 只能看到 `run_until_complete` / `_run_once` / epoll 的情况。
- watchdog 不是业务级 stall detector。它只判断“这一次 `asyncio_run()` 是否超过阈值还没返回”，
  不知道 producer 是否真的没有业务进展。

进程/线程设计：
- 诊断是按进程安装的。主进程和 Ray/torch 等子进程都有独立的 signal handler；
  只有实际调用 `asyncio_run()` 的进程才会尝试安装诊断。
- 安装状态记录为 `(pid, loop_id)`。fork 后 pid 会变化，event loop 重建后 loop_id 会变化，
  因此会重新安装 handler，让 `USR1` 指向当前进程正在使用的 loop。
- Python signal handler 只能在主线程安装。如果 `asyncio_run()` 首次发生在非主线程，
  `SIGUSR1` / `SIGUSR2` handler 可能安装失败；此时会输出 warning，不记录 installed，
  也不会启动 watchdog，后续主线程调用时仍可重试安装。
- 在 ready 日志出现前不要发送 `SIGUSR1`。Linux 对未处理的 `SIGUSR1` 的默认动作是终止进程。
- watchdog 依赖诊断安装成功后才启动。这样避免出现 signal 入口不可用但后台仍周期 dump 的混乱状态。

限制：
- 如果 event loop 仍在正常调度但某些 task 一直 pending，`USR1` 和 watchdog 都能 dump task 状态。
- 如果 event loop 被同步代码、C 扩展或长时间 CPU 计算占住，event loop timer 和 `USR1`
  callback 都可能无法执行；这时用 `USR2` 查看所有 Python 线程栈。
- 本模块只负责诊断输出，不改变 task 调度、取消或业务状态。
"""

import asyncio
import faulthandler
import inspect
import os
import signal
import sys
import time
from asyncio import AbstractEventLoop
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any, TypeAlias

from xtuner.v1.utils.logger import get_logger


logger = get_logger()
_INSTALLED_FOR: tuple[int, int] | None = None
_STARTED_AT = time.perf_counter()
_DEFAULT_DIAG_DIR_FOR: tuple[int, str] | None = None
WatchdogHandle: TypeAlias = dict[str, Any]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(f"[AsyncioDiag] ignoring invalid {name}={value!r}; expected int.")
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(f"[AsyncioDiag] ignoring invalid {name}={value!r}; expected float.")
        return default
    return parsed if parsed > 0 else default


def _diagnostics_dir() -> Path:
    dirname = os.getenv("XTUNER_ASYNCIO_DIAG_DIR")
    if dirname:
        return Path(dirname)

    global _DEFAULT_DIAG_DIR_FOR
    pid = os.getpid()
    if _DEFAULT_DIAG_DIR_FOR is None or _DEFAULT_DIAG_DIR_FOR[0] != pid:
        timestamp = time.strftime("%Y%m%d%H%M%S")
        _DEFAULT_DIAG_DIR_FOR = (pid, f"asyncio_diagnostics_{timestamp}")
    dirname = _DEFAULT_DIAG_DIR_FOR[1]

    work_dir = os.getenv("WORK_DIR")
    if work_dir:
        return Path(work_dir) / dirname

    return Path.cwd() / dirname


def _new_dump_path(kind: str) -> Path:
    now = time.strftime("%Y%m%d_%H%M%S")
    ns = time.time_ns() % 1_000_000_000
    return _diagnostics_dir() / f"{kind}_{now}_{ns:09d}_pid{os.getpid()}.txt"


def _write_lines(kind: str, lines: list[str]) -> Path | None:
    path = _new_dump_path(kind)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except Exception as exc:
        logger.warning(f"[AsyncioDiag] failed to write dump file {path}: {type(exc).__name__}: {exc}")
        for line in lines:
            logger.warning(line)
        return None


def _format_frame(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"


def _awaitable_name(obj: Any) -> str:
    if inspect.iscoroutine(obj):
        return obj.cr_code.co_name
    if inspect.isgenerator(obj):
        return obj.gi_code.co_name
    if inspect.isasyncgen(obj):
        return obj.ag_code.co_name
    return type(obj).__name__


def _await_chain(obj: Any, *, max_depth: int, depth: int = 0, seen: set[int] | None = None) -> list[str]:
    if obj is None:
        return []
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return [f"{'  ' * depth}<cycle: {obj!r}>"]
    if depth >= max_depth:
        return [f"{'  ' * depth}..."]

    seen.add(id(obj))
    indent = "  " * depth
    lines: list[str] = []

    if isinstance(obj, asyncio.Task):
        lines.append(f"{indent}Task name={obj.get_name()!r} state={getattr(obj, '_state', 'unknown')}")
        lines.extend(_await_chain(obj.get_coro(), max_depth=max_depth, depth=depth + 1, seen=seen))
        waiter = getattr(obj, "_fut_waiter", None)
        if waiter is not None:
            lines.append(f"{indent}  waiting_on={waiter!r}")
        return lines

    frame = getattr(obj, "cr_frame", None) or getattr(obj, "gi_frame", None) or getattr(obj, "ag_frame", None)
    if frame is None:
        lines.append(f"{indent}{_awaitable_name(obj)} {obj!r}")
    else:
        lines.append(f"{indent}{_awaitable_name(obj)} at {_format_frame(frame)}")

    next_obj = getattr(obj, "cr_await", None) or getattr(obj, "gi_yieldfrom", None) or getattr(obj, "ag_await", None)
    if next_obj is not None:
        lines.extend(_await_chain(next_obj, max_depth=max_depth, depth=depth + 1, seen=seen))
    return lines


def _iter_tasks(loop: AbstractEventLoop | None) -> Iterable[asyncio.Task]:
    if loop is not None:
        return asyncio.all_tasks(loop)
    try:
        return asyncio.all_tasks()
    except RuntimeError:
        return ()


def dump_asyncio_tasks(
    *,
    loop: AbstractEventLoop | None = None,
    reason: str = "manual",
    task_limit: int | None = None,
    stack_limit: int | None = None,
    await_depth: int | None = None,
) -> None:
    task_limit = task_limit or _env_int("XTUNER_ASYNCIO_DIAG_TASK_LIMIT", 100)
    stack_limit = stack_limit or _env_int("XTUNER_ASYNCIO_DIAG_STACK_LIMIT", 8)
    await_depth = await_depth or _env_int("XTUNER_ASYNCIO_DIAG_AWAIT_DEPTH", 12)

    tasks = sorted(_iter_tasks(loop), key=lambda task: task.get_name())
    loop_id = hex(id(loop)) if loop is not None else "current"
    lines = [
        f"[AsyncioDiag] dump reason={reason} pid={os.getpid()} loop_id={loop_id} "
        f"uptime={time.perf_counter() - _STARTED_AT:.1f}s task_count={len(tasks)}"
    ]
    for task in tasks[:task_limit]:
        lines.append("")
        lines.append(f"[AsyncioDiag] task={task!r}")
        for line in _await_chain(task, max_depth=await_depth):
            lines.append(f"[AsyncioDiag] await_chain {line}")

        stack = task.get_stack(limit=stack_limit)
        if not stack:
            lines.append("[AsyncioDiag] stack <empty>")
            continue
        for frame in stack:
            lines.append(f"[AsyncioDiag] stack {_format_frame(frame)}")
    if len(tasks) > task_limit:
        lines.append(f"[AsyncioDiag] skipped {len(tasks) - task_limit} tasks due to task_limit={task_limit}")

    path = _write_lines("asyncio_tasks", lines)
    if path is not None:
        logger.warning(f"[AsyncioDiag] wrote asyncio task dump reason={reason} task_count={len(tasks)} path={path}")


def start_asyncio_run_watchdog(
    *,
    loop: AbstractEventLoop,
    label: str = "asyncio_run",
    threshold_s: float | None = None,
    repeat_s: float | None = None,
) -> WatchdogHandle | None:
    if not _env_flag("XTUNER_ASYNCIO_DIAGNOSTICS"):
        return None

    threshold_s = threshold_s or _env_float("XTUNER_ASYNCIO_RUN_WATCHDOG_S", 600.0)
    repeat_s = repeat_s or _env_float("XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S", threshold_s)
    started_at = time.perf_counter()
    state: WatchdogHandle = {"count": 0, "handle": None}

    def _dump_and_reschedule() -> None:
        state["count"] += 1
        elapsed_s = time.perf_counter() - started_at
        dump_asyncio_tasks(
            loop=loop,
            reason=f"{label}:watchdog:{elapsed_s:.1f}s:#{state['count']}",
        )
        state["handle"] = loop.call_later(repeat_s, _dump_and_reschedule)

    state["handle"] = loop.call_later(threshold_s, _dump_and_reschedule)
    return state


def stop_asyncio_run_watchdog(watchdog: WatchdogHandle | None) -> None:
    if watchdog is None:
        return
    handle = watchdog.get("handle")
    if handle is not None:
        handle.cancel()


def _dump_all_thread_stacks(signum: int, frame: FrameType | None) -> None:
    path = _new_dump_path("thread_stacks")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write(
                f"[AsyncioDiag] dump all Python thread stacks signal={signum} "
                f"pid={os.getpid()} uptime={time.perf_counter() - _STARTED_AT:.1f}s\n\n"
            )
            file.flush()
            faulthandler.dump_traceback(file=file, all_threads=True)
        logger.warning(f"[AsyncioDiag] wrote Python thread stack dump signal={signum} path={path}")
    except Exception as exc:
        logger.warning(
            f"[AsyncioDiag] failed to write thread stack dump file {path}: "
            f"{type(exc).__name__}: {exc}; falling back to stderr"
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def install_asyncio_diagnostics(loop: AbstractEventLoop) -> bool:
    """Install low-intrusion asyncio diagnostics for Python 3.12.

    Environment knobs:
    - XTUNER_ASYNCIO_DIAGNOSTICS=1 enables installation.
    - XTUNER_ASYNCIO_DIAG_DIR overrides dump output directory.
    - XTUNER_ASYNCIO_RUN_WATCHDOG_S controls automatic asyncio_run dumps.
    - XTUNER_ASYNCIO_RUN_WATCHDOG_REPEAT_S controls repeated automatic dumps.
    - XTUNER_ASYNCIO_DIAG_TASK_LIMIT / STACK_LIMIT / AWAIT_DEPTH control dump size.
    - kill -USR1 <pid> dumps asyncio tasks for the shared event loop.
    - kill -USR2 <pid> dumps all Python thread stacks via faulthandler.
    """
    global _INSTALLED_FOR
    if not _env_flag("XTUNER_ASYNCIO_DIAGNOSTICS"):
        return False
    install_key = (os.getpid(), id(loop))
    if _INSTALLED_FOR == install_key:
        return True

    faulthandler.enable(file=sys.stderr, all_threads=True)

    sigusr1_installed = True
    try:
        loop.add_signal_handler(signal.SIGUSR1, partial(dump_asyncio_tasks, loop=loop, reason="SIGUSR1"))
    except (NotImplementedError, RuntimeError) as exc:
        sigusr1_installed = False
        logger.warning(f"[AsyncioDiag] failed to install SIGUSR1 handler: {type(exc).__name__}: {exc}")

    sigusr2_installed = True
    try:
        signal.signal(signal.SIGUSR2, _dump_all_thread_stacks)
    except (ValueError, OSError) as exc:
        sigusr2_installed = False
        logger.warning(f"[AsyncioDiag] failed to install SIGUSR2 handler: {type(exc).__name__}: {exc}")
    if not sigusr1_installed and not sigusr2_installed:
        logger.warning("[AsyncioDiag] no signal handlers installed; asyncio watchdog disabled.")
        return False

    _INSTALLED_FOR = install_key
    logger.warning(
        f"[AsyncioDiag] ready pid={os.getpid()} "
        f"loop_id={hex(id(loop))} "
        f"dump_dir={_diagnostics_dir()} "
        f"SIGUSR1={'ready: kill -USR1 dumps asyncio tasks' if sigusr1_installed else 'unavailable'} "
        f"SIGUSR2={'ready: kill -USR2 dumps all Python thread stacks' if sigusr2_installed else 'unavailable'}"
    )
    return True
