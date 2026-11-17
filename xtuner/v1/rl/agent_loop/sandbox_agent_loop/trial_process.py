"""Execute one trusted sandbox pipeline in a disposable CPU process.

Ray and the model proxy stay in the caller. The child owns the complete
infer/verify sequence; a stopped child can no longer create new resources.
Only its pre-allocation journal manifest is eligible for foreground recovery.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import importlib
import json
import logging
import math
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
from contextlib import ExitStack, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .schemas import AgentRolloutItem


_MANIFEST = "XTUNER_SANDBOX_TRIAL_MANIFEST"
_LATE_TRIALS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class TrialProcessLimits:
    """Wall-clock limits for execution, cooperative stop and exact-trial recovery.

    Args:
        execution (float): Maximum trial lifetime, including interpreter startup.
        stop (float): TERM grace before killing this trial's process group.
        recovery (float): Hard budget for foreground recovery and durable handoff.
    """

    execution: float = 14400
    stop: float = 120
    recovery: float = 30

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (self.execution, self.stop, self.recovery)):
            raise ValueError("trial process limits must be finite and positive")


@dataclass
class _Trial:
    directory: Path
    limits: TrialProcessLimits
    environment: dict[str, str]
    launch: asyncio.Task[asyncio.subprocess.Process]
    channel: asyncio.Task[None]
    parent_socket: socket.socket
    child_socket: socket.socket
    deadline: float
    process: asyncio.subprocess.Process | None = None
    exited: bool = False
    primary: BaseException | None = None
    receipt: dict[str, Any] = field(default_factory=dict)


async def run_trial_process(item: AgentRolloutItem, *, limits: TrialProcessLimits | None = None) -> AgentRolloutItem:
    """Run a pipeline, confirm process exit, then recover only its own journals.

    Args:
        item (AgentRolloutItem): Trusted task with a declarative pipeline or import path.
        limits (TrialProcessLimits | None): Explicit limits, or environment defaults.

    Returns:
        AgentRolloutItem: The child's result after exit and cleanup confirmation.
    """
    from .finalization import finalize_bounded

    trial = _start_trial(item, limits or _configured_limits())
    try:
        try:
            await _execute_trial(trial)
        except BaseException as error:
            trial.primary = error
        try:
            await finalize_bounded(
                _settle_trial(trial), timeout_seconds=trial.limits.stop + trial.limits.recovery + 10, diagnostic={}
            )
        except BaseException as error:
            trial.primary = trial.primary or error
        return _read_result(trial, item)
    finally:
        _dispose_trial(trial)


def _configured_limits() -> TrialProcessLimits:
    return TrialProcessLimits(
        execution=float(os.environ.get("XTUNER_SANDBOX_TRIAL_TIMEOUT", "14400")),
        stop=float(os.environ.get("XTUNER_SANDBOX_TRIAL_STOP_TIMEOUT", "120")),
        recovery=float(os.environ.get("XTUNER_SANDBOX_TRIAL_RECOVERY_TIMEOUT", "30")),
    )


def _start_trial(item: AgentRolloutItem, limits: TrialProcessLimits) -> _Trial:
    import cloudpickle

    from .trial_channel import serve_channel

    _validate_pipeline(item.pipeline)
    _validate_configuration(item.pipeline_overrides)
    # Serialize before spawning: unportable live objects must fail before allocation.
    payload = cloudpickle.dumps(item)
    with ExitStack() as rollback:
        directory = Path(tempfile.mkdtemp(prefix="xtuner-trial-"))
        rollback.callback(shutil.rmtree, directory, ignore_errors=True)
        request = directory / "request.pkl"
        request.write_bytes(payload)
        request.chmod(0o600)
        (directory / "journals").mkdir(mode=0o700)
        parent_socket, child_socket = socket.socketpair()
        rollback.callback(parent_socket.close)
        rollback.callback(child_socket.close)
        parent_socket.setblocking(False)
        environment = _child_environment(directory / "journals")
        deadline = time.monotonic() + limits.execution
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(directory),
                "--parent-pid",
                str(os.getpid()),
                "--control-fd",
                str(child_socket.fileno()),
                env=environment,
                pass_fds=(child_socket.fileno(),),
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
            )
        )
        rollback.callback(launch.cancel)
        channel = asyncio.create_task(serve_channel(parent_socket))
        trial = _Trial(directory, limits, environment, launch, channel, parent_socket, child_socket, deadline)
        rollback.pop_all()
        return trial


def _validate_pipeline(pipeline: Any) -> None:
    if not isinstance(pipeline, (dict, str)):
        raise TypeError("isolated trials require a pipeline config or import path, never a live Runner")
    _validate_configuration(pipeline)


def _validate_configuration(value: Any) -> None:
    from .sandbox import SandboxPool

    if isinstance(value, SandboxPool):
        raise TypeError("isolated trials cannot transfer a live SandboxPool")
    if (
        not isinstance(value, type)
        and callable(getattr(value, "create", None))
        and callable(getattr(value, "delete", None))
    ):
        raise TypeError("isolated trials cannot transfer a live sandbox provider")
    if isinstance(value, dict):
        for child in value.values():
            _validate_configuration(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_configuration(child)


async def _execute_trial(trial: _Trial) -> None:
    from .task_control import watch_task

    async with watch_task(os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")):
        trial.process = await asyncio.wait_for(
            asyncio.shield(trial.launch), timeout=max(0.01, trial.deadline - time.monotonic())
        )
        trial.child_socket.close()
        await _wait_worker(trial.process, trial.directory / "result.pkl", trial.deadline, trial.limits.stop)


async def _settle_trial(trial: _Trial) -> None:
    try:
        trial.process = trial.process or await asyncio.shield(trial.launch)
        trial.child_socket.close()
        await _stop_worker(trial.process, trial.limits.stop)
        trial.exited = True
        try:
            trial.receipt = await _recover_after_exit(trial.directory, trial.environment, trial.limits.recovery)
        except Exception as error:
            # Recovery failure is secondary to an already encoded business
            # failure. The result policy below still rejects successful work.
            trial.receipt = {"ok": False, "error_type": type(error).__name__}
    finally:
        if not trial.exited:
            trial.launch.add_done_callback(lambda done: _kill_late_worker(done, trial=trial))
        trial.child_socket.close()
        trial.channel.cancel()
        await asyncio.gather(trial.channel, return_exceptions=True)
        trial.parent_socket.close()


def _read_result(trial: _Trial, item: AgentRolloutItem) -> AgentRolloutItem:
    import cloudpickle

    if trial.primary is not None:
        raise trial.primary
    assert trial.process is not None
    if trial.process.returncode != 0:
        raise RuntimeError(f"Sandbox trial process exited with code {trial.process.returncode}")
    response = trial.directory / "result.pkl"
    if not response.is_file():
        raise RuntimeError("Sandbox trial process exited without a result")
    packet = cloudpickle.loads(response.read_bytes())
    if packet.get("error_type"):
        raise RuntimeError(f"Sandbox trial failed ({packet['error_type']}); inspect the worker log")
    result = packet["item"]
    if result.uid != item.uid or result.id != item.id:
        raise RuntimeError("Sandbox trial result identity mismatch")
    if not trial.receipt.get("ok"):
        from .schemas import RolloutError, RolloutStatus

        result.status = RolloutStatus.FAILED
        result.error = result.error or RolloutError(
            stage="cleanup", category="trial_cleanup_incomplete", message="Retain the run journal and recovery queue"
        )
    result.artifacts["trial_process"] = {
        "pid": trial.process.pid,
        "exit_code": trial.process.returncode,
        "cleanup": trial.receipt,
    }
    return result


def _dispose_trial(trial: _Trial) -> None:
    failed_launch = trial.launch.done() and not trial.launch.cancelled() and trial.launch.exception() is not None
    if trial.exited or failed_launch:
        try:
            shutil.rmtree(trial.directory)
        except OSError as error:
            # Staging disposal must not replace cancellation/business failure.
            logging.getLogger(__name__).warning("Trial staging cleanup failed (%s)", type(error).__name__)
    # An unconfirmed OS exit keeps its local request/manifest and global journals.


def register_trial_journal(path: Path) -> None:
    """Persist a child's exact resource scope before any provider allocation.

    Args:
        path (Path): Newly persisted Pool journal in the configured run directory.
    """
    manifest = os.environ.get(_MANIFEST)
    if not manifest:
        return
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    if not directory or path.parent.resolve() != Path(directory).resolve():
        raise ValueError("isolated trials require journals in the driver's cleanup directory")
    destination = Path(manifest) / path.name
    temporary = destination.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(str(path.resolve()), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_trial_cleanup(directory: str | None) -> None:
    """Require durable cleanup before an isolated Pool can allocate resources.

    Args:
        directory (str | None): Pool's configured persistent journal directory.
    """
    if os.environ.get(_MANIFEST):
        run = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
        if (
            not directory
            or not run
            or not Path(run).is_absolute()
            or Path(directory).resolve() != Path(run).resolve()
            or not os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
        ):
            raise ValueError("isolated sandbox tasks require a persistent run directory and cleanup factory")


def resolve_runner(pipeline: Any, session_id: str) -> Any:
    """Construct the trusted pipeline and preserve the model-proxy session.

    Args:
        pipeline (Any): Runner config, import path, or reconstructible runner.
        session_id (str): Original rollout session identity.

    Returns:
        Any: Configured Runner with its original recipe-specific behavior.
    """
    if isinstance(pipeline, str):
        module, separator, attribute = pipeline.rpartition(".")
        if not separator:
            raise ValueError("Pipeline must be a fully qualified import path")
        pipeline = getattr(importlib.import_module(module), attribute)
    if isinstance(pipeline, dict):
        from lagent.utils import create_object

        pipeline = deepcopy(pipeline)
        for entry in pipeline.get("infer", {}).get("entries", []):
            if isinstance(entry, dict) and entry.get("name") == "start_agent_daemon":
                entry.setdefault("env", {})["XTUNER_SESSION_ID"] = session_id
        pipeline = create_object(pipeline)
    if pipeline is None:
        raise ValueError("AgentRolloutItem.pipeline is required")
    return pipeline


def _child_environment(manifest: Path) -> dict[str, str]:
    environment = dict(os.environ)
    paths = [str(Path(path or os.getcwd()).resolve()) for path in sys.path]
    environment.update(
        PYTHONPATH=os.pathsep.join(paths),
        CUDA_VISIBLE_DEVICES="",
        XTUNER_SANDBOX_TRIAL_MANIFEST=str(manifest),
    )
    return environment


async def _wait_worker(
    process: asyncio.subprocess.Process, result: Path, deadline: float, stop_timeout: float
) -> None:
    waiter = asyncio.create_task(process.wait())
    published = False
    try:
        while not waiter.done():
            if result.is_file() and not published:
                published = True
                deadline = min(deadline, time.monotonic() + stop_timeout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Sandbox trial process exceeded its execution or shutdown deadline")
            await asyncio.wait({waiter}, timeout=min(0.1, remaining))
        waiter.result()
    finally:
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)


async def _stop_worker(process: asyncio.subprocess.Process, grace: float) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await asyncio.wait_for(process.wait(), timeout=5)
    # The root may have exited while a local helper remained in its group.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _kill_late_worker(task: asyncio.Task[asyncio.subprocess.Process], *, trial: _Trial | None = None) -> None:
    if not task.cancelled() and task.exception() is None:
        process = task.result()
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        recovery = asyncio.create_task(_reap_late_worker(process, trial), name="sandbox-late-trial-recovery")
        _LATE_TRIALS.add(recovery)

        def finished(done: asyncio.Task[None]) -> None:
            _LATE_TRIALS.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logging.getLogger(__name__).warning("Late trial recovery incomplete; retain the run journal")

        recovery.add_done_callback(finished)


async def _reap_late_worker(process: asyncio.subprocess.Process, trial: _Trial | None) -> None:
    await asyncio.wait_for(process.wait(), timeout=5)
    if trial is not None:
        trial.process = process
        trial.exited = True
        try:
            trial.receipt = await _recover_after_exit(trial.directory, trial.environment, trial.limits.recovery)
        finally:
            _dispose_trial(trial)


async def _recover_after_exit(directory: Path, environment: dict[str, str], timeout: float) -> dict[str, Any]:
    manifest = directory / "journals"
    if not any(manifest.glob("*.json")):
        return {"ok": True, "results": []}
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--recover",
            str(directory),
            "--recovery-timeout",
            str(timeout * 0.8),
            "--parent-pid",
            str(os.getpid()),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    )
    process = None
    deadline = time.monotonic() + timeout
    try:
        process = await asyncio.wait_for(asyncio.shield(launch), timeout=timeout)
        await asyncio.wait_for(process.wait(), timeout=max(0.01, deadline - time.monotonic()))
    finally:
        if process is None:
            launch.add_done_callback(_kill_late_worker)
        elif process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await asyncio.wait_for(process.wait(), timeout=5)
    receipt = directory / "recovery.json"
    if process.returncode != 0 or not receipt.is_file():
        return {"ok": False, "error_type": "TrialRecoveryFailed"}
    return json.loads(receipt.read_text())


def _recover(directory: Path, timeout: float) -> int:
    from cleanup import observe_journals, reconcile_file
    from recycler import request_journal_recovery

    run = Path(os.environ["XTUNER_SANDBOX_CLEANUP_DIR"]).resolve()
    paths = [Path(json.loads(entry.read_text())) for entry in (directory / "journals").glob("*.json")]
    if any(path.parent.resolve() != run or len(path.stem) != 32 or path.suffix != ".json" for path in paths):
        raise ValueError("Invalid trial journal scope")
    paths = list(dict.fromkeys(paths))
    report = observe_journals(paths)
    if not report["ok"]:
        factory = os.environ["XTUNER_SANDBOX_CLEANUP_FACTORY"]
        # Queue every unfinished exact journal before a slow provider can consume
        # the foreground budget. Existing recycler generations keep their bounds.
        pending = [path for path, row in zip(paths, report["results"]) if not row.get("ok")]
        request_journal_recovery(run, factory, pending, lock_timeout=timeout)
        module, name = os.environ["XTUNER_SANDBOX_CLEANUP_FACTORY"].split(":", 1)
        resolver = getattr(importlib.import_module(module), name)
        deadline = time.monotonic() + timeout

        async def recover() -> dict[str, Any]:
            while time.monotonic() < deadline:
                for path in pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        await reconcile_file(path, resolver, max(0.01, remaining * 0.6))
                    except Exception:
                        # Durable handoff already owns these paths. Report the
                        # observed confirmation, never a best-effort DELETE.
                        continue
                observed = observe_journals(paths)
                if observed["ok"]:
                    return observed
                pending[:] = [path for path, row in zip(paths, observed["results"]) if not row.get("ok")]
                await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
            return observe_journals(paths)

        report = asyncio.run(recover())
    (directory / "recovery.json").write_text(json.dumps(report))
    return 0


def _parent_death_guard(parent_pid: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot install trial parent-death protection")
    if os.getppid() != parent_pid:
        raise RuntimeError("Trial owner exited before the child installed its guard")


def _worker(directory: Path, parent_pid: int, control_fd: int) -> int:
    _parent_death_guard(parent_pid)
    import cloudpickle

    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trial_channel import child_channel

    item = cloudpickle.loads((directory / "request.pkl").read_bytes())

    async def execute() -> Any:
        owner = asyncio.current_task()
        assert owner is not None
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, owner.cancel)
        async with child_channel(control_fd):
            runner = resolve_runner(item.pipeline, str(item.uid))
            # The trace writer is process-local; the model trace store remains Ray-owned.
            from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trace import init_writer

            init_writer()
            return await runner.run(item)

    try:
        packet = {"item": asyncio.run(execute())}
    except BaseException as error:
        import traceback

        traceback.print_exc()
        packet = {"error_type": type(error).__name__}
    temporary = directory / "result.tmp"
    temporary.write_bytes(cloudpickle.dumps(packet))
    temporary.replace(directory / "result.pkl")
    return 1 if packet.get("error_type") else 0


def main() -> int:
    """Run the private child/recovery protocol; task data never appears in argv.

    Returns:
        int: Child exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", type=Path)
    mode.add_argument("--recover", type=Path)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--control-fd", type=int)
    parser.add_argument("--recovery-timeout", type=float, default=30)
    args = parser.parse_args()
    if args.worker:
        if args.parent_pid is None or args.control_fd is None:
            parser.error("--worker requires --parent-pid and --control-fd")
        return _worker(args.worker, args.parent_pid, args.control_fd)
    if args.parent_pid is None:
        parser.error("--recover requires --parent-pid")
    _parent_death_guard(args.parent_pid)
    signal.alarm(max(1, math.ceil(args.recovery_timeout)))
    return _recover(args.recover, args.recovery_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
