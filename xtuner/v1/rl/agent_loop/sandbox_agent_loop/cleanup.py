"""Run-owned sandbox recovery, usable without importing the training stack.

Run this file directly with --help for the offline reconciliation command.
Journals contain IDs and state only: provider factories and credentials are
supplied by trusted configuration, never serialized from a live provider.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import select
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, TextIO


if __package__:
    from . import locking, recycler, run_control
    from .finalization import CleanupError, finalize_bounded
else:
    import locking
    import recycler
    import run_control
    from finalization import CleanupError, finalize_bounded


_SUPERVISOR: tuple[int, str, str, bool, subprocess.Popen] | None = None
_DRIVER_LEASE: TextIO | None = None
_SHUTDOWN_GUARD: subprocess.Popen | None = None
_TASKS: set[asyncio.Task] = set()
_WRITES: set[asyncio.Task] = set()
_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("cleanup budget must be finite and positive")
    return value


_atomic_json = run_control.atomic_json


class CleanupJournal:
    """One atomic snapshot and kernel lease per Pool, held throughout its life.

    A reconciler must acquire this same lock, so it cannot reclaim an active
    Trial, including between training updates or between inference and judging.
    A dead process releases its lock without PID/clock/lease-expiry heuristics.
    Use a run-specific shared filesystem with validated cross-node OFD record-lock semantics.
    """

    @classmethod
    async def open(cls, directory: str | Path, provider_key: str, *, timeout_seconds: float = 5) -> CleanupJournal:
        """Register without blocking the actor, owning late initialization on cancellation.

        The I/O thread keeps ownership until the caller accepts the journal. If
        registration times out or is cancelled, that same thread detaches the
        unused journal after its initial write finishes. No event-loop callback
        is required to release a late owner, including during loop shutdown.
        """
        _positive(timeout_seconds)
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[CleanupJournal] = loop.create_future()
        resolved = threading.Event()
        adopted = False

        def publish(journal: CleanupJournal | None, error: BaseException | None) -> None:
            if ready.done():
                return
            if error is not None:
                ready.set_exception(error)
            else:
                assert journal is not None
                ready.set_result(journal)

        def initialize() -> None:
            try:
                journal = cls(directory, provider_key)
            except BaseException as exc:
                if not resolved.is_set():
                    loop.call_soon_threadsafe(publish, None, exc)
                return
            try:
                loop.call_soon_threadsafe(publish, journal, None)
            except RuntimeError:  # The event loop ended before registration.
                journal.detach()
                return
            resolved.wait()
            if not adopted:
                journal.detach()

        task = asyncio.create_task(asyncio.to_thread(initialize), name="sandbox-journal-open")
        _WRITES.add(task)

        def finished(result: asyncio.Task[None]) -> None:
            _WRITES.discard(result)
            if not result.cancelled():
                result.exception()

        task.add_done_callback(finished)

        async def receive() -> CleanupJournal:
            return await asyncio.shield(ready)

        try:
            journal = await finalize_bounded(receive(), timeout_seconds=timeout_seconds, diagnostic={})
            adopted = True
            return journal
        except CleanupError:
            if ready.done() and not ready.cancelled():
                error = ready.exception()
                if error is not None:
                    raise error  # Preserve admission rejection and allocation markers.
            raise
        finally:
            # Cancellation must also abandon a result already published but not
            # yet accepted. Cancelling this Future never cancels the I/O thread.
            if not adopted:
                if ready.done() and not ready.cancelled():
                    ready.exception()
                ready.cancel()
            resolved.set()

    def __init__(self, directory: str | Path, provider_key: str) -> None:
        if not provider_key or not isinstance(provider_key, str):
            raise ValueError("persistent cleanup requires an explicit provider key")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with run_control.admission(directory):
            self.path = directory / (uuid.uuid4().hex + ".json")
            self.lock = run_control.track_lock(locking.open_lock(self.path.with_suffix(".lock")))
            try:
                locking.acquire(self.lock)
            except BaseException:
                self.lock.close()
                raise
            self.data = {
                "version": 1,
                "provider_key": provider_key,
                "state": "active",
                "resources": {},
                "ambiguous_creates": 0,
                "creating": False,
            }
            # None is the legacy serial API's independent create intent.
            # Per-request tokens remain in memory; the persisted boolean still
            # conservatively records whether any POST lacks an exact ID.
            self._pending_creates: set[str | None] = set()
            self._write_task: asyncio.Task | None = None
            try:
                self.flush()
                if __package__:
                    from .trial_process import register_trial_journal
                else:
                    from trial_process import register_trial_journal

                register_trial_journal(self.path)
            except BaseException:
                self.lock.close()
                raise

    async def persist(self, mutation: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """Serialize disk writes, retaining them across coroutine cancellation.

        Detach queues behind earlier writes; no late write can recreate stale
        state after releasing the owner lock. Foreground waiting is bounded,
        but a stuck filesystem retains ownership until the process exits.
        """
        previous = self._write_task

        async def operation() -> None:
            if previous is not None:
                try:
                    await asyncio.shield(previous)
                except Exception:
                    pass  # The next full snapshot may recover a failed write.
            await asyncio.to_thread(mutation, *args, **kwargs)

        task = asyncio.create_task(operation(), name="sandbox-journal-write")
        self._write_task = task
        _WRITES.add(task)

        def done(result: asyncio.Task[None]) -> None:
            _WRITES.discard(result)
            if not result.cancelled():
                result.exception()

        task.add_done_callback(done)

        async def wait() -> None:
            await asyncio.shield(task)

        try:
            await finalize_bounded(wait(), timeout_seconds=5, diagnostic={})
        except CleanupError:
            # Admission rejection is a control decision, not a disk failure.
            # Keep its allocation marker so the outer group cannot retry it.
            if task.done() and not task.cancelled() and isinstance(task.exception(), run_control.RunStoppingError):
                stopped = task.exception()
                assert stopped is not None
                raise stopped
            raise

    @property
    def has_unresolved_creation(self) -> bool:
        """Whether recovery still lacks an exact ID for an admitted request."""
        return bool(self.data["creating"] or self.data["ambiguous_creates"])

    async def remember_created(self, env_id: str) -> None:
        """Persist newly observed ownership without rewriting already known IDs."""
        if env_id not in self.data["resources"]:
            await self.persist(self.created, env_id)

    def flush(self) -> None:
        self.data["updated_at"] = time.time()
        _atomic_json(self.path, self.data)

    def created(self, env_id: str) -> None:
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("sandbox ID must be a nonempty string")
        self.data["resources"].setdefault(env_id, {"confirmed": False})
        self.flush()

    def allocated(self, env_id: str, *, create_id: str | None = None) -> None:
        """Resolve only this request's intent before provider readiness waits."""
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("sandbox ID must be a nonempty string")
        self.data["resources"].setdefault(env_id, {"confirmed": False})
        self._pending_creates.discard(create_id)
        self.data["creating"] = bool(self._pending_creates)
        self.flush()

    def confirmed(self, env_id: str) -> None:
        self.data["resources"][env_id] = {"confirmed": True}
        self.flush()

    def begin_create(self, *, create_id: str | None = None) -> None:
        with run_control.admission(self.path.parent):
            self._pending_creates.add(create_id)
            self.data["creating"] = bool(self._pending_creates)
            self.flush()

    def identify(self, trial_id: str) -> None:
        self.data["trial_id"] = trial_id
        self.flush()

    def end_create(self, *, ambiguous: bool, create_id: str | None = None) -> None:
        # A prior allocated() resolved this request, or an earlier end_create()
        # already accounted for it. Repeated completion must not add ambiguity.
        unresolved = create_id in self._pending_creates
        self._pending_creates.discard(create_id)
        self.data["creating"] = bool(self._pending_creates)
        self.data["ambiguous_creates"] += int(ambiguous and unresolved)
        self.flush()

    def detach(self) -> None:
        """Publish terminal ownership once; failed publication retains the lease."""
        if self.lock.closed:
            return
        self.data["state"] = (
            "pending"
            if (
                self.data["creating"]
                or self.data["ambiguous_creates"]
                or any(not row["confirmed"] for row in self.data["resources"].values())
            )
            else "complete"
        )
        self.flush()
        self.lock.close()


def _absent(exc: Exception) -> bool:
    return (
        getattr(exc, "status_code", None) == 404 or getattr(getattr(exc, "response", None), "status_code", None) == 404
    )


async def _persist(persist: Callable[[], None]) -> None:
    # A cancelled thread future does not stop fsync/replace. Keep the owner
    # lock until the actual write has settled, preventing a late stale write
    # from overwriting another reconciler's result. The exit subprocess has
    # an independent wall-clock budget for filesystem/SDK stalls.
    task = asyncio.create_task(asyncio.to_thread(persist))
    _WRITES.add(task)
    cancellation = None
    try:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
        task.result()
    finally:
        _WRITES.discard(task)
    if cancellation is not None:
        raise cancellation


async def delete_confirmed(provider: Any, env_id: str, *, poll_seconds: float = 0.2) -> None:
    """DELETE acceptance is insufficient; require GET deleted or an exact 404.

    Providers normalize platform-specific terminal states. In particular,
    stopping/failed/expired are not assumed to mean resource reclamation.
    The caller supplies the total deadline, including repeated observations.
    """
    query = getattr(provider, "get", None)
    if not callable(query):
        raise RuntimeError("provider has no deletion confirmation API")
    try:
        await provider.delete(env_id)
    except Exception as exc:
        if _absent(exc):
            return
        raise
    while True:
        try:
            response = await query(env_id)
        except Exception as exc:
            if _absent(exc):
                return
            raise
        if response.get("env", response).get("status") == "deleted":
            return
        await asyncio.sleep(poll_seconds)


async def _recover(
    data: dict[str, Any],
    factory: Callable[[], Any],
    persist: Callable[[], None],
    timeout: float,
    *,
    verify_deletion: bool = True,
) -> dict[str, Any]:
    if not verify_deletion and "version" in data:
        raise ValueError("persistent recovery requires deletion confirmation")
    completion_key = "confirmed" if verify_deletion else "delete_accepted"
    deadline = time.monotonic() + _positive(timeout)
    pending = [key for key, value in data["resources"].items() if not value.get(completion_key, False)]
    provider = factory() if pending else None
    failures: dict[str, str] = {}
    try:
        # Reverse acquisition order preserves dependencies. A slow ID receives
        # only a share of the budget so other members still get an attempt.
        delay = 0.1
        while time.monotonic() < deadline:
            pending = [key for key, value in data["resources"].items() if not value.get(completion_key, False)][::-1]
            if not pending:
                break
            for index, env_id in enumerate(pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await finalize_bounded(
                        delete_confirmed(provider, env_id) if verify_deletion else provider.delete(env_id),
                        timeout_seconds=remaining / (len(pending) - index) / 2,
                        diagnostic={},
                    )
                except CleanupError as exc:
                    failures[env_id] = exc.kind
                else:
                    data["resources"][env_id][completion_key] = True
                    failures.pop(env_id, None)
                    await _persist(persist)
            if all(row.get(completion_key, False) for row in data["resources"].values()):
                break
            await asyncio.sleep(min(delay, max(0, deadline - time.monotonic())))
            delay = min(delay * 2, 5)
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            try:
                await finalize_bounded(close(), timeout_seconds=min(5, timeout), diagnostic={})
            except CleanupError:
                data["transport_close_failed"] = True
    pending = [key for key, value in data["resources"].items() if not value.get(completion_key, False)]
    complete = not pending and not data["creating"] and not data["ambiguous_creates"]
    state = ("complete" if verify_deletion else "delete_accepted") if complete else "pending"
    data.update(state=state, pending_ids=pending, failures=failures)
    await _persist(persist)
    return {
        "ok": complete,
        "deletion_confirmed": complete and verify_deletion,
        "pending_ids": pending,
        "ambiguous_creates": data["ambiguous_creates"] + int(data["creating"]),
    }


async def reconcile_file(path: Path, resolver: Callable[[str], Any], timeout: float) -> dict[str, Any]:
    """Skip live owners; an invalid journal fails this file, not the whole run."""
    with run_control.track_lock(locking.open_lock(path.with_suffix(".lock"))) as lock:
        try:
            locking.acquire(lock)
        except BlockingIOError:
            return {"journal": path.name, "ok": False, "state": "active"}
        try:
            data = json.loads(path.read_text())
            if (
                data["version"] != 1
                or not isinstance(data["resources"], dict)
                or not isinstance(data["provider_key"], str)
                or not data["provider_key"]
                or type(data["creating"]) is not bool
                or type(data["ambiguous_creates"]) is not int
                or data["ambiguous_creates"] < 0
                or any(
                    not isinstance(key, str) or not key or type(value.get("confirmed")) is not bool
                    for key, value in data["resources"].items()
                )
            ):
                raise ValueError("unsupported cleanup journal")
            if (
                data.get("state") == "complete"
                and not data["creating"]
                and not data["ambiguous_creates"]
                and all(row["confirmed"] for row in data["resources"].values())
            ):
                # Repeated sweeps must not fsync every completed Trial again.
                return {"journal": path.name, "ok": True, "pending_ids": [], "ambiguous_creates": 0}
            result = await _recover(
                data, lambda: resolver(data["provider_key"]), lambda: _atomic_json(path, data), timeout
            )
            return {"journal": path.name, **result}
        except Exception as exc:
            return {"journal": path.name, "ok": False, "error_type": type(exc).__name__}


def observe_journals(paths: list[Path]) -> dict[str, Any]:
    """Check durable completion without acquiring resources or issuing DELETE.

    A live owner, unresolved creation, malformed snapshot or missing journal is
    pending evidence even when a stale snapshot happens to say complete.
    """
    rows = []
    for path in paths:
        try:
            with run_control.track_lock(locking.open_lock(path.with_suffix(".lock"))) as lock:
                try:
                    locking.acquire(lock)
                except BlockingIOError:
                    rows.append({"journal": path.name, "ok": False, "state": "active"})
                    continue
                data = json.loads(path.read_text())
                confirmed = (
                    data.get("version") == 1
                    and isinstance(data.get("provider_key"), str)
                    and bool(data["provider_key"])
                    and data.get("state") == "complete"
                    and data.get("creating") is False
                    and type(data.get("ambiguous_creates")) is int
                    and data["ambiguous_creates"] == 0
                    and isinstance(data.get("resources"), dict)
                    and all(
                        isinstance(key, str) and bool(key) and isinstance(row, dict) and row.get("confirmed") is True
                        for key, row in data["resources"].items()
                    )
                )
                rows.append({"journal": path.name, "ok": confirmed})
        except (OSError, ValueError, AttributeError, TypeError) as error:
            rows.append({"journal": path.name, "ok": False, "error_type": type(error).__name__})
    return {"ok": all(row["ok"] for row in rows), "results": rows}


def defer_cleanup(
    env_ids: list[str],
    factory: Callable[[], Any],
    *,
    journal: CleanupJournal | None,
    timeout: float,
    verify_deletion: bool = True,
) -> asyncio.Task:
    """Strongly own one bounded retry worker after foreground cleanup ends."""
    if journal is not None and not verify_deletion:
        raise ValueError("persistent recovery requires deletion confirmation")
    loop = asyncio.get_running_loop()
    limit = _LIMITS.setdefault(loop, asyncio.Semaphore(8))

    async def work() -> dict[str, Any]:
        factory_path = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
        if journal is not None and factory_path:
            queued = await asyncio.to_thread(
                recycler.handoff_recovery, journal.path.parent, factory_path, journal.path, timeout=5
            )
            if not queued:
                raise RuntimeError("Sandbox recovery handoff unconfirmed; retain journal")
            return {"ok": False, "state": "queued", "journal": journal.path.name}
        async with limit:
            if journal is not None:
                return await reconcile_file(journal.path, lambda _: factory(), timeout)
            data = {
                "resources": {key: {"confirmed": False} for key in env_ids},
                "creating": False,
                "ambiguous_creates": 0,
            }
            return await _recover(data, factory, lambda: None, timeout, verify_deletion=verify_deletion)

    task = asyncio.create_task(work(), name="sandbox-cleanup")
    _TASKS.add(task)

    def finished(done: asyncio.Task[dict[str, Any]]) -> None:
        _TASKS.discard(done)
        if not done.cancelled():
            done.exception()

    task.add_done_callback(finished)
    return task


async def drain_cleanup_tasks(timeout: float = 30) -> bool:
    """Wait for this process's deferred workers without cancelling their owners."""
    if not _TASKS:
        return True
    _, pending = await asyncio.wait(set(_TASKS), timeout=_positive(timeout))
    return not pending


async def reconcile(
    directory: Path,
    resolver: Callable[[str], Any],
    timeout: float,
    concurrency: int = 8,
    journals: list[Path] | None = None,
) -> dict[str, Any]:
    _positive(timeout)
    if concurrency < 1:
        raise ValueError("cleanup concurrency must be positive")
    if not directory.is_dir():
        raise ValueError("cleanup directory does not exist")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(path: Path) -> dict[str, Any]:
        async with semaphore:
            try:
                return await reconcile_file(path, resolver, timeout)
            except Exception as exc:
                return {"journal": path.name, "ok": False, "error_type": type(exc).__name__}

    # UUID filenames exclude reports/configs and scope this command to one run.
    paths = recycler.journal_paths(directory) if journals is None else journals
    rows = await asyncio.gather(*(one(path) for path in paths))
    report = {"ok": all(row["ok"] for row in rows), "results": rows, "finished_at": time.time()}
    await asyncio.to_thread(_atomic_json, directory / "cleanup-report.json", report)
    return report


def start_cleanup_supervisor(*, peer: bool = False) -> None:
    """Register a passive driver-exit witness, without periodic resource scans.

    The environment is explicit because only the caller knows the provider's
    backend, namespace and credentials. An incomplete configuration fails before
    training allocates resources. The driver lease stays open until process exit;
    foreground fit cleanup does not stop recovery of still-live Ray workers.
    """
    global _SUPERVISOR, _DRIVER_LEASE
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory and not factory:
        return
    if not directory or not factory:
        raise ValueError("sandbox recovery requires both CLEANUP_DIR and CLEANUP_FACTORY")
    if _SUPERVISOR is not None and _SUPERVISOR[0] == os.getpid():
        if _SUPERVISOR[1:4] != (directory, factory, peer):
            raise RuntimeError("A process cannot switch sandbox recovery runs or roles")
        if run_control.is_stopping(Path(directory)):
            raise run_control.RunStoppingError()
        if _SUPERVISOR[4].poll() is not None:
            raise RuntimeError("sandbox recovery supervisor exited unexpectedly")
        return
    _positive(float(os.environ.get("XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT", "180")))
    _positive(float(os.environ.get("XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT", "30")))
    recycler.settings()
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("sandbox recovery directory must be absolute and shared by all workers")
    path.mkdir(parents=True, exist_ok=True)
    if peer:
        if not run_control.driver_alive(path) or run_control.is_stopping(path):
            raise RuntimeError("Cannot join sandbox recovery after driver shutdown")
    elif _DRIVER_LEASE is None or _DRIVER_LEASE.closed:
        _DRIVER_LEASE = run_control.claim_driver(path)
    with (path / "supervisor.log").open("a") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("supervisor.py"))],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
    assert child.stdout is not None
    ready, _, _ = select.select([child.stdout], [], [], 30)
    if not ready or child.stdout.readline() != b"READY\n":
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
        raise RuntimeError("sandbox recovery supervisor failed to start; inspect supervisor.log")
    child.stdout.close()
    _SUPERVISOR = (os.getpid(), directory, factory, peer, child)


@contextmanager
def guard_ray_shutdown(*, training_outcome: str | None = None) -> Iterator[None]:
    """Arm exit recovery before entering a possibly blocking Ray shutdown.

    Ray keeps its original calling thread. The guard only waits on a pipe:
    normal completion disarms it; timeout or driver death runs the bounded
    exit sweep, respecting Pool owner locks, then hands failures to recovery.
    """
    global _SHUTDOWN_GUARD
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory or not factory:
        yield
        return
    timeout = _positive(float(os.environ.get("XTUNER_SANDBOX_RAY_SHUTDOWN_TIMEOUT", "30")))
    read_fd, write_fd = os.pipe()
    try:
        try:
            _SHUTDOWN_GUARD = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--directory",
                    directory,
                    "--factory",
                    factory,
                    "--finalize",
                    "--watch-ray-fd",
                    str(read_fd),
                    "--watch-timeout",
                    str(timeout),
                    *(["--training-outcome", training_outcome] if training_outcome is not None else []),
                ],
                pass_fds=(read_fd,),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            # If no guard can be launched, do not defer the only recovery
            # opportunity until after the potentially blocking Ray call.
            cleanup_from_environment(finalize=True, training_outcome=training_outcome)
        finally:
            os.close(read_fd)
        yield
    finally:
        try:
            os.write(write_fd, b"D")
        except BrokenPipeError:
            pass  # The guard already took over after its deadline.
        finally:
            os.close(write_fd)
        if _SHUTDOWN_GUARD is not None:
            try:
                _SHUTDOWN_GUARD.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass  # Retain Popen while independent cleanup completes.


def request_stop_from_environment() -> bool:
    """Close creation admission in a killable child before stopping Ray workers."""
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory and not factory:
        return True
    if not directory:
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--directory",
                directory,
                "--stop-only",
            ],
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def cleanup_from_environment(*, finalize: bool = False, training_outcome: str | None = None) -> bool:
    """Run one exit sweep; hand failures to the durable background recycler."""
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory and not factory:
        return True
    if not directory or not factory:
        return False
    timeout = _positive(float(os.environ.get("XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT", "180")))
    deadline = time.monotonic() + timeout
    # The failure handoff is part of the exit budget, not an unbounded finally.
    handoff_reserve = min(5.0, timeout * 0.2) if finalize else 0.0
    report_reserve = min(2.0, timeout * 0.1) if finalize and training_outcome is not None else 0.0
    sweep_timeout = timeout - handoff_reserve - report_reserve
    confirmed = False
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--directory",
                directory,
                "--factory",
                factory,
                "--timeout",
                str(min(30, sweep_timeout / 2)),
                "--concurrency",
                str(recycler.settings()["concurrency"]),
                *(["--finalize", "--budget", str(sweep_timeout * 0.9)] if finalize else []),
                *(["--training-outcome", training_outcome] if training_outcome is not None else []),
            ],
            timeout=sweep_timeout,
            check=False,
        )
        confirmed = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        pass  # Durable queue handles the failed/terminated exit subprocess.
    finally:
        if finalize and training_outcome is not None:
            # This reports a fact already observed by the driver. A reporting
            # failure cannot undo a confirmed sweep or introduce a fit failure.
            reported = _record_fit_result(
                Path(directory),
                training_outcome,
                confirmed=confirmed,
                # A confirmed sweep needs no failure handoff, so its report can
                # use that remaining reserve without extending the exit budget.
                timeout=min(5.0 if confirmed else report_reserve, max(0, deadline - time.monotonic())),
            )
            if not reported:
                print("Sandbox fit report unconfirmed; retain the run journal", file=sys.stderr)
        if finalize and not confirmed:
            try:
                if not recycler.handoff_recovery(Path(directory), factory, timeout=deadline - time.monotonic()):
                    print("Sandbox recovery handoff unconfirmed; retain journals", file=sys.stderr)
            except Exception as exc:
                print(f"Sandbox recovery handoff failed ({type(exc).__name__}); retain journals", file=sys.stderr)
    return confirmed


def _record_fit_result(directory: Path, outcome: str, *, confirmed: bool, timeout: float) -> bool:
    """Bound driver acknowledgement/failure reporting, independently of the factory."""
    if timeout <= 0:
        return False
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--directory",
                str(directory),
                "--record-fit",
                "confirmed" if confirmed else "unconfirmed",
                "--training-outcome",
                outcome,
            ],
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def finish_training(training_outcome: str) -> None:
    """Require confirmed cleanup on successful fit; preserve an existing training error."""
    try:
        confirmed = cleanup_from_environment(finalize=True, training_outcome=training_outcome)
    except Exception:
        if training_outcome == "succeeded":
            raise CleanupError("cleanup_unconfirmed", "training_exit") from None
        return
    if not confirmed and training_outcome == "succeeded":
        raise CleanupError("cleanup_unconfirmed", "training_exit")


@asynccontextmanager
async def cleanup_owner(directory: Path, budget: float) -> AsyncIterator[None]:
    """Serialize head, peers and background attempts under one run-level lock."""
    deadline = time.monotonic() + _positive(budget)
    with run_control.track_lock(locking.open_lock(directory / "run-cleanup.lock")) as lock:
        while True:
            try:
                locking.acquire(lock)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CleanupError("cleanup_owner_busy", "run_exit") from None
                await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
        yield


async def finalize_run(
    directory: Path,
    resolver: Callable[[str], Any],
    timeout: float,
    budget: float,
    concurrency: int = 8,
    training_outcome: str | None = None,
    journals: list[Path] | None = None,
    known_journals: list[Path] | None = None,
    request_batch: Path | None = None,
) -> dict[str, Any]:
    """Perform one bounded exit sweep and publish the explicit training outcome."""
    await asyncio.to_thread(run_control.request_stop, directory, "training_exit")
    await asyncio.to_thread(
        run_control.record_result, directory, cleanup_confirmed=False, training_outcome=training_outcome
    )
    async with cleanup_owner(directory, budget):
        if request_batch is not None:
            # This process is bounded by the recycler's attempt and guardian
            # deadlines, including stop, directory I/O and the atomic seal.
            journals, known = await asyncio.to_thread(
                recycler.batch_journals, directory, request_batch, seal_exit=True
            )
            known_journals = [directory / name for name in sorted(known)]
        report = await reconcile(directory, resolver, min(timeout, budget), concurrency, journals)
        if journals is not None or known_journals is not None:
            # Selection controls DELETE budget ownership. Completion still
            # observes the whole stopped run without expanding that selection.
            # Preserve selected/ancestor paths even if their files disappeared:
            # a directory listing cannot turn missing ownership into success.
            observed = set(recycler.journal_paths(directory))
            observed.update(journals or ())
            observed.update(known_journals or ())
            report = await asyncio.to_thread(observe_journals, sorted(observed))
        return await asyncio.to_thread(
            run_control.record_result,
            directory,
            cleanup_confirmed=report["ok"],
            training_outcome=training_outcome,
            finalization_failed=not report["ok"] and training_outcome is not None,
        )


async def recover_queued(
    directory: Path,
    resolver: Callable[[str], Any],
    timeout: float,
    budget: float,
    concurrency: int,
    request_batch: Path | None = None,
) -> dict[str, Any]:
    """Recover only explicit failures; acknowledge immutable request snapshots."""
    queued = (
        recycler.requests(directory) if request_batch is None else recycler.batch_requests(directory, request_batch)[0]
    )
    if not queued:
        return {"ok": True, "requests": 0}
    rows = []
    invalid = []
    for path in queued:
        try:
            kind, name = recycler.read_request(path)
            row = {"kind": kind, "journal": name}
        except (OSError, ValueError) as exc:
            # Retain the original request for inspection. One damaged entry
            # must neither block valid work nor be acknowledged by an exit scan.
            invalid.append({"request": path.name, "error_kind": type(exc).__name__})
            continue
        rows.append((path, row))
    if not rows:
        return {"ok": False, "requests": len(queued), "acknowledged": 0, "invalid_requests": invalid}
    if any(row["kind"] == "exit" for _, row in rows):
        result = await finalize_run(
            directory,
            resolver,
            timeout,
            budget,
            concurrency,
            request_batch=request_batch,
        )
        ok = result["cleanup_confirmed"]
        acknowledged = [path for path, _ in rows] if ok else []
    else:
        journals = recycler.batch_journals(directory, request_batch)[0] if request_batch is not None else None
        async with cleanup_owner(directory, budget):
            names = sorted({row["journal"] for _, row in rows})
            requested = [directory / name for name in names]
            await reconcile(directory, resolver, timeout, concurrency, requested if journals is None else journals)
            result = await asyncio.to_thread(observe_journals, requested)
            completed = {row["journal"] for row in result["results"] if row["ok"]}
            acknowledged = [path for path, row in rows if row["journal"] in completed]
            ok = result["ok"]
    for path in acknowledged:
        path.unlink(missing_ok=True)  # Unique names cannot remove a later request.
    return {
        "ok": ok and not invalid,
        "requests": len(queued),
        "acknowledged": len(acknowledged),
        "invalid_requests": invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path, help="exact run-owned journal directory")
    parser.add_argument("--factory", help="trusted module:function resolving provider_key to a NEW provider")
    parser.add_argument("--timeout", type=float, default=30, help="per-journal retry budget in seconds")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--recover-queued", action="store_true", help="one pass over explicit failed work")
    parser.add_argument("--request-batch", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", help="permanently stop creates and finalize this run")
    parser.add_argument(
        "--budget", type=float, default=180, help="owner-lock wait budget; caller must bound total process lifetime"
    )
    parser.add_argument("--training-outcome", choices=["succeeded", "failed", "interrupted"])
    parser.add_argument("--record-fit", choices=["confirmed", "unconfirmed"], help=argparse.SUPPRESS)
    parser.add_argument("--stop-only", action="store_true", help="close admission without loading the provider")
    parser.add_argument("--watch-ray-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--watch-timeout", type=float, default=30, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.record_fit is not None:
        if not args.training_outcome or args.finalize or args.stop_only or args.recover_queued or args.watch_ray_fd:
            parser.error("--record-fit requires only --directory and --training-outcome")
        confirmed = args.record_fit == "confirmed"
        run_control.request_stop(args.directory, "training_exit")
        run_control.record_result(
            args.directory,
            cleanup_confirmed=confirmed,
            training_outcome=args.training_outcome,
            finalization_failed=not confirmed,
            fit_acknowledged=confirmed,
        )
        return 0
    if args.request_batch is not None and not args.recover_queued:
        parser.error("--request-batch requires --recover-queued")
    if args.watch_ray_fd is not None:
        if not args.finalize or not args.factory or args.recover_queued or args.stop_only:
            parser.error("--watch-ray-fd requires --finalize and --factory")
        # No shared-filesystem operations before the drain deadline. A normal
        # Ray return sends D; EOF means the driver disappeared mid-shutdown.
        try:
            ready, _, _ = select.select([args.watch_ray_fd], [], [], _positive(args.watch_timeout))
            if ready and os.read(args.watch_ray_fd, 1) == b"D":
                return 0
        finally:
            os.close(args.watch_ray_fd)
        return 0 if cleanup_from_environment(finalize=True, training_outcome=args.training_outcome) else 1
    if args.recover_queued and (args.finalize or args.stop_only or args.training_outcome):
        parser.error("--recover-queued cannot be combined with exit/stop flags")
    if args.training_outcome and not args.finalize:
        parser.error("--training-outcome requires --finalize")
    if args.stop_only:
        run_control.request_stop(args.directory, "driver_shutdown")
        return 0
    if not args.factory:
        parser.error("--factory is required for recovery")
    if args.finalize:
        run_control.request_stop(args.directory, "training_exit")
        run_control.record_result(args.directory, cleanup_confirmed=False, training_outcome=args.training_outcome)
    try:
        module, name = args.factory.split(":", 1)
        resolver = getattr(importlib.import_module(module), name)
        if args.recover_queued:
            report = asyncio.run(
                recover_queued(
                    args.directory, resolver, args.timeout, args.budget, args.concurrency, args.request_batch
                )
            )
            print(json.dumps(report))
            return 0 if report["ok"] else 1
        if args.finalize:
            report = asyncio.run(
                finalize_run(
                    args.directory, resolver, args.timeout, args.budget, args.concurrency, args.training_outcome
                )
            )
            print(json.dumps(report))
            # Resource confirmation is the child's only acknowledgement. The
            # driver records fit acknowledgement after observing this exit.
            return 0 if report["cleanup_confirmed"] else 1
    except BaseException:
        if args.finalize:
            run_control.record_result(
                args.directory, cleanup_confirmed=False, finalization_failed=args.training_outcome is not None
            )
        raise

    async def manual_sweep() -> dict[str, Any]:
        async with cleanup_owner(args.directory, args.budget):
            return await reconcile(args.directory, resolver, args.timeout, args.concurrency)

    report = asyncio.run(manual_sweep())
    print(json.dumps({"ok": report["ok"], "journals": len(report["results"])}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
