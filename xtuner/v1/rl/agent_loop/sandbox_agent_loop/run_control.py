"""Run-scoped sandbox admission and finalization state on a trusted shared filesystem."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO


_OWNER_LOCKS: weakref.WeakSet[TextIO] = weakref.WeakSet()


def track_lock(lock: TextIO) -> TextIO:
    """Prevent a forked helper from retaining the original owner's kernel lease."""
    _OWNER_LOCKS.add(lock)
    return lock


def _close_inherited_locks() -> None:
    for lock in list(_OWNER_LOCKS):
        lock.close()  # Never LOCK_UN: the parent still owns the shared open-file description.
    _OWNER_LOCKS.clear()


os.register_at_fork(after_in_child=_close_inherited_locks)


class RunStoppingError(RuntimeError):
    """Reject creation after run shutdown without replaying the request."""

    def __init__(self, allocation_state: str = "not_allocated") -> None:
        super().__init__("Sandbox run is stopping; no new allocation is allowed")
        self.allocation_state = allocation_state


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Persist an atomic snapshot, including its directory entry."""
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(data, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def admission(directory: Path) -> Iterator[None]:
    """Serialize journal registration/create intent against the stop barrier.

    This lock covers durable admission only, never network I/O. An admitted
    request can complete after stop; its locked Pool journal prevents a final
    successful sweep until that allocation has been accounted for.
    """
    with _gate(directory, shared=True):
        if is_stopping(directory):
            raise RunStoppingError()
        yield


def is_stopping(directory: Path) -> bool:
    """Return whether this run has permanently closed sandbox admission."""
    return (directory / "stop.requested.json").is_file()


def request_stop(directory: Path, reason: str) -> None:
    """Close admission permanently; repeated stop requests are idempotent."""
    with _gate(directory):
        if not is_stopping(directory):
            atomic_json(directory / "stop.requested.json", {"reason": reason, "requested_at": time.time()})


def claim_driver(directory: Path) -> TextIO:
    """Hold the driver lease until process death; reject reuse of a stopped run."""
    with admission(directory):
        lease = track_lock((directory / "driver.lock").open("a"))
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / "driver.started.json").exists():
                raise RuntimeError("Use a fresh cleanup directory for each training invocation")
            atomic_json(directory / "driver.started.json", {"pid": os.getpid(), "started_at": time.time()})
            return lease
        except BaseException:
            lease.close()
            raise


def driver_alive(directory: Path) -> bool:
    """Probe the kernel lease, without PID reuse or heartbeat expiry heuristics."""
    if not (directory / "driver.started.json").is_file():
        raise RuntimeError("Sandbox recovery peer requires a registered driver")
    with track_lock((directory / "driver.lock").open("a")) as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def record_result(
    directory: Path,
    *,
    cleanup_confirmed: bool,
    training_outcome: str | None = None,
    finalization_failed: bool = False,
) -> dict[str, Any]:
    """Publish final status without allowing a stale failed scan to undo confirmation.

    Finalization failure is sticky: a later recovery records reclaimed resources
    but cannot relabel an unsuccessful fit invocation as successful.
    """
    if training_outcome not in {None, "succeeded", "failed", "interrupted"}:
        raise ValueError("invalid training outcome")
    with _gate(directory):
        if not is_stopping(directory):
            raise RuntimeError("Cannot finalize sandbox cleanup before closing admission")
        path = directory / "run-cleanup.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        outcome = previous.get("training_outcome", "unknown")
        if training_outcome is not None:
            if outcome in {"unknown", "succeeded"}:
                outcome = training_outcome
        if (
            training_outcome == "succeeded"
            and (directory / "driver.started.json").exists()
            and not driver_alive(directory)
        ):
            outcome = "interrupted"
        confirmed = previous.get("cleanup_confirmed") is True or cleanup_confirmed
        fit_confirmed = previous.get("fit_cleanup_confirmed") is True or (
            training_outcome is not None and cleanup_confirmed
        )
        failed = previous.get("finalization_failed") is True or finalization_failed
        success = confirmed and fit_confirmed and outcome == "succeeded" and not failed
        result = {
            "training_outcome": outcome,
            "cleanup_confirmed": confirmed,
            "finalization_failed": failed,
            "fit_cleanup_confirmed": fit_confirmed,
            "success": success,
            "state": "succeeded" if success else "cleanup_complete" if confirmed else "cleanup_pending",
            "updated_at": time.time(),
        }
        atomic_json(path, result)
        return result


@contextmanager
def _gate(directory: Path, *, shared: bool = False) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with track_lock((directory / "admission.lock").open("a")) as lock:
        fcntl.flock(lock, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
