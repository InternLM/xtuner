"""Run-owned sandbox recovery, usable without importing the training stack.

Run this file directly with --help for the offline reconciliation command.
Journals contain IDs and state only: provider factories and credentials are
supplied by trusted configuration, never serialized from a live provider.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib
import json
import math
import os
import select
import subprocess
import sys
import time
import uuid
import weakref
from pathlib import Path
from typing import Any, Callable


if __package__:
    from .finalization import CleanupError, finalize_bounded
else:
    from finalization import CleanupError, finalize_bounded


_SUPERVISOR: tuple[int, subprocess.Popen] | None = None
_TASKS: set[asyncio.Task] = set()
_WRITES: set[asyncio.Task] = set()
_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("cleanup budget must be finite and positive")
    return value


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            # Preserve acquisition order for dependency-aware recovery.
            json.dump(data, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class CleanupJournal:
    """One atomic snapshot and kernel lease per Pool, held throughout its life.

    A reconciler must acquire this same lock, so it cannot reclaim an active
    Trial, including between training updates or between inference and judging.
    A dead process releases its lock without PID/clock/lease-expiry heuristics.
    Use a run-specific shared filesystem with working cross-node flock semantics.
    """

    def __init__(self, directory: str | Path, provider_key: str) -> None:
        if not provider_key or not isinstance(provider_key, str):
            raise ValueError("persistent cleanup requires an explicit provider key")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (uuid.uuid4().hex + ".json")
        self.lock = self.path.with_suffix(".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.data = {
            "version": 1,
            "provider_key": provider_key,
            "state": "active",
            "resources": {},
            "ambiguous_creates": 0,
            "creating": False,
        }
        self._write_task: asyncio.Task | None = None
        self.flush()

    async def write(self, method: str, *args: Any, **kwargs: Any) -> None:
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
            await asyncio.to_thread(getattr(self, method), *args, **kwargs)

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

        await finalize_bounded(wait(), timeout_seconds=5, diagnostic={})

    def flush(self) -> None:
        self.data["updated_at"] = time.time()
        _atomic_json(self.path, self.data)

    def created(self, env_id: str) -> None:
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("sandbox ID must be a nonempty string")
        self.data["resources"].setdefault(env_id, {"confirmed": False})
        self.flush()

    def confirmed(self, env_id: str) -> None:
        self.data["resources"][env_id] = {"confirmed": True}
        self.flush()

    def begin_create(self) -> None:
        self.data["creating"] = True
        self.flush()

    def identify(self, trial_id: str) -> None:
        self.data["trial_id"] = trial_id
        self.flush()

    def end_create(self, *, ambiguous: bool) -> None:
        self.data["creating"] = False
        self.data["ambiguous_creates"] += int(ambiguous)
        self.flush()

    def detach(self) -> None:
        """Publish terminal ownership before allowing another process to act."""
        self.data["state"] = (
            "pending"
            if (
                self.data["creating"]
                or self.data["ambiguous_creates"]
                or any(not row["confirmed"] for row in self.data["resources"].values())
            )
            else "complete"
        )
        try:
            self.flush()
        finally:
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
    data: dict[str, Any], factory: Callable[[], Any], persist: Callable[[], None], timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + _positive(timeout)
    pending = [key for key, value in data["resources"].items() if not value["confirmed"]]
    provider = factory() if pending else None
    failures: dict[str, str] = {}
    try:
        # Reverse acquisition order preserves dependencies. A slow ID receives
        # only a share of the budget so other members still get an attempt.
        delay = 0.1
        while time.monotonic() < deadline:
            pending = [key for key, value in data["resources"].items() if not value["confirmed"]][::-1]
            if not pending:
                break
            for index, env_id in enumerate(pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await finalize_bounded(
                        delete_confirmed(provider, env_id),
                        timeout_seconds=remaining / (len(pending) - index) / 2,
                        diagnostic={},
                    )
                except CleanupError as exc:
                    failures[env_id] = exc.kind
                else:
                    data["resources"][env_id]["confirmed"] = True
                    failures.pop(env_id, None)
                    await _persist(persist)
            if all(row["confirmed"] for row in data["resources"].values()):
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
    pending = [key for key, value in data["resources"].items() if not value["confirmed"]]
    complete = not pending and not data["creating"] and not data["ambiguous_creates"]
    data.update(state="complete" if complete else "pending", pending_ids=pending, failures=failures)
    await _persist(persist)
    return {
        "ok": complete,
        "pending_ids": pending,
        "ambiguous_creates": data["ambiguous_creates"] + int(data["creating"]),
    }


async def reconcile_file(path: Path, resolver: Callable[[str], Any], timeout: float) -> dict[str, Any]:
    """Skip live owners; an invalid journal fails this file, not the whole run."""
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                # Periodic sweeps must not fsync every completed Trial again.
                return {"journal": path.name, "ok": True, "pending_ids": [], "ambiguous_creates": 0}
            result = await _recover(
                data, lambda: resolver(data["provider_key"]), lambda: _atomic_json(path, data), timeout
            )
            return {"journal": path.name, **result}
        except Exception as exc:
            return {"journal": path.name, "ok": False, "error_type": type(exc).__name__}


def defer_cleanup(
    env_ids: list[str], factory: Callable[[], Any], *, journal: CleanupJournal | None, timeout: float
) -> asyncio.Task:
    """Strongly own one bounded retry worker after foreground cleanup ends."""
    loop = asyncio.get_running_loop()
    limit = _LIMITS.setdefault(loop, asyncio.Semaphore(8))

    async def work() -> dict[str, Any]:
        async with limit:
            if journal is not None:
                return await reconcile_file(journal.path, lambda _: factory(), timeout)
            data = {
                "resources": {key: {"confirmed": False} for key in env_ids},
                "creating": False,
                "ambiguous_creates": 0,
            }
            return await _recover(data, factory, lambda: None, timeout)

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
    directory: Path, resolver: Callable[[str], Any], timeout: float, concurrency: int = 8
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
    paths = sorted(
        path
        for path in directory.glob("*.json")
        if len(path.stem) == 32 and all(c in "0123456789abcdef" for c in path.stem)
    )
    rows = await asyncio.gather(*(one(path) for path in paths))
    report = {"ok": all(row["ok"] for row in rows), "results": rows, "finished_at": time.time()}
    await asyncio.to_thread(_atomic_json, directory / "cleanup-report.json", report)
    return report


def start_cleanup_supervisor() -> None:
    """Start one independent recovery owner for a configured training driver.

    The environment is explicit because only the caller knows the provider's
    backend, namespace and credentials. An incomplete configuration fails before
    training allocates resources. The pipe stays open until process exit;
    foreground fit cleanup does not stop recovery of still-live Ray workers.
    """
    global _SUPERVISOR
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory and not factory:
        return
    if not directory or not factory:
        raise ValueError("sandbox recovery requires both CLEANUP_DIR and CLEANUP_FACTORY")
    if _SUPERVISOR is not None and _SUPERVISOR[0] == os.getpid():
        if _SUPERVISOR[1].poll() is not None:
            raise RuntimeError("sandbox recovery supervisor exited unexpectedly")
        return
    _positive(float(os.environ.get("XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT", "180")))
    _positive(float(os.environ.get("XTUNER_SANDBOX_CLEANUP_INTERVAL", "30")))
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("sandbox recovery directory must be absolute and shared by all workers")
    path.mkdir(parents=True, exist_ok=True)
    with (path / "supervisor.log").open("a") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("supervisor.py"))],
            stdin=subprocess.PIPE,
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
        assert child.stdin is not None
        child.stdin.close()
        raise RuntimeError("sandbox recovery supervisor failed to start; inspect supervisor.log")
    _SUPERVISOR = (os.getpid(), child)


def cleanup_from_environment() -> bool:
    """Driver exit fallback, after Ray workers have stopped (fresh interpreter)."""
    directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
    factory = os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY")
    if not directory and not factory:
        return True
    if not directory or not factory:
        return False
    timeout = _positive(float(os.environ.get("XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT", "180")))
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
                str(min(30, timeout / 2)),
            ],
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path, help="exact run-owned journal directory")
    parser.add_argument(
        "--factory", required=True, help="trusted module:function resolving provider_key to a NEW provider"
    )
    parser.add_argument("--timeout", type=float, default=30, help="per-journal retry budget in seconds")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    module, name = args.factory.split(":", 1)
    resolver = getattr(importlib.import_module(module), name)
    report = asyncio.run(reconcile(args.directory, resolver, args.timeout, args.concurrency))
    print(json.dumps({"ok": report["ok"], "journals": len(report["results"])}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
