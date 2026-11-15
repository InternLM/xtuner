"""Durable failure handoff to one bounded, backoff-driven recycler per run."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


if __package__:
    from .run_control import atomic_json, track_lock
else:
    from run_control import atomic_json, track_lock


def settings() -> dict[str, Any]:
    """Read trusted retry limits; credentials are never written to the queue.

    Returns:
        dict[str, Any]: Concurrency, attempt and time budgets.
    """
    prefix = "XTUNER_SANDBOX_RECOVERY_"
    values = {
        "concurrency": int(os.environ.get(prefix + "CONCURRENCY", "8")),
        "attempts": int(os.environ.get(prefix + "ATTEMPTS", "6")),
        "backoff": float(os.environ.get(prefix + "BACKOFF", "2")),
        "max_backoff": float(os.environ.get(prefix + "MAX_BACKOFF", "60")),
        "attempt_timeout": float(os.environ.get(prefix + "ATTEMPT_TIMEOUT", "30")),
        "budget": float(os.environ.get(prefix + "BUDGET", "600")),
    }
    if not 1 <= values["concurrency"] <= 64 or not 1 <= values["attempts"] <= 100:
        raise ValueError("recovery concurrency must be 1..64 and attempts 1..100")
    if any(
        not math.isfinite(values[key]) or values[key] <= 0 for key in values if key not in {"concurrency", "attempts"}
    ):
        raise ValueError("recovery time budgets must be finite and positive")
    return values


def handoff_recovery(directory: Path, factory: str, journal: Path | None = None, *, timeout: float = 5) -> bool:
    """Bound enqueue, filesystem writes and worker launch in a killable child."""
    if not math.isfinite(timeout) or timeout <= 0:
        return False
    try:
        result = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()), "--enqueue",
                "--directory", str(directory), "--factory", factory,
                "--handoff-timeout", str(timeout * 0.9),
                *(["--journal", str(journal)] if journal is not None else []),
            ],
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def request_recovery(
    directory: Path, factory: str, journal: Path | None = None, *, lock_timeout: float = 5
) -> None:
    """Persist failed work before starting or notifying the run's recycler.

    Args:
        directory (Path): Exact run-owned directory.
        factory (str): Trusted provider factory import path.
        journal (Path | None): Failed Pool journal, or the entire stopped run.
    """
    settings()  # Reject bad retry policy before spawning.
    if not factory or not directory.is_absolute():
        raise ValueError("recovery requires a factory and absolute run directory")
    if journal is not None and journal.parent.resolve() != directory.resolve():
        raise ValueError("journal must belong to the recovery run")
    with queue_gate(directory, timeout=lock_timeout):
        queue = directory / "recovery-requests"
        queue.mkdir(exist_ok=True)
        atomic_json(
            queue / (uuid.uuid4().hex + ".json"),
            {
                "kind": "exit" if journal is None else "journal",
                "journal": None if journal is None else journal.name,
                "requested_at": time.time(),
            },
        )
        lease = track_lock((directory / "recovery-worker.lock").open("a"))
        try:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return  # The existing worker will observe the durable request.
            environment = {
                **os.environ,
                "XTUNER_SANDBOX_CLEANUP_DIR": str(directory),
                "XTUNER_SANDBOX_CLEANUP_FACTORY": factory,
            }
            with (directory / "recovery.log").open("a") as log:
                child = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--lease-fd", str(lease.fileno())],
                    pass_fds=(lease.fileno(),),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            # Reap the detached child if this caller remains alive; retaining
            # Popen in the waiting thread also avoids a live-child destructor.
            threading.Thread(target=child.wait, name="sandbox-recycler-reap", daemon=True).start()
        finally:
            # Close, never LOCK_UN: the recycler inherits this same kernel lease.
            lease.close()


@contextmanager
def queue_gate(directory: Path, *, timeout: float = 5) -> Iterator[None]:
    """Serialize enqueue with recycler exit, without an unbounded lock wait."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("queue lock timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    directory.mkdir(parents=True, exist_ok=True)
    with track_lock((directory / "recovery-queue.lock").open("a")) as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Sandbox recovery queue lock deadline exceeded") from None
                time.sleep(min(0.05, remaining))
        yield


def requests(directory: Path) -> list[Path]:
    """Return immutable pending request paths.

    Returns:
        list[Path]: Pending work; completed snapshots may be safely unlinked.
    """
    return sorted((directory / "recovery-requests").glob("*.json"))


def retry_delay(attempt: int, base: float, maximum: float) -> float:
    """Return capped exponential backoff with jitter in seconds.

    Returns:
        float: Delay before the next attempt.
    """
    return min(maximum, base * 2 ** (attempt - 1)) * random.uniform(0.8, 1.0)


def _release(directory: Path, lease: TextIO, *, only_if_empty: bool) -> bool:
    with queue_gate(directory):
        if only_if_empty and requests(directory):
            return False
        fcntl.flock(lease, fcntl.LOCK_UN)
        return True


def supervise_worker(lease_fd: int, budget: float) -> int:
    """Bound the entire worker, including filesystem I/O and its subprocesses.

    Keep the inherited lease until the worker group is reaped. The guardian
    performs no queue or status-file I/O, so those operations cannot stall its
    deadline. A later handoff can restart work retained after termination.
    """
    with track_lock(os.fdopen(lease_fd, "a")) as lease:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--lease-fd", str(lease.fileno()), "--run-worker"],
            pass_fds=(lease.fileno(),),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            return child.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            # Even stderr may be a log on the stalled shared filesystem.
            # Do not write status or logs from the deadline guardian.
            return 1
        finally:
            # A worker crash can leave a cleanup child alive. Terminate the
            # private group before releasing ownership to another recycler.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def main() -> int:
    """Retry only explicitly handed-off failures, with one process per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--lease-fd", type=int)
    action.add_argument("--enqueue", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--factory")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--handoff-timeout", type=float, default=5)
    parser.add_argument("--run-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_worker and args.lease_fd is None:
        parser.error("--run-worker requires --lease-fd")
    if args.enqueue:
        if args.directory is None or not args.factory:
            parser.error("--enqueue requires --directory and --factory")
        try:
            request_recovery(args.directory, args.factory, args.journal, lock_timeout=args.handoff_timeout)
        except Exception as exc:
            print(f"Sandbox recovery enqueue failed ({type(exc).__name__}); retain journals", file=sys.stderr)
            return 1
        return 0
    directory = Path(os.environ["XTUNER_SANDBOX_CLEANUP_DIR"])
    factory = os.environ["XTUNER_SANDBOX_CLEANUP_FACTORY"]
    policy = settings()
    if not args.run_worker:
        return supervise_worker(args.lease_fd, policy["budget"])
    deadline = time.monotonic() + policy["budget"]
    history: list[dict[str, Any]] = []
    with track_lock(os.fdopen(args.lease_fd, "a")) as lease:
        for attempt in range(1, policy["attempts"] + 1):
            if _release(directory, lease, only_if_empty=True):
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            timeout = min(policy["attempt_timeout"], remaining)
            started_at = time.time()
            status = "failed"
            try:
                child = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("cleanup.py")),
                        "--directory",
                        str(directory),
                        "--factory",
                        factory,
                        "--recover-queued",
                        "--concurrency",
                        str(policy["concurrency"]),
                        "--timeout",
                        str(min(10, timeout / 2)),
                        "--budget",
                        str(timeout * 0.9),
                    ],
                    timeout=timeout,
                    check=False,
                )
                status = "confirmed" if child.returncode == 0 else "pending"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except OSError:
                status = "launch_failed"
            delay = retry_delay(attempt, policy["backoff"], policy["max_backoff"])
            history.append(
                {
                    "attempt": attempt,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "next_delay_seconds": delay,
                }
            )
            atomic_json(
                directory / "recovery-status.json",
                {
                    "pid": os.getpid(),
                    "attempt": attempt,
                    "status": status,
                    "history": history,
                    "next_delay_seconds": delay,
                    "updated_at": time.time(),
                },
            )
            if _release(directory, lease, only_if_empty=True):
                return 0
            if attempt < policy["attempts"]:
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
        atomic_json(
            directory / "recovery-status.json",
            {
                "pid": os.getpid(),
                "status": "exhausted",
                "history": history,
                "updated_at": time.time(),
            },
        )
        _release(directory, lease, only_if_empty=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
