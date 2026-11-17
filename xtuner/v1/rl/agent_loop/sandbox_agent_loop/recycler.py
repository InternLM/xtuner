"""Durable failure handoff to one bounded, backoff-driven recycler per run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


if __package__:
    from . import locking
    from .run_control import atomic_json, is_stopping, track_lock
else:
    import locking
    from run_control import atomic_json, is_stopping, track_lock


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
                sys.executable,
                str(Path(__file__).resolve()),
                "--enqueue",
                "--directory",
                str(directory),
                "--factory",
                factory,
                "--handoff-timeout",
                str(timeout * 0.9),
                *(["--journal", str(journal)] if journal is not None else []),
            ],
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def request_recovery(directory: Path, factory: str, journal: Path | None = None, *, lock_timeout: float = 5) -> None:
    """Persist failed work before starting or notifying the run's recycler.

    Args:
        directory (Path): Exact run-owned directory.
        factory (str): Trusted provider factory import path.
        journal (Path | None): Failed Pool journal, or the entire stopped run.
    """
    _enqueue_recovery(directory, factory, [journal], lock_timeout=lock_timeout)


def request_journal_recovery(directory: Path, factory: str, journals: list[Path], *, lock_timeout: float = 5) -> None:
    """Durably enqueue an exact trial scope before starting one recovery chain.

    All paths are validated before any queue mutation. Individual requests keep
    the existing schema and ancestor budget rules; one queue gate covers the
    entire batch, so a worker cannot retire between its registrations.

    Args:
        directory (Path): Exact run-owned directory.
        factory (str): Trusted provider factory import path.
        journals (list[Path]): Proven journals belonging to the failed trial.
        lock_timeout (float): Existing queue-gate wait budget in seconds.
    """
    _enqueue_recovery(directory, factory, list(dict.fromkeys(journals)), lock_timeout=lock_timeout)


def _enqueue_recovery(directory: Path, factory: str, journals: Sequence[Path | None], *, lock_timeout: float) -> None:
    settings()  # Reject bad retry policy before spawning.
    if not factory or not directory.is_absolute():
        raise ValueError("recovery requires a factory and absolute run directory")
    if any(journal is not None and journal.parent.resolve() != directory.resolve() for journal in journals):
        raise ValueError("journal must belong to the recovery run")
    if not journals:
        return
    with queue_gate(directory, timeout=lock_timeout):
        queue = directory / "recovery-requests"
        queue.mkdir(exist_ok=True)
        for journal in journals:
            atomic_json(
                queue / (uuid.uuid4().hex + ".json"),
                {
                    "kind": "exit" if journal is None else "journal",
                    "journal": None if journal is None else journal.name,
                    "requested_at": time.time(),
                },
            )
        lease = track_lock(locking.open_lock(directory / "recovery-worker.lock"))
        try:
            try:
                locking.acquire(lease)
            except BlockingIOError:
                return  # Retirement transfers requests outside the active batch.
            _launch_batch(directory, factory, lease, requests(directory), seen=set())
        finally:
            # Close, never LOCK_UN: the recycler inherits this same kernel lease.
            lease.close()


def _launch_batch(
    directory: Path,
    factory: str,
    lease: TextIO,
    pending: list[Path],
    *,
    seen: set[str],
    seen_journals: set[str] | None = None,
) -> None:
    """Freeze requests and ancestor ownership, deferring exit scope until stop."""
    names = {path.name for path in pending}
    owned = set() if seen_journals is None else seen_journals
    journals: set[str] = set()
    has_exit = False
    for path in pending:
        try:
            kind, name = read_request(path)
        except (OSError, ValueError):
            continue  # The worker reports invalid requests without acknowledging them.
        if kind == "exit":
            has_exit = True
        else:
            assert name is not None
            journals.add(name)
    journals.difference_update(owned)
    batches = directory / "recovery-batches"
    batches.mkdir(exist_ok=True)
    manifest = batches / (uuid.uuid4().hex + ".json")
    atomic_json(
        manifest,
        {
            "requests": sorted(names),
            "seen": sorted(seen | names),
            "journals": sorted(journals),
            "seen_journals": sorted(owned | journals),
            "exit_scope": "pending" if has_exit else "sealed",
        },
    )
    environment = {
        **os.environ,
        "XTUNER_SANDBOX_CLEANUP_DIR": str(directory),
        "XTUNER_SANDBOX_CLEANUP_FACTORY": factory,
    }
    with (directory / "recovery.log").open("a") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--lease-fd",
                str(lease.fileno()),
                "--batch",
                str(manifest),
            ],
            pass_fds=(lease.fileno(),),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    threading.Thread(target=child.wait, name="sandbox-recycler-reap", daemon=True).start()


def batch_requests(directory: Path, manifest: Path) -> tuple[list[Path], set[str]]:
    """Read the immutable batch; reject paths outside the run's request queue."""
    if manifest.parent.resolve() != (directory / "recovery-batches").resolve():
        raise ValueError("recovery batch must belong to the run")
    data = json.loads(manifest.read_text())
    for field in ("requests", "seen"):
        names = data.get(field) if isinstance(data, dict) else None
        if not isinstance(names, list) or any(
            not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") for name in names
        ):
            raise ValueError("invalid recovery batch")
    if not set(data["requests"]) <= set(data["seen"]):
        raise ValueError("batch requests must have an owner")
    queue = directory / "recovery-requests"
    return [queue / name for name in data["requests"] if (queue / name).exists()], set(data["seen"])


def journal_paths(directory: Path) -> list[Path]:
    """Select exact run-owned journal names, excluding reports and configuration."""
    return sorted(path for path in directory.glob("*.json") if _journal_name(path.name))


def _journal_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and len(name) == 37
        and name.endswith(".json")
        and all(character in "0123456789abcdef" for character in name[:-5])
    )


def read_request(path: Path) -> tuple[str, str | None]:
    """Validate one immutable request without expanding its deletion scope."""
    row = json.loads(path.read_text())
    if not isinstance(row, dict):
        raise ValueError("request must be an object")
    kind, name = row.get("kind"), row.get("journal")
    if kind == "exit" and name is None:
        return kind, None
    if kind == "journal" and _journal_name(name):
        return kind, name
    raise ValueError("invalid recovery request")


def batch_journals(directory: Path, manifest: Path, *, seal_exit: bool = False) -> tuple[list[Path], set[str]]:
    """Return frozen deletion work and cumulative budget ownership, never completion.

    Legacy journal-only batches can be upgraded from their immutable requests.
    Missing ancestor requests or exit requests lack an exact historical scope;
    retain those batches for an explicit idle handoff instead of renewing work.
    A new exit batch is sealed once by its bounded cleanup worker after stop;
    retirement cannot guess its scope if that worker dies before sealing.
    """
    batch_requests(directory, manifest)  # Validate the manifest location and request names.
    data = json.loads(manifest.read_text())
    if "journals" not in data or "seen_journals" not in data:
        current: set[str] = set()
        ancestors: set[str] = set()
        for name in data["seen"]:
            kind, journal = read_request(directory / "recovery-requests" / name)
            if kind != "journal":
                raise ValueError("Legacy exit batch requires an explicit recovery handoff")
            assert journal is not None
            (current if name in data["requests"] else ancestors).add(journal)
        data.update(
            journals=sorted(current - ancestors), seen_journals=sorted(current | ancestors), exit_scope="sealed"
        )
        atomic_json(manifest, data)
    for field in ("journals", "seen_journals"):
        if not isinstance(data[field], list) or any(not _journal_name(name) for name in data[field]):
            raise ValueError("invalid batch journal ownership")
    if not set(data["journals"]) <= set(data["seen_journals"]):
        raise ValueError("batch journals must have a budget owner")
    if "exit_scope" not in data:
        # Older snapshots froze exit work at enqueue, before the stop barrier.
        # A stop marker now cannot prove their historical listing was complete.
        for name in data["requests"]:
            if read_request(directory / "recovery-requests" / name)[0] != "journal":
                raise ValueError("Legacy exit batch requires an explicit recovery handoff")
        data["exit_scope"] = "sealed"
        atomic_json(manifest, data)
    scope = data["exit_scope"]
    if scope not in {"pending", "sealed"}:
        raise ValueError("invalid exit scope state")
    if scope == "pending":
        if not seal_exit or not is_stopping(directory):
            raise ValueError("exit batch scope requires a successful stop barrier")
        queued, _ = batch_requests(directory, manifest)
        has_exit = False
        for path in queued:
            try:
                has_exit |= read_request(path)[0] == "exit"
            except (OSError, ValueError):
                continue  # Keep damaged requests without blocking valid exit work.
        if not has_exit:
            raise ValueError("pending exit scope requires an exit request")
        # request_stop serialized registration before this listing. Existing
        # admitted creates keep their journal lease until the final ID is saved.
        # Ancestor ownership is cumulative: exit must never renew its budget.
        known = set(data["seen_journals"])
        fresh = {path.name for path in journal_paths(directory)} - known
        data.update(
            journals=sorted(set(data["journals"]) | fresh),
            seen_journals=sorted(known | fresh),
            exit_scope="sealed",
        )
        atomic_json(manifest, data)  # Persist the exact scope before any DELETE.
    return [directory / name for name in data["journals"]], set(data["seen_journals"])


def retire_batch(directory: Path, factory: str, lease: TextIO, manifest: Path) -> None:
    """Atomically transfer only new requests, or release the idle run lease.

    Earlier exhausted batches stay durable without receiving another budget.
    A later explicit handoff, after this chain is idle, may retry them again.
    """
    with queue_gate(directory):
        _, seen = batch_requests(directory, manifest)
        _, seen_journals = batch_journals(directory, manifest)
        fresh = [path for path in requests(directory) if path.name not in seen]
        if fresh:
            _launch_batch(directory, factory, lease, fresh, seen=seen, seen_journals=seen_journals)
            # The new guardian inherits the same open file description. Never
            # LOCK_UN here, even though the old guardian still holds its copy.
        else:
            # The old worker group has already been reaped. Release under the
            # queue gate so a concurrent enqueue either transfers or launches.
            locking.release(lease)


@contextmanager
def queue_gate(directory: Path, *, timeout: float = 5) -> Iterator[None]:
    """Serialize enqueue with recycler exit, without an unbounded lock wait."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("queue lock timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    directory.mkdir(parents=True, exist_ok=True)
    with track_lock(locking.open_lock(directory / "recovery-queue.lock")) as lock:
        while True:
            try:
                locking.acquire(lock)
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


def supervise_worker(lease_fd: int, budget: float, manifest: Path) -> int:
    """Bound the entire worker, including filesystem I/O and its subprocesses.

    Keep the inherited lease until the worker group is reaped. The guardian
    performs no queue or status-file I/O. Reserve part of this generation's
    budget for a killable retirement child, including after worker timeout.
    """
    with track_lock(os.fdopen(lease_fd, "a")) as lease:
        deadline = time.monotonic() + budget
        retirement_budget = min(5.0, budget / 2)
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--lease-fd",
                str(lease.fileno()),
                "--batch",
                str(manifest),
                "--run-worker",
            ],
            pass_fds=(lease.fileno(),),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            result = child.wait(timeout=max(0.001, deadline - time.monotonic() - retirement_budget))
        except subprocess.TimeoutExpired:
            # Even stderr may be a log on the stalled shared filesystem.
            # Do not write status or logs from the deadline guardian.
            result = 1
        finally:
            # A worker crash can leave a cleanup child alive. Terminate the
            # private group before releasing ownership to another recycler.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        # Retirement shares the lease, but has a different process group. A
        # successor starts its own session, so neither old timeout can kill it.
        retire = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--lease-fd",
                str(lease.fileno()),
                "--batch",
                str(manifest),
                "--retire",
            ],
            pass_fds=(lease.fileno(),),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            if retire.wait(timeout=max(0.001, deadline - time.monotonic())) != 0:
                result = 1
        except subprocess.TimeoutExpired:
            # A stalled filesystem cannot be safely handed over within budget.
            # Retain every request for a later explicit or external recovery.
            result = 1
        finally:
            try:
                os.killpg(retire.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            retire.wait()
        return result


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
    parser.add_argument("--retire", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_worker and args.lease_fd is None:
        parser.error("--run-worker requires --lease-fd")
    if args.lease_fd is not None and args.batch is None:
        parser.error("--lease-fd requires --batch")
    if args.retire and (args.lease_fd is None or args.run_worker):
        parser.error("--retire requires --lease-fd without --run-worker")
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
    if args.retire:
        with track_lock(os.fdopen(args.lease_fd, "a")) as lease:
            retire_batch(directory, factory, lease, args.batch)
        return 0
    if not args.run_worker:
        return supervise_worker(args.lease_fd, policy["budget"], args.batch)
    return run_batch(directory, factory, args.lease_fd, args.batch, policy)


def run_batch(directory: Path, factory: str, lease_fd: int, manifest: Path, policy: dict[str, Any]) -> int:
    """Retry a fixed request generation; the guardian owns retirement and handoff."""
    deadline = time.monotonic() + policy["budget"]
    history: list[dict[str, Any]] = []
    with track_lock(os.fdopen(lease_fd, "a")):
        for attempt in range(1, policy["attempts"] + 1):
            if not batch_requests(directory, manifest)[0]:
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
                        "--request-batch",
                        str(manifest),
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
            if not batch_requests(directory, manifest)[0]:
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
