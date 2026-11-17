"""Exercise lock ownership using independent processes and real kernel descriptors."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TextIO
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "xtuner/v1/rl/agent_loop/sandbox_agent_loop"
sys.path.insert(0, str(SOURCE))
import cleanup  # noqa: E402
import locking  # noqa: E402
import run_control  # noqa: E402


class TestRecordLocks:
    def test_same_process_fresh_probe_and_unrelated_close_preserve_owner(self, tmp_path: Path) -> None:
        path = tmp_path / "owner.lock"
        with locking.open_lock(path) as owner:
            locking.acquire(owner)
            with locking.open_lock(path) as unrelated:
                assert unrelated.fileno() != owner.fileno()
            for _ in range(2):
                with locking.open_lock(path) as probe:
                    with pytest.raises(BlockingIOError):
                        locking.acquire(probe)
        with locking.open_lock(path) as successor:
            locking.acquire(successor)

    def test_shared_admission_blocks_exclusive_fresh_process(self, tmp_path: Path) -> None:
        path = tmp_path / "admission.lock"
        with locking.open_lock(path) as first, locking.open_lock(path) as second:
            locking.acquire(first, shared=True)
            locking.acquire(second, shared=True)
            script = """
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import locking
with locking.open_lock(Path(sys.argv[2])) as lock:
    try:
        locking.acquire(lock)
    except BlockingIOError:
        sys.exit(17)
sys.exit(19)
"""
            result = subprocess.run([sys.executable, "-c", script, str(SOURCE), str(path)], timeout=10)
            assert result.returncode == 17

    def test_transferred_description_survives_parent_close_until_child_death(self, tmp_path: Path) -> None:
        path = tmp_path / "guardian.lock"
        owner = locking.open_lock(path)
        locking.acquire(owner)
        script = """
import os,sys
with os.fdopen(int(sys.argv[1]), 'a+') as inherited:
    print('ready', flush=True)
    sys.stdin.read()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(owner.fileno())],
            pass_fds=(owner.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline() == "ready\n"
            owner.close()
            with locking.open_lock(path) as probe:
                with pytest.raises(BlockingIOError):
                    locking.acquire(probe)
            child.kill()
            child.wait(timeout=5)
            with locking.open_lock(path) as probe:
                locking.acquire(probe)
        finally:
            owner.close()
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)

    def test_fork_child_closes_owner_copy_without_unlocking_parent(self, tmp_path: Path) -> None:
        path = tmp_path / "fork.lock"
        with run_control.track_lock(locking.open_lock(path)) as owner:
            locking.acquire(owner)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                os.write(write_fd, bytes([owner.closed]))
                os.close(write_fd)
                os._exit(0)
            os.close(write_fd)
            try:
                assert os.read(read_fd, 1) == b"\x01"
                _, status = os.waitpid(pid, 0)
                assert status == 0
                with locking.open_lock(path) as probe:
                    with pytest.raises(BlockingIOError):
                        locking.acquire(probe)
            finally:
                os.close(read_fd)

    def test_legacy_flock_namespace_is_rejected_before_provider_resolution(self, tmp_path: Path) -> None:
        path = tmp_path / ("1" * 32 + ".json")
        path.write_text("{}")
        called = []
        with path.with_suffix(".lock").open("a") as legacy:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(OSError, match="Unverified cleanup lock protocol"):
                asyncio.run(cleanup.reconcile_file(path, lambda key: called.append(key), 1))
        assert called == []

    def test_unsupported_ofd_never_falls_back_to_flock(self, tmp_path: Path) -> None:
        with locking.open_lock(tmp_path / "unsupported.lock") as lock:
            with (
                patch.object(locking.fcntl, "fcntl", side_effect=OSError(errno.EINVAL, "unsupported")),
                patch.object(locking.fcntl, "flock", side_effect=AssertionError("unsafe fallback")),
            ):
                with pytest.raises(OSError):
                    locking.acquire(lock)

    def test_ineffective_ofd_capability_check_releases_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "ineffective.lock"
        actual = locking.fcntl.fcntl

        def broken_query(fd: int, operation: int, argument: bytes) -> bytes | int:
            if operation == fcntl.F_GETLK:
                query = locking._RecordLock(fcntl.F_UNLCK, os.SEEK_SET, 0, 0, 0)
                return bytes(query)
            return actual(fd, operation, argument)

        with locking.open_lock(path) as lock:
            with patch.object(locking.fcntl, "fcntl", side_effect=broken_query):
                with pytest.raises(OSError, match="does not expose OFD"):
                    locking.acquire(lock)
            with locking.open_lock(path) as probe:
                locking.acquire(probe)

    def test_driver_probe_preserves_live_same_process_lease(self, tmp_path: Path) -> None:
        with run_control.claim_driver(tmp_path):
            assert run_control.driver_alive(tmp_path)
            assert run_control.driver_alive(tmp_path)
        assert not run_control.driver_alive(tmp_path)

    def test_live_pool_in_other_process_never_resolves_provider_then_crash_recovers(self, tmp_path: Path) -> None:
        script = """
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cleanup
journal = cleanup.CleanupJournal(Path(sys.argv[2]), 'record-lock-test')
journal.allocated('live-resource')
print(journal.path.name, flush=True)
sys.stdin.read()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(SOURCE), str(tmp_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            path = tmp_path / child.stdout.readline().strip()
            calls = []
            result = asyncio.run(cleanup.reconcile_file(path, lambda key: calls.append(key), 1))
            assert result["state"] == "active"
            assert calls == []
            child.kill()
            child.wait(timeout=5)

            class Provider:
                async def delete(self, env_id: str) -> None:
                    calls.append(env_id)

                async def get(self, env_id: str) -> dict[str, str]:
                    return {"status": "deleted"}

            result = asyncio.run(cleanup.reconcile_file(path, lambda key: Provider(), 1))
            assert result["ok"]
            assert calls == ["live-resource"]
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)

    def test_concurrent_same_process_journals_keep_independent_ownership(self, tmp_path: Path) -> None:
        first = cleanup.CleanupJournal(tmp_path, "record-lock-test")
        second = cleanup.CleanupJournal(tmp_path, "record-lock-test")
        first.allocated("first-resource")
        second.allocated("second-resource")
        calls = []
        try:
            for journal in (first, second):
                result = asyncio.run(cleanup.reconcile_file(journal.path, lambda key: calls.append(key), 1))
                assert result["state"] == "active"
            first.detach()
            result = asyncio.run(cleanup.reconcile_file(second.path, lambda key: calls.append(key), 1))
            assert result["state"] == "active"
            assert calls == []
        finally:
            first.detach()
            second.detach()

    def test_journal_acquire_failure_closes_descriptor_before_allocation(self, tmp_path: Path) -> None:
        actual = locking.open_lock
        opened = []

        def observe_open(path: Path) -> TextIO:
            result = actual(path)
            opened.append(result)
            return result

        actual_acquire = locking.acquire

        def unsupported_journal(lock: TextIO, **kwargs: Any) -> None:
            if Path(os.readlink(f"/proc/self/fd/{lock.fileno()}")).name != "admission.lock":
                raise OSError(errno.EINVAL, "unsupported journal lock")
            return actual_acquire(lock, **kwargs)

        with (
            patch.object(locking, "open_lock", side_effect=observe_open),
            patch.object(locking, "acquire", side_effect=unsupported_journal),
        ):
            with pytest.raises(OSError):
                cleanup.CleanupJournal(tmp_path, "record-lock-test")
        assert opened and all(stream.closed for stream in opened)
        assert not list(tmp_path.glob("*.json"))

    def test_concurrent_first_open_publishes_one_initialized_inode(self, tmp_path: Path) -> None:
        path = tmp_path / "first-open.lock"
        barrier = threading.Barrier(8)

        def open_together(_: int) -> tuple[int, int]:
            barrier.wait(timeout=5)
            with locking.open_lock(path) as lock:
                locking.acquire(lock, shared=True)
                metadata = os.fstat(lock.fileno())
                return metadata.st_dev, metadata.st_ino

        with ThreadPoolExecutor(max_workers=8) as workers:
            identities = list(workers.map(open_together, range(8)))
        assert len(set(identities)) == 1
        assert not list(tmp_path.glob("*.tmp"))

    @pytest.mark.parametrize("code", [errno.EACCES, errno.EAGAIN])
    def test_record_lock_contention_reports_active_for_both_kernel_errnos(self, tmp_path: Path, code: int) -> None:
        with locking.open_lock(tmp_path / "busy.lock") as lock:
            with patch.object(locking.fcntl, "fcntl", side_effect=OSError(code, "busy")):
                with pytest.raises(BlockingIOError):
                    locking.acquire(lock)

    def test_shared_ofd_accepts_existing_posix_reader_and_blocks_exclusive(self, tmp_path: Path) -> None:
        path = tmp_path / "shared-posix-reader.lock"
        with locking.open_lock(path):
            pass
        script = """
import fcntl,sys
with open(sys.argv[1], 'r+') as reader:
    fcntl.lockf(reader, fcntl.LOCK_SH)
    print('ready', flush=True)
    sys.stdin.read()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline() == "ready\n"
            actual = locking.fcntl.fcntl
            observed = []

            def record_query(fd: int, command: int, argument: bytes) -> bytes | int:
                result = actual(fd, command, argument)
                if command == fcntl.F_GETLK:
                    observed.append(locking._RecordLock.from_buffer_copy(result).pid)
                return result

            with locking.open_lock(path) as shared:
                with patch.object(locking.fcntl, "fcntl", side_effect=record_query):
                    locking.acquire(shared, shared=True)
                assert child.pid in observed
                with locking.open_lock(path) as exclusive:
                    with pytest.raises(BlockingIOError):
                        locking.acquire(exclusive)
                child.kill()
                child.wait(timeout=5)
                with locking.open_lock(path) as exclusive:
                    with pytest.raises(BlockingIOError):
                        locking.acquire(exclusive)
            with locking.open_lock(path) as exclusive:
                locking.acquire(exclusive)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
