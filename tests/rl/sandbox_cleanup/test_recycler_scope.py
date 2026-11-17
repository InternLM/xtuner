"""Exercise exit scope sealing with real journals and bounded recovery workers."""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "xtuner/v1/rl/agent_loop/sandbox_agent_loop"
sys.path.insert(0, str(SOURCE))
import cleanup  # noqa: E402
import recycler  # noqa: E402


class Provider:
    def __init__(self, calls):
        self.calls = calls

    async def delete(self, env_id):
        self.calls.append(env_id)

    async def get(self, env_id):
        return {"status": "deleted"}

    async def aclose(self):
        pass


def _journal(directory, env_id, *, detach=True):
    journal = cleanup.CleanupJournal(directory, "scope-test")
    journal.allocated(env_id)
    if detach:
        journal.detach()
    return journal


def _request(directory, journal=None):
    queue = directory / "recovery-requests"
    queue.mkdir(exist_ok=True)
    request = queue / (uuid.uuid4().hex + ".json")
    recycler.atomic_json(
        request, {"kind": "exit" if journal is None else "journal", "journal": journal.name if journal else None}
    )
    return request


def _launch(directory, pending, *, seen=None, ancestors=None):
    with (
        cleanup.locking.open_lock(directory / "recovery-worker.lock") as lease,
        patch.object(recycler.subprocess, "Popen", return_value=SimpleNamespace(wait=lambda: 0)) as launch,
        patch.object(recycler.threading, "Thread"),
    ):
        recycler._launch_batch(
            directory, "scope_fixture:resolve", lease, pending, seen=seen or set(), seen_journals=ancestors
        )
    command = launch.call_args.args[0]
    return Path(command[command.index("--batch") + 1])


def _recover(directory, manifest, calls):
    return asyncio.run(cleanup.recover_queued(directory, lambda _: Provider(calls), 0.05, 1, 2, manifest))


def test_exit_scope_includes_journal_registered_between_enqueue_and_stop(tmp_path):
    request = _request(tmp_path)
    with patch.object(recycler, "journal_paths", side_effect=AssertionError("enqueue must not list journals")):
        manifest = _launch(tmp_path, [request])
    assert not cleanup.run_control.is_stopping(tmp_path)
    assert json.loads(manifest.read_text())["exit_scope"] == "pending"
    late = _journal(tmp_path, "late-orphan")
    calls = []
    assert _recover(tmp_path, manifest, calls)["ok"]
    assert calls == ["late-orphan"]
    assert not request.exists()
    assert json.loads(late.path.read_text())["resources"]["late-orphan"]["confirmed"]
    scope, owned = recycler.batch_journals(tmp_path, manifest)
    assert scope == [late.path]
    assert owned == {late.path.name}
    with pytest.raises(cleanup.run_control.RunStoppingError):
        cleanup.CleanupJournal(tmp_path, "scope-test")


def test_admitted_active_journal_retains_same_scope_until_late_allocation(tmp_path):
    manifest = _launch(tmp_path, [_request(tmp_path)])
    journal = cleanup.CleanupJournal(tmp_path, "scope-test")
    calls = []
    try:
        assert not _recover(tmp_path, manifest, calls)["ok"]
        sealed = manifest.read_bytes()
        assert recycler.batch_journals(tmp_path, manifest)[0] == [journal.path]
        assert calls == []
        # A create admitted before stop may report its ID afterwards; the
        # already selected journal remains owned while the original Pool lives.
        journal.allocated("completed-after-stop")
    finally:
        journal.detach()
    assert _recover(tmp_path, manifest, calls)["ok"]
    assert manifest.read_bytes() == sealed
    assert calls == ["completed-after-stop"]


def test_failed_stop_keeps_pending_scope_and_same_generation_can_retry(tmp_path):
    request = _request(tmp_path)
    manifest = _launch(tmp_path, [request])
    initial = manifest.read_bytes()
    calls = []
    with patch.object(cleanup.run_control, "request_stop", side_effect=OSError("stop unavailable")):
        with pytest.raises(OSError, match="stop unavailable"):
            _recover(tmp_path, manifest, calls)
    assert manifest.read_bytes() == initial
    assert not cleanup.run_control.is_stopping(tmp_path)
    late = _journal(tmp_path, "after-failed-stop")
    assert _recover(tmp_path, manifest, calls)["ok"]
    assert recycler.batch_journals(tmp_path, manifest)[0] == [late.path]
    assert calls == ["after-failed-stop"]


def test_atomic_seal_failure_precedes_delete_and_can_retry(tmp_path):
    request = _request(tmp_path)
    manifest = _launch(tmp_path, [request])
    _journal(tmp_path, "seal-retry")
    initial = manifest.read_bytes()
    calls = []
    with patch.object(recycler, "atomic_json", side_effect=OSError("seal unavailable")):
        with pytest.raises(OSError, match="seal unavailable"):
            _recover(tmp_path, manifest, calls)
    assert cleanup.run_control.is_stopping(tmp_path)
    assert manifest.read_bytes() == initial
    assert request.exists()
    assert calls == []
    assert _recover(tmp_path, manifest, calls)["ok"]
    assert calls == ["seal-retry"]


@pytest.mark.parametrize("crash_point", ["before_stop", "before_seal", "before_replace", "after_seal"])
def test_worker_crash_at_seal_preserves_queue_and_fixed_scope(tmp_path, crash_point):
    request = _request(tmp_path)
    manifest = _launch(tmp_path, [request])
    journal = _journal(tmp_path, "crash-orphan")
    script = """
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cleanup, recycler
manifest = Path(sys.argv[3])
crash_point = sys.argv[4]
if crash_point == 'before_stop':
    cleanup.run_control.request_stop = lambda *args: os._exit(73)
atomic = recycler.atomic_json
def crash(path, data):
    if path == manifest:
        if crash_point == 'before_replace':
            replace = os.replace
            def crash_replace(source, destination):
                if Path(destination) == manifest:
                    os._exit(73)
                replace(source, destination)
            os.replace = crash_replace
            atomic(path, data)
        if crash_point == 'after_seal':
            atomic(path, data)
        os._exit(73)
    atomic(path, data)
recycler.atomic_json = crash
asyncio.run(cleanup.recover_queued(Path(sys.argv[2]), lambda _: None, .05, 1, 2, manifest))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(SOURCE), str(tmp_path), str(manifest), crash_point],
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 73, result.stdout + result.stderr
    assert request.exists()
    assert cleanup.run_control.is_stopping(tmp_path) is (crash_point != "before_stop")
    after_seal = crash_point == "after_seal"
    assert json.loads(manifest.read_text())["exit_scope"] == ("sealed" if after_seal else "pending")
    selected = [journal.path]
    expected_calls = {"crash-orphan"}
    if crash_point == "before_stop":
        selected.append(_journal(tmp_path, "registered-after-crash").path)
        expected_calls.add("registered-after-crash")
    if crash_point == "before_replace":
        assert list(manifest.parent.glob(manifest.name + ".*.tmp"))
    if after_seal:
        # Simulate a later stray file on shared storage. Observation must retain
        # it as pending without giving sealed DELETE work a larger scope.
        stray = tmp_path / (uuid.uuid4().hex + ".json")
        data = json.loads(journal.path.read_text())
        data["resources"] = {"outside-seal": {"confirmed": False}}
        recycler.atomic_json(stray, data)
    calls = []
    report = _recover(tmp_path, manifest, calls)
    assert report["ok"] is not after_seal
    assert set(calls) == expected_calls
    assert len(calls) == len(expected_calls)
    assert recycler.batch_journals(tmp_path, manifest)[0] == sorted(selected)
    if after_seal:
        assert request.exists()
        assert not _recover(tmp_path, manifest, calls)["ok"]
        assert calls == ["crash-orphan"]


def test_successor_exit_never_renews_exhausted_or_missing_ancestor_work(tmp_path):
    old = _journal(tmp_path, "ancestor-exhausted")
    old_request = _request(tmp_path, old.path)
    first = _launch(tmp_path, [old_request])
    _, ancestors = recycler.batch_journals(tmp_path, first)
    missing = uuid.uuid4().hex + ".json"
    exit_request = _request(tmp_path)
    successor = _launch(tmp_path, [exit_request], seen={old_request.name}, ancestors=ancestors | {missing})
    # This resource appears after successor enqueue and must receive the new
    # budget, while both present and absent ancestor obligations stay excluded.
    fresh = _journal(tmp_path, "successor-window")
    calls = []
    assert not _recover(tmp_path, successor, calls)["ok"]
    assert calls == ["successor-window"]
    assert recycler.batch_journals(tmp_path, successor) == ([fresh.path], {old.path.name, missing, fresh.path.name})
    assert old_request.exists() and exit_request.exists()
    assert not _recover(tmp_path, successor, calls)["ok"]
    assert calls == ["successor-window"]


def test_unsealed_crash_cannot_retire_by_guessing_exit_ownership(tmp_path):
    request = _request(tmp_path)
    manifest = _launch(tmp_path, [request])
    fresh = _request(tmp_path, _journal(tmp_path, "fresh").path)
    with cleanup.locking.open_lock(tmp_path / "recovery-worker.lock") as lease:
        cleanup.locking.acquire(lease)
        with patch.object(recycler.subprocess, "Popen") as launch:
            with pytest.raises(ValueError, match="stop barrier"):
                recycler.retire_batch(tmp_path, "scope_fixture:resolve", lease, manifest)
            launch.assert_not_called()
        with cleanup.locking.open_lock(tmp_path / "recovery-worker.lock") as probe:
            with pytest.raises(BlockingIOError):
                cleanup.locking.acquire(probe)
    assert set(recycler.requests(tmp_path)) == {request, fresh}
    assert json.loads(manifest.read_text())["exit_scope"] == "pending"


@pytest.mark.parametrize("with_old_scope", [False, True])
def test_legacy_exit_scope_never_guesses_stop_order(tmp_path, with_old_scope):
    request = _request(tmp_path)
    journal = _journal(tmp_path, "legacy-orphan")
    manifest = _launch(tmp_path, [request])
    data = {"requests": [request.name], "seen": [request.name]}
    if with_old_scope:
        data.update(journals=[journal.path.name], seen_journals=[journal.path.name])
    recycler.atomic_json(manifest, data)
    cleanup.run_control.request_stop(tmp_path, "test")
    calls = []
    with pytest.raises(ValueError, match="Legacy exit batch"):
        _recover(tmp_path, manifest, calls)
    assert calls == []
    assert request.exists()
    assert json.loads(manifest.read_text()) == data


def test_damaged_request_does_not_block_valid_exit_scope(tmp_path):
    valid = _request(tmp_path)
    invalid = _request(tmp_path)
    invalid.write_text("broken")
    manifest = _launch(tmp_path, [invalid, valid])
    journal = _journal(tmp_path, "valid-exit")
    calls = []
    report = _recover(tmp_path, manifest, calls)
    assert not report["ok"]
    assert report["acknowledged"] == 1
    assert invalid.exists() and not valid.exists()
    assert calls == ["valid-exit"]
    assert recycler.batch_journals(tmp_path, manifest)[0] == [journal.path]


def test_stalled_exit_seal_remains_inside_existing_hard_budget(tmp_path):
    request = _request(tmp_path)
    manifest = _launch(tmp_path, [request])
    fixture = tmp_path / "scope_fixture.py"
    fixture.write_text(
        "import time, recycler\n"
        "from pathlib import Path\n"
        "def stall(directory):\n"
        "    (Path(directory) / 'seal-entered').touch()\n"
        "    time.sleep(60)\n"
        "recycler.journal_paths = stall\n"
        "def resolve(key):\n"
        "    raise AssertionError('seal must precede provider access')\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(SOURCE)]),
        "XTUNER_SANDBOX_CLEANUP_DIR": str(tmp_path),
        "XTUNER_SANDBOX_CLEANUP_FACTORY": "scope_fixture:resolve",
        "XTUNER_SANDBOX_RECOVERY_BUDGET": "4",
        "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "1.5",
        "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "1",
        "XTUNER_SANDBOX_RECOVERY_BACKOFF": "0.001",
    }
    started = time.monotonic()
    with cleanup.locking.open_lock(tmp_path / "recovery-worker.lock") as lease:
        cleanup.locking.acquire(lease)
        result = subprocess.run(
            [sys.executable, str(SOURCE / "recycler.py"), "--lease-fd", str(lease.fileno()), "--batch", str(manifest)],
            pass_fds=(lease.fileno(),),
            env=environment,
            timeout=8,
            capture_output=True,
            text=True,
        )
    assert result.returncode != 0
    assert time.monotonic() - started < 7
    assert (tmp_path / "seal-entered").exists(), result.stdout + result.stderr
    assert cleanup.run_control.is_stopping(tmp_path)
    assert request.exists()
    assert json.loads(manifest.read_text())["exit_scope"] == "pending"
    with cleanup.locking.open_lock(tmp_path / "recovery-worker.lock") as probe:
        cleanup.locking.acquire(probe)


class TestBatchRecovery:
    def test_all_exact_journals_are_queued_before_one_launch(self, tmp_path: Path) -> None:
        journals = [_journal(tmp_path, f"batch-{index}").path for index in range(8)]
        launched = []

        def capture(*args: Any, **kwargs: Any) -> None:
            launched.append(args[3])
            assert len(recycler.requests(tmp_path)) == len(journals)

        with patch.object(recycler, "_launch_batch", side_effect=capture) as launch:
            recycler.request_journal_recovery(tmp_path, "scope_fixture:resolve", journals)
        launch.assert_called_once()
        assert len(launched[0]) == 8
        assert {recycler.read_request(path) for path in recycler.requests(tmp_path)} == {
            ("journal", path.name) for path in journals
        }

    def test_invalid_path_rejects_entire_batch_before_queue_mutation(self, tmp_path: Path) -> None:
        good = tmp_path / ("a" * 32 + ".json")
        foreign = tmp_path.parent / ("b" * 32 + ".json")
        with patch.object(recycler, "_launch_batch") as launch:
            with pytest.raises(ValueError, match="belong"):
                recycler.request_journal_recovery(tmp_path, "scope_fixture:resolve", [good, foreign])
        launch.assert_not_called()
        assert not recycler.requests(tmp_path)
        assert not (tmp_path / "recovery-queue.lock").exists()

    def test_batch_notifies_active_chain_without_renewing_exhausted_ancestor(self, tmp_path: Path) -> None:
        ancestor = _journal(tmp_path, "exhausted")
        old = _request(tmp_path, ancestor.path)
        manifest = _launch(tmp_path, [old])
        fresh = [_journal(tmp_path, f"fresh-{index}").path for index in range(8)]
        with cleanup.locking.open_lock(tmp_path / "recovery-worker.lock") as lease:
            cleanup.locking.acquire(lease)
            with patch.object(recycler, "_launch_batch") as launch:
                recycler.request_journal_recovery(tmp_path, "scope_fixture:resolve", fresh)
            launch.assert_not_called()
            with patch.object(recycler, "_launch_batch") as launch:
                recycler.retire_batch(tmp_path, "scope_fixture:resolve", lease, manifest)
            launch.assert_called_once()
            selected = launch.call_args.args[3]
            assert old not in selected
            assert {recycler.read_request(path) for path in selected} == {("journal", path.name) for path in fresh}
            assert launch.call_args.kwargs["seen"] == {old.name}
            assert launch.call_args.kwargs["seen_journals"] == {ancestor.path.name}
        assert old.exists()
