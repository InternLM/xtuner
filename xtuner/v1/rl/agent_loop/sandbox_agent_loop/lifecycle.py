"""Ownership, phase release and final recovery for one sandbox Pool.

Dependency planning and task preparation belong to SandboxPool. This module
owns the lifecycle protocol for the resulting resources, including nonpersistent
operation; callers never inspect journal rows or choose a recovery strategy.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lagent.utils import create_object

from .cleanup import CleanupJournal, defer_cleanup, delete_confirmed
from .finalization import CleanupError, finalize_bounded
from .run_control import RunStoppingError, is_stopping
from .task_control import watch_task
from .trial_process import validate_trial_cleanup


if TYPE_CHECKING:
    from lagent.serving.sandbox.providers.base import SandboxProvider


@dataclass
class SandboxLease:
    """One allocated member in a dependency group, in acquisition order."""

    name: str
    client: Any
    env_id: str


# A capability for one named member: the Lifecycle registers ownership before
# the Pool can await health checks or provisioners.
CreateSandbox = Callable[[str, Callable[[], Awaitable[dict[str, Any]]]], Awaitable[SandboxLease]]


@dataclass
class _Finalization:
    release_attempted: bool = False
    sealed: bool = False
    handed_off: bool = False
    completed: bool = False
    release_error: tuple[str, str] | None = None


class SandboxLifecycle:
    """Own a trial's resources until deletion or durable recovery handoff.

    The Pool supplies dependency plans and may specialize phase-release checks.
    This object exclusively owns confirmation, journal, cancellation budgets,
    provider ownership and recovery policy. One object serves both persistent
    and in-process-only modes.
    """

    def __init__(
        self,
        provider: Any,
        *,
        cleanup_timeout_sec: float = 120,
        close_provider: bool | None = None,
        verify_deletion: bool | None = None,
        cleanup_dir: str | None = None,
        cleanup_provider_key: str | None = None,
        cleanup_provider: dict[str, Any] | None = None,
        cleanup_retry_timeout_sec: float = 60,
    ) -> None:
        if not math.isfinite(cleanup_timeout_sec) or cleanup_timeout_sec <= 0:
            raise ValueError("cleanup_timeout_sec must be finite and positive")
        self._cleanup_timeout_sec = cleanup_timeout_sec
        # A passed instance may serve other pools. Only a dedicated provider
        # may donate orphaned creates or be closed by this pool.
        self._close_provider = isinstance(provider, dict) if close_provider is None else close_provider
        if not math.isfinite(cleanup_retry_timeout_sec) or cleanup_retry_timeout_sec <= 0:
            raise ValueError("cleanup retry timeout must be finite and positive")
        self._cleanup_retry_timeout_sec = cleanup_retry_timeout_sec
        if cleanup_provider is not None and not isinstance(cleanup_provider, dict):
            raise TypeError("cleanup_provider must be a reconstructible provider config")
        recovery_config = cleanup_provider if cleanup_provider is not None else provider
        self._recovery_config = deepcopy(recovery_config) if isinstance(recovery_config, dict) else None
        run_directory = os.environ.get("XTUNER_SANDBOX_CLEANUP_DIR")
        if cleanup_dir and run_directory and Path(cleanup_dir).resolve() != Path(run_directory).resolve():
            raise ValueError("Pool cleanup_dir must match the driver's run directory")
        self._journal_dir = cleanup_dir or run_directory
        validate_trial_cleanup(self._journal_dir)
        self._provider_key = cleanup_provider_key or os.environ.get("XTUNER_SANDBOX_PROVIDER_KEY")
        if self._journal_dir and not self._provider_key:
            raise ValueError("cleanup_dir requires cleanup_provider_key for exact backend recovery")
        self._journal: CleanupJournal | None = None
        self._cleanup_trial_id: str | None = None
        self._known_ids: dict[str, None] = {}
        self._unclaimed_creates: dict[str, Any] = {}
        self._release_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._closing = False
        self._deferred_cleanup: asyncio.Task | None = None
        self._cleanup: dict[str, Any] = {"ok": False, "error_kind": None}
        self._release_complete = False
        self._provider_closed = False
        self._finalization = _Finalization()
        self._cleanup_clients: list[Any] = []
        self._pending_delete_ids: set[str] = set()
        self._pending_verification_ids: set[str] = set()
        self._releasing = False
        self._phase_delete_deadline: float | None = None
        self._provider: SandboxProvider[Any] = create_object(provider)
        query_supported = callable(getattr(self._provider, "get", None))
        self._verify_deletion = (
            getattr(self._provider, "supports_deletion_confirmation", False) is True
            if verify_deletion is None
            else verify_deletion
        )
        if self._verify_deletion and not query_supported:
            raise ValueError("verify_deletion requires provider.get")
        if self._journal_dir and not self._verify_deletion:
            raise ValueError("persistent cleanup requires deletion verification")
        self._clients: dict[str, Any] = {}
        self._env_ids: dict[str, str] = {}
        self._urls: dict[str, str | None] = {}
        self._groups: dict[str, list[SandboxLease]] = {}
        self._name_locks: dict[str, asyncio.Lock] = {}
        self._active_acquisitions: dict[asyncio.Future[None], asyncio.Task[Any]] = {}

    async def acquire_group(
        self,
        name: str,
        acquire: Callable[[CreateSandbox, int], Awaitable[None]],
        *,
        max_attempts: int,
    ) -> Any:
        """Commit one ready group, or roll back before considering another attempt.

        The callback plans/prepares members and checks application readiness.
        Its create capability registers every lease before returning it. The
        lifecycle alone decides whether a failed transaction is safe to replay.
        Calls for one name share a committed group; different names may acquire
        concurrently. Waiting calls remain part of the release/close drain.
        """
        self._assert_acquisition_open()
        task = asyncio.current_task()
        assert task is not None
        completed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_acquisitions[completed] = task
        try:
            # Retain locks for the Pool's named plans: replacing a lock while
            # it still has waiters would allow two owners of the same name.
            async with self._name_locks.setdefault(name, asyncio.Lock()):
                self._assert_acquisition_open()
                if self._pending_delete_ids or self._pending_verification_ids:
                    raise RuntimeError("Previous sandbox cleanup is still pending")
                if name in self._clients:
                    return self._clients[name]
                if self._journal_dir:
                    await self._ensure_journal()
                self._assert_acquisition_open()
                return await self._acquire_group(name, acquire, max_attempts)
        finally:
            self._active_acquisitions.pop(completed)
            completed.set_result(None)

    def _assert_acquisition_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("SandboxPool is closed")
        if self._releasing:
            raise RuntimeError("Previous sandbox cleanup is still pending")

    async def _acquire_group(
        self, name: str, acquire: Callable[[CreateSandbox, int], Awaitable[None]], max_attempts: int
    ) -> Any:
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            self._assert_acquisition_open()
            group: list[SandboxLease] = []

            async def create_member(
                member_name: str, prepare: Callable[[], Awaitable[dict[str, Any]]]
            ) -> SandboxLease:
                self._assert_acquisition_open()
                client, env_id = await self._create(prepare)
                lease = SandboxLease(name=member_name, client=client, env_id=env_id)
                group.append(lease)
                self._assert_acquisition_open()
                return lease

            try:
                await acquire(create_member, attempt)
                self._assert_acquisition_open()
                primary = group[-1]
                self._groups[name] = group
                self._clients[name] = primary.client
                self._env_ids[name] = primary.env_id
                self._urls[name] = self._url_of(primary.client)
                return primary.client
            except asyncio.CancelledError:
                await self._rollback_group(name, group)
                raise
            except Exception as exc:
                last_err = exc
                await self._rollback_group(name, group)
                if not self._can_retry_acquisition(exc):
                    raise
                if attempt < max_attempts:
                    await asyncio.sleep(min(2**attempt, 8))

        last_err_msg = (
            "".join(traceback.format_exception(type(last_err), last_err, last_err.__traceback__)).rstrip()
            if last_err is not None
            else "unknown"
        )
        raise RuntimeError(f"could not acquire sandbox group after {max_attempts} attempts: {last_err_msg}")

    @property
    def cleanup(self) -> dict[str, Any]:
        """Return an independent diagnostic snapshot, never lifecycle authority."""
        return deepcopy(self._cleanup)

    @property
    def pending_resource_ids(self) -> frozenset[str]:
        return frozenset(self._pending_delete_ids | self._pending_verification_ids | self._owned_orphan_ids())

    def _can_retry_acquisition(self, error: Exception) -> bool:
        """Never replay classified allocations or accumulate unresolved ownership."""
        return getattr(error, "allocation_state", None) is None and not (
            self._closed
            or self._closing
            or self._unclaimed_creates
            or self._pending_delete_ids
            or self._pending_verification_ids
            or self._cleanup_clients
        )

    @asynccontextmanager
    async def task_scope(self, trial_id: str, close_pool: Callable[[], Awaitable[None]]) -> AsyncIterator[None]:
        async with watch_task(self._journal_dir) as stopped:
            try:
                self.set_cleanup_context(trial_id)
                yield
            except BaseException:
                # Only an exception escaping this task body is primary. An
                # unrelated caller's handled exception must not hide cleanup.
                try:
                    await close_pool()
                except CleanupError:
                    pass  # The cleanup snapshot retains the secondary failure.
                raise
            else:
                await close_pool()
            if stopped.is_set():
                raise asyncio.CancelledError("sandbox_run_stopping")

    def peek(self, name: str) -> Any | None:
        return self._clients.get(name)

    async def _ensure_journal(self) -> None:
        # Share final-close serialization: initialization must not publish a
        # second journal or adopt an owner after finalization has detached it.
        async with self._close_lock:
            if self._closed or self._closing:
                raise RuntimeError("SandboxPool is closed")
            if self._journal is None:
                assert self._provider_key is not None and self._journal_dir is not None
                self._journal = await CleanupJournal.open(
                    self._journal_dir, self._provider_key, timeout_seconds=min(5, self._cleanup_timeout_sec)
                )
                self._cleanup["journal"] = str(self._journal.path)
                if self._cleanup_trial_id is not None:
                    await self._journal.persist(self._journal.identify, self._cleanup_trial_id)
            if self._closing:
                raise RuntimeError("SandboxPool is closed")

    def set_cleanup_context(self, trial_id: str) -> None:
        """Associate the journal with a rollout UID; never serialize task content."""
        if self._journal is not None:
            raise RuntimeError("Cleanup context must be set before acquisition")
        self._cleanup_trial_id = trial_id

    def env_id(self, name: str) -> str | None:
        return self._env_ids.get(name)

    def url(self, name: str) -> str | None:
        return self._urls.get(name)

    async def release_all(self) -> None:
        """Release current resources without closing the provider between phases."""
        if self._closed and (self._journal is not None or self._deferred_cleanup is not None):
            raise RuntimeError("Cleanup ownership has been transferred; use the run reconciler")

        async def serialized() -> None:
            async with self._release_lock:
                await self._release_all()

        await finalize_bounded(
            serialized(),
            timeout_seconds=self._cleanup_timeout_sec * 1.1,
            diagnostic=self._cleanup.setdefault("release_wait", {}),
        )

    async def _release_all(self) -> None:
        if (
            self._release_complete
            and not self._clients
            and not self._pending_delete_ids
            and not self._pending_verification_ids
            and not self._owned_orphan_ids()
            and not self._unclaimed_creates
            and not self._cleanup_clients
            and not self._active_acquisitions
        ):
            return
        self._releasing = True
        self._release_complete = False
        self._cleanup.update(ok=False, error_kind=None, resources_deleted=False, clients_closed=False)
        seconds = self._cleanup_timeout_sec
        started = time.monotonic()
        # First deletion pass, retry pass and final observation share the outer
        # deadline, leaving a final slice for closing HTTP connections.
        delete_deadline = started + seconds * 0.4
        # Only rollbacks caused by this release's drain share its deadline.
        # Never let a later acquisition inherit a completed phase's budget.
        self._phase_delete_deadline = delete_deadline

        try:
            await finalize_bounded(
                self._release_resources(delete_deadline, started + seconds * 0.8),
                timeout_seconds=seconds,
                diagnostic=self._cleanup,
            )
            self._release_complete = True
        finally:
            self._phase_delete_deadline = None
            self._cleanup["unverified_env_ids"] = sorted(self._pending_verification_ids)
            self._cleanup["pending_env_ids"] = sorted(self._pending_delete_ids)
            # A cancellation-resistant create still owns its journal/transport.
            # Keep admission stopped until a later release drains it completely.
            self._releasing = bool(self._active_acquisitions)

    async def _release_resources(self, delete_deadline: float, retry_deadline: float) -> None:
        self._cleanup["phase"] = "drain_acquisitions"
        await self._drain_acquisitions()
        await self._record_phase_ownership()
        try:
            self._cleanup["phase"] = "delete_sandboxes"
            await self._release_groups(delete_deadline)
            self._cleanup["phase"] = "retry_deletions"
            await self._delete_pass(self._reverse_acquisition_order(set(self.pending_resource_ids)), retry_deadline)
            if self._pending_delete_ids or self._pending_verification_ids:
                raise RuntimeError("Sandbox deletion not confirmed")
            self._cleanup["resources_deleted"] = True
        except Exception:
            self._cleanup["failure_phase"] = self._cleanup["phase"]
            raise
        finally:
            await self._close_phase_clients()

    async def _drain_acquisitions(self) -> None:
        # The caller owns the release budget. wait() lets its cancellation stop
        # waiting without detaching ownership or implicitly waiting forever for
        # a provider which ignores cancellation. Acquisition finally removes
        # each task only after its rollback has finished.
        current = asyncio.current_task()
        active = {done: task for done, task in self._active_acquisitions.items() if task is not current}
        for task in set(active.values()):
            task.cancel()
        if active:
            # Completion belongs to get(), not its caller task: a caller may
            # catch cancellation and continue unrelated work after get unwinds.
            await asyncio.wait(active)

    async def _record_phase_ownership(self) -> None:
        for env_id, client in self._unclaimed_creates.items():
            self._pending_delete_ids.add(env_id)
            self._cleanup_clients.append(client)
        self._unclaimed_creates.clear()
        owned_ids = set(self.pending_resource_ids)
        owned_ids.update(lease.env_id for group in self._groups.values() for lease in group)
        for env_id in owned_ids:
            await self._remember_created(env_id)
        if self._verify_deletion:
            # Track confirmation separately from DELETE acceptance before any
            # release pass can remove the original lease or provider ownership.
            self._pending_verification_ids.update(owned_ids)

    async def _close_phase_clients(self) -> None:
        self._cleanup["phase"] = "close_clients"
        clients = list(self._cleanup_clients)
        self._cleanup["provider_closed"] = self._provider_closed
        outcomes = await asyncio.gather(
            *(self._close_phase_client(client) for client in clients), return_exceptions=True
        )
        self._cleanup["clients_closed"] = not any(isinstance(value, BaseException) for value in outcomes)
        if not self._cleanup["clients_closed"]:
            raise RuntimeError("Sandbox HTTP client closure not confirmed")

    async def _close_phase_client(self, client: Any) -> None:
        await client.aclose()
        # Record each completion immediately, even if another close stalls.
        self._cleanup_clients.remove(client)
        self._cleanup["closed_client_count"] = self._cleanup.get("closed_client_count", 0) + 1

    async def _remember_created(self, env_id: str) -> None:
        # Memory ownership precedes disk I/O so a failed write still rolls back.
        self._known_ids[env_id] = None
        if self._journal is not None:
            await self._journal.remember_created(env_id)

    async def finish(self, release_phase: Callable[[], Awaitable[None]]) -> None:
        """Bound final release, record sealing, recovery handoff and transport close."""
        self._closing = True
        try:
            await finalize_bounded(
                self._finish_exclusive(release_phase),
                timeout_seconds=self._cleanup_timeout_sec * 1.2 + 6,
                diagnostic=self._cleanup.setdefault("finalize", {}),
            )
        except CleanupError:
            self._cleanup["ok"] = False
            raise

    async def _finish_exclusive(self, release_phase: Callable[[], Awaitable[None]]) -> None:
        async with self._close_lock:
            state = self._finalization
            if state.completed:
                if state.release_error is not None:
                    raise CleanupError(*state.release_error)
                return
            try:
                await self._release_for_finalization(release_phase)
            finally:
                if self._active_acquisitions:
                    # Do not seal a still-writing journal or close a transport
                    # that a late allocation/rollback still needs. The caller
                    # may retry final close after these acquisitions settle.
                    self._cleanup.update(ok=False, error_kind="cleanup_acquisition_pending")
                    raise CleanupError("cleanup_acquisition_pending", "drain_acquisitions")
                pending = await self._seal_resources()
                recoverable = await self._handoff(pending)
                await self._close_management(pending, recoverable)
                if self._journal is not None and self._journal.has_unresolved_creation:
                    raise CleanupError("cleanup_ambiguous_create", "create")
                state.completed = (not pending or recoverable) and not self._cleanup_clients
                if pending and not recoverable and self._journal is None:
                    # An untransferred in-memory owner can explicitly retry
                    # release on its original transport, with no new worker.
                    state.sealed = False
                    state.release_attempted = False
            if state.release_error is not None:
                raise CleanupError(*state.release_error)
            self._cleanup.update(ok=True, error_kind=None)

    async def _release_for_finalization(self, release_phase: Callable[[], Awaitable[None]]) -> None:
        state = self._finalization
        if state.release_attempted:
            if self._cleanup_clients:
                await self._close_phase_clients()
            return
        state.release_error = None
        try:
            await release_phase()
        except CleanupError as error:
            state.release_error = (error.kind, error.phase)
            raise
        except Exception:
            state.release_error = ("cleanup_failed", "release")
            raise
        finally:
            state.release_attempted = not self._active_acquisitions

    async def _seal_resources(self) -> frozenset[str]:
        pending = self.pending_resource_ids
        if self._finalization.sealed:
            return pending
        for env_id in pending:
            await self._remember_created(env_id)
        self._closed = True
        if self._journal is not None:
            await self._journal.persist(self._journal.detach)
        self._finalization.sealed = True
        return pending

    async def _handoff(self, pending: frozenset[str]) -> bool:
        durable = self._journal is not None and bool(os.environ.get("XTUNER_SANDBOX_CLEANUP_FACTORY"))
        recoverable = self._recovery_config is not None or durable
        ambiguous = self._journal is not None and self._journal.has_unresolved_creation
        if (pending or ambiguous) and recoverable and not self._finalization.handed_off:
            task = self._deferred_cleanup
            if task is None or task.cancelled() or (task.done() and task.exception() is not None):
                task = self._deferred_cleanup = defer_cleanup(
                    [key for key in self._known_ids if key in pending],
                    self._fresh_provider,
                    journal=self._journal,
                    timeout=self._cleanup_retry_timeout_sec,
                    verify_deletion=self._verify_deletion,
                )
                self._cleanup["deferred"] = True
                task.add_done_callback(self._record_recovery_result)
            if durable:
                # Ownership is not handed off until the durable enqueue finishes.
                await asyncio.shield(task)
            self._finalization.handed_off = True
        return recoverable

    def _fresh_provider(self) -> SandboxProvider[Any]:
        if self._recovery_config is None:
            raise RuntimeError("Recovery requires the configured durable factory")
        provider: SandboxProvider[Any] = create_object(deepcopy(self._recovery_config))
        if provider is self._provider:
            raise RuntimeError("Cleanup factory must return a fresh provider")
        return provider

    def _record_recovery_result(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            result = {"ok": False, "error_kind": "cancelled"}
        elif task.exception() is not None:
            result = {"ok": False, "error_kind": "recovery_failed"}
        else:
            result = task.result()
        self._cleanup["deferred_result"] = result

    async def _close_management(self, pending: frozenset[str], recoverable: bool) -> None:
        # Keep unrecoverable failed ownership on the original transport for an
        # explicit retry; a shared provider is never closed by this lifecycle.
        if self._close_provider and not self._provider_closed and (not pending or recoverable):
            close = getattr(self._provider, "aclose", None)
            if close is not None:
                await finalize_bounded(close(), timeout_seconds=5, diagnostic={})
            self._cleanup["provider_closed"] = True
            self._provider_closed = True

    @staticmethod
    def _url_of(client: Any) -> str | None:
        for attr in ("base_url", "url", "endpoint", "_base_url"):
            val = getattr(client, attr, None)
            if val:
                return str(val)
        return None

    async def _create(self, prepare: Callable[[], Awaitable[dict[str, Any]]]) -> tuple[Any, str]:
        create_id = uuid.uuid4().hex
        try:
            create_kwargs = await prepare()
            self._assert_acquisition_open()
        except Exception as exc:
            # Validation/rate limiting have not issued a provider request.
            setattr(exc, "allocation_state", "not_allocated")
            raise
        before = self._owned_orphan_ids()
        started = False
        allocated_ids: set[str] = set()
        allocation_callback = getattr(self._provider, "supports_allocation_callback", False)

        async def on_allocated(env_id: str) -> None:
            if not isinstance(env_id, str) or not env_id:
                raise ValueError("sandbox ID must be a nonempty string")
            if env_id not in allocated_ids:
                allocated_ids.add(env_id)
                self._known_ids[env_id] = None  # Retain ownership even if persistence fails.
                if self._journal is not None:
                    await self._journal.persist(self._journal.allocated, env_id, create_id=create_id)
            if self._journal is not None and is_stopping(self._journal.path.parent):
                raise RunStoppingError("allocated")

        try:
            if self._journal is not None:
                await self._journal.persist(self._journal.begin_create, create_id=create_id)
            started = True
            # Explicit capability opt-in: never retry a POST after TypeError,
            # and never forward a callback through an unaware provider's **kwargs.
            if allocation_callback:
                create_kwargs["on_allocated"] = on_allocated
            client, env_id = await self._provider.create(**create_kwargs)
            self._unclaimed_creates[env_id] = client
            await on_allocated(env_id)
        except BaseException as exc:
            known = set(allocated_ids)
            if not allocation_callback:
                current = asyncio.current_task()
                concurrent = any(task is not current for task in self._active_acquisitions.values())
                if not concurrent:
                    # Legacy providers expose only global ownership. A returned
                    # resource already belongs to its own transaction; while
                    # another acquisition is in flight, even an unclaimed ID
                    # cannot be attributed safely to this failed POST.
                    known.update((self._owned_orphan_ids() - before).difference(self._known_ids))
            for env_id in known:
                self._pending_delete_ids.add(env_id)
                await self._remember_created(env_id)
            ambiguous = started and not known and getattr(exc, "allocation_state", None) != "not_allocated"
            if self._journal is not None:
                await self._journal.persist(self._journal.end_create, ambiguous=ambiguous, create_id=create_id)
            if ambiguous and isinstance(exc, Exception):
                setattr(exc, "allocation_state", "unknown")
            elif started and isinstance(exc, Exception) and not hasattr(exc, "allocation_state"):
                setattr(exc, "allocation_state", "allocated" if known else "unknown")
            raise
        else:
            if self._journal is not None:
                await self._journal.persist(self._journal.end_create, ambiguous=False, create_id=create_id)
            self._unclaimed_creates.pop(env_id, None)
            return client, env_id

    async def _rollback_group(self, name: str, group: list[SandboxLease]) -> None:
        diagnostic: dict[str, Any] = {"name": name, "phase": "acquire_rollback"}
        self._cleanup.setdefault("acquire_rollbacks", []).append(diagnostic)
        try:
            await finalize_bounded(
                self._release_group(name, group, delete_deadline=self._phase_delete_deadline),
                timeout_seconds=self._cleanup_timeout_sec,
                diagnostic=diagnostic,
            )
        except CleanupError:
            # Preserve the acquisition failure; release_all retries retained
            # leases and sockets. Repeated cancellation cannot skip this pass.
            pass

    async def _release_groups(self, delete_deadline: float) -> None:
        for name in list(reversed(self._clients)):
            client = self._clients.pop(name)
            env_id = self._env_ids.pop(name, None)
            self._urls.pop(name, None)
            group = self._groups.pop(name, None)
            if group is None and env_id is not None:
                group = [SandboxLease(name="primary", client=client, env_id=env_id)]
            if group:
                await self._release_group(name, group, delete_deadline=delete_deadline)

    async def _delete_pass(self, env_ids: list[str], deadline: float) -> None:
        env_ids = list(dict.fromkeys(env_ids))
        for index, env_id in enumerate(env_ids):
            self._pending_delete_ids.add(env_id)
            remaining = deadline - time.monotonic()
            diagnostic = {"env_id": env_id, "phase": "delete_sandbox"}
            self._cleanup.setdefault("delete_attempts", []).append(diagnostic)
            if remaining <= 0:
                diagnostic.update(ok=False, error_kind="delete_budget_exhausted")
                continue
            # Include finalize_bounded's cancellation grace in this member's
            # share. A stalled primary must leave time to delete dependencies.
            share = remaining / (len(env_ids) - index)
            try:
                await finalize_bounded(
                    delete_confirmed(self._provider, env_id)
                    if self._verify_deletion
                    else self._provider.delete(env_id),
                    timeout_seconds=share / 1.1,
                    diagnostic=diagnostic,
                )
            except CleanupError:
                continue
            self._pending_delete_ids.discard(env_id)
            self._pending_verification_ids.discard(env_id)
            if self._verify_deletion:
                self._cleanup.setdefault("final_states", []).append({"env_id": env_id, "status": "deleted"})
                if self._journal is not None:
                    await self._journal.persist(self._journal.confirmed, env_id)

    def _reverse_acquisition_order(self, env_ids: set[str]) -> list[str]:
        """Keep dependency order in retries; sort only IDs without observed acquisition order."""
        known = [env_id for env_id in reversed(self._known_ids) if env_id in env_ids]
        return known + sorted(env_ids.difference(self._known_ids))

    async def _release_group(
        self, name: str, group: list[SandboxLease], *, delete_deadline: float | None = None
    ) -> None:
        # A stalled HTTP close must not prevent deleting the next dependency.
        # Preserve reverse dependency deletion, then close clients independently.
        leases = list(reversed(group))
        self._cleanup_clients.extend(lease.client for lease in leases)
        diagnostic = {"name": name, "phase": "delete_sandboxes"}
        self._cleanup.setdefault("group_releases", []).append(diagnostic)
        seconds = self._cleanup_timeout_sec
        deadline = time.monotonic() + seconds * 0.4 if delete_deadline is None else delete_deadline
        env_ids = [lease.env_id for lease in leases]
        self._pending_delete_ids.update(env_ids)
        try:
            # Also called by upstream while unwinding a partial acquisition,
            # before Runner reaches release_all(). Never replace its primary
            # cancellation/error; owned IDs are retried by the final pool pass.
            await finalize_bounded(
                self._delete_pass(env_ids, deadline),
                timeout_seconds=max(0.001, deadline - time.monotonic()) * 1.1,
                diagnostic=diagnostic,
            )
        except CleanupError:
            pass
        if self._pending_delete_ids.intersection(env_ids):
            diagnostic.update(
                ok=False,
                error_kind="cleanup_timeout"
                if any(
                    row.get("error_kind") == "cleanup_timeout" and row["env_id"] in env_ids
                    for row in self._cleanup.get("delete_attempts", [])
                )
                else "delete_failed",
            )

        if not self._releasing:
            retry_ids = self._pending_delete_ids.intersection(env_ids)
            await self._delete_pass(self._reverse_acquisition_order(retry_ids), time.monotonic() + seconds * 0.4)
            # Acquisition rollback must release sockets even when callers use
            # Pool.get directly and never reach Runner's final release.
            clients = [lease.client for lease in leases]

            async def close_rollback() -> None:
                outcomes = await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
                for client, outcome in zip(clients, outcomes):
                    if not isinstance(outcome, BaseException):
                        self._cleanup_clients.remove(client)
                if any(isinstance(value, BaseException) for value in outcomes):
                    raise RuntimeError("Sandbox rollback client closure failed")

            try:
                await finalize_bounded(
                    close_rollback(),
                    timeout_seconds=seconds * 0.2,
                    diagnostic=diagnostic.setdefault("close", {}),
                )
            except CleanupError:
                # Keep failed clients for the final release, preserving the
                # acquisition exception that caused rollback.
                pass

    def _owned_orphan_ids(self) -> set[str]:
        return set(getattr(self._provider, "owned_env_ids", ())) if self._close_provider else set()
