"""Exit-only budgets and receipts, independent of inference request timeouts.

Timeouts are failures: a timed-out thread is not claimed to have stopped. The
actor event loop stays available to retrieve shutdown_status while it exits.
"""

import asyncio
import copy
import time

import ray

CONTROLLER_SHUTDOWN_TIMEOUT = 300.0
WORKER_SHUTDOWN_TIMEOUT = 170.0
SESSION_SHUTDOWN_TIMEOUT = 90.0
DRIVER_SHUTDOWN_TIMEOUT = 330.0
WORKER_SESSION_PHASE_TIMEOUT = 95.0
WORKER_BACKEND_PHASE_TIMEOUT = 65.0


class ShutdownError(RuntimeError):
    pass


class ShutdownTrace:
    def __init__(self, owner, role, blocking_tasks=None):
        self.owner = owner
        self.started = time.monotonic()
        self.report = {"role": role, "state": "running", "phases": [], "errors": []}
        owner._shutdown_report = self.report
        self.blocking_tasks = blocking_tasks

    def begin(self, name):
        phase = {"name": name, "state": "running", "started_s": time.monotonic() - self.started}
        self.report["phases"].append(phase)
        self.owner.logger.info(f"Shutdown phase started: {name}")
        return phase, time.monotonic()

    def end(self, phase, started, exc=None):
        phase.update(state="failed" if exc else "complete", elapsed_s=time.monotonic() - started)
        if exc is not None:
            phase["error"] = f"{type(exc).__name__}: {exc}"
            self.report["errors"].append({"phase": phase["name"], "error": phase["error"]})
        self.owner.logger.info(f"Shutdown phase ended: {phase}")

    def finish(self):
        self.report.update(state="failed" if self.report["errors"] else "complete",
                           elapsed_s=time.monotonic() - self.started)
        if self.report["errors"]:
            raise ShutdownError(str(self.report))
        return copy.deepcopy(self.report)

    async def run(self, name, fn, timeout):
        phase, started = self.begin(name)
        task = asyncio.create_task(asyncio.to_thread(fn))
        if self.blocking_tasks is not None:
            self.blocking_tasks.add(task)

        def completed(work):
            if self.blocking_tasks is not None:
                self.blocking_tasks.discard(work)
            if not work.cancelled():
                work.exception()  # Also retrieve late errors after a phase timeout.

        task.add_done_callback(completed)
        try:
            # A timeout ends this wait, not the blocking function's lifetime.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except Exception as exc:
            self.end(phase, started, exc)
            return None
        self.end(phase, started)
        return result


def shutdown_status(owner):
    return copy.deepcopy(getattr(owner, "_shutdown_report", {"state": "not_started"}))


class WorkerShutdown:
    """One backend lifetime's partial/full close, including unfinished threads."""

    def __init__(self, owner):
        self.owner = owner
        self.logger = owner.logger
        self._future = None
        self._full_close = False
        self._blocking_tasks = set()

    def prepare_start(self):
        """Reject overlap before native init can replace any resource handles."""
        if self._full_close:
            raise ShutdownError("Worker is sealed for final shutdown")
        if ((self._future is not None and not self._future.done())
                or any(not task.done() for task in self._blocking_tasks)):
            raise ShutdownError("Cannot start a backend while its previous close is still running")
        if self.owner.server_task is not None or self.owner.server_process is not None:
            raise ShutdownError("Cannot start a backend before the previous backend is closed")
        self._future = None
        self._full_close = False
        self._shutdown_report = {"state": "not_started"}

    async def close(self, *, stop_session_server=False):
        if self._future is None or (stop_session_server and not self._full_close):
            previous = self._future
            self._full_close = stop_session_server
            # Register the stronger operation before yielding: reinit must see
            # it even while it is waiting for an earlier partial close.
            self._future = asyncio.create_task(self._close(previous, stop_session_server))
        return await asyncio.shield(self._future)

    async def _close(self, previous, stop_session_server):
        if previous is not None:
            try:
                await asyncio.shield(previous)
            except ShutdownError:
                # A stronger close may retry a completed partial failure, but
                # must never race a timed-out thread still using old handles.
                if any(not task.done() for task in self._blocking_tasks):
                    raise ShutdownError("Previous backend close still has running blocking work")
        trace = ShutdownTrace(self, "worker", self._blocking_tasks)
        if stop_session_server:
            await trace.run("session_server", self.owner._stop_session_server, WORKER_SESSION_PHASE_TIMEOUT)
        await trace.run("backend", self.owner._shutdown_backend, WORKER_BACKEND_PHASE_TIMEOUT)
        return trace.finish()

    def status(self):
        return shutdown_status(self)


async def shutdown_controller(owner, *, health_timeout=35.0, worker_timeout=WORKER_SHUTDOWN_TIMEOUT,
                              proxy_timeout=90.0):
    # Repeated RPCs share one operation, including its failure; no duplicate kills.
    if getattr(owner, "_shutdown_future", None) is not None:
        return await asyncio.shield(owner._shutdown_future)

    async def close():
        trace = ShutdownTrace(owner, "controller")

        def stop_health():
            owner.health_manager.stop()
            thread = getattr(owner.health_manager, "_thread", None)
            if thread is not None and thread.is_alive():
                raise ShutdownError("Health recovery thread is still alive")

        await trace.run("health_manager", stop_health, health_timeout)

        def stop_workers():
            actors = owner.registry.all_actors()
            # Shutdown-only timeout; never reuse the multi-hour rollout budget.
            return ray.get([actor.shutdown.remote(stop_session_server=True) for actor in actors],
                           timeout=worker_timeout)

        await trace.run("workers", stop_workers, worker_timeout + 1)
        if owner.proxy_manager is not None:
            await trace.run("proxy_routes", owner.proxy_manager.close, proxy_timeout)
        return trace.finish()

    owner._shutdown_future = asyncio.create_task(close())
    return await asyncio.shield(owner._shutdown_future)
