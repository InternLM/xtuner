"""Bounded read-only process/exit-code observation; unknown is never missing."""

import asyncio
import time

import httpx


TERMINAL = {"deleted", "terminated", "failed", "stopped", "expired"}


async def observe_process(
    client,
    *,
    pid,
    timeout,
    recovery_timeout,
    poll_interval,
    retry_interval,
    rc_wait_timeout,
    rc_poll_interval,
    read_rc,
    max_failures=3,
):
    deadline = time.monotonic() + timeout
    failures = {}  # Independent PID and rc outages: a good PID cannot heal rc.
    consecutive = {"pid": 0, "rc": 0}
    errors = {"pid": 0, "rc": 0}
    observations, rc_checks = [], []
    polls = 0
    missing_since = None

    def end():
        return (
            min([deadline] + [t + recovery_timeout for t in failures.values()]) if recovery_timeout > 0 else deadline
        )

    def result(reason, rc=None):
        return (
            reason,
            rc,
            dict(
                pid=pid,
                polls=polls,
                pid_check_errors_total=errors["pid"],
                rc_check_errors_total=errors["rc"],
                monitor_observations=observations,
                rc_checks=rc_checks,
                probe_kind=next(iter(failures), None),
                deadline_reached=time.monotonic() >= deadline,
                recovery_deadline_reached=bool(failures) and recovery_timeout > 0 and time.monotonic() >= end(),
            ),
        )

    async def bounded(call, limit):
        remaining = min(limit, end() - time.monotonic())
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(call(), remaining)

    def failed(kind, exc):
        failures.setdefault(kind, time.monotonic())
        errors[kind] += 1
        consecutive[kind] += 1
        row = dict(poll=polls, probe_kind=kind, error_type=type(exc).__name__)
        observations.append(row)
        del observations[:-32]
        return row

    def recovered(kind):
        failures.pop(kind, None)
        consecutive[kind] = 0

    while time.monotonic() < end():
        polls += 1
        running = None
        try:
            running = await bounded(lambda: client.is_pid_running(pid), 30)
            if type(running) is not bool:
                raise httpx.RemoteProtocolError("Non-boolean process observation")
            recovered("pid")
        except (httpx.HTTPError, asyncio.TimeoutError, TimeoutError) as exc:
            row = failed("pid", exc)
            query = getattr(client, "get_lifecycle_state", None)
            if query:
                try:
                    row["sandbox_state"] = await bounded(query, 5)
                    if row["sandbox_state"] in TERMINAL:
                        return result("sandbox_unavailable")
                except (httpx.HTTPError, RuntimeError, asyncio.TimeoutError, TimeoutError):
                    pass

        if running is not True or "rc" in failures:
            started = time.monotonic()
            check = {}
            try:
                rc = await bounded(read_rc, 5)
                if rc is not None and type(rc) is not int:
                    raise httpx.RemoteProtocolError("Invalid exit code")
                recovered("rc")
                check["missing"] = rc is None
                if rc is not None:
                    return result("entry_completed" if rc == 0 else "entry_failed", rc)
                if running is False:
                    missing_since = missing_since or time.monotonic()
                else:
                    missing_since = None
            except (httpx.HTTPError, asyncio.TimeoutError, TimeoutError) as exc:
                failed("rc", exc)
                missing_since = None
                check["error_type"] = type(exc).__name__
                if isinstance(exc, httpx.HTTPStatusError):
                    check["http_status"] = exc.response.status_code
            finally:
                check["elapsed_seconds"] = time.monotonic() - started
                rc_checks.append(check)
                del rc_checks[:-3]
        else:
            missing_since = None

        if failures and (
            time.monotonic() >= end() or (recovery_timeout <= 0 and max(consecutive.values()) >= max_failures)
        ):
            return result("pid_monitor_error")
        if missing_since is not None and time.monotonic() >= missing_since + rc_wait_timeout:
            return result("rc_missing")
        delay = retry_interval if failures else poll_interval if running else rc_poll_interval
        wake = end()
        if missing_since is not None:
            wake = min(wake, missing_since + rc_wait_timeout)
        await asyncio.sleep(min(max(0.01, delay), max(0, wake - time.monotonic())))
    return result("pid_monitor_error" if failures else "rc_missing" if missing_since is not None else "timeout")
