"""Optional bounded observation recovery, adapted from native e4b26352.

Unknown observations never establish process death. No Agent/tool request is
replayed, and the recovery budget cannot extend the entry deadline.
"""
import asyncio
import time
import httpx

TERMINAL = {'deleted', 'terminated', 'failed', 'stopped', 'expired'}

async def observe_process(client, *, pid, timeout, recovery_timeout, poll_interval,
                          retry_interval, rc_wait_timeout, rc_poll_interval, read_rc):
    deadline = time.monotonic() + timeout
    first_failure = None
    errors, observations, polls = 0, [], 0
    dead_since = None
    last_unknown = False

    def result(reason, rc=None):
        return reason, rc, {'pid': pid, 'polls': polls, 'pid_check_errors_total': errors,
                            'monitor_observations': observations,
                            'deadline_reached': time.monotonic() >= deadline,
                            'recovery_deadline_reached': first_failure is not None and time.monotonic() >= first_failure + recovery_timeout}

    async def bounded(call, limit, end):
        remaining = min(limit, end - time.monotonic())
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(call(), timeout=remaining)

    while time.monotonic() < deadline:
        observation_end = min(deadline, first_failure + recovery_timeout) if first_failure is not None else deadline
        if time.monotonic() >= observation_end:
            return result('pid_monitor_error')
        polls += 1
        try:
            running = await bounded(lambda: client.is_pid_running(pid), 30, observation_end)
            if type(running) is not bool:
                raise httpx.RemoteProtocolError('Non-boolean process observation')
            first_failure = None
            last_unknown = False
        except (httpx.HTTPError, TimeoutError) as exc:
            last_unknown = True
            errors += 1
            first_failure = first_failure or time.monotonic()
            observation_end = min(deadline, first_failure + recovery_timeout)
            state = None
            query_state = getattr(client, 'get_lifecycle_state', None)
            if query_state is not None:
                try:
                    state = await bounded(query_state, 5, observation_end)
                except (httpx.HTTPError, RuntimeError, TimeoutError):
                    pass
            observations.append({'poll':polls, 'error_type':type(exc).__name__, 'sandbox_state':state})
            if len(observations) > 32:
                del observations[1]
            if state in TERMINAL:
                return result('sandbox_unavailable')
            # The trusted wrapper's rc can complete during a probe outage.
            try:
                rc = await bounded(read_rc, 5, observation_end)
            except (httpx.HTTPError, TimeoutError):
                rc = None
            if rc is not None:
                return result('entry_completed' if rc == 0 else 'entry_failed', rc)
            if time.monotonic() >= observation_end:
                return result('pid_monitor_error')
            await asyncio.sleep(min(retry_interval, max(0, observation_end-time.monotonic())))
            continue

        if running:
            dead_since = None
        else:
            dead_since = dead_since or time.monotonic()
            completion_end = min(deadline, dead_since + rc_wait_timeout)
            try:
                rc = await bounded(read_rc, 5, completion_end)
            except (httpx.HTTPError, TimeoutError):
                rc = None
            if rc is not None:
                return result('entry_completed' if rc == 0 else 'entry_failed', rc)
            if time.monotonic() >= completion_end:
                return result('rc_missing')
        interval = poll_interval if running else rc_poll_interval
        await asyncio.sleep(min(max(.01, interval), max(0, deadline-time.monotonic())))
    return result('pid_monitor_error' if last_unknown else 'rc_missing' if dead_since else 'timeout')
