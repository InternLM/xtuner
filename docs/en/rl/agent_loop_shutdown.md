# AgentLoop lifecycle hooks

`AgentLoop` has three asynchronous lifecycle hooks. All of them are no-ops by
default, so loops without persistent resources or external clients need no
changes. Routers and Ray actors forward each hook to their workers.

## `close()`

Both RL trainers close their training and evaluation loop managers when `fit()`
exits, including after an error. A disaggregated trainer stops its background
producer first. A loop that owns sessions, background tasks or external clients
should override `close()` to stop admitting work, wait for active work to clean
up and close its clients. Bound any external I/O in the hook. Close failures are
logged and do not replace the training result.

The built-in sandbox loop rejects new runs, cancels active runners and waits
for their `finally` blocks to release their pools.

## `prepare_pause()` and `resume()`

Managers call `prepare_pause()` on every active loop before
`pause_generation()`, and `resume()` after `continue_generation()`. Loops whose
model calls come from an external harness use `prepare_pause()` to stop
admitting new calls and let accepted ones finish, so the abort that follows
does not cut a request the harness is still waiting on. If `prepare_pause()`
raises, the manager does not abort inference.
