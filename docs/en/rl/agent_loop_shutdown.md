# AgentLoop shutdown

Both colocated and disaggregated RL trainers close their training and evaluation
loop managers when `fit()` exits, including after an error or cancellation.
Managers forward `close()` to their local loops and Ray actors; routers forward
it to their workers. A disaggregated trainer stops and joins its background
producer before closing loop owners.

## Implementing a resource-owning loop

`AgentLoop.close()` is an asynchronous hook with a default no-op implementation.
Existing loops without persistent resources do not need to change. A loop that
owns sessions, background tasks or external clients should override it to:

1. Stop admitting new work.
2. Cancel or finish its active work and await the associated cleanup.
3. Close owned clients and connections.

Make the hook safe to call repeatedly, and apply appropriate deadlines to your
backend's external I/O. If multiple callers can close the loop concurrently,
share one cleanup task. `await_cleanup(task)` from `xtuner.v1.rl.utils.async_utils`
waits for that task even if the caller is cancelled repeatedly, then propagates
the cancellation. It does not impose a backend-specific timeout or retry policy.

The built-in sandbox loop rejects new runs during shutdown and joins active
runners so their existing `finally` blocks can release their pools. It shares a
single close task for repeated and concurrent calls.

## Pausing for weight updates

Loops whose model calls come from an external harness can also override
`AgentLoop.prepare_pause()` and `AgentLoop.resume()`, both no-ops by default.
Managers call `prepare_pause()` on every active loop before
`pause_generation()`, and `resume()` after `continue_generation()`, in both the
colocated and disaggregated managers. Use `prepare_pause()` to stop admitting new
model calls and let accepted ones finish, so the abort that follows does not cut
a request the harness is still waiting on. If `prepare_pause()` raises, the
manager does not abort inference. Routers and Ray actors forward both hooks the
same way as `close()`.

## Errors and cancellation

Each owner gets a chance to close even if another owner fails. Cleanup failures
are collected in `CleanupError.errors`, preserving the original exception
objects. A child close operation that is cancelled is also a cleanup failure;
it is not silently reported as success.

Cancellation of the caller waits for cleanup to settle before propagating.
If cleanup also fails, the failure is retained as the cancellation's cause.
Likewise, if training and shutdown both fail, the training exception remains the
top-level exception and the cleanup failure is chained as its cause. Unexpected
background producer errors are propagated, including errors raised while that
producer is being stopped.

## Compatibility and scope

There is no environment API migration: `Runner(pool=...)`, `runner.run(item)`,
Judger's pool argument and `item.pipeline_overrides` remain unchanged.
`AgentInSandboxLoopConfig` does not require an environment provider.

The hook gives loop owners a cleanup opportunity while the worker is running.
Environment abstractions, SandboxPool allocation/deletion behavior, deletion
retries and recovery after a worker process is killed are outside this contract.
