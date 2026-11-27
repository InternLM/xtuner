# Command-ID based sandbox entries

`BackgroundShellEntry` is an opt-in alternative to `DetachedShellEntry` for
clients which own a background-command record. A command may have finished
successfully while a shell wrapper's `.rc` file is absent; command-ID execution
uses the backend's terminal status instead of shell PID/capture files.

```python
from xtuner.v1.rl.agent_loop.sandbox_agent_loop import BackgroundShellEntry

entry = dict(
    type=BackgroundShellEntry,
    cmd="python /app/run_agent.py",
    timeout=3600,             # remote execution budget, seconds
    host_grace_sec=60,        # extra host observation budget
    poll_interval_sec=5,
    cleanup_timeout_sec=120,  # interrupt / terminal observation budget
)
```

The provider's client must expose these asynchronous methods:

| Method | Return |
| --- | --- |
| `start_command(command, cwd, timeout_sec)` | nonempty command ID string |
| `get_command_status(command_id)` | `{"running": bool, "return_code": int or None}` |
| `get_command_output(command_id)` | `{"stdout": str, "stderr": str}` |
| `interrupt_command(command_id)` | no required result |

The entry launches once. Status/output operations always address that same ID.
Providers can retry transient read failures within the caller's deadline, but
must not transparently replay command submission. Missing exit status is
`exit_code_missing`, not success. A zero exit status completes the entry; the
Runner still executes the configured judger before producing a task reward.

On timeout or cancellation the entry retrieves the ID from an in-flight launch,
interrupts the same command, and waits for a terminal status within the cleanup
budget. Cleanup errors remain in `EntryOutcome.details`; a timed-out cleanup is
not evidence that a remote process or sandbox was released. Runner/Pool retain
sandbox deletion ownership. Cancellation propagates after bounded cleanup.

Command environment values are shell-quoted, and the command runs under bash in
`StageRecord.workspace` (default `/root`). No PID or capture paths are created.
Diagnostics requiring `entry_file="rc"`, `"stdout"` or `"stderr"` must instead
name application-owned files, or read `EntryRecord.result` for command output.
An SDK with only combined output may report that in stdout and leave stderr empty.
Existing clients lacking this protocol should continue using their configured
synchronous or detached entries. XTuner itself does not depend on an SDK or on
recipe-private clients.
