"""Independent recovery supervisor; stdin EOF means the owning driver exited.

Only the driver owns the pipe's write end. SIGKILL closes it in the kernel.
Each scan uses a fresh interpreter with a wall-clock limit, so a stuck provider
cannot prevent subsequent scans. The supervisor must run outside any cgroup
that the scheduler kills if recovery after whole-job termination is required.
"""

from __future__ import annotations

import importlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> int:
    """Watch the driver and reconcile abandoned Pool journals until exit."""
    directory = Path(os.environ["XTUNER_SANDBOX_CLEANUP_DIR"])
    factory = os.environ["XTUNER_SANDBOX_CLEANUP_FACTORY"]
    budget = float(os.environ.get("XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT", "180"))
    interval = float(os.environ.get("XTUNER_SANDBOX_CLEANUP_INTERVAL", "30"))
    module, name = factory.split(":", 1)
    if not callable(getattr(importlib.import_module(module), name)):
        raise TypeError("cleanup factory must be callable")
    print("READY", flush=True)
    deadline: float | None = None
    last_ok = False
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], interval if deadline is None else min(interval, 1))
        if readable and not os.read(sys.stdin.fileno(), 4096) and deadline is None:
            deadline = time.monotonic() + budget
        remaining = budget if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            return 0 if last_ok else 1
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "cleanup.py"),
                    "--directory",
                    str(directory),
                    "--factory",
                    factory,
                    "--timeout",
                    str(min(30, remaining / 2)),
                ],
                timeout=min(budget, remaining),
                stdout=subprocess.DEVNULL,
                # Errors go to the supervisor log; no provider payloads are printed.
                check=False,
            )
            last_ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            last_ok = False
        if deadline is not None and last_ok:
            return 0
        if deadline is not None:
            # EOF is always readable; avoid spinning on failed or live journals.
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        (directory / "supervisor-status.json").write_text(
            json.dumps({"ok": last_ok, "driver_exited": deadline is not None, "updated_at": time.time()}) + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
