"""Passive exit witness: no periodic sandbox scans while the driver is alive.

Normal exit cleanup belongs to the trainer. A surviving head/peer witness takes
that role only after driver death; failed cleanup is handed to the recycler.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
import time
from pathlib import Path

from cleanup import cleanup_from_environment
from run_control import driver_alive


def main() -> int:
    """Observe driver exit and execute the same bounded exit-cleanup path."""
    directory = Path(os.environ["XTUNER_SANDBOX_CLEANUP_DIR"])
    factory = os.environ["XTUNER_SANDBOX_CLEANUP_FACTORY"]
    module, name = factory.split(":", 1)
    with contextlib.redirect_stdout(sys.stderr):
        if not callable(getattr(importlib.import_module(module), name)):
            raise TypeError("cleanup factory must be callable")
    print("READY", flush=True)
    # The parent closes its handshake pipe after READY. Exit subprocesses must
    # write reports to the log, not that now-unread pipe.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    failures = 0
    while True:
        try:
            alive = driver_alive(directory)
        except OSError:
            # An unreadable lease is unknown, never evidence of driver death.
            # Retry without sandbox scans; log only at powers of two to avoid
            # flooding logs during a sustained filesystem outage.
            failures += 1
            if failures & (failures - 1) == 0:
                logging.warning("Driver lease probe failed (%s attempts); retrying", failures)
            time.sleep(min(5.0, 0.2 * 2 ** min(failures - 1, 5)))
            continue
        failures = 0
        if not alive:
            break
        time.sleep(0.2)
    # Stop admission and read/write final status only inside the bounded child.
    # A live worker may hold admission while its shared-filesystem write stalls.
    return 0 if cleanup_from_environment(finalize=True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
