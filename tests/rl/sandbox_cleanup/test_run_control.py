"""Keep import-only dependency substitutions isolated from the pytest process."""

import subprocess
import sys
from pathlib import Path


def test_run_control_integration() -> None:
    script = Path(__file__).resolve().parents[1] / "sandbox_run_control_cpu.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
