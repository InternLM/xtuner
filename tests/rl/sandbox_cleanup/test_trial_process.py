"""Run fresh-interpreter lifecycle tests without global optional-import stubs."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "sandbox_trial_cpu.py"
CASES = [
    node.name
    for cls in ast.parse(SCRIPT.read_text()).body
    if isinstance(cls, ast.ClassDef) and cls.name == "TestTrialProcess"
    for node in cls.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_process_")
]


class TestTrialProcess:
    @pytest.mark.parametrize("case", CASES)
    def test_process_lifecycle(self, case):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("XTUNER_SANDBOX_")}
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "TestTrialProcess." + case],
            capture_output=True,
            text=True,
            env=environment,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
