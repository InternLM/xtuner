"""Run CPU lifecycle cases in subprocesses to isolate optional-import stubs."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "sandbox_cleanup_cpu.py"
IMPORT_SCRIPT = SCRIPT.with_name("sandbox_cleanup_import_cpu.py")
CASES = [
    node.name
    for cls in ast.parse(SCRIPT.read_text()).body
    if isinstance(cls, ast.ClassDef) and cls.name == "TestSandboxCleanup"
    for node in cls.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
]
IMPORT_CASES = [
    node.name
    for cls in ast.parse(IMPORT_SCRIPT.read_text()).body
    if isinstance(cls, ast.ClassDef) and cls.name == "ImportCompatibility"
    for node in cls.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
]


def _run_isolated(script: Path, case: str) -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("XTUNER_SANDBOX_") and key != "XTUNER_TEST_POOL_SOURCE"
    }
    result = subprocess.run(
        [sys.executable, str(script), case],
        capture_output=True,
        text=True,
        env=env,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestSandboxCleanup:
    @pytest.mark.parametrize("case", CASES)
    def test_lifecycle(self, case: str) -> None:
        _run_isolated(SCRIPT, f"TestSandboxCleanup.{case}")

    @pytest.mark.parametrize("case", IMPORT_CASES)
    def test_import_compatibility(self, case: str) -> None:
        _run_isolated(IMPORT_SCRIPT, f"ImportCompatibility.{case}")
