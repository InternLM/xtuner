import pytest
import subprocess


def run_cmd(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
    except Exception as e:
        pytest.fail(f"Unknown error: {e}")
    if result.returncode != 0:
        pytest.fail(f"run command error:{result.stderr}")
