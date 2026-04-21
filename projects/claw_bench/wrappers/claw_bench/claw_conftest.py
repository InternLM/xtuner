"""Shared pytest plugin for claw-bench-style verifiers.

Registers the ``--workspace`` CLI option and the ``weight`` marker.  The
``pytest_ctrf.sh`` wrapper copies this file next to the pytest target at
runtime, so tasks don't need to ship their own conftest.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--workspace",
        action="store",
        default=None,
        help="Path to the agent's workspace (TASK_WORKSPACE)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "weight(n): per-test weight used by the judger aggregator",
    )
