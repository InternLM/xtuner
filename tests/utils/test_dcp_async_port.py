"""Regression tests for the DCP async-checkpoint daemon port patch.

The original bug: PyTorch's process-based async checkpoint picks the daemon TCPStore
port via ``get_free_port`` (``bind(("localhost", 0))`` -> ephemeral port -> close),
so a transient outbound connection can grab the same port during the spawn window and
the daemon ``listen`` fails with ``EADDRINUSE`` (more likely at large scale).

``patch_dcp_async_daemon_port`` relocates the daemon port to a fixed range *outside*
the kernel ephemeral range, which the kernel never auto-assigns. These tests drive the
public patch through the symbol torch actually calls and assert observable behavior:
the chosen port lands in the configured range and is bindable, a single-port range is
pinned, occupied ports are skipped, and an exhausted range fails fast.
"""

import socket

import pytest
import torch.distributed.checkpoint._async_process_executor as _async_process_executor

import xtuner.v1.patch.dcp_async_port as dcp_async_port
from xtuner.v1.patch import patch_dcp_async_daemon_port


def _free_port_for_test() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def reset_patch_state():
    original = _async_process_executor.get_free_port
    was_patched = dcp_async_port._PATCHED
    dcp_async_port._PATCHED = False
    yield
    _async_process_executor.get_free_port = original
    dcp_async_port._PATCHED = was_patched


def _get_daemon_port() -> int:
    """Call the port selector exactly as torch's async-checkpoint executor does."""
    return _async_process_executor.get_free_port()


class TestDcpAsyncPort:
    def test_patch_routes_daemon_port_into_configured_range(self, reset_patch_state, monkeypatch):
        # Pick a free range so the assertion is deterministic on any host.
        base = _free_port_for_test()
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_RANGE", f"{base}:{base + 50}")

        patch_dcp_async_daemon_port()
        port = _get_daemon_port()

        # The patch must take effect on the symbol torch actually calls, so the returned
        # port falls in our configured range and is genuinely bindable.
        assert base <= port < base + 50
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("", port))

    def test_patch_is_idempotent(self, reset_patch_state):
        patch_dcp_async_daemon_port()
        sentinel = lambda: 0  # noqa: E731 - stand-in for a later reassignment
        _async_process_executor.get_free_port = sentinel
        patch_dcp_async_daemon_port()  # second call must be a no-op
        assert _async_process_executor.get_free_port is sentinel

    def test_single_port_range_is_pinned(self, reset_patch_state, monkeypatch):
        base = _free_port_for_test()
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_RANGE", f"{base}:{base + 1}")

        patch_dcp_async_daemon_port()

        assert _get_daemon_port() == base

    def test_degenerate_range_falls_back_to_single_port(self, reset_patch_state, monkeypatch):
        # "X:X" is empty; the patch must clamp it to a single usable port (X) rather
        # than failing to find any port.
        base = _free_port_for_test()
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_RANGE", f"{base}:{base}")

        patch_dcp_async_daemon_port()

        assert _get_daemon_port() == base

    def test_patch_skips_occupied_port(self, reset_patch_state, monkeypatch):
        occupied = _free_port_for_test()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", occupied))
        listener.listen(1)
        try:
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_RANGE", f"{occupied}:{occupied + 5}")
            patch_dcp_async_daemon_port()
            port = _get_daemon_port()
            # SO_REUSEADDR still cannot bind over an actively listening socket, so the
            # base port is occupied and selection must move past it.
            assert occupied < port < occupied + 5
        finally:
            listener.close()

    def test_patch_raises_when_no_free_port(self, reset_patch_state, monkeypatch):
        occupied = _free_port_for_test()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", occupied))
        listener.listen(1)
        try:
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_RANGE", f"{occupied}:{occupied + 1}")
            patch_dcp_async_daemon_port()
            with pytest.raises(RuntimeError, match="no free daemon port"):
                _get_daemon_port()
        finally:
            listener.close()
