"""Regression tests for the DCP async-checkpoint daemon port patch.

The original bug: PyTorch's process-based async checkpoint picks the daemon TCPStore
port via ``get_free_port`` (``bind(("localhost", 0))`` -> ephemeral port -> close),
so a transient outbound connection can grab the same port during the spawn window and
the daemon ``listen`` fails with ``EADDRINUSE`` (more likely at large scale).

``patch_dcp_async_daemon_port`` relocates the daemon port to a fixed range *outside*
the kernel ephemeral range, which the kernel never auto-assigns. These tests verify the
patch is wired to the symbol torch actually calls and that the chosen port is kept out
of the ephemeral range.
"""

import socket
from unittest.mock import MagicMock

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


class TestDcpAsyncPort:
    def test_patch_replaces_get_free_port(self, reset_patch_state):
        assert _async_process_executor.get_free_port is not dcp_async_port._xtuner_dcp_get_free_port
        patch_dcp_async_daemon_port()
        # torch calls the name bound in the executor module namespace, so that is what
        # must be replaced for the fix to take effect.
        assert _async_process_executor.get_free_port is dcp_async_port._xtuner_dcp_get_free_port

    def test_patch_is_idempotent(self, reset_patch_state):
        patch_dcp_async_daemon_port()
        sentinel = lambda: 0  # noqa: E731 - stand-in for a later reassignment
        _async_process_executor.get_free_port = sentinel
        patch_dcp_async_daemon_port()  # second call must be a no-op
        assert _async_process_executor.get_free_port is sentinel

    def test_chosen_port_is_outside_ephemeral(self, monkeypatch):
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", "29600")
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "400")
        monkeypatch.setattr(dcp_async_port, "_ephemeral_range", lambda: (32768, 60999))
        port = dcp_async_port._xtuner_dcp_get_free_port()
        assert 29600 <= port < 30000
        assert port < 32768

    def test_env_overrides_range(self, monkeypatch):
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", "31234")
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "7")
        assert dcp_async_port._port_base() == 31234
        assert dcp_async_port._port_span() == 7

    def test_port_span_clamped_to_minimum_one(self, monkeypatch):
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "0")
        assert dcp_async_port._port_span() == 1

    def test_probe_skips_occupied_port(self, monkeypatch):
        occupied = _free_port_for_test()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", occupied))
        listener.listen(1)
        try:
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", str(occupied))
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "5")
            monkeypatch.setattr(dcp_async_port, "_ephemeral_range", lambda: None)
            port = dcp_async_port._xtuner_dcp_get_free_port()
            # SO_REUSEADDR still cannot bind over an actively listening socket, so the
            # probe must move past the occupied base port.
            assert occupied < port < occupied + 5
        finally:
            listener.close()

    def test_no_free_port_raises(self, monkeypatch):
        occupied = _free_port_for_test()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", occupied))
        listener.listen(1)
        try:
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", str(occupied))
            monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "1")
            monkeypatch.setattr(dcp_async_port, "_ephemeral_range", lambda: None)
            with pytest.raises(RuntimeError, match="no free daemon port"):
                dcp_async_port._xtuner_dcp_get_free_port()
        finally:
            listener.close()

    @pytest.mark.parametrize(
        "base,hi_excl,eph,expected",
        [
            (29600, 30000, (32768, 60999), True),  # fully below ephemeral range
            (61000, 61400, (32768, 60999), True),  # fully above ephemeral range
            (40000, 40400, (32768, 60999), False),  # inside ephemeral range
            (29600, 30000, None, False),  # unknown range -> conservative False
            (32768, 33000, (32768, 60999), False),  # overlaps the lower bound
        ],
    )
    def test_is_outside_ephemeral(self, base, hi_excl, eph, expected):
        assert dcp_async_port._is_outside_ephemeral(base, hi_excl, eph) is expected

    def test_patch_warns_when_range_overlaps_ephemeral(self, reset_patch_state, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr(dcp_async_port, "logger", mock_logger)
        monkeypatch.setattr(dcp_async_port, "_ephemeral_range", lambda: (29000, 60999))
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", "29600")
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "400")
        patch_dcp_async_daemon_port()
        assert mock_logger.warning.called
        assert not mock_logger.info.called

    def test_patch_logs_info_when_range_outside_ephemeral(self, reset_patch_state, monkeypatch):
        mock_logger = MagicMock()
        monkeypatch.setattr(dcp_async_port, "logger", mock_logger)
        monkeypatch.setattr(dcp_async_port, "_ephemeral_range", lambda: (32768, 60999))
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_BASE", "29600")
        monkeypatch.setenv("XTUNER_DCP_DAEMON_PORT_SPAN", "400")
        patch_dcp_async_daemon_port()
        assert mock_logger.info.called
        assert not mock_logger.warning.called
