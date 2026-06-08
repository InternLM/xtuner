"""Patch DCP async-checkpoint daemon port selection.

PyTorch's process-based async checkpoint (``dcp.async_save`` +
``AsyncCheckpointerType.PROCESS``) spawns a background daemon. The coordinator rank
calls ``torch.distributed.elastic.utils.distributed.get_free_port`` to pick a port,
broadcasts it to all ranks, and each rank's daemon initializes a GLOO ``TCPStore``
with it.

``get_free_port`` binds ``("localhost", 0)`` so the kernel assigns a port from the
ephemeral range (``net.ipv4.ip_local_port_range``, typically 32768-60999) and then
immediately ``close()``s it. Between that ``close()`` and the daemon subprocess
actually calling ``bind()/listen()`` there is a spawn window (hundreds of ms) during
which the kernel may hand the same port to a transient NCCL/gloo outbound connection,
making the daemon TCPStore ``listen`` fail with ``EADDRINUSE``. The larger the job
(more connections, slower spawn/broadcast), the more likely it triggers.

This patch picks the daemon port from a fixed range *outside* the ephemeral range.
The kernel only auto-assigns outbound ports from ``ip_local_port_range``, so a port
outside it is *never* taken by the kernel's automatic source-port assignment -- this
categorically removes the dominant production vector (the EADDRINUSE we actually hit),
regardless of training scale.

Scope of the guarantee:
- Kernel auto-assignment (``bind(port=0)`` / unbound ``connect``): eliminated entirely.
- An explicit ``bind(("", <our_port>))`` by some co-located process (monitoring agent,
  another framework's rendezvous, etc.) is NOT governed by ``ip_local_port_range`` and
  is still allowed by the kernel. This residual is rare, scale-independent, and the
  probe below skips ports already taken; only an explicit bind landing in the spawn
  window could still collide. Note ``ip_local_reserved_ports`` also does not stop an
  explicit bind -- fully closing this window requires holding the port continuously
  (have the daemon bind first and report back), which needs an upstream change.
"""

import os
import socket

import torch.distributed.checkpoint._async_process_executor as _async_process_executor

from xtuner.v1.utils.logger import get_logger


logger = get_logger()

# Default daemon port range "low:high" (high exclusive), outside the ephemeral range
# (>=32768) and clear of the common training MASTER_PORT (29500).
_DEFAULT_PORT_RANGE = "29600:30000"

_PATCHED = False


def _port_range() -> tuple[int, int]:
    """Parse ``XTUNER_DCP_DAEMON_PORT_RANGE`` as ``"low:high"`` (high
    exclusive).

    Returns ``(low, high)`` with ``high > low`` guaranteed (a degenerate or inverted
    range is clamped to a single port).
    """
    raw = os.environ.get("XTUNER_DCP_DAEMON_PORT_RANGE", _DEFAULT_PORT_RANGE)
    low_str, high_str = raw.split(":")
    low, high = int(low_str), int(high_str)
    if high <= low:
        high = low + 1
    return low, high


def _read_kernel_ephemeral_range() -> tuple[int, int] | None:
    """Return the kernel ephemeral port range
    (net.ipv4.ip_local_port_range)."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = f.read().split()[:2]
            return int(low), int(high)
    except Exception:
        return None


def _is_outside_ephemeral_range(
    range_low: int, range_high_exclusive: int, ephemeral_range: tuple[int, int] | None
) -> bool:
    """Whether [range_low, range_high_exclusive) does not intersect the kernel
    ephemeral range.

    The kernel only auto-assigns outbound ports from the ephemeral range, so a daemon port outside it cannot be grabbed
    by a transient connection during the spawn window.
    """
    if ephemeral_range is None:
        return False
    ephemeral_low, ephemeral_high = ephemeral_range
    return range_high_exclusive <= ephemeral_low or range_low > ephemeral_high


def _xtuner_dcp_get_free_port() -> int:
    """Scan for a free port from a fixed range (kept outside the ephemeral
    range by default).

    Only the coordinator rank (global rank 0) calls this; the chosen port is then
    broadcast to all nodes (which all connect to master_addr=node0), so it only needs
    to be free on node0.

    ``SO_REUSEADDR`` is set to mirror the daemon's actual TCPStore listener (c10d's
    ``socket.cpp`` calls ``enableAddressReuse()`` before bind on non-Windows). Matching
    it makes this probe a faithful predictor of the daemon bind: we no longer reject a
    port the daemon could reuse (e.g. lingering TIME_WAIT from a previous run). It does
    NOT weaken in-use detection -- ``SO_REUSEADDR`` still cannot bind over an actively
    listening socket (that would require ``SO_REUSEPORT``).
    """
    range_low, range_high = _port_range()
    ephemeral_range = _read_kernel_ephemeral_range()
    last_error: OSError | None = None
    for port in range(range_low, range_high):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind the wildcard address ("") rather than localhost: the real daemon
            # TCPStore server (GLOO PG init with MASTER_ADDR=node0's FQDN) binds all
            # interfaces so other nodes can reach it, so probing the wildcard is the
            # faithful predictor of the daemon bind. (torch's etcd_server binds localhost
            # only because it is a single-host rendezvous; this daemon is cross-node.)
            sock.bind(("", port))
            # backlog is irrelevant for a probe we close immediately; we only confirm the
            # port can enter LISTEN, mirroring the daemon's listener.
            sock.listen(0)
            # Per-checkpoint evidence emitted at debug level: "outside" means the port is
            # safe; "in-ephemeral" means the race can still trigger. Enable debug logging
            # to grep this at scale.
            status = "outside" if _is_outside_ephemeral_range(port, port + 1, ephemeral_range) else "in-ephemeral"
            logger.debug(
                f"[DCP async_save] DCP daemon port chosen: {port} ({status} ephemeral range {ephemeral_range})"
            )
            return port
        except OSError as error:
            last_error = error
        finally:
            sock.close()
    raise RuntimeError(
        f"[DCP async_save] no free daemon port in [{range_low}, {range_high}); "
        f'set XTUNER_DCP_DAEMON_PORT_RANGE (e.g. "29600:30000") to relocate. last error: {last_error}'
    )


def patch_dcp_async_daemon_port() -> None:
    """Route the DCP async checkpoint daemon port to a fixed range (outside the
    ephemeral range by default)."""
    global _PATCHED
    if _PATCHED:
        return
    _async_process_executor.get_free_port = _xtuner_dcp_get_free_port
    _PATCHED = True

    range_low, range_high = _port_range()
    ephemeral_range = _read_kernel_ephemeral_range()
    if _is_outside_ephemeral_range(range_low, range_high, ephemeral_range):
        logger.debug(
            f"[DCP async_save] patched daemon port selection to range "
            f"[{range_low}, {range_high}), outside kernel ephemeral range {ephemeral_range}"
        )
    else:
        # NOTE: keep the real error signatures ("EADDRINUSE" / "address already in use")
        # out of this benign log line, so a naive grep for those tokens only matches
        # actual daemon failures, not this warning.
        logger.warning(
            f"[DCP async_save] daemon port range [{range_low}, {range_high}) overlaps the kernel "
            f"ephemeral range {ephemeral_range}; the TCPStore daemon-port bind race may persist (the "
            f"port can be grabbed during the spawn window). Set XTUNER_DCP_DAEMON_PORT_RANGE "
            f"to a range below {ephemeral_range[0] if ephemeral_range else 32768} to avoid it."
        )
