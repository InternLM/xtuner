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

# Default daemon port range, outside the ephemeral range (>=32768) and clear of the
# common training MASTER_PORT (29500).
_DEFAULT_PORT_BASE = 29600
_DEFAULT_PORT_SPAN = 400

_PATCHED = False


def _port_base() -> int:
    return int(os.environ.get("XTUNER_DCP_DAEMON_PORT_BASE", str(_DEFAULT_PORT_BASE)))


def _port_span() -> int:
    return max(1, int(os.environ.get("XTUNER_DCP_DAEMON_PORT_SPAN", str(_DEFAULT_PORT_SPAN))))


def _ephemeral_range() -> tuple[int, int] | None:
    """Return the kernel ephemeral port range
    (net.ipv4.ip_local_port_range)."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = f.read().split()[:2]
            return int(lo), int(hi)
    except Exception:
        return None


def _is_outside_ephemeral(base: int, hi_excl: int, eph: tuple[int, int] | None) -> bool:
    """Whether [base, hi_excl) does not intersect the kernel ephemeral range.

    The kernel only auto-assigns outbound ports from the ephemeral range, so a daemon port outside it cannot be grabbed
    by a transient connection during the spawn window.
    """
    if eph is None:
        return False
    eph_lo, eph_hi = eph
    return hi_excl <= eph_lo or base > eph_hi


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
    base = _port_base()
    span = _port_span()
    eph = _ephemeral_range()
    last_err: OSError | None = None
    for port in range(base, base + span):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)
            # Per-checkpoint evidence for verification: every line must say "outside"; any
            # "in-ephemeral" means the race can still trigger. Greppable at scale.
            status = "outside" if _is_outside_ephemeral(port, port + 1, eph) else "in-ephemeral"
            logger.info(f"[DCP async_save] DCP daemon port chosen: {port} ({status} ephemeral range {eph})")
            return port
        except OSError as exc:
            last_err = exc
        finally:
            sock.close()
    raise RuntimeError(
        f"[DCP async_save] no free daemon port in [{base}, {base + span}); "
        f"set XTUNER_DCP_DAEMON_PORT_BASE / XTUNER_DCP_DAEMON_PORT_SPAN to relocate. last error: {last_err}"
    )


def patch_dcp_async_daemon_port() -> None:
    """Route the DCP async checkpoint daemon port to a fixed range (outside the
    ephemeral range by default)."""
    global _PATCHED
    if _PATCHED:
        return
    _async_process_executor.get_free_port = _xtuner_dcp_get_free_port
    _PATCHED = True

    base = _port_base()
    hi_excl = base + _port_span()
    eph = _ephemeral_range()
    if _is_outside_ephemeral(base, hi_excl, eph):
        logger.info(
            f"[DCP async_save] patched daemon port selection to range "
            f"[{base}, {hi_excl}), outside kernel ephemeral range {eph}"
        )
    else:
        # NOTE: keep the real error signatures ("EADDRINUSE" / "address already in use")
        # out of this benign log line, so a naive grep for those tokens only matches
        # actual daemon failures, not this warning.
        logger.warning(
            f"[DCP async_save] daemon port range [{base}, {hi_excl}) overlaps the kernel "
            f"ephemeral range {eph}; the TCPStore daemon-port bind race may persist (the "
            f"port can be grabbed during the spawn window). Set XTUNER_DCP_DAEMON_PORT_BASE "
            f"below {eph[0] if eph else 32768} to avoid it."
        )
