"""Linux file-description record locks for shared-filesystem cleanup ownership.

OFD locks share POSIX record-lock coordination, while keeping ownership on the
open file description rather than the process. Closing an unrelated descriptor
cannot release them; explicit descriptor transfer to a guardian remains valid.
Filesystem-wide coordination still requires deployment-level cross-node tests.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import sys
import uuid
from pathlib import Path
from typing import TextIO


_PROTOCOL = b"xtuner-cleanup-ofd-v1\n"


def open_lock(path: Path) -> TextIO:
    """Open a protocol-marked lock, refusing unverified historical namespaces.

    Args:
        path (Path): Persistent lock path, never unlinked while a run exists.

    Returns:
        TextIO: Independently opened, non-inheritable read/write description.
    """
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except FileNotFoundError:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            os.write(descriptor, _PROTOCOL)
            os.fsync(descriptor)
            try:
                # Never replace a lock inode: its owner may still be alive.
                os.link(temporary, path)
            except FileExistsError:
                replacement = os.open(path, os.O_RDWR | os.O_CLOEXEC)
                os.close(descriptor)
                descriptor = replacement
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            os.close(descriptor)
            raise
        finally:
            temporary.unlink(missing_ok=True)
    try:
        if os.pread(descriptor, len(_PROTOCOL) + 1, 0) != _PROTOCOL:
            raise OSError(errno.EOPNOTSUPP, "Unverified cleanup lock protocol; preserve this run for offline recovery")
        return os.fdopen(descriptor, "a+")
    except BaseException:
        os.close(descriptor)
        raise


def acquire(lock: TextIO, *, shared: bool = False, blocking: bool = False) -> None:
    """Acquire and verify an OFD lock, without a weaker API fallback.

    Args:
        lock (TextIO): Read/write lock description from ``open_lock``.
        shared (bool): Acquire a shared admission lock.
        blocking (bool): Wait in the kernel rather than reporting contention.
    """
    _require_abi()
    command = fcntl.F_OFD_SETLKW if blocking else fcntl.F_OFD_SETLK
    request = _RecordLock(fcntl.F_RDLCK if shared else fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
    try:
        fcntl.fcntl(lock, command, bytes(request))
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise BlockingIOError(errno.EAGAIN, "Cleanup lock has another owner") from None
        raise
    try:
        # F_GETLK must expose a conflicting record lock through a separate
        # description. With shared owners it may return any existing reader,
        # including a remote process PID, rather than this description's OFD
        # owner (-1). This local capability check does not prove cross-node
        # exclusion; the acquired OFD lock remains the ownership contract.
        descriptor = os.open(f"/proc/self/fd/{lock.fileno()}", os.O_RDWR | os.O_CLOEXEC)
        try:
            query = _RecordLock(fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
            result = _RecordLock.from_buffer_copy(fcntl.fcntl(descriptor, fcntl.F_GETLK, bytes(query)))
            if result.kind == fcntl.F_UNLCK:
                raise OSError(errno.EOPNOTSUPP, "Filesystem does not expose OFD ownership in POSIX record locks")
        finally:
            os.close(descriptor)
    except BaseException:
        release(lock)
        raise


def release(lock: TextIO) -> None:
    """Explicitly unlock a transferred description only after all work is reaped.

    Args:
        lock (TextIO): Shared description whose remaining holders must go idle.
    """
    _require_abi()
    fcntl.fcntl(lock, fcntl.F_OFD_SETLK, bytes(_RecordLock(fcntl.F_UNLCK, os.SEEK_SET, 0, 0, 0)))


class _RecordLock(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_short),
        ("whence", ctypes.c_short),
        ("start", ctypes.c_longlong),
        ("length", ctypes.c_longlong),
        ("pid", ctypes.c_int),
    ]


def _require_abi() -> None:
    if (
        sys.platform != "linux"
        or ctypes.sizeof(ctypes.c_long) != 8
        or ctypes.sizeof(_RecordLock) != 32
        or _RecordLock.start.offset != 8
        or not hasattr(fcntl, "F_OFD_SETLK")
        or not hasattr(fcntl, "F_OFD_SETLKW")
    ):
        raise OSError(errno.EOPNOTSUPP, "Cleanup requires Linux 64-bit OFD record locks")
