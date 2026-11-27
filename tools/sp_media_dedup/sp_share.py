#!/usr/bin/env python3
"""CC8 D2 shared-media storage layer: node-local POSIX shm, read-only consumer maps.

Design (per plan §必须解决):
- unit of sharing = one PACK's collated pixel_values (+ grid) — the level at which
  memory actually dedups (per-sample sharing would be re-copied by the collator's cat).
- key = (run_id, pack_idx); the segment header records the media content sha list and
  the processing-config fingerprint — same path ≠ same content is guarded by the
  producer recording post-read content hashes, consumers verifying them.
- exactly-one-producer via O_EXCL create; consumers attach read-only, verify header,
  refcount++, release → refcount-- → unlink at zero.
- every wait is bounded (timeout); on timeout/producer-failure the consumer falls back
  to native local processing, which is bitwise identical because processing is
  deterministic (H2) — fallback is never a correctness divergence.
- capacity budget per round; publish over budget refuses (→ all-local).
- cleanup touches only own-prefixed segments; stale WRITING segments past their
  deadline may be marked ABANDONED by anyone and unlinked at refcount 0.

This file is the storage/protocol layer for the D2 CPU prototype. Trainer/collator
integration is a separate patch (D2-full) and is NOT done here.
"""
from __future__ import annotations

import fcntl
import mmap
import os
import struct
import time
from dataclasses import dataclass, field

MAGIC = b"CC8SHM1"
HDR_FMT = "<8sIIQQQQQ16sQQ"  # magic, state, ver, producer_starttime? -> see fields below
STATE_WRITING, STATE_READY, STATE_ABANDONED = 1, 2, 3
HDR_SIZE = 256  # fixed header area, meta JSON follows, then payload


@dataclass
class SegmentMeta:
    run_id: str = ""
    pack_idx: int = -1
    producer_pid: int = 0
    content_sha16: str = ""      # blake2b16 of the payload (collated pixel bytes)
    media_sha_list: str = ""     # json list of per-media content sha16, order = grid order
    cfg_sha16: str = ""          # processing-config fingerprint
    shape: tuple = field(default_factory=tuple)
    dtype: str = ""
    payload_nbytes: int = 0
    grid_shape: tuple = field(default_factory=tuple)
    grid_nbytes: int = 0

    def to_json(self) -> str:
        import json
        return json.dumps({
            "run_id": self.run_id, "pack_idx": self.pack_idx, "producer_pid": self.producer_pid,
            "content_sha16": self.content_sha16, "media_sha_list": self.media_sha_list,
            "cfg_sha16": self.cfg_sha16, "shape": list(self.shape), "dtype": self.dtype,
            "payload_nbytes": self.payload_nbytes, "grid_shape": list(self.grid_shape),
            "grid_nbytes": self.grid_nbytes})

    @classmethod
    def from_json(cls, s: str) -> "SegmentMeta":
        import json
        d = json.loads(s)
        d["shape"] = tuple(d["shape"])
        d["grid_shape"] = tuple(d["grid_shape"])
        return cls(**d)


def _seg_path(name: str) -> str:
    return f"/dev/shm/{name}"


class PackShare:
    """Producer/consumer protocol over /dev/shm segments named `{prefix}/{key}`."""

    def __init__(self, prefix: str, budget_bytes: int, now=time.time):
        self.prefix = prefix.lstrip("/")
        self.budget_bytes = budget_bytes
        self._now = now

    # ---------- accounting ----------
    def current_usage(self) -> int:
        total = 0
        try:
            for fn in os.listdir("/dev/shm"):
                if fn.startswith(self.prefix):
                    try:
                        total += os.stat(_seg_path(fn)).st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def sweep_own(self, force: bool = False) -> int:
        """Unlink own-prefixed segments that are abandoned/expired (or all if force).
        Never touches other prefixes. Returns number unlinked."""
        n = 0
        for fn in list(self._own_names()):
            p = _seg_path(fn)
            try:
                with open(p, "rb") as f:
                    hdr = f.read(HDR_SIZE)
                if len(hdr) < HDR_SIZE or hdr[:8] != MAGIC:
                    continue
                state, = struct.unpack_from("<I", hdr, 8)
                deadline, refcnt = struct.unpack_from("<QQ", hdr, 24)
                expired_writing = state == STATE_WRITING and self._now() > deadline
                if force or state == STATE_ABANDONED or expired_writing:
                    if refcnt == 0 or expired_writing or force:
                        os.unlink(p)
                        n += 1
            except (OSError, struct.error):
                continue
        return n

    def _own_names(self):
        try:
            names = os.listdir("/dev/shm")
        except OSError:
            return
        for fn in names:
            if fn.startswith(self.prefix):
                yield fn

    # ---------- producer ----------
    def try_publish(self, key: str, meta: SegmentMeta, payload: bytes, grid_bytes: bytes,
                    lease_s: float) -> bool:
        """Create segment exclusively, write payload, mark READY. False if exists/over-budget."""
        name = f"{self.prefix}-{key}"
        path = _seg_path(name)
        total = HDR_SIZE + len(meta.to_json().encode()) + len(payload) + len(grid_bytes)
        if self.current_usage() + total > self.budget_bytes:
            return False
        # refcount lock file
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            return False
        try:
            os.ftruncate(fd, total)
            with mmap.mmap(fd, total, prot=mmap.PROT_WRITE) as mm:
                mm[0:8] = MAGIC
                struct.pack_into("<I", mm, 8, STATE_WRITING)
                struct.pack_into("<Q", mm, 16, self._now() + lease_s)
                struct.pack_into("<Q", mm, 24, 1)  # refcount: producer
                meta_json = meta.to_json().encode()
                struct.pack_into("<Q", mm, 32, len(meta_json))
                mm[HDR_SIZE:HDR_SIZE + len(meta_json)] = meta_json
                off = HDR_SIZE + len(meta_json)
                mm[off:off + len(payload)] = payload
                mm[off + len(payload):off + len(payload) + len(grid_bytes)] = grid_bytes
                # mark READY last (publish barrier)
                struct.pack_into("<I", mm, 8, STATE_READY)
                mm.flush()
            return True
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            return False
        finally:
            os.close(fd)

    def abandon(self, key: str) -> None:
        try:
            path = _seg_path(f"{self.prefix}-{key}")
            fd = os.open(path, os.O_RDWR)
            with mmap.mmap(fd, HDR_SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
                if mm[0:8] == MAGIC:
                    struct.pack_into("<I", mm, 8, STATE_ABANDONED)
            os.close(fd)
        except (OSError, ValueError):
            pass

    # ---------- consumer ----------
    def attach(self, key: str, timeout_s: float, poll_s: float = 0.05,
               expect_content_sha: str | None = None):
        """Wait for READY, map read-only, bump refcount. Returns (mm, meta, grid_off) or None."""
        path = _seg_path(f"{self.prefix}-{key}")
        deadline = self._now() + timeout_s
        while self._now() < deadline:
            try:
                fd = os.open(path, os.O_RDONLY)
                break
            except FileNotFoundError:
                time.sleep(poll_s)
        else:
            return None
        try:
            while self._now() < deadline:
                mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                if mm[0:8] != MAGIC or len(mm) < HDR_SIZE:
                    mm.close(); os.close(fd); return None
                state, = struct.unpack_from("<I", mm, 8)
                if state == STATE_ABANDONED:
                    mm.close(); os.close(fd); return None
                if state == STATE_READY:
                    meta_len, = struct.unpack_from("<Q", mm, 32)
                    meta = SegmentMeta.from_json(mm[HDR_SIZE:HDR_SIZE + meta_len].decode())
                    if expect_content_sha and meta.content_sha16 != expect_content_sha:
                        mm.close(); os.close(fd); return None
                    self._ref(path, +1)
                    payload_off = HDR_SIZE + meta_len
                    grid_off = payload_off + meta.payload_nbytes
                    return _SharedView(mm, meta, payload_off, grid_off, path)
                mm.close()
                time.sleep(poll_s)
            return None  # timeout; producer still WRITING
        finally:
            os.close(fd)  # mapping keeps the file alive after fd close

    def release(self, shared: "_SharedView") -> None:
        shared.mm.close()
        self._ref(shared.path, -1)

    def _ref(self, path: str, delta: int) -> None:
        lock = path + ".reflock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            main = os.open(path, os.O_RDWR)
            try:
                buf = os.pread(main, HDR_SIZE, 0)
                if buf[:8] == MAGIC:
                    refcnt, = struct.unpack_from("<Q", buf, 24)
                    refcnt = max(0, refcnt + delta)
                    os.pwrite(main, struct.pack("<Q", refcnt), 24)
                    if refcnt == 0:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                        try:
                            os.unlink(lock)
                        except OSError:
                            pass
            finally:
                os.close(main)
        finally:
            os.close(fd)


class _SharedView:
    def __init__(self, mm, meta, payload_off, grid_off, path):
        self.mm = mm
        self.meta = meta
        self.payload_off = payload_off
        self.grid_off = grid_off
        self.path = path

    def pixel_tensor(self):
        """Zero-copy read-only tensor over the shared payload."""
        import numpy as np
        import torch
        dt = getattr(torch, self.meta.dtype)
        np_arr = np.frombuffer(self.mm, dtype=dt, count=int(np.prod(self.meta.shape)) if self.meta.shape else 0,
                               offset=self.payload_off)
        t = torch.from_numpy(np_arr.reshape(self.meta.shape))
        return t

    def grid_tensor(self):
        import numpy as np
        import torch
        n = self.meta.grid_nbytes // 4
        np_arr = np.frombuffer(self.mm, dtype=np.int32, count=n, offset=self.grid_off)
        return torch.from_numpy(np_arr.reshape(self.meta.grid_shape))
