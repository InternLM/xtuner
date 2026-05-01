"""Named Ray actor that owns all ``routed_experts`` ObjectRefs.

Motivation:
    lmdeploy workers produce ``routed_experts`` per rollout and encode them as
    Ray ``ObjectRef`` values.  Those refs travel through HTTP/JSON (the agent
    sandbox roundtrip) as cloudpickled hex strings, spending minutes to hours
    outside any Python process.  Ray's distributed refcount + cloudpickle
    out-of-band tracking is fragile at that timescale — under heartbeat flakes
    or owner restart the object can be evicted, and training later hits
    ``ObjectFetchTimedOutError`` with "no locations were found".

    Routing the refs through a dedicated long-lived store decouples their
    lifetime from lmdeploy's lifecycle.  Transport between actors/processes
    becomes a short uuid string (``put_tensor`` → ``get_ref`` → ``release``)
    instead of a cloudpickled ObjectRef blob, so the failure mode is replaced
    by a plain dict lookup that either hits or misses.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import ray
from ray import ObjectRef

from xtuner.v1.utils import get_logger


_STORE_NAME = "routed_expert_store"
# Generous default — covers long rollouts that may not be consumed immediately
# (e.g. eval groups, aborted tasks awaiting GC, etc).  Overridable via env var
# for quick tuning without re-editing the source.
_DEFAULT_TTL_SEC = int(os.environ.get("XTUNER_ROUTED_EXPERT_TTL_SEC", 24 * 3600))
_DEFAULT_GC_INTERVAL_SEC = int(os.environ.get("XTUNER_ROUTED_EXPERT_GC_INTERVAL_SEC", 300))


@ray.remote(num_cpus=0)
class RoutedExpertStore:
    """Dedicated store actor for routed_experts ObjectRefs.

    Every ref in ``self._store`` has the store as its Ray owner (via
    ``ray.put``), so its lifetime is bounded purely by the store's Python
    dict — no distributed heartbeats or cloudpickle TTL involved.

    GC runs inline on each ``put_*`` call: if the last sweep happened more
    than ``gc_interval_sec`` ago, stale keys older than ``ttl_sec`` are
    evicted.  Keeps the store honest without a background task.
    """

    def __init__(self, ttl_sec: int = _DEFAULT_TTL_SEC, gc_interval_sec: int = _DEFAULT_GC_INTERVAL_SEC):
        self._store: dict[str, tuple[ObjectRef, float]] = {}
        self._ttl = ttl_sec
        self._gc_interval = gc_interval_sec
        self._last_gc = time.monotonic()

    def put_tensor(self, tensor: Any) -> str:
        """Take ownership of a tensor via ray.put; return a uuid key."""
        self._maybe_gc()
        ref = ray.put(tensor)
        return self._stash(ref)

    def get_ref(self, key: str) -> ObjectRef:
        """Return the stashed ObjectRef; caller runs ``ray.get`` to
        materialize."""
        entry = self._store.get(key)
        if entry is None:
            raise KeyError(f"RoutedExpertStore: key not found: {key}")
        self._store[key] = (entry[0], time.monotonic())
        return entry[0]

    def release(self, key: str) -> None:
        self._store.pop(key, None)

    def release_many(self, keys: list[str]) -> None:
        for k in keys:
            self._store.pop(k, None)

    def stats(self) -> dict:
        return {"count": len(self._store), "ttl_sec": self._ttl}

    def _stash(self, ref: ObjectRef) -> str:
        key = uuid.uuid4().hex
        self._store[key] = (ref, time.monotonic())
        return key

    def _maybe_gc(self) -> None:
        now = time.monotonic()
        if now - self._last_gc < self._gc_interval:
            return
        self._last_gc = now
        stale = [k for k, (_, t) in self._store.items() if now - t > self._ttl]
        for k in stale:
            self._store.pop(k, None)
        if stale:
            get_logger().warning(
                f"RoutedExpertStore GC: evicted {len(stale)} stale keys (ttl={self._ttl}s, remaining={len(self._store)})"
            )


_handle_cache: Any = None


def get_store():
    """Process-local cached handle to the singleton store actor.

    First tries ``ray.get_actor`` (fast path if another caller created it
    already).  On NotFound, tries to create; if creation races with another
    caller (raises because name is taken), falls back to another lookup.
    """
    global _handle_cache
    if _handle_cache is not None:
        return _handle_cache

    # Fast path: already exists.
    try:
        _handle_cache = ray.get_actor(_STORE_NAME)
        return _handle_cache
    except ValueError:
        pass

    # Slow path: try to create.  Race with concurrent callers is handled by
    # catching the "name already taken" case and re-looking-up.
    import time as _time

    for attempt in range(10):
        try:
            _handle_cache = RoutedExpertStore.options(name=_STORE_NAME).remote()
            return _handle_cache
        except ValueError as exc:
            # Either "name already taken" (someone else won) or transient
            # issue.  Try to look up; if that also fails, back off.
            try:
                _handle_cache = ray.get_actor(_STORE_NAME)
                return _handle_cache
            except ValueError:
                get_logger().debug(f"RoutedExpertStore bootstrap retry {attempt}: {exc}")
                _time.sleep(0.2 * (attempt + 1))
                continue

    raise RuntimeError(f"RoutedExpertStore: failed to acquire named actor {_STORE_NAME!r} after retries")
