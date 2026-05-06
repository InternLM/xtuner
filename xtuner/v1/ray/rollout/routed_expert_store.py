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
        # Diagnostic counters — help locate double-consume / leak symptoms
        # after the fact without changing runtime behaviour.
        self._n_put = 0
        self._n_get = 0
        self._n_release = 0
        self._n_missing_get = 0
        self._n_missing_release = 0

    def put_tensor(self, tensor: Any) -> str:
        """Take ownership of a tensor via ray.put; return a uuid key.

        NOTE: this places the tensor in the STORE actor's local plasma —
        under heavy traffic this can cause one-node plasma saturation (all
        rollouts across the cluster funnel to whichever node this actor
        runs on).  Prefer ``put_ref`` + worker-side ``ray.put`` for
        multi-node scale.
        """
        self._maybe_gc()
        ref = ray.put(tensor)
        self._n_put += 1
        return self._stash(ref)

    def put_ref(self, wrapped: list) -> str:
        """Register an externally-owned ObjectRef, return a uuid key.

        The caller must wrap the ref in a single-element list.  Ray
        auto-dereferences bare ObjectRef args; the list wrapper bypasses
        that so the store receives the ref itself (not the materialized
        tensor).

        Ownership stays with the original ``ray.put`` caller (typically
        the rollout worker); data lives in that caller's node plasma.
        The store just holds a Python-level strong reference so Ray's
        distributed refcount keeps the object alive across consumers.

        This distributes plasma pressure across all rollout nodes instead
        of funneling it to the store's node.  Trade-off: if the owner
        worker dies, the ref's object is lost; with ``put_tensor`` only
        the store dying would lose data.
        """
        self._maybe_gc()
        if not (isinstance(wrapped, list) and len(wrapped) == 1):
            raise TypeError(f"put_ref expects [ObjectRef] to bypass auto-deref, got {type(wrapped)}")
        ref = wrapped[0]
        if not isinstance(ref, ObjectRef):
            raise TypeError(f"put_ref expects [ObjectRef], got [{type(ref)}]")
        self._n_put += 1
        return self._stash(ref)

    def get_ref(self, key: str) -> ObjectRef:
        """Return the stashed ObjectRef; caller runs ``ray.get`` to
        materialize."""
        entry = self._store.get(key)
        if entry is None:
            self._n_missing_get += 1
            raise KeyError(f"RoutedExpertStore: key not found: {key}")
        self._store[key] = (entry[0], time.monotonic())
        self._n_get += 1
        return entry[0]

    def release(self, key: str) -> None:
        if self._store.pop(key, None) is None:
            self._n_missing_release += 1
        else:
            self._n_release += 1

    def release_many(self, keys: list[str]) -> None:
        for k in keys:
            self.release(k)

    def stats(self) -> dict:
        return {
            "live": len(self._store),
            "ttl_sec": self._ttl,
            "n_put": self._n_put,
            "n_get": self._n_get,
            "n_release": self._n_release,
            "n_missing_get": self._n_missing_get,
            "n_missing_release": self._n_missing_release,
        }

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
