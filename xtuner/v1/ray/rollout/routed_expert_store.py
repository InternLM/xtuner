"""Named Ray actor that owns all ``routed_experts`` ObjectRefs.

Motivation:
    Rollout workers produce ``routed_experts`` ndarrays per turn.  These must
    survive HTTP/JSON roundtrips through the agent sandbox (minutes to hours)
    and be concat'd at training time.  Historically they travelled as
    cloudpickled ObjectRef hex strings, whose distributed refcount is fragile
    at that timescale — heartbeat flakes or owner restarts evict the object
    and training later hits ``ObjectFetchTimedOutError``.  A second failure
    mode: the old concat-on-rollout path called ``ray.get + ray.internal.free``
    inline, so any exception between get and free would silently drop the
    data; a retry would then find nothing.

    Routing all ``routed_experts`` refs through a dedicated long-lived store
    decouples lifetime from the lmdeploy worker lifecycle, and moves release
    out of the rollout path entirely — the training worker calls
    ``release_many`` only after it has successfully consumed the keys.
    Transport becomes a short uuid string instead of a cloudpickled ref blob,
    and retries are safe because rollout never frees.
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
_DEFAULT_TTL_SEC = int(os.environ.get("XTUNER_ROUTED_EXPERT_TTL_SEC", 24 * 3600))
_DEFAULT_GC_INTERVAL_SEC = int(os.environ.get("XTUNER_ROUTED_EXPERT_GC_INTERVAL_SEC", 300))


@ray.remote(num_cpus=0)
class RoutedExpertStore:
    """Dedicated store actor for routed_experts ObjectRefs.

    Each registered ref is held by the store's Python dict, which pins the
    Ray-level refcount across HTTP roundtrips and retries.  GC runs inline
    on each ``put_ref`` call as a safety net for leaked keys (aborted /
    filtered samples that never reach training); it is not a correctness
    mechanism.

    Args:
        ttl_sec (int): Time-to-live for an unaccessed key before the inline
            GC sweep evicts it.  Defaults to ``XTUNER_ROUTED_EXPERT_TTL_SEC``
            env var or 24 hours.
        gc_interval_sec (int): Minimum seconds between GC sweeps.  Defaults
            to ``XTUNER_ROUTED_EXPERT_GC_INTERVAL_SEC`` env var or 5 minutes.
    """

    def __init__(self, ttl_sec: int = _DEFAULT_TTL_SEC, gc_interval_sec: int = _DEFAULT_GC_INTERVAL_SEC):
        self._store: dict[str, tuple[ObjectRef, float]] = {}
        self._ttl = ttl_sec
        self._gc_interval = gc_interval_sec
        self._last_gc = time.monotonic()
        self._n_put = 0
        self._n_get = 0
        self._n_release = 0
        self._n_missing_get = 0
        self._n_missing_release = 0

    def put_ref(self, wrapped: list) -> str:
        """Register an externally-owned ObjectRef and return a uuid key.

        The caller must wrap the ref in a single-element list.  Ray auto-
        dereferences bare ObjectRef args; the list wrapper bypasses that so
        the store receives the ref itself rather than the materialized tensor.

        Ownership stays with the original ``ray.put`` caller (typically the
        rollout worker); the store just holds a Python-level strong reference
        so Ray's distributed refcount keeps the object alive across consumers.
        This distributes plasma pressure across all rollout nodes instead of
        funneling it to the store's node.

        Args:
            wrapped (list): Single-element list holding the ObjectRef.

        Returns:
            str: Short uuid key that later callers pass to ``get_ref``.
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
        """Return the stashed ObjectRef.  Caller runs ``ray.get`` to
        materialize.

        Args:
            key (str): Key returned by a prior ``put_ref``.

        Returns:
            ObjectRef: The stashed ref.
        """
        entry = self._store.get(key)
        if entry is None:
            self._n_missing_get += 1
            raise KeyError(f"RoutedExpertStore: key not found: {key}")
        self._store[key] = (entry[0], time.monotonic())
        self._n_get += 1
        return entry[0]

    def release(self, key: str) -> None:
        """Drop the stored ref for ``key`` and force-evict the object from
        plasma.  Idempotent: no-op if already released.

        Uses ``ray.internal.free`` on top of the Python-level pop so plasma is
        reclaimed immediately instead of waiting for Ray's distributed
        refcount GC (which lags under heavy rollout pressure).  Callers just
        invoke ``release`` / ``release_many`` and don't reason about plasma.
        """
        entry = self._store.pop(key, None)
        if entry is None:
            self._n_missing_release += 1
            return
        ref, _ = entry
        try:
            ray.internal.free([ref], local_only=False)
        except Exception as e:
            # free is best-effort — log and continue; the Python pop above
            # still drops our strong ref so Ray GC catches it eventually.
            get_logger().debug(f"RoutedExpertStore: ray.internal.free failed for {key}: {e}")
        self._n_release += 1

    def release_many(self, keys: list[str]) -> None:
        """Batch release.

        Idempotent per key.
        """
        for k in keys:
            self.release(k)

    def stats(self) -> dict:
        """Return live-count + lifetime operation counters for observability.

        Returns:
            dict: Snapshot of live key count, TTL, and put/get/release counters.
        """
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
            entry = self._store.pop(k, None)
            if entry is None:
                continue
            ref, _ = entry
            try:
                ray.internal.free([ref], local_only=False)
            except Exception as e:
                get_logger().debug(f"RoutedExpertStore GC: ray.internal.free failed for {k}: {e}")
        if stale:
            get_logger().warning(
                f"RoutedExpertStore GC: evicted {len(stale)} stale keys (ttl={self._ttl}s, "
                f"remaining={len(self._store)})"
            )


_handle_cache: Any = None


def get_store():
    """Process-local cached handle to the singleton store actor.

    Fast path: ``ray.get_actor`` if another caller has already created it.
    Slow path: create the actor under the reserved name; race with concurrent
    creators is handled by catching the "name already taken" ValueError and
    re-looking-up.

    Returns:
        ActorHandle: Handle to the ``RoutedExpertStore`` actor.
    """
    global _handle_cache
    if _handle_cache is not None:
        return _handle_cache

    try:
        _handle_cache = ray.get_actor(_STORE_NAME)
        return _handle_cache
    except ValueError:
        pass

    import time as _time

    for attempt in range(10):
        try:
            _handle_cache = RoutedExpertStore.options(name=_STORE_NAME).remote()
            return _handle_cache
        except ValueError as exc:
            try:
                _handle_cache = ray.get_actor(_STORE_NAME)
                return _handle_cache
            except ValueError:
                get_logger().debug(f"RoutedExpertStore bootstrap retry {attempt}: {exc}")
                _time.sleep(0.2 * (attempt + 1))
                continue

    raise RuntimeError(f"RoutedExpertStore: failed to acquire named actor {_STORE_NAME!r} after retries")


def collect_routed_expert_keys_from_env(env: Any) -> list[str]:
    """Collect every ``routed_experts`` store key attached to an env item.

    Keys can live in two places in the current schema:

    * **Legacy single-turn** — ``env.rollout.extra_info["routed_experts"]``
      is a single ``{"key": str, "length": int}`` dict (one delta per sample).
    * **Agent multi-turn** — each assistant message carries its own
      ``extra_info["routed_experts"] = {"key", "length"}``.  Messages live
      in one of ``agent_message_dict`` / ``message_dict`` /
      ``agent_state_dict`` inside either ``env.rollout.extra_info`` (where
      :class:`AgentEnvironment` stashes them) or ``env.agent.extra_info``
      (where :class:`InstallAgentEnvironment` stashes them).  The container
      value is a dict whose values are message-lists.

    Used by both the dataflow skipped/failed release path and the replay
    buffer expired/aborted strip path — without this, per-message keys
    leak when samples never reach training (``_pop_routed_experts_from_extra_info``
    used to only handle the legacy single-turn form and was a no-op for
    agent rollouts).

    Args:
        env (Any): An ``RLEnvDataItem``-shaped object with ``rollout`` and
            ``agent`` attributes.  Safe to pass a partial object.

    Returns:
        list[str]: Deduped store keys to release.  Empty list if nothing
            is attached.
    """
    keys: list[str] = []
    rollout_extra = getattr(getattr(env, "rollout", None), "extra_info", None) or {}
    agent_extra = getattr(getattr(env, "agent", None), "extra_info", None) or {}

    # Legacy single-turn wire: one dict on rollout.extra_info.
    if isinstance(rollout_extra, dict):
        legacy = rollout_extra.get("routed_experts")
        if isinstance(legacy, dict) and isinstance(legacy.get("key"), str):
            keys.append(legacy["key"])

    # Agent multi-turn wire: walk message containers on both sides.
    message_containers = ("agent_message_dict", "message_dict", "agent_state_dict")
    for side in (rollout_extra, agent_extra):
        if not isinstance(side, dict):
            continue
        for container_key in message_containers:
            container = side.get(container_key)
            if not isinstance(container, dict):
                continue
            for maybe_msgs in container.values():
                if not isinstance(maybe_msgs, list):
                    continue
                for msg in maybe_msgs:
                    if not isinstance(msg, dict):
                        continue
                    ei = msg.get("extra_info")
                    if not isinstance(ei, dict):
                        continue
                    ref = ei.get("routed_experts")
                    if isinstance(ref, dict) and isinstance(ref.get("key"), str):
                        keys.append(ref["key"])

    # Dedup; same key can appear in e.g. agent_state_dict and message_dict.
    return list(dict.fromkeys(keys))
