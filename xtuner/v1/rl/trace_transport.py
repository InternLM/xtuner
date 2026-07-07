from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Protocol

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.trace import inject_trace_context


_TRACE_CARRIER_EXTRA_FIELD = "_xtuner_trace_carrier"
TRACE_CARRIER_EXTRA_FIELD = _TRACE_CARRIER_EXTRA_FIELD

_MISSING = object()

class RayRemoteMethod(Protocol):
    def remote(self, *args: Any, **kwargs: Any) -> Any: ...


class _RolloutTraceCarrier:
    field_name = _TRACE_CARRIER_EXTRA_FIELD

    @classmethod
    @contextmanager
    def attach_temporarily(cls, rollout_state: RolloutState, carrier: Mapping[str, str]):
        if not carrier:
            yield
            return

        extra_fields = rollout_state.extra_fields
        if extra_fields is None:
            extra_fields = {}
            rollout_state.extra_fields = extra_fields
        previous_carrier = extra_fields.get(cls.field_name, _MISSING)
        extra_fields[cls.field_name] = dict(carrier)
        try:
            yield
        finally:
            if previous_carrier is _MISSING:
                extra_fields.pop(cls.field_name, None)
            else:
                extra_fields[cls.field_name] = previous_carrier

    @classmethod
    def pop_from_rollout_state(cls, rollout_state: RolloutState) -> dict[str, str] | None:
        if not isinstance(rollout_state, RolloutState):
            raise TypeError("extract_rollout_trace_parent_carrier() requires a RolloutState")

        extra_fields = rollout_state.extra_fields
        if not extra_fields:
            return None
        return cls.extract_from_mapping(extra_fields, remove=True)

    @classmethod
    def extract_from_mapping(cls, payload: Mapping[str, Any] | None, *, remove: bool = False) -> dict[str, str] | None:
        if not isinstance(payload, Mapping):
            return None
        carrier = payload.get(cls.field_name)
        if remove and isinstance(payload, dict):
            payload.pop(cls.field_name, None)
        if not isinstance(carrier, Mapping):
            return None
        return {str(key): str(value) for key, value in carrier.items()}


def trace_remote(
    remote_method: RayRemoteMethod,
    *args: Any,
    target: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Call a Ray remote method and propagate the current trace context."""

    rollout_state = resolve_rollout_state_target(args, kwargs, target=target, owner="trace_remote")
    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    with _RolloutTraceCarrier.attach_temporarily(rollout_state, carrier):
        return remote_method.remote(*args, **kwargs)


def extract_rollout_trace_parent_carrier(rollout_state: RolloutState) -> dict[str, str] | None:
    """Remove and return the parent trace carrier attached to a rollout call."""

    return _RolloutTraceCarrier.pop_from_rollout_state(rollout_state)


def pop_rollout_trace_carrier(rollout_state: RolloutState) -> dict[str, str] | None:
    """Compatibility wrapper for existing internal tests and call sites."""

    return extract_rollout_trace_parent_carrier(rollout_state)


def resolve_rollout_state_target(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    target: Any | None = None,
    owner: str,
) -> RolloutState:
    if target is not None:
        if not isinstance(target, RolloutState):
            raise TypeError(f"{owner} target must be a RolloutState")
        return target

    rollout_states: list[RolloutState] = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, RolloutState):
            rollout_states.append(value)
            continue
        if _contains_rollout_state_collection(value):
            raise TypeError(f"{owner} supports a single RolloutState, not a RolloutState collection")

    if len(rollout_states) != 1:
        raise ValueError(f"{owner} requires exactly one RolloutState argument")
    return rollout_states[0]


def _contains_rollout_state_collection(value: Any) -> bool:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return False
    return any(isinstance(item, RolloutState) for item in value)


__all__ = [
    "trace_remote",
]
