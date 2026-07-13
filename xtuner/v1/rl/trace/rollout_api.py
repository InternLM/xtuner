from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any, Protocol, TypeVar

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils import get_logger

from . import api as trace_api
from . import otel_utils
from .runtime import is_trace_enabled


F = TypeVar("F", bound=Callable[..., Any])

TRACE_ROLLOUT_ENABLED_ENV = "XTUNER_TRACE_ENABLE_ROLLOUT"
TRACE_CARRIER_EXTRA_FIELD = "_xtuner_trace_carrier"
TRACE_CALL_CHAIN_EXTRA_FIELD = "_xtuner_trace_call_chain"


class _RayRemoteMethod(Protocol):
    def remote(self, *args: Any, **kwargs: Any) -> Any: ...


def is_rollout_trace_enabled() -> bool:
    return os.environ.get(TRACE_ROLLOUT_ENABLED_ENV) == "1" and is_trace_enabled()


def trace_rollout_endpoint(
    span_name: str,
    *,
    target_arg: str = "rollout_state",
    initial_attributes: Callable[[Any, Any], Mapping[str, Any]] | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("trace_rollout_endpoint() only supports async functions")

        signature = inspect.signature(func)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_rollout_trace_enabled():
                return await func(*args, **kwargs)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if target_arg not in bound.arguments:
                raise TypeError(f"trace_rollout_endpoint target argument {target_arg!r} was not bound")
            target_value = bound.arguments[target_arg]
            if isinstance(target_value, (list, tuple)) or not isinstance(target_value, RolloutState):
                get_logger().warning(
                    f"XTuner rollout trace disabled for this endpoint: span={span_name!r}, "
                    f"target_arg={target_arg!r}, target_type={type(target_value).__name__}."
                )
                return await func(*args, **kwargs)
            rollout_state = target_value

            owner = bound.arguments.get("self", args[0] if args else None)
            attributes = (
                dict(initial_attributes(owner, target_value))
                if initial_attributes is not None
                else rollout_state_initial_attributes(rollout_state)
            )
            attributes.setdefault("xtuner.stage", span_name)
            parent_carrier = extract_rollout_trace_parent_carrier(rollout_state)

            with _attach_rollout_call_chain(rollout_state, span_name, parent_carrier) as (
                call_chain,
                cleanup_call_chain_on_exit,
            ):
                attributes["xtuner.span_name_path"] = call_chain
                with trace_api.trace_span(span_name, attributes=attributes, parent_carrier=parent_carrier):
                    result = await func(*args, **kwargs)
                    result_rollout_state = result if isinstance(result, RolloutState) else rollout_state
                    trace_api.set_trace_attributes(rollout_state_final_attributes(result_rollout_state))
                    if cleanup_call_chain_on_exit and isinstance(result, RolloutState):
                        result.extra_fields.pop(TRACE_CALL_CHAIN_EXTRA_FIELD, None)
                    return result

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_rollout_remote(
    remote_method: _RayRemoteMethod,
    *args: Any,
    target: RolloutState | None = None,
    **kwargs: Any,
) -> Any:
    if not is_rollout_trace_enabled():
        return remote_method.remote(*args, **kwargs)

    rollout_state = _resolve_rollout_state_target(args, kwargs, target=target, owner="trace_rollout_remote")
    carrier = trace_api.inject_trace_context({})
    with attach_rollout_trace_carrier(rollout_state, carrier):
        return remote_method.remote(*args, **kwargs)


@contextmanager
def attach_rollout_trace_carrier(rollout_state: RolloutState, carrier: Mapping[str, str]):
    if not carrier:
        yield
        return

    extra_fields = rollout_state.extra_fields
    had_previous_carrier = TRACE_CARRIER_EXTRA_FIELD in extra_fields
    previous_carrier = extra_fields.get(TRACE_CARRIER_EXTRA_FIELD)
    extra_fields[TRACE_CARRIER_EXTRA_FIELD] = dict(carrier)
    try:
        yield
    finally:
        if had_previous_carrier:
            extra_fields[TRACE_CARRIER_EXTRA_FIELD] = previous_carrier
        else:
            extra_fields.pop(TRACE_CARRIER_EXTRA_FIELD, None)


def extract_rollout_trace_parent_carrier(rollout_state: RolloutState) -> dict[str, str] | None:
    extra_fields = rollout_state.extra_fields
    carrier = extra_fields.pop(TRACE_CARRIER_EXTRA_FIELD, None)
    if not isinstance(carrier, Mapping):
        return None
    return {str(key): str(value) for key, value in carrier.items()}


def rollout_state_initial_attributes(rollout_state: RolloutState) -> dict[str, Any]:
    extra_fields = rollout_state.extra_fields
    attributes: dict[str, Any] = {
        "xtuner.status": rollout_state.status.value,
        "xtuner.rollout_id": rollout_state.rollout_id,
        "xtuner.group_id": rollout_state.group_id,
        "xtuner.session_id": rollout_state.session_id,
        "xtuner.task_name": rollout_state.task_name or extra_fields.get("task_name"),
        "xtuner.producer_future_step": extra_fields.get("producer_future_step"),
    }
    if rollout_state.prompt_ids is not None:
        attributes["prompt.tokens"] = len(rollout_state.prompt_ids)
    return {key: value for key, value in attributes.items() if value is not None}


def rollout_state_final_attributes(rollout_state: RolloutState) -> dict[str, Any]:
    attributes = rollout_state_initial_attributes(rollout_state)
    attributes.update(
        {
            "finish_reason": rollout_state.finish_reason,
            "error.message": rollout_state.error_msg,
        }
    )
    if rollout_state.response_ids is not None:
        attributes["completion.tokens"] = len(rollout_state.response_ids)
    reward = rollout_state.reward
    if isinstance(reward, Mapping):
        attributes.update(
            {
                "reward.score": reward.get("score"),
                "reward.pass": reward.get("pass"),
            }
        )
    return {key: value for key, value in attributes.items() if value is not None}


@contextmanager
def _attach_rollout_call_chain(
    rollout_state: RolloutState,
    span_name: str,
    parent_carrier: Mapping[str, str] | None,
):
    extra_fields = rollout_state.extra_fields
    cleanup_call_chain_on_exit = TRACE_CALL_CHAIN_EXTRA_FIELD not in extra_fields
    call_chain = (*_rollout_call_chain(rollout_state, parent_carrier), span_name)
    extra_fields[TRACE_CALL_CHAIN_EXTRA_FIELD] = list(call_chain)
    try:
        yield call_chain, cleanup_call_chain_on_exit
    finally:
        if cleanup_call_chain_on_exit:
            rollout_state.extra_fields.pop(TRACE_CALL_CHAIN_EXTRA_FIELD, None)


def _rollout_call_chain(
    rollout_state: RolloutState,
    parent_carrier: Mapping[str, str] | None,
) -> tuple[str, ...]:
    value = rollout_state.extra_fields.get(TRACE_CALL_CHAIN_EXTRA_FIELD)
    if value is None and parent_carrier:
        parent_context = otel_utils.extract_otel_context(parent_carrier)
        value = trace_api._extract_span_name_path(parent_context, otel_utils=otel_utils)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split("->") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _resolve_rollout_state_target(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    target: RolloutState | None = None,
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
        elif isinstance(value, (list, tuple, set, frozenset)) and any(
            isinstance(item, RolloutState) for item in value
        ):
            raise TypeError(f"{owner} supports a single RolloutState, not a RolloutState collection")

    if len(rollout_states) != 1:
        raise ValueError(f"{owner} requires exactly one RolloutState argument")
    return rollout_states[0]


__all__ = [
    "TRACE_CARRIER_EXTRA_FIELD",
    "TRACE_ROLLOUT_ENABLED_ENV",
    "extract_rollout_trace_parent_carrier",
    "is_rollout_trace_enabled",
    "rollout_state_final_attributes",
    "rollout_state_initial_attributes",
    "trace_rollout_endpoint",
    "trace_rollout_remote",
]
