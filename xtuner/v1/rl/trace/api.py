from __future__ import annotations

import contextvars
import inspect
import json
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from . import otel_utils
from .runtime import (
    ensure_trace_runtime_from_env as _ensure_trace_runtime_from_env,
)
from .runtime import is_trace_enabled


F = TypeVar("F", bound=Callable[..., Any])


_SPAN_NAME_PATH_BAGGAGE_KEY = "xtuner.span_name_path"
_SPAN_NAME_PATH_ATTRIBUTE = "xtuner.span_name_path"
# OTel context propagation carries trace/span IDs, not the current span names.
# XTuner keeps this local path so child spans and remote carriers can expose a
# readable execution chain for the viewer.
_CURRENT_SPAN_NAME_PATH: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "xtuner_trace_span_name_path",
    default=(),
)


@contextmanager
def trace_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    parent_carrier: Mapping[str, str] | None = None,
):
    """Create a current trace span.

    XTuner wraps OTel spans with runtime auto-initialization, trace-enabled
    gating, attribute normalization, failure recording, and a
    ``xtuner.span_name_path`` attribute for viewer-friendly call chains.
    ``parent_carrier`` accepts the W3C carrier produced by
    ``inject_trace_context`` across a process or request boundary.
    """

    span_name, normalized_attributes = _validate_trace_inputs(
        name=name,
        attributes=attributes,
    )
    assert span_name is not None
    _ensure_trace_runtime_from_env()
    if not is_trace_enabled():
        yield
        return

    with _attach_parent_carrier(parent_carrier):
        span_name_path = (*_CURRENT_SPAN_NAME_PATH.get(), span_name)
        normalized_attributes = dict(normalized_attributes)
        normalized_attributes.setdefault(_SPAN_NAME_PATH_ATTRIBUTE, span_name_path)
        path_token = _CURRENT_SPAN_NAME_PATH.set(span_name_path)
        try:
            with otel_utils.start_span(span_name, attributes=normalized_attributes):
                try:
                    yield
                except Exception as exc:
                    otel_utils.record_failure(exc)
                    raise
        finally:
            _CURRENT_SPAN_NAME_PATH.reset(path_token)


def trace_function(
    name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorate a sync or async function with ``trace_span``.

    This provides the same XTuner additions as ``trace_span`` while keeping
    call sites free of direct OTel SDK usage.
    """

    def decorator(func: F) -> F:
        span_name, _ = _validate_trace_inputs(
            name=name or f"{func.__module__}.{func.__qualname__}",
            attributes=None,
        )
        assert span_name is not None

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with trace_span(span_name, attributes=attributes):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(span_name, attributes=attributes):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    """Add an event to the current span through XTuner's normalized API."""

    event_name, normalized_attributes = _validate_trace_inputs(
        name=name,
        attributes=attributes,
    )
    assert event_name is not None
    if not is_trace_enabled():
        return
    otel_utils.add_event(event_name, attributes=normalized_attributes)


def set_trace_attributes(attributes: Mapping[str, Any] | None) -> None:
    """Set current-span attributes after applying XTuner value
    normalization."""

    _, normalized_attributes = _validate_trace_inputs(attributes=attributes)
    if not is_trace_enabled():
        return
    otel_utils.set_attributes(normalized_attributes)


def inject_trace_context(carrier: dict[str, str] | None = None) -> dict[str, str]:
    """Inject OTel context plus XTuner's span-name path into a W3C carrier.

    OTel's trace context links parent/child spans. XTuner adds W3C Baggage for
    ``xtuner.span_name_path`` so downstream spans can keep a readable chain.
    """

    target = carrier if carrier is not None else {}
    _ensure_trace_runtime_from_env()
    if not is_trace_enabled():
        return target
    span_name_path = _CURRENT_SPAN_NAME_PATH.get()
    context = None
    if span_name_path:
        context = otel_utils.context_with_baggage(
            _SPAN_NAME_PATH_BAGGAGE_KEY,
            json.dumps(span_name_path, separators=(",", ":")),
        )
    otel_utils.inject_otel_context(context=context, carrier=target)
    return target


@contextmanager
def _attach_parent_carrier(parent_carrier: Mapping[str, str] | None):
    if parent_carrier is None:
        yield
        return

    parent_context = otel_utils.extract_otel_context(parent_carrier)
    token = otel_utils.attach_otel_context(parent_context)
    span_name_path = _extract_span_name_path(parent_context, otel_utils=otel_utils)
    path_token = _CURRENT_SPAN_NAME_PATH.set(span_name_path) if span_name_path else None
    try:
        yield
    finally:
        if path_token is not None:
            _CURRENT_SPAN_NAME_PATH.reset(path_token)
        otel_utils.detach_otel_context(token)


def _extract_span_name_path(parent_context: Any, *, otel_utils: Any) -> tuple[str, ...]:
    payload = otel_utils.get_baggage(_SPAN_NAME_PATH_BAGGAGE_KEY, context=parent_context)
    if not payload:
        return ()
    return _parse_span_name_path_payload(payload)


def _parse_span_name_path_payload(payload: object) -> tuple[str, ...]:
    try:
        value = json.loads(str(payload))
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())


def _validate_trace_inputs(
    *,
    name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    normalized_name: str | None = None
    if name is not None:
        if not isinstance(name, str):
            raise TypeError("trace name must be a string")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("trace name cannot be empty")

    if attributes is None:
        return normalized_name, {}
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")

    normalized = {}
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise TypeError("attribute key must be a string")
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("attribute key cannot be empty")
        normalized[normalized_key] = _normalize_trace_attribute_value(value)
    return normalized_name, normalized


def _normalize_trace_attribute_value(value: Any) -> Any:
    def normalize_scalar(item: Any) -> str | bool | int | float:
        if isinstance(item, bool):
            return item
        if isinstance(item, int):
            return item if -(2**63) <= item <= 2**63 - 1 else str(item)
        if isinstance(item, (str, float)):
            return item
        return f"<{type(item).__name__}>"

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -(2**63) <= value <= 2**63 - 1 else str(value)
    if isinstance(value, (str, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = tuple(normalize_scalar(item) for item in value)
        if not items:
            return items
        if all(isinstance(item, str) for item in items):
            return items
        if all(isinstance(item, bool) for item in items):
            return items
        if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
            return items
        if all(isinstance(item, float) for item in items):
            return items
        return tuple(str(item) for item in items)
    return f"<{type(value).__name__}>"


__all__ = [
    "inject_trace_context",
    "set_trace_attributes",
    "trace_event",
    "trace_function",
    "trace_span",
]
