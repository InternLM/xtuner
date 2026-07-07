from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xtuner.v1.rl.telemetry import otel_utils
from xtuner.v1.rl.telemetry.runtime import (
    TraceRuntime,
    configure_trace_runtime,
    current_trace_runtime,
    is_trace_enabled,
)
from xtuner.v1.rl.telemetry.runtime import (
    close_trace as _close_trace_runtime,
)
from xtuner.v1.rl.telemetry.runtime import (
    ensure_trace_runtime_from_env as _ensure_trace_runtime_from_env,
)


F = TypeVar("F", bound=Callable[..., Any])


_OTEL_INT64_MIN = -(2**63)
_OTEL_INT64_MAX = 2**63 - 1


class TraceConfig(BaseModel):
    """Public rollout tracing configuration.

    The interface is intentionally XTuner-level. OTel endpoint, exporter, and collector choices are runtime
    implementation details.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    enabled: bool = False
    output_dir: Path | str | None = Field(default=None)
    service_name: str = "xtuner-rollout"

    @model_validator(mode="before")
    @classmethod
    def _accept_deprecated_otlp_output_dir(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "otlp_output_dir" not in data:
            return data
        alias_value = data.pop("otlp_output_dir")
        if "output_dir" in data and data["output_dir"] != alias_value:
            raise ValueError("output_dir and otlp_output_dir cannot differ")
        data["output_dir"] = alias_value
        return data

    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()


def configure_trace(config: TraceConfig | None = None) -> TraceRuntime:
    return configure_trace_runtime(config or TraceConfig())


def close_trace() -> None:
    _close_trace_runtime()


@contextmanager
def trace_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    parent_carrier: Mapping[str, str] | None = None,
):
    """Create a current trace span using XTuner trace semantics."""

    span_name = _validate_trace_name(name, kind="span name")
    normalized_attributes = _normalize_trace_attributes(attributes)
    _ensure_trace_runtime_from_env()
    if not is_trace_enabled():
        yield
        return

    with _attach_parent_carrier(parent_carrier):
        with otel_utils.start_span(span_name, attributes=normalized_attributes):
            try:
                yield
            except Exception as exc:
                otel_utils.record_failure(exc)
                raise


def trace_function(
    name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorate a function so each call is wrapped in a trace span."""

    def decorator(func: F) -> F:
        span_name = _validate_trace_name(
            name or f"{func.__module__}.{func.__qualname__}",
            kind="span name",
        )

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
    event_name = _validate_trace_name(name, kind="event name")
    normalized_attributes = _normalize_trace_attributes(attributes)
    if not is_trace_enabled():
        return
    otel_utils.add_event(event_name, attributes=normalized_attributes)


def set_trace_attribute(key: str, value: Any) -> None:
    set_trace_attributes({key: value})


def set_trace_attributes(attributes: Mapping[str, Any] | None) -> None:
    normalized_attributes = _normalize_trace_attributes(attributes)
    if not is_trace_enabled():
        return
    otel_utils.set_attributes(normalized_attributes)


def set_trace_error(message: str | None = None) -> None:
    if not is_trace_enabled():
        return
    otel_utils.set_error_status(message)


def inject_trace_context(carrier: dict[str, str] | None = None) -> dict[str, str]:
    target = carrier if carrier is not None else {}
    _ensure_trace_runtime_from_env()
    if not is_trace_enabled():
        return target
    otel_utils.inject_otel_context(carrier=target)
    return target


@contextmanager
def _attach_parent_carrier(parent_carrier: Mapping[str, str] | None):
    if parent_carrier is None:
        yield
        return

    parent_context = otel_utils.extract_otel_context(parent_carrier)
    token = otel_utils.attach_otel_context(parent_context)
    try:
        yield
    finally:
        otel_utils.detach_otel_context(token)


def _validate_trace_name(name: str, *, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} must be a string")
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError(f"{kind} cannot be empty")
    return normalized_name


def _normalize_trace_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if attributes is None:
        return {}
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
    return normalized


def _normalize_trace_attribute_value(value: Any) -> Any:
    def normalize_scalar(item: Any) -> str | bool | int | float:
        if isinstance(item, bool):
            return item
        if isinstance(item, int):
            return item if _OTEL_INT64_MIN <= item <= _OTEL_INT64_MAX else str(item)
        if isinstance(item, (str, float)):
            return item
        return f"<{type(item).__name__}>"

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if _OTEL_INT64_MIN <= value <= _OTEL_INT64_MAX else str(value)
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
    "TraceConfig",
    "TraceRuntime",
    "close_trace",
    "configure_trace",
    "current_trace_runtime",
    "inject_trace_context",
    "set_trace_attribute",
    "set_trace_attributes",
    "set_trace_error",
    "trace_event",
    "trace_function",
    "trace_span",
]
