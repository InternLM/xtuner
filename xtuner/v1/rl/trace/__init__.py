from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "TraceConfig": (".runtime", "TraceConfig"),
    "TraceRuntime": (".runtime", "TraceRuntime"),
    "close_trace": (".runtime", "close_trace"),
    "configure_trace": (".runtime", "configure_trace"),
    "configure_trace_runtime": (".runtime", "configure_trace_runtime"),
    "current_trace_runtime": (".runtime", "current_trace_runtime"),
    "get_trace_env_vars": (".runtime", "get_trace_env_vars"),
    "inject_trace_context": (".api", "inject_trace_context"),
    "set_trace_attributes": (".api", "set_trace_attributes"),
    "trace_event": (".api", "trace_event"),
    "trace_function": (".api", "trace_function"),
    "trace_span": (".api", "trace_span"),
}


__all__ = [
    "TraceConfig",
    "TraceRuntime",
    "close_trace",
    "configure_trace",
    "configure_trace_runtime",
    "current_trace_runtime",
    "get_trace_env_vars",
    "inject_trace_context",
    "set_trace_attributes",
    "trace_event",
    "trace_function",
    "trace_span",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
