from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xtuner.v1.rl.telemetry.runtime import (
    TraceRuntime,
    configure_trace_runtime,
    current_trace_runtime,
)
from xtuner.v1.rl.telemetry.runtime import (
    close_trace as _close_trace_runtime,
)
from xtuner.v1.rl.telemetry.runtime import (
    ensure_trace_runtime_from_env as _ensure_trace_runtime_from_env,
)
from xtuner.v1.rl.telemetry.runtime import (
    reset_trace_for_test as _reset_trace_runtime_for_test,
)


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


def ensure_trace_runtime_from_env() -> bool:
    return _ensure_trace_runtime_from_env()


def close_trace() -> None:
    _close_trace_runtime()


def reset_trace_for_test() -> None:
    _reset_trace_runtime_for_test()


__all__ = [
    "TraceConfig",
    "TraceRuntime",
    "close_trace",
    "configure_trace",
    "current_trace_runtime",
    "ensure_trace_runtime_from_env",
    "reset_trace_for_test",
]
