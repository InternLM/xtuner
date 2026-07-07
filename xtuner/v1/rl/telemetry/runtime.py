from __future__ import annotations

import atexit
import contextlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

from xtuner.v1.rl.telemetry.collector import OTelCollector
from xtuner.v1.rl.telemetry.otel_utils import (
    _reset_otel_tracer_provider_for_reconfigure,
    configure_tracer_provider,
)
from xtuner.v1.rl.utils.misc import find_free_ports
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from xtuner.v1.rl.trace import TraceConfig


logger = get_logger()

TraceRuntimeMode = Literal["disabled", "driver", "inherited"]


TRACE_ENV_KEYS = (
    "XTUNER_OTEL_ENABLED",
    "XTUNER_OTEL_OUTPUT_DIR",
    "XTUNER_OTEL_RUN_ID",
    "XTUNER_OTEL_RUN_DIR",
    "XTUNER_OTEL_JSONL_PATH",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
)


@dataclass(frozen=True)
class TraceRuntime:
    enabled: bool
    mode: TraceRuntimeMode
    run_id: str
    run_dir: Path
    trace_jsonl_path: Path
    service_name: str


@dataclass
class _ActiveTraceRuntime:
    runtime: TraceRuntime
    endpoint: str
    env_vars: dict[str, str]
    collector_port: int | None = None
    collector: OTelCollector | None = None
    provider: Any | None = None

    def start(self) -> None:
        apply_trace_env(self.env_vars)
        if not self.runtime.enabled:
            logger.info("XTuner OTel tracing disabled.")
            return
        try:
            if self.runtime.mode == "driver":
                if self.collector_port is None:
                    raise RuntimeError("driver trace runtime requires a collector port")
                self.collector = OTelCollector.start(
                    port=self.collector_port,
                    output_path=self.runtime.trace_jsonl_path,
                )
            self.provider = configure_tracer_provider(
                service_name=self.runtime.service_name,
                run_id=self.runtime.run_id,
                endpoint=self.endpoint,
                protocol=self.env_vars["OTEL_EXPORTER_OTLP_PROTOCOL"],
            )
        except Exception:
            self.close()
            clear_trace_env()
            raise
        logger.info(
            f"XTuner OTel tracing enabled: run_id={self.runtime.run_id}, endpoint={self.endpoint}, "
            f"traces={self.runtime.trace_jsonl_path}"
        )

    def close(self) -> None:
        provider = self.provider
        self.provider = None
        if provider is not None:
            with contextlib.suppress(Exception):
                provider.shutdown()

        collector = self.collector
        self.collector = None
        if collector is not None:
            with contextlib.suppress(Exception):
                collector.close()


_RUNTIME: _ActiveTraceRuntime | None = None
_ATEXIT_REGISTERED = False


def configure_trace_runtime(config: TraceConfig) -> TraceRuntime:
    """Configure Layer1 OTel runtime for the current process."""

    global _RUNTIME

    close_trace()
    active_runtime = _build_active_trace_runtime(config)
    active_runtime.start()
    _RUNTIME = active_runtime
    if active_runtime.runtime.enabled:
        register_atexit_once(close_trace)
    return active_runtime.runtime


def _build_active_trace_runtime(config: TraceConfig) -> _ActiveTraceRuntime:
    if not config.enabled:
        return _ActiveTraceRuntime(
            runtime=TraceRuntime(
                enabled=False,
                mode="disabled",
                run_id="",
                run_dir=Path(),
                trace_jsonl_path=Path(),
                service_name=config.service_name,
            ),
            endpoint="",
            env_vars={},
        )

    output_dir = Path(config.output_dir or Path.cwd() / "otel_traces").expanduser()
    run_id = new_run_id()
    run_dir = output_dir / run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl_path = traces_dir / "traces.jsonl"

    try:
        port = find_free_ports(nums=1, host="127.0.0.1", start_port=4317, end_port=4318)[0]
    except RuntimeError:
        port = find_free_ports(nums=1, host="127.0.0.1")[0]
    endpoint = f"http://127.0.0.1:{port}"
    protocol = "grpc"

    env_vars = {
        "XTUNER_OTEL_ENABLED": "1",
        "XTUNER_OTEL_OUTPUT_DIR": os.fspath(output_dir),
        "XTUNER_OTEL_RUN_ID": run_id,
        "XTUNER_OTEL_RUN_DIR": os.fspath(run_dir),
        "XTUNER_OTEL_JSONL_PATH": os.fspath(trace_jsonl_path),
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_PROTOCOL": protocol,
        "OTEL_SERVICE_NAME": config.service_name,
    }
    return _ActiveTraceRuntime(
        runtime=TraceRuntime(
            enabled=True,
            mode="driver",
            run_id=run_id,
            run_dir=run_dir,
            trace_jsonl_path=trace_jsonl_path,
            service_name=config.service_name,
        ),
        endpoint=endpoint,
        env_vars=env_vars,
        collector_port=port,
    )


def get_trace_env_vars() -> dict[str, str]:
    """Return the env vars that should be injected into child Ray processes."""

    if _RUNTIME is None or not _RUNTIME.runtime.enabled:
        return get_trace_env_vars_from_env()
    return dict(_RUNTIME.env_vars)


def current_trace_runtime() -> TraceRuntime | None:
    """Return the active trace runtime owned by this process, if any."""

    if _RUNTIME is None:
        return None
    return _RUNTIME.runtime


def get_trace_env_vars_from_env() -> dict[str, str]:
    """Return inherited trace env vars before process-local runtime exists."""

    if os.environ.get("XTUNER_OTEL_ENABLED") != "1":
        return {}
    return {key: os.environ[key] for key in TRACE_ENV_KEYS if key in os.environ}


def ensure_trace_runtime_from_env() -> bool:
    """Lazily configure trace runtime in Ray child processes from inherited
    env."""

    return _ensure_trace_runtime_from_env_vars(get_trace_env_vars_from_env())


def _ensure_trace_runtime_from_env_vars(env_vars: Mapping[str, str]) -> bool:
    """Lazily configure trace runtime from already-resolved env vars."""

    global _RUNTIME

    if _RUNTIME is not None and _RUNTIME.runtime.enabled:
        return True
    if not env_vars:
        return False

    env_vars = {key: str(env_vars[key]) for key in TRACE_ENV_KEYS if key in env_vars}
    if "OTEL_EXPORTER_OTLP_ENDPOINT" not in env_vars:
        return False
    env_vars.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"])
    env_vars.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    env_vars.setdefault("OTEL_TRACES_EXPORTER", "otlp")

    run_dir = Path(env_vars.get("XTUNER_OTEL_RUN_DIR") or Path.cwd()).expanduser()
    trace_jsonl_path = Path(env_vars.get("XTUNER_OTEL_JSONL_PATH") or run_dir / "traces" / "traces.jsonl").expanduser()
    active_runtime = _ActiveTraceRuntime(
        runtime=TraceRuntime(
            enabled=True,
            mode="inherited",
            run_id=env_vars.get("XTUNER_OTEL_RUN_ID", ""),
            run_dir=run_dir,
            trace_jsonl_path=trace_jsonl_path,
            service_name=env_vars.get("OTEL_SERVICE_NAME", "xtuner-rollout"),
        ),
        endpoint=env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"],
        env_vars=env_vars,
    )
    active_runtime.start()
    _RUNTIME = active_runtime
    register_atexit_once(close_trace)
    return True


def is_trace_enabled() -> bool:
    """Return whether XTuner trace runtime is enabled in this process."""

    return _RUNTIME is not None and _RUNTIME.runtime.enabled


def close_trace() -> None:
    """Flush provider state and stop the local collector if this process owns
    it."""

    global _RUNTIME

    runtime = _RUNTIME
    _RUNTIME = None
    if runtime is not None:
        runtime.close()
    clear_trace_env()


def reset_trace_for_test() -> None:
    """Reset process-global OTel state for isolated unit tests."""

    close_trace()
    _reset_otel_tracer_provider_for_reconfigure()


def apply_trace_env(env_vars: Mapping[str, str]) -> None:
    clear_trace_env()
    os.environ.update(env_vars)


def clear_trace_env() -> None:
    for key in TRACE_ENV_KEYS:
        os.environ.pop(key, None)


def register_atexit_once(close_fn) -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(close_fn)
        _ATEXIT_REGISTERED = True


def new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
