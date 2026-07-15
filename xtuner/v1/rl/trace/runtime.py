from __future__ import annotations

import atexit
import contextlib
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xtuner.v1.rl.utils.misc import find_free_ports
from xtuner.v1.utils import get_logger


logger = get_logger()

TraceRuntimeMode = Literal["disabled", "driver", "inherited"]


TRACE_ENV_KEYS = (
    "XTUNER_OTEL_ENABLED",
    "XTUNER_OTEL_OUTPUT_DIR",
    "XTUNER_OTEL_RUN_ID",
    "XTUNER_OTEL_RUN_DIR",
    "XTUNER_OTEL_JSONL_PATH",
    "XTUNER_TRACE_ENABLE_ROLLOUT",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
)


DEFAULT_JAEGER_OTLP_GRPC_ENDPOINT = "127.0.0.1:14317"
_TRACE_VIEWER_SERVER_MODULE = "recipe.trace_viewer.server"
_READY_TIMEOUT_S = 3.0
_READY_POLL_INTERVAL_S = 0.05
_LOG_TAIL_CHARS = 4000
_TRACE_VIEWER_STARTUP_CHECK_TIMEOUT_S = 0.5

_OTELCOL_CONFIG_YAML_TEMPLATE = """
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:{port}

exporters:
  file:
    path: {output_path}
    rotation:
      max_megabytes: 64
      max_days: 0
      max_backups: 0

service:
  telemetry:
    metrics:
      level: none
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [{exporters}]
""".lstrip()

_OTELCOL_OTLP_GRPC_EXPORTER_YAML_TEMPLATE = """
  otlp/jaeger:
    endpoint: {endpoint}
    tls:
      insecure: true
""".rstrip()


def _configure_tracer_provider(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    protocol: str,
) -> Any:
    try:
        from xtuner.v1.rl.trace.otel_utils import configure_tracer_provider
    except ModuleNotFoundError as exc:
        if exc.name == "opentelemetry" or (exc.name or "").startswith("opentelemetry."):
            raise RuntimeError(
                "XTuner OTel tracing requires OpenTelemetry packages. "
                "Install `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-grpc` "
                "before enabling trace."
            ) from exc
        raise
    return configure_tracer_provider(
        service_name=service_name,
        run_id=run_id,
        endpoint=endpoint,
        protocol=protocol,
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
    xtuner_viewer_enabled: bool = False
    xtuner_viewer_host: str = "127.0.0.1"
    xtuner_viewer_port: int = Field(default=18080, ge=0, le=65535)
    xtuner_viewer_jaeger_query_url: str | None = None
    enable_rollout_trace: bool = False

    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()


@dataclass(frozen=True)
class TraceRuntime:
    enabled: bool
    mode: TraceRuntimeMode
    run_id: str
    run_dir: Path
    trace_jsonl_path: Path
    service_name: str
    trace_viewer_url: str | None = None
    trace_viewer_port: int | None = None


@dataclass
class _OTelCollector:
    port: int
    output_path: Path
    _binary_path: str = field(repr=False)
    _config_path: Path = field(repr=False)
    _stdout_path: Path = field(repr=False)
    _stderr_path: Path = field(repr=False)
    _process: subprocess.Popen | None = field(repr=False)

    @classmethod
    def start(
        cls,
        *,
        port: int,
        output_path: Path,
    ) -> _OTelCollector:
        otelcol = shutil.which("otelcol-contrib") or shutil.which("otelcol")
        if otelcol is None:
            raise RuntimeError(
                "XTuner OTel tracing requires `otelcol-contrib` or `otelcol` on PATH. "
                "Install an official OpenTelemetry Collector binary before enabling trace."
            )

        jaeger_exporter = "\n" + _OTELCOL_OTLP_GRPC_EXPORTER_YAML_TEMPLATE.format(
            endpoint=DEFAULT_JAEGER_OTLP_GRPC_ENDPOINT
        )

        config_path = output_path.parent / "otelcol.yaml"
        config_yaml = _OTELCOL_CONFIG_YAML_TEMPLATE.format(
            port=port,
            output_path=output_path,
            exporters="file, otlp/jaeger",
        ).replace("\nservice:", f"{jaeger_exporter}\n\nservice:")
        config_path.write_text(config_yaml, encoding="utf-8")
        stdout_path = output_path.parent / "otelcol.stdout.log"
        stderr_path = output_path.parent / "otelcol.stderr.log"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                [otelcol, "--config", os.fspath(config_path)],
                stdout=stdout_file,
                stderr=stderr_file,
            )
        collector = cls(
            port=port,
            output_path=output_path,
            _binary_path=otelcol,
            _config_path=config_path,
            _stdout_path=stdout_path,
            _stderr_path=stderr_path,
            _process=process,
        )
        try:
            collector._wait_until_ready()
        except Exception:
            collector.close()
            raise
        return collector

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                stderr_tail = ""
                with contextlib.suppress(OSError):
                    stderr_tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:]
                raise RuntimeError(
                    "OpenTelemetry collector failed to start: collector process is not available. "
                    f"binary={self._binary_path}, config={self._config_path}, port={self.port}, "
                    f"stderr_tail={stderr_tail!r}"
                )
            exit_code = process.poll()
            if exit_code is not None:
                stderr_tail = ""
                with contextlib.suppress(OSError):
                    stderr_tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:]
                raise RuntimeError(
                    f"OpenTelemetry collector failed to start: collector exited with code {exit_code}. "
                    f"binary={self._binary_path}, config={self._config_path}, port={self.port}, "
                    f"stderr_tail={stderr_tail!r}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(_READY_POLL_INTERVAL_S)

        detail = f"collector did not become ready within {_READY_TIMEOUT_S:.1f}s"
        if last_error is not None:
            detail += f"; last connection error: {last_error}"
        stderr_tail = ""
        with contextlib.suppress(OSError):
            stderr_tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:]
        raise RuntimeError(
            f"OpenTelemetry collector failed to start: {detail}. "
            f"binary={self._binary_path}, config={self._config_path}, port={self.port}, "
            f"stderr_tail={stderr_tail!r}"
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)


def _build_xtuner_viewer_command(
    *,
    trace_jsonl_path: Path,
    jaeger_query_url: str | None,
    service_name: str,
    run_id: str,
    host: str,
    port: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        _TRACE_VIEWER_SERVER_MODULE,
        "--trace-jsonl",
        os.fspath(trace_jsonl_path),
        "--service",
        service_name,
        "--run-id",
        run_id,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if jaeger_query_url is not None:
        command.extend(["--jaeger-query-url", jaeger_query_url])
    return command


@dataclass
class _TraceRuntimeHandle:
    runtime: TraceRuntime
    endpoint: str
    env_vars: dict[str, str]
    collector_port: int | None = None
    collector: _OTelCollector | None = None
    provider: Any | None = None
    xtuner_viewer_host: str | None = None
    xtuner_viewer_port: int = 0
    xtuner_viewer_jaeger_query_url: str | None = None
    xtuner_viewer_process: subprocess.Popen | None = None
    xtuner_viewer_command: list[str] | None = None
    xtuner_viewer_url: str | None = None

    def start(self) -> None:
        apply_trace_env(self.env_vars)
        if not self.runtime.enabled:
            logger.info("XTuner OTel tracing disabled.")
            return
        try:
            if self.runtime.mode == "driver":
                if self.collector_port is None:
                    raise RuntimeError("driver trace runtime requires a collector port")
                self.collector = _OTelCollector.start(
                    port=self.collector_port,
                    output_path=self.runtime.trace_jsonl_path,
                )
            self.provider = _configure_tracer_provider(
                service_name=self.runtime.service_name,
                run_id=self.runtime.run_id,
                endpoint=self.endpoint,
                protocol=self.env_vars["OTEL_EXPORTER_OTLP_PROTOCOL"],
            )
            if self.runtime.mode == "driver" and self.xtuner_viewer_host is not None:
                if self.xtuner_viewer_port == 0:
                    self.xtuner_viewer_port = find_free_ports(nums=1, host=self.xtuner_viewer_host)[0]
                self.xtuner_viewer_command = _build_xtuner_viewer_command(
                    trace_jsonl_path=self.runtime.trace_jsonl_path,
                    jaeger_query_url=self.xtuner_viewer_jaeger_query_url,
                    service_name=self.runtime.service_name,
                    run_id=self.runtime.run_id,
                    host=self.xtuner_viewer_host,
                    port=self.xtuner_viewer_port,
                )
                self.xtuner_viewer_process = subprocess.Popen(self.xtuner_viewer_command)
                try:
                    exit_code = self.xtuner_viewer_process.wait(timeout=_TRACE_VIEWER_STARTUP_CHECK_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    raise RuntimeError(
                        "XTuner trace viewer failed to start: "
                        f"exit_code={exit_code}. Manual command: {shlex.join(self.xtuner_viewer_command)}"
                    )
                self.xtuner_viewer_url = f"http://{self.xtuner_viewer_host}:{self.xtuner_viewer_port}"
                self.runtime = replace(
                    self.runtime,
                    trace_viewer_url=self.xtuner_viewer_url,
                    trace_viewer_port=self.xtuner_viewer_port,
                )
        except Exception:
            self.close(stop_viewer=True)
            clear_trace_env()
            raise
        logger.info(
            f"XTuner OTel tracing enabled: run_id={self.runtime.run_id}, endpoint={self.endpoint}, "
            f"traces={self.runtime.trace_jsonl_path}"
        )
        if self.xtuner_viewer_process is not None:
            logger.info(
                f"XTuner trace viewer enabled: url={self.xtuner_viewer_url}. "
                f"Manual command: {shlex.join(self.xtuner_viewer_command or [])}"
            )

    def close(self, *, stop_viewer: bool = True) -> None:
        xtuner_viewer_process = self.xtuner_viewer_process
        self.xtuner_viewer_process = None
        if xtuner_viewer_process is not None and stop_viewer:
            logger.info("XTuner trace viewer stopped with training process.")
            if xtuner_viewer_process.poll() is None:
                xtuner_viewer_process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    xtuner_viewer_process.wait(timeout=5)
                if xtuner_viewer_process.poll() is None:
                    xtuner_viewer_process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        xtuner_viewer_process.wait(timeout=5)

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


_RUNTIME: _TraceRuntimeHandle | None = None
_ATEXIT_REGISTERED = False


def configure_trace(config: TraceConfig | None = None) -> TraceRuntime:
    return configure_trace_runtime(config or TraceConfig())


def configure_trace_runtime(config: TraceConfig) -> TraceRuntime:
    """Configure Layer1 OTel runtime for the current process."""

    global _RUNTIME

    close_trace()
    runtime_handle = _build_trace_runtime_handle(config)
    runtime_handle.start()
    _RUNTIME = runtime_handle
    if runtime_handle.runtime.enabled:
        register_atexit_once(close_trace)
    return runtime_handle.runtime


def _build_trace_runtime_handle(config: TraceConfig) -> _TraceRuntimeHandle:
    if not config.enabled:
        return _TraceRuntimeHandle(
            runtime=TraceRuntime(
                enabled=False,
                mode="disabled",
                run_id="",
                run_dir=Path(),
                trace_jsonl_path=Path(),
                service_name=config.service_name,
                trace_viewer_url=None,
                trace_viewer_port=None,
            ),
            endpoint="",
            env_vars={},
        )

    output_dir = Path(config.output_dir or Path.cwd() / "otel_traces").expanduser()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_id = f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl_path = traces_dir / "traces.jsonl"
    trace_jsonl_path.touch(exist_ok=True)

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
        "XTUNER_TRACE_ENABLE_ROLLOUT": "1" if config.enable_rollout_trace else "0",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_PROTOCOL": protocol,
        "OTEL_SERVICE_NAME": config.service_name,
    }
    return _TraceRuntimeHandle(
        runtime=TraceRuntime(
            enabled=True,
            mode="driver",
            run_id=run_id,
            run_dir=run_dir,
            trace_jsonl_path=trace_jsonl_path,
            service_name=config.service_name,
            trace_viewer_url=None,
            trace_viewer_port=None,
        ),
        endpoint=endpoint,
        env_vars=env_vars,
        collector_port=port,
        xtuner_viewer_host=config.xtuner_viewer_host if config.xtuner_viewer_enabled else None,
        xtuner_viewer_port=config.xtuner_viewer_port,
        xtuner_viewer_jaeger_query_url=config.xtuner_viewer_jaeger_query_url,
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

    global _RUNTIME

    env_vars = get_trace_env_vars_from_env()
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
    runtime_handle = _TraceRuntimeHandle(
        runtime=TraceRuntime(
            enabled=True,
            mode="inherited",
            run_id=env_vars.get("XTUNER_OTEL_RUN_ID", ""),
            run_dir=run_dir,
            trace_jsonl_path=trace_jsonl_path,
            service_name=env_vars.get("OTEL_SERVICE_NAME", "xtuner-rollout"),
            trace_viewer_url=None,
            trace_viewer_port=None,
        ),
        endpoint=env_vars["OTEL_EXPORTER_OTLP_ENDPOINT"],
        env_vars=env_vars,
    )
    runtime_handle.start()
    _RUNTIME = runtime_handle
    register_atexit_once(close_trace)
    return True


def is_trace_enabled() -> bool:
    """Return whether XTuner trace runtime is enabled in this process."""

    if _RUNTIME is None:
        ensure_trace_runtime_from_env()
    return _RUNTIME is not None and _RUNTIME.runtime.enabled


def close_trace() -> None:
    """Flush provider state and stop local trace processes owned by this
    process."""

    global _RUNTIME

    runtime = _RUNTIME
    _RUNTIME = None
    if runtime is not None:
        runtime.close()
    clear_trace_env()


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
