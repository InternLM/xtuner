from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_JAEGER_OTLP_GRPC_ENDPOINT = "127.0.0.1:14317"
_READY_TIMEOUT_S = 3.0
_READY_POLL_INTERVAL_S = 0.05
_LOG_TAIL_CHARS = 4000

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


@dataclass
class OTelCollector:
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
    ) -> OTelCollector:
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
