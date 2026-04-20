from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GatewayConfig:
    _CAPTURE_PATH_FOLDER = "gateway_captures"
    """Configuration for the XTuner gateway HTTP server.

    Examples::

        # Auto-start with RolloutController:
        cfg = GatewayConfig(port=8080)

        # Opt-out of auto-start (start manually later):
        cfg = GatewayConfig(port=8080, auto_start=False)

        # With request capture (writes one JSONL file per API key):
        cfg = GatewayConfig(port=8080, capture_folder="/tmp/gateway_captures")
    """

    port: int
    """TCP port to bind the server on."""

    host: str = "0.0.0.0"
    """Interface to bind the server on."""

    auto_start: bool = True
    """Whether to start the gateway automatically when the RolloutController
    initialises.

    Set to False if you want to start the gateway manually via
    :func:`~xtuner.v1.rl.gateway.serve_gateway` or
    :meth:`~xtuner.v1.rl.rollout.controller.RolloutController.start_gateway`.
    """

    capture_folder: str | None = None
    """Optional folder for writing per-request trace records.

    The gateway writes one JSONL file per API key inside this folder. If
    omitted, this resolves to ``./worker_dirs/gateway_captures``; when started
    by :class:`~xtuner.v1.rl.rollout.controller.RolloutController`, an omitted
    value resolves relative to ``RolloutConfig.worker_log_dir`` instead.
    """
    title: str = "XTuner Gateway"
    """FastAPI application title shown in /docs."""

    version: str = "0.1.0"
    """FastAPI application version string."""

    log_level: str = "warning"
    """Uvicorn log level (debug/info/warning/error/critical)."""

    def __post_init__(self) -> None:
        if self.capture_folder is None:
            self.capture_folder = str(Path.cwd() / "worker_dirs" / self._CAPTURE_PATH_FOLDER)
            print(f"GatewayConfig.capture_folder is not specified, use default capture_folder: {self.capture_folder}")
