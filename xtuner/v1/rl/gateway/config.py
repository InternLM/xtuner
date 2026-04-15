from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    """Configuration for the XTuner gateway HTTP server.

    Examples::

        # Auto-start with RolloutController:
        cfg = GatewayConfig(port=8080)

        # Opt-out of auto-start (start manually later):
        cfg = GatewayConfig(port=8080, auto_start=False)

        # With request capture (writes JSON files per request):
        cfg = GatewayConfig(port=8080, capture_path="/tmp/gateway_captures")
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

    capture_path: str | None = None
    """Optional directory path for writing per-request trace files."""

    title: str = "XTuner Gateway"
    """FastAPI application title shown in /docs."""

    version: str = "0.1.0"
    """FastAPI application version string."""

    log_level: str = "warning"
    """Uvicorn log level (debug/info/warning/error/critical)."""
