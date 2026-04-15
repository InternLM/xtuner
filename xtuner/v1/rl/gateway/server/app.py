from __future__ import annotations

import threading
from typing import Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.reasoning_parser import ThinkTagReasoningParser
from xtuner.v1.rl.rollout.tool_call_parser import CompositeToolCallParser, JsonToolCallParser, RegexToolCallParser

from ..adapters import AnthropicChatAdapter, OpenAIChatAdapter
from ..adapters.responses import OpenAIResponsesAdapter
from ..backend.local_backend import LocalRolloutBackend
from ..backend.protocol import GatewayBackend
from ..config import GatewayConfig
from ..core.exceptions import ContextLengthExceededError, GatewayError
from .routes import build_anthropic_router, build_openai_router, build_responses_router, build_runtime_router


TokenizerArg = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str]


# ---------------------------------------------------------------------------
# Internal base builder
# ---------------------------------------------------------------------------


def _create_base_gateway_app(
    backend: GatewayBackend,
    *,
    title: str = "XTuner Gateway",
    version: str = "0.1.0",
) -> FastAPI:
    """Create the base FastAPI app with runtime routes and global error
    handlers.

    This is an internal builder used by higher-level factory functions. The returned app exposes /livez, /readyz, and
    /capabilities but no protocol-specific endpoints.
    """
    app = FastAPI(title=title, version=version)
    app.state.gateway_backend = backend
    app.include_router(build_runtime_router())

    @app.exception_handler(ContextLengthExceededError)
    async def context_length_error_handler(request: Request, exc: ContextLengthExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "context_length_exceeded", "code": "context_too_long"}},
        )

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__, "code": "gateway_error"}},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": "internal_server_error"}},
        )

    return app


# ---------------------------------------------------------------------------
# Generic public factory (works with any GatewayBackend)
# ---------------------------------------------------------------------------


def build_gateway_app(
    backend: GatewayBackend,
    *,
    tokenizer: TokenizerArg,
    model_name: str,
    context_length: int,
    config: GatewayConfig | None = None,
) -> FastAPI:
    """Build a gateway FastAPI app wired to *any* :class:`GatewayBackend`.

    This is the lowest-level public factory.  Use this when you have a custom
    backend (e.g. a future ``RemoteRolloutBackend``) and want to wire it into
    the full gateway stack (OpenAI / Anthropic / Responses endpoints).

    Args:
        backend: An object that satisfies the :class:`~xtuner.v1.rl.gateway.backend.protocol.GatewayBackend` protocol.
        tokenizer: Tokenizer used for prompt encoding and token-count helpers.
            Accepts a :class:`~transformers.PreTrainedTokenizer`,
            :class:`~transformers.PreTrainedTokenizerFast`, or a **string**
            path/identifier which will be loaded via
            :func:`~transformers.AutoTokenizer.from_pretrained`.
        model_name: Default model name reported by the ``/capabilities`` endpoint.
        context_length: Maximum context length enforced by the gateway.
        config: Gateway configuration (title, version, capture_path, …).
            Defaults to a bare :class:`~xtuner.v1.rl.gateway.config.GatewayConfig`
            with ``port=8080`` when not provided.

    Returns:
        A fully-configured :class:`fastapi.FastAPI` instance ready to serve.
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    cfg = config or GatewayConfig(port=8080)
    app = _create_base_gateway_app(backend, title=cfg.title, version=cfg.version)
    adapter_kwargs = {
        "generate_handler": backend.generate,
        "tokenizer": tokenizer,
        "default_model_name": model_name,
        "context_length": context_length,
        "capture_path": cfg.capture_path,
    }
    app.state.gateway_openai_adapter = OpenAIChatAdapter(**adapter_kwargs)
    app.state.gateway_anthropic_adapter = AnthropicChatAdapter(**adapter_kwargs)
    app.state.gateway_responses_adapter = OpenAIResponsesAdapter(**adapter_kwargs)
    app.include_router(build_openai_router())
    app.include_router(build_anthropic_router())
    app.include_router(build_responses_router())
    return app


# ---------------------------------------------------------------------------
# LocalRolloutBackend convenience factory
# ---------------------------------------------------------------------------


def build_local_gateway_app(
    controller: RolloutController,
    config: GatewayConfig | None = None,
) -> FastAPI:
    """Build a gateway app backed by a local :class:`RolloutController`.

    Loads the tokenizer from ``controller.config.tokenizer_path``, configures
    output parsers on the controller, and wires a
    :class:`~xtuner.v1.rl.gateway.backend.local_backend.LocalRolloutBackend`
    into the full OpenAI / Anthropic / Responses endpoint stack.

    Args:
        controller: The local :class:`~xtuner.v1.rl.rollout.controller.RolloutController` instance.
        config: Gateway configuration.  ``port`` is not used during app
            construction (only needed for :func:`serve_gateway`).

    Returns:
        A fully-configured :class:`fastapi.FastAPI` instance.
    """
    cfg = config or GatewayConfig(port=8080)
    tokenizer = AutoTokenizer.from_pretrained(controller.config.tokenizer_path, trust_remote_code=True)

    strip_tokens: list[str] = []
    if tokenizer.eos_token:
        strip_tokens.append(tokenizer.eos_token)
    for tok in getattr(tokenizer, "additional_special_tokens", []):
        if any(marker in tok.lower() for marker in ("im_end", "eot", "end_of_turn", "turn_end")):
            strip_tokens.append(tok)

    controller.configure_output_parsers(
        tool_call_parser=CompositeToolCallParser(parsers=[JsonToolCallParser(), RegexToolCallParser()]),
        reasoning_parser=ThinkTagReasoningParser(strip_tokens=strip_tokens),
    )

    model_name = controller.config.model_name
    if model_name is None:
        raise ValueError("controller.config.model_name must be set when building a local gateway app")
    context_length = controller.config.context_length
    if context_length is None:
        raise ValueError("controller.config.context_length must be set when building a local gateway app")

    backend = LocalRolloutBackend(controller, tokenizer=tokenizer)
    return build_gateway_app(
        backend,
        tokenizer=tokenizer,
        model_name=model_name,
        context_length=context_length,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Standalone serve helpers
# ---------------------------------------------------------------------------


def serve_gateway(app: FastAPI, config: GatewayConfig) -> None:
    """Start the gateway server in the **current thread** (blocking).

    Use this for a fully standalone gateway process::

        from xtuner.v1.rl.gateway import (
            GatewayConfig, build_local_gateway_app, serve_gateway
        )

        config = GatewayConfig(port=8080, auto_start=False)
        app = build_local_gateway_app(controller, config)
        serve_gateway(app, config)  # blocks until interrupted

    For a custom backend::

        from xtuner.v1.rl.gateway import (
            GatewayConfig, build_gateway_app, serve_gateway
        )

        config = GatewayConfig(port=8080)
        app = build_gateway_app(
            my_backend,
            tokenizer=tokenizer,
            model_name="my-model",
            context_length=32768,
            config=config,
        )
        serve_gateway(app, config)

    Args:
        app: A FastAPI application previously built by :func:`build_gateway_app`
            or :func:`build_local_gateway_app`.
        config: Gateway configuration supplying ``host``, ``port``, and
            ``log_level``.
    """
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def serve_gateway_in_thread(app: FastAPI, config: GatewayConfig) -> threading.Thread:
    """Start the gateway server in a **daemon thread** (non-blocking).

    Returns the :class:`threading.Thread` that is running uvicorn so callers
    can monitor it if needed.  The thread is daemonised so it will not prevent
    the process from exiting.

    Args:
        app: A FastAPI application previously built by :func:`build_gateway_app`
            or :func:`build_local_gateway_app`.
        config: Gateway configuration supplying ``host``, ``port``, and
            ``log_level``.

    Returns:
        The started daemon thread.
    """
    thread = threading.Thread(
        target=serve_gateway,
        args=(app, config),
        daemon=True,
        name="gateway-server",
    )
    thread.start()
    return thread
