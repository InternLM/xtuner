from __future__ import annotations

import contextlib
import secrets
from typing import Any, Mapping, MutableMapping

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util._once import Once


class _SystemRandomIdGenerator(IdGenerator):
    """Generate OTel IDs without using Python's global random state."""

    def generate_span_id(self) -> int:
        span_id = secrets.randbits(64)
        while span_id == trace.INVALID_SPAN_ID:
            span_id = secrets.randbits(64)
        return span_id

    def generate_trace_id(self) -> int:
        trace_id = secrets.randbits(128)
        while trace_id == trace.INVALID_TRACE_ID:
            trace_id = secrets.randbits(128)
        return trace_id


def configure_tracer_provider(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    protocol: str = "grpc",
) -> TracerProvider:
    exporter = _build_otlp_span_exporter(endpoint, protocol=protocol)
    resource = Resource.create({"service.name": service_name, "run.id": run_id})
    provider = TracerProvider(resource=resource, id_generator=_SystemRandomIdGenerator())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def _build_otlp_span_exporter(endpoint: str, *, protocol: str = "grpc"):
    if protocol == "http/protobuf":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError(
                "XTuner OTel tracing requires the official OpenTelemetry OTLP HTTP trace exporter. "
                "Install `opentelemetry-exporter-otlp-proto-http` before enabling HTTP trace export."
            ) from exc
        return OTLPSpanExporter(endpoint=endpoint)
    if protocol != "grpc":
        raise ValueError(f"Unsupported OTel trace export protocol: {protocol!r}")
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise RuntimeError(
            "XTuner OTel tracing requires the official OpenTelemetry OTLP gRPC trace exporter. "
            "Install `opentelemetry-exporter-otlp-proto-grpc` before enabling trace."
        ) from exc
    return OTLPSpanExporter(endpoint=endpoint, insecure=True)


def inject_otel_context(
    context: Any | None = None,
    carrier: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Inject the current or provided OTel context into a W3C carrier."""

    carrier = carrier if carrier is not None else {}
    propagate.inject(carrier, context=context)
    return carrier


def extract_otel_context(
    carrier: Mapping[str, str],
    context: Any | None = None,
) -> Any:
    """Extract W3C TraceContext from a carrier into an OTel context."""

    return propagate.extract(carrier, context=context)


def attach_otel_context(context: Any) -> object:
    """Attach extracted context to the current execution scope."""

    return otel_context.attach(context)


def detach_otel_context(token: object) -> None:
    """Detach a previously attached OTel context token."""

    otel_context.detach(token)


def get_otel_tracer(name: str = "xtuner"):
    """Return an OTel tracer from the current process-global provider."""

    return trace.get_tracer(name)


def start_span(name: str, *, attributes: Mapping[str, Any] | None = None):
    """Start a current OTel span with XTuner-managed exception handling."""

    return get_otel_tracer().start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    )


def add_event(name: str, *, attributes: Mapping[str, Any] | None = None) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.add_event(name, attributes=attributes)


def set_attributes(attributes: Mapping[str, Any]) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)


def set_error_status(message: str | None = None) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute("error", True)
    description = message or "error"
    span.set_attribute("error.message", description)
    span.set_status(Status(StatusCode.ERROR, description))


def record_failure(exc: BaseException) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    error_attributes = {
        "error.type": type(exc).__name__,
        "error.message": str(exc),
        "error": True,
    }
    for key, value in error_attributes.items():
        span.set_attribute(key, value)
    span.record_exception(exc, attributes=error_attributes)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _reset_otel_tracer_provider_for_reconfigure() -> None:
    """Clear OTel's once-only global provider slot before XTuner reconfigures
    trace."""

    with contextlib.suppress(Exception):
        import opentelemetry.trace as trace_api

        trace_api._TRACER_PROVIDER = None
        trace_api._TRACER_PROVIDER_SET_ONCE = Once()
