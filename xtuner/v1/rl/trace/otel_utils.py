from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from opentelemetry import baggage, propagate, trace
from opentelemetry import context as otel_context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


def configure_tracer_provider(
    *,
    service_name: str,
    run_id: str,
    endpoint: str,
    protocol: str = "grpc",
) -> TracerProvider:
    if protocol != "grpc":
        raise ValueError(f"Unsupported OTel trace export protocol: {protocol!r}")
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise RuntimeError(
            "XTuner OTel tracing requires the official OpenTelemetry OTLP gRPC trace exporter. "
            "Install `opentelemetry-exporter-otlp-proto-grpc` before enabling trace."
        ) from exc
    resource = Resource.create({"service.name": service_name, "run.id": run_id})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    return provider


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


def context_with_baggage(name: str, value: object, context: Any | None = None) -> Any:
    """Return a context with one W3C Baggage item attached."""

    return baggage.set_baggage(name, value, context=context)


def get_baggage(name: str, context: Any | None = None) -> object | None:
    """Read one W3C Baggage item from a context."""

    return baggage.get_baggage(name, context=context)


def attach_otel_context(context: Any) -> object:
    """Attach extracted context to the current execution scope."""

    return otel_context.attach(context)


def detach_otel_context(token: object) -> None:
    """Detach a previously attached OTel context token."""

    otel_context.detach(token)


def start_span(name: str, *, attributes: Mapping[str, Any] | None = None):
    """Start a current OTel span with XTuner-managed exception handling."""

    return trace.get_tracer("xtuner").start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    )


def current_span_ids() -> dict[str, str] | None:
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return None
    return {
        "trace_id": f"{span_context.trace_id:032x}",
        "span_id": f"{span_context.span_id:016x}",
    }


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
