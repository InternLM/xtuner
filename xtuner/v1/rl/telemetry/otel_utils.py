from __future__ import annotations

import contextlib
import secrets

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
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


def _reset_otel_tracer_provider_for_reconfigure() -> None:
    """Clear OTel's once-only global provider slot before XTuner reconfigures trace."""

    with contextlib.suppress(Exception):
        import opentelemetry.trace as trace_api

        trace_api._TRACER_PROVIDER = None
        trace_api._TRACER_PROVIDER_SET_ONCE = Once()
