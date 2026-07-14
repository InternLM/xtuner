import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter

import xtuner.v1.rl.trace.api as trace_api


def _json_safe(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _install_in_memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True))


def record_span() -> None:
    exporter = _install_in_memory_exporter()

    with (
        mock.patch.object(trace_api, "_ensure_trace_runtime_from_env"),
        mock.patch.object(trace_api, "is_trace_enabled", return_value=True),
    ):
        with trace_api.trace_span("unit.parent", attributes={"xtuner.stage": "unit"}):
            trace_api.set_trace_attributes({"unit.count": 1})
            trace_api.trace_event("unit.event", {"ok": True})

        try:
            with trace_api.trace_span("unit.failure"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    success = spans["unit.parent"]
    failure = spans["unit.failure"]
    _emit(
        {
            "success_attributes": {
                key: _json_safe(value) for key, value in success.attributes.items()
            },
            "success_events": [event.name for event in success.events],
            "failure_status": failure.status.status_code.name,
            "failure_attributes": {
                key: _json_safe(value) for key, value in failure.attributes.items()
            },
        }
    )


def child_span() -> None:
    carrier = json.loads(os.environ["XTUNER_TEST_TRACE_CARRIER"])
    exporter = _install_in_memory_exporter()

    with (
        mock.patch.object(trace_api, "_ensure_trace_runtime_from_env"),
        mock.patch.object(trace_api, "is_trace_enabled", return_value=True),
    ):
        with trace_api.trace_span("child.phase", parent_carrier=carrier):
            pass

    (span,) = exporter.get_finished_spans()
    _emit(
        {
            "trace_id": f"{span.context.trace_id:032x}",
            "span_id": f"{span.context.span_id:016x}",
            "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
            "span_name_path": list(span.attributes.get("xtuner.span_name_path") or []),
        }
    )


def parent_child() -> None:
    exporter = _install_in_memory_exporter()

    with (
        mock.patch.object(trace_api, "_ensure_trace_runtime_from_env"),
        mock.patch.object(trace_api, "is_trace_enabled", return_value=True),
    ):
        with trace_api.trace_span("parent.phase"):
            carrier = trace_api.inject_trace_context({})
            env = os.environ.copy()
            env["XTUNER_TEST_TRACE_CARRIER"] = json.dumps(carrier)
            child_result = subprocess.run(
                [sys.executable, os.fspath(Path(__file__).resolve()), "child-span"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

    (parent_span,) = exporter.get_finished_spans()
    child = json.loads(child_result.stdout.strip().splitlines()[-1])
    _emit(
        {
            "parent_trace_id": f"{parent_span.context.trace_id:032x}",
            "parent_span_id": f"{parent_span.context.span_id:016x}",
            "carrier": carrier,
            "child": child,
        }
    )


def nested_span_order() -> None:
    exporter = _install_in_memory_exporter()

    with (
        mock.patch.object(trace_api, "_ensure_trace_runtime_from_env"),
        mock.patch.object(trace_api, "is_trace_enabled", return_value=True),
    ):
        with trace_api.trace_span("order.parent"):
            with trace_api.trace_span("order.child"):
                pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    parent = spans["order.parent"]
    child = spans["order.child"]
    _emit(
        {
            "parent_span_id": f"{parent.context.span_id:016x}",
            "child_parent_span_id": f"{child.parent.span_id:016x}" if child.parent else None,
            "span_name_paths": {
                name: list(span.attributes.get("xtuner.span_name_path") or [])
                for name, span in spans.items()
            },
        }
    )


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "record-span":
        record_span()
    elif command == "child-span":
        child_span()
    elif command == "parent-child":
        parent_child()
    elif command == "nested-span-order":
        nested_span_order()
    else:
        raise SystemExit(f"unknown trace utils command: {command}")


if __name__ == "__main__":
    main()
