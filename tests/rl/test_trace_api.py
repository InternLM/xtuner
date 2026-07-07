import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:  # pragma: no cover - older OTel SDK path
    from opentelemetry.sdk.trace.export import InMemorySpanExporter

import xtuner.v1.rl.trace as trace_api
import xtuner.v1.rl.telemetry as telemetry_api
from xtuner.v1.rl.trace import (
    TraceConfig,
    TraceContext,
    attach_trace_context,
    configure_trace,
    extract_trace_context,
    inject_trace_context,
    reset_trace_for_test,
    set_trace_attribute,
    set_trace_attributes,
    trace_event,
    trace_function,
    trace_span,
)


class _DummyCollector:
    def close(self) -> None:
        return None


class TestTraceAPI(unittest.TestCase):
    def setUp(self) -> None:
        reset_trace_for_test()
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        trace.set_tracer_provider(self.provider)

    def tearDown(self) -> None:
        self.provider.shutdown()
        reset_trace_for_test()

    def _configure_trace_with_in_memory_provider(self, tmpdir: str) -> None:
        patches = [
            patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
            patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=_DummyCollector()),
            patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=self.provider),
        ]
        for trace_patch in patches:
            trace_patch.start()
        self.addCleanup(lambda: [trace_patch.stop() for trace_patch in reversed(patches)])
        configure_trace(TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test"))

    def test_trace_span_records_attributes_and_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._configure_trace_with_in_memory_provider(tmpdir)

            with trace_span(
                "test.span",
                attributes={"global_step": 7, "ignored": object(), "mixed": [1, "two"]},
            ):
                set_trace_attribute("rollout.backend", "vllm")
                set_trace_attributes({"request.id": "req-1", "scores": [1, 2, 3]})
                trace_event("request.sent", {"payload.bytes": 128})

        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "test.span")
        self.assertEqual(span.attributes["global_step"], 7)
        self.assertEqual(span.attributes["ignored"], "<object>")
        self.assertEqual(span.attributes["mixed"], ("1", "two"))
        self.assertEqual(span.attributes["rollout.backend"], "vllm")
        self.assertEqual(span.attributes["request.id"], "req-1")
        self.assertEqual(span.attributes["scores"], (1, 2, 3))
        self.assertEqual(len(span.events), 1)
        self.assertEqual(span.events[0].name, "request.sent")
        self.assertEqual(span.events[0].attributes["payload.bytes"], 128)

    def test_trace_span_records_exception_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._configure_trace_with_in_memory_provider(tmpdir)

            with self.assertRaisesRegex(ValueError, "bad request"):
                with trace_span("rollout_controller.generate"):
                    raise ValueError("bad request")

        (span,) = self.exporter.get_finished_spans()
        self.assertEqual(span.status.status_code.name, "ERROR")
        self.assertEqual(span.attributes["error.type"], "ValueError")
        self.assertEqual(span.attributes["error.message"], "bad request")
        self.assertTrue(any(event.name == "exception" for event in span.events))

    def test_trace_function_wraps_sync_and_async_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._configure_trace_with_in_memory_provider(tmpdir)

            @trace_function("reward.compute", attributes={"reward.kind": "rule"})
            def compute_reward(value):
                return value + 1

            @trace_function("rollout.async_step")
            async def async_step(value):
                return value * 2

            self.assertEqual(compute_reward(3), 4)
            self.assertEqual(asyncio.run(async_step(5)), 10)

        spans = self.exporter.get_finished_spans()
        self.assertEqual([span.name for span in spans], ["reward.compute", "rollout.async_step"])
        self.assertEqual(spans[0].attributes["reward.kind"], "rule")

    def test_otel_provider_alone_does_not_enable_xtuner_trace(self):
        with trace_span("rollout.batch"):
            trace_event("request.sent")
            set_trace_attribute("global_step", 1)

        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

    def test_trace_apis_noop_without_configured_provider_or_current_span(self):
        reset_trace_for_test()

        trace_event("request.sent", {"request.id": "req-1"})
        set_trace_attribute("global_step", 1)
        set_trace_attributes({"rollout.backend": "vllm"})
        carrier = inject_trace_context()
        context = extract_trace_context(carrier)

        self.assertEqual(carrier, {})
        with attach_trace_context(context):
            with trace_span("disabled.span"):
                return_value = "ok"
        self.assertEqual(return_value, "ok")

    def test_context_propagation_restores_parent_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._configure_trace_with_in_memory_provider(tmpdir)

            with trace_span("parent"):
                carrier: dict[str, str] = {}
                returned_carrier = inject_trace_context(carrier)

            self.assertIs(returned_carrier, carrier)

            extracted_context = extract_trace_context(carrier)
            with attach_trace_context(extracted_context):
                with trace_span("child"):
                    pass

        spans = self.exporter.get_finished_spans()
        parent = next(span for span in spans if span.name == "parent")
        child = next(span for span in spans if span.name == "child")
        self.assertEqual(child.parent.span_id, parent.context.span_id)
        self.assertEqual(child.context.trace_id, parent.context.trace_id)

    def test_extract_trace_context_returns_opaque_trace_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._configure_trace_with_in_memory_provider(tmpdir)

            with trace_span("parent"):
                carrier = inject_trace_context()

        extracted_context = extract_trace_context(carrier)

        self.assertIsInstance(extracted_context, TraceContext)
        self.assertFalse(hasattr(extracted_context, "trace_id"))

    def test_attach_trace_context_detaches_after_exit(self):
        context = extract_trace_context({})
        before = otel_context.get_current()

        with attach_trace_context(context):
            pass

        self.assertIs(otel_context.get_current(), before)

    def test_attach_trace_context_requires_trace_context_value(self):
        with self.assertRaisesRegex(TypeError, "TraceContext"):
            with attach_trace_context(object()):
                pass

    def test_context_propagation_noops_when_xtuner_trace_disabled(self):
        carrier = {"traceparent": "invalid-value-kept-as-data"}

        returned_carrier = inject_trace_context(carrier)
        extracted_context = extract_trace_context(carrier)

        self.assertIs(returned_carrier, carrier)
        self.assertEqual(carrier, {"traceparent": "invalid-value-kept-as-data"})
        self.assertIsInstance(extracted_context, TraceContext)
        with attach_trace_context(extracted_context):
            with trace_span("disabled.child"):
                pass
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

    def test_invalid_names_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "span name"):
            with trace_span(""):
                pass
        with self.assertRaisesRegex(ValueError, "event name"):
            trace_event("")
        with self.assertRaisesRegex(ValueError, "attribute key"):
            set_trace_attribute("", "value")
        with self.assertRaisesRegex(TypeError, "attributes"):
            set_trace_attributes([("key", "value")])

# Trace config tests
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from xtuner.v1.rl.telemetry.runtime import get_trace_env_vars
from xtuner.v1.rl.trace import TraceConfig, configure_trace, reset_trace_for_test


_TRACE_ENV_KEYS = (
    "XTUNER_OTEL_ENABLED",
    "XTUNER_OTEL_OUTPUT_DIR",
    "XTUNER_OTEL_RUN_ID",
    "XTUNER_OTEL_RUN_DIR",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
)


class TestTraceConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {key: os.environ.get(key) for key in _TRACE_ENV_KEYS}
        reset_trace_for_test()

    def tearDown(self) -> None:
        reset_trace_for_test()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_public_config_uses_output_dir_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test")

            self.assertEqual(cfg.output_dir, Path(tmpdir))
            self.assertFalse(hasattr(cfg, "otlp_output_dir"))
            self.assertEqual(cfg.service_name, "xtuner-test")

    def test_public_config_has_no_environment_factory(self):
        self.assertFalse(hasattr(TraceConfig, "from_env"))

    def test_deprecated_otlp_output_dir_is_validation_alias_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TraceConfig.model_validate({"enabled": True, "otlp_output_dir": tmpdir})

            self.assertEqual(cfg.output_dir, Path(tmpdir))
            self.assertFalse(hasattr(cfg, "otlp_output_dir"))

    def test_conflicting_output_dir_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            with self.assertRaisesRegex(ValueError, "output_dir and otlp_output_dir cannot differ"):
                TraceConfig.model_validate({"enabled": True, "output_dir": left, "otlp_output_dir": right})

    def test_otel_runtime_options_are_not_public_config_fields(self):
        with self.assertRaises(ValidationError):
            TraceConfig(enabled=True, otel_endpoint="http://127.0.0.1:4317")

        with self.assertRaises(ValidationError):
            TraceConfig(enabled=True, otel_auto_start_collector=False)

        with self.assertRaises(ValidationError):
            TraceConfig(enabled=True, max_jsonl_lines=1)

    def test_runtime_mode_type_is_not_public_trace_api(self):
        self.assertIn("TraceRuntime", trace_api.__all__)
        self.assertNotIn("TraceRuntimeMode", trace_api.__all__)
        self.assertNotIn("ensure_trace_runtime_from_env", trace_api.__all__)
        self.assertNotIn("get_trace_env_vars", trace_api.__all__)
        self.assertNotIn("reset_trace_for_test", trace_api.__all__)
        self.assertFalse(hasattr(telemetry_api, "TraceRuntimeMode"))
        self.assertFalse(hasattr(telemetry_api, "ensure_trace_runtime_from_env"))
        self.assertFalse(hasattr(telemetry_api, "get_trace_env_vars"))
        self.assertFalse(hasattr(telemetry_api, "reset_trace_for_test"))

    def test_configure_trace_uses_output_dir_and_managed_local_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = _ConfigDummyCollector()
            provider = _ConfigDummyProvider()
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector),
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider),
            ):
                runtime = configure_trace(TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test"))

                self.assertTrue(runtime.enabled)
                self.assertEqual(runtime.run_dir.parent, Path(tmpdir))
                env = get_trace_env_vars()
                self.assertEqual(env["XTUNER_OTEL_OUTPUT_DIR"], os.fspath(Path(tmpdir)))
                self.assertEqual(env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"], "http://127.0.0.1:14317")
                self.assertEqual(env["OTEL_SERVICE_NAME"], "xtuner-test")


class _ConfigDummyProvider:
    def shutdown(self) -> None:
        return None


class _ConfigDummyCollector:
    def start(self):
        return self

    def close(self) -> None:
        return None
