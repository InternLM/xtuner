import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.rl.telemetry.runtime import get_trace_env_vars
from xtuner.v1.rl.trace import (
    TraceConfig,
    close_trace,
    configure_trace,
    current_trace_runtime,
    ensure_trace_runtime_from_env,
    reset_trace_for_test,
)
from xtuner.v1.rl.utils.ray_utils import merge_trace_runtime_env, with_trace_runtime_env


_TRACE_ENV_KEYS = (
    "XTUNER_OTEL_ENABLED",
    "XTUNER_OTEL_OUTPUT_DIR",
    "XTUNER_OTEL_RUN_ID",
    "XTUNER_OTEL_RUN_DIR",
    "XTUNER_OTEL_JSONL_PATH",
    "OTEL_TRACES_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_SERVICE_NAME",
    "XTUNER_OTEL_ENDPOINT",
)


class _DummyProvider:
    def __init__(self) -> None:
        self.shutdown_count = 0

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _DummyCollector:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class TestTraceRuntime(unittest.TestCase):
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

    def test_configure_trace_creates_run_dir_and_exports_otel_env(self):
        provider = _DummyProvider()
        collector = _DummyCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14318]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector) as start_collector,
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider),
            ):
                runtime = configure_trace(
                    TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test")
                )

        self.assertTrue(runtime.enabled)
        self.assertEqual(runtime.mode, "driver")
        self.assertEqual(runtime.run_dir.parent, Path(tmpdir))
        self.assertEqual(runtime.trace_jsonl_path, runtime.run_dir / "traces" / "traces.jsonl")
        self.assertTrue(runtime.trace_jsonl_path.parent.is_dir())
        self.assertEqual(get_trace_env_vars()["XTUNER_OTEL_RUN_ID"], runtime.run_id)
        self.assertEqual(get_trace_env_vars()["OTEL_SERVICE_NAME"], "xtuner-test")
        start_collector.assert_called_once_with(port=14318, output_path=runtime.trace_jsonl_path)

    def test_configure_trace_ignores_external_forward_endpoint_configuration(self):
        provider = _DummyProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["XTUNER_OTEL_ENDPOINT"] = "http://127.0.0.1:24317"
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14318]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start"),
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider) as configure,
            ):
                runtime = configure_trace(
                    TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-jaeger-smoke")
                )

        env = get_trace_env_vars()
        self.assertEqual(env["OTEL_EXPORTER_OTLP_ENDPOINT"], "http://127.0.0.1:14318")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"], "http://127.0.0.1:14318")
        self.assertEqual(env["OTEL_EXPORTER_OTLP_PROTOCOL"], "grpc")
        self.assertEqual(env["XTUNER_OTEL_JSONL_PATH"], os.fspath(runtime.trace_jsonl_path))
        self.assertNotIn("XTUNER_OTEL_FORWARD_ENDPOINT", env)
        configure.assert_called_once_with(
            service_name="xtuner-jaeger-smoke",
            run_id=runtime.run_id,
            endpoint="http://127.0.0.1:14318",
            protocol="grpc",
        )

    def test_child_process_can_lazily_configure_trace_from_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ.update(
                {
                    "XTUNER_OTEL_ENABLED": "1",
                    "XTUNER_OTEL_OUTPUT_DIR": tmpdir,
                    "XTUNER_OTEL_RUN_ID": "run-from-parent",
                    "XTUNER_OTEL_RUN_DIR": os.fspath(Path(tmpdir) / "run-from-parent"),
                    "OTEL_TRACES_EXPORTER": "otlp",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:14317",
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://127.0.0.1:14317",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
                    "OTEL_SERVICE_NAME": "xtuner-child",
                }
            )
            provider = _DummyProvider()
            with (
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start") as start_collector,
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider) as configure,
            ):
                self.assertTrue(ensure_trace_runtime_from_env())

        runtime = current_trace_runtime()
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.mode, "inherited")
        start_collector.assert_not_called()
        configure.assert_called_once_with(
            service_name="xtuner-child",
            run_id="run-from-parent",
            endpoint="http://127.0.0.1:14317",
            protocol="grpc",
        )

    def test_close_trace_owns_provider_and_collector_once(self):
        provider = _DummyProvider()
        collector = _DummyCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector),
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider),
            ):
                configure_trace(TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test"))

        close_trace()
        close_trace()

        self.assertEqual(provider.shutdown_count, 1)
        self.assertEqual(collector.close_count, 1)

    def test_configure_trace_replaces_previous_runtime_and_clears_env_on_close(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            first_provider = _DummyProvider()
            second_provider = _DummyProvider()
            first_collector = _DummyCollector()
            second_collector = _DummyCollector()
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", side_effect=[[14317], [14318]]),
                patch(
                    "xtuner.v1.rl.telemetry.runtime.OTelCollector.start",
                    side_effect=[first_collector, second_collector],
                ),
                patch(
                    "xtuner.v1.rl.telemetry.runtime.configure_tracer_provider",
                    side_effect=[first_provider, second_provider],
                ),
            ):
                first = configure_trace(TraceConfig(enabled=True, output_dir=Path(left), service_name="xtuner-test"))
                second = configure_trace(TraceConfig(enabled=True, output_dir=Path(right), service_name="xtuner-test"))

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(second.run_dir.parent, Path(right))
        self.assertEqual(first_provider.shutdown_count, 1)
        self.assertEqual(first_collector.close_count, 1)

    def test_configure_trace_cleans_up_when_provider_fails(self):
        collector = _DummyCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector),
                patch(
                    "xtuner.v1.rl.telemetry.runtime.configure_tracer_provider",
                    side_effect=RuntimeError("provider failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    configure_trace(TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-test"))

        self.assertEqual(collector.close_count, 1)
        self.assertEqual(get_trace_env_vars(), {})

    def test_merge_trace_runtime_env_preserves_existing_env_and_adds_active_runtime(self):
        provider = _DummyProvider()
        collector = _DummyCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector),
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider),
            ):
                runtime = configure_trace(
                    TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-runtime-env-test")
                )
                runtime_env = merge_trace_runtime_env(
                    {
                        "working_dir": ".",
                        "env_vars": {
                            "BACKEND_ENV": "1",
                            "OTEL_SERVICE_NAME": "backend-service",
                        },
                    }
                )

        self.assertEqual(runtime_env["working_dir"], ".")
        self.assertEqual(runtime_env["env_vars"]["BACKEND_ENV"], "1")
        self.assertEqual(runtime_env["env_vars"]["XTUNER_OTEL_RUN_ID"], runtime.run_id)
        self.assertEqual(runtime_env["env_vars"]["OTEL_SERVICE_NAME"], "backend-service")

    def test_merge_trace_runtime_env_is_noop_when_trace_disabled(self):
        runtime_env = {"env_vars": {"BACKEND_ENV": "1"}}

        self.assertEqual(merge_trace_runtime_env(runtime_env), runtime_env)
        self.assertIsNot(merge_trace_runtime_env(runtime_env), runtime_env)

    def test_with_trace_runtime_env_preserves_options_and_merges_runtime_env(self):
        provider = _DummyProvider()
        collector = _DummyCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("xtuner.v1.rl.telemetry.runtime.find_free_ports", return_value=[14317]),
                patch("xtuner.v1.rl.telemetry.runtime.OTelCollector.start", return_value=collector),
                patch("xtuner.v1.rl.telemetry.runtime.configure_tracer_provider", return_value=provider),
            ):
                runtime = configure_trace(
                    TraceConfig(enabled=True, output_dir=Path(tmpdir), service_name="xtuner-runtime-options-test")
                )
                options = {"num_cpus": 1, "runtime_env": {"env_vars": {"BACKEND_ENV": "1"}}}
                merged_options = with_trace_runtime_env(options)

        self.assertEqual(merged_options["num_cpus"], 1)
        self.assertEqual(merged_options["runtime_env"]["env_vars"]["BACKEND_ENV"], "1")
        self.assertEqual(merged_options["runtime_env"]["env_vars"]["XTUNER_OTEL_RUN_ID"], runtime.run_id)
        self.assertNotIn("XTUNER_OTEL_RUN_ID", options["runtime_env"]["env_vars"])

    def test_rl_trainer_common_init_configures_trace_from_train_config(self):
        from xtuner.v1.train import rl_trainer

        trainer = object.__new__(rl_trainer.BaseRLTrainer)
        trace_runtime = object()
        cfg = SimpleNamespace(
            trace_config=TraceConfig(
                enabled=True,
                output_dir=Path(tempfile.gettempdir()) / "xtuner-trainer-trace-unused",
                service_name="xtuner-trainer-test",
            ),
            advantage_estimator_config=SimpleNamespace(build=lambda: object()),
            exp_tracker="jsonl",
        )
        log_dir = Path(tempfile.gettempdir())

        with (
            patch("xtuner.v1.train.rl_trainer.check_fa3"),
            patch("xtuner.v1.train.rl_trainer.configure_trace", return_value=trace_runtime) as configure_trace_mock,
            patch("xtuner.v1.train.rl_trainer.get_writer", return_value=object()),
            patch.object(rl_trainer.BaseRLTrainer, "_init_work_dir_and_meta"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_load_source"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_save_config"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_logger", return_value=log_dir),
            patch.object(rl_trainer.BaseRLTrainer, "_save_runtime_environment"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_train_state"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_train_worker_config"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_rollout_config"),
            patch.object(rl_trainer.BaseRLTrainer, "_ensure_rollout_proxy_config"),
            patch.object(rl_trainer.BaseRLTrainer, "_init_runtime_flags"),
        ):
            trainer._init_common(cfg, meta_path=".meta", logger_tag="RLTrainer")

        configure_trace_mock.assert_called_once_with(cfg.trace_config)
        self.assertIs(trainer._trace_runtime, trace_runtime)
