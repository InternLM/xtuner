import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from contextlib import contextmanager
from pathlib import Path


def _run_trace_utils(repo_root: Path, command: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, os.fspath(Path(__file__).with_name("trace_utils.py")), command],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTrace(unittest.TestCase):
    def test_trace_span_records_attributes_events_and_errors(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "record-span")

        self.assertEqual(output["success_attributes"]["xtuner.stage"], "unit")
        self.assertEqual(output["success_attributes"]["unit.count"], 1)
        self.assertEqual(output["success_events"], ["unit.event"])
        self.assertEqual(output["failure_status"], "ERROR")
        self.assertEqual(output["failure_attributes"]["error"], True)
        self.assertEqual(output["failure_attributes"]["error.type"], "RuntimeError")
        self.assertEqual(output["failure_attributes"]["error.message"], "boom")

    def test_mixed_case_parent_carrier_links_child_span_in_another_process(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "parent-child")

        self.assertEqual(output["child"]["trace_id"], output["parent_trace_id"])
        self.assertEqual(output["child"]["parent_span_id"], output["parent_span_id"])
        self.assertIn("Traceparent", output["child"]["carrier_keys"])

    def test_nested_trace_span_preserves_parent_to_child_order(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "nested-span-order")

        self.assertEqual(output["child_parent_span_id"], output["parent_span_id"])
        self.assertEqual(output["span_name_paths"]["order.parent"], ["order.parent"])
        self.assertEqual(output["span_name_paths"]["order.child"], ["order.parent", "order.child"])

    def test_rollout_remote_propagates_and_cleans_batch_carriers(self):
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.trace import rollout_api

        states = [RolloutState(message=[], rollout_id=index) for index in (1, 2)]
        observed_carriers = []

        class RemoteMethod:
            def remote(self, rollout_states):
                observed_carriers.extend(
                    dict(state.extra_fields[rollout_api.TRACE_CARRIER_EXTRA_FIELD]) for state in rollout_states
                )
                return "object-ref"

        with (
            unittest.mock.patch.object(rollout_api, "is_rollout_trace_enabled", return_value=True),
            unittest.mock.patch.object(
                rollout_api.trace_api,
                "inject_trace_context",
                return_value={"traceparent": "00-trace-span-01"},
            ),
        ):
            result = rollout_api.trace_rollout_remote(
                RemoteMethod(),
                states,
                target=states,
            )

        self.assertEqual(result, "object-ref")
        self.assertEqual(
            observed_carriers,
            [{"traceparent": "00-trace-span-01"}, {"traceparent": "00-trace-span-01"}],
        )
        self.assertTrue(all(rollout_api.TRACE_CARRIER_EXTRA_FIELD not in state.extra_fields for state in states))

    def test_rollout_remote_restores_existing_single_carrier(self):
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.trace import rollout_api

        previous_carrier = {"traceparent": "previous"}
        state = RolloutState(
            message=[],
            rollout_id=1,
            extra_fields={rollout_api.TRACE_CARRIER_EXTRA_FIELD: previous_carrier},
        )
        observed = {}

        class RemoteMethod:
            def remote(self, rollout_state):
                observed.update(rollout_state.extra_fields[rollout_api.TRACE_CARRIER_EXTRA_FIELD])
                return "object-ref"

        with (
            unittest.mock.patch.object(rollout_api, "is_rollout_trace_enabled", return_value=True),
            unittest.mock.patch.object(
                rollout_api.trace_api,
                "inject_trace_context",
                return_value={"traceparent": "current"},
            ),
        ):
            result = rollout_api.trace_rollout_remote(RemoteMethod(), state, target=state)

        self.assertEqual(result, "object-ref")
        self.assertEqual(observed, {"traceparent": "current"})
        self.assertIs(state.extra_fields[rollout_api.TRACE_CARRIER_EXTRA_FIELD], previous_carrier)

    def test_rollout_remote_accepts_empty_group_without_changing_call(self):
        from xtuner.v1.rl.trace import rollout_api

        class RemoteMethod:
            def __init__(self):
                self.calls = []

            def remote(self, rollout_states):
                self.calls.append(rollout_states)
                return "object-ref"

        for trace_enabled in (False, True):
            with self.subTest(trace_enabled=trace_enabled):
                remote_method = RemoteMethod()
                states = []
                with (
                    unittest.mock.patch.object(
                        rollout_api, "is_rollout_trace_enabled", return_value=trace_enabled
                    ),
                    unittest.mock.patch.object(
                        rollout_api.trace_api, "inject_trace_context", return_value={}
                    ),
                ):
                    result = rollout_api.trace_rollout_remote(remote_method, states, target=states)

                self.assertEqual(result, "object-ref")
                self.assertEqual(remote_method.calls, [states])

    def test_viewer_uses_span_name_path_for_display_chain(self):
        from recipe.trace.viewer.payload import build_rollout_view_payload_from_jaeger_traces

        traces = [
            {
                "traceID": "trace-1",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-1",
                        "spanID": "span-1",
                        "operationName": "parent.phase",
                        "processID": "p1",
                        "startTime": 1_000,
                        "duration": 2_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.span_name_path", "value": ["parent.phase"]},
                        ],
                    },
                    {
                        "traceID": "trace-1",
                        "spanID": "span-2",
                        "operationName": "child.phase",
                        "processID": "p1",
                        "startTime": 2_000,
                        "duration": 1_000,
                        "references": [{"refType": "CHILD_OF", "traceID": "trace-1", "spanID": "span-1"}],
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.span_name_path", "value": ["parent.phase", "child.phase"]},
                        ],
                    },
                ],
            }
        ]

        payload = build_rollout_view_payload_from_jaeger_traces(traces, train_step="all")

        self.assertEqual(
            [node["name"] for node in payload["samples"][0]["display_path"]],
            ["parent.phase", "child.phase"],
        )
        self.assertEqual(payload["samples"][0]["chain"], "parent.phase -> child.phase")

    def test_viewer_filters_latest_train_step_and_renders_payload(self):
        from recipe.trace.viewer.payload import build_rollout_view_payload_from_jaeger_traces
        from recipe.trace.viewer.render import render_rollout_trace_html

        traces = [
            {
                "traceID": "trace-1",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-1",
                        "spanID": "span-1",
                        "operationName": "old.operation",
                        "processID": "p1",
                        "startTime": 1_000,
                        "duration": 1_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-1"},
                            {"key": "xtuner.producer_future_step", "value": 1},
                            {"key": "xtuner.stage", "value": "stage_one"},
                        ],
                    }
                ],
            },
            {
                "traceID": "trace-2",
                "processes": {"p1": {"serviceName": "xtuner-test", "tags": []}},
                "spans": [
                    {
                        "traceID": "trace-2",
                        "spanID": "span-2",
                        "operationName": "new.operation",
                        "processID": "p1",
                        "startTime": 2_000,
                        "duration": 1_000,
                        "tags": [
                            {"key": "xtuner.rollout_id", "value": "rollout-2"},
                            {"key": "xtuner.producer_future_step", "value": 2},
                            {"key": "xtuner.stage", "value": "stage_two"},
                        ],
                    }
                ],
            },
        ]

        payload = build_rollout_view_payload_from_jaeger_traces(traces)
        html = render_rollout_trace_html(payload)

        self.assertEqual(payload["selected_train_step"], 2)
        self.assertEqual(payload["available_train_steps"], [1, 2])
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["samples"][0]["rollout_id"], "rollout-2")
        self.assertEqual(payload["samples"][0]["stage"], "stage_two")
        self.assertIn("XTuner Rollout Trace Viewer", html)
        self.assertIn("stage_two", html)


class TestRolloutEndpointTrace(unittest.IsolatedAsyncioTestCase):
    async def test_deep_copied_batch_results_do_not_leak_call_chain(self):
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.trace import rollout_api

        @contextmanager
        def passthrough_span(*args, **kwargs):
            yield

        for container_type in (list, tuple):
            with self.subTest(container_type=container_type.__name__):

                @rollout_api.trace_rollout_endpoint("test.endpoint")
                async def endpoint(rollout_state):
                    return container_type(
                        (rollout_state.model_copy(deep=True), rollout_state.model_copy(deep=True))
                    )

                state = RolloutState(message=[], rollout_id=1)
                with (
                    unittest.mock.patch.object(
                        rollout_api, "is_rollout_trace_enabled", return_value=True
                    ),
                    unittest.mock.patch.object(
                        rollout_api.trace_api, "trace_span", side_effect=passthrough_span
                    ),
                    unittest.mock.patch.object(rollout_api.trace_api, "set_trace_attributes"),
                ):
                    result = await endpoint(state)

                self.assertNotIn(rollout_api.TRACE_CALL_CHAIN_EXTRA_FIELD, state.extra_fields)
                self.assertTrue(
                    all(rollout_api.TRACE_CALL_CHAIN_EXTRA_FIELD not in item.extra_fields for item in result)
                )

    async def test_nested_endpoint_restores_existing_call_chain(self):
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.trace import rollout_api

        @rollout_api.trace_rollout_endpoint("test.inner")
        async def endpoint(rollout_state):
            return rollout_state.model_copy(deep=True)

        previous_call_chain = ["test.outer"]
        state = RolloutState(
            message=[],
            rollout_id=1,
            extra_fields={rollout_api.TRACE_CALL_CHAIN_EXTRA_FIELD: previous_call_chain},
        )
        observed_paths = []

        @contextmanager
        def capture_span(name, attributes=None, *, parent_carrier=None):
            observed_paths.append(attributes["xtuner.span_name_path"])
            yield

        with (
            unittest.mock.patch.object(rollout_api, "is_rollout_trace_enabled", return_value=True),
            unittest.mock.patch.object(rollout_api.trace_api, "trace_span", side_effect=capture_span),
            unittest.mock.patch.object(rollout_api.trace_api, "set_trace_attributes"),
        ):
            result = await endpoint(state)

        self.assertEqual(observed_paths, [("test.outer", "test.inner")])
        self.assertIs(state.extra_fields[rollout_api.TRACE_CALL_CHAIN_EXTRA_FIELD], previous_call_chain)
        self.assertIs(result.extra_fields[rollout_api.TRACE_CALL_CHAIN_EXTRA_FIELD], previous_call_chain)


class TestSessionServerTrace(unittest.IsolatedAsyncioTestCase):
    async def test_request_span_extracts_parent_and_records_status(self):
        from multidict import CIMultiDict

        from xtuner.v1.rl.rollout import session_server

        server = object.__new__(session_server.SessionServer)
        request = types.SimpleNamespace(
            match_info={"path": "v1/chat/completions"},
            headers=CIMultiDict(
                {"Traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}
            ),
            method="POST",
        )
        response = types.SimpleNamespace(status=201)
        observed = {}

        @contextmanager
        def capture_span(name, attributes=None, *, parent_carrier=None):
            observed.update(
                name=name,
                attributes=dict(attributes or {}),
                parent_carrier=parent_carrier,
            )
            yield

        with (
            unittest.mock.patch.object(session_server, "trace_span", side_effect=capture_span),
            unittest.mock.patch.object(session_server, "set_trace_attributes") as set_attributes,
            unittest.mock.patch.object(
                session_server.SessionServer,
                "_handle_request_impl",
                new=unittest.mock.AsyncMock(return_value=response),
            ),
        ):
            result = await server._handle_request(request)

        self.assertIs(result, response)
        self.assertEqual(observed["name"], "session_server.request")
        self.assertIs(observed["parent_carrier"], request.headers)
        self.assertEqual(observed["parent_carrier"]["traceparent"], request.headers["Traceparent"])
        set_attributes.assert_called_once_with({"http.response.status_code": 201, "error": False})

    async def test_backend_roundtrip_injects_current_context(self):
        from xtuner.v1.rl.rollout import session_server

        server = object.__new__(session_server.SessionServer)
        response = types.SimpleNamespace(status=503)
        forwarded = {}
        observed_spans = []

        class RequestContext:
            async def __aenter__(self):
                return response

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Client:
            def request(self, **kwargs):
                forwarded.update(kwargs)
                return RequestContext()

        @contextmanager
        def capture_span(name, attributes=None, **kwargs):
            observed_spans.append((name, dict(attributes or {})))
            yield

        def inject(headers):
            headers["traceparent"] = "injected"
            return headers

        headers = {}
        with (
            unittest.mock.patch.object(session_server, "trace_span", side_effect=capture_span),
            unittest.mock.patch.object(session_server, "inject_trace_context", side_effect=inject),
            unittest.mock.patch.object(session_server, "set_trace_attributes") as set_attributes,
        ):
            async with server._backend_request(
                Client(),
                method="POST",
                url="http://worker",
                headers=headers,
                trace_session_id="session-1",
            ):
                pass

        self.assertEqual(headers["traceparent"], "injected")
        self.assertIs(forwarded["headers"], headers)
        set_attributes.assert_called_once_with({"http.upstream.status_code": 503, "error": True})
        self.assertEqual(
            observed_spans,
            [
                (
                    "session_server.backend_roundtrip",
                    {
                        "xtuner.stage": "llm_backend_roundtrip",
                        "xtuner.session_id": "session-1",
                    },
                )
            ],
        )


class TestSandboxTraceBridge(unittest.TestCase):
    def test_legacy_span_is_preserved_and_otel_schema_stays_minimal(self):
        trace_path = Path(__file__).resolve().parents[2] / "xtuner/v1/rl/agent_loop/sandbox_agent_loop/trace.py"
        spec = importlib.util.spec_from_file_location("sandbox_trace_test", trace_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        sandbox_trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sandbox_trace)

        observed_spans = []
        observed_final_attributes = []

        @contextmanager
        def capture_span(name, attributes=None):
            observed_spans.append((name, dict(attributes or {})))
            yield

        with tempfile.TemporaryDirectory() as temp_dir, unittest.mock.patch.dict(
            os.environ, {"WORK_DIR": temp_dir}
        ):
            sandbox_trace._reset_for_testing()
            sandbox_trace.init_writer(actor_id="test")
            with (
                unittest.mock.patch.object(sandbox_trace, "trace_span", side_effect=capture_span),
                unittest.mock.patch.object(
                    sandbox_trace,
                    "set_trace_attributes",
                    side_effect=lambda attrs: observed_final_attributes.append(dict(attrs)),
                ),
            ):
                with sandbox_trace.span("session-1", "acquire", task_id="task-1") as handle:
                    handle.annotate(
                        sandbox_name="default",
                        sandbox_image="sandbox:latest",
                        sandbox_url="http://sandbox.internal/id",
                    )
                    handle.mark_error("acquire failed")
                with sandbox_trace.span("session-1", "run_total"):
                    pass
            sandbox_trace._reset_for_testing()

            legacy_files = list((Path(temp_dir) / "trace").glob("spans.*.jsonl"))
            self.assertEqual(len(legacy_files), 1)
            legacy_records = [json.loads(line) for line in legacy_files[0].read_text().splitlines()]

        self.assertEqual(observed_spans[0][0], "sandbox.acquire")
        self.assertEqual(observed_spans[0][1]["xtuner.session_id"], "session-1")
        self.assertEqual(observed_spans[1][0], "sandbox_agent.execute")
        self.assertEqual(observed_final_attributes[0]["sandbox.sandbox_name"], "default")
        self.assertTrue(observed_final_attributes[0]["error"])
        self.assertNotIn("sandbox.sandbox_url", observed_final_attributes[0])
        self.assertNotIn("sandbox.duration_ms", observed_final_attributes[0])
        self.assertNotIn("sandbox.ok", observed_final_attributes[0])
        self.assertEqual([record["event"] for record in legacy_records], ["enter", "exit", "enter", "exit"])
        self.assertEqual(legacy_records[1]["sandbox_url"], "http://sandbox.internal/id")
        self.assertEqual(legacy_records[1]["ok"], False)
        self.assertEqual(legacy_records[1]["err"], "acquire failed")


if __name__ == "__main__":
    unittest.main()
