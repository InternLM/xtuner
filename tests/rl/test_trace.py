import json
import os
import subprocess
import sys
import unittest
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

    def test_injected_parent_carrier_links_child_span_in_another_process(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "parent-child")

        self.assertEqual(output["child"]["trace_id"], output["parent_trace_id"])
        self.assertEqual(output["child"]["parent_span_id"], output["parent_span_id"])

    def test_nested_trace_span_preserves_parent_to_child_order(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = _run_trace_utils(repo_root, "nested-span-order")

        self.assertEqual(output["child_parent_span_id"], output["parent_span_id"])
        self.assertEqual(output["span_name_paths"]["order.parent"], ["order.parent"])
        self.assertEqual(output["span_name_paths"]["order.child"], ["order.parent", "order.child"])

    def test_viewer_uses_span_name_path_for_display_chain(self):
        from xtuner.tools.trace_viewer.payload import build_rollout_view_payload_from_jaeger_traces

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
        from xtuner.tools.trace_viewer.payload import build_rollout_view_payload_from_jaeger_traces
        from xtuner.tools.trace_viewer.render import render_rollout_trace_html

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


if __name__ == "__main__":
    unittest.main()
