import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import ray
import torch
from fastapi.testclient import TestClient

from xtuner.v1.rl.gateway.adapters import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponsesRequest,
    ResponsesResponse,
)
from xtuner.v1.rl.gateway import build_local_gateway_app
from xtuner.v1.rl.gateway.config import GatewayConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
RESOURCE_MAP = {
    "npu": "NPU",
    "cuda": "GPU",
}


@unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
class TestGatewayProtocolChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]

    def setUp(self):
        ray.init(address="local", ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.capture_output_path = Path(self.temp_dir.name) / "gateway_capture_output.jsonl"
        self.openai_body_output_path = Path(self.temp_dir.name) / "openai_body.json"
        self.anthropic_body_output_path = Path(self.temp_dir.name) / "anthropic_body.json"
        self.responses_body_output_path = Path(self.temp_dir.name) / "responses_body.json"
        self.controller = None
        self.placement_group = None
        self._capture_line_count_before = self._read_capture_records_count()

    def tearDown(self):
        if self.controller is not None:
            self.controller.shutdown()
        if self.placement_group is not None:
            ray.util.remove_placement_group(self.placement_group)
        ray.shutdown()
        self._cleanup_lmdeploy_ray_worker_wrapper()
        self.temp_dir.cleanup()

    def _cleanup_lmdeploy_ray_worker_wrapper(self):
        try:
            subprocess.run(
                ["pkill", "-f", "ray::RayWorkerWrapper*"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            return

    def _build_controller(self) -> RolloutController:
        resource_config = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=16,
            cpu_memory_per_worker=8 * 1024**3,
        )
        self.placement_group = AutoAcceleratorWorkers.build_placement_group(resource_config, name="gateway_protocol_pg")
        rollout_config = RolloutConfig(
            env="test_gateway_protocol",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=4,
            expert_parallel_size=1,
            context_length=1536,
            worker_log_dir=os.path.join(self.worker_log_dir, "gateway"),
            dist_port_base=42000,
            api_host="127.0.0.1",
            api_port=30080,
        )
        return RolloutController(rollout_config, self.placement_group)

    def _read_capture_records_count(self) -> int:
        if not self.capture_output_path.exists():
            return 0
        with self.capture_output_path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _read_new_capture_records(self) -> list[dict]:
        if not self.capture_output_path.exists():
            return []
        with self.capture_output_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[self._capture_line_count_before :]]

    def _write_json_output(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _capture_records_by_protocol(self, capture_records: list[dict]) -> dict[str, dict]:
        return {record["protocol"]: record for record in capture_records}

    def _assert_trace_record_matches_capture(
        self,
        trace_record,
        capture_record: dict,
        *,
        expected_request_field: str,
        expected_request_role: str | None,
        expected_response_field: str,
    ) -> None:
        self.assertIsNotNone(trace_record)
        self.assertEqual(trace_record.request_id, capture_record["request_id"])
        self.assertEqual(trace_record.finish_reason, capture_record["rollout_finish_reason"] or capture_record["finish_reason"])
        self.assertEqual(trace_record.status.value, capture_record["status"])
        self.assertGreater(len(trace_record.prompt_ids), 0)
        self.assertGreater(len(trace_record.response_ids), 0)
        self.assertGreater(capture_record["prompt_tokens"], 0)
        self.assertGreater(capture_record["completion_tokens"], 0)
        self.assertTrue(capture_record["input_text"])
        self.assertIn(expected_request_field, trace_record.request_snapshot)
        if expected_request_role is not None:
            self.assertEqual(trace_record.request_snapshot[expected_request_field][0]["role"], expected_request_role)
        self.assertIn(expected_response_field, trace_record.response_snapshot)

    def test_gateway_routes_with_real_rollout_controller_capture_protocol_traces(self):
        self.controller = self._build_controller()
        app = build_local_gateway_app(
            self.controller,
            config=GatewayConfig(port=8080, capture_path=str(self.capture_output_path)),
        )

        openai_payload = {
            "model": self.controller.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Think before answering."},
                {"role": "user", "content": "Use the search tool and then summarize the weather."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_seed_chat",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"q": "Beijing weather"}, ensure_ascii=False),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_seed_chat",
                    "content": json.dumps({"result": "Sunny, 26C"}, ensure_ascii=False),
                },
                {"role": "user", "content": "Now give me a short final answer."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the latest weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
        }
        anthropic_payload = {
            "model": self.controller.config.model_name,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Please reason and call the weather tool."}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "Need weather data first."},
                        {
                            "type": "tool_use",
                            "id": "toolu_seed_messages",
                            "name": "search",
                            "input": {"q": "Beijing weather"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_seed_messages",
                            "content": [{"type": "text", "text": "Sunny, 26C"}],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the latest weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "search"},
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        responses_payload = {
            "model": self.controller.config.model_name,
            "instructions": "You are a helpful assistant. Reason before you answer.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Need the weather in Beijing."}],
                },
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Need to inspect the tool result before answering."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_seed_responses",
                    "name": "search",
                    "arguments": json.dumps({"q": "Beijing weather"}, ensure_ascii=False),
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_seed_responses",
                    "output": [{"type": "text", "text": "Sunny, 26C"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize it in one sentence."}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search the latest weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"effort": "medium"},
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 64,
        }

        with TestClient(app) as client:
            openai_adapter = app.state.gateway_openai_adapter
            anthropic_adapter = app.state.gateway_anthropic_adapter
            responses_adapter = app.state.gateway_responses_adapter

            openai_response = client.post("/v1/chat/completions", json=openai_payload)
            self.assertEqual(openai_response.status_code, 200, openai_response.text)
            openai_body = openai_response.json()
            self._write_json_output(self.openai_body_output_path, openai_body)
            print("openai_body:", json.dumps(openai_body, ensure_ascii=False, indent=2))
            self.assertEqual(openai_body["model"], self.controller.config.model_name)
            self.assertEqual(openai_body["choices"][0]["message"]["role"], "assistant")
            self.assertIn(openai_body["choices"][0]["finish_reason"], {"stop", "length", "tool_calls"})
            self.assertGreater(openai_body["usage"]["prompt_tokens"], 0)
            self.assertNotIn("<think>", openai_body["choices"][0]["message"].get("content") or "")
            self.assertTrue(openai_body["choices"][0]["message"].get("reasoning_content"))

            anthropic_response = client.post("/v1/messages", json=anthropic_payload)
            self.assertEqual(anthropic_response.status_code, 200, anthropic_response.text)
            anthropic_body = anthropic_response.json()
            self._write_json_output(self.anthropic_body_output_path, anthropic_body)
            print("anthropic_body:", json.dumps(anthropic_body, ensure_ascii=False, indent=2))
            self.assertEqual(anthropic_body["type"], "message")
            self.assertEqual(anthropic_body["role"], "assistant")
            self.assertEqual(anthropic_body["model"], self.controller.config.model_name)
            self.assertGreater(anthropic_body["usage"]["input_tokens"], 0)
            self.assertTrue(anthropic_body["content"])

            responses_response = client.post("/v1/responses", json=responses_payload)
            self.assertEqual(responses_response.status_code, 200, responses_response.text)
            responses_body = responses_response.json()
            self._write_json_output(self.responses_body_output_path, responses_body)
            print("responses_body:", json.dumps(responses_body, ensure_ascii=False, indent=2))
            self.assertEqual(responses_body["object"], "response")
            self.assertEqual(responses_body["model"], self.controller.config.model_name)
            self.assertGreater(responses_body["usage"]["input_tokens"], 0)
            self.assertTrue(responses_body["output"])

            openai_trace = openai_adapter.get_trace_by_request_response(
                ChatCompletionRequest.model_validate(openai_payload),
                ChatCompletionResponse.model_validate(openai_body),
            )
            self.assertIsNotNone(openai_trace)
            self.assertIsNotNone(openai_adapter.get_trace_by_response_hash(openai_trace.response_hash))

            anthropic_trace = anthropic_adapter.get_trace_by_request_response(
                AnthropicMessagesRequest.model_validate(anthropic_payload),
                AnthropicMessagesResponse.model_validate(anthropic_body),
            )
            self.assertIsNotNone(anthropic_trace)
            self.assertIsNotNone(anthropic_adapter.get_trace_by_response_hash(anthropic_trace.response_hash))

            responses_trace = responses_adapter.get_trace_by_request_response(
                ResponsesRequest.model_validate(responses_payload),
                ResponsesResponse.model_validate(responses_body),
            )
            self.assertIsNotNone(responses_trace)
            self.assertIsNotNone(responses_adapter.get_trace_by_response_hash(responses_trace.response_hash))

        capture_records = self._read_new_capture_records()
        self.assertGreaterEqual(len(capture_records), 3)
        protocol_records = {record["protocol"]: record for record in capture_records[-3:]}
        self.assertIn("OpenAIChatAdapter", protocol_records)
        self.assertIn("AnthropicChatAdapter", protocol_records)
        self.assertIn("OpenAIResponsesAdapter", protocol_records)

        openai_record = protocol_records["OpenAIChatAdapter"]
        self.assertTrue(any(message.get("tool_calls") for message in openai_record["request"]["messages"]))
        self.assertTrue(openai_record["internal_messages"])
        self.assertEqual(openai_record["request_id"], openai_trace.request_id)
        self.assertEqual(openai_record["output_messages"][0]["role"], "assistant")
        self.assertTrue(any(item["type"] == "reasoning" for item in openai_record["output_messages"][0]["content"]))
        self.assertTrue(openai_record["input_text"])
        self._assert_trace_record_matches_capture(
            openai_trace,
            openai_record,
            expected_request_field="messages",
            expected_request_role="system",
            expected_response_field="choices",
        )
        self.assertEqual(openai_trace.request_snapshot["messages"][2]["tool_calls"][0]["function"]["name"], "search")
        self.assertEqual(openai_trace.response_snapshot["choices"][0]["message"]["role"], "assistant")

        anthropic_record = protocol_records["AnthropicChatAdapter"]
        anthropic_blocks = anthropic_record["request"]["messages"][1]["content"]
        self.assertTrue(any(block.get("type") == "tool_use" for block in anthropic_blocks))
        self.assertTrue(any(block.get("type") == "thinking" for block in anthropic_blocks))
        self.assertEqual(anthropic_record["request_id"], anthropic_trace.request_id)
        self.assertTrue(anthropic_record["output_messages"][0]["content"])
        self._assert_trace_record_matches_capture(
            anthropic_trace,
            anthropic_record,
            expected_request_field="messages",
            expected_request_role="user",
            expected_response_field="content",
        )
        self.assertEqual(anthropic_trace.request_snapshot["messages"][0]["role"], "user")
        self.assertEqual(anthropic_trace.request_snapshot["messages"][1]["role"], "assistant")
        self.assertEqual(anthropic_trace.response_snapshot["role"], "assistant")

        responses_record = protocol_records["OpenAIResponsesAdapter"]
        self.assertTrue(any(item.get("type") == "reasoning" for item in responses_record["request"]["input"]))
        self.assertTrue(any(item.get("type") == "function_call" for item in responses_record["request"]["input"]))
        self.assertTrue(responses_record["output_messages"])
        self.assertEqual(responses_record["request_id"], responses_trace.request_id)
        self.assertTrue(any(item["type"] == "reasoning" for item in responses_record["output_messages"][0]["content"]))
        self.assertTrue(responses_record["input_text"])
        self._assert_trace_record_matches_capture(
            responses_trace,
            responses_record,
            expected_request_field="input",
            expected_request_role=None,
            expected_response_field="output",
        )
        self.assertEqual(responses_trace.request_snapshot["input"][2]["type"], "function_call")
        self.assertEqual(responses_trace.response_snapshot["status"], "completed")

    def test_gateway_runtime_endpoints_with_real_rollout_controller(self):
        self.controller = self._build_controller()
        app = build_local_gateway_app(
            self.controller,
            config=GatewayConfig(port=8080, capture_path=str(self.capture_output_path)),
        )

        with TestClient(app) as client:
            livez_response = client.get("/livez")
            self.assertEqual(livez_response.status_code, 200, livez_response.text)
            self.assertEqual(livez_response.json(), {"status": "ok"})

            readyz_response = client.get("/readyz")
            self.assertEqual(readyz_response.status_code, 200, readyz_response.text)
            readyz_body = readyz_response.json()
            self.assertTrue(readyz_body["ready"])
            self.assertEqual(readyz_body["status"], "ready")
            self.assertIsInstance(readyz_body["details"], dict)

            capabilities_response = client.get("/capabilities")
            self.assertEqual(capabilities_response.status_code, 200, capabilities_response.text)
            capabilities_body = capabilities_response.json()
            self.assertEqual(capabilities_body["model"], self.controller.config.model_name)
            self.assertEqual(capabilities_body["backend"], self.controller.config.rollout_backend)
            self.assertEqual(capabilities_body["context_length"], self.controller.config.context_length)
            self.assertTrue(capabilities_body["supports_stream"])
            self.assertTrue(capabilities_body["supports_tools"])
            self.assertFalse(capabilities_body["supports_cancel"])
            self.assertTrue(capabilities_body["supports_parallel_tool_calls"])
            self.assertTrue(capabilities_body["supports_reasoning"])

    def test_gateway_routes_with_real_rollout_controller_capture_ir_fallback_behavior(self):
        self.controller = self._build_controller()
        app = build_local_gateway_app(
            self.controller,
            config=GatewayConfig(port=8080, capture_path=str(self.capture_output_path)),
        )

        openai_payload = {
            "model": self.controller.config.model_name,
            "messages": [
                {"role": "user", "content": "Call the search tool if you need it."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bad_openai",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "not-json",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bad_openai",
                    "content": "Sunny, 26C",
                },
                {"role": "user", "content": "Finish the answer in one sentence. DONE"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the latest weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"},
            },
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.4,
            "stop": ["DONE"],
            "max_tokens": 32,
        }
        openai_invalid_n_payload = {
            **openai_payload,
            "n": 2,
        }
        anthropic_payload = {
            "model": self.controller.config.model_name,
            "system": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 32,
        }
        responses_payload = {
            "model": self.controller.config.model_name,
            "instructions": "Follow the system rule.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Use concise answers."}],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Need private reasoning first."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_bad_responses",
                    "name": "search",
                    "arguments": "not-json",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_bad_responses",
                    "output": [{"type": "text", "text": "Sunny, 26C"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Answer now."}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search the latest weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
                {
                    "type": "web_search_preview",
                    "name": "web_search_preview",
                },
            ],
            "tool_choice": {"type": "function", "name": "search"},
            "parallel_tool_calls": True,
            "store": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "high"},
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 32,
        }
        responses_invalid_content_payload = {
            **responses_payload,
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "Use concise answers."},
                        {"type": "image", "image_url": "https://example.com/ignored.png"},
                    ],
                }
            ],
        }
        responses_stream_payload = {
            **responses_payload,
            "stream": True,
        }
        anthropic_stream_payload = {
            "model": self.controller.config.model_name,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Say hello briefly."}]},
            ],
            "max_tokens": 32,
            "stream": True,
        }
        openai_stream_payload = {
            **openai_payload,
            "stream": True,
        }

        with TestClient(app) as client:
            openai_response = client.post("/v1/chat/completions", json=openai_payload)
            self.assertEqual(openai_response.status_code, 200, openai_response.text)

            openai_invalid_n_response = client.post("/v1/chat/completions", json=openai_invalid_n_payload)
            self.assertEqual(openai_invalid_n_response.status_code, 400, openai_invalid_n_response.text)
            openai_invalid_n_body = openai_invalid_n_response.json()
            self.assertEqual(openai_invalid_n_body["error"]["type"], "invalid_request_error")
            self.assertEqual(openai_invalid_n_body["error"]["code"], "n_not_supported")

            anthropic_response = client.post("/v1/messages", json=anthropic_payload)
            self.assertEqual(anthropic_response.status_code, 400, anthropic_response.text)
            anthropic_error_body = anthropic_response.json()
            self.assertEqual(anthropic_error_body["type"], "error")
            self.assertEqual(anthropic_error_body["error"]["type"], "invalid_request_error")
            self.assertIn("Unsupported Anthropic content block type(s) in system: image", anthropic_error_body["error"]["message"])

            openai_stream_response = client.post("/v1/chat/completions", json=openai_stream_payload)
            self.assertEqual(openai_stream_response.status_code, 200, openai_stream_response.text)
            self.assertIn("text/event-stream", openai_stream_response.headers["content-type"])
            self.assertIn("data: [DONE]", openai_stream_response.text)

            anthropic_stream_response = client.post("/v1/messages", json=anthropic_stream_payload)
            self.assertEqual(anthropic_stream_response.status_code, 200, anthropic_stream_response.text)
            self.assertIn("text/event-stream", anthropic_stream_response.headers["content-type"])
            self.assertIn("event: message_start", anthropic_stream_response.text)
            self.assertIn("event: message_stop", anthropic_stream_response.text)

            responses_response = client.post("/v1/responses", json=responses_payload)
            self.assertEqual(responses_response.status_code, 200, responses_response.text)

            responses_invalid_content_response = client.post("/v1/responses", json=responses_invalid_content_payload)
            self.assertEqual(responses_invalid_content_response.status_code, 400, responses_invalid_content_response.text)
            responses_invalid_content_body = responses_invalid_content_response.json()
            self.assertEqual(responses_invalid_content_body["error"]["type"], "invalid_request_error")
            self.assertEqual(responses_invalid_content_body["error"]["code"], "unsupported_content_block")

            responses_stream_response = client.post("/v1/responses", json=responses_stream_payload)
            self.assertEqual(responses_stream_response.status_code, 200, responses_stream_response.text)
            self.assertIn("text/event-stream", responses_stream_response.headers["content-type"])
            self.assertIn("event: response.created", responses_stream_response.text)
            self.assertIn("event: response.completed", responses_stream_response.text)

        capture_records = self._read_new_capture_records()
        protocol_records = self._capture_records_by_protocol(capture_records)
        self.assertIn("OpenAIChatAdapter", protocol_records)
        self.assertIn("OpenAIResponsesAdapter", protocol_records)
        self.assertNotIn("AnthropicChatAdapter", protocol_records)

        openai_record = protocol_records["OpenAIChatAdapter"]
        self.assertEqual(openai_record["rollout_tool_choice"], {"type": "function", "function": {"name": "search"}})
        self.assertEqual(len(openai_record["rollout_tools"]), 1)
        self.assertEqual(openai_record["rollout_tools"][0]["function"]["name"], "search")
        self.assertEqual(openai_record["rollout_sample_params"]["presence_penalty"], 0.6)
        self.assertEqual(openai_record["rollout_sample_params"]["frequency_penalty"], 0.4)
        self.assertEqual(openai_record["rollout_sample_params"]["temperature"], 0.2)
        self.assertEqual(openai_record["rollout_sample_params"]["top_p"], 0.9)
        self.assertEqual(openai_record["rollout_sample_params"]["stops"], ["DONE"])
        self.assertEqual(
            openai_record["internal_messages"][1]["tool_calls"][0]["function"]["arguments"],
            "not-json",
        )

        responses_record = protocol_records["OpenAIResponsesAdapter"]
        self.assertEqual(responses_record["rollout_tool_choice"], {"type": "function", "function": {"name": "search"}})
        self.assertEqual(len(responses_record["rollout_tools"]), 1)
        self.assertEqual(responses_record["rollout_tools"][0]["function"]["name"], "search")
        self.assertTrue(responses_record["rollout_sample_params"]["max_tokens"] <= 32)
        self.assertEqual(responses_record["rollout_sample_params"]["temperature"], 0.1)
        self.assertEqual(responses_record["rollout_sample_params"]["top_p"], 0.8)
        self.assertNotIn("store", responses_record["rollout_sample_params"])
        self.assertNotIn("include", responses_record["rollout_sample_params"])
        self.assertEqual(responses_record["internal_messages"][0]["role"], "system")
        self.assertEqual(responses_record["internal_messages"][0]["content"], "Follow the system rule.")
        self.assertEqual(responses_record["internal_messages"][1]["role"], "system")
        self.assertEqual(responses_record["internal_messages"][1]["content"], "Use concise answers.")
        self.assertEqual(
            responses_record["internal_messages"][3]["tool_calls"][0]["function"]["arguments"],
            "not-json",
        )

    def test_gateway_routes_with_real_rollout_controller_return_context_length_error_on_context_overflow(self):
        self.controller = self._build_controller()
        app = build_local_gateway_app(
            self.controller,
            config=GatewayConfig(port=8080, capture_path=str(self.capture_output_path)),
        )
        overflow_prompt = "Beijing weather " * max(self.controller.config.context_length, 1)
        openai_payload = {
            "model": self.controller.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": overflow_prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
        }

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/chat/completions", json=openai_payload)

        self.assertEqual(response.status_code, 400, response.text)
        response_body = response.json()
        self.assertEqual(response_body["error"]["type"], "context_length_exceeded")
        self.assertEqual(response_body["error"]["code"], "context_too_long")
        self.assertIn("Input is too long", response_body["error"]["message"])
        capture_records = self._read_new_capture_records()
        self.assertFalse(capture_records)


if __name__ == "__main__":
    unittest.main()
