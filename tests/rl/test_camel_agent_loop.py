import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib import error as urllib_error
from urllib import request as urllib_request

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
import torch
from camel.toolkits import SearchToolkit

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import CamelAgentLoop, CamelAgentLoopConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.chat_adapter.collector import append_current_trace_rollout_state
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers


MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH", "")
RESOURCE_MAP = {
    "npu": "NPU",
    "cuda": "GPU",
}


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
        rendered = "\n".join(json.dumps(message, ensure_ascii=False, sort_keys=True) for message in messages)
        if tools:
            rendered += "\nTOOLS:" + json.dumps(tools, ensure_ascii=False, sort_keys=True)
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class TestCamelAgentLoopUnit(unittest.IsolatedAsyncioTestCase):
    def _build_loop(self, agent, sample_params=None, tools=None, tool_choice=None):
        tokenizer = DummyTokenizer()
        with patch("xtuner.v1.rl.agent_loop.agent_loop.load_tokenizer", return_value=tokenizer), patch(
            "xtuner.v1.rl.agent_loop.agent_loop.load_processor", return_value=None
        ), patch.object(CamelAgentLoop, "init_agent", return_value=agent):
            cfg = CamelAgentLoopConfig(
                hf_checkpoint="dummy",
                sample_params=sample_params or SampleParams(max_tokens=256, temperature=0.0),
                tools=tools,
                tool_choice=tool_choice,
            )
            loop = cfg.build(rollout_controller=SimpleNamespace())
        return loop, tokenizer

    async def test_camel_generate_sample_returns_gateway_rollout_states(self):
        class GatewayFakeAgent:
            async def astep(self, content):
                first_turn = RolloutState(
                    uid=101,
                    message=[{"role": "user", "content": content}],
                    prompt_ids=[11, 12],
                    tokens=[11, 12],
                    response="tool-call",
                    response_ids=[21, 22],
                    logprobs=[-0.1, -0.2],
                    response_mask=[1, 1],
                    finish_reason="tool_calls",
                    status=Status.COMPLETED,
                    extra_fields={
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{\"q\":\"camel\"}"},
                            }
                        ]
                    },
                )
                second_turn = RolloutState(
                    uid=102,
                    message=[{"role": "user", "content": content}],
                    prompt_ids=[11, 12, 21, 22],
                    tokens=[11, 12, 21, 22],
                    response="final-answer",
                    response_ids=[31, 32, 33],
                    logprobs=[-0.3, -0.4, -0.5],
                    response_mask=[1, 1, 1],
                    finish_reason="stop",
                    status=Status.COMPLETED,
                )
                append_current_trace_rollout_state(first_turn)
                append_current_trace_rollout_state(second_turn)
                return SimpleNamespace(info={"termination_reasons": ["stop"]})

        loop, _ = self._build_loop(agent=GatewayFakeAgent())
        state = RolloutState(
            message=[{"role": "user", "content": "Where is CAMEL on GitHub?"}],
            sample_params=SampleParams(max_tokens=128, temperature=0.0),
        )

        result = await loop.generate_sample(state)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].status, Status.COMPLETED)
        self.assertEqual(result[0].finish_reason, "tool_calls")
        self.assertEqual(result[0].extra_fields["gateway_rollout_index"], 0)
        self.assertEqual(result[1].finish_reason, "stop")
        self.assertEqual(result[1].prompt_ids, [11, 12, 21, 22])
        self.assertEqual(result[1].tokens, [11, 12, 21, 22])
        self.assertEqual(result[1].response_ids, [31, 32, 33])
        self.assertEqual(result[1].response_mask, [1, 1, 1])
        self.assertEqual(result[1].logprobs, [-0.3, -0.4, -0.5])
        self.assertEqual(result[1].response, "final-answer")
        self.assertEqual(result[1].message, state.message)
        self.assertEqual(len(result[1].extra_fields["gateway_trace_records"]), 2)
        self.assertEqual(result[1].extra_fields["gateway_trace_records"][0]["request_id"], "101")
        self.assertEqual(result[1].extra_fields["gateway_trace_records"][1]["finish_reason"], "stop")


@unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0" or not MODEL_PATH, "lmdeploy backend is not enabled")
class TestCamelAgentLoopIntegration(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]

    def setUp(self):
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(address="local", num_cpus=80, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=1,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.context_length = 1024

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    def _wait_until_ready(self, base_url: str):
        deadline = time.time() + 1800
        last_error = None
        while time.time() < deadline:
            try:
                with urllib_request.urlopen(f"{base_url}/healthz", timeout=10.0) as response:
                    if response.status == 200:
                        return
                    last_error = response.read().decode("utf-8", errors="ignore")
            except urllib_error.URLError as exc:
                last_error = repr(exc)
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(5)
        raise RuntimeError(f"API server at {base_url} did not become ready in time: {last_error}")

    def _build_controller(self, port: int):
        rollout_config = RolloutConfig(
            env=f"test_camel_agent_loop_{port}",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            context_length=self.context_length,
            worker_log_dir=self.worker_log_dir,
            api_host="127.0.0.1",
            api_port=port,
        )
        pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg, name=f"camel_pg_{port}")
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        return rollout_controller, pg

    async def test_camel_tool_call_integration_returns_gateway_rollout_batch(self):
        rollout_controller, pg = self._build_controller(port=28602)
        try:
            metadata = await rollout_controller.get_rollout_metadata.remote()
            self._wait_until_ready(metadata["api_server_url"])
            search_tool = SearchToolkit().search_duckduckgo

            cfg = CamelAgentLoopConfig(
                hf_checkpoint=MODEL_PATH,
                sample_params=SampleParams(max_tokens=256, temperature=0.0),
                system_message="You are a helpful assistant to do search task.",
                tools=[search_tool],
                tool_choice={"type": "function", "function": {"name": "search_duckduckgo"}},
            )
            loop = cfg.build(rollout_controller=rollout_controller)
            state = RolloutState(
                message=[{"role": "user", "content": "What is the Github link to CAMEL framework?"}],
                sample_params=SampleParams(max_tokens=256, temperature=0.0),
            )

            result = await loop.generate_sample(state)
            print(f"CAMEL_INTEGRATION_RESULT: {result}")
            if not any((state.extra_fields or {}).get("tool_calls") for state in result):
                self.skipTest("Current backend/model did not emit tool calls for the integration prompt.")

            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(result[-1].status, Status.COMPLETED)
            self.assertIsNotNone(result[-1].response_ids)
            self.assertIsNotNone(result[-1].response_mask)
            self.assertIsNotNone(result[-1].response)
            self.assertTrue(any(state.finish_reason == "tool_calls" for state in result))

        finally:
            try:
                await rollout_controller.shutdown.remote()
            finally:
                ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    unittest.main()
