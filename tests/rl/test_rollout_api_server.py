import os
import subprocess
import tempfile
import time
import unittest

import httpx
import ray
import torch
from transformers import AutoTokenizer

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers


TEST_TEXT_MESSAGES = [{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
MOE_MODEL_PATH = os.environ.get("QWEN3_MOE_PATH") or os.environ["QWEN30B_MODEL_PATH"]
RESOURCE_MAP = {
    "npu": "NPU",
    "cuda": "GPU",
}


@unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
class TestRolloutAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]

    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.max_prompt_length = 512
        self.max_response_length = 1024
        self.context_length = self.max_prompt_length + self.max_response_length
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()

    def tearDown(self):
        ray.shutdown()
        self._cleanup_lmdeploy_ray_worker_wrapper()
        self.temp_dir.cleanup()

    def _cleanup_lmdeploy_ray_worker_wrapper(self):
        try:
            result = subprocess.run(
                ["pkill", "-f", "ray::RayWorkerWrapper*"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                print(
                    f"pkill command failed with return code {result.returncode}: {result.stderr}."
                    " Maybe no lmdeploy ray::RayWorkerWrapper processes found."
                )
        except Exception as exc:
            print(f"Error stopping ray::RayWorkerWrapper cluster: {exc}")

    def _wait_until_ready(self, base_url: str):
        deadline = time.time() + 1800
        last_error = None
        while time.time() < deadline:
            try:
                response = httpx.get(f"{base_url}/healthz", timeout=10.0)
                if response.status_code == 200:
                    return
                last_error = f"healthz returned {response.status_code}: {response.text}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(5)
        raise RuntimeError(f"API server at {base_url} did not become ready in time: {last_error}")

    def test_dense_model(self):
        resource_config = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=16,
            cpu_memory_per_worker=8 * 1024**3,
        )
        pg = AutoAcceleratorWorkers.build_placement_group(resource_config, name="dense_api_pg")
        dense_worker_log_dir = os.path.join(self.worker_log_dir, "dense")
        rollout_config = RolloutConfig(
            env="test_rollout_api_server_dense",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=4,
            expert_parallel_size=1,
            context_length=self.context_length,
            worker_log_dir=dense_worker_log_dir,
            dist_port_base=38000,
            api_host="127.0.0.1",
            api_port=28000,
        )
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        try:
            metadata = ray.get(rollout_controller.get_rollout_metadata.remote(), timeout=1800)
            base_url = metadata["api_server_url"]
            self._wait_until_ready(base_url)

            text_prompt = self.tokenizer.apply_chat_template(
                TEST_TEXT_MESSAGES,
                tokenize=False,
                add_generation_prompt=True,
            )
            test_input_ids = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]

            with httpx.Client(timeout=300.0) as client:
                generate = client.post(
                    f"{base_url}/generate",
                    json={
                        "message": TEST_TEXT_MESSAGES,
                        "tokens": test_input_ids,
                        "sample_params": {
                            "return_token_ids": True,
                            "temperature": 0.0,
                            "top_k": 1,
                            "max_tokens": 16,
                        },
                    },
                )
                self.assertEqual(generate.status_code, 200, generate.text)
                generate_body = generate.json()
                self.assertEqual(generate_body["status"], "completed")
                self.assertIn(generate_body["finish_reason"], {"stop", "length"})
                self.assertTrue(generate_body["extra_fields"]["request_id"])
                self.assertGreater(len(generate_body["response_ids"]), 0)
                self.assertIsInstance(generate_body["response"], str)

                chat = client.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": rollout_config.model_name,
                        "messages": TEST_TEXT_MESSAGES,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 16,
                    },
                )
                self.assertEqual(chat.status_code, 200, chat.text)
                chat_body = chat.json()
                print("chat_body: ", chat_body)
                self.assertEqual(chat_body["object"], "chat.completion")
                self.assertEqual(chat_body["model"], rollout_config.model_name)
                self.assertTrue(chat_body["id"].startswith("chatcmpl-"))
                self.assertEqual(chat_body["choices"][0]["message"]["role"], "assistant")
                self.assertTrue(chat_body["choices"][0]["message"]["content"])
                self.assertIn(chat_body["choices"][0]["finish_reason"], {"stop", "length"})
                self.assertGreater(chat_body["usage"]["prompt_tokens"], 0)
                self.assertGreater(chat_body["usage"]["total_tokens"], chat_body["usage"]["completion_tokens"])

                anthropic = client.post(
                    f"{base_url}/v1/messages",
                    json={
                        "model": rollout_config.model_name,
                        "system": "You are helpful.",
                        "messages": TEST_TEXT_MESSAGES,
                        "max_tokens": 16,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                )
                self.assertEqual(anthropic.status_code, 200, anthropic.text)
                anthropic_body = anthropic.json()
                self.assertEqual(anthropic_body["type"], "message")
                self.assertEqual(anthropic_body["role"], "assistant")
                self.assertEqual(anthropic_body["model"], rollout_config.model_name)
                self.assertTrue(anthropic_body["id"].startswith("msg_"))
                self.assertTrue(anthropic_body["content"][0]["text"])
                self.assertIn(anthropic_body["stop_reason"], {"stop", "length"})
                self.assertGreater(anthropic_body["usage"]["input_tokens"], 0)
                self.assertGreaterEqual(anthropic_body["usage"]["output_tokens"], 1)

                invalid_block = client.post(
                    f"{base_url}/v1/messages",
                    json={
                        "model": rollout_config.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "look"},
                                    {"type": "image", "text": ""},
                                ],
                            }
                        ],
                        "max_tokens": 8,
                    },
                    timeout=30.0,
                )
                self.assertEqual(invalid_block.status_code, 400, invalid_block.text)
                self.assertEqual(invalid_block.json()["type"], "error")
                self.assertEqual(invalid_block.json()["error"]["type"], "invalid_request_error")

                health = client.get(f"{base_url}/healthz", timeout=30.0)
                meta = client.get(f"{base_url}/metadata", timeout=30.0)
                self.assertEqual(health.status_code, 200, health.text)
                self.assertEqual(health.json()["status"], "ok")
                self.assertGreaterEqual(health.json()["active_workers"], 1)
                self.assertEqual(meta.status_code, 200, meta.text)
                self.assertEqual(meta.json()["api_server_url"], base_url)
                self.assertEqual(metadata["api_server_url"], base_url)
                self.assertEqual(meta.json()["api_server_url"].rsplit(":", 1)[-1], str(rollout_config.api_port))
                self.assertTrue(all(meta.json()["worker_server_urls_status"].values()))

                offload = client.post(f"{base_url}/offload", timeout=120.0)
                self.assertEqual(offload.status_code, 200, offload.text)
                self.assertEqual(offload.json()["action"], "offload")

                onload = client.post(f"{base_url}/onload", timeout=120.0)
                self.assertEqual(onload.status_code, 200, onload.text)
                self.assertEqual(onload.json()["action"], "onload")

                regenerated = client.post(
                    f"{base_url}/generate",
                    json={
                        "message": TEST_TEXT_MESSAGES,
                        "sample_params": {
                            "return_token_ids": True,
                            "temperature": 0.0,
                            "top_k": 1,
                            "max_tokens": 8,
                        },
                    },
                )
                self.assertEqual(regenerated.status_code, 200, regenerated.text)
                self.assertEqual(regenerated.json()["status"], "completed")
        finally:
            try:
                ray.get(rollout_controller.shutdown.remote(), timeout=300)
            finally:
                ray.util.remove_placement_group(pg)

    def test_moe_model(self):
        resource_config = AcceleratorResourcesConfig(
            accelerator=RESOURCE_MAP[torch.accelerator.current_accelerator().type],
            num_workers=4,
            num_cpus_per_worker=16,
            cpu_memory_per_worker=8 * 1024**3,
        )
        pg = AutoAcceleratorWorkers.build_placement_group(resource_config, name="moe_api_pg")
        moe_worker_log_dir = os.path.join(self.worker_log_dir, "moe")
        rollout_config = RolloutConfig(
            env="test_rollout_api_server_moe",
            model_path=MOE_MODEL_PATH,
            model_name=os.path.basename(MOE_MODEL_PATH).lower(),
            tokenizer_path=MOE_MODEL_PATH,
            tensor_parallel_size=1,
            expert_parallel_size=4,
            context_length=self.context_length,
            worker_log_dir=moe_worker_log_dir,
            dist_port_base=38000 + 1024 * 4,
            api_host="127.0.0.1",
            api_port=29000,
            enable_return_routed_experts=True,
        )
        rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
        try:
            metadata = ray.get(rollout_controller.get_rollout_metadata.remote(), timeout=1800)
            base_url = metadata["api_server_url"]
            self._wait_until_ready(base_url)

            request = RolloutState(
                message=[{"role": "user", "content": "Briefly explain what mixture of experts means."}],
                sample_params=SampleParams(
                    return_token_ids=True,
                    return_logprob=False,
                    temperature=0.0,
                    top_k=1,
                    max_tokens=32,
                ),
            )
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{base_url}/generate",
                    json=request.model_dump(mode="json"),
                )
                meta = client.get(f"{base_url}/metadata", timeout=30.0)

            self.assertEqual(response.status_code, 200, response.text)
            rollout_state = RolloutState.model_validate_json(response.text)
            self.assertIsNotNone(rollout_state.routed_experts)
            self.assertEqual(meta.status_code, 200, meta.text)
            self.assertEqual(meta.json()["api_server_url"], base_url)
            self.assertEqual(metadata["api_server_url"], base_url)
            self.assertEqual(meta.json()["api_server_url"].rsplit(":", 1)[-1], str(rollout_config.api_port))
        finally:
            try:
                ray.get(rollout_controller.shutdown.remote(), timeout=300)
            finally:
                ray.util.remove_placement_group(pg)

if __name__ == "__main__":
    unittest.main()
