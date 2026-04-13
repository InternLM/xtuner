import os
import sys
import tempfile
import time
import unittest
from urllib import error as urllib_error
from urllib import request as urllib_request

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
import torch

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import CamelAgentLoopConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
RESOURCE_MAP = {
    "npu": "NPU",
    "cuda": "GPU",
}


@unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
class TestCamelAgentLoop(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
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

    async def test_camel_single_turn_builds_chat_agent_with_official_async_openai(self):
        from openai import AsyncOpenAI

        rollout_controller, pg = self._build_controller(port=28601)
        try:
            metadata = await rollout_controller.get_rollout_metadata.remote()
            self._wait_until_ready(metadata["api_server_url"])

            cfg = CamelAgentLoopConfig(
                hf_checkpoint=MODEL_PATH,
                sample_params=SampleParams(max_tokens=64, temperature=0.0),
            )
            loop = cfg.build(rollout_controller=rollout_controller)
            state = RolloutState(
                message=[{"role": "user", "content": "Say hello in one short sentence."}],
                sample_params=SampleParams(max_tokens=64, temperature=0.0),
            )

            result = await loop.generate_sample(state)
            trace = await rollout_controller.get_openai_chat_trace_by_messages.remote(
                result.extra_fields["camel_request"],
                result.extra_fields["camel_response"],
                result.extra_fields["camel_finish_reason"],
            )

            print(f"CHAT_HISTORY_BEFORE: {result.extra_fields['camel_chat_history_before']}")
            print(f"CHAT_HISTORY_AFTER: {result.extra_fields['camel_chat_history_after']}")
            print(
                f"NEW_ENTRIES: "
                f"{result.extra_fields['camel_chat_history_after'][len(result.extra_fields['camel_chat_history_before']):]}"
            )
            print(f"EXTRACTED_REQUEST: {result.extra_fields['camel_request']}")
            print(f"EXTRACTED_RESPONSE: {result.extra_fields['camel_response']}")
            print(f"EXTRACTED_FINISH_REASON: {result.extra_fields['camel_finish_reason']}")
            print(f"TRACE_LOOKUP: {trace}")

            self.assertEqual(result.status, Status.COMPLETED)
            self.assertIsNotNone(trace)
            self.assertEqual(result.prompt_ids, trace["prompt_ids"])
            self.assertEqual(result.response_ids, trace["response_ids"])
            self.assertEqual(result.finish_reason, trace["finish_reason"])
            self.assertEqual(result.response, loop.tokenizer.decode(trace["response_ids"]))
        finally:
            try:
                await rollout_controller.shutdown.remote()
            finally:
                ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    unittest.main()
