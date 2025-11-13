import json
import multiprocessing
import time
from typing import Any, Dict, List

import httpx
import ray
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.ray.judger.native import NativeJudger
from xtuner.v1.ray.rollout import LMDeployWorker

from .httpx_utils import HttpRequestErrorType, HttpRequestResult


app = FastAPI()


def get_eos_token(model_path: str) -> int | List[int]:
    import os

    from xtuner.v1.utils.logger import get_logger

    logger = get_logger()
    generation_config_path = os.path.join(model_path, "generation_config.json")
    if not os.path.exists(generation_config_path):
        logger.warning(
            f"Config {generation_config_path} does not exist and thus cannot get eos_token. You must provide eos_token manually."
        )
        return []
    with open(generation_config_path) as f:
        generation_config = json.load(f)
    eos_token_id = generation_config.get("eos_token_id")
    return eos_token_id


@ray.remote
class MockTimeoutRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            raise httpx.TimeoutException("Mocked timeout error")
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked exception: {e.__class__.__name__}")
            return result

    def launch_server(self):
        pass  # Override


@ray.remote
class MockRequestErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            raise httpx.RequestError("Mocked httpx request error")
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked exception: {e.__class__.__name__}")
            return result

    def launch_server(self):
        pass  # Override


@ray.remote
class MockClientErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            req = httpx.Request("POST", url)
            res = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("Mocked client error", request=req, response=res)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked exception: {e.__class__.__name__}")
            return result

    def launch_server(self):
        pass  # Override


@ray.remote
class MockServerErrorRolloutWorker(LMDeployWorker):
    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            req = httpx.Request("POST", url)
            res = httpx.Response(500, request=req)
            raise httpx.HTTPStatusError("Mocked server error", request=req, response=res)
        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            self.logger.info(f"Caught mocked exception: {e.__class__.__name__}")
            return result

    def launch_server(self):
        pass  # Override


class JudgeRequest(BaseModel):
    response: str
    label: str
    extra_info: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float


@app.post("/judge", response_model=JudgeResponse)
async def judge(request: JudgeRequest):
    from xtuner.v1.ray.judger.gsm8k import compute_reward

    """Endpoint to compute reward for a given response and label."""
    # The compute_reward function returns a float, we wrap it in a dict
    # to match what the client-side post-processing function might expect.
    reward_value = compute_reward(request.response, request.label, request.extra_info)
    return reward_value


def run_server(host="127.0.0.1", port=8000):
    """Utility to run the server."""
    uvicorn.run(app, host=host, port=port)


class JudgerServer:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.process = None
        self.url = f"http://{self.host}:{self.port}/judge"

    def start(self):
        """Starts the server in a background process."""
        self.process = multiprocessing.Process(target=run_server, args=(self.host, self.port))
        self.process.start()
        # Wait for the server to be ready
        for _ in range(10):
            try:
                response = requests.get(f"http://{self.host}:{self.port}/docs")
                if response.status_code == 200:
                    print("Server started successfully.")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        raise RuntimeError("Server failed to start.")

    def stop(self):
        """Stops the server process."""
        if self.process:
            self.process.terminate()
            self.process.join()
            print("Server stopped.")


def custom_postprocessor_for_gsm8k(result):
    from xtuner.v1.data_proto.rl_data import RLJudgerResponseItem

    if not isinstance(result, list):
        result = [result]
    judger_response_item = [RLJudgerResponseItem(uid=result[i]["uid"], reward=result[i]) for i in range(len(result))]
    return judger_response_item


class GSM8KRemoteJudgerConfig(BaseModel):
    judger_name: str
    remote_url: str
    extra_info: dict = {"score": 1, "format_score": 0}
    model_config = ConfigDict(extra="forbid")

    def build(self):
        return NativeJudger(
            judger_name=self.judger_name,
            remote_url=self.remote_url,
            postprocess_func=custom_postprocessor_for_gsm8k,
            extra_info=self.extra_info,
        )
