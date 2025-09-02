import multiprocessing
import time
from typing import Any, Dict

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from xtuner.v1.ray.judger.gsm8k import compute_reward
from xtuner.v1.ray.judger.native import NativeJudger


app = FastAPI()


class JudgeRequest(BaseModel):
    response: str
    label: str
    extra_info: Dict[str, Any] = Field(default_factory=dict)


class JudgeResponse(BaseModel):
    reward: float


@app.post("/judge", response_model=JudgeResponse)
async def judge(request: JudgeRequest):
    """Endpoint to compute reward for a given response and label."""
    # The compute_reward function returns a float, we wrap it in a dict
    # to match what the client-side post-processing function might expect.
    reward_value = compute_reward(request.response, request.label, request.extra_info)
    return {"reward": reward_value}


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
    return result["reward"]


class GSM8KRemoteJudgerConfig(BaseModel):
    remote_url: str
    extra_info: dict = {"score": 1, "format_score": 0}

    def build(self):
        return NativeJudger(
            remote_url=self.remote_url, postprocess_func=custom_postprocessor_for_gsm8k, extra_info=self.extra_info
        )
