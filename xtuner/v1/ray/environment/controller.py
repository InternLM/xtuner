import json
from typing import List

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.ray.judger.controller import JudgerController
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.utils import get_logger


logger = get_logger()


class SampleParams(BaseModel):
    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 50
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 0.95
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 0.6
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 2
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 2048
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True


@ray.remote
class EnvController:
    def __init__(self, environment: str, rollout_controller: RolloutController, judger_controller: JudgerController):
        self.environment = environment
        self.judger_controller = judger_controller
        self.rollout_controller = rollout_controller
        self.received_samples = 0
        self.finished_samples = 0

    def pause(self):
        return ray.get(self.rollout_controller.pause.remote())

    def shutdown(self):
        return ray.get(self.rollout_controller.shutdown.remote())

    async def rollout(self, prompt):
        return await self.rollout_controller.rollout.remote(prompt)

    async def run(self, rollout_input, reward_input, sample_params: SampleParams = SampleParams()) -> str:
        self.received_samples += 1
        logger.debug(f"env controller received_samples: {self.received_samples}")
        rollout_res, state = await self.rollout_controller.rollout.remote(rollout_input, sample_params.dict())  # type: ignore[attr-defined]
        if state == "unfinished":
            return json.dumps({"response": rollout_res, "state": state})
        reward = await self.judger_controller.judge.remote(rollout_res, reward_input)  # type: ignore[attr-defined]
        self.finished_samples += 1
        logger.debug(f"env controller finished_samples: {self.finished_samples}")
        return json.dumps({"response": rollout_res, "reward": reward, "state": state})

    def get_rollout_info(self):
        return ray.get(self.rollout_controller.get_rollout_info.remote())

    def onload(self, *args, **kwargs):
        return ray.get(self.rollout_controller.onload.remote(*args, **kwargs))

    def offload(self):
        return ray.get(self.rollout_controller.offload.remote())

    def agent(self):
        # todo: add agent logic here
        pass
