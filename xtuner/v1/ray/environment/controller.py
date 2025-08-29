import asyncio
from typing import List, Optional, Union

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers
from xtuner.v1.ray.judger.controller import JudgerController
from xtuner.v1.ray.rollout.controller import RolloutController


class SampleParams(BaseModel):
    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 0
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 1.0
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 1.0
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 0
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 2048
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True
    do_sample: Annotated[bool, Parameter(help="Whether to sample or not.")] = True


@ray.remote
class EnvController:
    def __init__(self, environment: str, placement_group, rollout_cfg=None, judger_cfg=None, sample_params=None):
        self.environment = environment
        self.sample_params = sample_params
        self.init_rollout_controller(placement_group, rollout_cfg)
        self.init_judger_controller(placement_group, judger_cfg)

    def init_rollout_controller(self, placement_group, rollout_cfg):
        if rollout_cfg is None:
            self.rollout_controller = None
            return
        if rollout_cfg.backend == "lmdeploy":
            from xtuner.v1.ray.rollout import LMDeployWorker

            rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
                LMDeployWorker, rollout_cfg, placement_group
            )
            self.rollout_controller = RolloutController.remote(rollout_cfg, rollout_workers_map)
        elif rollout_cfg.backend == "vllm":
            from xtuner.v1.ray.rollout import vLLMWorker

            rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(vLLMWorker, rollout_cfg, placement_group)
            self.rollout_controller = RolloutController.remote(rollout_cfg, rollout_workers_map)
        else:
            raise NotImplementedError(f"Rollout backend '{rollout_cfg.backend}' is not supported.")

    def init_judger_controller(self, placement_group, judger_cfg):
        if judger_cfg is None:
            self.judger_controller = None
            return
        self.judger_controller = JudgerController.remote(judger_cfg)

    async def run(
        self, data: Union[str, RLTextDataItem, List[RLTextDataItem]], flow_sample_params: Optional[SampleParams] = None
    ) -> Union[str, RLTextDataItem, List[RLTextDataItem]]:
        if isinstance(data, str):
            group_samples = [RLTextDataItem(prompt_str=data)]
        elif not isinstance(data, list):
            group_samples = [data]
        else:
            group_samples = data

        if self.rollout_controller:
            # 优先使用dataflow的sample_params
            sample_params = flow_sample_params if flow_sample_params is not None else self.sample_params

            response_future = [
                self.rollout_controller.rollout.remote(sample["prompt_str"], sample_params.dict())
                for sample in group_samples
            ]
            response = await asyncio.gather(*response_future)
            for i in range(len(group_samples)):
                group_samples[i]["response_str"] = response[i][0]
                group_samples[i]["state"] = response[i][1]

        if self.judger_controller:
            rewards = await self.judger_controller.run.remote(group_samples)
            assert len(rewards) == len(group_samples)
            for i in range(len(group_samples)):
                group_samples[i]["reward"] = rewards[i]

        if isinstance(data, str):
            return group_samples[0]["response_str"] or ""
        elif not isinstance(data, list):
            return group_samples[0]
        else:
            return group_samples

    def pause(self):
        return ray.get(self.rollout_controller.pause.remote())

    def shutdown(self):
        if self.rollout_controller:
            return ray.get(self.rollout_controller.shutdown.remote())

    def restart(self):
        return ray.get(self.rollout_controller.restart.remote())

    async def rollout(self, prompt):
        return await self.rollout_controller.rollout.remote(prompt)

    def get_rollout_info(self):
        return ray.get(self.rollout_controller.get_rollout_info.remote())

    def onload(self, *args, **kwargs):
        return ray.get(self.rollout_controller.onload.remote(*args, **kwargs))

    def offload(self, *args, **kwargs):
        return ray.get(self.rollout_controller.offload.remote(*args, **kwargs))
