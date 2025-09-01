import asyncio
from typing import List, Union

import ray

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.ray.environment.base_env import BaseEnvironment


@ray.remote
class SingleTurnEnvironment(BaseEnvironment):
    def __init__(self, environment: str, placement_group, rollout_cfg=None, judger_cfg=None):
        super().__init__(environment, placement_group, rollout_cfg, judger_cfg)

    async def generate(
        self, data: Union[list[str], RLTextDataItem, List[RLTextDataItem]], sample_params: None
    ) -> Union[list[str], RLTextDataItem, List[RLTextDataItem]]:
        if not isinstance(data, list):
            group_samples = [data]
        elif len(data) == 0:
            return []
        elif isinstance(data[0], str):
            group_samples = [RLTextDataItem(messages=data)]  # type: ignore[typeddict-item]
        else:
            group_samples = data  # type: ignore[assignment]

        if self.rollout_controller:
            response_future = [
                self.rollout_controller.rollout.remote(prompt=sample["messages"], sample_params=sample_params)
                for sample in group_samples
            ]
            response = await asyncio.gather(*response_future)
            for i in range(len(group_samples)):
                group_samples[i]["response_str"] = response[i][0]
                group_samples[i]["state"] = response[i][1]
        if isinstance(data, str):
            return group_samples[0]["response_str"] or ""
        elif not isinstance(data, list):
            return group_samples[0]
        else:
            return group_samples

    async def run(
        self, data: Union[list[str], RLTextDataItem, List[RLTextDataItem]], sample_params: None
    ) -> Union[list[str], RLTextDataItem, List[RLTextDataItem]]:
        if not isinstance(data, list):
            group_samples = [data]
        elif not isinstance(data[0], str):
            group_samples = data  # type: ignore[assignment]
        else:
            return await self.generate(data, sample_params)

        group_samples = await self.generate(group_samples, sample_params)  # type: ignore[assignment]
        if self.judger_controller:
            group_samples = await self.judger_controller.run.remote(group_samples)
        if not isinstance(data, list) and not isinstance(data, str):
            return group_samples[0]
        return group_samples
