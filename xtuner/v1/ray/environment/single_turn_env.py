import asyncio
from typing import List, Union

import ray

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.ray.environment.base_env import BaseEnvironment


@ray.remote
class SingleTurnEnvironment(BaseEnvironment):
    """A single-turn environment for handling generation and evaluation tasks.

    This class extends `BaseEnvironment` to provide a concrete implementation for
    single-turn interactions. It manages the rollout process for generating responses
    and can coordinate with a judger for evaluation.

    Args:
        environment (str): The name of the environment.
        placement_group: The placement group for scheduling Ray actors.
        rollout_cfg (optional): Configuration for the rollout controller. Defaults to None.
        judger_cfg (optional): Configuration for the judger controller. Defaults to None.
    """

    def __init__(self, environment: str, placement_group, rollout_cfg=None, judger_cfg=None):
        super().__init__(environment, placement_group, rollout_cfg, judger_cfg)

    async def generate(
        self, data: Union[list[str], RLTextDataItem, List[RLTextDataItem]], sample_params: None
    ) -> Union[list[str], RLTextDataItem, List[RLTextDataItem]]:
        """Generates responses for the given data using the rollout controller.

        This method takes input data, which can be a single prompt, a list of prompts,
        or `RLTextDataItem` objects, and uses the rollout controller to generate
        responses. The generated responses and their states are then added to the
        input data items.

        Args:
            data: The input data for generation. Can be a list of strings,
                a single `RLTextDataItem`, or a list of `RLTextDataItem`.
            sample_params: Sampling parameters for the generation process.

        Returns:
            The data enriched with the generated responses. The format of the return
            value matches the format of the input `data`.
        """
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
        """Runs a full generation and judger cycle.

        This method first generates responses using the `generate` method and then,
        if a judger controller is available, it uses the judger to evaluate the
        generated responses.

        Args:
            data: The input data for the cycle. Can be a list of strings,
                a single `RLTextDataItem`, or a list of `RLTextDataItem`.
            sample_params: Sampling parameters for the generation process.

        Returns:
            The data enriched with generated responses and evaluation results.
            The format of the return value matches the format of the input `data`.
        """
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
