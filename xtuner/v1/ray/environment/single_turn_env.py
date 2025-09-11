import asyncio
from typing import List

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

    async def generate(self, group_samples: List[RLTextDataItem], sample_params: None) -> List[RLTextDataItem]:
        """Generate responses for a batch of RLTextDataItem using the rollout
        controller.

        Each item in `group_samples` will be sent to the rollout controller for response generation
        with the provided sampling parameters. The generated response string and state will be
        added to each RLTextDataItem in-place as `response_str` and `state` fields.

        Args:
            group_samples (List[RLTextDataItem]):
                A list of RLTextDataItem objects containing the prompts/messages for generation.
            sample_params: Sampling parameters for the generation process. The type should match
                the rollout controller's expected sampling parameter type (e.g., SampleParams or dict).

        Returns:
            List[RLTextDataItem]:
                The same list of RLTextDataItem, with each item enriched with the generated response
                and state from the rollout controller.
        """
        if self.rollout_controller:
            response_future = [
                self.rollout_controller.rollout.remote(prompt=sample["messages"], sample_params=sample_params)
                for sample in group_samples
            ]
            response = await asyncio.gather(*response_future)
            for i in range(len(group_samples)):
                group_samples[i]["response_str"] = response[i].response
                group_samples[i]["state"] = response[i].finish_reason

        return group_samples

    async def run(self, group_samples: List[RLTextDataItem], sample_params: None) -> List[RLTextDataItem]:
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
        group_samples = await self.generate(group_samples, sample_params)  # type: ignore[assignment]
        if self.judger_controller:
            group_samples = await self.judger_controller.run.remote(group_samples)
        return group_samples
