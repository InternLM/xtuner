import asyncio
from typing import Dict, List

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.utils import get_logger

from .native import NativeJudger


class JudgerConfig(BaseModel):
    enable_batch_reward: Annotated[
        bool, Parameter(help="Whether to enable batch reward calculation for multiple samples at once.")
    ] = False
    reward_judger_configs: Annotated[
        Dict[str, BaseModel],
        Parameter(help="A custom Python function for computing reward given model output and label."),
    ] = {}


@ray.remote
class JudgerController:
    def __init__(self, judger_config: JudgerConfig, placement_group=None):
        self.judger_config = judger_config
        # note: placement_group is used to control the placement of Ray tasks.
        # It will be implemented when gpu judger is needed
        self.placement_group = placement_group
        self.reward_judger = {}
        for name, config in self.judger_config.reward_judger_configs.items():
            self.reward_judger[name] = config.build()
        self.logger = get_logger()

    async def _call_custom_reward_judger(
        self,
        active_judgers: Dict[str, NativeJudger],
        responses: List[str],
        labels: List[str],
    ) -> Dict[str, List[float]]:
        group_size = len(responses)
        if self.judger_config.enable_batch_reward:
            tasks = {name: judger.judge(responses, labels) for name, judger in active_judgers.items()}
            results = await asyncio.gather(*tasks.values())
            return dict(zip(tasks.keys(), [results] * group_size))

        else:
            tasks_per_sample = [
                [(name, judger.judge(responses[i], labels[i])) for name, judger in active_judgers.items()]
                for i in range(len(responses))
            ]

            flat_tasks_with_names = [task for sample_tasks in tasks_per_sample for task in sample_tasks]
            coroutines = [item[1] for item in flat_tasks_with_names]

            flat_results = await asyncio.gather(*coroutines)
            final_rewards: Dict[str, List[float]] = {
                name: [] for name in active_judgers
            }  # name: [sample1, sample2, ...]
            active_reward_size = len(active_judgers)
            for name_index in range(active_reward_size):
                reward_list = []
                for index in range(group_size):
                    reward_list.append(flat_results[index * active_reward_size + name_index])
                final_rewards[list(active_judgers.keys())[name_index]] = reward_list
            return final_rewards

    async def run(
        self, group_data_item: RLTextDataItem | List[RLTextDataItem]
    ) -> RLTextDataItem | List[RLTextDataItem]:
        if not group_data_item:
            return []
        input_list = True
        if not isinstance(group_data_item, List):
            group_data_item = [group_data_item]
            input_list = False
        batch_responses = [item["response_str"] or "" for item in group_data_item]
        batch_labels = [item["reward_model"]["ground_truth"] for item in group_data_item]
        data_source = group_data_item[0]["data_source"]
        assert data_source, "No data source found for the given datsetes"

        active_reward_judger = {name: func for name, func in self.reward_judger.items() if name in data_source}
        assert active_reward_judger, f"No active reward judger found for the given data source {data_source}."

        rewards_by_name = await self._call_custom_reward_judger(active_reward_judger, batch_responses, batch_labels)
        num_samples = len(group_data_item)
        final_rewards = [0.0] * num_samples

        for i in range(num_samples):
            for name, scores in rewards_by_name.items():
                weight = data_source.get(name, 1.0)
                final_rewards[i] += scores[i] * weight

        assert len(final_rewards) == num_samples
        for i, item in enumerate(group_data_item):
            item["reward"] = final_rewards[i]
        if not input_list:
            return group_data_item[0]
        return group_data_item
