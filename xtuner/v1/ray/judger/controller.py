import asyncio
from pathlib import Path
from typing import List

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLJudgerResponseItem
from xtuner.v1.utils import get_logger

from .native import NativeJudger


class JudgerConfig(BaseModel):
    """Judger configuration for XTuner.

    Configuration for the judging system managing batch processing and custom judger
    implementations for model evaluation and reward computation.

    Args:
        enable_batch_reward (bool): Enable calculate reward within the data group of repeat_prompt_k. Defaults to False.

        reward_judger_configs (Dict[str, BaseModel]): Dictionary mapping judger names
            to their configuration objects. We provided the example GSM8KJudgerConfig
            for GSM8K mathematical reasoning tasks (see ``xtuner/v1/ray/judger/gsm8k.py``). Defaults to empty dict.

    **Examples:**

        Example configuration for single judger::

            config = JudgerConfig(
                enable_batch_reward=False,
                reward_judger_configs={
                    "gsm8k": GSM8KJudgerConfig(...)
                }
            )

        Example configuration for multiple judgers::

            config = JudgerConfig(
                reward_judger_configs={
                    "gsm8k": GSM8KJudgerConfig(...),
                    "math_qa": MathQAJudgerConfig(...),
                    "custom_eval": CustomJudgerConfig(...)
                }
            )

    .. note::
        You should ensure each dataset item specifies data_source with dictionary mapping judger names to their weight ratios

        Example dataset item::

            data_item = {
                "data_source": {"gsm8k": 0.7, "math_qa": 0.3},
                "response_str": "...",
                "reward_model": {"ground_truth": "..."}
            }
    """

    enable_batch_reward: Annotated[
        bool, Parameter(help="Whether to enable batch reward calculation for multiple samples at once.")
    ] = False
    enable_weighted_judgers: Annotated[
        bool, Parameter(help="Whether to enable weighted reward calculation on multi judgers.")
    ] = False
    reward_judger_configs: Annotated[
        List[BaseModel],
        Parameter(help="A custom Python function for computing reward given model output and label."),
    ] = []
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"


@ray.remote
class JudgerController:
    """Controller for judging model outputs and calculating rewards."""

    def __init__(self, judger_config: JudgerConfig, placement_group=None):
        """Initialize the JudgerController.

        Args:
            judger_config (JudgerConfig): The configuration for the judger.
            placement_group: The Ray placement group for resource allocation.
                Defaults to None.
        """
        self.judger_config = judger_config
        # note: placement_group is used to control the placement of Ray tasks.
        # It will be implemented when gpu judger is needed
        self.placement_group = placement_group
        self.reward_judger = []
        for config in self.judger_config.reward_judger_configs:
            self.reward_judger.append(config.build())
        self.enable_weighted_judgers = (
            False if len(self.reward_judger) == 1 else self.judger_config.enable_weighted_judgers
        )
        self.logger = get_logger(log_dir=judger_config.worker_log_dir, tag="Judger")

    async def _call_single_reward_judger(self, judger: NativeJudger, group_data_item: List[RLDataFlowItem]):
        """Call a single custom reward judger to calculate rewards.

        Args:
            judger (NativeJudger): An instance of a custom judger.
            responses (List[str]): A list of model-generated responses.
            labels (List[str]): A list of ground-truth labels.

        Returns:
            List[RLJudgerResponseItem]: A list of RLJudgerResponseItem containing
                calculated rewards for each sample.
        """
        tasks = []
        if self.judger_config.enable_batch_reward:
            task = judger.judge(group_data_item)
            tasks.append(task)
        else:
            for data_item in group_data_item:
                task = judger.judge([data_item])
                tasks.append(task)
        return tasks

    async def _call_custom_reward_judger(
        self,
        active_judgers: List[NativeJudger],
        group_data_item: List[RLDataFlowItem],
    ) -> List[RLJudgerResponseItem]:
        """Call custom reward judgers to calculate rewards.

        Args:
            active_judgers (Dict[str, NativeJudger]): A dictionary of active
                judgers.
            responses (List[str]): A list of model-generated responses.
            labels (List[str]): A list of ground-truth labels.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are judger names
                and values are lists of calculated rewards for each sample.
        """
        active_judgers_len = len(active_judgers)
        task_len_list = [0]
        judger_list = []
        all_tasks = []
        for judger in active_judgers:
            tasks = await self._call_single_reward_judger(judger, group_data_item)
            all_tasks.extend(tasks)
            judger_list.append(judger.judger_name)
            task_len_list.append(task_len_list[-1] + len(tasks))

        all_results = await asyncio.gather(*all_tasks)

        assert len(all_results) == len(group_data_item) * len(active_judgers), (
            f"Expected {len(group_data_item) * len(active_judgers)} results, but got {len(all_results)}"
        )

        active_judger_results = {}
        for i in range(active_judgers_len):
            active_judger_results[judger_list[i]] = all_results[task_len_list[i] : task_len_list[i + 1]]

        # 为每个样本创建一个 RLJudgerResponseItem，不同judger的结果放在同一个item中
        uid_list = [item.uid.observation_id for item in group_data_item]
        judger_response_items_dict = {uid: RLJudgerResponseItem(uid=uid) for uid in uid_list}
        for judger_name, results in active_judger_results.items():
            for result in results:
                for data in result:
                    return_uid = data.uid
                    judger_response_items_dict[return_uid].reward.update(data.reward)
                    judger_response_items_dict[return_uid].reward.update({judger_name: data.reward})
                    judger_response_items_dict[return_uid].extra_info.update(data.extra_info)
        return list(judger_response_items_dict.values())

    async def run(
        self, group_data_item: RLDataFlowItem | List[RLDataFlowItem]
    ) -> RLJudgerResponseItem | List[RLJudgerResponseItem]:
        """Run the judging process for a group of data items.

        Args:
            group_data_item (List[RLTextDataItem]): A list of RLTextDataItem,
                each containing the response and other relevant information.

        Returns:
            List[float]: A list of final calculated rewards for each data item.
        """
        input_type_is_list = True
        if not isinstance(group_data_item, list):
            input_type_is_list = False
            group_data_item = [group_data_item]

        if self.enable_weighted_judgers:
            data_source = group_data_item[0].data.data_source
            # 如果要使用多个judger并且进行加权打分，则必须在数据集中指定data_source的分数
            assert data_source, "No data source found for the given datasets when multiple judgers are provided."
            judger_names = [judger.judger_name for judger in self.reward_judger]
            active_reward_judger = [func for func in self.reward_judger if func.judger_name in data_source]
            assert active_reward_judger, (
                f"No active reward judger in {judger_names} found for the given data source {data_source}."
            )
            judger_response_item = await self._call_custom_reward_judger(active_reward_judger, group_data_item)

            # NOTE: 只计算score的加权和
            for item in judger_response_item:
                final_reward = 0
                for name, weight in data_source.items():
                    if name in item.reward:
                        final_reward += item.reward[name]["score"] * weight
                item.reward["weighted_score"] = final_reward
        else:
            judger_response_item = await self._call_custom_reward_judger(self.reward_judger, group_data_item)
        if input_type_is_list is False:
            return judger_response_item[0]
        return judger_response_item
