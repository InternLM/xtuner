import asyncio
import os
from typing import List

import ray

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RLRolloutResponseItem,
    update_dataflow_item,
)
from xtuner.v1.ray.environment.base_env import BaseEnvironment
from xtuner.v1.utils import get_logger


@ray.remote(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))
class SingleTurnEnvironment(BaseEnvironment):
    """A single-turn environment for handling generation and evaluation tasks.

    This class extends `BaseEnvironment` to provide a concrete implementation for
    single-turn interactions. It manages the rollout process for generating responses
    and can coordinate with a judger for evaluation.

    Args:
        environment (str): The name of the environment.
        rollout_pg: The placement group for scheduling rollout Ray actors.
        rollout_cfg (optional): Configuration for the rollout controller. Defaults to None.
        judger_pg (Any): The placement group for scheduling judger Ray actors.
                         Defaults to None indicates using the rollout_pg.
        judger_cfg (optional): Configuration for the judger controller. Defaults to None.
    """

    def __init__(self, environment: str, rollout_pg, rollout_cfg=None, judger_pg=None, judger_cfg=None):
        super().__init__(environment, rollout_pg, rollout_cfg, judger_pg, judger_cfg)
        worker_log_dir = rollout_cfg.worker_log_dir if rollout_cfg else judger_cfg.worker_log_dir
        self.logger = get_logger(log_dir=worker_log_dir, tag="SingleTurnEnv")
        if rollout_cfg.enable_return_routed_experts:
            self.logger.info("！！！ Enable `return routed experts` in rollout controller. ！！！")
        self.rollout_timeout = rollout_cfg.rollout_timeout if rollout_cfg else 1200.0
        self.judger_timeout = judger_cfg.judger_timeout if judger_cfg else 1200.0

    async def generate(
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        """Generate responses for a batch of RLTextDataItem using the rollout
        controller.

        Each item in `group_data_items` will be sent to the rollout controller for response generation
        with the provided sampling parameters. The generated response string and state will be
        added to each RLTextDataItem in-place as `response_str` and `state` fields.

        Args:
            group_data_items (List[RLTextDataItem]):
                A list of RLTextDataItem objects containing the prompts/messages for generation.
            sample_params: Sampling parameters for the generation process. The type should match
                the rollout controller's expected sampling parameter type (e.g., SampleParams or dict).

        Returns:
            List[RLTextDataItem]:
                The same list of RLTextDataItem, with each item enriched with the generated response
                and state from the rollout controller.
        """
        if self.rollout_controller:
            # 在env中对输入的数据进行转换，是为了支持rollout_controller单独作为rollout engine使用，使各个模块进行解耦
            # 每个模块返回独立的data item, 在env中进行更新
            response_futures = []
            for sample in group_data_items:
                extra_info = sample.data.extra_info if hasattr(sample.data, "extra_info") else {}
                extra_info.update({"action_id": sample.uid.action_id})
                response_future = self.rollout_controller.rollout.remote(
                    prompt=sample.data.messages,
                    input_ids=sample.data.input_ids,
                    sample_params=sample_params,
                    extra_params=extra_params,
                    extra_info=extra_info,
                )
                for sample in group_data_items
            ]
            try:
                rollout_responses = await asyncio.wait_for(
                    asyncio.gather(*response_future), timeout=self.rollout_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error("Get rollout controller response timeout and return the failed response.")
                rollout_responses = [
                    RLRolloutResponseItem(
                        finish_reason="failed",
                    )
                    for _ in group_data_items
                ]
            group_data_items = update_dataflow_item(group_data_items, "env.rollout", rollout_responses)
        return group_data_items

    async def run(
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
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
        group_data_items = await self.generate(group_data_items, sample_params, extra_params)  # type: ignore[assignment]
        skip_judger = any(
            item.env.rollout.finish_reason == "abort" or item.env.rollout.finish_reason == "failed"
            for item in group_data_items
        )
        if self.judger_controller and not skip_judger:
            try:
                judger_responses: List[RLJudgerResponseItem] = await asyncio.wait_for(
                    self.judger_controller.run.remote(group_data_items), timeout=self.judger_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error("Get judger controller response timeout and return the failed response.")
                judger_responses = [
                    RLJudgerResponseItem(
                        extra_info={"state": "failed"},
                    )
                    for _ in group_data_items
                ]
            group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_responses)
        return group_data_items
