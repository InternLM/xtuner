import asyncio
import os
from pathlib import Path
from typing import List

import ray

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RLRolloutResponseItem,
    update_dataflow_item,
    update_rollout_item,
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
        rollout_controller (optional): An instance of the rollout controller. Defaults to None.
        judger_controller (optional): An instance of the judger controller. Defaults to None.
    """

    def __init__(
        self,
        environment: str,
        rollout_pg,
        rollout_cfg=None,
        judger_pg=None,
        judger_cfg=None,
        rollout_controller=None,
        judger_controller=None,
    ):
        super().__init__(
            environment, rollout_pg, rollout_cfg, judger_pg, judger_cfg, rollout_controller, judger_controller
        )
        if rollout_cfg:
            worker_log_dir = rollout_cfg.worker_log_dir
        elif judger_cfg:
            worker_log_dir = judger_cfg.worker_log_dir
        else:
            worker_log_dir = Path.cwd() / "work_dir"
        self.logger = get_logger(log_dir=worker_log_dir, tag="SingleTurnEnv")
        if rollout_cfg and rollout_cfg.enable_return_routed_experts:
            self.logger.info("!!! Enable `return routed experts` in rollout controller. !!!")
        self.rollout_timeout = rollout_cfg.rollout_timeout if rollout_cfg else 1200.0
        self.judger_timeout = judger_cfg.judger_timeout if judger_cfg else 1200.0
        # The timeout for the environment to wait for the rollout controller's response.
        # This should be longer than the controller's internal timeout (`rollout_timeout`)
        # to account for potential queuing delays and other overheads.
        self.timeout_multiplier = 2.0

    async def generate(
        self,
        group_data_items: List[RLDataFlowItem],
        sample_params=None,
        extra_params=None,
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
            response_future = []
            for sample in group_data_items:
                sample.data.extra_info["root_id"] = sample.uid.root_id
                sample.data.extra_info["action_id"] = sample.uid.action_id
                fut = self.rollout_controller.rollout.remote(
                    prompt=sample.data.messages,
                    input_ids=sample.data.input_ids,
                    sample_params=sample_params,
                    extra_params=extra_params,
                    extra_info=sample.data.extra_info,
                )
                response_future.append(fut)
            try:
                rollout_responses = await asyncio.wait_for(
                    asyncio.gather(*response_future), timeout=self.rollout_timeout * self.timeout_multiplier
                )
            except asyncio.TimeoutError:
                self.logger.error("Get rollout controller response timeout and return the failed response.")
                rollout_responses = [RLRolloutResponseItem(state="skipped") for _ in group_data_items]
            group_data_items = update_rollout_item(group_data_items, rollout_responses)
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
        continue_judger = all(item.env.rollout.state == "completed" for item in group_data_items)
        if self.judger_controller and continue_judger:
            try:
                judger_responses: List[RLJudgerResponseItem] = await asyncio.wait_for(
                    self.judger_controller.run.remote(group_data_items),
                    timeout=self.judger_timeout * self.timeout_multiplier,
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
