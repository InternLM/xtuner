import asyncio
import inspect
from typing import Any, Callable, Dict, List

import httpx
import ray
import ray.util.queue
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.utils import get_logger


class JudgerConfig(BaseModel):
    enable_batch_reward: Annotated[
        bool, Parameter(help="Whether to enable batch reward calculation for multiple samples at once.")
    ] = False
    reward_functions: Annotated[
        Dict[str, Callable],
        Parameter(help="A custom Python function for computing reward given model output and label."),
    ] = {}
    remote_urls: Annotated[
        Dict[str, str], Parameter(help="The remote server URL for external judger service, if used.")
    ] = {}
    extra_info: Annotated[
        Dict[str, Dict[Any, Any]],
        Parameter(help="Extra information to be passed to the reward function or remote service."),
    ] = {}
    request_timeout: Annotated[float, Parameter(help="Timeout in seconds for remote reward model requests.")] = 300.0
    reward_ratio: Annotated[
        Dict[str, float], Parameter(help="The weight of each reward function when combining multiple rewards.")
    ] = {}


@ray.remote
def batch_reward_task(reward_func: Callable, responses: List[str], labels: List[str], extra_info: dict):
    """Ray Task for batch-processing reward functions."""
    if inspect.iscoroutinefunction(reward_func):
        return asyncio.run(reward_func(responses, labels, extra_info))
    else:
        return reward_func(responses, labels, extra_info)


@ray.remote
def single_reward_task(reward_func: Callable, response: str, label: str, extra_info: dict):
    """Ray Task for single-item-processing reward functions."""
    if inspect.iscoroutinefunction(reward_func):
        return asyncio.run(reward_func(response, label, extra_info))
    else:
        return reward_func(response, label, extra_info)


@ray.remote
class JudgerController:
    def __init__(self, config: JudgerConfig, placement_group=None):
        self.config = config
        self.placement_group = placement_group
        self.reward_functions = self.config.reward_functions
        self.remote_urls = self.config.remote_urls
        self.http_client = httpx.AsyncClient(timeout=self.config.request_timeout)
        self.logger = get_logger()

    async def _call_custom_reward_functions(self, responses: List[str], labels: List[str]):
        if not self.config.reward_functions:
            return {}
        if self.config.enable_batch_reward:
            # --- 批量处理逻辑 (Batch Mode) ---
            tasks = {}
            for name, func in self.config.reward_functions.items():
                task_builder = batch_reward_task
                if self.placement_group:
                    task_builder = task_builder.options(placement_group=self.placement_group)
                tasks[name] = task_builder.remote(func, responses, labels, self.config.extra_info[name])

            results = await asyncio.gather(*tasks.values())
            return dict(zip(tasks.keys(), results))
        else:
            # --- 单个处理逻辑 (Iterative Mode) ---
            tasks_per_function = {}
            for name, func in self.config.reward_functions.items():
                sample_tasks = []
                for i in range(len(responses)):
                    task_builder = single_reward_task
                    if self.placement_group:
                        task_builder = task_builder.options(placement_group=self.placement_group)
                    task = task_builder.remote(func, responses[i], labels[i], self.config.extra_info[name])
                    sample_tasks.append(task)
                tasks_per_function[name] = asyncio.gather(*sample_tasks)

            all_results = await asyncio.gather(*tasks_per_function.values())
            return dict(zip(tasks_per_function.keys(), all_results))

    async def _call_remote_reward_models(self, responses: List[str], labels: List[str]):
        if not self.config.remote_urls or self.config.enable_batch_reward:
            return {}

        payload = {"responses": responses, "labels": labels}

        async def fetch(name, url):
            try:
                payload["extra_info"] = self.config.extra_info[name]
                resp = await self.http_client.post(url, json=payload)
                resp.raise_for_status()
                return name, resp.json().get("scores", [])
            except httpx.RequestError as exc:
                self.logger.error(f"Request error to {name} ({url}): {exc}")
            except httpx.HTTPStatusError as exc:
                self.logger.error(f"HTTP error from {name} ({url}): {exc.response.status_code}")
            return name, [0.0] * len(responses)

        tasks = [fetch(name, url) for name, url in self.config.remote_urls.items()]
        results = await asyncio.gather(*tasks)
        return {name: scores for name, scores in results if scores}

    async def run(self, group_data_item: List[RLTextDataItem]) -> List[float]:
        if not group_data_item:
            return []
        batch_responses = [item["response_str"] or "" for item in group_data_item]
        batch_labels = [item["reward_model"]["ground_truth"] for item in group_data_item]
        local_rewards_task = self._call_custom_reward_functions(batch_responses, batch_labels)
        remote_rewards_task = self._call_remote_reward_models(batch_responses, batch_labels)
        all_reward_results = await asyncio.gather(local_rewards_task, remote_rewards_task)

        num_samples = len(group_data_item)
        rewards_per_sample: List[Dict[str, float]] = [{} for _ in range(num_samples)]
        for reward_dict in all_reward_results:
            for name, scores in reward_dict.items():
                if len(scores) == num_samples:
                    for i in range(num_samples):
                        rewards_per_sample[i][name] = scores[i]
                else:
                    self.logger.warning(f"Reward '{name}' returned {len(scores)} scores, but expected {num_samples}.")

        reward_list = []
        for i, item in enumerate(group_data_item):
            reward = 0.0
            for name, score in rewards_per_sample[i].items():
                reward += score * self.config.reward_ratio.get(name, 0.0)
            reward_list.append(reward)
        return reward_list

    async def __aexit__(self, exc_type, exc, tb):
        """在 Actor 退出时关闭 HTTP 客户端。"""
        if self.http_client:
            await self.http_client.aclose()
