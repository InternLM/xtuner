import asyncio
import os
from itertools import chain, groupby
from typing import Callable, Dict, List

import ray
from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RLDataFlowItem

from .base_env import BaseEnvironment


@ray.remote(max_concurrency=int(os.environ.get("XTUNER_MAX_CONCURRENCY", 2000)))
class ComposedEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment: str,
        rollout_controller,
        environments: Dict[str, BaseEnvironment | dict],
        router: Callable[[RLDataFlowItem], str] = lambda item: item.data.extra_info['environment'],
    ):
        super().__init__(environment, None, None, None, None)
        self.rollout_controller = rollout_controller
        self.environments = {name: create_object(env) for name, env in environments.items()}
        self.router = router

    async def generate(self, data, sample_params, extra_params):
        return await super().generate(data, sample_params, extra_params)

    async def run(
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        results = await asyncio.gather(
            *[
                self.environments[env_name].run.remote(
                    list(items), sample_params=sample_params, extra_params=extra_params
                )
                for env_name, items in groupby(sorted(group_data_items, key=self.router), key=self.router)
            ]
        )
        return list(chain.from_iterable(results))
