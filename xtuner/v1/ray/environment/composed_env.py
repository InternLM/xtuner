import asyncio
import os
from itertools import chain, count, groupby
from typing import Any, Callable, Dict, List, Union

import ray
from lagent.utils import create_object

from xtuner.v1.data_proto.rl_data import RLDataFlowItem

from .base_env import BaseEnvironment


EnvSpec = Union[BaseEnvironment, dict, List[Union[BaseEnvironment, dict]]]


@ray.remote(max_concurrency=int(os.environ.get("XTUNER_MAX_CONCURRENCY", 2000)))  # type: ignore[call-overload]
class ComposedEnvironment(BaseEnvironment):
    """Dispatches rollout requests to registered sub-environments.

    Each registered name maps to a pool of one or more sub-environment
    actors. When a pool has more than one actor, ``run`` round-robins
    across them so concurrent rollout traffic is spread over multiple
    Python processes (and, with ``scheduling_strategy="SPREAD"`` on the
    sub-environment class, across multiple nodes). A bare ``dict`` or
    actor handle registers a single-actor pool and preserves the old
    behavior.

    Args:
        environment (str): Name tag stored on :class:`BaseEnvironment`.
        rollout_controller: Shared rollout controller handle forwarded
            to sub-environments.
        environments (Dict[str, EnvSpec]): Mapping of env name to a spec
            or list of specs. A list triggers pool mode with one actor
            per entry.
        router (Callable[[RLDataFlowItem], str]): Selects which
            registered name a data item goes to. Defaults to reading
            ``item.data.extra_info["environment"]``.
    """

    def __init__(
        self,
        environment: str,
        rollout_controller,
        environments: Dict[str, EnvSpec],
        router: Callable[[RLDataFlowItem], str] = lambda item: item.data.extra_info["environment"],
    ):
        super().__init__(environment, None, None, None, None)
        self.rollout_controller = rollout_controller
        self.environments: Dict[str, List[Any]] = {
            name: [create_object(e) for e in env] if isinstance(env, list) else [create_object(env)]
            for name, env in environments.items()
        }
        self._pool_cursors: Dict[str, Any] = {name: count() for name in self.environments}
        self.router = router

    async def generate(self, data, sample_params, extra_params):
        return await super().generate(data, sample_params, extra_params)

    async def run(  # type: ignore[override]
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        results = await asyncio.gather(
            *[
                self._pick(env_name).run.remote(list(items), sample_params=sample_params, extra_params=extra_params)
                for env_name, items in groupby(sorted(group_data_items, key=self.router), key=self.router)
            ]
        )
        return list(chain.from_iterable(results))

    def _pick(self, name: str):
        pool = self.environments[name]
        return pool[next(self._pool_cursors[name]) % len(pool)]
