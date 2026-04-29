from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils import CPUActorLauncher
from xtuner.v1.utils.logger import get_logger
from xtuner.v1.utils.type_helper import ray_method


logger = get_logger()


class Judger(ABC):
    @abstractmethod
    async def judge(self, rollout_state: RolloutState) -> RolloutState: ...


class NativeJudger(Judger):
    """Local judger implementation backed by a Python callable or HTTP
    endpoint."""

    def __init__(
        self,
        judger_name: str = "native_judger",
        reward_handler: Callable | str | None = None,
        extra_info: dict | None = None,
        request_timeout: float = 30.0,
    ):
        self._judger_name = judger_name
        self.extra_info = extra_info or {}
        self.reward_handler = reward_handler
        self.request_timeout = request_timeout

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:  # type: ignore[override]
        assert rollout_state.response is not None, (
            "RolloutState must have a response for judging. You should detokenize the response_ids in AgentLoop"
        )
        assert rollout_state.reward_model is not None and "ground_truth" in rollout_state.reward_model, (
            "RolloutState must have reward_model with 'ground_truth' for judging. You should set reward_model in AgentLoop"
        )

        input_kwargs = {
            "response": rollout_state.response,
            "label": rollout_state.reward_model["ground_truth"],
            "extra_info": {**self.extra_info},
        }

        judger_response = None
        if isinstance(self.reward_handler, str):
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(self.reward_handler, json=input_kwargs)
                response.raise_for_status()
                judger_response = response.json()
        elif callable(self.reward_handler):
            if inspect.iscoroutinefunction(self.reward_handler):
                judger_response = await self.reward_handler(**input_kwargs)
            else:
                judger_response = self.reward_handler(**input_kwargs)

        assert judger_response is not None, "Reward handler did not return a response."
        assert isinstance(judger_response, dict), (
            f"Reward handler must return a dict, but got {type(judger_response)}."
        )
        rollout_state.reward = judger_response
        return rollout_state

    def get_judger_name(self) -> str:
        return self._judger_name


class RemoteJudger(Judger):
    def __init__(self, actor: RayJudgerProxy, judger_name: str):
        self.actor = actor
        self._judger_name = judger_name

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:  # type: ignore[override]
        return await self.actor.judge.remote(rollout_state)

    def get_judger_name(self) -> str:
        return self._judger_name


class JudgerPool(Judger):
    """Round-robin dispatch across replicas of the same judger type."""

    def __init__(self, replicas: list[Judger], judger_name: str):
        if not replicas:
            raise ValueError("JudgerPool requires at least one replica.")
        self.replicas = replicas
        self._judger_name = judger_name
        self._rr_index = 0
        self._lock = asyncio.Lock()
        self._worker_loads = dict.fromkeys(range(len(replicas)), 0)

    async def _pick_replica(self) -> tuple[int, Judger]:
        async with self._lock:
            replica_idx = self._rr_index % len(self.replicas)
            self._rr_index = (self._rr_index + 1) % len(self.replicas)
            self._worker_loads[replica_idx] += 1
            return replica_idx, self.replicas[replica_idx]

    async def _release_replica(self, replica_idx: int) -> None:
        async with self._lock:
            self._worker_loads[replica_idx] -= 1

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:  # type: ignore[override]
        replica_idx, replica = await self._pick_replica()
        try:
            return await replica.judge(rollout_state)
        finally:
            await self._release_replica(replica_idx)

    def get_worker_status(self) -> dict[str, int]:
        return {f"{self._judger_name}[{idx}]": load for idx, load in self._worker_loads.items()}

    def get_judger_name(self) -> str:
        return self._judger_name


class JudgerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    judger_name: str
    reward_handler: Callable | str | None = Field(default=None, exclude=True)
    request_timeout: float = 30.0
    extra_info: dict = Field(default_factory=dict, exclude=True)
    num_ray_actors: int = Field(default=0, ge=0, description="0 means local mode, >0 means remote Ray actors.")
    num_cpus_per_actor: int = Field(default=1, gt=0, description="CPU cores per remote judger actor.")
    cpu_memory_per_actor: int = Field(
        default=1024**3, gt=0, description="CPU memory in bytes per remote judger actor."
    )

    @model_validator(mode="after")
    def _validate_ray_actor_config(self) -> JudgerConfig:
        if self.num_ray_actors == 0:
            if self.num_cpus_per_actor != 1 or self.cpu_memory_per_actor != 1024**3:
                logger.warning(
                    "num_cpus_per_actor and cpu_memory_per_actor are ignored when Judger runs in local mode."
                )
        return self

    def get_num_placement_group_bundles(self) -> int:
        return self.num_ray_actors

    def get_cpu_bundles(self) -> list[dict[str, float | int]]:
        return [
            {
                "CPU": self.num_cpus_per_actor,
                "memory": self.cpu_memory_per_actor,
            }
            for _ in range(self.get_num_placement_group_bundles())
        ]

    def build_local(self) -> Judger:
        return NativeJudger(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def _build_remote_actor(self, pg: PlacementGroup | None = None, bundle_idx: int = 0) -> RayJudgerProxy:
        return CPUActorLauncher.build_actor(
            JudgerActor,
            self,
            pg=pg,
            bundle_idx=bundle_idx,
            actor_num_cpus=self.num_cpus_per_actor,
            actor_memory=self.cpu_memory_per_actor,
        )

    def _build_remote_actors(
        self,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
        num_ray_actors: int | None = None,
    ) -> list[RayJudgerProxy]:
        actor_count = self.num_ray_actors if num_ray_actors is None else num_ray_actors
        return CPUActorLauncher.build_actors(
            JudgerActor,
            self,
            pg=pg,
            start_bundle_idx=start_bundle_idx,
            num_workers=actor_count,
            actor_num_cpus_per_worker=self.num_cpus_per_actor,
            actor_memory_per_worker=self.cpu_memory_per_actor,
        )

    def _build_remote_judger(self, pg: PlacementGroup | None = None, bundle_idx: int = 0) -> Judger:
        return RemoteJudger(self._build_remote_actor(pg=pg, bundle_idx=bundle_idx), judger_name=self.judger_name)

    def _build_remote_judgers(
        self,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
        num_ray_actors: int | None = None,
    ) -> list[Judger]:
        return [
            RemoteJudger(actor, judger_name=self.judger_name)
            for actor in self._build_remote_actors(
                pg=pg,
                start_bundle_idx=start_bundle_idx,
                num_ray_actors=num_ray_actors,
            )
        ]

    def build(self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> Judger:
        from .factory import build_judger

        return build_judger(self, pg=pg, start_bundle_idx=start_bundle_idx)


class JudgerActor:
    def __init__(self, judger_config: JudgerConfig):
        self.judger = judger_config.build_local()

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        return await self.judger.judge(rollout_state)


RayJudger = cast(ActorClass[JudgerActor], CPUActorLauncher.to_actor_class(JudgerActor))
RayJudgerProxy: TypeAlias = ActorProxy[JudgerActor]
