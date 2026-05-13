"""
Judger 体系关系图
=================

                        ┌─────────────────┐
                        │   Judger (ABC)  │  ← 所有 judger 的统一接口
                        │   judge(state)  │
                        └────────┬────────┘
                                 │ 继承
              ┌──────────────────┼───────────────────┐
              │                  │                   │
     ┌────────▼───────┐  ┌───────▼──────┐  ┌────────▼────────┐
     │  NativeJudger  │  │ RemoteJudger │  │   JudgerPool    │
     │                │  │              │  │                 │
     │ 本地执行        │  │ Ray Actor 代理│  │ 多副本负载均衡   │
     │ 调用 reward_fn │  │ 调用.remote() │  │ round-robin 分发│
     └────────────────┘  └───────┬──────┘  └────────┬────────┘
                                 │ 包含              │ 包含多个
                         ┌───────▼──────┐   ┌───────▼──────┐
                         │  JudgerActor │   │ RemoteJudger │
                         │  (Ray Actor) │   │  (同左)       │
                         │ 包装NativeJ  │   └──────────────┘
                         └───────┬──────┘
                                 │ 内部调用
                         ┌───────▼──────┐
                         │ NativeJudger │
                         └──────────────┘

     ┌──────────────────────────────────────┐
     │         ComposedJudger               │
     │                                      │
     │  select_fn → 选 branch → judge       │
     │  merge_fn  → 合并多个 branch 的结果  │
     │                                      │
     │  branches: dict[str, Judger]         │
     │  (每个 branch 可以是上面任意一种)      │
     └──────────────────────────────────────┘

构建模式
--------
未配置 external CPU pool → NativeJudger                     （纯本地，无 Ray）

配置 external CPU pool 且 num_actors = 1 → RemoteJudger
                                          └─► JudgerActor (Ray Worker)
                                                  └─► NativeJudger

配置 external CPU pool 且 num_actors > 1 → JudgerPool
                                          ├─► RemoteJudger → JudgerActor → NativeJudger
                                          ├─► RemoteJudger → JudgerActor → NativeJudger
                                          └─► RemoteJudger → JudgerActor → NativeJudger

调用链示例（remote 模式，单条打分）
------------------------------------
AgentLoop
  └─► RemoteJudger.judge(state)
        └─► JudgerActor.judge.remote(state)   ← Ray 跨进程/机器调用
              └─► NativeJudger.judge(state)
                    └─► reward_handler(response, label)
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, cast, overload

import httpx
from pydantic import BaseModel, ConfigDict, Field
from ray.actor import ActorClass, ActorProxy

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils import CPUActorLauncher, CPUActorPoolAllocation, CPUActorPoolConfig
from xtuner.v1.utils.logger import get_logger
from xtuner.v1.utils.type_helper import ray_method


logger = get_logger()


class Judger(ABC):
    @overload
    async def judge(self, rollout_state: RolloutState) -> RolloutState: ...
    @overload
    async def judge(self, rollout_state: list[RolloutState]) -> list[RolloutState]: ...
    @abstractmethod
    async def judge(self, rollout_state): ...


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
    async def judge(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:  # type: ignore[override]
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
    async def judge(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:  # type: ignore[override]
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
    """Configuration for a native judger.

    ``JudgerConfig`` describes the reward logic and optionally names the
    external CPU pool used to run the judger as Ray actors. CPU quantities are
    owned by ``CPUResourceManagerConfig``.

    Args:
        judger_name (str): Logical judger name used in logs and reward output.
        reward_handler (Callable | str | None): Reward function or HTTP
            endpoint used to score a rollout. Defaults to None.
        request_timeout (float): Timeout in seconds for HTTP reward handlers.
            Defaults to 30.0.
        extra_info (dict): Extra static information passed to the reward
            handler. Defaults to an empty dict.

    **Examples:**

    Example local judger::

        config = JudgerConfig(
            judger_name="custom/math",
            reward_handler=compute_reward,
            extra_info={"score": 1.0},
        )

    Remote actor judgers are enabled by setting ``external_cpu``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    judger_name: str
    reward_handler: Callable | str | None = Field(default=None, exclude=True)
    request_timeout: float = 30.0
    extra_info: dict = Field(default_factory=dict, exclude=True)
    external_cpu: CPUActorPoolConfig | None = None

    def build_local(self) -> Judger:
        return NativeJudger(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def _build_remote_actor(self, external_cpu_allocation: CPUActorPoolAllocation) -> RayJudgerProxy:
        return CPUActorLauncher.build_actor(
            JudgerActor,
            self,
            actor_num_cpus=external_cpu_allocation.num_cpus_per_actor,
            actor_memory=external_cpu_allocation.memory_per_actor,
        )

    def _build_remote_actors(
        self,
        external_cpu_allocation: CPUActorPoolAllocation,
    ) -> list[RayJudgerProxy]:
        return [self._build_remote_actor(external_cpu_allocation) for _ in range(external_cpu_allocation.num_actors)]

    def _build_remote_judger(self, external_cpu_allocation: CPUActorPoolAllocation) -> Judger:
        return RemoteJudger(self._build_remote_actor(external_cpu_allocation), judger_name=self.judger_name)

    def _build_remote_judgers(
        self,
        external_cpu_allocation: CPUActorPoolAllocation,
    ) -> list[Judger]:
        return [
            RemoteJudger(actor, judger_name=self.judger_name)
            for actor in self._build_remote_actors(external_cpu_allocation)
        ]

    def build(self) -> Judger:
        from .factory import build_judger

        return build_judger(self)


class JudgerActor:
    def __init__(self, judger_config: JudgerConfig):
        self.judger = judger_config.build_local()

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        return await self.judger.judge(rollout_state)


RayJudger = cast(ActorClass[JudgerActor], CPUActorLauncher.to_actor_class(JudgerActor))
RayJudgerProxy: TypeAlias = ActorProxy[JudgerActor]
