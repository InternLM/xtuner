"""
Judger 体系关系图
=================

                        ┌─────────────────┐
                        │     Judger      │  ← 所有 judger 的统一接口
                        │   judge(state)  │
                        │ batch_judge(list)│
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
     │  data_source → 选 branch → judge     │
     │  merge_fn  → 合并多个 branch 的结果  │
     │                                      │
     │  branches: dict[str, Judger]         │
     │  (每个 branch 可以是上面任意一种)      │
     └──────────────────────────────────────┘

构建模式
--------
未配置 external CPU resources → NativeJudger                     （纯本地，无 Ray）

配置 external CPU resources 且 num_workers = 1 → RemoteJudger
                                          └─► JudgerActor (Ray Worker)
                                                  └─► NativeJudger

配置 external CPU resources 且 num_workers > 1 → JudgerPool
                                          ├─► RemoteJudger → JudgerActor → NativeJudger
                                          ├─► RemoteJudger → JudgerActor → NativeJudger
                                          └─► RemoteJudger → JudgerActor → NativeJudger

调用链示例（remote 模式，单条打分）
------------------------------------
AgentLoop
  └─► RemoteJudger.judge(state)
        ├─► preprocess(state)                 ← driver 侧提取轻量 payload
        └─► JudgerActor.judge_payload.remote(payload)
              └─► NativeJudger.judge_payload(payload)
                    └─► reward_handler(response, label)

批量打分语义
------------
Judger.judge 只处理单条 RolloutState。需要批量打分时调用
Judger.batch_judge(list[RolloutState])。不是所有具体 judger 都支持 batch；
比如 NativeJudger 和 CompassVerifierV2 会在 batch_judge 入口直接报错。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, TypeAlias, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field
from ray.actor import ActorClass, ActorProxy

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.utils import CPUActorLauncher, CPUResourcesConfig
from xtuner.v1.utils.logger import get_logger
from xtuner.v1.utils.type_helper import ray_method


logger = get_logger()

JudgerPayload: TypeAlias = dict[str, Any]
JudgerPayloadBatch: TypeAlias = JudgerPayload | list[JudgerPayload]
JudgerOutput: TypeAlias = dict[str, Any]
JudgerOutputBatch: TypeAlias = JudgerOutput | list[JudgerOutput]


class Judger:
    def __init__(self, judger_name: str | None = None):
        self._judger_name = judger_name or self.__class__.__name__

    def preprocess(self, rollout_state: RolloutState) -> JudgerPayload:
        return {
            "response": rollout_state.response,
            "label": rollout_state.reward_model.get("ground_truth") if rollout_state.reward_model else None,
            "message": rollout_state.message,
            "status": rollout_state.status,
            "data_source": rollout_state.data_source,
            "task_name": rollout_state.task_name,
        }

    def postprocess(self, rollout_state: RolloutState, output: JudgerOutput) -> RolloutState:
        rollout_state.reward = output
        return rollout_state

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        payload = self.preprocess(rollout_state)
        output = await self.judge_payload(payload)
        if isinstance(output, list):
            raise TypeError("Judger returned a list output for a single rollout state.")
        return self.postprocess(rollout_state, output)

    async def batch_judge(self, rollout_states: list[RolloutState]) -> list[RolloutState]:
        payloads = [self.preprocess(state) for state in rollout_states]
        outputs = await self.judge_payload(payloads)
        if not isinstance(outputs, list):
            raise TypeError(f"Judger returned a single output for {len(rollout_states)} rollout states.")
        if len(outputs) != len(rollout_states):
            raise ValueError(f"Judger returned {len(outputs)} outputs for {len(rollout_states)} rollout states.")
        return [self.postprocess(state, output) for state, output in zip(rollout_states, outputs)]

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        raise NotImplementedError(f"{self.__class__.__name__}.judge_payload() is not implemented.")

    def get_judger_name(self) -> str:
        return self._judger_name


class NativeJudger(Judger):
    """Local judger implementation backed by a Python callable or HTTP
    endpoint.

    ``NativeJudger`` calls one reward handler for one rollout sample. It does
    not support ``batch_judge(list[RolloutState])``; callers that need grouped
    routing should use ``ComposedJudger`` or a judger implementation that
    explicitly supports batch payloads.
    """

    def __init__(
        self,
        judger_name: str = "native_judger",
        reward_handler: Callable | str | None = None,
        extra_info: dict | None = None,
        request_timeout: float = 30.0,
    ):
        super().__init__(judger_name=judger_name)
        self.extra_info = extra_info or {}
        self.reward_handler = reward_handler
        self.request_timeout = request_timeout

    async def batch_judge(self, rollout_states: list[RolloutState]) -> list[RolloutState]:
        raise NotImplementedError("NativeJudger does not support batch_judge.")

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        assert not isinstance(payload, list), "NativeJudger does not support batch payloads."
        assert payload["response"] is not None, (
            "RolloutState must have a response for judging. You should detokenize the response_ids in AgentLoop"
        )
        assert payload["label"] is not None, (
            "RolloutState must have reward_model with 'ground_truth' for judging. You should set reward_model in "
            "AgentLoop"
        )
        input_kwargs = {
            "response": payload["response"],
            "label": payload["label"],
            "extra_info": {**self.extra_info},
        }

        judger_response = None
        if isinstance(self.reward_handler, str):
            # TODO: 如果超时或者返回状态错误，会如何？
            # TODO: 这里不好 try 的原因是在异常情况下，我们应该给 -1 还是 0 分呢？
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
        return cast(JudgerOutput, judger_response)


class RemoteJudger(Judger):
    """Driver-side proxy for a Ray-hosted judger.

    ``RemoteJudger`` keeps the same ``Judger`` interface as local judgers, so
    callers still pass ``RolloutState`` to ``judge``. The base ``Judger`` logic
    converts that state to a lightweight payload on the driver, then this proxy
    sends only the payload to ``JudgerActor``. ``JudgerActor`` lives in the Ray
    worker process and owns the real local judger instance that executes
    ``judge_payload``. Batch support is determined by that actor-side judger.
    """

    def __init__(self, actor: RayJudgerProxy, judger_name: str):
        super().__init__(judger_name=judger_name)
        self.actor = actor

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        return await self.actor.judge_payload.remote(payload)


class JudgerPool(Judger):
    """Round-robin dispatch across replicas of the same judger type."""

    def __init__(self, replicas: list[Judger], judger_name: str):
        super().__init__(judger_name=judger_name)
        if not replicas:
            raise ValueError("JudgerPool requires at least one replica.")
        self.replicas = replicas
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

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        replica_idx, replica = await self._pick_replica()
        try:
            return await replica.judge_payload(payload)
        finally:
            await self._release_replica(replica_idx)

    def get_worker_status(self) -> dict[str, int]:
        return {f"{self._judger_name}[{idx}]": load for idx, load in self._worker_loads.items()}


class JudgerConfig(BaseModel):
    """Configuration for a native judger.

    ``JudgerConfig`` describes the reward logic and optionally names the
    external CPU resources used to run the judger as Ray actors. CPU quantities are
    declared by ``CPUResourcesConfig`` and validated by ``CPUResourceManager``.

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

    Remote actor judgers are enabled by setting ``cpu_resources``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    judger_name: str
    reward_handler: Callable | str | None = Field(default=None, exclude=True)
    request_timeout: float = 30.0
    extra_info: dict = Field(default_factory=dict, exclude=True)
    cpu_resources: CPUResourcesConfig | None = None

    def build_local(self) -> Judger:
        return NativeJudger(
            judger_name=self.judger_name,
            reward_handler=self.reward_handler,
            request_timeout=self.request_timeout,
            extra_info=self.extra_info,
        )

    def build(self) -> Judger:
        from .factory import build_judger

        return build_judger(self)


class JudgerActor:
    def __init__(self, judger_config: JudgerConfig):
        self.judger = judger_config.build_local()

    @ray_method
    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        return await self.judger.judge_payload(payload)


RayJudger = cast(ActorClass[JudgerActor], CPUActorLauncher.to_actor_class(JudgerActor))
RayJudgerProxy: TypeAlias = ActorProxy[JudgerActor]
