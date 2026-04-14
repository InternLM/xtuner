import inspect
from typing import Awaitable, Callable, TypeAlias

import ray

from xtuner.v1.data_proto import RolloutState

from .native import Judger, JudgerConfig, RayJudgerProxy


JudgerCallable: TypeAlias = Callable[[RolloutState], RolloutState | Awaitable[RolloutState]]
JudgerLike: TypeAlias = Judger | RayJudgerProxy | JudgerCallable
JudgerSpec: TypeAlias = JudgerLike | dict[str, JudgerLike] | None
JudgerConfigLike: TypeAlias = JudgerConfig | JudgerCallable
JudgerConfigSpec: TypeAlias = JudgerConfigLike | dict[str, JudgerConfigLike] | None


class JudgerSpecConfig:
    def __init__(self, judger_config: JudgerConfigSpec):
        self.judger_config = judger_config

    @classmethod
    def from_judger_config(cls, judger_config: JudgerConfig) -> "JudgerSpecConfig":
        return cls(judger_config)

    @classmethod
    def from_judger_config_dict(cls, judger_config: dict[str, JudgerConfigLike]) -> "JudgerSpecConfig":
        return cls(judger_config)

    @classmethod
    def from_judger_callable(cls, judger_callable: JudgerCallable) -> "JudgerSpecConfig":
        return cls(judger_callable)

    @classmethod
    def from_value(cls, judger_config: JudgerConfigSpec) -> "JudgerSpecConfig":
        return cls(judger_config)

    def build(self) -> JudgerSpec:
        judger_config = self.judger_config
        if judger_config is None:
            return None

        if isinstance(judger_config, dict):
            judger_dict = {}
            for key, config in judger_config.items():
                if isinstance(config, JudgerConfig):
                    judger_dict[key] = config.build()
                elif callable(config):
                    judger_dict[key] = config
                else:
                    raise ValueError(f"Invalid judger config type: {type(config)} for key {key}")
            return judger_dict

        if isinstance(judger_config, JudgerConfig):
            return judger_config.build()

        if callable(judger_config):
            return judger_config

        raise ValueError(f"Invalid judger config type: {type(judger_config)}")


def _resolve_judger_from_dict(judger_dict: dict[str, JudgerLike], rollout_state: RolloutState) -> JudgerLike:
    if not judger_dict:
        raise ValueError("judger dict must not be empty.")

    candidate_keys: list[str] = []
    if rollout_state.task_name:
        candidate_keys.append(rollout_state.task_name)

    data_source = rollout_state.data_source
    if isinstance(data_source, str):
        candidate_keys.append(data_source)
    elif isinstance(data_source, dict):
        for field in ("name", "id", "type", "data_source"):
            value = data_source.get(field)
            if isinstance(value, str):
                candidate_keys.append(value)

    for key in candidate_keys:
        if key in judger_dict:
            return judger_dict[key]

    if "default" in judger_dict:
        return judger_dict["default"]

    if len(judger_dict) == 1:
        return next(iter(judger_dict.values()))

    raise KeyError(
        "Unable to resolve judger from dict with "
        f"task_name={rollout_state.task_name!r}, data_source={rollout_state.data_source!r}, "
        f"available_keys={sorted(judger_dict)}"
    )


async def judge_sample(judger: JudgerSpec, rollout_state: RolloutState) -> RolloutState:
    if judger is None:
        return rollout_state

    if isinstance(judger, dict):
        judger = _resolve_judger_from_dict(judger, rollout_state)

    if isinstance(judger, Judger):
        rollout_state = await judger.judge(rollout_state)
    elif isinstance(judger, ray.actor.ActorHandle):
        rollout_state = await judger.judge.remote(rollout_state)
    elif callable(judger):
        judger_result = judger(rollout_state)
        if inspect.isawaitable(judger_result):
            rollout_state = await judger_result
        else:
            rollout_state = judger_result
    else:
        raise ValueError(f"Invalid judger type: {type(judger)}")

    if not isinstance(rollout_state, RolloutState):
        raise TypeError(f"Judger must return RolloutState, but got {type(rollout_state)}")
    return rollout_state
