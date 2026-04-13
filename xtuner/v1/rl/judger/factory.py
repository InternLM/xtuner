from typing import Awaitable, Callable, TypeAlias

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
