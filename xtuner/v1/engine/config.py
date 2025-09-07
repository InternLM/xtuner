from typing import Protocol, runtime_checkable

from pydantic import BaseModel as PydanticBaseModel

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.compose.intern_s1 import InternS1BaseConfig


@runtime_checkable
class ModelConfigProto(Protocol):
    def build(self) -> BaseModel:
        """Build the model configuration."""
        raise NotImplementedError


class EngineConfig(PydanticBaseModel):
    model_config = {"arbitrary_types_allowed": True}
    fsdp_cfg: FSDPConfig
    optim_cfg: OptimConfig
    model_cfg: ModelConfigProto

    def build(self):
        from xtuner.v1.engine.intern_s1_train_engine import InternS1TrainEngine
        from xtuner.v1.engine.train_engine import TrainEngine

        # TODO: (hha) Remove this hardcode. Use Composable Engine rather than InternS1BaseEngine
        if isinstance(self.model_cfg, InternS1BaseConfig):
            return InternS1TrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
        else:
            return TrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
