from typing import Protocol, runtime_checkable

from pydantic import BaseModel as PydanticBaseModel

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeConfigProtocol, VisionComposeTrainEngine
from xtuner.v1.model.base import BaseModel, ConfigDict


@runtime_checkable
class ModelConfigProto(Protocol):
    def build(self) -> BaseModel:
        """Build the model configuration."""
        raise NotImplementedError


class EngineConfig(PydanticBaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )
    fsdp_cfg: FSDPConfig
    optim_cfg: OptimConfig
    model_cfg: ModelConfigProto

    def build(self):
        if isinstance(self.model_cfg, VisionComposeConfigProtocol):
            return VisionComposeTrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
        else:
            return TrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
