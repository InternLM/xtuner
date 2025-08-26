from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from .fsdp import FSDPConfig
from .optim import OptimConfig


if TYPE_CHECKING:
    from xtuner.v1.model.base import BaseModel
    from xtuner.v1.model.interns1 import InternS1Config


@runtime_checkable
class ModelConfigProto(Protocol):
    def build(self) -> "BaseModel":
        """Build the model configuration."""
        raise NotImplementedError


# TODO: (caoweihan) Is this engine config necessary? Or just provide a `build_engine` function
class EngineConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    fsdp_cfg: FSDPConfig
    optim_cfg: OptimConfig
    model_cfg: ModelConfigProto

    def build(self):
        from xtuner.v1.engine.interns1_train_engine import InternS1TrainEngine
        from xtuner.v1.engine.train_engine import TrainEngine

        if isinstance(self.model_cfg, InternS1Config):
            return InternS1TrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
        else:
            return TrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
