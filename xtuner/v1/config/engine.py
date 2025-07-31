from pydantic import BaseModel

from .base_model import MoEConfig, TransformerConfig
from .fsdp import FSDPConfig
from .optim import OptimConfig


# TODO: (caoweihan) Is this engine config necessary? Or just provide a `build_engine` function
class EngineConfig(BaseModel):
    fsdp_cfg: FSDPConfig
    optim_cfg: OptimConfig
    model_cfg: TransformerConfig

    def build(self):
        from xtuner.v1.engine.dense_train_engine import DenseTrainEngine
        from xtuner.v1.engine.moe_train_engine import MoETrainEngine

        if isinstance(self.model_cfg, MoEConfig):
            return MoETrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
        else:
            return DenseTrainEngine(model_cfg=self.model_cfg, optim_cfg=self.optim_cfg, fsdp_cfg=self.fsdp_cfg)
