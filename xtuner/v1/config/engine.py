from pydantic import BaseModel

from .base_model import MoEConfig, TransformerConfig
from .fsdp import FSDPConfig
from .moe_loss import MoELossConfig
from .optim import LRConfig, OptimConfig


class EngineConfig(BaseModel):
    model: TransformerConfig
    optim: OptimConfig
    lr: LRConfig
    fsdp: FSDPConfig


class MoEEngineConfig(EngineConfig):
    model: MoEConfig
    moe_loss: MoELossConfig
