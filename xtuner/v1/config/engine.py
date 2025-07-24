from pydantic import BaseModel

from .base_model import MoEConfig, TransformerConfig
from .fsdp import FSDPConfig
from .optim import OptimConfig


class EngineConfig(BaseModel):
    model: TransformerConfig
    optim: OptimConfig
    fsdp: FSDPConfig


class MoEEngineConfig(EngineConfig):
    model: MoEConfig
    # The number of micro-batches for intra-layer all2all overlap.
    intra_layer_micro_batch: int = 1
