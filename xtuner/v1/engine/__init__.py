from xtuner.v1.config import EngineConfig, MoEConfig, MoEEngineConfig

from .dense_train_engine import DenseTrainEngine
from .moe_train_engine import MoETrainEngine


def build_engine(config: EngineConfig) -> DenseTrainEngine | MoETrainEngine:
    """Build the appropriate training engine based on the configuration.

    Args:
        config (MoEEngineConfig): The configuration for the training engine.

    Returns:
        DenseTrainEngine | MoETrainEngine: An instance of the appropriate training engine.
    """
    if isinstance(config, MoEEngineConfig):
        return MoETrainEngine(
            model_cfg=config.model,
            moe_loss_cfg=config.moe_loss,  # TODO: Loss calculation will be moved to model
            optim_cfg=config.optim,
            lr_cfg=config.lr,
            fsdp_cfg=config.fsdp,
        )
    else:
        return DenseTrainEngine(
            model_cfg=config.model,
            optim_cfg=config.optim,
            lr_cfg=config.lr,
            fsdp_cfg=config.fsdp,
        )


__all__ = [
    "DenseTrainEngine",
    "MoETrainEngine",
]
