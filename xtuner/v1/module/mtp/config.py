"""Configuration for Multi-Token Prediction (MTP)."""

from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict


class MTPConfig(BaseModel):
    """Configuration for Multi-Token Prediction (MTP).

    MTP extends the prediction scope to multiple future tokens at each position,
    creating denser training signals and potentially improving data efficiency.

    This config only contains training-related hyperparameters. The actual
    construction of MTP layers (including choosing Dense vs MoE decoder layers)
    is handled by the model (Dense/MoE) which knows how to create the appropriate
    decoder layers.

    Args:
        num_layers (int): Number of MTP layers (prediction depths). Each layer
            predicts tokens at increasing future positions (i+1, i+2, ..., i+D).
        loss_scaling_factor (float): Scaling factor for MTP loss. The total MTP loss
            is computed as the average of losses across all depths, multiplied by
            this factor. Default: 0.1.

    Example:
        >>> # In model config
        >>> config = TransformerConfig(
        ...     ...,
        ...     mtp_config=MTPConfig(
        ...         num_layers=2,
        ...         loss_scaling_factor=0.1,
        ...     ),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    num_layers: Annotated[int, Parameter(group="model")]
    loss_scaling_factor: Annotated[float, Parameter(group="model")] = 0.1
