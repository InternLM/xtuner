"""Configuration for Multi-Token Prediction (MTP)."""

from typing import Annotated, Literal

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
        share_weights (bool): Whether to share the weights of the MTP layers.
            If True, the weights of the MTP layers are shared across all layers.
            Default: False.
        detach_mtp_lm_head_weight (bool): Whether to detach the LM head weight.
            This is used in RL training. Default is False.
        detach_mtp_inputs (bool): Whether to detach the input embeddings and hidden states.
            This is used in RL training. Default is False.
        loss_scaling_factor (float): Scaling factor for MTP loss. Per-depth CE is
            averaged across depths; e2e TV is already normalized by the configured
            speculative horizon. The selected objective is multiplied by this
            factor. Default: 0.1.
        loss_type (str): MTP training objective. ``"ce"`` preserves the existing
            per-depth cross-entropy objective. ``"e2e_tv"`` directly optimizes the
            expected multi-step rejection-sampling acceptance length. Default: ``"ce"``.
        tv_loss_chunk_size (int): Number of token positions processed together when
            computing the exact full-vocabulary TV overlap. Default: 128.

    Example:
        >>> # In model config
        >>> config = TransformerConfig(
        ...     ...,
        ...     mtp_config=MTPConfig(
        ...         num_layers=2,
        ...         share_weights=True,
        ...         loss_scaling_factor=0.1,
        ...     ),
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    num_layers: Annotated[int, Parameter(group="model")]
    share_weights: Annotated[bool, Parameter(group="model")] = False
    detach_mtp_lm_head_weight: Annotated[bool, Parameter(group="model")] = False
    detach_mtp_inputs: Annotated[bool, Parameter(group="model")] = False
    loss_scaling_factor: Annotated[float, Parameter(group="model")] = 0.1
    loss_type: Annotated[Literal["ce", "e2e_tv"], Parameter(group="model")] = "ce"
    tv_loss_chunk_size: Annotated[int, Parameter(group="model")] = 128
