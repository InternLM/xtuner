"""Configuration contract for Xtuner's optional UltraEP execution path."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, model_validator


class UltraEPConfig(BaseModel):
    """Runtime-only redundant-expert configuration.

    ``MoEConfig.ultraep_cfg is None`` is the only disabled state. Replica weights
    and gradients remain owned by the UltraEP runtime; this configuration never
    changes model parameters, optimizer state, or checkpoints.
    """

    model_config = ConfigDict(extra="forbid")

    num_redundant_experts_per_rank: Annotated[
        int, Parameter(help="UltraEP redundant-expert slots reserved on each EP rank")
    ]

    @model_validator(mode="after")
    def validate_redundant_experts(self) -> "UltraEPConfig":
        if self.num_redundant_experts_per_rank <= 0:
            raise ValueError("num_redundant_experts_per_rank must be > 0")
        return self
