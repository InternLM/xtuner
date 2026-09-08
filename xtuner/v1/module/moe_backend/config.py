from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from .sonicmoe import SonicMoEBackend


class SonicMoEBackendConfig(BaseModel):
    """Configuration for the official SonicMoE routed-expert backend.

    Args:
        routing_mode (Literal["general", "token_rounding"]): Routing metadata
            policy. General preserves the model's fixed top-k assignments.
        rounding_mode (Literal["nearest", "up", "down"]): Token rounding
            direction.
        rounding_quantum (int): Expert token-count tile size.
    """

    model_config = ConfigDict(extra="forbid")

    routing_mode: Literal["general", "token_rounding"] = "general"
    rounding_mode: Literal["nearest", "up", "down"] = "nearest"
    rounding_quantum: int = Field(default=128, gt=0)

    def build(self) -> "SonicMoEBackend":
        """Build a SonicMoE expert backend.

        Returns:
            SonicMoEBackend: Configured expert backend module.
        """
        from .sonicmoe import SonicMoEBackend

        return SonicMoEBackend(self)
