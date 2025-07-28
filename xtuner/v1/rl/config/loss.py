from typing import Literal

from cyclopts import Parameter
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class BaseLossConfig(BaseModel):
    """Base configuration for loss function."""

    type: Annotated[
        Literal["grpo", "ppo"],
        Parameter(group="Loss Types", description="Type of the loss function."),
    ] = Field(..., discriminator="type")
