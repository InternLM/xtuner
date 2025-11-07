from typing import Literal

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated


class BaseLossConfig(BaseModel):
    """Base configuration for loss function."""

    model_config = ConfigDict(extra="forbid")
    type: Annotated[
        Literal["grpo", "ppo"],
        Parameter(group="Loss Types", help="Type of the loss function."),
    ]
