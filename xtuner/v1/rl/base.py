from typing import Literal

from cyclopts import Parameter
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class BaseTrainerConfig(BaseModel):
    type: Annotated[
        Literal["xtuner", "lmdeploy", "sglang", "vllm"],
        Parameter(group="Worker Types", description="Type of the worker."),
    ] = Field(..., discriminator="type")
