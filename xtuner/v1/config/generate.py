from typing import Annotated, Literal, TypeVar

from cyclopts import Parameter
from pydantic import BaseModel


T = TypeVar("T")


class GenerateConfig(BaseModel):
    max_batch_size: Annotated[int, Parameter(group="generate")] = 32
    max_prefill_batch: Annotated[int, Parameter(group="generate")] = 16
    max_length: Annotated[int, Parameter(group="generate")] = 2048
    block_size: Annotated[int, Parameter(group="generate")] = 128
    dtype: Annotated[Literal["bf16", "fp8"], Parameter(group="generate")] = "bf16"
