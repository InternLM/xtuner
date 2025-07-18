from typing import Optional

from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.float8.float8_tensor import ScalingGranularity


class Float8Config(BaseModel):
    scaling_granularity_gemm: Optional[
        Annotated[
            ScalingGranularity,
            Parameter(help="Scaling granularity for GEMM operations, e.g., TILEWISE and TENSORWISE"),
        ]
    ] = None
    scaling_granularity_grouped_gemm: Optional[
        Annotated[
            ScalingGranularity,
            Parameter(help="Scaling granularity for grouped GEMM operations. Currently only TILEWISE is supported"),
        ]
    ] = None

    @property
    def enable_float8(self) -> bool:
        """Whether to enable float8 quantization."""
        return self.scaling_granularity_gemm is not None or self.scaling_granularity_grouped_gemm is not None

    @property
    def is_tilewise(self) -> bool:
        """Whether the scaling granularity is TILEWISE."""
        return (
            self.scaling_granularity_gemm == ScalingGranularity.TILEWISE
            or self.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE
        )

    @property
    def is_tensorwise(self) -> bool:
        """Whether the scaling granularity is TENSORWISE."""
        return self.scaling_granularity_gemm == ScalingGranularity.TENSORWISE
