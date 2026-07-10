from typing import Any, Literal, Optional

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated


class PipelineParallelConfig(BaseModel):
    """Configuration for pipeline parallel (PP).

    PP is configured independently of FSDP: a model is sharded across ``pp_size`` pipeline stages,
    each stage owning a contiguous range of decoder layers. Expert parallel keeps using
    ``FSDPConfig.ep_size``; the two combine as ``world_size == pp_size * ep_size`` (validated where PP
    meets the model/engine, not here, since this config does not know the world size or layer count).

    Args:
        pp_size (int): Number of pipeline stages. ``1`` disables PP. Defaults to ``1``.
        schedule (Literal["1f1b"]): Pipeline schedule. Only ``"1f1b"`` is supported for now;
            ``gpipe`` / interleaved variants are future work. Defaults to ``"1f1b"``.
        layer_split (Optional[list[int]]): Explicit number of decoder layers per stage, length
            ``pp_size``. ``None`` splits layers as evenly as possible. The sum must equal the model's
            ``num_hidden_layers`` (checked against the model in ``split_for_pipeline``). Defaults to
            ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    pp_size: Annotated[int, Parameter(help="Number of pipeline-parallel stages")] = 1
    schedule: Annotated[Literal["1f1b"], Parameter(help="Pipeline schedule")] = "1f1b"
    layer_split: Annotated[
        Optional[list[int]], Parameter(help="Per-stage decoder layer counts; None splits evenly")
    ] = None

    def model_post_init(self, __context: Any) -> None:
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")
        if self.layer_split is not None:
            if len(self.layer_split) != self.pp_size:
                raise ValueError(
                    f"layer_split must have one entry per stage (pp_size={self.pp_size}), "
                    f"got {len(self.layer_split)} entries"
                )
            if any(n <= 0 for n in self.layer_split):
                raise ValueError(f"layer_split entries must be positive, got {self.layer_split}")
