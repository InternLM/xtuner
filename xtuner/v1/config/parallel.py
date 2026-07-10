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
        num_virtual_stages (int): Virtual stages (model chunks) per rank for interleaved 1F1B; ``1``
            disables interleaving. Defaults to ``1``.
        layer_split (Optional[list[int]]): Explicit decoder-layer count for each virtual stage, in
            global stage order, length ``pp_size * num_virtual_stages``. ``None`` splits layers as
            evenly as possible. The sum must equal the model's ``num_hidden_layers`` (checked against
            the model in ``split_for_pipeline``). Defaults to ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    pp_size: Annotated[int, Parameter(help="Number of pipeline-parallel stages")] = 1
    schedule: Annotated[Literal["1f1b"], Parameter(help="Pipeline schedule")] = "1f1b"
    num_virtual_stages: Annotated[
        int, Parameter(help="Virtual stages (model chunks) per rank for interleaved 1F1B; 1 disables it")
    ] = 1
    layer_split: Annotated[
        Optional[list[int]], Parameter(help="Per-stage decoder layer counts; None splits evenly")
    ] = None

    def model_post_init(self, __context: Any) -> None:
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")
        if self.num_virtual_stages < 1:
            raise ValueError(f"num_virtual_stages must be >= 1, got {self.num_virtual_stages}")
        if self.layer_split is not None:
            # One entry per virtual stage (global stage order); equals pp_size when not interleaving.
            expected = self.pp_size * self.num_virtual_stages
            if len(self.layer_split) != expected:
                raise ValueError(
                    f"layer_split must have one entry per virtual stage "
                    f"(pp_size * num_virtual_stages = {expected}), got {len(self.layer_split)} entries"
                )
            if any(n <= 0 for n in self.layer_split):
                raise ValueError(f"layer_split entries must be positive, got {self.layer_split}")
