from functools import partial
from typing import Literal, Protocol

import torch
from pydantic import BaseModel


class MoEActFnProtocol(Protocol):
    def __call__(self, fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor: ...


class MoEActFnConfig(BaseModel):
    act_type: Literal["clipped_swiglu", "swiglu"] = "swiglu"

    clip_alpha: float | None = None
    clip_limit: float | None = None

    def build(self) -> MoEActFnProtocol:
        from xtuner.v1.ops.moe_act_fn import get_moe_act_fn

        act_fn = get_moe_act_fn(self.act_type)

        if self.act_type == "clipped_swiglu":
            act_fn = partial(act_fn, alpha=self.clip_alpha, limit=self.clip_limit)
        return act_fn
