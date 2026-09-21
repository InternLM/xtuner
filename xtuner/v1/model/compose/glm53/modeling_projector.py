# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash projector: `downsample` (2x2 spatial merge) + `merger`, see
design doc F2 §5.2.

Local naming mirrors HF exactly (`merger.proj/post_projection_norm/gate_proj/up_proj/down_proj`,
`downsample`) per §6.1, so `from_hf()` needs no per-model copy loop -- only `to_hf_key_list`
remaps the `model.visual.` prefix.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model import BaseModel
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils.init_weight import default_init_weights

from .glm53_config import Glm53ProjectorConfig
from .modeling_vision import init_world_mesh


class Glm53VisionPatchMerger(nn.Module):
    def __init__(self, config: Glm53ProjectorConfig) -> None:
        super().__init__()
        dim, context_dim = config.out_hidden_size, config.projection_intermediate_size
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.act1 = nn.GELU()
        self.act_fn = get_act_fn(config.hidden_act)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_state).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm53Projector(BaseModel):
    config: Glm53ProjectorConfig

    def __init__(self, config: Glm53ProjectorConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.spatial_merge_size = config.spatial_merge_size
        self.downsample = nn.Conv2d(
            in_channels=config.vision_hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm53VisionPatchMerger(config)

        self._hf_prefix = "model.visual."
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    @torch.no_grad()
    def init_weights(self) -> None:
        initialized = default_init_weights(self)
        if missing := {name for name, _ in self.named_parameters()} - initialized:
            raise RuntimeError(f"{missing} is not initialized")

    @override
    def fully_shard(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config
        self.fsdp_mesh = init_world_mesh()

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        setattr(module, p_name, torch.nn.Parameter(param.to(dtype=torch.float32)))
        else:
            for param in self.parameters():
                param.requires_grad = False

        if self.config.fully_shard:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
            )
            self._fully_shard(
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m = self.spatial_merge_size
        hidden_states = hidden_states.view(-1, m, m, hidden_states.shape[-1])
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        return self.merger(hidden_states)
