from typing_extensions import override, Self
from torch import nn
import torch

from functools import partial

from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .intern_s1_config import InternS1ProjectorConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from .modeling_vision import init_world_mesh

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class InternS1MultiModalProjector(BaseModel):
    config: InternS1ProjectorConfig

    def __init__(self, config: InternS1ProjectorConfig):
        super().__init__(config)  # type: ignore[arg-type]
        self.layer_norm = nn.LayerNorm(config.vision_hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.linear_1 = nn.Linear(
            config.vision_hidden_size * int(1 / config.downsample_ratio) ** 2, config.text_hidden_size
        )
        self.act = get_act_fn(config.hidden_act)
        self.linear_2 = nn.Linear(config.text_hidden_size, config.text_hidden_size)

        self._hf_prefix = "model.multi_modal_projector."
        self._init_load_spec()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    # Note: 因为 model 本身就是 self，暂时无法实现在 fully_shard 时候进行 checkpoint
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
    ) -> Self:
        self.fsdp_config = fsdp_config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else OffloadPolicy(),
        )
        return self
