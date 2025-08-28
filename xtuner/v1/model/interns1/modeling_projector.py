from typing_extensions import override
from torch import nn
import torch

from functools import partial

# TODO: 等 interns1 合入后全部换成 interns1 的实现
try:
    from transformers.models.internvl.modeling_internvl import ACT2FN
except:
    ACT2FN = None

from xtuner.v1.utils import get_device, get_torch_device_module, init_params
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from .interns1_config import InternS1ProjectorConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from .modeling_vision import init_world_mesh

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class InternS1MultiModalProjector(BaseModel):
    config: InternS1ProjectorConfig

    def __init__(self, config: InternS1ProjectorConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.text_config.hidden_size
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

        self._hf_prefix = "model.multi_modal_projector."
        self._init_load_spec()

    @maybe_compile(fullgraph=True)
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
        float8_handler: Float8Handler | None = None,
    ):
        assert float8_handler is None
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
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    @torch.no_grad()
    def init_weights(self):
        init_params(self.layer_norm.weight, nn.init.ones_)
        init_params(self.layer_norm.bias, nn.init.zeros_)
        init_params(self.linear_1.bias, nn.init.zeros_)
        init_params(self.linear_1.weight, partial(nn.init.normal_, mean=0.0, std=0.02))
        init_params(self.linear_2.bias, nn.init.zeros_)
        init_params(self.linear_2.weight, partial(nn.init.normal_, mean=0.0, std=0.02))
