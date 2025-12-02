from xtuner.v1.ops.act_fn import get_act_fn
import torch.nn as nn
import torch
from typing_extensions import override
from .qwen3_vl_config import Qwen3VLProjectorConfig
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from .modeling_vision import init_world_mesh
from xtuner.v1.utils import get_device, get_torch_device_module, init_params
from functools import partial


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class Qwen3VLVisionPatchMerger(nn.Module):
    config: Qwen3VLProjectorConfig

    def __init__(self, config: Qwen3VLProjectorConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.factor = config.spatial_merge_size ** 2
        self.hidden_size = config.vision_hidden_size * self.factor
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if self.use_postshuffle_norm else config.vision_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = get_act_fn("gelu")
        self.linear_fc2 = nn.Linear(self.hidden_size, config.text_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    @torch.no_grad()
    def init_weights(self):
        init_params(self.norm.weight, nn.init.ones_)
        init_params(self.norm.bias, nn.init.zeros_)
        init_params(self.linear_fc1.bias, nn.init.zeros_)
        init_params(self.linear_fc1.weight, partial(nn.init.normal_, mean=0.0, std=0.02))
        init_params(self.linear_fc2.bias, nn.init.zeros_)
        init_params(self.linear_fc2.weight, partial(nn.init.normal_, mean=0.0, std=0.02))


class Qwen3VLProjector(BaseModel):
    config: Qwen3VLProjectorConfig

    def __init__(self, config: Qwen3VLProjectorConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self._hf_prefix = "model.visual."
        self._init_load_spec()

    @maybe_compile(fullgraph=True)
    def forward(self, hidden_states: torch.Tensor, deepstack_feature_lists: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.merger(hidden_states)
        deepstack_projected_features = []
        for i, deepstack_feature in enumerate(deepstack_feature_lists):
            deepstack_projected_features.append(self.deepstack_merger_list[i](deepstack_feature))
        return hidden_states, deepstack_projected_features

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    # TODO: 暂时不开 checkpoint
    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None
        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        self._maybe_compile_layers()

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
        self.merger.init_weights()
        for merger in self.deepstack_merger_list:
            merger.init_weights()
