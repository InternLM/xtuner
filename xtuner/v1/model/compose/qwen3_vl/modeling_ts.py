import torch
from xtuner.v1.model import BaseModel
from transformers import AutoModel, AutoConfig
from xtuner.v1.config import FSDPConfig
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from typing import Optional
from typing_extensions import override
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from xtuner.v1.utils import get_device, get_torch_device_module
from tqdm import tqdm
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


class Qwen3VLTimeSeriesModel(BaseModel):

    def __init__(self, path) -> None:
        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        ts_config = hf_config.ts_config
        from xtuner.v1.model.base import HFSaveCfg
        ts_config.hf_save_cfg = HFSaveCfg()
        ts_config.hf_config = None
        super().__init__(ts_config)  # type: ignore[arg-type]

        # ts_config.attn_implementation = "flash_attention_2"
        ts_config.use_cache = False
        ts_config.name_or_path = path  # 必须要
        self.time_series = AutoModel.from_config(ts_config, trust_remote_code=True)

        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        key = key.replace('time_series', 'model.time_series')
        return [key]

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

        # NOTE: 在 cpu_offload 模式下，mesh 应该是 cuda 的，在 meta fully_shard 后在调用 .to_empty(device=cpu)
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

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        num_recompute_layers = int(len(self.time_series.encoder.layers) * fsdp_config.vision_recompute_ratio)
        for layer_idx in tqdm(list(range(len(self.time_series.encoder.layers))), desc="[TimeSeries Fully Shard]"):
            layer = self.time_series.encoder.layers[layer_idx]

            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer,
                                           preserve_rng_state=checkpoint_preserve_rng_state,
                                           checkpoint_impl=CheckpointImpl.REENTRANT)
                if self.compile_cfg:
                    layer.forward = torch.compile(layer.forward, fullgraph=True)

            self.time_series.encoder.layers[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy()
                if fsdp_config.cpu_offload
                else None,
            )
        for layer_cur, layer_next in zip(self.time_series.encoder.layers[:-1], self.time_series.encoder.layers[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    def forward(self,
                time_series_signals: Optional[torch.FloatTensor] = None,
                ts_lens: Optional[torch.Tensor] = None,
                sr: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        ts_embeds, ts_pad_mask = self.time_series(
            time_series_signals,
            ts_lens=ts_lens,
            sr=sr
        )
        return ts_embeds, ts_pad_mask

