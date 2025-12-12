import json
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from pydantic import ConfigDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model import BaseModel
from xtuner.v1.model.base import XTunerBaseModelConfig
from xtuner.v1.utils import get_device, get_logger


DEVICE = get_device()
logger = get_logger()


class BaseComposeConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    vision_config: XTunerBaseModelConfig
    projector_config: XTunerBaseModelConfig
    text_config: XTunerBaseModelConfig

    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False
    dcp_ignore_frozen_params: bool = True


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


def to_hf_key_list_wrapper(fn: Callable[[str], list[str]], convertor: Callable[[str], str]):
    def wrapper(self, *args, **kwargs):
        return [convertor(i) for i in fn(*args, **kwargs)]

    return wrapper


def modify_safetensors_index_json(hf_dir: Path, weight_map_dict: dict):
    if dist.get_rank() == 0:
        with open(hf_dir / "model.safetensors.index.json") as f:
            weight_map_dict.update(json.load(f)["weight_map"])


class BaseComposeModel(BaseModel):
    def __init__(self, config: BaseComposeConfig):
        super().__init__(config)  # type: ignore[arg-type]
        self._hf_path: Path | None = None

        self.vision_tower = config.vision_config.build()
        self.multi_modal_projector = config.projector_config.build()
        self.language_model = config.text_config.build()

        self._maybe_enable_compile(self.compile_cfg)
        self._freeze_modules()

    def _freeze_modules(self):
        freeze_vision = self.config.freeze_vision
        if freeze_vision:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
            logger.info("Freeze vision tower")
        freeze_projector = self.config.freeze_projector
        if freeze_projector:
            self.multi_modal_projector.requires_grad_(False)
            self.multi_modal_projector.eval()
            logger.info("Freeze multi modal projector")
        freeze_language = self.config.freeze_language
        if freeze_language:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
            logger.info("Freeze language model")

    @override
    def init_weights(self) -> None:
        self.vision_tower.init_weights()
        self.language_model.init_weights()
        self.multi_modal_projector.init_weights()

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        # TODO: 判断其余模块是否已经被 fsdp 切分了

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)

        self.fsdp_mesh = init_world_mesh()
        # Note: 非常关键，不能删除这个 assert
        assert self.fsdp_mesh is not None

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

        if isinstance(self.vision_tower.blocks[0], FSDPModule):
            self.language_model.embed_tokens.set_modules_to_forward_prefetch(  # type: ignore
                [self.vision_tower.blocks[0]]
            )
            self.vision_tower.blocks[-1].set_modules_to_forward_prefetch(  # type: ignore
                [self.multi_modal_projector]
            )
        self.multi_modal_projector.set_modules_to_forward_prefetch([self.language_model])  # type: ignore
        self.language_model.set_modules_to_forward_prefetch([self.language_model.layers["0"]])  # type: ignore

        self._to_empty_meta()
        return self

    def from_hf(self, hf_path: str | Path, strict=True):
        self._hf_path = Path(hf_path)

        if isinstance(hf_path, Path):
            hf_path = str(hf_path)

        _, _, missing_llm_keys = self.language_model.from_hf(hf_path, strict=False)
        _, _, missing_vision_keys = self.vision_tower.from_hf(hf_path, strict=False)
        _, _, missing_project_keys = self.multi_modal_projector.from_hf(hf_path, strict=False)

        missing = missing_llm_keys | missing_vision_keys | missing_project_keys
        if strict:
            if missing:
                raise RuntimeError(f"Missing parameters from {hf_path}: {list(missing)}. ")

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16, prefix: str = "model"):
        hf_dir = Path(hf_dir)
        self.language_model.save_hf(hf_dir, save_dtype, "model-language")

        weight_map_dict: dict = {}
        modify_safetensors_index_json(hf_dir, weight_map_dict)

        self.vision_tower.save_hf(hf_dir, save_dtype, "model-vision")
        modify_safetensors_index_json(hf_dir, weight_map_dict)

        self.multi_modal_projector.save_hf(hf_dir, save_dtype, "model-project")
        modify_safetensors_index_json(hf_dir, weight_map_dict)

        if dist.get_rank() == 0:
            with open(hf_dir / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map_dict, "metadata": {}}, f, indent=4)
        dist.barrier()

    def scale_and_reduce_grad(self):
        self.language_model.scale_and_reduce_grad()
