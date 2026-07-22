from pathlib import Path
from typing import cast

import torch

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.utils import get_torch_device_module


DEVICE_MODULE = get_torch_device_module()

FrozenModel = BaseComposeModel | XtunerBaseModel


def build_frozen_model(
    model_cfg: TransformerConfig | BaseComposeConfig,
    load_from: str | Path,
    fsdp_cfg: FSDPConfig | None = None,
) -> FrozenModel:
    """Build a frozen FSDP model and leave it resident on CPU."""
    with torch.device("meta"):
        model = model_cfg.build()

    if isinstance(model_cfg, BaseComposeConfig):
        assert model_cfg.text_config.float8_cfg is None, "BaseComposeConfig does not support float8"
        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
        model = model.fully_shard(fsdp_cfg)
        model.from_hf(hf_path=load_from)
        model.eval()  # type: ignore
    else:
        model_cfg = cast(TransformerConfig, model_cfg)
        if model_cfg.float8_cfg is not None and model_cfg.float8_cfg.enable_float8:
            float8_handler = Float8Handler(
                scaling_granularity_gemm=model_cfg.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
            )
        else:
            float8_handler = None
        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
        model = model.fully_shard(fsdp_cfg)  # type: ignore
        model.from_hf(hf_path=load_from)
        model.eval()  # type: ignore
        if float8_handler is not None:
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)  # type: ignore

    model.to_device("cpu")  # type: ignore
    DEVICE_MODULE.empty_cache()
    return model
