from pathlib import Path
from typing import cast

import torch

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import PreparedHFParams, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.utils import get_device, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

FrozenModel = BaseComposeModel | XtunerBaseModel


def _precompute_float8_dynamic_scale(model: FrozenModel) -> None:
    """Precompute FP8 scales after the model's FSDP shards reach the GPU."""
    if isinstance(model, BaseComposeModel):
        target = model.language_model
    else:
        target = model
    float8_handler = target.float8_handler
    if float8_handler is not None:
        float8_handler.precompute_float8_dynamic_scale_for_fsdp(target)


def build_frozen_model(
    model_cfg: TransformerConfig | BaseComposeConfig,
    load_from: str | Path,
    fsdp_cfg: FSDPConfig | None = None,
    *,
    target_device: torch.device | str = "cpu",
    load_device: torch.device | str | None = None,
) -> FrozenModel:
    """Build a frozen FSDP model and move it to the target device.

    Common device combinations:
    - ``load_device=None, target_device="cpu"``: load on GPU, then return a CPU-resident model.
    - ``load_device=DEVICE, target_device=DEVICE``: load directly on GPU for immediate Teacher compute.
    - ``load_device="cpu", target_device="cpu"``: build entirely on CPU for background Teacher prefetch.
    """
    materialize_device = torch.device(load_device) if load_device is not None else None
    target_device = torch.device(target_device)
    with torch.device("meta"):
        model = model_cfg.build()

    if isinstance(model_cfg, BaseComposeConfig):
        text_float8_cfg = getattr(model_cfg.text_config, "float8_cfg", None)
        if materialize_device is not None and materialize_device.type == "cpu":
            if text_float8_cfg is not None and text_float8_cfg.enable_float8:
                raise NotImplementedError("CPU materialization is not supported for Float8 Teachers")
        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
        model = model.fully_shard(fsdp_cfg, materialize_device=materialize_device)
        model.from_hf(hf_path=load_from, load_device=load_device)
        model.eval()  # type: ignore
    else:
        model_cfg = cast(TransformerConfig, model_cfg)
        if materialize_device is not None and materialize_device.type == "cpu":
            if model_cfg.float8_cfg is not None and model_cfg.float8_cfg.enable_float8:
                raise NotImplementedError("CPU materialization is not supported for Float8 Teachers")
        if fsdp_cfg is None:
            fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
        model = model.fully_shard(fsdp_cfg)  # type: ignore
        model.from_hf(hf_path=load_from, load_device=load_device)
        model.eval()  # type: ignore

    if materialize_device is None or materialize_device != target_device:
        model.to_device(target_device)  # type: ignore
    # ``to_device`` may recreate FP8 tensor wrappers and drop their cached
    # dynamic scales.  Always compute scales after the final device move.
    _precompute_float8_dynamic_scale(model)
    DEVICE_MODULE.empty_cache()
    return model


def build_frozen_model_shell(
    model_cfg: TransformerConfig | BaseComposeConfig,
    fsdp_cfg: FSDPConfig | None = None,
) -> FrozenModel:
    """Build an FSDP-wrapped model on meta without loading checkpoint
    weights."""
    if fsdp_cfg is None:
        fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
    with torch.device("meta"):
        model = model_cfg.build()
    if isinstance(model_cfg, BaseComposeConfig):
        model = model.fully_shard(fsdp_cfg, materialize_device="meta")
    else:
        model = model.fully_shard(fsdp_cfg)  # type: ignore
    return model


def prepare_frozen_model(
    model: FrozenModel,
    load_from: str | Path,
) -> tuple[FrozenModel, PreparedHFParams | dict[str, PreparedHFParams]]:
    """Prepare ordinary CPU tensors for a previously built model shell."""
    return model, model.prepare_hf(load_from)


def materialize_frozen_model(
    model: FrozenModel,
    prepared: PreparedHFParams | dict[str, PreparedHFParams],
) -> FrozenModel:
    """Materialize prepared tensors and move the model to the compute
    device."""
    model.materialize_prepared_hf(prepared)  # type: ignore[arg-type]
    model.eval()  # type: ignore
    model.to_device(DEVICE)  # type: ignore
    _precompute_float8_dynamic_scale(model)
    DEVICE_MODULE.empty_cache()
    return model
