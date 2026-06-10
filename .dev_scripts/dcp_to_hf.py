from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.moe.moe import MoEConfig
from cyclopts import App, Parameter
from pathlib import Path
import torch.distributed as dist
import torch
from typing import Annotated, Literal
import pickle

from xtuner.v1.utils import profile_time_and_memory
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from xtuner.v1.model.base import BaseModel
from xtuner.v1.train.trainer import TrainerConfig
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTensorWiseFloat8CastTensor,
    WeightWithDynamicTilewiseFloat8CastTensor,
)
import torch.distributed.checkpoint as dcp




usage = """Usage
torchrun --nproc-per-node 8 .dev_scripts/dcp_to_hf.py <dcp-path> --hf-path <hf-path> [--remote-hf-path <remote-hf-path>] [--dtype bf16|fp8]

Arguments:
  <dcp-path>                 DCP checkpoint dir, e.g. <work_dirs>/<timestamp>/checkpoints/ckpt-step-6
  --hf-path <hf-path>        Output dir for the HF checkpoint (defaults to a subfolder of <dcp-path>)
  --remote-hf-path <path>    Source of the remote code/config; required when the model has no `hf_config`
                             (falls back to the run's `load_from` when omitted)
  --dtype bf16|fp8           Save dtype, defaults to bf16; `fp8` saves per-block float8_e4m3fn
"""

cli = App(usage=usage)


@cli.default
def dcp_to_hf(
    dcp_path: Annotated[
        Path,
        Parameter(
            help="Path to the DCP checkpoint, <work_dirs>/<timestamp>/checkpoints/ckpt-step-6"
        ),
    ],
    hf_path: Annotated[
        Path | None,
        Parameter(
            help="Path to save hf checkpoint, defaults to a subfolder of dcp path"
        ),
    ] = None,
    remote_hf_path: Annotated[
        Path | None,
        Parameter(
            help="Path to save hf checkpoint, defaults to a subfolder of dcp path"
        ),
    ] = None,
    dtype: Annotated[
        Literal["fp8", "bf16"],
        Parameter(
            help="Path to save hf checkpoint, defaults to a subfolder of dcp path"
        ),
    ] = "bf16",
):
    dist.init_process_group(backend="cuda:nccl,cpu:gloo")
    torch.serialization.add_safe_globals(
        [
            WeightWithDynamicTilewiseFloat8CastTensor,
            WeightWithDynamicTensorWiseFloat8CastTensor,
        ]
    )
    ep_size = 16
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    if hf_path is None:
        hf_path = dcp_path / (dcp_path.name + "_hf")

    trainer_cfg_path = dcp_path / "trainer_config.bin"
    trainer_cfg: TrainerConfig = pickle.loads(trainer_cfg_path.read_bytes())

    model_cfg = trainer_cfg.model_cfg
    fsdp_cfg = trainer_cfg.fsdp_cfg
    assert fsdp_cfg is not None, "FSDPConfig is required to load DCP checkpoint"

    if isinstance(model_cfg, MoEConfig):
        model_cfg.ep_size = ep_size
        fsdp_cfg.ep_size = ep_size


    with torch.device("meta"):
        model = model_cfg.build()

    model.fully_shard(fsdp_cfg)

    # Remote-code HF models keep `model_cfg.hf_config` as None, so every sub-model needs `_hf_path`
    # pointing at the original repo for `save_hf` to copy back its config and modeling code. The path
    # comes from `--remote-hf-path`, falling back to the run's `load_from`.
    if model_cfg.hf_config is None:
        remote_hf_path = remote_hf_path or trainer_cfg.load_from
        if remote_hf_path is None:
            raise RuntimeError("Remote code found! please set --remote-hf-path")

        for module in model.modules():
            if module is model:
                continue
            if isinstance(module, BaseModel):
                module._hf_path = remote_hf_path

    load_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)
    set_options = StateDictOptions(cpu_offload=True, strict=True)

    with profile_time_and_memory(f"[Load DCP Model from {dcp_path}]"):
        shard_model_state_dict = get_model_state_dict(model, options=load_options)
        # inplace state_dict
        dcp.load(
            state_dict={"model": shard_model_state_dict},
            checkpoint_id=dcp_path / "weights",
        )
        set_model_state_dict(model, shard_model_state_dict, options=set_options)

    if dtype == "bf16":
        model.save_hf(hf_path)
    else:
        model.save_hf(hf_path, save_dtype=torch.float8_e4m3fn)

    dist.destroy_process_group()


if __name__ == "__main__":
    cli()
