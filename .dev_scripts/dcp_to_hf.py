from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.moe.moe import MoEConfig
from transformers import AutoTokenizer
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
from xtuner.v1.train.trainer import TrainerConfig
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTensorWiseFloat8CastTensor,
    WeightWithDynamicTilewiseFloat8CastTensor,
)
import torch.distributed.checkpoint as dcp




usage = """Usage
clusterx  run --image <image> --no-env -N 8 --gpus-per-task 8 --cpus-per-task 150 --memory-per-task 1500  export PYTHONPATH=<custom-xtuner>:<interntrain>:<xtuenr-path> '&&' torchrun --nproc-
per-node 8  --master-addr '$MASTER_ADDR' --master-port '$MASTER_PORT' --nnodes '$WORLD_SIZE' --node-rank '$RANK' .dev_scripts/dcp_to_hf.py <dcp-path> --hf-path <hf-path>
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
    tokenizer_path: Annotated[
        Path,
        Parameter(
            help="Path to the tokenizer folder, usually the same as the hf_path"
        ),
    ],
    hf_path: Annotated[
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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

    load_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)
    set_options = StateDictOptions(cpu_offload=True, strict=True)

    with profile_time_and_memory(f"[Load DCP Model from {dcp_path}]"):
        shard_model_state_dict = get_model_state_dict(model, options=load_options)
        # inplace state_dict
        dcp.load(
            state_dict=shard_model_state_dict,
            checkpoint_id=dcp_path / "model",
        )
        set_model_state_dict(model, shard_model_state_dict, options=set_options)

    if dtype == "bf16":
        model.save_hf(hf_path)
    else:
        model.save_hf(hf_path, save_dtype=torch.float8_e4m3fn)

    if dist.get_rank() == 0:
        tokenizer.save_pretrained(hf_path)


if __name__ == "__main__":
    cli()
