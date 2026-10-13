"""Inspect the actual FP32 ignored-parameter mesh on four NCCL ranks."""

import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist

from xtuner.v1.config import FSDPConfig


test_file = Path(__file__).resolve().parents[2] / "tests/model/test_moe_ignored_grad_sync.py"
spec = importlib.util.spec_from_file_location("ignored_grad_sync_test", test_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
try:
    for ep_size in (1, 2, 4):
        with torch.device("meta"):
            model = module._config(ep_size, sp_size=1).build()
        model.fully_shard(FSDPConfig(ep_size=ep_size, param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        parameter = model.layers["0"].input_layernorm.weight
        print(
            f"rank={rank} ep_size={ep_size} world_mesh={model.world_mesh.mesh.tolist()} "
            f"parameter_mesh={parameter.device_mesh.mesh.tolist()} placements={parameter.placements}",
            flush=True,
        )
        dist.barrier()
finally:
    dist.destroy_process_group()
