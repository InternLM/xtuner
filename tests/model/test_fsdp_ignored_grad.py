"""Two-rank regression for FP32 parameters excluded from FSDP."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig


class IgnoredGradModel(BaseModel):
    def __init__(self, mesh_shape):
        super().__init__(
            XTunerBaseModelConfig(
                compile_cfg=False,
                hf_key_mapping={r"^": "model."},
                hf_save_cfg=HFSaveCfg(fp32_keys_pattern=[r"^model\.(scale|unused)$"]),
            )
        )
        self.mesh_shape = mesh_shape
        self.proj = nn.Linear(4, 4, bias=False)
        self.scale = nn.Parameter(torch.ones(4))
        self.unused = nn.Parameter(torch.ones(4))
        nn.init.ones_(self.proj.weight)
        self._init_load_spec()

    def to_hf_key_list(self, key):
        return [key]

    def _init_world_mesh(self):
        names = ("shard",) if len(self.mesh_shape) == 1 else ("replicate", "shard")
        return init_device_mesh("cuda", self.mesh_shape, mesh_dim_names=names)

    def forward(self, x):
        return (self.proj(x).float() * self.scale.to_local()).sum()


def _check_ignored_gradients(rank, init_method, mesh_shape):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model = IgnoredGradModel(mesh_shape).cuda()
        model.fully_shard(FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, torch_compile=False))
        assert model._fsdp_ignored_param_names == {"scale", "unused"}
        assert isinstance(model.scale, DTensor)
        assert all(isinstance(p, Replicate) for p in model.scale.placements)
        assert model.scale.dtype == torch.float32
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125)

        model(torch.full((2, 4), float(rank + 1), device="cuda")).backward()
        # The ignored gradient is 8 on rank 0 and 16 on rank 1; FSDP already
        # averages the managed projection gradient to 3 on both ranks.
        torch.testing.assert_close(model.scale.grad.to_local(), torch.full((4,), 8.0 * (rank + 1), device="cuda"))
        managed_grad = model.proj.weight.grad.to_local().clone()
        torch.testing.assert_close(managed_grad, torch.full_like(managed_grad, 3.0), rtol=0, atol=0)
        model.scale_and_reduce_grad()
        torch.testing.assert_close(model.scale.grad.to_local(), torch.full((4,), 12.0, device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(model.proj.weight.grad.to_local(), managed_grad, rtol=0, atol=0)
        assert model.unused.grad is None

        optimizer.step()
        torch.testing.assert_close(model.scale.to_local(), torch.full((4,), -0.5, device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(
            model.proj.weight.full_tensor(), torch.full((4, 4), 0.625, device="cuda"), rtol=0, atol=0
        )
        torch.testing.assert_close(model.unused.to_local(), torch.ones(4, device="cuda"), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
@pytest.mark.parametrize("mesh_shape", [(2,), (1, 2)])
def test_fsdp_ignored_gradients(tmp_path, mesh_shape):
    mp.spawn(_check_ignored_gradients, args=((tmp_path / "rendezvous").as_uri(), mesh_shape), nprocs=2, join=True)
