"""Check SP convolution gradient scale against a full-batch CE reference."""

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.ops.gated_deltanet.causal_conv1d import causal_conv1d_fn
from xtuner.v1.utils import set_deterministic


class SPConvModel(BaseModel):
    def __init__(self):
        super().__init__(
            XTunerBaseModelConfig(
                compile_cfg=False,
                hf_save_cfg=HFSaveCfg(fp32_keys_pattern=[r"^conv1d\.weight$"]),
            )
        )
        self.conv1d = nn.Conv1d(64, 64, 4, groups=64, bias=False)
        self.head = nn.Linear(64, 64, bias=False)
        # An identity head avoids different GEMM shapes obscuring gradient scaling.
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(64))
        self._init_load_spec()

    def to_hf_key_list(self, key):
        return [key]

    def forward(self, x, seq_idx, sp_mesh, loss_ctx):
        weight = self.conv1d.weight.to_local().squeeze(1)
        if sp_mesh.size() > 1:
            # Same sequence-to-channel redistribution as GatedDeltaNet.forward_for_sp.
            x = ulysses_all_to_all(x.transpose(1, 2), scatter_dim=1, gather_dim=2, mesh=sp_mesh)
            x = x.transpose(1, 2).contiguous()
            weight = weight.chunk(sp_mesh.size(), dim=0)[sp_mesh.get_local_rank()]
        out = causal_conv1d_fn(x, weight, seq_idx=seq_idx, activation="silu")
        if sp_mesh.size() > 1:
            out = ulysses_all_to_all(out, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        loss, _ = loss_ctx(out, self.head.weight)
        return loss


def _check_conv_gradient_scale(rank, init_method, dtype):
    set_deterministic()
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for sp in (1, 2):
            torch.manual_seed(123)
            model = SPConvModel().cuda()
            x = torch.randn(1, 32, 64, device="cuda", dtype=dtype)
            labels = torch.arange(32, device="cuda").unsqueeze(0)
            labels[:, [4, 15, 31]] = -100  # Unequal valid-token counts across ranks.
            seq_idx = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(16).unsqueeze(0)

            # Independent normalization: one full-batch mean CE, with no distributed
            # loss reduction, gradient reduction, clipping, or optimizer normalization.
            initial_weight = model.conv1d.weight.detach().clone()
            ref_weight = initial_weight.clone().requires_grad_()
            ref_out = causal_conv1d_fn(x, ref_weight.squeeze(1), seq_idx=seq_idx, activation="silu")
            ref_logits = F.linear(ref_out, model.head.weight.detach().to(dtype)).float()
            ref_loss = F.cross_entropy(ref_logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            (ref_grad,) = torch.autograd.grad(ref_loss, ref_weight)

            mesh = init_device_mesh("cuda", (2 // sp, sp), mesh_dim_names=("dp", "sp"))
            sp_mesh = mesh["sp"]
            model.fully_shard(FSDPConfig(param_dtype=dtype, reduce_dtype=torch.float32, torch_compile=False))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
            docs = tuple(torch.zeros(1, 16, dtype=torch.long, device="cuda") for _ in range(sp))
            data = [
                {
                    "seq_ctx": SequenceContext.from_input_ids(docs),
                    "shifted_labels": labels if sp == 2 else labels.chunk(2, dim=1)[rank].contiguous(),
                }
            ]
            loss_ctx = model.build_loss_ctx_batch(data, sp_mesh=sp_mesh)[0]["lm"]
            local_idx = seq_idx if sp == 2 else torch.zeros(1, 16, dtype=torch.int32, device="cuda")
            loss = model(x.chunk(2, dim=1)[rank].contiguous(), local_idx, sp_mesh, loss_ctx)
            torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-5)
            loss.backward()

            # Negative control: reproduce the proposed SUM / DP reduction directly.
            dp_only_grad = model.conv1d.weight.grad.to_local().clone()
            dist.all_reduce(dp_only_grad, op=dist.ReduceOp.SUM)
            dp_only_grad.div_(2 // sp)
            model.scale_and_reduce_grad()
            grad = model.conv1d.weight.grad.to_local()
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=1e-5)
            if sp > 1:
                torch.testing.assert_close(dp_only_grad, sp * ref_grad, rtol=0, atol=2e-5)
                assert (dp_only_grad - ref_grad).abs().max().item() > 1e-5

            optimizer.step()
            expected_weight = initial_weight - 0.125 * ref_grad
            torch.testing.assert_close(model.conv1d.weight.to_local(), expected_weight, rtol=0, atol=1e-5)
            if rank == 0:
                print(
                    "CONV_GRAD_SCALE",
                    json.dumps(
                        {
                            "dtype": str(dtype),
                            "sp": sp,
                            "gradient_max_abs": (grad - ref_grad).abs().max().item(),
                            "gradient_norm_ratio": (grad.norm() / ref_grad.norm()).item(),
                            "dp_only_gradient_norm_ratio": (dp_only_grad.norm() / ref_grad.norm()).item(),
                            "sgd_update_max_abs": (model.conv1d.weight.to_local() - expected_weight)
                            .abs()
                            .max()
                            .item(),
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sp_conv_gradient_scale(tmp_path, dtype):
    mp.spawn(_check_conv_gradient_scale, args=((tmp_path / "rendezvous").as_uri(), dtype), nprocs=2, join=True)
